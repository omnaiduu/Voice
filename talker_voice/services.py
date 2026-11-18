"""Pipecat services that call the local sidecars and the talker's JSON stream."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

from .structured import IncrementalSayParser


class ParakeetSTT(SegmentedSTTService):
    """Sends each finished user utterance (16 kHz PCM16) to the Parakeet sidecar."""

    def __init__(self, *, url: str, on_result=None, **kwargs):
        super().__init__(**kwargs)
        self._url = url
        self._session: aiohttp.ClientSession | None = None
        self._on_result = on_result  # callback(text, ms) for the UI

    @property
    def wants_wav_segments(self) -> bool:
        return False  # raw PCM16

    @property
    def segment_sample_rate(self) -> int:
        return 16000

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        t0 = time.perf_counter()
        try:
            async with self._session.post(self._url, data=audio) as resp:
                data = await resp.json()
        except Exception as e:  # noqa: BLE001
            yield ErrorFrame(error=f"ASR failed: {e}")
            return
        text = (data.get("text") or "").strip()
        if self._on_result:
            await self._on_result(text, (time.perf_counter() - t0) * 1000)
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601())

    async def cleanup(self):
        await super().cleanup()
        if self._session:
            await self._session.close()


class KokoroTTS(TTSService):
    """One HTTP request per sentence to the Kokoro sidecar; audio is streamed back."""

    def __init__(self, *, url: str, voice: str, speed: float = 1.0, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True, sample_rate=24000, **kwargs)
        self._url = url
        self._voice = voice
        self._speed = speed
        self._session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if not text.strip():
            return
        if self._session is None:
            self._session = aiohttp.ClientSession()
        try:
            await self.start_tts_usage_metrics(text)
            payload = {"text": text, "voice": self._voice, "speed": self._speed}
            async with self._session.post(self._url, json=payload) as resp:
                leftover = b""
                async for chunk in resp.content.iter_any():
                    await self.stop_ttfb_metrics()
                    chunk = leftover + chunk
                    cut = len(chunk) - (len(chunk) % 2)  # keep whole int16 samples
                    leftover = chunk[cut:]
                    if cut:
                        yield TTSAudioRawFrame(
                            audio=chunk[:cut], sample_rate=24000, num_channels=1, context_id=context_id
                        )
        except Exception as e:  # noqa: BLE001
            yield ErrorFrame(error=f"TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()

    async def cleanup(self):
        await super().cleanup()
        if self._session:
            await self._session.close()


class SayExtractor(FrameProcessor):
    """Sits between the LLM and TTS.

    Consumes the talker's streamed JSON tokens and forwards ONLY the decoded
    `say` text (as LLMTextFrames) so TTS can start on the first sentence while
    the JSON is still being generated. Reports the route to the UI.
    """

    def __init__(self, *, on_route=None, on_first_say=None, on_done=None, **kwargs):
        super().__init__(**kwargs)
        self._p = IncrementalSayParser()
        self._on_route = on_route
        self._on_first_say = on_first_say
        self._on_done = on_done
        self._said_any = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._p.reset()
            self._said_any = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            ev = self._p.feed(frame.text)
            if ev.route and self._on_route:
                await self._on_route(ev.route)
            if ev.say_delta:
                if not self._said_any:
                    self._said_any = True
                    if self._on_first_say:
                        await self._on_first_say()
                await self.push_frame(LLMTextFrame(ev.say_delta), direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            route, say = self._p.finalize()
            if not self._said_any and say:  # fallback if streaming parse failed
                await self.push_frame(LLMTextFrame(say), direction)
            if self._on_done:
                await self._on_done(route, say, self._p.buf)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._p.reset()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


__all__ = ["ParakeetSTT", "KokoroTTS", "SayExtractor", "logger"]
