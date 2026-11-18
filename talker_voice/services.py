"""Pipecat services that call the local sidecars and the talker's JSON stream."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

import numpy as np

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    SystemFrame,
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


class SidecarTTS(TTSService):
    """One HTTP request per sentence to a local TTS sidecar (Kokoro or Qwen3-TTS).

    All sidecars share one contract: POST {text, voice, speed, tone} -> PCM16 24 kHz.
    `tone` is set per reply by the talker (via SayExtractor) before speech starts.
    """

    def __init__(self, *, url: str, voice: str | None, speed: float = 1.0, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True, sample_rate=24000, **kwargs)
        self._url = url
        self._voice = voice
        self._speed = speed
        self.tone = "neutral"
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
            payload = {"text": text, "voice": self._voice, "speed": self._speed, "tone": self.tone}
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

    def __init__(self, *, on_route=None, on_tone=None, on_first_say=None, on_done=None, **kwargs):
        super().__init__(**kwargs)
        self._p = IncrementalSayParser()
        self._on_route = on_route
        self._on_tone = on_tone
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
            if ev.tone and self._on_tone:
                await self._on_tone(ev.tone)
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


KokoroTTS = SidecarTTS  # backwards-compatible name


class SoftInterruptOutput(FrameProcessor):
    """Fade the bot out instead of cutting it off when the user barges in.

    Placed right before transport.output(). It releases the bot's audio in real
    time, keeping only ~`lead_ms` queued in the transport (TTS produces audio much
    faster than real time, so normally the whole sentence would already sit in
    the transport's buffer where we can't touch it). On an interruption it takes
    the next `fade_ms` of not-yet-played speech, applies a smooth fade-out, plays
    that, and drops the rest, the way a person trails off when someone cuts in.
    """

    def __init__(self, *, fade_ms: int = 300, lead_ms: int = 40, **kwargs):
        super().__init__(**kwargs)
        self._fade_s = fade_ms / 1000
        self._lead_s = lead_ms / 1000
        self._q: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._play_until = 0.0  # monotonic time when audio already released ends

    async def _release_loop(self):
        while True:
            frame, direction = await self._q.get()
            if isinstance(frame, OutputAudioRawFrame):
                now = time.monotonic()
                wait = (self._play_until - now) - self._lead_s
                if wait > 0:
                    await asyncio.sleep(wait)
                dur = len(frame.audio) / 2 / frame.sample_rate / max(frame.num_channels, 1)
                self._play_until = max(self._play_until, time.monotonic()) + dur
            await self.push_frame(frame, direction)

    def _ensure_task(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._release_loop())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._interrupt(frame, direction)
        elif isinstance(frame, SystemFrame) or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        else:
            self._ensure_task()
            await self._q.put((frame, direction))

    async def _interrupt(self, frame: Frame, direction: FrameDirection):
        # stop releasing, grab the next fade_ms of queued (unplayed) speech
        if self._task is not None:
            self._task.cancel()
            self._task = None
        chunks, need, sr = [], None, 24000
        while not self._q.empty():
            f, _ = self._q.get_nowait()
            if isinstance(f, OutputAudioRawFrame) and (need is None or need > 0):
                sr = f.sample_rate
                if need is None:
                    need = int(self._fade_s * sr) * 2
                chunks.append(f.audio[: need - (need % 2)])
                need -= len(chunks[-1])
        self._play_until = 0.0
        await self.push_frame(frame, direction)  # transport clears its small buffer
        if chunks:
            a = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32)
            ramp = np.linspace(1.0, 0.0, a.size, dtype=np.float32) ** 1.5
            faded = (a * ramp).astype(np.int16).tobytes()
            await asyncio.sleep(0.02)  # let the transport finish handling the interruption
            await self.push_frame(OutputAudioRawFrame(audio=faded, sample_rate=sr, num_channels=1))

    async def cleanup(self):
        await super().cleanup()
        if self._task is not None:
            self._task.cancel()


__all__ = ["ParakeetSTT", "SidecarTTS", "KokoroTTS", "SayExtractor", "SoftInterruptOutput", "logger"]
