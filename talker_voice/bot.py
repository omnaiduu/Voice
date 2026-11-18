"""The voice pipeline for one WebRTC session.

browser mic -> WebRTC -> Silero VAD + Smart Turn v3 (end of turn)
            -> Parakeet (whole utterance) -> context -> talker LLM (JSON, streamed)
            -> SayExtractor (streams only "say") -> sentence chunker -> Kokoro -> WebRTC -> speaker
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

try:  # optional: only needed for the LiveKit mode
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
except Exception:  # noqa: BLE001
    LiveKitParams = LiveKitTransport = None

from .config import ASR_PORT, DEFAULT_VOICE, TALKERS, TTS_ENGINES
from .services import ParakeetSTT, SayExtractor, SidecarTTS, SoftInterruptOutput
from .structured import RESPONSE_FORMAT, SYSTEM_PROMPT
from .ws_serializer import RawPCMSerializer

VAD_STOP_SECS = 0.2  # short; Smart Turn v3 decides if the turn is really over


class TurnTimer:
    """Collects per-turn timestamps and sends them to the browser."""

    def __init__(self, send):
        self._send = send
        self.reset()

    def reset(self):
        self.t: dict[str, float] = {}
        self.route = None
        self.asr_ms = None

    def mark(self, key: str, at: float | None = None):
        if key not in self.t:
            self.t[key] = at if at is not None else time.time()

    def report(self):
        base = self.t.get("user_stopped")
        if base is None:
            return
        rel = {k: round((v - base) * 1000) for k, v in self.t.items() if k != "user_stopped"}
        self._send({"type": "metrics", "route": self.route, "asr_ms": self.asr_ms, "ms": rel})


class LatencyObserver(BaseObserver):
    """Watches frames anywhere in the pipeline (no pipeline changes needed)."""

    def __init__(self, timer: TurnTimer, send, llm, tts):
        super().__init__(observe_every_push=False)
        self._timer, self._send, self._llm, self._tts = timer, send, llm, tts
        self._state = None
        self._bot_speaking = False

    def _set_state(self, state: str):
        # system frames travel both up- and downstream: only report real changes
        if state != self._state:
            self._state = state
            self._send({"type": "state", "state": state})

    async def on_push_frame(self, data: FramePushed):
        f, src = data.frame, data.source
        tm = self._timer
        if isinstance(f, UserStartedSpeakingFrame):
            self._set_state("listening")
        elif isinstance(f, VADUserStoppedSpeakingFrame):
            # VAD confirms silence after stop_secs: the voice actually ended earlier.
            tm.reset()
            tm.mark("user_stopped", f.timestamp - (f.stop_secs or VAD_STOP_SECS))
        elif isinstance(f, UserStoppedSpeakingFrame):
            tm.mark("turn_end_decided")
            self._set_state("thinking")
        elif isinstance(f, LLMFullResponseStartFrame) and src is self._llm:
            tm.mark("llm_request")
        elif isinstance(f, LLMTextFrame) and src is self._llm:
            tm.mark("llm_first_token")
        elif isinstance(f, TTSAudioRawFrame) and src is self._tts:
            tm.mark("tts_first_audio")
        elif isinstance(f, BotStartedSpeakingFrame):
            if "bot_started" not in tm.t:
                tm.mark("bot_started")
                tm.report()
            self._bot_speaking = True
            self._set_state("speaking")
        elif isinstance(f, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._set_state("idle")
        elif isinstance(f, InterruptionFrame) and self._bot_speaking:
            self._bot_speaking = False  # user barged in while the bot was talking
            self._send({"type": "interrupted"})


async def run_bot(
    talker_key: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    *,
    webrtc_connection=None,
    websocket=None,
    livekit: tuple[str, str, str] | None = None,
    tts_engine: str = "kokoro",
):
    """Run one call. Pass exactly one of `webrtc_connection` (P2P), `websocket`,
    or `livekit=(url, bot_token, room_name)` (LiveKit Cloud SFU)."""
    talker = TALKERS[talker_key]
    holder: dict = {}

    if webrtc_connection is not None:
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True, audio_out_10ms_chunks=2),
        )

        def send(msg: dict):
            try:
                webrtc_connection.send_app_message(msg)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"send_app_message failed: {e}")

    elif livekit is not None:
        lk_url, lk_token, lk_room = livekit
        transport = LiveKitTransport(
            url=lk_url,
            token=lk_token,
            room_name=lk_room,
            params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True, audio_out_10ms_chunks=2),
        )

        def send(msg: dict):
            task = holder.get("task")
            if task is not None:
                asyncio.ensure_future(task.queue_frame(OutputTransportMessageUrgentFrame(message=msg)))

    else:
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_10ms_chunks=2,
                add_wav_header=False,
                serializer=RawPCMSerializer(),
            ),
        )

        def send(msg: dict):
            # events travel on the same socket as the audio, as urgent (system) frames
            task = holder.get("task")
            if task is not None:
                asyncio.ensure_future(task.queue_frame(OutputTransportMessageUrgentFrame(message=msg)))

    timer = TurnTimer(send)

    async def on_asr(text: str, ms: float):
        timer.asr_ms = round(ms)
        timer.mark("transcript")
        send({"type": "user", "text": text, "asr_ms": round(ms)})

    stt = ParakeetSTT(url=f"http://127.0.0.1:{ASR_PORT}/transcribe", on_result=on_asr)

    llm = OpenAILLMService(
        base_url=f"http://127.0.0.1:{talker['port']}/v1",
        api_key="local",
        settings=OpenAILLMService.Settings(
            model=talker["hf_id"],
            temperature=0.4,
            max_tokens=300,
            extra={
                "response_format": RESPONSE_FORMAT,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
    )

    async def on_route(route: str):
        timer.route = route
        timer.mark("route_decided")
        send({"type": "route", "route": route})

    async def on_tone(tone: str):
        tts.tone = tone  # arrives before "say", so the first sentence already uses it
        send({"type": "tone", "tone": tone})

    async def on_first_say():
        timer.mark("first_say_text")

    async def on_done(route, say, raw):
        send({"type": "bot", "route": route, "tone": tts.tone, "text": say, "raw": raw})

    say = SayExtractor(on_route=on_route, on_tone=on_tone, on_first_say=on_first_say, on_done=on_done)
    tts = SidecarTTS(url=f"http://127.0.0.1:{TTS_ENGINES[tts_engine]['port']}/tts", voice=voice, speed=speed)
    soft = SoftInterruptOutput(fade_ms=300)

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # start_secs 0.3: ignore very short noises/"mm" so the bot isn't cut off by accident
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.3, stop_secs=VAD_STOP_SECS)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            say,
            tts,
            soft,  # fades the bot out (~0.3 s) instead of a hard cut when you interrupt
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, audio_in_sample_rate=16000, audio_out_sample_rate=24000),
        enable_rtvi=False,
        idle_timeout_secs=600,
        observers=[LatencyObserver(timer, send, llm, tts)],
    )
    holder["task"] = task

    if livekit is not None:

        @transport.event_handler("on_first_participant_joined")
        async def on_first_joined(transport, participant_id):
            logger.info(f"user joined LiveKit room, talker={talker_key}")
            await asyncio.sleep(0.5)  # let the user's data subscription settle
            send({"type": "ready", "talker": talker_key})

        @transport.event_handler("on_participant_disconnected")
        async def on_participant_left(transport, participant_id):
            logger.info("user left LiveKit room")
            await task.cancel()

    else:

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"client connected, talker={talker_key}")
            send({"type": "ready", "talker": talker_key})

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("client disconnected")
            await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)
