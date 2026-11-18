"""Web server + process supervisor for the single GPU container.

- Starts the sidecars in the background: Parakeet (ASR), Kokoro (TTS), and one
  vLLM server per enabled talker model. The page polls /api/status while they
  load (a cold start takes a few minutes; warm restarts are instant).
- Serves the UI, handles the WebRTC offer/answer, and runs one bot per call.
- /api/chat runs a single text turn against a talker (used by the routing eval).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import aiohttp
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from .config import (
    ASR_PORT,
    DEFAULT_VOICE,
    ENABLED_TALKERS,
    ENABLED_TTS,
    MAX_MODEL_LEN,
    TALKERS,
    TTS_ENGINES,
    VOICES,
)
from .structured import RESPONSE_FORMAT, SYSTEM_PROMPT

STATIC = Path(__file__).parent / "static"
VENV_ASR = os.environ.get("VENV_ASR", "/opt/asr/bin/python")
VENV_TTS = os.environ.get("VENV_TTS", "/opt/tts/bin/python")
VENV_QWEN_TTS = os.environ.get("VENV_QWEN_TTS", "/opt/qwentts/bin/python")
VLLM_BIN = os.environ.get("VLLM_BIN", "/opt/vllm/bin/vllm")
SIDECARS = Path(__file__).parent / "sidecars"
ICE_SERVERS = ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]

STATUS: dict[str, str] = (
    {"asr": "starting"} | {k: "starting" for k in ENABLED_TTS} | {k: "waiting" for k in ENABLED_TALKERS}
)
TTS_SIDECARS = {
    "kokoro": (VENV_TTS, "tts_server.py", "TTS_PORT"),
    "qwen3-tts": (VENV_QWEN_TTS, "qwen_tts_server.py", "QWEN_TTS_PORT"),
}
_procs: list[subprocess.Popen] = []
_started = False
_lock = threading.Lock()


def _wait_http(url: str, name: str, proc: subprocess.Popen, timeout_s: float = 1500) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            STATUS[name] = f"crashed (exit {proc.returncode}), see Modal logs"
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    STATUS[name] = "ready"
                    logger.info(f"{name} ready in {time.time() - t0:.0f}s")
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    STATUS[name] = "timeout"
    return False


def _spawn(cmd: list[str], env_extra: dict | None = None) -> subprocess.Popen:
    env = os.environ.copy() | (env_extra or {})
    p = subprocess.Popen(cmd, env=env)  # inherits stdout/stderr -> Modal logs
    _procs.append(p)
    return p


def _boot():
    # 1) speech models (small) first, in parallel
    waits = []
    asr = _spawn([VENV_ASR, str(SIDECARS / "asr_server.py")], {"ASR_PORT": str(ASR_PORT)})
    waits.append(threading.Thread(target=_wait_http, args=(f"http://127.0.0.1:{ASR_PORT}/health", "asr", asr)))
    for eng in ENABLED_TTS:
        venv, script, port_env = TTS_SIDECARS[eng]
        port = TTS_ENGINES[eng]["port"]
        p = _spawn([venv, str(SIDECARS / script)], {port_env: str(port)})
        waits.append(threading.Thread(target=_wait_http, args=(f"http://127.0.0.1:{port}/health", eng, p)))
    for t in waits:
        t.start()
    for t in waits:
        t.join()

    # 2) talkers one at a time (vLLM sizes its memory from what is free at start)
    for key in ENABLED_TALKERS:
        cfg = TALKERS[key]
        STATUS[key] = "loading"
        cmd = [
            VLLM_BIN, "serve", cfg["hf_id"],
            "--host", "127.0.0.1",
            "--port", str(cfg["port"]),
            "--gpu-memory-utilization", str(cfg["gpu_mem"]),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", "4",
            "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0}),
        ]
        extra = os.environ.get("VLLM_EXTRA_ARGS", "").split()
        p = _spawn(cmd + extra)
        _wait_http(f"http://127.0.0.1:{cfg['port']}/health", key, p)
    logger.info(f"boot finished: {STATUS}")


def ensure_started():
    global _started
    with _lock:
        if not _started:
            _started = True
            threading.Thread(target=_boot, daemon=True).start()


def shutdown():
    for p in _procs:
        if p.poll() is None:
            p.terminate()


# ---------------------------------------------------------------- web app
app = FastAPI(title="Talker voice demo")
_current: dict = {"conn": None, "task": None}


@app.on_event("startup")
async def _startup():
    ensure_started()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def status():
    return {
        "status": STATUS,
        "talkers": {k: TALKERS[k]["label"] for k in ENABLED_TALKERS},
        "voices": VOICES,
        "default_voice": DEFAULT_VOICE,
        "tts_engines": {
            k: {"label": TTS_ENGINES[k]["label"], "voices": TTS_ENGINES[k]["voices"],
                "default_voice": TTS_ENGINES[k]["default_voice"]}
            for k in ENABLED_TTS
        },
        "region": os.environ.get("MODAL_REGION", "unknown"),
    }


def _resolve(talker: str, tts: str, voice: str | None) -> tuple[str, str]:
    """Validate a call request; returns (tts_engine, voice) or raises HTTPException."""
    if talker not in ENABLED_TALKERS:
        raise HTTPException(400, f"unknown talker {talker}")
    if tts not in ENABLED_TTS:
        raise HTTPException(400, f"unknown tts {tts}")
    for need in ("asr", tts, talker):
        if STATUS.get(need) != "ready":
            raise HTTPException(503, f"{need} is {STATUS.get(need)}; wait for it to be ready")
    voices = TTS_ENGINES[tts]["voices"]
    return tts, (voice if voice in voices else TTS_ENGINES[tts]["default_voice"])


class Offer(BaseModel):
    sdp: str
    type: str
    talker: str
    tts: str = "kokoro"
    voice: str | None = None
    speed: float = 1.0


@app.post("/api/offer")
async def offer(o: Offer):
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    from .bot import run_bot

    tts, voice = _resolve(o.talker, o.tts, o.voice)

    # one user at a time: drop any previous call
    if _current["conn"] is not None:
        try:
            await _current["conn"].disconnect()
        except Exception:  # noqa: BLE001
            pass
        if _current["task"] is not None:
            _current["task"].cancel()

    conn = SmallWebRTCConnection(ICE_SERVERS)
    await conn.initialize(sdp=o.sdp, type=o.type)
    _current["conn"] = conn
    _current["task"] = asyncio.create_task(
        run_bot(o.talker, voice, max(0.7, min(o.speed, 1.4)), webrtc_connection=conn, tts_engine=tts)
    )
    return conn.get_answer()


@app.websocket("/ws")
async def ws_call(websocket: WebSocket, talker: str, tts: str = "kokoro", voice: str | None = None, speed: float = 1.0):
    """Audio over a WebSocket on the same HTTPS host: works behind any firewall/NAT."""
    from .bot import run_bot

    await websocket.accept()
    try:
        tts, voice = _resolve(talker, tts, voice)
    except HTTPException as e:
        await websocket.send_text(json.dumps({"type": "error", "text": e.detail}))
        await websocket.close()
        return
    await run_bot(talker, voice, max(0.7, min(speed, 1.4)), websocket=websocket, tts_engine=tts)


class LiveKitStart(BaseModel):
    talker: str
    tts: str = "kokoro"
    voice: str | None = None
    speed: float = 1.0


@app.post("/api/livekit/start")
async def livekit_start(r: LiveKitStart):
    """Create a fresh LiveKit room, start the bot in it, return the user's join token."""
    import secrets

    from livekit import api as lk_api

    from .bot import run_bot

    url, key, secret = (os.environ.get(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"))
    if not (url and key and secret):
        raise HTTPException(500, "LiveKit secret missing: create Modal secret 'livekit' (see README)")
    tts, voice = _resolve(r.talker, r.tts, r.voice)

    if _current["task"] is not None:  # one user at a time
        _current["task"].cancel()

    room = f"talker-{secrets.token_hex(4)}"

    def token(identity: str, agent: bool) -> str:
        t = lk_api.AccessToken(key, secret).with_identity(identity).with_name(identity)
        return t.with_grants(lk_api.VideoGrants(room_join=True, room=room, agent=agent)).to_jwt()

    _current["task"] = asyncio.create_task(
        run_bot(r.talker, voice, max(0.7, min(r.speed, 1.4)), livekit=(url, token("talker-bot", True), room),
                tts_engine=tts)
    )
    return {"url": url, "token": token("you", False), "room": room}


class ChatReq(BaseModel):
    talker: str
    text: str
    history: list[dict] = []


@app.post("/api/chat")
async def chat(r: ChatReq):
    """One text turn through a talker, returning the parsed JSON + latency."""
    if STATUS.get(r.talker) != "ready":
        raise HTTPException(503, f"{r.talker} is {STATUS.get(r.talker)}")
    cfg = TALKERS[r.talker]
    body = {
        "model": cfg["hf_id"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *r.history, {"role": "user", "content": r.text}],
        "temperature": 0.0,
        "max_tokens": 300,
        "stream": True,
        "response_format": RESPONSE_FORMAT,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first = None
    raw = ""
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions", json=body) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
                if delta and first is None:
                    first = time.perf_counter()
                raw += delta
    total = time.perf_counter() - t0
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
        parsed = None
    return {
        "raw": raw,
        "parsed": parsed,
        "ttft_ms": round(((first or time.perf_counter()) - t0) * 1000, 1),
        "total_ms": round(total * 1000, 1),
    }
