"""Qwen3-TTS 1.7B CustomVoice server (runs in its own venv: /opt/qwentts).

Same HTTP contract as the Kokoro sidecar:
POST /tts  json {"text", "voice", "speed", "tone"} -> raw PCM16 mono 24 kHz.

Expressiveness comes from Qwen3-TTS's natural-language `instruct`: the talker
picks a `tone` for each reply and we turn it into a short style instruction.
Generation is per sentence (non-streaming inside the sentence); Pipecat's
sentence chunking keeps each request short.
"""

import os
import re
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

PORT = int(os.environ.get("QWEN_TTS_PORT", "8103"))
MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
OUT_SR = 24000
GAP_MS = int(os.environ.get("TTS_GAP_MS", "180"))

TONE_INSTRUCT = {
    "neutral": "Speak naturally in a relaxed, conversational way.",
    "warm": "Speak warmly and kindly, like a friendly person who cares.",
    "cheerful": "Speak in a bright, cheerful and upbeat tone with a smile in your voice.",
    "excited": "Speak with genuine excitement and energy.",
    "calm": "Speak calmly and softly, slow and reassuring.",
    "empathetic": "Speak gently and empathetically, with understanding and care.",
    "serious": "Speak in a clear, serious and focused tone.",
    "playful": "Speak in a playful, light-hearted and teasing tone.",
}
TAGS = re.compile(r"\[(laugh|laughs|chuckle|chuckles|sigh|sighs|cough)\]", re.I)

tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
SPEAKERS = [s for s in (tts.get_supported_speakers() or [])]
DEFAULT_SPEAKER = next((s for s in ("Ryan", "Aiden", "Vivian", "Serena") if s.lower() in [x.lower() for x in SPEAKERS]),
                       SPEAKERS[0] if SPEAKERS else None)
print(f"Qwen3-TTS speakers: {SPEAKERS}; default {DEFAULT_SPEAKER}", flush=True)


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == OUT_SR or a.size == 0:
        return a
    n = int(round(a.size * OUT_SR / sr))
    return np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a).astype(np.float32)


def _trim(a: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(a) > thresh)[0]
    if idx.size == 0:
        return a[:0]
    return a[max(int(idx[0]) - 240, 0) : min(int(idx[-1]) + 720, a.size)]


def synth(text: str, speaker: str | None, tone: str) -> bytes:
    text = TAGS.sub("", text).strip()
    if not text:
        return b""
    wavs, sr = tts.generate_custom_voice(
        text=text,
        speaker=speaker or DEFAULT_SPEAKER,
        language="English",
        instruct=TONE_INSTRUCT.get(tone, TONE_INSTRUCT["neutral"]),
    )
    a = _trim(_resample(np.asarray(wavs[0], dtype=np.float32).reshape(-1), sr))
    a = np.concatenate([a, np.zeros(int(OUT_SR * GAP_MS / 1000), dtype=np.float32)])
    return (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()


for _ in range(2):  # warm up kernels
    t0 = time.perf_counter()
    synth("Hello there, I am warming up my voice.", None, "warm")
    print(f"Qwen3-TTS warmup {time.perf_counter() - t0:.2f}s", flush=True)

app = FastAPI()


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0
    tone: str = "neutral"


@app.get("/health")
def health():
    return {"ok": True, "speakers": SPEAKERS}


@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    voice = req.voice if req.voice and req.voice.lower() in [s.lower() for s in SPEAKERS] else None
    return Response(synth(req.text, voice, req.tone), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
