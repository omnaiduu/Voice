"""Kokoro-82M TTS server on GPU (runs in its own venv: /opt/tts).

POST /tts  json {"text": str, "voice": str, "speed": float}
  -> streaming body of raw PCM16 mono 24 kHz, sent as soon as each piece is ready.

Pipecat already splits the talker's text into sentences (with abbreviation and
decimal handling), so each request is one sentence. Leading/trailing silence
is trimmed so the pause between sentences is consistent (the bot adds its own
small gap).
"""

import os

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro import KModel, KPipeline
from pydantic import BaseModel

PORT = int(os.environ.get("TTS_PORT", "8102"))
SAMPLE_RATE = 24000
GAP_MS = int(os.environ.get("TTS_GAP_MS", "180"))  # pause appended after each sentence

kmodel = KModel().to("cuda").eval()
pipeline = KPipeline(lang_code="a", model=kmodel, device="cuda")


def _trim(audio: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(audio) > thresh)[0]
    if idx.size == 0:
        return audio[:0]
    start = max(int(idx[0]) - int(0.01 * SAMPLE_RATE), 0)
    end = min(int(idx[-1]) + int(0.03 * SAMPLE_RATE), audio.size)
    return audio[start:end]


def _to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def synth(text: str, voice: str, speed: float):
    pieces = list(pipeline(text, voice=voice, speed=speed))  # kokoro splits long text itself
    for n, (_, _, audio) in enumerate(pieces):
        if audio is None:
            continue
        a = audio.detach().float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
        a = _trim(a)
        if n == len(pieces) - 1:
            a = np.concatenate([a, np.zeros(int(SAMPLE_RATE * GAP_MS / 1000), dtype=np.float32)])
        yield _to_pcm16(a)


# warm up all voices we expose
for v in ("af_heart", "am_michael"):
    for _ in synth("Hello there, warming up.", v, 1.0):
        pass

app = FastAPI()


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/tts")
def tts(req: TTSRequest):
    return StreamingResponse(synth(req.text, req.voice, req.speed), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
