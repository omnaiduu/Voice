"""Parakeet TDT 0.6B v3 transcription server (runs in its own venv: /opt/asr).

POST /transcribe  body: raw PCM16 mono 16 kHz bytes  ->  {"text": ..., "ms": ...}

Whole-utterance (segmented) transcription: Pipecat's VAD + Smart Turn decide
when the user finished; we transcribe that segment in one shot. On a GPU this
takes tens of milliseconds, which is what Modal measured to be faster overall
than open streaming STT.
"""

import logging
import os
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request

MODEL = os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
PORT = int(os.environ.get("ASR_PORT", "8101"))

logging.getLogger("nemo_logger").setLevel(logging.ERROR)
import nemo.collections.asr as nemo_asr  # noqa: E402

model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
model = model.to("cuda").eval()  # keep fp32 weights; autocast runs the math in bf16
dec = model.cfg.decoding
if dec.strategy != "beam":
    dec.strategy = "greedy_batch"
    model.change_decoding_strategy(dec)


def transcribe(pcm16: bytes) -> str:
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < 1600:  # < 0.1 s
        return ""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.transcribe([torch.from_numpy(audio)], batch_size=1, verbose=False)
    hyp = out[0]
    return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()


# warm up (CUDA kernels, decoding graphs)
_t = np.zeros(16000, dtype=np.int16).tobytes()
for _ in range(3):
    transcribe(_t)

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.post("/transcribe")
async def transcribe_endpoint(request: Request):
    body = await request.body()
    t0 = time.perf_counter()
    text = transcribe(body)
    return {"text": text, "ms": round((time.perf_counter() - t0) * 1000, 1)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
