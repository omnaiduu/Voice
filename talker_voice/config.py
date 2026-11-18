"""Model registry and ports. Everything runs inside ONE GPU container."""

import os

ASR_PORT = 8101
TTS_PORT = 8102  # Kokoro
QWEN_TTS_PORT = 8103

# Talker models served by vLLM (each in its own process/port on the same GPU).
# gpu_gib = GPU memory budget for that vLLM server (weights + KV cache + overhead).
# Measured: E2B weights 9.2 GiB, E4B 14.5 GiB; ~2 GiB KV is plenty for short voice turns.
# The server converts it into --gpu-memory-utilization for whatever GPU we got.
TALKERS: dict[str, dict] = {
    "gemma-4-e2b": {
        "hf_id": "google/gemma-4-E2B-it",
        "label": "Gemma 4 E2B (2B effective)",
        "port": 8201,
        "gpu_gib": float(os.environ.get("E2B_GPU_GIB", "12")),
    },
    "gemma-4-e4b": {
        "hf_id": "google/gemma-4-E4B-it",
        "label": "Gemma 4 E4B (4B effective)",
        "port": 8202,
        "gpu_gib": float(os.environ.get("E4B_GPU_GIB", "17.5")),
    },
}

# Which talkers to load, comma separated. Default: both (needs a 48 GB GPU).
# Set TALKERS=gemma-4-e2b and GPU=L4 to run a single model on a cheaper 24 GB GPU.
ENABLED_TALKERS = [
    t.strip() for t in os.environ.get("TALKERS", "gemma-4-e2b,gemma-4-e4b").split(",") if t.strip()
]

DEFAULT_VOICE = "af_heart"
VOICES = {"af_heart": "Heart (female, US)", "am_michael": "Michael (male, US)"}

# TTS engines; all speak the same sidecar HTTP contract (see services.SidecarTTS).
TTS_ENGINES: dict[str, dict] = {
    "kokoro": {
        "label": "Kokoro 82M (fast, stable)",
        "port": TTS_PORT,
        "voices": VOICES,
        "default_voice": DEFAULT_VOICE,
    },
    "qwen3-tts": {
        "label": "Qwen3-TTS 1.7B (expressive)",
        "port": QWEN_TTS_PORT,
        "voices": {"Ryan": "Ryan (male)", "Aiden": "Aiden (male)", "Vivian": "Vivian (female)", "Serena": "Serena (female)"},
        "default_voice": "Ryan",
    },
}
ENABLED_TTS = [t.strip() for t in os.environ.get("TTS_ENGINES", "kokoro,qwen3-tts").split(",") if t.strip()]

MAX_MODEL_LEN = 4096  # short voice turns; keeps KV cache small
