"""Talker voice demo on Modal: one GPU container runs everything.

    modal deploy modal_app.py          # prints the URL, open it and talk

Inside the container (all on localhost, no network hops between models):
  - Pipecat bot + SmallWebRTC (peer-to-peer WebRTC with your browser, no SFU)
  - Silero VAD + Smart Turn v3 (end-of-turn), CPU
  - Parakeet TDT 0.6B v3 (STT)                      -> venv /opt/asr
  - Kokoro 82M (TTS)                                -> venv /opt/tts
  - vLLM: Gemma 4 E2B and Gemma 4 E4B talkers       -> venv /opt/vllm

Deploy-time knobs (environment variables on YOUR machine when you run deploy):
  VOICE_REGION   default "ap-south" (Mumbai). Comma list allowed, e.g. "ap-south,ap".
  VOICE_GPU      default "L40S,A100-80GB" (48 GB+ needed for both talkers; first available wins).
  TALKERS        default "gemma-4-e2b,gemma-4-e4b". Use "gemma-4-e2b" with VOICE_GPU=L4 to save money.
  HF_TOKEN       optional, only if Hugging Face asks for auth when downloading models.
"""

import os

import modal

APP_NAME = "talker-voice"
REGION = [r.strip() for r in os.environ.get("VOICE_REGION", "ap-south,ap").split(",") if r.strip()]
GPU = [g.strip() for g in os.environ.get("VOICE_GPU", "L40S,A100-80GB").split(",") if g.strip()]
TALKERS = os.environ.get("TALKERS", "gemma-4-e2b,gemma-4-e4b")
# Route the HTTP signalling through Mumbai too (only valid on first deploy of the app).
ROUTING_REGION = os.environ.get("VOICE_ROUTING_REGION", "ap-south")

PY = "/usr/local/bin/python"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg", "espeak-ng", "git", "libsndfile1", "build-essential")
    .run_commands(f"{PY} -m pip install -q uv")
    # Separate virtualenvs so vLLM / NeMo / Kokoro can never fight over torch or numpy versions.
    .run_commands(
        f"uv venv /opt/vllm --python {PY}",
        "uv pip install --python /opt/vllm/bin/python 'vllm==0.34.0' hf_transfer",
    )
    .run_commands(
        f"uv venv /opt/asr --python {PY}",
        "uv pip install --python /opt/asr/bin/python 'nemo_toolkit[asr]==2.7.3' "
        "'fastapi==0.115.12' 'uvicorn[standard]' hf_transfer soundfile",
    )
    .run_commands(
        f"uv venv /opt/tts --python {PY}",
        "uv pip install --python /opt/tts/bin/python 'kokoro>=0.9.4,<1' 'misaki[en]' "
        "'fastapi==0.115.12' 'uvicorn[standard]' soundfile",
        # pre-download the spaCy model misaki uses, so the first call is fast
        "uv pip install --python /opt/tts/bin/python "
        "'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'",
    )
    # The bot itself (Pipecat) lives in the main Python.
    .uv_pip_install(
        "pipecat-ai[webrtc,silero,local-smart-turn,openai]==1.16.0",
        "aiohttp",
        "fastapi[standard]",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_HOME": "/cache/torch",
            "VLLM_CACHE_ROOT": "/cache/vllm",
        }
    )
    .add_local_dir("talker_voice", remote_path="/root/talker_voice")
)

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)
secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []


@app.function(
    image=image,
    gpu=GPU,
    region=REGION,
    routing_region=ROUTING_REGION,
    cpu=4,
    memory=32 * 1024,
    volumes={"/cache": cache},
    secrets=secrets,
    env={"TALKERS": TALKERS, "MODAL_REGION": ",".join(REGION)},
    max_containers=1,  # one user, one container: the WebRTC peer is always this container
    scaledown_window=5 * 60,  # GPU shuts down 5 min after the page stops pinging
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    from talker_voice.server import app as fastapi_app

    return fastapi_app


@app.function(image=image, cpu=2, memory=8 * 1024, timeout=20 * 60)
def smoke():
    """Cheap CPU-only check that every venv imports cleanly (run before spending GPU time):

    modal run modal_app.py::smoke
    """
    import subprocess

    checks = {
        "asr (NeMo)": ["/opt/asr/bin/python", "-c", "import nemo.collections.asr, numpy; print('nemo ok, numpy', numpy.__version__)"],
        "tts (Kokoro)": ["/opt/tts/bin/python", "-c", "from kokoro import KModel, KPipeline; import spacy; spacy.load('en_core_web_sm'); print('kokoro ok')"],
        "vllm": ["/opt/vllm/bin/python", "-c", "import vllm; print('vllm', vllm.__version__)"],
        "bot (Pipecat)": ["python", "-c", "import talker_voice.bot, talker_voice.server; print('bot ok')"],
    }
    failed = []
    for name, cmd in checks.items():
        r = subprocess.run(cmd, capture_output=True, text=True, cwd="/root")
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print(f"[{'OK' if r.returncode == 0 else 'FAIL'}] {name}: {' | '.join(tail)}")
        if r.returncode:
            failed.append(name)
    if failed:
        raise SystemExit(f"FAILED: {failed}")


@app.function(image=image, gpu="T4", volumes={"/cache": cache}, timeout=20 * 60)
def asr_check():
    """Cheap GPU check of the Parakeet sidecar (load + warm-up transcription): modal run modal_app.py::asr_check"""
    import subprocess

    code = "import runpy; runpy.run_path('/root/talker_voice/sidecars/asr_server.py', run_name='check'); print('ASR_OK')"
    r = subprocess.run(["/opt/asr/bin/python", "-c", code], capture_output=True, text=True)
    print((r.stdout + r.stderr)[-1500:])
    if "ASR_OK" not in r.stdout:
        raise SystemExit("ASR check failed")
