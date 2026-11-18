# Talker Voice Lab

Talk to **Gemma 4 E2B** and **Gemma 4 E4B** as voice "talkers" and feel the difference.
Everything runs in **one Modal GPU container in Mumbai (`ap-south`)**, with peer-to-peer WebRTC
to your browser (no SFU), and per-turn latency shown in the UI.

```
browser mic ──WebRTC (P2P)──► Silero VAD + Smart Turn v3 ──► Parakeet TDT 0.6B v3 (STT, whole utterance)
                                                               │
             speaker ◄──WebRTC── Kokoro 82M (TTS, per sentence) ◄── sentence chunker ◄── SayExtractor
                                                                                          ▲
                                     Gemma 4 E2B / E4B on vLLM, JSON-constrained: {"route", "say"}
```

* **Structured output:** vLLM guided decoding forces `{"route": INSTANT|DELEGATE|CLARIFY, "say": "..."}`.
  `route` is generated first; `say` is streamed to TTS *while* it is generated (no waiting for the JSON to close).
  There is **no thinker yet**: `route` is shown as "what the talker would do", and the talker always answers.
* **Chunking:** Pipecat's sentence aggregator (handles `Dr.`, `3.5`, etc.) sends one sentence per Kokoro request;
  the prompt asks for a short 3–8 word first sentence so audio starts fast. Kokoro trims silence and adds a
  fixed 180 ms pause between sentences so it sounds even.
* **Turn-taking:** Silero VAD (0.2 s) + Smart Turn v3 decides you're done (semantic + intonation), Parakeet then
  transcribes the whole segment in tens of ms. Barge-in (interrupting the bot) works; use headphones.

## Connection: LiveKit Cloud (default)

Direct peer-to-peer WebRTC to a Modal container is blocked on many networks, so the default is
LiveKit Cloud (free Build plan: 5,000 participant-minutes/month, hard cap, no card). On **Start talking**
the server creates a fresh room, the bot joins it from the GPU container, and the page joins with
`livekit-client`. Create the credentials once in the Modal dashboard → **Secrets** → Custom, named
**`livekit`**:

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=API...
LIVEKIT_API_SECRET=...
```

(cloud.livekit.io → your project → Settings → Keys.) "Direct peer-to-peer" is still in the dropdown.

## Deploy (from your laptop, ~2 commands)

Modal's client talks gRPC, which the Claude cloud sandbox can't proxy, so deploy from your machine:

```bash
pip install modal
modal profile activate omnaidu42        # the profile you created with `modal token set ... --profile=omnaidu42`

git clone https://github.com/omnaiduu/Voice.git && cd Voice
git checkout claude/adoring-einstein-5h40aj
modal deploy modal_app.py
```

Open the printed URL (`https://<workspace>--talker-voice-web.modal.run`).

* **First deploy:** the image build takes ~10–20 min (vLLM + NeMo + Kokoro). Later deploys reuse it.
* **First page load / cold start:** the page shows each model loading (ASR, TTS, E2B, E4B). The first time
  downloads weights into a Modal Volume (a few minutes); after that a cold start is ~2–4 min (vLLM loading +
  CUDA graphs). When everything is green, press **Start talking**.
* Switch **Talker model** in the dropdown to compare E2B vs E4B (both are loaded, switching is instant).

## Cost (one user)

| Setting | GPU | ~Cost while the container is up |
|---|---|---|
| default (`ap-south`, both talkers) | L40S 48 GB | ≈ $3.5–4 / hour (L40S $1.95/h + CPU/RAM, × 1.75 for pinning to Mumbai) |
| `VOICE_REGION=ap` (anywhere in Asia-Pacific) | L40S | ≈ $2.5 / hour (× 1.15), maybe +30–60 ms network |
| `TALKERS=gemma-4-e2b VOICE_GPU=L4` | L4 24 GB | ≈ $1.6 / hour in Mumbai |

* It **scales to zero**: the page pings the server while open; 15 min after your last call the page stops pinging
  and the GPU shuts down 10 min later. Closing the tab = shuts down after ~10 min. Cost when idle: $0.
* Modal's free Starter plan includes $30/month.
* Stop it immediately any time: `modal app stop talker-voice`.

Deploy-time knobs (set on your machine before `modal deploy`):

```bash
VOICE_REGION=ap-south          # or "ap", "ap-south,ap", ...
VOICE_GPU=L40S,A100-80GB       # first available wins; both talkers need 48 GB+
TALKERS=gemma-4-e2b,gemma-4-e4b
HF_TOKEN=hf_...                # only if Hugging Face asks for auth
VLLM_EXTRA_ARGS="--kv-cache-dtype fp8"   # optional extra vLLM flags (set as a Modal env if needed)
```

## Compare 2B vs 4B routing (text, no audio)

```bash
python eval/routing_eval.py https://<workspace>--talker-voice-web.modal.run
```

Prints route accuracy, valid-JSON rate, TTFT / full-reply latency (p50/p90), a confusion matrix and the
misrouted examples for each talker. Add your own lines to `eval/routing_cases.jsonl` (60 starter cases).
JSON validity should be 100% (guided decoding); accuracy is the real 2B-vs-4B signal.

## Reading the latency chips

Each bot reply shows times measured from **the moment your voice stopped**:

* `turn end +X` – when VAD + Smart Turn decided you were done (biggest knob for "snappiness")
* `LLM 1st tok` – talker time-to-first-token
* `say starts +X` – first spoken character decoded from the JSON
* `TTS audio +X` – first Kokoro audio for sentence one
* **`first word`** – first audio sent to you (add ~50–150 ms network + browser playback)

Wiring-only overhead measured without models (fake STT/LLM/TTS, `tests/e2e_fake_models.py`) is ~0.4 s;
real models add their compute on top.

## Troubleshooting

* **"WebRTC connection failed"**: your network blocks direct UDP. Try another network/hotspot; if needed we add a
  TURN server (e.g. Cloudflare Calls TURN or Twilio) to `ICE_SERVERS` in `talker_voice/server.py` and the page.
* **A model shows `crashed` / `timeout`**: `modal app logs talker-voice` shows the vLLM / NeMo / Kokoro output.
  Out-of-memory on start → lower `E2B_GPU_MEM` / `E4B_GPU_MEM` in `talker_voice/config.py` or use `A100-80GB`.
* **No GPU in ap-south for a long time**: deploy with `VOICE_REGION=ap-south,ap`.
* **Security**: the URL is public (anyone with it can use your GPU). Stop the app when you're done testing.

## Layout

```
modal_app.py                 Modal image (3 isolated venvs) + the single GPU web function
talker_voice/server.py       FastAPI: UI, /api/status, /api/offer (WebRTC), /api/chat (eval); boots sidecars
talker_voice/bot.py          Pipecat pipeline + latency observer
talker_voice/services.py     Parakeet STT, Kokoro TTS, SayExtractor (JSON → streamed "say")
talker_voice/structured.py   schema, talker system prompt, incremental JSON parser
talker_voice/sidecars/       asr_server.py (NeMo Parakeet), tts_server.py (Kokoro)
talker_voice/static/         the web UI
eval/                        2B vs 4B routing eval
tests/                       parser unit tests + end-to-end wiring test with fake models
```

Local checks (no GPU): `pytest tests/test_structured.py` and
`PYTHONPATH=. python tests/e2e_fake_models.py some_speech.wav` (needs `pipecat-ai[webrtc,silero,local-smart-turn,openai]`, `uvicorn`).
