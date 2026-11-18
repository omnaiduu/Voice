# Talker Voice Lab: engineering notes (source material for a blog post)

> Purpose of this file: a complete, factual record of what we built, why, what we tried, what broke,
> what we measured, how it felt, and what we decided **not** to do. A future writer (human or agent)
> should be able to write an elaborate engineering blog post from this file alone.
>
> Dates: work done on 17–18 Nov 2026. Repo: this one (`Voice`), branch `claude/adoring-einstein-5h40aj`.
> Numbers marked **(measured)** come from our own logs; numbers marked **(vendor)** are claims from
> model cards/blogs; numbers marked **(estimate)** are back-of-envelope.

---

## 1. The question that started it

"What is the **talker–thinker** architecture in voice AI, how do people build voice bots with it, and is
this what OpenAI does?" Then: "What are the pros/cons of a cascaded **STT → small LLM (talker, ~2B) →
TTS**, with a bigger model as the thinker? How do STT and TTS stream?" Then: "Build it, host it on
Modal, cheap, for one user, so I can talk to it and feel 2B vs 4B."

## 2. Research findings (the "what is talker–thinker" part)

There are **two different things** called talker/thinker. This distinction is the core insight of the blog.

### 2a. Thinker–Talker *inside one model* (Alibaba Qwen2.5-Omni, Mar 2025; later Qwen3-Omni)
- Source: Qwen2.5-Omni technical report, arXiv 2503.20215.
- **Thinker** = the LLM "brain": encoders for audio/vision + Transformer decoder, produces text and
  hidden states. **Talker** = a dual-track autoregressive decoder, the "mouth", that consumes the
  thinker's **hidden representations + the sampled text tokens** and emits speech codec tokens.
  - Hidden states carry tone/attitude so the voice can be expressive *before* the sentence is complete.
  - The discrete text tokens disambiguate phonetics (hidden states are semantic, not phonetic).
- Streaming: block-wise (2 s) audio encoder for prefill; a **sliding-window DiT** (flow matching, lookback 2
  blocks, lookahead 1) turns codec tokens into mel chunks; BigVGAN vocoder streams waveform.
- End-to-end jointly trained. It's a **model architecture**.

### 2b. Talker + Thinker as *two separate models* (what OpenAI and most production voice agents do)
- Google DeepMind "Agents Thinking Fast and Slow: A Talker-Reasoner Architecture" (2024) named the pattern.
- **OpenAI** docs call it the **"Responder-Thinker"** architecture (prompting guide for realtime models) and
  the open-source `openai-realtime-agents` repo calls it **"Chat-Supervisor"**:
  realtime speech-to-speech model = responder (talks), a stronger text model = thinker, called via a tool
  (`getNextResponseFromSupervisor`). Rules from OpenAI's guide: say a neutral filler that doesn't imply
  success/failure ("One moment", "Let me check"), never mention the supervisor, then **rephrase** the
  thinker's text into ≤2 short spoken sentences.
- **GPT-Live** (OpenAI, Jul 2026): full-duplex voice model (listens while speaking; decides many times per
  second whether to speak, listen, pause, interrupt, call a tool) that **delegates** search/reasoning to a
  frontier model (GPT-5.5 at launch) while it keeps the conversation going. OpenAI's Aug 2026 engineering
  post: media on a dedicated fast path (Go + WebRTC), delegation behind an **async RPC boundary**, frontier
  model session **pre-created and prefilled** at call start with session affinity + prompt caching; the whole
  delegation loop counted in the latency budget. (OpenAI has not published the voice model's internals.)
- OpenAI's voice-agents guide now lists three architectures: GPT-Live (client delegation or Responses
  delegation), a single Realtime API session, or a chained STT→LLM→TTS pipeline.
- Other implementations we looked at: LiveKit "subagent delegation" (async tool, `ctx.update()` makes it
  non-blocking, `CANCELLABLE` flag), ShipVoice "Talker and Thinker", UW **ConvFill** (135M–1.7B on-device
  talker produces filler until a cloud reasoner streams "knowledge chunks"; first sound ~0.3 s vs ~1 s),
  Nick Lackman's multi-thinker `responder-thinker` repo.
- **Takeaway:** Qwen's Thinker-Talker = model architecture (brain + mouth sharing hidden states).
  OpenAI's = **system** architecture (fast voice front end + delegated reasoning). For building today, copy
  the system pattern.

### 2c. Cascaded talker–thinker (STT + small LLM + TTS + big thinker): pros/cons we wrote down
Pros: cheap (small talker handles most turns), fast first word, can be private/on-device, text at every
stage (logs, guardrails, evals), swap any component, focused prompts.
Cons: loses paralinguistics (tone/emotion dropped by STT, TTS doesn't know context), latency stacking,
half-duplex turn-taking (needs VAD/turn detection/barge-in), **a 2B model is a weak router** (tends to
answer instead of delegating; may paraphrase the thinker's facts wrongly), STT errors propagate, two-brain
state sync, more engineering.
Mitigations: make routing a structured classification, "when unsure, DELEGATE", pass names/numbers from
the thinker verbatim, LoRA-finetune the small model on routing data.

## 3. Design decisions (and the reasoning)

| Decision | Choice | Why | Alternatives considered |
|---|---|---|---|
| Overall | Cascaded STT → talker LLM → TTS, **no thinker yet** | User wanted to feel 2B vs 4B talkers first | Speech-to-speech (GPT-Live/Realtime), thinker delegation (deferred) |
| Talker models | **Gemma 4 E2B** and **E4B** (Apache-2.0, Apr 2026), both loaded, switchable live | E2B ≈ 2.3B effective params, E4B ≈ 4.5B; native function calling + system role; user prefers 2B for speed | Gemma 2 2B (older), Qwen3 1.7B/4B, Qwen3-4B-Instruct (Modal's demo used it) |
| Structured output | vLLM guided decoding, JSON schema `{route, tone, say}` in that order | Guaranteed-valid JSON; `route` decided in ~1–3 tokens; `tone` known before speech; `say` streamed to TTS *while generated* (incremental JSON parser) | Free text; separate classifier call |
| Note on structured output | Guarantees **format**, not the **decision** | Routing accuracy must be evaluated separately (`eval/routing_eval.py`) | — |
| STT | **Parakeet TDT 0.6B v3** (NeMo), whole-utterance after turn end | Fast enough that streaming STT buys nothing for latency; Modal measured VAD-segmented Parakeet faster end-to-end than open streaming STTs; punctuated, accurate | Streaming **Parakeet Realtime EOU 120M** (EOU p50 160 ms, p90 280 ms (vendor), English only, no punctuation, 9.3% WER), Kyutai STT, Deepgram Flux, Whisper (not streaming-native; hallucinates on silence) |
| User insight we agreed with | "Parakeet is very fast, so why stream?" | Correct: the bottleneck is **endpointing** (deciding the user finished), not transcription | — |
| Turn detection | **Silero VAD (stop 0.2 s) + Pipecat Smart Turn v3** (audio model on CPU, <100 ms, 23 languages) | Built into Pipecat (default), multilingual, semantic+intonation; VAD = silence-based, Smart Turn = "are they really done" | Parakeet EOU model (could be ~100 ms faster; easy A/B later via the "turn end" chip) |
| TTS | **Kokoro 82M** (Apache), **sentence-chunked** | Most reliable open TTS; tiny; chunking by sentence gives fast first audio with any TTS | True streaming TTS (Kyutai, Qwen3-TTS streaming), Orpheus, Chatterbox |
| Chunking | Pipecat's sentence aggregator (handles `Dr.`, `3.5`), prompt asks for a short 3–8 word first sentence | First chunk short → first audio fast; split only at sentence ends (mid-sentence splits sound like list reading) | Custom chunker (we wrote one as a teaching example) |
| Framework | **Pipecat 1.16** | Transports, VAD, Smart Turn, interruptions, metrics all built in | LiveKit Agents |
| Hosting | **Modal**, ONE GPU container running everything (bot + vLLM×2 + Parakeet + Kokoro) on localhost | Lowest latency (no inter-service hops) and lowest cost for one user; scale-to-zero | Separate autoscaling services (Modal's own demo does this; better for many users) |
| Isolation | Separate virtualenvs: `/opt/vllm`, `/opt/asr` (NeMo), `/opt/tts` (Kokoro), `/opt/qwentts`, main (Pipecat) | vLLM/NeMo/Kokoro/qwen-tts pin conflicting torch/transformers/numpy | One env (dependency hell) |
| Region | `ap-south` (Mumbai) preferred, then `ap` | User is in India; `["ap-south","ap"]` bills at the cheaper 1.15× multiplier instead of 1.75× | `us-east` (Modal input plane, farther) |
| Transport (final) | **LiveKit Cloud** free Build plan | Direct P2P WebRTC to the Modal container **failed** on the user's network | P2P SmallWebRTC (kept as option), Daily (key issues), Cloudflare Realtime (more wiring), our own WebSocket audio (built, browser side not wired) |

## 4. Architecture (as built)

```
Browser (mic, speaker, UI with latency chips)
   │  WebRTC via LiveKit Cloud (default)  |  direct P2P SmallWebRTC (option)
   ▼
ONE Modal GPU container (ap-south preferred; L40S 48 GB or A100-80GB fallback)
 ├─ Pipecat bot
 │    ├─ Silero VAD (start 0.3 s, stop 0.2 s) + Smart Turn v3  → end of user turn
 │    ├─ ParakeetSTT  → POST localhost:8101 (whole utterance, PCM16 16 kHz)
 │    ├─ OpenAILLMService → vLLM localhost:8201 (E2B) | :8202 (E4B), JSON schema {route,tone,say}, stream
 │    ├─ SayExtractor: incremental JSON parse → forwards ONLY "say" text; reports route + tone
 │    ├─ SidecarTTS: sentence → POST localhost:8102 (Kokoro) | :8103 (Qwen3-TTS) with {text,voice,speed,tone}
 │    ├─ SoftInterruptOutput: real-time pacing (≈40 ms lead) + 300 ms fade-out on barge-in
 │    └─ LatencyObserver: per-turn timestamps → browser (data channel)
 ├─ /opt/asr   : Parakeet TDT 0.6B v3 (FastAPI)
 ├─ /opt/tts   : Kokoro 82M (FastAPI, trims silence, fixed 180 ms gap between sentences)
 ├─ /opt/qwentts: Qwen3-TTS 1.7B CustomVoice (FastAPI; tone → natural-language "instruct")
 └─ /opt/vllm  : two vLLM servers (Gemma 4 E2B, E4B), fixed GiB memory budgets
```

One turn: you stop talking → VAD 0.2 s + Smart Turn decides → Parakeet transcribes (tens of ms) →
Gemma streams `{"route":…, "tone":…, "say":"…` → first `say` sentence complete → TTS → audio out
while the rest is still generating. Interrupting → bot fades out over 0.3 s.

Files: `modal_app.py` (image + GPU function + smoke/asr checks), `talker_voice/server.py` (FastAPI, boot
supervisor, `/api/status`, `/api/livekit/start`, `/api/offer`, `/ws`, `/api/chat`), `talker_voice/bot.py`
(pipeline), `talker_voice/services.py` (STT/TTS/SayExtractor/SoftInterruptOutput),
`talker_voice/structured.py` (schema, prompt, parser), `talker_voice/sidecars/*`, `talker_voice/static/index.html`,
`eval/` (routing eval, 60 cases), `tests/` (parser unit tests, end-to-end test with fake models).

### Talker system prompt essentials
JSON only; route rules (INSTANT small talk/simple facts; DELEGATE deep reasoning/fresh data/account data;
CLARIFY ambiguous); tone from a fixed list; `say`: short first sentence (3–8 words), then natural 8–20 word
sentences, 2–4 sentences; no markdown/lists/emojis/URLs; numbers as words; occasional `[laugh]`/`[chuckle]`
allowed (stripped by Kokoro/Qwen sidecars). No thinker yet, so on DELEGATE it still gives its best answer.

## 5. Measurements

### 5a. Real calls (user talking through LiveKit, 18 Nov 2026) (measured, from Pipecat metrics in logs)

| Metric | Gemma 4 E2B (n=20 turns) | Gemma 4 E4B (n=14–15 turns) |
|---|---|---|
| LLM time-to-first-token | **median 48 ms** (min 40) | **median 52 ms** (min 47) |
| First LLM call after load (one-off warm-up) | 3.7 s | 5.0 s |
| Parakeet STT "TTFB" (from VAD stop, includes 0.2 s VAD wait) | median 270 ms | median 267 ms |
| Kokoro time-to-first-audio per sentence | median 106 ms (69–163) | median 123 ms (109–145) |

Example E4B turn timeline (measured): transcript ready and turn-end decided ~268 ms after VAD stop; LLM
first token +56 ms; first sentence ready for TTS +~220 ms (route+tone tokens + first sentence);
Kokoro first audio +109 ms → **first audio out ≈ 0.4 s after the turn end, ≈ 0.65 s after the voice
actually stopped**, plus LiveKit/network (~50–150 ms, estimate).

Text-only `/api/chat` checks on E2B (measured): warm TTFT **41 ms**, full JSON reply **~180–210 ms**;
routes correct on 3/3 probes ("Hi, how are you?" → INSTANT; "What's the weather in Mumbai right now?" →
DELEGATE; "Book it." → CLARIFY).

### 5b. Pipeline overhead without models (fake STT/LLM/TTS, real Pipecat + real WebRTC) (measured locally)
~0.28 s from voice end to turn-end decision (VAD 0.2 s + Smart Turn), **~0.40 s to first bot audio** with
instant fake models. Sentence chunking verified: `"Sure!"` → `"It looks warm and sunny in Mumbai today."` →
`"Carry some water, Dr. Smith said 3.5 litres."` (no split at "Dr." or "3.5").

### 5c. Soft interruption (measured locally, fake models)
On barge-in the bot's audio RMS ramps ~4000 → 62 over ~300 ms (`[…4007, 3626, 3113, … 741, 525, 332, 175, 62]`)
instead of a hard cut. Tone `cheerful` reached the TTS for every sentence.

### 5c-bis. Why the fade sometimes did NOT happen (user report, 18 Nov) and the fix
User: "sometimes with the 4B, the volume is not decreasing the way I had told you". Root cause was in the
audio path, not the model: Kokoro returns a **whole sentence as one PCM chunk** (often 2–4 s). Our
`SoftInterruptOutput` paced frames in real time, but it released a frame as a unit, so a whole sentence was
handed to LiveKit at once. LiveKit's `AudioSource` keeps a ~1 s internal queue
(`audio_out_queue_size_ms=1000`) and on interruption calls `clear_queue()` → hard cut of exactly the
audio we wanted to fade. The fade only works on audio we still hold. With the fake-TTS test the chunks were
small (4801 bytes), so the bug was invisible locally. 4B seemed worse only because its replies were longer
(bigger sentences = bigger chunks), and by chance.
**Fix** (commit ac32e56): slice every incoming audio frame into **20 ms pieces** before queueing, then pace
with a 40 ms lead. Now at most ~40 ms is inside LiveKit when barge-in happens and the next 300 ms is
still ours to ramp down. The e2e test got a `BIG_CHUNKS=1` mode that mimics Kokoro; it fails before the fix
and passes after. Not yet re-confirmed by the user live (credits ran out right after).

### 5d. Cold start and resources (measured)
- First image build: ~7.7 min (`App deployed in 461 s`); later deploys reuse layers (~3–66 s).
- Cold start with cached weights: Kokoro ready ~23–36 s, Parakeet ~45–58 s, E2B ~3 min, **all four ~5 min (312 s)**.
- vLLM: E2B weights **9.18 GiB**, E4B **14.5 GiB** (measured). The run landed on an **A100-80GB**
  (Modal fallback; fraction-based `gpu-memory-utilization` 0.32/0.42 then gave huge 15–18 GiB KV caches).
  Fixed afterwards: budgets in GiB (E2B 14, E4B 19.5) converted to a fraction for whichever GPU we get.
- **Parallel boot** (commit 99bb149): before, sidecars started one after another and the talkers waited for
  the remote Qwen service. Now Kokoro, Parakeet and the first talker start together; remote TTS boots in a
  background thread and never blocks. Measured on the last boot (18 Nov, L40S-class GPU, cached weights):
  Kokoro ready **30 s**, Parakeet **52 s**, **E2B 129 s (~2:10)**, E4B 228 s (~3:50). Before: E2B ~5 min.
  The two talkers still load one after another (two vLLM processes starting together on one GPU fight over
  memory profiling).
- Modal prices (Nov 2026): L40S $1.95/h, A100-80GB $2.50/h, L4 $0.80/h; region pinning ×1.15 (broad) or
  ×1.75 (narrow). Our setup ≈ $2.5–4/h while running; scale-to-zero when idle; Starter plan $30/month credit.

## 6. How it felt (user feedback, verbatim-ish)
- "The bot is doing very fine. The 2B and the 4B. I couldn't see most of the delays."
- "**2B is snappy**." "**4B is also very reasonable.**"
- "The STT is working great."
- "It is speaking in small sentences" → by design (prompt asked for 8–15 word sentences, 1–3 sentences);
  we relaxed to 8–20 words, 2–4 sentences, varied rhythm.
- Interruptions felt abrupt: "a better option would be to gradually lower the volume, like naturally
  someone…" → we built the 300 ms fade (SoftInterruptOutput) and raised VAD start to 0.3 s.
- Wanted more **expressiveness** from TTS → Qwen3-TTS with talker-chosen tone.
- Later session (18 Nov): "I see there is **no difference between the 4B and the 2B** model", "I think
  **Kokoro is the only better option**. At the smaller level, it is fast", "the STT is also working great",
  "the **turn detection** … is working completely **flawlessly**". Conclusion for the blog: for a talker
  that only has to produce 2–4 short sentences + a route + a tone, **E2B is enough**; the 4B buys nothing
  audible. Pick E2B (smaller GPU, faster boot).
- On Qwen3-TTS (reference implementation): it was "too slow" and "not using the tags properly"
  (`[laugh]` was read out / ignored). Loading time was the main complaint: "Loading takes so much long".

## 7. Things that broke and how we fixed them (good "war stories")
1. **Couldn't deploy from the Claude cloud sandbox**: Modal's client speaks gRPC. First diagnosis was
   "the sandbox proxy doesn't carry gRPC". The user said "download the Modal proxy"; debug logs showed
   Modal *does* support HTTP proxies but needs `pip install 'modal[api-proxy-support]'` (python-socks).
   After that, deploy/logs/stop all worked from the sandbox. Lesson: read the client's debug log before
   concluding a network path is impossible.
2. **STT crash #1**: `ImportError: pyarrow requires NumPy 2.0` → I had copied a `numpy<2` pin from Modal's
   older NeMo 2.3 example; NeMo 2.7 needs numpy 2. Fix: drop the pin. Added a **CPU smoke test**
   (`modal run modal_app.py::smoke`) that imports every venv before spending GPU money.
3. **STT crash #2**: `RuntimeError: mat1 and mat2 must have the same dtype, Float and BFloat16` → I had
   cast Parakeet to bf16, but NeMo's CUDA-graph TDT decoder feeds fp32. Fix: keep fp32 weights + bf16
   autocast. Verified on a cheap **T4** (`modal run modal_app.py::asr_check`) before redeploying.
4. **Direct WebRTC failed** in the user's browser ("WebRTC connection failed"): P2P from the browser to a
   Modal container didn't traverse the NATs. Options: TURN server, our own WebSocket audio, or an SFU.
5. **Daily** (first SFU choice; Pipecat is made by Daily): API returned 401 twice. The first value saved
   was the account domain (`cloud-…`, Pipecat-Cloud style), the second also rejected. Switched to LiveKit.
6. **LiveKit Cloud**: worked first time. The Modal workspace already had a `livekit` secret from an older
   project (bankbot); verified it with a tiny Modal function (list rooms) instead of overwriting it.
7. **Sandbox restarts** killed background watchers mid-deploy; Modal itself was unaffected.
8. **Qwen3-TTS reference implementation too slow**: first version ran `qwen-tts` (the HF reference package)
   in-process in our GPU container. Warm, it took **6.89 s per sentence** (no streaming, full sentence then
   play) → unusable for conversation. Also crashed once on start because our image sets
   `HF_HUB_ENABLE_HF_TRANSFER=1` but that venv had no `hf_transfer` (fix: set it to 0 for that sidecar).
9. **Tags**: Qwen3-TTS CustomVoice doesn't read inline `[laugh]` tags; it takes a natural-language
   `instructions` field. Fix: `qwen_instructions(tone, text)` strips the tags from the text and folds them
   into the instruction ("…with a light laugh at the start"). The prompt also had to say "You CAN laugh…
   never say you can't laugh" because Gemma kept apologising that it is an AI and cannot laugh.
10. **Moved Qwen3-TTS to vLLM-Omni** on its own Modal GPU function (`qwen_tts`, `@modal.web_server`,
    `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --deploy-config …/qwen3_tts.yaml`, OpenAI-style
    `/v1/audio/speech` with `stream: true, response_format: pcm`). Separate GPU so it can't starve the
    talkers, and separate image because vLLM-Omni 0.32 pins vllm 0.32 while the talkers use vllm 0.34.
11. **vLLM-Omni crash #1**: `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'`
    (the sampler JIT-compiles a kernel). The image was `debian_slim` + pip CUDA wheels, which has no nvcc.
    Fix: base the image on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` and set `CUDA_HOME` (commit 226aaec).
12. **vLLM-Omni crash #2 (unsolved, the last thing we saw)**: model weights load fine (9.3 GiB, KV 4.2 GiB),
    then stage 0 captures CUDA graphs including the multi-token-prediction ("MRv2 mtp") graphs for sizes
    64…1, and the whole stage exceeds vLLM-Omni's **built-in 300 s `stage_init_timeout`**:
    `TimeoutError: LLM stage-0 replica-0 did not become ready within 300s (stage_init_timeout)`.
    Likely fixes, not tried: raise the stage init timeout (deploy config / CLI flag), run with
    `enforce_eager` (skip graph capture; slower decode), capture fewer graph sizes, or use a faster GPU
    than L4/A10. The 0.6B model would also start faster.
13. **Credit leak from health polling**: the main container polls the remote TTS `/health` for up to 25
    minutes and pings it every 2 min to keep it warm. When Qwen kept crashing, every poll made Modal start
    a **fresh GPU container**, which crashed again ~5 min later. Stopping containers by hand didn't help
    (a new one appeared within seconds). What worked: cap the function at zero containers
    (`modal.Function.from_name("talker-voice","qwen_tts").update_autoscaler(max_containers=0)`), then stop
    the pending ones. Lesson: a keep-warm loop must give up after N failures, and an optional engine should
    be opt-in. Afterwards the default TTS list was changed to Kokoro only (`TTS_ENGINES=kokoro`).
14. **GPU capacity in India**: one deploy sat waiting because `ap-south`/`ap` had no free L40S/A100; we
    waited rather than leave the region (latency to the user matters more than a few minutes).
15. **Monitoring from the sandbox**: tailing `modal app logs` in a loop replayed old log lines and
    confused progress; polling our own `/api/status` JSON was far more reliable.
16. Cosmetic: Pipecat logs `STTSettings/TTSSettings NOT_GIVEN` as ERROR; harmless.

## 8. Things we deliberately did NOT do (and why)
- **No thinker model yet**: user wanted to test the talkers alone first. `route` is diagnostic only.
- **No speech-to-speech model** (GPT-Live/Realtime/Qwen-Omni): goal was an open, cheap, inspectable cascade.
- **No audio-in to Gemma** (Gemma 4 E2B/E4B accept audio): no turn detection, 30 s clip limit, no
  transcript for logs/thinker, lower ASR accuracy than Parakeet. Possible future A/B.
- **No streaming STT / Parakeet EOU model**: batch Parakeet after Smart Turn was good enough; EOU model is
  English-only, no punctuation; may save ~100 ms; easy to A/B later.
- **No Chatterbox**: retired by Resemble AI (Aug 2026, archive only), pins torch 2.6 / old deps (another
  heavy venv), always watermarks, ~3 GB more VRAM. Chose **Qwen3-TTS** (maintained, Apache-2.0,
  natural-language style control) instead.
- **No Higgs TTS 3 / Fish S2-Pro**: very expressive but non-commercial licenses. Fish Drama 3: community
  license (<$500K revenue), 35B MoE.
- **No Daily**: key problems; LiveKit worked. **No Cloudflare Realtime**: lower-level, more wiring.
- **No Tailscale / running on the user's laptop**: not needed once the sandbox could drive Modal.
- **No GPU memory snapshots** yet (could cut cold start to seconds; experimental with 5 GPU processes).
- **No TURN server**: LiveKit solved connectivity. **WebSocket audio transport** exists server-side
  (`/ws`, raw PCM16 + JSON events + `clear` on barge-in) but the browser client isn't wired.
- **No "duck and resume"** on false interruptions (fade then come back if it was a cough): designed, not built.

## 9. TTS landscape research (Nov 2026) (for the "why is open TTS hard" section)
Why open TTS lags ElevenLabs/Cartesia in production: (1) expressive models are autoregressive codec LMs
→ hallucinate/skip/repeat words, drift on long text (e.g. Qwen3-TTS speaking-rate drift >100 chars, issue
#239); stable non-AR models (Kokoro) sound flat; (2) labs ship weights, not serving (streaming + batching);
Voxtral TTS 70 ms → 552 ms at 32 concurrent; Qwen3-TTS "streaming" initially returned full audio;
(3) best expressive models are non-commercial; (4) expressive conversational data is expensive/private;
(5) TTS doesn't know conversational context unless the LLM tells it (→ our `tone` field).

| Model | License | Notes |
|---|---|---|
| Qwen3-TTS 0.6B/1.7B (Jan 2026) | Apache-2.0 | instruct style control, voice design, 3 s cloning, 10 langs, ~97 ms (vendor) / 180–230 ms (independent) |
| Orpheus 3B | Apache + Llama terms | tags `<laugh>`, vLLM serving, ~200 ms; most battle-tested expressive |
| Chatterbox Turbo/Multilingual V3 | MIT | natural, `[laugh]` tags, watermark, weak concurrency, **retired Aug 2026** |
| CosyVoice 3 0.5B | Apache-2.0 | instruct emotion, bi-streaming ~150 ms |
| Ming-3.0-tts (Sep 2026) | MIT | top open model on Artificial Analysis (1188), 14 tags, no streaming documented |
| ZONOS3-Mini (Nov 2026) | Apache-2.0 | word-level emotion, 43 langs, brand new |
| Fish Audio Drama 3 (Oct 2026) | community (<$500K) | very expressive, 35B MoE |
| Higgs TTS 3 | non-commercial | 21 emotions, sfx tags |
| Kokoro 82M | Apache-2.0 | fast, stable, flat; our baseline |

Transports compared: Daily free 10,000 participant-min/month; LiveKit Build free 5,000 participant-min,
1,000 agent-min, hard cap, no card; Cloudflare Realtime 1 TB free.

## 10. Current state and next steps (as of writing)
**Final state (18 Nov 2026, end of the project's first phase: Modal credits used up)**
- **Live-tested and working**: LiveKit + Parakeet + Gemma E2B/E4B + Kokoro, latency chips, `tone` field,
  soft barge-in fade, VAD start 0.3 s, longer/more natural replies, fixed-GiB vLLM budgets, parallel boot.
- **Recommended config**: Parakeet + Gemma 4 **E2B** + **Kokoro**, LiveKit transport, one GPU
  (`TALKERS=gemma-4-e2b`, an L4/L40S is enough). ~0.4 s from end of turn to first audio.
- **Fade fix with Kokoro**: built and passes the e2e test in Kokoro-like mode; not re-confirmed live.
- **Qwen3-TTS**: reference implementation ran (6.9 s/sentence, too slow); vLLM-Omni version gets past
  loading but dies on the 300 s stage-init timeout. **Off by default now** (opt in with
  `TTS_ENGINES=kokoro,qwen3-tts`). Its real streaming TTFA on our setup was never measured.
- Modal: app `talker-voice` is deployed but scaled to zero (0 containers); `qwen_tts` was capped at
  max_containers=0 at runtime (a redeploy resets that).
- Cost: the billing report showed **~$4.65 for talker-voice** this month (before the last day's runs were
  fully counted); the workspace's $30 monthly credit was used up overall (other projects ~$25).
- Next candidates: add the thinker (delegate on DELEGATE, filler, rephrase), routing eval at scale + LoRA
  for E2B, Parakeet-EOU A/B, duck-and-resume, Qwen3-TTS streaming (vLLM-Omni), GPU snapshots, password on URL.

## 10b. Cost lessons (for the blog)
- Scale-to-zero (`scaledown_window=300`, `max_containers=1`) kept a demo with 5 models on one GPU at a few
  dollars total for days of on-and-off testing.
- The expensive part was not talking to the bot, it was **cold starts and failed boots** (each failed
  Qwen boot ≈ 5–6 GPU-minutes). Do CPU smoke tests and cheap-GPU checks (T4) before a full deploy.
- Anything that auto-wakes a GPU service (health checks, keep-alive pings) needs a failure budget.
- Region pinning to India costs ×1.15 on Modal; worth it for latency, but capacity can be thin.

## 11. Reproduce
```bash
pip install 'modal[api-proxy-support]'   # proxy extra needed when behind an HTTP proxy
modal run modal_app.py::smoke            # CPU: all venvs import
modal run modal_app.py::asr_check        # T4: Parakeet loads + transcribes
modal deploy modal_app.py                # app "talker-voice", region ap-south/ap
modal app logs talker-voice              # watch boot + calls
modal app stop talker-voice              # stop GPU immediately
python eval/routing_eval.py <app-url>    # 2B vs 4B routing accuracy + latency
PYTHONPATH=. python tests/e2e_fake_models.py speech.wav   # local wiring test, no GPU
```
Requires a Modal secret `livekit` (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET).

## 12. Security / hygiene notes
- A Modal token was pasted in chat during setup: **rotate it**.
- The app URL is public; anyone with it can spend GPU credits. Stop the app after testing; add auth later.
- Third-party keys (Daily, LiveKit) were kept out of chat: Daily via the sandbox's credential form,
  LiveKit via a Modal secret.

## 13. Sources
Qwen2.5-Omni report (arXiv 2503.20215); OpenAI: "Prompting Realtime models" (Responder-Thinker),
"Introducing GPT-Live", "How we built a realtime system for responsive voice AI in six months", Voice agents
guide; Google DeepMind Talker-Reasoner (2024); LiveKit "Talker-Reasoner pattern" blog; ConvFill (UW);
Modal "One-second voice-to-voice latency with Modal, Pipecat, and open models" + `open-source-av-ragbot`;
Pipecat docs (SmallWebRTC, Smart Turn, choosing a transport); NVIDIA `parakeet_realtime_eou_120m-v1` card,
NeMo voice agent docs, Nemotron streaming ASR blog; Kyutai STT; Deepgram Flux docs; Gemma 4 model card;
vLLM Gemma 4 recipe; Qwen3-TTS repo/model card; Chatterbox repo; open-TTS comparisons (micdrop.dev,
softcery.com, dograh.com, quantacloud.net, ai-opensource.com, lokahq benchmark, Picovoice TTS benchmark,
arunbaby.com cost cliff); Fish Audio Drama 3 report; Higgs TTS 3 card; Modal pricing + region docs;
LiveKit and Daily pricing.
