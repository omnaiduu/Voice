"""End-to-end wiring test WITHOUT a GPU.

Runs the real FastAPI server + real Pipecat pipeline + a real WebRTC client
(aiortc) that "speaks" a WAV file. ASR, the talker LLM and TTS are replaced by
tiny fake HTTP servers on the same ports, so this checks everything except the
models themselves: signalling, WebRTC audio both ways, VAD + turn detection,
STT call, JSON streaming -> SayExtractor -> sentence chunking -> TTS, and the
data-channel events the UI shows.

    python tests/e2e_fake_models.py path/to/speech.wav
"""

import asyncio
import json
import math
import struct
import sys
import time

import uvicorn
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

from talker_voice import server
from talker_voice.config import ASR_PORT, TALKERS, TTS_PORT

WAV = sys.argv[1]
REPLY = {"route": "INSTANT", "say": "Sure! It looks warm and sunny in Mumbai today. Carry some water, Dr. Smith said 3.5 litres."}
tts_requests: list[str] = []


async def fake_asr(req):
    body = await req.read()
    return web.json_response({"text": f"hello what is the weather ({len(body)} bytes)", "ms": 12.0})


async def fake_tts(req):
    data = await req.json()
    tts_requests.append(data["text"])
    resp = web.StreamResponse()
    await resp.prepare(req)
    sr = 24000
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / sr))) for i in range(sr // 2))
    for i in range(0, len(pcm), 4801):  # odd chunk size on purpose (tests int16 re-alignment)
        await resp.write(pcm[i : i + 4801])
    return resp


async def fake_vllm(req):
    body = await req.json()
    assert body.get("response_format", {}).get("type") == "json_schema", body.get("response_format")
    assert body.get("stream") is True
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    text = json.dumps(REPLY)
    for i in range(0, len(text), 3):
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                 "choices": [{"index": 0, "delta": {"content": text[i : i + 3]}, "finish_reason": None}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(0.005)
    end = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    await resp.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
    return resp


async def start_fakes():
    runners = []
    for port, routes in [
        (ASR_PORT, [web.post("/transcribe", fake_asr)]),
        (TTS_PORT, [web.post("/tts", fake_tts)]),
        (TALKERS["gemma-4-e2b"]["port"], [web.post("/v1/chat/completions", fake_vllm)]),
    ]:
        a = web.Application()
        a.add_routes(routes)
        r = web.AppRunner(a)
        await r.setup()
        await web.TCPSite(r, "127.0.0.1", port).start()
        runners.append(r)
    return runners


async def main():
    await start_fakes()
    server.ensure_started = lambda: None  # don't boot real sidecars
    server.STATUS.update({"asr": "ready", "tts": "ready", "gemma-4-e2b": "ready"})
    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=8800, log_level="warning")
    srv = uvicorn.Server(cfg)
    asyncio.create_task(srv.serve())
    await asyncio.sleep(1.5)

    pc = RTCPeerConnection()
    player = MediaPlayer(WAV)
    pc.addTransceiver(player.audio, direction="sendrecv")
    pc.addTransceiver("video", direction="recvonly")
    dc = pc.createDataChannel("app")
    msgs, audio_frames = [], [0]

    @dc.on("message")
    def on_msg(m):
        msgs.append(json.loads(m))
        print("EVENT", m[:200])

    @dc.on("open")
    def on_open():
        async def ping():
            while dc.readyState == "open":
                dc.send("ping")
                await asyncio.sleep(1)
        asyncio.ensure_future(ping())

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            async def drain():
                while True:
                    try:
                        await track.recv()
                        audio_frames[0] += 1
                    except Exception:
                        return
            asyncio.ensure_future(drain())

    await pc.setLocalDescription(await pc.createOffer())
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.post("http://127.0.0.1:8800/api/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "talker": "gemma-4-e2b"}) as r:
            assert r.status == 200, await r.text()
            ans = await r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))

    t0 = time.time()
    while time.time() - t0 < 25:
        if any(m.get("type") == "metrics" for m in msgs):
            await asyncio.sleep(2)
            break
        await asyncio.sleep(0.2)

    types = [m["type"] for m in msgs]
    print("\nevent types:", types)
    print("TTS sentence requests:", tts_requests)
    print("received bot audio frames:", audio_frames[0])
    metrics = [m for m in msgs if m["type"] == "metrics"]
    print("metrics:", metrics)
    ok = ("user" in types and "bot" in types and metrics and len(tts_requests) >= 2 and audio_frames[0] > 0)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    await pc.close()
    srv.should_exit = True
    await asyncio.sleep(0.5)
    sys.exit(0 if ok else 1)


asyncio.run(main())
