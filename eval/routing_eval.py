"""Compare talkers (Gemma 4 E2B vs E4B) on routing accuracy and speed.

Runs each case in routing_cases.jsonl through the deployed app's /api/chat
(same system prompt + JSON schema the voice bot uses, temperature 0).

    python eval/routing_eval.py https://<your-workspace>--talker-voice-web.modal.run

Prints, per talker: route accuracy, valid-JSON rate, TTFT and full-reply
latency (p50 / p90), a confusion matrix, and the misrouted cases.
Note: guided decoding guarantees the JSON *format*; accuracy measures the *decision*.
"""

import json
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

URL = sys.argv[1].rstrip("/")
CASES = [json.loads(line) for line in (Path(__file__).parent / "routing_cases.jsonl").read_text().splitlines() if line.strip()]
ROUTES = ["INSTANT", "DELEGATE", "CLARIFY"]


def post(path, payload):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


status = json.loads(urllib.request.urlopen(URL + "/api/status", timeout=60).read())
talkers = [t for t, s in status["status"].items() if t in status["talkers"] and s == "ready"]
if not talkers:
    sys.exit(f"No talker ready yet: {status['status']}")

for talker in talkers:
    post("/api/chat", {"talker": talker, "text": "hello"})  # warm-up
    ok = valid = 0
    ttft, total = [], []
    confusion = Counter()
    misses = []
    for c in CASES:
        r = post("/api/chat", {"talker": talker, "text": c["text"]})
        ttft.append(r["ttft_ms"])
        total.append(r["total_ms"])
        got = (r["parsed"] or {}).get("route")
        valid += r["parsed"] is not None
        confusion[(c["route"], got)] += 1
        if got == c["route"]:
            ok += 1
        else:
            misses.append((c["route"], got, c["text"], (r["parsed"] or {}).get("say", r["raw"])[:90]))
    n = len(CASES)
    print(f"\n=== {status['talkers'][talker]} ===")
    print(f"route accuracy : {ok}/{n} = {ok / n:.0%}")
    print(f"valid JSON     : {valid}/{n}")
    print(f"TTFT ms        : p50 {pct(ttft, 50):.0f}  p90 {pct(ttft, 90):.0f}")
    print(f"full reply ms  : p50 {pct(total, 50):.0f}  p90 {pct(total, 90):.0f}  mean {statistics.mean(total):.0f}")
    print("confusion (rows = expected, cols = got):")
    print("            " + "".join(f"{r:>10}" for r in ROUTES))
    for exp in ROUTES:
        print(f"{exp:>10}  " + "".join(f"{confusion[(exp, got)]:>10}" for got in ROUTES))
    for exp, got, text, say in misses:
        print(f"  MISS expected {exp:<8} got {str(got):<8} | {text} -> {say}")
