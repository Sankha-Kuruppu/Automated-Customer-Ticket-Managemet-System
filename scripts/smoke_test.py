"""End-to-end check against a running service (local container or hosted URL).

Usage:
    BASE_URL=http://localhost:8000 API_KEY=<key> python scripts/smoke_test.py [--jsonl data/validation.jsonl] [--repeat 1]

Mimics the organizers' harness: health, auth, single predict, sync batch, then an async job with
every ticket from the JSONL file, polled to completion and paged back. Prints accuracy if labels
are present, and /health latency while the job is running.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
KEY = os.environ.get("API_KEY", "")


def call(method, path, body=None, headers=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", "X-API-Key": KEY, **(headers or {})}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, r.headers, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, e.headers, (json.loads(raw) if raw else None)


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        check.failed += 1


check.failed = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/validation.jsonl")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the file N times (load test, max 5000)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    tickets = []
    for k in range(args.repeat):
        for r in rows:
            tickets.append({"ticket_id": f"{r['ticket_id']}#{k}", "channel": r["channel"],
                            "subject": r.get("subject") or "", "text": r["text"]})
    tickets = tickets[:5000]
    truth = {f"{r['ticket_id']}#{k}": r for k in range(args.repeat) for r in rows}

    print(f"Target: {BASE}")
    if not KEY:
        sys.exit("API_KEY is not set. Set BASE_URL and API_KEY first, for example (PowerShell):\n"
                 '  $env:BASE_URL="https://<host>"\n  $env:API_KEY="<key>"')
    try:
        s, _, b = call("GET", "/health", headers={"X-API-Key": ""}, timeout=30)
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach {BASE} ({e.reason}). Is the service running, and is BASE_URL set correctly?")
    check("health 200", s == 200, str(b))
    version = (b or {}).get("model_version")

    s, h, b = call("POST", "/predict", tickets[0], headers={"X-API-Key": "wrong"})
    check("wrong key -> 401", s == 401 and h.get("WWW-Authenticate") == "Bearer")
    s, _, b = call("POST", "/predict", {"channel": "chat", "text": "   "})
    check("empty text -> 422", s == 422)
    s, _, b = call("POST", "/predict", tickets[0])
    check("predict 200", s == 200 and b.get("model_version") == version, json.dumps(b, ensure_ascii=False))

    t = time.time()
    s, _, b = call("POST", "/predict/batch", {"tickets": tickets[:100]})
    check("batch(100) 200", s == 200 and len(b["predictions"]) == min(100, len(tickets)),
          f"{time.time() - t:.1f}s")

    t = time.time()
    s, h, job = call("POST", "/batch/jobs", {"tickets": tickets}, headers={"Idempotency-Key": f"smoke-{t}"})
    check(f"job submit ({len(tickets)}) 202", s == 202, f"{time.time() - t:.1f}s location={h.get('Location')}")
    if s != 202:
        sys.exit(1)

    lat, stop = [], threading.Event()

    def probe():
        while not stop.is_set():
            t1 = time.time()
            call("GET", "/health", headers={"X-API-Key": ""})
            lat.append(time.time() - t1)
            time.sleep(0.5)

    th = threading.Thread(target=probe, daemon=True)
    th.start()
    while True:
        s, h, st = call("GET", f"/batch/jobs/{job['job_id']}")
        if st["status"] not in ("queued", "running"):
            break
        print(f"   {st['status']} {st['processed']}/{st['total']}", end="\r")
        time.sleep(int(h.get("Retry-After", 2)))
    stop.set()
    th.join()
    dur = time.time() - t
    check("job succeeded", st["status"] == "succeeded", f"in {dur:.0f}s ({len(tickets) / dur:.1f} tickets/s)")
    if lat:
        p95 = sorted(lat)[int(0.95 * (len(lat) - 1))]
        check("/health p95 < 1s during job", p95 < 1.0, f"p95={p95 * 1000:.0f}ms median={statistics.median(lat) * 1000:.0f}ms")

    preds, offset = [], 0
    while offset is not None:
        s, _, page = call("GET", f"/batch/jobs/{job['job_id']}/results?offset={offset}&limit=1000")
        preds += page["predictions"]
        offset = page["next_offset"]
    check("results complete and ordered", [p["ticket_id"] for p in preds] == [x["ticket_id"] for x in tickets])

    if "category" in rows[0]:
        print("   (note: scores on data the deployed model was trained on are not a held-out estimate)")
        cat = sum(p["category"] == truth[p["ticket_id"]]["category"] for p in preds) / len(preds)
        urg_tp = sum(p["is_urgent"] and truth[p["ticket_id"]]["is_urgent"] for p in preds)
        urg_p = sum(p["is_urgent"] for p in preds)
        urg_t = sum(truth[p["ticket_id"]]["is_urgent"] for p in preds)
        f1 = 2 * urg_tp / (urg_p + urg_t) if urg_p + urg_t else 0
        print(f"   category accuracy={cat:.4f}  urgency F1={f1:.4f}")
    s, _, _ = call("DELETE", f"/batch/jobs/{job['job_id']}")
    check("delete 204", s == 204)
    print("FAILED" if check.failed else "ALL CHECKS PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
