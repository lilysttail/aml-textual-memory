#!/usr/bin/env python3
"""Local smoke test against a running server (default http://localhost:8000).

Usage:
    API_KEY=dev-key-change-me python scripts/smoke_local.py [--base http://host:8000]
"""
import argparse
import sys
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://localhost:8000")
args = p.parse_args()

import os
KEY = os.environ.get("API_KEY", "dev-key-change-me")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
base = args.base.rstrip("/")
fails = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)

r = requests.get(f"{base}/health", timeout=5)
check("health public", r.status_code == 200, r.text[:120])

r = requests.post(f"{base}/add", json={})  # no auth
check("add rejects without auth", r.status_code == 401, str(r.status_code))

ts = int(time.time() * 1000)
add_body = {
    "request_id": "smoke-1", "user_id": "smoke-u", "session_id": "smoke-s",
    "messages": [
        {"role": "user", "timestamp": ts,
         "content": "LESSON | trigger: flaky integration test | fix: retry with exponential backoff"},
        {"role": "assistant", "timestamp": ts,
         "content": "acknowledged, will retry flaky tests with backoff"},
    ],
}
r = requests.post(f"{base}/add", headers=H, json=add_body, timeout=10)
check("add 200 + success", r.status_code == 200 and r.json().get("success"), r.text[:200])

r = requests.post(f"{base}/add", headers=H, json=add_body, timeout=10)
check("add idempotent replay", r.status_code == 200 and r.json().get("success"), r.text[:200])

r = requests.post(f"{base}/search", headers=H, timeout=10, json={
    "query": "flaky integration test retry backoff",
    "user_id": "smoke-u", "top_k": 5,
})
data = r.json().get("data", []) if r.status_code == 200 else []
check("search returns hits", r.status_code == 200 and len(data) >= 1, r.text[:200])
if data:
    check("content embeds timestamp+role",
          data[0]["content"].startswith("[") and "user:" in data[0]["content"] or "assistant:" in data[0]["content"],
          data[0]["content"][:120])
    scores = [h["score"] for h in data]
    check("search sorted desc", scores == sorted(scores, reverse=True), str(scores))

r = requests.post(f"{base}/search", headers=H, timeout=10, json={
    "query": "flaky integration test", "user_id": "other-user", "top_k": 5})
check("user isolation", r.status_code == 200 and r.json().get("data") == [], r.text[:120])

print(f"\n{len(fails)} failures" if fails else "\nALL SMOKE TESTS PASSED")
sys.exit(1 if fails else 0)
