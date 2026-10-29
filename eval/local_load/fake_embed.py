"""A local stand-in for the embedding service, for load tests only.

It answers the same request the service's DashScope adapter sends (OpenAI-compatible
`POST <base>/embeddings` with `{"model", "input": [texts], "dimensions", "encoding_format"}`)
with the same reply shape (`{"data": [{"index", "embedding"}], "usage": {...}}`), so a load
test exercises the real adapter, the real gate and the real deadlines — only the far end is
local. Nothing here is used by the service itself.

- The same text always gets the same vector (seeded from the text's sha256), `dimensions` long.
- Every call waits a FIXED delay before answering (`--delay`, default 0.5 s; one number, not a
  range), so the latency is the same in every run.
- `GET /stats` returns how many calls / texts it has answered; `POST /stats/reset` zeroes them.
- Many calls are served at once (one thread per request), like a remote service.

    python fake_embed.py --port 8099 --delay 0.5
"""
import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_LOCK = threading.Lock()
_STATS = {"calls": 0, "texts": 0, "in_flight": 0, "max_in_flight": 0}


def vector_for(text, dim):
    """The fixed vector of one text: dim floats from a generator seeded by the text's sha256."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32).tolist()


def make_handler(delay):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):          # quiet: the load test reads /stats, not this log
            pass

        def _send(self, code, obj):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/stats"):
                with _LOCK:
                    return self._send(200, dict(_STATS, delay_s=delay))
            if self.path.rstrip("/").endswith("/health"):
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            if self.path.rstrip("/").endswith("/stats/reset"):
                with _LOCK:
                    for k in _STATS:
                        _STATS[k] = 0
                return self._send(200, {"ok": True})
            if not self.path.rstrip("/").endswith("/embeddings"):
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(body.decode("utf-8"))
                texts = req["input"]
                if isinstance(texts, str):
                    texts = [texts]
                dim = int(req.get("dimensions") or 1024)
            except Exception:
                return self._send(400, {"error": {"code": "InvalidParameter", "message": "bad request"}})
            with _LOCK:
                _STATS["calls"] += 1
                _STATS["texts"] += len(texts)
                _STATS["in_flight"] += 1
                _STATS["max_in_flight"] = max(_STATS["max_in_flight"], _STATS["in_flight"])
            try:
                time.sleep(delay)
                data = [{"object": "embedding", "index": i, "embedding": vector_for(t, dim)} for i, t in enumerate(texts)]
                tokens = sum(max(1, len(t) // 4) for t in texts)
                self._send(200, {"object": "list", "data": data, "model": req.get("model"),
                                 "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})
            finally:
                with _LOCK:
                    _STATS["in_flight"] -= 1
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds every call waits before answering (one fixed value)")
    args = ap.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.delay))
    server.daemon_threads = True
    print(f"fake embeddings on {args.host}:{args.port}, delay {args.delay}s", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
