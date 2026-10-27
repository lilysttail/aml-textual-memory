"""The thing measured, spoken to over real HTTP (step 3): the same service, in this process, on a loopback port.

Ported from an earlier stand-alone measurement script: the service runs INSIDE this
Python process (uvicorn on a loopback port, built by the very same `app.main.make_app`), not in a container. Why: a provider key
then lives in exactly one process environment (the one that starts this run) — never in a container, a command line
or a file — and the code under test is the code the container runs. It is what the gate check (many people asking at the
same time) needs, because the in-process test client does not exercise the real server and its threads.

`search_status` / `add_status` turn any transport failure into a status string ("exc:<Type>") so a load run counts failures
instead of dying; `busy_reasons` counts why the core said "busy" (a hash prefix or a lock name, never corpus text).
"""
import socket
import threading
import time
from collections import Counter
from typing import List, Optional, Tuple

from .dense import LOCAL_KEY, DenseUseCounter, service_env
from .target import Hit, render_like_service


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class HttpServiceTarget:
    record_sep = "#"

    def __init__(self, db_path: str, neighbours: bool = False, dense=None, request_timeout: float = 300.0, stem_english: bool = False):
        import httpx
        import uvicorn
        self._httpx = httpx
        self._timeout = request_timeout
        self.stem_english = stem_english
        with service_env(db_path, neighbours, dense, stem_english=stem_english) as (hooks, m):
            self._app = m.make_app(m.load())
            self.port = _free_port()
            self.base = f"http://127.0.0.1:{self.port}"
            self._server = uvicorn.Server(uvicorn.Config(self._app, host="127.0.0.1", port=self.port, log_level="warning"))
            self._thread = threading.Thread(target=self._server.run, daemon=True)
            self._thread.start()
            self.health_status: Optional[int] = None
            for _ in range(300):
                try:
                    # ANY HTTP answer means the process is up (a 500 from /health is a finding of its own, recorded, not hidden)
                    self.health_status = httpx.get(f"{self.base}/health", timeout=5.0, trust_env=False).status_code
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                self._server.should_exit = True
                raise RuntimeError("the in-process service did not come up")
        self.embed_meter = hooks.embed_meter
        self._caching = hooks.caching
        self._cache = hooks.cache                  # the vector file: closed with the target
        self.dense_use = DenseUseCounter().install() if dense is not None else None      # counts the searches that really ran the dense path
        self._auth = {"Authorization": f"Bearer {LOCAL_KEY}"}
        self.busy_reasons: Counter = Counter()
        self._busy_lock = threading.Lock()
        self._count_busy()

    def _count_busy(self) -> None:
        from core.db import StoreBusy
        core = self._app.state.core
        for name in ("search", "write"):
            orig = getattr(core, name)

            def wrap(*a, __orig=orig, __name=name, **kw):
                try:
                    return __orig(*a, **kw)
                except StoreBusy as e:
                    with self._busy_lock:
                        self.busy_reasons[f"{__name}: {str(e)[:70]}"] += 1
                    raise
            setattr(core, name, wrap)

    def record_id(self, request_id: str, i: int) -> str:
        return f"{request_id}#{i}"

    def vector_stats(self) -> dict:
        return self._caching.stats() if self._caching is not None else {}

    def config_readback(self) -> dict:
        """What the RUNNING service says it is set to (read from its core, not copied from the environment this process set): which paths
        are built and enabled, whether reranking is on, and what the embedder is (provider, model, and the chain of wrappers around it)."""
        core = self._app.state.core
        emb = core.config.embedder
        chain, e = [], emb
        while e is not None and len(chain) < 8:
            chain.append(type(e).__name__)
            e = e.__dict__.get("_inner")
        return {"paths": list(core.config.paths), "recalls": {rc.name: bool(rc.enabled) for rc in core.recalls},
                "rerank_enabled": bool(core.config.rerank_enabled), "embedder_chain": chain,
                "embed_provider": getattr(emb, "provider", None), "embed_model": getattr(emb, "model_name", None)}

    def add_status(self, user_id: str, session_id: str, request_id: str, messages: List[dict]) -> Tuple[str, float]:
        body = {"request_id": request_id, "user_id": user_id, "session_id": session_id, "messages": messages}
        t0 = time.perf_counter()
        try:
            code = str(self._httpx.post(f"{self.base}/add", json=body, headers=self._auth, timeout=self._timeout, trust_env=False).status_code)
        except Exception as e:
            code = f"exc:{type(e).__name__}"
        return code, time.perf_counter() - t0

    def add(self, user_id: str, session_id: str, request_id: str, messages: List[dict]) -> int:
        code, _dt = self.add_status(user_id, session_id, request_id, messages)
        if not code.isdigit():
            raise RuntimeError(f"add failed ({code})")
        return int(code)

    def search_status(self, user_id: str, query: str, top_k: int = 100, timeout: Optional[float] = None) -> Tuple[str, List[Hit], float]:
        t0 = time.perf_counter()
        try:
            r = self._httpx.post(f"{self.base}/search", json={"query": query, "user_id": user_id, "top_k": top_k}, headers=self._auth,
                                 timeout=timeout or self._timeout, trust_env=False)
            hits = [Hit(id=h["id"], content=h["content"]) for h in r.json().get("data", [])] if r.status_code == 200 else []
            return str(r.status_code), hits, time.perf_counter() - t0
        except Exception as e:
            return f"exc:{type(e).__name__}", [], time.perf_counter() - t0

    def search(self, user_id: str, query: str, top_k: int = 100) -> List[Hit]:
        code, hits, _dt = self.search_status(user_id, query, top_k)
        if code != "200":
            raise RuntimeError(f"search answered {code}")
        return hits

    def stored_texts(self, user_id: str) -> List[str]:
        core = self._app.state.core
        with core.db.connect() as conn:
            rows = conn.execute("SELECT speaker, ts, text FROM core_records WHERE space_id=? ORDER BY session_id, turn_index",
                                (user_id,)).fetchall()
        return [render_like_service(ts, speaker, text) for speaker, ts, text in rows]

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
        if self.dense_use is not None:
            self.dense_use.uninstall()
        if self._cache is not None:
            self._cache.close()
