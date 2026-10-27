"""Switching the service path to "keyword + dense" for the ruler (step 3): the settings, and the one place that builds the app.

`DenseConfig` says how to run the meaning-based path: keyword + dense, the DashScope embedder (the key comes from the process
environment as `DASHSCOPE_API_KEY`, never from a file and never as an argument), reranking off by default. Around the embedder
the ruler puts, from the outside in: the invocation counter, a fixed local vector store (a text embedded once is never embedded
again), and the metering transport (counts calls and tokens, refuses locally at the token cap or the spending limit).

`service_env` is the only place that sets the process environment for a service and puts the embedder hooks in; both the
in-process test client and the loopback HTTP server are built through it, and what it changed is put back when it exits. The
environment and `app.main._build_embedder` must stay in place until the app's start-up has run (the core is built there), so the
caller keeps the `with` block open around start-up.
"""
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

LOCAL_KEY = "ruler-local-key"                 # the in-process service's own throwaway key; it protects nothing; gate:allow-secret
KEYWORD_SWITCHES = {"RECALL_PATHS": "keyword", "RERANK_ENABLED": "0", "EMBED_PROVIDER": "local"}
EMBED_PRICE_KEY = "text-embedding-v4"  # gate:allow-secret


@dataclass
class DenseConfig:
    transport: Optional[Callable] = None       # None = the real HTTP transport of the embedder
    cap_tokens: int = 600_000
    cost_meter: Optional[object] = None
    price_key: str = EMBED_PRICE_KEY
    cache_path: Optional[str] = None
    rerank: bool = False
    # A concurrency measurement only: the embedding requests are not queued behind the paid gate (see embed_meter.EmbedMeter).
    concurrent_embed: bool = False

    def switches(self) -> dict:
        return {"RECALL_PATHS": "keyword,dense", "RERANK_ENABLED": "1" if self.rerank else "0", "EMBED_PROVIDER": "dashscope"}


class DenseUseCounter:
    """Counts that the meaning-based path RAN, not just that it was not skipped (step 4). While installed, every call of
    `core.recall.DenseRecall.candidates` is counted (`calls`), and how many of them returned something (`nonempty`; a call whose query
    vector was skipped, or that found nothing to rank, returns nothing) or raised (`raised`). The result of the call is passed on
    untouched. Several counters can be installed at once (several targets in one process); the class method is wrapped once and is
    put back when the last one leaves. A target installs its counter after start-up and removes it in `close()`."""
    _guard = threading.Lock()
    _active: list = []
    _original = None

    def __init__(self):
        self.calls = 0
        self.nonempty = 0
        self.raised = 0
        self._own = threading.Lock()

    def _record(self, n_returned) -> None:
        """`n_returned` = how many candidates the call returned; None = the call raised."""
        with self._own:
            self.calls += 1
            if n_returned is None:
                self.raised += 1
            elif n_returned > 0:
                self.nonempty += 1

    def marks(self) -> tuple:
        with self._own:
            return (self.calls, self.nonempty)

    def snapshot(self) -> dict:
        with self._own:
            return {"calls": self.calls, "nonempty": self.nonempty, "raised": self.raised}

    @classmethod
    def _notify(cls, n_returned) -> None:
        for c in list(cls._active):
            c._record(n_returned)

    def install(self) -> "DenseUseCounter":
        from core import recall
        with DenseUseCounter._guard:
            if not DenseUseCounter._active:
                original = recall.DenseRecall.candidates

                def counted(this, space_id, query, options, limit):
                    try:
                        out = original(this, space_id, query, options, limit)
                    except BaseException:
                        DenseUseCounter._notify(None)
                        raise
                    DenseUseCounter._notify(len(out))
                    return out
                DenseUseCounter._original = original
                recall.DenseRecall.candidates = counted
            DenseUseCounter._active.append(self)
        return self

    def uninstall(self) -> None:
        from core import recall
        with DenseUseCounter._guard:
            if self not in DenseUseCounter._active:
                return
            DenseUseCounter._active.remove(self)
            if not DenseUseCounter._active:
                recall.DenseRecall.candidates = DenseUseCounter._original
                DenseUseCounter._original = None


class Hooks:
    """What `service_env` built around the embedder, for the target to read afterwards."""

    def __init__(self):
        self.embed_meter = None
        self.caching = None
        self.cache = None


@contextmanager
def service_env(db_path: str, neighbours: bool, dense: Optional[DenseConfig] = None, stem_english: bool = False):
    # STEM_ENGLISH (English word-form matching of the keyword path) is set explicitly for every run: what the outer environment
    # says never counts, and the setting is written into the run's conditions and results.
    env = dict(dense.switches() if dense else KEYWORD_SWITCHES, DB_PATH=db_path, API_KEY=LOCAL_KEY,
               NEIGHBOUR_CANDIDATES="1" if neighbours else "0", STEM_ENGLISH="1" if stem_english else "0")
    if dense is not None and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("the dense path needs DASHSCOPE_API_KEY in the process environment (set it in the environment of the process that starts the run)")
    saved = {k: os.environ.get(k) for k in list(env) + ["DASHSCOPE_API_KEY"]}
    os.environ.update(env)
    if dense is None:
        os.environ.pop("DASHSCOPE_API_KEY", None)                   # the keyword path never sees a key
    hooks = Hooks()
    import app.main as m
    original = m._build_embedder
    if dense is not None:
        from app import embedder as dash
        from . import embed_meter, vector_cache

        def build(settings):
            emb = original(settings)
            hooks.embed_meter = embed_meter.EmbedMeter(dense.transport or dash._urllib_transport, dense.cap_tokens,
                                                      cost_meter=dense.cost_meter, price_key=dense.price_key,
                                                      concurrent=dense.concurrent_embed)
            emb._transport = hooks.embed_meter.transport
            outer = emb
            if dense.cache_path:
                hooks.cache = vector_cache.DiskVectorCache(dense.cache_path)
                outer = hooks.caching = vector_cache.CachingEmbedder(emb, hooks.cache)
            return hooks.embed_meter.count_invocations(outer)
        m._build_embedder = build
    try:
        yield hooks, m
    finally:
        m._build_embedder = original
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
