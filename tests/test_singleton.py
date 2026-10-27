"""Startup builds EVERYTHING once — embedder, VectorCache, recall paths — at
lifespan time, not lazily on the first /search (ke/栖 review: the lazy build
had no lock; even after locking it was still building per-request state).

Test with a counting fake-embedder factory:
  - after app startup, WITHOUT any request: embedder (+ VectorCache) and the
    recall paths are already built, exactly once each;
  - after 32 concurrent /search: build counts are unchanged and the recall
    path object is the same instance.
"""
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient  # noqa: E402

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)
os.environ["API_KEY"] = "test-key"
os.environ.setdefault("DB_PATH", os.path.join(_TMPROOT, f"boot_{uuid.uuid4().hex}.db"))

import app.main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402


def _settings(**ov):
    d = dict(db_path=os.path.join(_TMPROOT, f"singleton_{uuid.uuid4().hex}.db"),
             api_key="test-key", recall_paths=("dense",), rerank_enabled=False,
             rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
             embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    d.update(ov)
    return Settings(**d)


def test_builds_once_at_startup_and_stays_singleton(monkeypatch):
    builds = {"n": 0}
    lock = threading.Lock()

    class CountingEmbedder:
        model_name = "counting-v1"
        dim = 8
        embedded_count = 0

        def __init__(self):
            with lock:
                builds["n"] += 1
            time.sleep(0.02)  # widen any race window

        def embed(self, texts, timeout_s=None):
            return [[0.0] * self.dim for _ in texts]

    monkeypatch.setattr(app_main, "_build_embedder", lambda s: CountingEmbedder())

    with TestClient(app_main.make_app(_settings())) as client:
        state = client.app.state
        # built at startup, BEFORE any request hit the app
        assert builds["n"] == 1, f"embedder must be built once at startup: {builds}"
        assert state.core is not None, "the core is built at startup"
        assert len(state.core.recalls) == 1, "recall paths built at startup"
        recalls_ref = state.core.recalls
        core_ref = state.core

        auth = {"Authorization": "Bearer test-key"}

        def one(i):
            return client.post("/search", headers=auth, json={
                "query": f"cold query number {i}", "user_id": "u1", "top_k": 5
            }).status_code

        with ThreadPoolExecutor(max_workers=32) as ex:
            codes = list(ex.map(one, range(32)))
        assert all(c == 200 for c in codes), f"non-200: {set(codes)}"

    assert builds["n"] == 1, \
        f"embedder rebuilt after requests (built {builds['n']} times)"
    assert state.core.recalls is recalls_ref, "recall paths rebuilt per request"
    assert state.core is core_ref, "the core rebuilt per request"