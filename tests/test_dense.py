"""B — dense recall (written RED first).

Fast tests use a deterministic FAKE embedder (no onnx, no model files) so they
run anywhere. Exactly one integration test uses the real local bge-small and
SKIPS (loudly) if the dev deps / model files are missing — a skip is not a pass.

Covers:
  - KeywordRecall / DenseRecall implement the Recall shape: candidates(...) ->
    [(id, score)], and `name` / `enabled`
  - vector cache: the same content is embedded only once (cache hit second time)
  - DenseRecall full vs rerank modes
  - retrieval.rank() orders a single path's results and cuts to top_k
  - the service rebuilds the {id, content, score, created_at} contract
"""
import hashlib
import math
import os
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)
os.environ["API_KEY"] = "test-key"
os.environ.setdefault("DB_PATH", os.path.join(_TMPROOT, f"dense_boot_{uuid.uuid4().hex}.db"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from core.db import Database  # noqa: E402
from core.models import Record  # noqa: E402
from core.recall import KeywordRecall, DenseRecall  # noqa: E402
from core.store import Store  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402
from core.vector_cache import VectorCache  # noqa: E402
from core import rerank as retrieval  # noqa: E402
from app import main as app_main  # noqa: E402


class FakeEmbedder:
    """Deterministic 8-dim embedder; counts how many texts it actually embeds."""
    model_name = "fake-v1"
    dim = 8

    def __init__(self):
        self.calls = 0

    def embed(self, texts, timeout_s=None):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(t):
        h = hashlib.sha256(t.encode("utf-8")).digest()[:8]
        v = [b / 255.0 for b in h]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def _fresh_store():
    """(db, store, vec) — the core's own assembly."""
    db = Database(os.path.join(_TMPROOT, f"dense_{uuid.uuid4().hex}.db"))
    db.create_schema()
    store = Store(db)
    vec = VectorStore(db)
    return db, store, vec


def _seed(store, docs):
    for rid, text in docs.items():
        store.write([Record(record_id=f"{rid}#0", session_id="s", text=text)],
                    space_id="u", batch_id=rid)


def test_keyword_recall_shape():
    _db, store, _vec = _fresh_store()
    _seed(store, {"a": "database connection pool", "b": "cat on the mat"})
    kw = KeywordRecall(store)
    assert kw.name == "keyword" and kw.enabled is True
    out = kw.candidates("u", "database pool", None, 10)
    assert isinstance(out, list) and out
    assert all(isinstance(i, str) and isinstance(s, float) for i, s in out)


def test_rerank_reorders_pool_and_caches():
    _db, store, vec = _fresh_store()
    _seed(store, {"a": "database connection pool exhausted",
                  "b": "the pool of db connections ran out",
                  "c": "a cat sleeping on a windowsill"})
    emb = FakeEmbedder()
    kw = KeywordRecall(store)
    pool = kw.candidates("u", "db connection pool", None, 50)
    vc = VectorCache(vec, emb)
    out = retrieval.rerank(store, "u", "db connection pool", pool, lambda: vc,
                           top_n=50)
    assert out and all(isinstance(i, str) and isinstance(s, float) for i, s in out)
    calls_after_first = emb.calls
    assert calls_after_first > 0
    # second identical rerank: content vectors are cached -> no re-embed
    retrieval.rerank(store, "u", "db connection pool", pool, lambda: vc, top_n=50)
    # only the query itself might be re-embedded (<= 1), never the contents
    assert emb.calls - calls_after_first <= 1, "cached content should not be re-embedded"


def _make_dense(store, vec, emb, **kw):
    """Shared DenseRecall construction for ALL dense tests (fake and real
    embedder alike): a signature change here fails the fake tests that always
    run, instead of hiding behind the real-model skip."""
    return DenseRecall(store, VectorCache(vec, emb), **kw)


def test_dense_recall_full_mode():
    _db, store, vec = _fresh_store()
    _seed(store, {"a": "alpha beta", "b": "gamma delta", "c": "epsilon zeta"})
    emb = FakeEmbedder()
    dense = _make_dense(store, vec, emb)
    out = dense.candidates("u", "alpha beta", None, 10)
    ids = [i for i, _ in out]
    assert set(ids) == {"a#0", "b#0", "c#0"}, "full mode ranks the whole pool"


def test_disabled_flag():
    _db, store, vec = _fresh_store()
    dense = _make_dense(store, vec, FakeEmbedder(), enabled=False)
    assert dense.enabled is False


def test_retrieval_rank_orders_and_cuts():
    pool = [("x", 0.1), ("y", 0.9), ("z", 0.5)]
    ranked = retrieval.rank(pool, top_k=2)
    assert ranked == [("y", 0.9), ("z", 0.5)]


def test_assemble_hits_rebuilds_contract():
    from app.config import Settings
    from app.main import make_app
    settings = Settings(
        db_path=os.path.join(_TMPROOT, f"dense_{uuid.uuid4().hex}.db"),
        api_key="test-key", recall_paths=("keyword",), rerank_enabled=False,
        rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
        embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
        embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    with TestClient(make_app(settings)) as client:
        auth = {"Authorization": "Bearer test-key"}
        r = client.post("/add", headers=auth, json={
            "request_id": "a", "user_id": "u", "session_id": "s1",
            "messages": [{"role": "user", "timestamp": 1728000000000,
                           "content": "hello world"}]})
        assert r.status_code == 200
        s = client.post("/search", headers=auth, json={
            "query": "hello world", "user_id": "u", "top_k": 5})
        hits = s.json()["data"]
        assert len(hits) == 1
        h = hits[0]
        assert set(h.keys()) == {"id", "content", "score", "created_at"}
        assert h["id"] == "a#0" and isinstance(h["score"], float)
        assert "hello world" in h["content"]


@pytest.mark.skipif(
    True if os.environ.get("SKIP_REAL_EMBEDDER") else False,
    reason="set SKIP_REAL_EMBEDDER to force-skip")
def test_dense_beats_keyword_on_paraphrase_real_model():
    try:
        from app.embedder import LocalEmbedder
        emb = LocalEmbedder()
    except Exception as e:  # missing dev deps or model files
        pytest.skip(f"real local embedder unavailable: {e}")
    _db, store, vec = _fresh_store()
    # paraphrase: query shares meaning but few words with the right doc
    _seed(store, {
        "right": "the worker process leaks memory after repeated reconnect attempts",
        "distractor1": "database connection pool exhausted under heavy write load",
        "distractor2": "a cat was sleeping on the warm windowsill in the sun",
    })
    query = "RAM usage keeps growing when the service reconnects over and over"
    dense = _make_dense(store, vec, emb)
    dense_top = dense.candidates("u", query, None, 3)[0][0]
    kw = KeywordRecall(store)
    kw_out = kw.candidates("u", query, None, 3)
    kw_top = kw_out[0][0] if kw_out else None
    assert dense_top == "right#0", f"dense should surface the paraphrase, got {dense_top}"
    assert kw_top != "right#0", "keyword should miss the paraphrase (few shared words)"
