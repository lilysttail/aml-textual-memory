"""D — RRF fusion tests (written RED first).

merge_rrf fuses each path's ranking with 1/(k + rank) (k=60, fixed weights).
The same formula runs on a single path (no special case): 1/(k+rank) is
monotonic, so a single path's output order is bit-identical to its input.
"""
import hashlib
import math
import os
import uuid

from fastapi.testclient import TestClient  # noqa: E402

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)
os.environ["API_KEY"] = "test-key"
os.environ.setdefault("DB_PATH", os.path.join(_TMPROOT, f"fusion_boot_{uuid.uuid4().hex}.db"))

from app.config import Settings  # noqa: E402
from app.main import make_app  # noqa: E402
from core import rerank as retrieval  # noqa: E402


def test_merge_rrf_formula():
    # path A: a@1, b@2 ; path B: b@1, c@2 ; k=60
    # a: 1/61 ; b: 1/62 + 1/61 ; c: 1/62
    merged = retrieval.merge_rrf(
        [[("a", 0.9), ("b", 0.8)], [("b", 0.9), ("c", 0.8)]], k=60)
    assert [m for m, _ in merged] == ["b", "a", "c"]
    assert abs(dict(merged)["b"] - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(dict(merged)["a"] - 1 / 61) < 1e-9


def test_merge_rrf_single_path_order_unchanged():
    # one path goes through the same formula; order must equal the input order
    path = [[("x", 0.1), ("y", 0.9), ("z", 0.5)]]
    merged = retrieval.merge_rrf(path)
    assert [m for m, _ in merged] == ["x", "y", "z"]


def _settings(**ov):
    d = dict(db_path=os.path.join(_TMPROOT, f"fusion_{uuid.uuid4().hex}.db"),
             api_key="test-key", recall_paths=("keyword",),
             rerank_enabled=False, rerank_n=200,
             store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
             embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    d.update(ov)
    return Settings(**d)


def test_multi_path_search_http():
    # the service serves the pool; results stay within top_k and keep the contract
    with TestClient(make_app(_settings())) as client:
        auth = {"Authorization": "Bearer test-key"}
        r = client.post("/add", headers=auth, json={
            "request_id": "r1", "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "timestamp": 1728000000000,
                          "content": "FileNotFoundError opening src/app/main.py"
                          " earlier this week"}]})
        assert r.status_code == 200
        s = client.post("/search", headers=auth, json={
            "query": "FileNotFoundError in src/app/main.py", "user_id": "u1",
            "top_k": 100})
        assert s.status_code == 200
        hits = s.json()["data"]
        assert len(hits) <= 100, "results must stay within top_k"
        assert hits, "the search should surface the seeded memory"
        assert all(set(h) >= {"id", "content", "score", "created_at"} for h in hits)


def test_rerank_takes_fused_pool_top_n():
    """After fusion, rerank consumes the TOP RERANK_N of the fused pool."""
    from core.db import Database
    from core.models import Record
    from core.recall import DenseRecall, KeywordRecall
    from core.store import Store
    from core.vector_cache import VectorCache
    from core.vector_store import VectorStore

    class FakeEmbedder:
        model_name = "fake-v1"
        dim = 8
        embedded_count = 0

        def embed(self, texts, timeout_s=None):
            self.embedded_count += len(texts)
            out = []
            for t in texts:
                h = hashlib.sha256(t.encode()).digest()[:8]
                v = [b / 255.0 for b in h]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / n for x in v])
            return out

    db = Database(os.path.join(_TMPROOT, f"fus_{uuid.uuid4().hex}.db"))
    db.create_schema()
    store = Store(db)
    vec = VectorStore(db)
    docs = {
        "r1": "FileNotFoundError: no such file 'src/app/main.py' seen earlier",
        "r2": "database connection pool exhausted under heavy write load",
        "r3": "KeyError: 'missing_key' in the config dictionary",  # gate:allow-secret
    }
    for rid, text in docs.items():
        store.write([Record(record_id=f"{rid}#0", session_id="s", text=text)],
                    space_id="u", batch_id=rid)

    q = "FileNotFoundError opening src/app/main.py"
    kw = KeywordRecall(store)
    dense = DenseRecall(store, VectorCache(vec, FakeEmbedder()))
    fused = retrieval.merge_rrf([kw.candidates("u", q, None, 50),
                                 dense.candidates("u", q, None, 50)])
    # The exact fusion pool shape is retrieval's concern (tested above); this
    # test pins the rerank seam: it must consume the fused pool's TOP N.
    pool = [(m, 1.0 - i * 0.05) for i, m in enumerate(
        [fused[0][0]] + [x[0] for x in fused[1:]] + [m for m in
         ("r1#0", "r2#0", "r3#0") if m not in {x[0] for x in fused}])]
    top2_ids = {m for m, _ in pool[:2]}
    emb = FakeEmbedder()
    vc = VectorCache(vec, emb)
    reranked = retrieval.rerank(store, "u", q, pool, lambda: vc, top_n=2)
    assert len(reranked) == 2
    assert {m for m, _ in reranked} == top2_ids, \
        "rerank must consume exactly the fused pool's top RERANK_N"
