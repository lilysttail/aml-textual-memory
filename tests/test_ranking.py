"""Ranking correctness tests (written RED first).

Bug: keyword_search mapped BM25 to score = 1/(1-bm25), so a BETTER match
(more negative bm25) got a SMALLER score, while retrieval sorted by score
descending and took top_k — handing back the least-relevant of the keyword
candidates. Net effect: a memory could not even retrieve itself.

These lock the contract the score must satisfy:
  - querying a stored memory with its own text ranks it #1 (over HTTP)
  - score increases with match quality (more query terms matched -> higher)
"""
import os
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)

os.environ["API_KEY"] = "test-key"
os.environ.setdefault("DB_PATH", os.path.join(_TMPROOT, f"rank_boot_{uuid.uuid4().hex}.db"))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import make_app  # noqa: E402

AUTH = {"Authorization": "Bearer test-key"}


def _settings():
    return Settings(db_path=os.path.join(_TMPROOT, f"rank_{uuid.uuid4().hex}.db"),
                    api_key="test-key", recall_paths=("keyword",), rerank_enabled=False,
                    rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
                    embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
                    embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)


def _add(client, rid, text, ts_ms=1728000000000):
    return client.post("/add", headers=AUTH, json={
        "request_id": rid, "user_id": "u1", "session_id": "s1",
        "messages": [{"role": "user", "timestamp": ts_ms, "content": text}],
    })


def _search(client, query, top_k=10):
    r = client.post("/search", headers=AUTH,
                    json={"query": query, "user_id": "u1", "top_k": top_k})
    assert r.status_code == 200
    return r.json()["data"]


def _strip(mem_id):
    return mem_id.rsplit("#", 1)[0]


def test_self_retrieval_ranks_first():
    with TestClient(make_app(_settings())) as client:
        # Documents deliberately share vocabulary (database / connection / pool /
        # concurrent / write / load) so the query matches several of them. The
        # one whose text IS the query must still rank #1. Under the old reversed
        # score it got buried last among the matches.
        docs = {
            "m1": "database connection pool exhausted under heavy concurrent write load",
            "m2": "database connection refused when the server restarts under load",
            "m3": "connection pool leak causes the worker to exhaust available sockets",
            "m4": "heavy write load triggers a timeout in the database replication lag",
            "m5": "concurrent read load on the database is served from the cache layer",
            "m6": "the write ahead log grows under sustained concurrent write pressure",
            "m7": "a pool of database connections is shared across all worker threads",
            "m8": "the load balancer spreads concurrent requests across database replicas",
        }
        for rid, text in docs.items():
            assert _add(client, rid, text).status_code == 200
        data = _search(client, docs["m1"], top_k=8)
        ids = [_strip(h["id"]) for h in data]
        assert ids[0] == "m1", f"a memory must retrieve itself first; got order {ids}"


def test_score_increases_with_match():
    with TestClient(make_app(_settings())) as client:
        _add(client, "hi", "alpha beta gamma delta epsilon zeta")   # all terms
        _add(client, "lo", "alpha and nothing else relevant here")  # one term
        _add(client, "none", "completely unrelated wording xyzzy")
        data = _search(client, "alpha beta gamma delta epsilon zeta", top_k=10)
        score = {_strip(h["id"]): h["score"] for h in data}
        ids = [_strip(h["id"]) for h in data]
        assert "hi" in score and "lo" in score
        assert score["hi"] > score["lo"], f"better match must score higher: {score}"
        assert ids.index("hi") < ids.index("lo"), f"better match must rank earlier: {ids}"
