"""search-path failure policy with write-time embedding.

New shape: vectors are computed at write time, not at search time. If the
provider fails during /add, the write still succeeds (fail-open) and the
missing vectors are tracked in core_pending_vectors. At search time, dense
recall uses cached vectors + limited backfill; if backfill fails, dense is
skipped and keyword still answers. /search never 503s because of embedding.
"""
import logging
import os
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)
os.environ["API_KEY"] = "test-key"
os.environ.setdefault("DB_PATH", os.path.join(_TMPROOT, f"ds_boot_{uuid.uuid4().hex}.db"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embedder import DashScopeEmbedder, EmbedUnavailable  # noqa: E402

AUTH = {"Authorization": "Bearer test-key"}


class DownEmbedder:
    """An embedder that always raises EmbedUnavailable — simulates a provider outage."""
    model_name, dim = "down-v1", 8

    def embed(self, texts, timeout_s=None):
        raise EmbedUnavailable("DashScope HTTP 503 (Throttling)")


class BuggyEmbedder:
    """An embedder that raises a real bug (not EmbedUnavailable) — should propagate."""
    model_name, dim = "buggy-v1", 8

    def embed(self, texts, timeout_s=None):
        raise RuntimeError("a real bug, not an outage")


def _settings(**ov):
    d = dict(db_path=os.path.join(_TMPROOT, f"ds_{uuid.uuid4().hex}.db"),
             api_key="test-key", recall_paths=("keyword", "dense"),
             rerank_enabled=False, rerank_n=200, store_busy_timeout_ms=5000,
             store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32,
             embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
             bge_small_dir=None, neighbour_candidates=False,
             dense_backfill_m=8, dense_backfill_t=2.0)
    d.update(ov)
    return Settings(**d)


def _seed(client, content="connection pool exhausted in the database client"):
    r = client.post("/add", headers=AUTH, json={
        "request_id": "r1", "user_id": "u1", "session_id": "s1",
        "messages": [{"role": "user", "timestamp": 1728000000000,
                      "content": content}]})
    assert r.status_code == 200
    return r


def _search(client, query="database connection pool"):
    return client.post("/search", headers=AUTH, json={"query": query, "user_id": "u1", "top_k": 10})


def test_write_time_fail_open_add_still_200(monkeypatch):
    """Write with a down embedder: /add still returns 200, keyword search works."""
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: DownEmbedder())
    with TestClient(app_main.make_app(_settings())) as client:
        r = _seed(client)
        assert r.status_code == 200, "write must succeed even if embedder is down"
        # Keyword search should still find the record.
        sr = _search(client)
        assert sr.status_code == 200
        assert sr.json()["data"], "keyword route must still return the seeded memory"


def test_search_with_down_embedder_dense_skipped_keyword_answers(monkeypatch, caplog):
    """Search with a down embedder: dense is skipped, keyword still answers, no 503."""
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: DownEmbedder())
    with TestClient(app_main.make_app(_settings())) as client:
        _seed(client)
        with caplog.at_level(logging.INFO):
            r = _search(client)
        assert r.status_code == 200, "search must not 503 when embedder is down"
        assert r.json()["data"], "keyword route must still answer"
        # Dense route should be skipped (logged).
        skips = [rec for rec in caplog.records if "skip" in rec.getMessage().lower()
                 or "skipped" in rec.getMessage().lower()]
        assert any("dense" in s.getMessage().lower() for s in skips), \
            "dense route should be logged as skipped"


def test_only_dense_route_and_it_is_down_returns_empty_not_an_error(monkeypatch):
    """Only dense route, and it's down: search returns empty data, not an error."""
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: DownEmbedder())
    with TestClient(app_main.make_app(_settings(recall_paths=("dense",)))) as client:
        _seed(client)
        r = _search(client)
        assert r.status_code == 200, "search must not 503"
        assert r.json()["data"] == [], "dense down with no cached vectors returns empty"


def test_other_exceptions_still_propagate(monkeypatch):
    """Buggy embedder (not EmbedUnavailable): exception propagates, /add returns 500."""
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: BuggyEmbedder())
    with TestClient(app_main.make_app(_settings()), raise_server_exceptions=False) as client:
        # The bug is in the write-time embed call; it should propagate as 500.
        r = client.post("/add", headers=AUTH, json={
            "request_id": "r1", "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "timestamp": 1728000000000,
                          "content": "test"}]})
        assert r.status_code == 500, "real bugs must not be swallowed"


def test_garbage_vector_reply_skips_route_not_500(monkeypatch, caplog):
    """A bad vector reply (garbage data) causes EmbedUnavailable → route skipped, not 500."""
    import json

    def _bad_reply_embedder(component):
        def transport(url, headers, body, timeout):
            n = len(json.loads(body)["input"])
            data = [{"object": "embedding", "index": i, "embedding": [1.0] * 64}
                    for i in range(n)]
            data[0]["embedding"][3] = component
            return 200, json.dumps({"data": data, "usage": {"total_tokens": n}}).encode("utf-8")
        return DashScopeEmbedder(api_key="test-key-NOT-A-REAL-KEY-123", dim=64,
                                 transport=transport)

    for component in [None, "x", float("nan")]:
        monkeypatch.setattr(app_main, "_build_embedder", lambda s, c=component: _bad_reply_embedder(c))
        with TestClient(app_main.make_app(_settings()), raise_server_exceptions=False) as client:
            _seed(client)
            with caplog.at_level(logging.INFO):
                r = _search(client)
            assert r.status_code == 200, "garbage vector reply must not 500"
            assert r.json()["data"], "keyword route still answers"


def test_provider_switch_builds_dashscope_embedder():
    """Provider switch in app.main builds a DashScopeEmbedder for dashscope provider."""
    s = _settings(embed_provider="dashscope", dashscope_api_key="test-key-NOT-A-REAL-KEY-123")
    emb = app_main._build_embedder(s)
    assert isinstance(emb, DashScopeEmbedder)
    assert emb.model_name == "text-embedding-v4:1024" and emb.dim == 1024
    assert "NOT-A-REAL-KEY" not in repr(emb)