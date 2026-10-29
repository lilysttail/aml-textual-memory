"""/health with the dense path on.

This shape was never tested before: the health endpoint used `core.db` but
Core stores `self._db`, so with RECALL_PATHS=keyword,dense (the deploy
configuration) GET /health raised AttributeError and returned 500. The old
`except Exception: pass` in the health body used to swallow it; removing the
swallow (narrow exception scope) exposed it.

Tests here pin:
  - /health returns 2xx while dense is on, for both EMBED_PROVIDER values
    (dashscope / local); it no longer carries pending_vectors (a full read of
    every space) — scripts/count_missing_vectors.py gives that number;
  - a real database failure makes /health return 500, never "0 gaps".
"""
import logging
import os
import sqlite3
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)
os.environ["API_KEY"] = "test-key"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embedder import EmbedUnavailable  # noqa: E402

AUTH = {"Authorization": "Bearer test-key"}


class DownEmbedder:
    """An embedder that always raises — /health must not depend on it."""
    model_name, dim = "down-v1", 8

    def embed(self, texts, timeout_s=None):
        raise EmbedUnavailable("provider down (test)")


def _settings(**ov):
    d = dict(db_path=os.path.join(_TMPROOT, f"hd_{uuid.uuid4().hex}.db"),
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
    assert r.status_code == 200, f"seed failed: {r.status_code}"


@pytest.mark.parametrize("embed_provider", ["dashscope", "local"])
def test_health_ok_while_dense_on_both_providers(monkeypatch, embed_provider):
    """开着 dense 打 /health：两种 EMBED_PROVIDER 都是 2xx、回 ok，不再带 pending_vectors。"""
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: DownEmbedder())
    ov = {"embed_provider": embed_provider}
    if embed_provider == "dashscope":
        ov["dashscope_api_key"] = "test-key-NOT-A-REAL-KEY-123"
    with TestClient(app_main.make_app(_settings(**ov))) as client:
        _seed(client)
        r = client.get("/health")
        assert r.status_code == 200, f"/health must be 2xx while dense is on, got {r.status_code}"
        body = r.json()
        assert body.get("ok") is True and "pending_vectors" not in body, f"/health body: {body!r}"


def test_health_db_failure_returns_500_not_zero_gaps(monkeypatch):
    """数据库真出错：/health 返回 500，不许伪装成「缺 0 条」。"""
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: DownEmbedder())
    with TestClient(app_main.make_app(_settings()), raise_server_exceptions=False) as client:
        _seed(client)
        # Drop the vector table: the table probe fails.
        db_path = client.app.state.core.db.path
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE core_vectors")
        conn.commit()
        conn.close()
        r = client.get("/health")
        assert r.status_code == 500, \
            f"database failure must surface as 500, got {r.status_code}"
