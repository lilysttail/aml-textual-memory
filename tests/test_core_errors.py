"""壳层要把核心的异常接住：忙 -> 503，邻居失败 -> 可观察（栖 P1-4）。

app.db.StoreBusy 和 core.db.StoreBusy 是两个不同的类，壳原来只接住了自己
那个：核心忙时 /search 回 500 而不是约定的 503。而邻居补充失败被一句宽泛的
except Exception 吞掉，回 200、上下文悄悄没了——那是在把共享核心的失效
伪装成成功。
"""
import os
import shutil
import tempfile
import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app
from core.db import StoreBusy as CoreStoreBusy

AUTH = {"Authorization": "Bearer test-key"}


def _settings(**ov):
    d = os.path.join(tempfile.gettempdir(), "errors-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    s = dict(db_path=os.path.join(d, "errors.db"),
             api_key="test-key", recall_paths=("keyword",), rerank_enabled=False,
             rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
             embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    s.update(ov)
    return Settings(**s)


def _add(client):
    return client.post("/add", headers=AUTH, json={
        "request_id": "r1", "user_id": "u1", "session_id": "s1",
        "messages": [{"role": "user", "timestamp": 1728000000000,
                      "content": "一句话 alpha"}]})


def test_a_busy_core_answers_503_not_500():
    with TestClient(make_app(_settings())) as client:
        _add(client)
        state = client.app.state

        def busy(*args, **kwargs):
            raise CoreStoreBusy("core is busy")

        state.core.search = busy
        r = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5})
        assert r.status_code == 503, r.status_code


def test_a_failed_core_lookup_is_never_silent():
    """The neighbours come from the core now, so a failure there IS a failure
    of the search: a real error (500, the platform retries), never a 200 that
    quietly lost the context."""
    with TestClient(make_app(_settings()), raise_server_exceptions=False) as client:
        _add(client)
        state = client.app.state

        def broken(*args, **kwargs):
            raise RuntimeError("neighbour lookup blew up")

        state.core.search = broken
        r = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5})
        assert r.status_code == 500, r.status_code
