"""Contract tests for the AML Add/Search scaffold.

Covers: /health public, auth rejection, Add success + idempotency,
Search ordering + user isolation, timestamp-in-content, concurrent writes.
"""
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["API_KEY"] = "test-key"
_tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(_tmp, "test.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import make_app  # noqa: E402


def _settings(**ov):
    d = dict(db_path=os.environ["DB_PATH"], api_key="test-key",
             recall_paths=("keyword",), rerank_enabled=False, rerank_n=200,
             store_busy_timeout_ms=5000,
             store_write_lock_timeout_ms=10000, embed_max_concurrency=2,
             embed_batch=32, embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
             bge_small_dir=None, neighbour_candidates=False)
    d.update(ov)
    return Settings(**d)


@pytest.fixture()
def client():
    with TestClient(make_app(_settings())) as c:
        yield c


AUTH = {"Authorization": "Bearer test-key"}


def _add(client, rid, uid, text, ts_ms=1728000000000):
    return client.post("/add", headers=AUTH, json={
        "request_id": rid, "user_id": uid, "session_id": "s1",
        "messages": [{"role": "user", "timestamp": ts_ms, "content": text}],
    })


def test_health_public(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_add_requires_auth(client):
    r = client.post("/add", json={})
    assert r.status_code == 401


def test_search_requires_auth(client):
    r = client.post("/search", json={})
    assert r.status_code == 401


def test_add_success_and_retrievable(client):
    r = _add(client, "r1", "u1", "flaky integration test fixed by retrying with backoff")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["request_id"] == "r1"
    s = client.post("/search", headers=AUTH,
                    json={"query": "flaky integration test backoff",
                          "user_id": "u1", "top_k": 5})
    assert s.status_code == 200
    data = s.json()["data"]
    assert len(data) >= 1
    assert "[2024-10-04" in data[0]["content"]  # timestamp embedded in text


def test_add_idempotent(client):
    before = client.app.state.core.store.count("u1")
    _add(client, "r-dup", "u1", "duplicate lesson text here")
    _add(client, "r-dup", "u1", "duplicate lesson text here")
    assert client.app.state.core.store.count("u1") == before + 1  # written once
    s = client.post("/search", headers=AUTH,
                    json={"query": "duplicate lesson text here",
                          "user_id": "u1", "top_k": 100})
    ids = [h["id"] for h in s.json()["data"]]
    assert ids.count("r-dup#0") == 1


def test_search_sorted_desc_and_user_isolated(client):
    _add(client, "ra", "u2", "aaa unique alpha beta gamma")
    _add(client, "rb", "u2", "aaa unique alpha")
    s = client.post("/search", headers=AUTH,
                    json={"query": "aaa unique alpha beta gamma",
                          "user_id": "u2", "top_k": 10})
    scores = [h["score"] for h in s.json()["data"]]
    assert scores == sorted(scores, reverse=True)
    s2 = client.post("/search", headers=AUTH,
                     json={"query": "aaa unique alpha", "user_id": "u-other",
                           "top_k": 10})
    assert s2.json()["data"] == []


def test_concurrent_adds(client):
    def w(i):
        return _add(client, f"rc-{i}", "u3", f"concurrent lesson number {i}").status_code
    with ThreadPoolExecutor(max_workers=16) as ex:
        codes = list(ex.map(w, range(32)))
    assert all(c == 200 for c in codes)
    assert client.app.state.core.store.count("u3") == 32
