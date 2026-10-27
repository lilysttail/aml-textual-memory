"""壳接上核心之后的端到端：/add /search 的契约没变，只是多了邻居。

- /add 写完立刻能搜到；
- 同 request_id 再 add 不重复；
- user 之间硬隔离；
- 邻居并进 content（每行有自己的说话人和日期），id / score / created_at
  仍对准主命中；
- 字符预算只裁邻居，命中那条一字不裁。
"""
import os
import shutil
import tempfile
import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app

AUTH = {"Authorization": "Bearer test-key"}


def _settings(**ov):
    d = os.path.join(tempfile.gettempdir(), "bridge-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    s = dict(db_path=os.path.join(d, "bridge.db"),
             api_key="test-key", recall_paths=("keyword",), rerank_enabled=False,
             rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
             embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    s.update(ov)
    return Settings(**s)


def _add(client, rid, uid, session, messages):
    return client.post("/add", headers=AUTH, json={
        "request_id": rid, "user_id": uid, "session_id": session,
        "messages": [{"role": role, "timestamp": ts, "content": content}
                     for role, ts, content in messages]})


def test_add_then_search_finds_it_and_the_content_is_the_candidate():
    with TestClient(make_app(_settings())) as client:
        assert _add(client, "r1", "u1", "s1", [
            ("user", 1728000000000, "第一句话 alpha"),
            ("assistant", 1728000060000, "第二句话 alpha"),
            ("user", 1728000120000, "第三句话 alpha"),
        ]).status_code == 200
        r = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data, "the seeded line must be findable right after the Add"
        hit = data[0]
        # the contract's fields are unchanged
        assert set(hit) >= {"id", "content", "score", "created_at"}
        # the content is the candidate itself — the neighbours are NOT folded
        # into it (the platform's answer model reads it verbatim)
        assert len(hit["content"].splitlines()) == 1, hit["content"]
        # the id / score / created_at still point at the hit itself
        assert hit["id"].startswith("r1#")


def test_the_neighbours_are_candidates_of_their_own_when_the_switch_is_on():
    with TestClient(make_app(_settings(neighbour_candidates=True))) as client:
        _add(client, "r1", "u1", "s1", [
            ("user", 1728000000000, "第一句话 alpha"),
            ("assistant", 1728000060000, "第二句话 alpha"),
            ("user", 1728000120000, "第三句话 alpha"),
        ])
        data = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5}).json()["data"]
        ids = [h["id"] for h in data]
        assert len(ids) == len(set(ids)), ids
        assert len(data) <= 5, data
        assert any("第一句话" in h["content"] for h in data), data
        assert any("第三句话" in h["content"] for h in data), data


def test_the_same_request_id_twice_writes_once():
    with TestClient(make_app(_settings())) as client:
        for _ in range(2):
            assert _add(client, "r1", "u1", "s1", [
                ("user", 1728000000000, "只写一次 alpha")]).status_code == 200
        r = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 10})
        assert len(r.json()["data"]) == 1


def test_one_user_cannot_see_another():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", [("user", 1728000000000, "甲的 alpha")])
        _add(client, "r2", "u2", "s1", [("user", 1728000000000, "乙的 alpha")])
        r1 = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 10}).json()["data"]
        r2 = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u2", "top_k": 10}).json()["data"]
        assert [h["id"] for h in r1] == ["r1#0"]
        assert [h["id"] for h in r2] == ["r2#0"]


def test_the_budget_cuts_the_neighbours_not_the_hit():
    long_line = "很长的一行 " * 200
    with TestClient(make_app(_settings(content_max_chars=200))) as client:
        _add(client, "r1", "u1", "s1", [
            ("user", 1728000000000, "第一句 " + long_line),
            ("assistant", 1728000060000, "第二句 alpha"),
            ("user", 1728000120000, "第三句 " + long_line),
        ])
        r = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5})
        hit = r.json()["data"][0]
        # the content is the candidate itself — the long neighbours are not
        # folded into it, so no budget can cut them out of it
        assert "第二句 alpha" in hit["content"]
        assert "很长的一行" not in hit["content"]


def test_the_neighbours_follow_the_event_time_even_when_the_add_arrives_out_of_order():
    """The platform does not promise that one request's messages come in time
    order; the neighbours must still be the ones around it in TIME (栖)."""
    with TestClient(make_app(_settings(neighbour_candidates=True))) as client:
        _add(client, "r1", "u1", "s1", [
            ("user", 1728000120000, "第三句 alpha"),
            ("assistant", 1728000060000, "第二句 alpha"),
            ("user", 1728000000000, "第一句 alpha"),
        ])
        data = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5}).json()["data"]
        neighbours = " | ".join(h["content"] for h in data if h["id"] != "r1#1")
        assert "第一句" in neighbours and "第三句" in neighbours, neighbours
        # the earlier neighbour comes before the later one
        assert neighbours.index("第一句") < neighbours.index("第三句"), neighbours


def test_the_hit_keeps_its_own_text_verbatim():
    text = "带  空格\n和换行的一行 alpha"
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", [("user", 1728000000000, text)])
        r = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 5})
        content = r.json()["data"][0]["content"]
        assert text in content
