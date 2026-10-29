"""文本赛道接口的契约（官网 API Guide 逐条）。

- timestamp 可选（官方："timestamp is optional Unix milliseconds"）：不带时间的 Add 照收；事件时间未知就是未知，
  **不拿入库时间冒充，也不在 content 里渲染出一个编出来的日期**。
- 输入检查不比官方合约严（2026-10-29 起）：合约必填的字段缺了才 422；别的照收，没文字的消息跳过这一条
  （细节见 tests/test_add_tolerant.py）。
- 比赛的 content 只放原候选；邻居作为单独候选的开关默认开。本文件里测「只有原候选」的用例显式传 neighbour_candidates=False。
"""
import os
import re
import shutil
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app

AUTH = {"Authorization": "Bearer test-key"}


def _settings(**ov):
    d = os.path.join(tempfile.gettempdir(), "textual-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    s = dict(db_path=os.path.join(d, "textual.db"),
             api_key="test-key", recall_paths=("keyword",), rerank_enabled=False,
             rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
             embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    s.update(ov)
    return Settings(**s)


def _add(client, rid, uid, session, messages):
    return client.post("/add", headers=AUTH, json={
        "request_id": rid, "user_id": uid, "session_id": session,
        "messages": messages})


def _search(client, query, uid, top_k=10, **extra):
    body = {"query": query, "user_id": uid, "top_k": top_k}
    body.update(extra)
    return client.post("/search", headers=AUTH, json=body)


def _msgs():
    return [
        {"role": "user", "timestamp": 1728000000000, "content": "第一句 alpha"},
        {"role": "assistant", "timestamp": 1728000060000, "content": "第二句 alpha"},
        {"role": "user", "timestamp": 1728000120000, "content": "第三句 alpha"},
    ]


# ── 没有时间戳的 Add（栖第 1 条）────────────────────────────────────────────

def test_T_N1_an_add_without_timestamps_is_accepted_and_searchable():
    with TestClient(make_app(_settings())) as client:
        r = _add(client, "r1", "u1", "s1", [
            {"role": "user", "content": "第一条没有时间的原话 alpha"},
            {"role": "assistant", "content": "第二条没有时间的原话 alpha"},
        ])
        assert r.status_code == 200, r.text
        data = _search(client, "alpha", "u1", top_k=5).json()["data"]
        assert len(data) == 2, data
        assert "第一条没有时间的原话" in data[0]["content"]
        assert "第二条没有时间的原话" in data[1]["content"]
        # the replay is still idempotent
        again = _add(client, "r1", "u1", "s1", [
            {"role": "user", "content": "第一条没有时间的原话 alpha"},
            {"role": "assistant", "content": "第二条没有时间的原话 alpha"},
        ])
        assert again.status_code == 200
        assert len(_search(client, "alpha", "u1", top_k=5).json()["data"]) == 2


def test_T_N2_turn_indexes_without_timestamps_follow_the_arrival_order():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", [
            {"role": "user", "content": "甲"},
            {"role": "assistant", "content": "乙"},
            {"role": "user", "content": "丙"},
        ])
        rows = client.app.state.core.store.get_rows("u1", ["r1#0", "r1#1", "r1#2"])
        assert rows["r1#0"]["turn_index"] == 0
        assert rows["r1#1"]["turn_index"] == 1
        assert rows["r1#2"]["turn_index"] == 2


def test_T_N3_a_record_without_a_timestamp_shows_no_date():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", [
            {"role": "user", "content": "没有时间的一句 alpha"}])
        data = _search(client, "alpha", "u1", top_k=5).json()["data"]
        content = data[0]["content"]
        assert "没有时间的一句" in content
        assert not re.search(r"\d{4}-\d{2}-\d{2}", content), content


# ── content 只放原候选；邻居另作独立候选（设计时已定）──────────────────────

def test_T_C8_by_default_the_content_is_the_candidate_only():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        data = _search(client, "alpha", "u1", top_k=5).json()["data"]
        assert data
        for h in data:
            assert len(h["content"].splitlines()) == 1, h["content"]


def test_T_C8b_the_neighbour_switch_makes_neighbours_separate_candidates():
    with TestClient(make_app(_settings(neighbour_candidates=True))) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        data = _search(client, "alpha", "u1", top_k=5).json()["data"]
        ids = [h["id"] for h in data]
        assert len(ids) == len(set(ids)), ids          # no duplicates
        assert len(data) <= 5, data                  # never more than top_k
        assert any("第一句" in h["content"] for h in data), data


def test_T_C2_the_response_never_exceeds_top_k():
    with TestClient(make_app(_settings(neighbour_candidates=True))) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        for top_k in (1, 2, 5, 100):
            data = _search(client, "alpha", "u1", top_k=top_k).json()["data"]
            assert len(data) <= top_k, (top_k, data)


def test_T_C1_every_item_carries_id_and_content():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        data = _search(client, "alpha", "u1", top_k=5).json()["data"]
        assert data
        for h in data:
            assert isinstance(h["id"], str) and h["id"]
            assert isinstance(h["content"], str) and h["content"]
            assert isinstance(h["score"], (int, float))
            assert isinstance(h["created_at"], str) and h["created_at"]


def test_T_B5_a_very_long_content_is_taken_and_stays_searchable():
    """The official contract sets no length limit, so a long message is legal
    input: it must be stored and findable, not refused (柯 10-21 判：保持)."""
    long_text = "alpha " + ("x" * (1024 * 1024))
    with TestClient(make_app(_settings())) as client:
        r = _add(client, "r1", "u1", "s1", [
            {"role": "user", "timestamp": 1728000000000, "content": long_text}])
        assert r.status_code == 200, r.text
        data = _search(client, "alpha", "u1", top_k=5).json()["data"]
        assert data, "the long message must be findable"
        assert long_text in data[0]["content"]


def test_T_C4_the_options_do_not_change_the_retrieval():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        plain = _search(client, "alpha", "u1", top_k=5).json()["data"]
        with_opts = _search(client, "alpha", "u1", top_k=5,
                            options=["A. 第一句", "B. 第二句"]).json()["data"]
        assert [h["id"] for h in plain] == [h["id"] for h in with_opts]


# ── 输入边界：不比官方合约严（2026-10-29 改）──────────────────────
# 原来这里是「fail closed（422），不静默接受」（栖 P2）。正式评测被我们自己的 422 停下（平台对 422 不重试）之后改成：
# 合约允许的一律收下；一条消息没有文字就跳过这一条、不拒整批；时间读不成日期就当没有时间。缺合约必填字段仍是 422。
# 细节和其余情况见 tests/test_add_tolerant.py。

@pytest.mark.parametrize("messages", [
    [],
    [{"role": "user", "content": ""}],
    [{"role": "user", "content": "   "}],
])
def test_T_B1_messages_without_text_are_taken_and_nothing_is_written(messages):
    with TestClient(make_app(_settings())) as client:
        r = client.post("/add", headers=AUTH, json={
            "request_id": "r1", "user_id": "u1", "session_id": "s1",
            "messages": messages})
        assert r.status_code == 200, r.status_code
        assert r.json() == {"success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"}
        with client.app.state.core.db.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 0


def test_T_B2_a_top_k_below_one_answers_with_no_results():
    with TestClient(make_app(_settings())) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        for k in (0, -1):
            r = client.post("/search", headers=AUTH, json={
                "query": "alpha", "user_id": "u1", "top_k": k})
            assert r.status_code == 200, r.status_code
            assert r.json() == {"data": []}


def test_T_B3_an_out_of_range_timestamp_is_taken_as_no_time():
    with TestClient(make_app(_settings())) as client:
        r = client.post("/add", headers=AUTH, json={
            "request_id": "r1", "user_id": "u1", "session_id": "s1",
            "messages": [{"role": "user", "timestamp": 10 ** 100,
                          "content": "x alpha"}]})
        assert r.status_code == 200, r.status_code
        hits = _search(client, "alpha", "u1").json()["data"]
        assert [h["content"] for h in hits] == ["[user] x alpha"], "读不成日期：当没有时间，不编日期"


def test_T_B4_a_neighbour_that_is_already_a_hit_is_not_handed_over_twice():
    """The de-dup is on the UNDERLYING record, not on the generated id."""
    with TestClient(make_app(_settings(neighbour_candidates=True))) as client:
        _add(client, "r1", "u1", "s1", _msgs())
        data = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 10}).json()["data"]
        contents = [h["content"] for h in data]
        assert len(contents) == len(set(contents)), contents
        assert len(data) <= 3, data


# ── 默认就是开（设计时定的）──────────────────────────────────────────

def _default_settings():
    """The settings as the repository ships them: the neighbour switch is
    NOT set, so the default decides."""
    from app.config import Settings
    d = os.path.join(tempfile.gettempdir(), "textual-default-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return Settings(db_path=os.path.join(d, "textual.db"),
                    api_key="test-key", recall_paths=("keyword",),
                    rerank_enabled=False, rerank_n=200,
                    store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
                    embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
                    embed_acquire_s=30.0, bge_small_dir=None)


def test_T_C9_the_neighbour_switch_is_on_by_default():
    """The version we submit is the version we ship: the switch is on unless
    the environment turns it off (NEIGHBOUR_CANDIDATES=0)."""
    assert _default_settings().neighbour_candidates is True


def test_T_C9b_with_the_default_the_neighbours_come_as_candidates_of_their_own():
    with TestClient(make_app(_default_settings())) as client:
        # only the first and the third turn match; the second one is their
        # neighbour, so it can only reach the platform as a candidate of its own
        _add(client, "r1", "u1", "s1", [
            {"role": "user", "timestamp": 1728000000000, "content": "第一句 alpha"},
            {"role": "assistant", "timestamp": 1728000060000, "content": "第二句 beta"},
            {"role": "user", "timestamp": 1728000120000, "content": "第三句 alpha"},
        ])
        data = client.post("/search", headers=AUTH, json={
            "query": "alpha", "user_id": "u1", "top_k": 10}).json()["data"]
        ids = [h["id"] for h in data]
        contents = [h["content"] for h in data]
        assert len(ids) == len(set(ids)), ids
        assert len(contents) == len(set(contents)), contents
        assert len(data) <= 10, data
        assert any("第二句 beta" in c for c in contents), contents
