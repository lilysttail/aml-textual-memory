"""2026-10-29 起：/add、/search 的输入检查不比官方合约严。

正式评测停在 ADD_API_CONTRACT_MISMATCH · HTTP 422：我们的检查比合约严，有几批评测数据被我们拒了，平台对 422 不重试。
改成：合约要求的字段必须在（缺了仍 422，日志只记字段名和毛病种类，不记请求内容）；合约允许的一律收下——
一条消息没有文字就跳过这一条、不拒整批；role 缺记 user；时间读不成日期当没有时间；session_id 空就以这一批的 request_id 当会话；
/search 的空 query、top_k < 1 回 200 空结果。正常请求写进去的和搜出来的跟交上去那一版（63a2da9）逐字一样（金样）。
"""
import json
import logging
import os
import shutil
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app, normalize_message

AUTH = {"Authorization": "Bearer test-key"}
GOLDEN = os.path.join(os.path.dirname(__file__), "golden_add_search_63a2da9.json")


def _settings(**ov):
    d = os.path.join(tempfile.gettempdir(), "tolerant-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    s = dict(db_path=os.path.join(d, "t.db"), api_key="test-key", recall_paths=("keyword",), rerank_enabled=False,
             rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000, embed_max_concurrency=2,
             embed_batch=32, embed_backfill_wait_s=60.0, embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    s.update(ov)
    return Settings(**s)


@pytest.fixture()
def client():
    s = _settings()
    with TestClient(make_app(s)) as c:
        yield c
    shutil.rmtree(os.path.dirname(s.db_path), ignore_errors=True)


def add(c, messages, request_id="r1", user_id="u1", session_id="s1"):
    return c.post("/add", headers=AUTH, json={"request_id": request_id, "user_id": user_id, "session_id": session_id, "messages": messages})


def rows(c, user_id="u1"):
    with c.app.state.core.db.connect() as conn:
        return [dict(zip(("record_id", "session_id", "speaker", "ts", "text"), r)) for r in conn.execute(
            "SELECT record_id, session_id, speaker, ts, text FROM core_records WHERE space_id=? ORDER BY record_id", (user_id,)).fetchall()]


# ── 以前 422、现在收下：每一种核写进去的是什么、跳过的是哪条 ─────────────────────────

def test_an_empty_message_list_is_taken_and_writes_nothing(client):
    r = add(client, [])
    assert r.status_code == 200 and r.json() == {"success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"}
    assert rows(client) == []


def test_messages_without_text_are_skipped_one_by_one_and_the_rest_is_stored_with_its_original_position(client):
    r = add(client, [
        {"role": "user", "content": "first garden"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "   \n\t"},
        {"role": "user", "content": None},
        {"role": "user"},
        {"role": "user", "content": True},
        "not an object",
        None,
        {"role": "assistant", "content": "last garden"},
    ])
    assert r.status_code == 200
    got = rows(client)
    assert [x["record_id"] for x in got] == ["r1#0", "r1#8"], "跳过的不重新编号：保留原始下标"
    assert [x["text"] for x in got] == ["first garden", "last garden"]


def test_a_batch_where_every_message_is_skipped_answers_success_and_writes_nothing(client):
    r = add(client, [{"role": "user", "content": ""}, {"content": None}, 5])
    assert r.status_code == 200 and r.json()["success"] is True
    assert rows(client) == []


def test_content_that_is_not_a_string_keeps_its_text_parts_joined_with_spaces(client):
    r = add(client, [
        {"role": "user", "content": ["part one", {"type": "text", "text": "part two"}, {"type": "image", "url": "x"}, 7, ""]},
        {"role": "user", "content": {"type": "text", "text": "single part"}},
        {"role": "user", "content": [{"type": "image"}, "  "]},
        {"role": "user", "content": 42},
    ])
    assert r.status_code == 200
    got = {x["record_id"]: x["text"] for x in rows(client)}
    assert got == {"r1#0": "part one part two", "r1#1": "single part", "r1#3": "42"}, "拼出来是空的那条（#2）跳过"


def test_a_missing_or_odd_role_is_taken(client):
    r = add(client, [{"content": "no role"}, {"role": None, "content": "null role"}, {"role": 3, "content": "number role"},
                     {"role": "tool", "content": "tool role"}])
    assert r.status_code == 200
    assert [x["speaker"] for x in rows(client)] == ["user", "user", "user", "tool"], "缺 / 不是字符串记 user；不认识的字符串原样存"


@pytest.mark.parametrize("value,stored", [
    (-86400000, "1969-12-31T00:00:00+00:00"),                     # 1970 年以前：照收
    (1728000000123.9, "2024-10-04T00:00:00.123000+00:00"),         # 带小数：取整
    ("1728000000123", "2024-10-04T00:00:00.123000+00:00"),         # 数字字符串
    (" 1728000000123.7 ", "2024-10-04T00:00:00.123000+00:00"),    # 带小数的数字字符串
    ("1:56 pm on 8 May, 2023", None),                              # 读不成：当没有时间
    (10 ** 100, None),                                              # 超大：当没有时间，不 500
    (4102444800001, None),                                          # 2100 年以后：当没有时间
    (-10 ** 15, None),                                              # 公元 1 年以前：表示不了，当没有时间，不 500
    (True, None),
    ([1, 2], None),
])
def test_a_timestamp_the_old_check_refused_is_taken(client, value, stored):
    r = add(client, [{"role": "user", "timestamp": value, "content": "a dated garden note"}])
    assert r.status_code == 200, r.text
    assert rows(client)[0]["ts"] == stored


def test_a_time_far_out_of_range_never_becomes_a_500(client):
    for v in (-62135596800001, 253402300800000, 10 ** 30, -10 ** 30, "1e400", "nan", "inf"):
        r = add(client, [{"role": "user", "timestamp": v, "content": "x"}], request_id=f"r-{v}")
        assert r.status_code == 200, (v, r.status_code)


def test_empty_ids_are_taken_and_given_back_as_sent(client):
    r = add(client, [{"role": "user", "content": "garden"}], request_id="", user_id="", session_id="")
    assert r.status_code == 200 and r.json() == {"success": True, "request_id": "", "user_id": "", "session_id": ""}


def test_an_empty_request_id_keys_the_batch_by_its_content_so_different_batches_are_all_kept_and_a_resend_is_kept_once(client):
    one = [{"role": "user", "timestamp": 1728000000000, "content": "first empty-id garden"}]
    two = [{"role": "user", "timestamp": 1728000060000, "content": "second empty-id garden"}]
    for batch in (one, two, one, two, one):
        r = add(client, batch, request_id="", session_id="s1")
        assert r.status_code == 200 and r.json()["request_id"] == "", "回复照样带回空的 request_id"
    got = rows(client)
    assert sorted(x["text"] for x in got) == ["first empty-id garden", "second empty-id garden"], "两批不同内容都写进去，重发不重复"
    keys = {x["record_id"].split("#")[0] for x in got}
    assert len(keys) == 2 and all(k.startswith("anon-") and len(k) == 5 + 24 for k in keys)
    # the same content under another session id or another user is another batch
    add(client, one, request_id="", session_id="s2")
    add(client, one, request_id="", session_id="s1", user_id="u2")
    assert len(rows(client)) == 3 and len(rows(client, "u2")) == 1
    # a non-empty request id is still the key, exactly as before
    add(client, one, request_id="rq")
    assert [x["record_id"] for x in rows(client) if x["record_id"].startswith("rq")] == ["rq#0"]


def test_an_empty_request_id_and_an_empty_session_id_still_keep_batches_apart(client):
    add(client, [{"role": "user", "content": "alpha one"}], request_id="", session_id="")
    add(client, [{"role": "user", "content": "beta two"}], request_id="", session_id="")
    got = rows(client)
    assert len(got) == 2 and len({x["session_id"] for x in got}) == 2, "每批自成一个会话"


def test_an_empty_session_id_makes_each_batch_its_own_session_so_batches_never_become_neighbours():
    s = _settings(neighbour_candidates=True)
    with TestClient(make_app(s)) as c:
        add(c, [{"role": "user", "content": "alpha garden"}, {"role": "user", "content": "beta"}], request_id="ra", session_id="")
        add(c, [{"role": "user", "content": "gamma"}, {"role": "user", "content": "delta"}], request_id="rb", session_id="")
        got = rows(c)
        assert {x["record_id"]: x["session_id"] for x in got} == {"ra#0": "ra", "ra#1": "ra", "rb#0": "rb", "rb#1": "rb"}
        ids = [h["id"] for h in c.post("/search", headers=AUTH, json={"query": "alpha", "user_id": "u1", "top_k": 10}).json()["data"]]
        assert ids == ["ra#0", "ra#0#nbra#1"], "邻居只在同一批里找，不跨到另一批"
    shutil.rmtree(os.path.dirname(s.db_path), ignore_errors=True)


def test_unknown_fields_are_ignored(client):
    r = client.post("/add", headers=AUTH, json={"request_id": "r1", "user_id": "u1", "session_id": "s1", "extra": {"a": 1},
                                                "messages": [{"role": "user", "content": "garden", "name": "x", "metadata": {}}]})
    assert r.status_code == 200 and len(rows(client)) == 1


# ── 还会 422 的：只记字段名和毛病，不记请求内容 ─────────────────────────────────────────

SECRET_TEXT = "PRIVATE-EVAL-CONTENT-do-not-log-7731"  # gate:allow-secret


@pytest.mark.parametrize("missing", ["request_id", "user_id", "session_id", "messages"])
def test_a_missing_required_field_is_still_a_422_and_the_log_names_the_field_without_the_content(client, caplog, missing):
    body = {"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": [{"role": "user", "content": SECRET_TEXT}]}
    del body[missing]
    with caplog.at_level(logging.WARNING, logger="app.main"):
        r = client.post("/add", headers=AUTH, json=body)
    assert r.status_code == 422
    lines = [rec.getMessage() for rec in caplog.records if rec.name == "app.main" and "refused 422" in rec.getMessage()]
    assert len(lines) == 1, lines
    assert f"loc=body.{missing}" in lines[0] and "type=missing" in lines[0] and "msg=" in lines[0]
    assert SECRET_TEXT not in "\n".join(rec.getMessage() for rec in caplog.records), "请求内容不进日志"
    assert "input" not in lines[0] and "url" not in lines[0]


def test_a_wrong_type_is_logged_by_field_and_kind_only(client, caplog):
    with caplog.at_level(logging.WARNING, logger="app.main"):
        r = client.post("/add", headers=AUTH, json={"request_id": {"x": SECRET_TEXT}, "user_id": "u1", "session_id": "s1", "messages": []})
    assert r.status_code == 422
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "loc=body.request_id" in text and "type=string_type" in text
    assert SECRET_TEXT not in text


def test_skipped_messages_are_logged_as_a_count_only(client, caplog):
    with caplog.at_level(logging.WARNING, logger="app.main"):
        add(client, [{"role": "user", "content": "   "}, {"role": "user", "content": SECRET_TEXT + " kept"}])
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "1 of 2 message(s) skipped" in text and SECRET_TEXT not in text


# ── /search ───────────────────────────────────────────────────────────────────

def test_search_with_a_blank_query_an_empty_user_or_a_top_k_below_one_answers_with_no_results(client):
    add(client, [{"role": "user", "content": "garden alpha"}])
    for body in ({"query": "", "user_id": "u1"}, {"query": "   ", "user_id": "u1"}, {"query": "alpha", "user_id": "u1", "top_k": 0},
                 {"query": "alpha", "user_id": "u1", "top_k": -5}, {"query": "alpha", "user_id": ""}):
        r = client.post("/search", headers=AUTH, json=body)
        assert r.status_code == 200 and r.json() == {"data": []}, body
    assert client.post("/search", headers=AUTH, json={"query": "alpha", "user_id": "u1"}).json()["data"], "正常的照常有结果"


def test_a_blank_query_or_a_top_k_below_one_never_reaches_retrieval(client):
    """The keyword path answers a blank query with nothing anyway, but the deployed meaning-based path would embed it: the empty answer
    is given before any search is run."""
    def boom(*a, **k):
        raise AssertionError("retrieval must not run")
    client.app.state.core.search = boom
    for body in ({"query": "", "user_id": "u1"}, {"query": " \t ", "user_id": "u1"}, {"query": "alpha", "user_id": "u1", "top_k": 0}):
        r = client.post("/search", headers=AUTH, json=body)
        assert r.status_code == 200 and r.json() == {"data": []}, body


def test_search_still_needs_its_required_fields(client):
    assert client.post("/search", headers=AUTH, json={"user_id": "u1"}).status_code == 422
    assert client.post("/search", headers=AUTH, json={"query": "x"}).status_code == 422


def test_the_key_is_still_required(client):
    assert client.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s", "messages": []}).status_code == 401


# ── 重发不重复写（幂等） ───────────────────────────────────────────────────────────

def test_the_same_batch_sent_again_writes_nothing_twice_including_a_batch_with_skipped_messages(client):
    batch = [{"role": "user", "content": "garden one"}, {"role": "user", "content": " "}, {"role": "user", "content": "garden two"}]
    for _ in range(3):
        assert add(client, batch, request_id="rx").status_code == 200
    assert [x["record_id"] for x in rows(client)] == ["rx#0", "rx#2"]


# ── 正常请求逐字不变（对交上去那一版 63a2da9 的金样）──────────────────────────────────────

T0 = 1728000000123
MIN = 60_000
BATCHES = [
    ("r-a", "s1", [{"role": "user", "timestamp": T0, "content": "the garden needs water today"},
                   {"role": "assistant", "timestamp": T0 + MIN, "content": "we planted tomatoes in the garden"},
                   {"role": "user", "timestamp": T0 + 2 * MIN, "content": "garden garden tomatoes harvest soon"}]),
    ("r-b", "s2", [{"role": "user", "content": "morning walk with the dog"},
                   {"role": "assistant", "content": "the dog found a tomato in the garden"}]),
    ("r-c", "s3", [{"role": "user", "timestamp": 0, "content": "epoch zero garden note"},
                   {"role": "system", "timestamp": 4102444800000, "content": "end of century tomatoes"},
                   {"role": "user", "timestamp": T0 + 5 * MIN, "content": "  padded garden text  "}]),
]
QUERIES = [("garden tomatoes", 10), ("dog", 5), ("water", 3)]


def _ordinary(neighbours):
    s = _settings(neighbour_candidates=neighbours)
    out = {"add": [], "search": [], "rows": None}
    with TestClient(make_app(s)) as c:
        for rid, sid, msgs in BATCHES:
            r = add(c, msgs, request_id=rid, session_id=sid)
            out["add"].append([r.status_code, r.json()])
        for q, k in QUERIES:
            r = c.post("/search", headers=AUTH, json={"query": q, "user_id": "u1", "top_k": k})
            body = r.json()
            for h in body.get("data", []):
                if not h["content"].startswith("[1") and not h["content"].startswith("[2"):
                    h["created_at"] = "<ingestion time: a message without a timestamp>"
            out["search"].append([q, k, r.status_code, body])
        with c.app.state.core.db.connect() as conn:
            out["rows"] = [list(x) for x in conn.execute(
                "SELECT record_id, session_id, speaker, ts, text FROM core_records ORDER BY record_id").fetchall()]
    shutil.rmtree(os.path.dirname(s.db_path), ignore_errors=True)
    return json.loads(json.dumps(out))


@pytest.mark.parametrize("neighbours", [True, False])
def test_ordinary_requests_are_stored_and_answered_exactly_as_the_submitted_version_did(neighbours):
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    assert _ordinary(neighbours) == golden["neighbours_on" if neighbours else "neighbours_off"]


def test_normalize_message_on_its_own():
    assert normalize_message({"role": "user", "timestamp": 0, "content": "x"})[0::2] == ("user", "x")
    assert normalize_message({"content": "  "}) is None
    assert normalize_message("x") is None
