"""被测找法走 HTTP 的那一种：同一个服务在本进程的回环端口上起来，用真 HTTP 讲话（从 measure_dense 并进来）。向量接口是假的。"""
import socket
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from eval.ruler import corpus, dense, http_target, shapes, target
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def fill(t, mini, shape="c"):
    for conv in mini.conversations:
        for b in corpus.chunk_conversation(conv):
            assert t.add(b.user_id, b.session_id, b.request_id, [shapes.render(shape, m, s) for m, s in b.pairs()]) == 200


def test_it_speaks_the_real_contract_over_a_loopback_port_and_answers_like_the_in_process_one(tmp_path, mini):
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"), neighbours=True)
    t = target.ServiceTarget(str(tmp_path / "t.sqlite"), neighbours=True)
    try:
        fill(h, mini)
        fill(t, mini)
        assert h.health_status == 200 and h.base.startswith("http://127.0.0.1:")
        for user, q in (("mini-1", "puppy"), ("mini-1", "pottery class"), ("mini-2", "lighthouse")):
            a = [(x.id, x.content) for x in h.search(corpus.user_id_of(user), q, 20)]
            b = [(x.id, x.content) for x in t.search(corpus.user_id_of(user), q, 20)]
            assert a == b and a
    finally:
        h.close()
        t.close()


def test_a_wrong_key_is_refused_by_the_service_itself(tmp_path):
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"))
    try:
        r = httpx.post(h.base + "/search", json={"query": "x", "user_id": "u", "top_k": 5}, headers={"Authorization": "Bearer nope"}, trust_env=False)
        assert r.status_code == 401
    finally:
        h.close()


def test_many_searches_at_once_all_come_back_ok(tmp_path, mini):
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"), neighbours=False)
    try:
        fill(h, mini)
        with ThreadPoolExecutor(max_workers=8) as pool:
            res = list(pool.map(lambda i: h.search_status(corpus.user_id_of("mini-1"), "puppy lighthouse", 10), range(40)))
        assert {code for code, _hits, _dt in res} == {"200"} and all(dt >= 0 for _c, _h, dt in res)
    finally:
        h.close()


def test_a_transport_failure_is_a_status_string_not_an_exception(tmp_path):
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"))
    h.close()
    code, hits, dt = h.search_status("u", "x", 5)                  # 服务已经关了：连接失败
    assert code.startswith("exc:") and hits == []


def test_the_dense_path_works_over_http_and_is_counted(tmp_path, mini):
    api = FakeEmbedApi()
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=api))
    try:
        fill(h, mini)
        assert h.search(corpus.user_id_of("mini-1"), "puppy", 5)
        snap = h.embed_meter.snapshot()
        assert snap["calls"] == api.calls >= 1 and snap["embed_invocations"] >= 1
        assert h.busy_reasons == {}
    finally:
        h.close()


def test_close_frees_the_port_and_leaves_the_environment_alone(tmp_path):
    import os
    before = {k: os.environ.get(k) for k in ("RECALL_PATHS", "EMBED_PROVIDER", "DB_PATH")}
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"))
    port = h.port
    h.close()
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))                                # 能再绑上 = 端口放掉了
    finally:
        s.close()
    assert {k: os.environ.get(k) for k in before} == before


def test_the_record_id_spelling_is_the_services(tmp_path):
    h = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"))
    try:
        assert h.record_id("req", 3) == "req#3" and h.record_sep == "#"
    finally:
        h.close()
