"""缓存同一段文本只许一个线程去请求、封顶要硬（发请求前预留）、题单校验要全、
向量缓存要关得掉、词干开没开要写进每次运行。假接口、假钥匙，不联网。"""
import json
import os
import threading
import time
from collections import Counter

import pytest

from core.embedder import EmbedUnavailable
from eval.ruler import corpus, cost, dense, embed_meter, qa_main, sampling, target, vector_cache
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


# ============ 1. 同一段文本只算一次（并发）============

class BlockingInner:
    provider, model_name, dim = "dashscope", "text-embedding-v4:8", 8

    def __init__(self, fail_first=False):
        self.calls = []
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_first = fail_first

    def embed(self, texts, timeout_s=None):
        with self.lock:
            self.calls.append(list(texts))
            n = len(self.calls)
        self.entered.set()
        self.release.wait(10)
        if self.fail_first and n == 1:
            raise EmbedUnavailable("first request failed")
        return [[float(len(t))] + [0.0] * 7 for t in texts]


def _run(fn, out, key):
    def go():
        try:
            out[key] = fn()
        except Exception as e:                                  # noqa: BLE001
            out[key] = e
    t = threading.Thread(target=go)
    t.start()
    return t


def test_two_threads_asking_for_the_same_new_text_ask_the_provider_only_once(tmp_path):
    inner = BlockingInner()
    e = vector_cache.CachingEmbedder(inner, vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite")))
    out = {}
    t1 = _run(lambda: e.embed(["same text"]), out, "a")
    assert inner.entered.wait(5)
    t2 = _run(lambda: e.embed(["same text"]), out, "b")
    time.sleep(0.4)                                             # 第二个线程此刻应当在等第一个，不是自己去请求
    assert len(inner.calls) == 1
    inner.release.set()
    t1.join(10)
    t2.join(10)
    assert len(inner.calls) == 1 and out["a"] == out["b"] and not isinstance(out["a"], Exception)
    assert e.stats()["inner_calls"] == 1


def test_overlapping_batches_ask_for_each_text_once(tmp_path):
    inner = BlockingInner()
    e = vector_cache.CachingEmbedder(inner, vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite")))
    out = {}
    t1 = _run(lambda: e.embed(["a", "b"]), out, "x")
    assert inner.entered.wait(5)
    t2 = _run(lambda: e.embed(["b", "c"]), out, "y")
    time.sleep(0.4)
    inner.release.set()
    t1.join(10)
    t2.join(10)
    asked = Counter(t for call in inner.calls for t in call)
    assert asked == Counter({"a": 1, "b": 1, "c": 1}), asked
    assert len(out["x"]) == 2 and len(out["y"]) == 2 and out["x"][1] == out["y"][0]


def test_when_the_thread_that_was_asking_fails_the_waiting_one_asks_for_itself_and_nothing_hangs(tmp_path):
    inner = BlockingInner(fail_first=True)
    e = vector_cache.CachingEmbedder(inner, vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite")))
    out = {}
    t1 = _run(lambda: e.embed(["z"]), out, "a")
    assert inner.entered.wait(5)
    t2 = _run(lambda: e.embed(["z"]), out, "b")
    time.sleep(0.4)
    inner.release.set()
    t1.join(10)
    t2.join(10)
    assert isinstance(out["a"], EmbedUnavailable)               # 第一个的失败照样抛给第一个
    assert out["b"] == [[1.0] + [0.0] * 7]                       # 等着的那个自己去问了一次，拿到了
    assert len(inner.calls) == 2
    assert e.embed(["z"]) == out["b"] and len(inner.calls) == 2   # 占位已清掉、结果已落库


# ============ 2. 封顶要硬 ============

def body(*texts):
    return json.dumps({"model": "text-embedding-v4", "input": list(texts)}).encode("utf-8")


def reply(tokens, n=1):
    return 200, json.dumps({"usage": {"total_tokens": tokens}, "data": [{"index": i, "embedding": [1.0]} for i in range(n)]}).encode("utf-8")


class Held:
    """真传输的替身：第一个进来的请求卡住，直到放行；记下每一次真的发出去的请求。"""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, url, headers, body_, timeout):
        with self.lock:
            self.calls += 1
            r = self.replies.pop(0)
        self.started.set()
        self.release.wait(10)
        return r


def test_the_estimate_is_an_upper_bound_of_what_the_provider_can_charge():
    # 一个 token 至少盖住一个字节；每段文本另有头尾两个特殊记号
    assert embed_meter.estimate_request_tokens(body("abcdefgh")) == 8 + 2
    assert embed_meter.estimate_request_tokens(body("你好")) == 6 + 2
    assert embed_meter.estimate_request_tokens(body("a", "bb", "ccc")) == 6 + 6
    assert embed_meter.estimate_request_tokens(b"") is None and embed_meter.estimate_request_tokens(b"not json") is None


def test_a_request_that_could_never_fit_even_alone_is_refused_and_closes_the_door():
    real = Held(reply(1))
    m = embed_meter.EmbedMeter(real, cap_tokens=5)
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, body("abcdefgh"), 5)               # 上界 10 > 5
    assert real.calls == 0 and m.cap_reached.is_set()


def test_the_room_left_is_shared_out_by_upper_bounds_and_trued_up_by_the_real_usage():
    real = Held(reply(4), reply(10), reply(10))
    real.release.set()
    m = embed_meter.EmbedMeter(real, cap_tokens=22)
    m.transport("u", {}, body("abcdefgh"), 5)                   # 上界 10，实际 4：多退 6
    assert m.snapshot()["tokens_by_replies"] == 4
    m.transport("u", {}, body("abcdefgh"), 5)                   # 4 + 10 <= 22，实际 10 → 14
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, body("abcdefgh"), 5)               # 14 + 10 = 24 > 22：发出去之前就拒了
    assert real.calls == 2 and m.snapshot()["tokens_by_replies"] == 14


def test_a_reply_that_charges_more_than_the_bound_is_flagged_and_stops_everything_after_it():
    real = Held(reply(500))
    real.release.set()
    m = embed_meter.EmbedMeter(real, cap_tokens=10 ** 6)
    m.transport("u", {}, body("abcdefgh"), 5)                   # 上界 10，回来说 500：上界不成立
    s = m.snapshot()
    assert s["replies_above_bound"] == 1 and s["tokens_by_replies"] == 500
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, body("a"), 5)
    assert m.cap_reached.is_set()


def test_a_reply_without_usage_is_charged_the_reserved_amount_not_zero():
    real = Held((200, b'{"data": []}'))
    real.release.set()
    m = embed_meter.EmbedMeter(real, cap_tokens=100)
    m.transport("u", {}, body("abcdefgh"), 5)
    s = m.snapshot()
    assert s["tokens_by_replies"] == 10 and s["replies_without_usage"] == 1


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_only_a_2xx_reply_counts_its_usage_an_error_reply_is_never_charged_even_when_it_carries_a_usage_block(status):
    """栖第二轮 P2：带 usage 的 4xx / 5xx 也不计费（没处理的请求）；也不能因为它报的数比预留大就关门。"""
    cm = cost.CostMeter({"text-embedding-v4": (1.0, 0.0)}, limit_usd=1.0)
    real = Held((status, json.dumps({"usage": {"total_tokens": 999}}).encode("utf-8")))
    real.release.set()
    m = embed_meter.EmbedMeter(real, cap_tokens=10_000, cost_meter=cm, price_key="text-embedding-v4")  # gate:allow-secret
    m.transport("u", {}, body("abcdefgh"), 5)
    s = m.snapshot()
    assert s["tokens_by_replies"] == 0 and s["replies_above_bound"] == 0 and s["http_status"] == {str(status): 1}
    assert cm.spent == 0 and not cm.gate.closed
    m.transport  # 之后的请求照常
    assert m.cap_reached.is_set() is False


def test_a_request_that_raised_may_still_have_been_charged_so_the_reserved_amount_stays_counted():
    class Boom:
        def __call__(self, *a):
            raise TimeoutError()
    m = embed_meter.EmbedMeter(Boom(), cap_tokens=25)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            m.transport("u", {}, body("abcdefgh"), 5)           # 每次按 10 记（不知道服务端有没有算）
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, body("abcdefgh"), 5)               # 20 + 10 > 25
    s = m.snapshot()
    assert s["transport_exceptions"] == {"TimeoutError": 2} and s["tokens_charged_without_reply"] == 20


def test_a_body_that_cannot_be_read_gets_the_whole_room_so_only_one_such_request_is_in_the_air():
    real = Held(reply(5), reply(5))
    m = embed_meter.EmbedMeter(real, cap_tokens=10)
    got = {}
    t = _run(lambda: m.transport("u", {}, b"", 5), got, "a")
    assert real.started.wait(5)
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, b"", 5)
    real.release.set()
    t.join(10)
    assert real.calls == 1 and m.snapshot()["tokens_by_replies"] == 5


# ============ 3. 题单校验要全 ============

GOOD_MAIN = ["q3", "q2", "q5", "q1"]                              # 类别 1 / 2 / 3 / 4 各一题


def _check(mini, main, cross, per=1, cross_per=1):
    sampling.check_picks(mini, {"main": main, "cross": cross}, ["mini-1", "mini-2"], per_category=per, cross_per_category=cross_per)


def test_a_clean_picks_set_passes_with_the_generation_parameters(mini):
    _check(mini, GOOD_MAIN, GOOD_MAIN)
    made = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    sampling.check_picks(mini, made, ["mini-1", "mini-2"], per_category=1, cross_per_category=1)


@pytest.mark.parametrize("main,cross,why", [
    (GOOD_MAIN[:3], GOOD_MAIN[:3], "少一题"),
    (GOOD_MAIN + ["q4"], GOOD_MAIN, "多一题"),
    (["q3", "q2", "q1", "q4"], ["q3", "q2", "q1", "q4"], "总数对，但类别 3 换成了第二题类别 4"),
    (["q1", "q1", "q2", "q3"], ["q1"], "同一题两次"),
    (GOOD_MAIN, GOOD_MAIN + ["q3"], "cross 里同一题两次"),
    (GOOD_MAIN, GOOD_MAIN[:3], "cross 少一题"),
    (GOOD_MAIN, [], "cross 空了"),
])
def test_a_hand_edited_picks_set_is_refused(mini, main, cross, why):
    with pytest.raises(ValueError):
        _check(mini, main, cross)


def test_a_question_named_twice_in_the_cross_set_is_refused_even_without_the_generation_parameters(mini):
    """没有传生成参数的老用法也不放过：cross 里同一题两次（只是 main 的子集不够，集合比较会把重复吞掉）"""
    with pytest.raises(ValueError):
        sampling.check_picks(mini, {"main": ["q1", "q2"], "cross": ["q1", "q1"]}, ["mini-1", "mini-2"])


def test_every_category_must_have_exactly_the_asked_number(mini):
    """总数 4 对，但两题在类别 4、一题在类别 3、一题在类别 2、类别 1 没有"""
    with pytest.raises(ValueError):
        _check(mini, ["q1", "q4", "q5", "q2"], ["q1", "q4", "q5", "q2"])
    with pytest.raises(ValueError):                              # cross 每类数量也要对：这里 cross 在类别 4 里放了两题
        _check(mini, GOOD_MAIN, ["q3", "q2", "q5", "q1"][:2] + ["q1", "q4"])


def test_the_run_passes_the_generation_parameters_to_both_checks_and_a_changed_request_is_refused(tmp_path, mini):
    import inspect
    src = inspect.getsource(qa_main.execute_run)
    assert src.count("check_picks(corpus, picks, conv_ids, per_category=per_category, cross_per_category=cross_per_category)") == 2
    with pytest.raises(ValueError):                              # 落盘的是每类 1 题；这次要每类 3 题 = 不是同一份题单
        out = tmp_path / "out"
        out.mkdir()
        (out / "picks.json").write_text(json.dumps({"main": GOOD_MAIN, "cross": GOOD_MAIN}), encoding="utf-8")
        qa_main.execute_run(mini, str(out), "main", ["G2"], "main", 1.0, ask=lambda q, m: None, price_key="x", judge_client=None,
                            conv_ids=["mini-1", "mini-2"], per_category=3, cross_per_category=1)


# ============ 4. 向量缓存要关得掉 ============

def _fill(t, mini):
    from eval.ruler import shapes
    for conv in mini.conversations:
        if conv.sample_id != "mini-1":
            continue
        for b in corpus.chunk_conversation(conv):
            t.add(b.user_id, b.session_id, b.request_id, [shapes.render("c", m, s) for m, s in b.pairs()])


def test_closing_a_service_target_closes_the_vector_file_so_it_can_be_deleted(tmp_path, mini):
    path = tmp_path / "vectors.sqlite"
    cfg = dense.DenseConfig(transport=FakeEmbedApi(), cache_path=str(path))
    t = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=cfg)
    _fill(t, mini)
    t.close()
    os.remove(path)                                              # Windows：连接没关，这里是 PermissionError
    assert not path.exists()
    t.close()                                                    # 关两次不出事


def test_closing_an_http_target_closes_the_vector_file_too(tmp_path, mini):
    from eval.ruler import http_target
    path = tmp_path / "vectors_http.sqlite"
    cfg = dense.DenseConfig(transport=FakeEmbedApi(), cache_path=str(path))
    t = http_target.HttpServiceTarget(str(tmp_path / "h.sqlite"), neighbours=False, dense=cfg)
    _fill(t, mini)
    t.close()
    os.remove(path)
    assert not path.exists()


def test_a_closed_cache_refuses_further_use_loudly_not_silently(tmp_path):
    c = vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite"))
    c.close()
    c.close()                                                    # 幂等
    with pytest.raises(Exception):
        c.get_many("m", ["x"])


# ============ 5. 词干开没开要写进每次运行 ============

def test_the_stemming_switch_is_always_set_explicitly_whatever_the_outer_environment_says(tmp_path, monkeypatch):
    monkeypatch.setenv("STEM_ENGLISH", "1")
    with dense.service_env(str(tmp_path / "a.sqlite"), False, None, stem_english=False) as (hooks, m):
        assert os.environ["STEM_ENGLISH"] == "0" and m.load().stem_english is False
    assert os.environ["STEM_ENGLISH"] == "1"                     # 退出时原样放回
    monkeypatch.delenv("STEM_ENGLISH")
    with dense.service_env(str(tmp_path / "b.sqlite"), False, None, stem_english=True) as (hooks, m):
        assert os.environ["STEM_ENGLISH"] == "1" and m.load().stem_english is True
    assert "STEM_ENGLISH" not in os.environ


def test_the_default_is_off_and_it_is_off_even_when_the_outer_environment_has_it_on(tmp_path, monkeypatch):
    monkeypatch.setenv("STEM_ENGLISH", "1")
    with dense.service_env(str(tmp_path / "c.sqlite"), False, None) as (hooks, m):
        assert m.load().stem_english is False


def test_a_run_writes_the_stemming_setting_into_its_conditions_the_rows_and_the_report(tmp_path, mini, monkeypatch):
    from eval.ruler import chat
    monkeypatch.setenv("STEM_ENGLISH", "1")                      # 外面开着也不算数：以每次运行明说的为准

    class J:
        def complete(self, prompt):
            return chat.ChatResult(text='{"label": "CORRECT"}', prompt_tokens=10, completion_tokens=2, model="j")

    def ask(q, memories):
        return chat.ChatResult(text="a puppy", prompt_tokens=10, completion_tokens=2, model="a")

    pr = {"a": (1.0, 2.0), "j": (1.0, 2.0)}
    out = tmp_path / "out"
    rows = qa_main.execute_run(mini, str(out), "main", ["G2"], "main", 1.0, ask=ask, price_key="a", judge_client=J(), judge_price_key="j",
                               prices=pr, conv_ids=["mini-1", "mini-2"], per_category=1, cross_per_category=1, stem_english=False)
    cond = json.loads((out / "conditions_keyword.json").read_text(encoding="utf-8"))
    assert cond["stem_english"] is False and cond["variant"] == "keyword"
    assert all(r["stem_english"] is False for r in rows)
    rep = qa_main.write_report(str(out))
    assert rep["conditions"]["stem_english"] is False


def test_the_command_line_has_the_switch_and_it_defaults_to_off():
    p = qa_main.build_parser()
    base = ["--out", "x", "--data", "y"]
    assert p.parse_args(["run", *base, "--who", "main", "--groups", "G2"]).stem_english == "off"
    assert p.parse_args(["run", *base, "--who", "main", "--groups", "G2", "--stem-english", "on"]).stem_english == "on"
    assert p.parse_args(["load", *base, "--search", "keyword", "--shape", "c", "--workers", "1", "--probe-n", "0"]).stem_english == "off"
    assert p.parse_args(["load", *base, "--stem-english", "on"]).stem_english == "on"
    with pytest.raises(SystemExit):
        p.parse_args(["run", *base, "--who", "main", "--groups", "G2", "--stem-english", "maybe"])
