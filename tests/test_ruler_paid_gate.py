"""所有付费调用（答题、判分、判分一致性、算向量）过同一道门：
门里一把锁，整个调用在锁里做完（先看「已花 + 这次的上界」够不够 → 发 → 记实际 → 实际比上界还多就关门，之后全拒）；
同一时间只有一个付费调用在外面，就没有两个调用抢额度。reserve / settle / release / 在途额度整套删掉。假接口、假钥匙。"""
import json
import os
import threading
import time

import pytest

from eval.ruler import chat, cost, embed_meter, paid_gate

PRICE = {"m": (1.0, 2.0), "free": None, "emb": (1.0, 0.0)}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_CHAT_KEY", "test-key-NOT-A-REAL-KEY")


class Probe:
    """记下任何时刻同时在外面的调用数（最大值）。"""
    def __init__(self):
        self.now = 0
        self.max = 0
        self.total = 0
        self.lock = threading.Lock()

    def enter(self):
        with self.lock:
            self.now += 1
            self.total += 1
            self.max = max(self.max, self.now)

    def leave(self):
        with self.lock:
            self.now -= 1


def ok(prompt_tokens, completion_tokens, text="hi"):
    return 200, json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "model": "m",
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}).encode("utf-8")


def emb_reply(tokens):
    return 200, json.dumps({"usage": {"total_tokens": tokens}, "data": [{"index": 0, "embedding": [1.0]}]}).encode("utf-8")


def emb_body(*texts):
    return json.dumps({"model": "text-embedding-v4", "input": list(texts)}).encode("utf-8")


# ---- 门本身 ----

def test_the_gate_runs_a_call_that_fits_books_what_it_really_cost_and_refuses_one_that_would_pass_the_limit_without_sending_it():
    m = cost.CostMeter(PRICE, limit_usd=0.001)
    sent = []
    out = m.gate.run("m", 100, 100, lambda: (sent.append(1) or "answer", (50, 10)))
    assert out == "answer" and m.tokens["m"] == {"prompt": 50, "completion": 10, "calls": 1}
    assert m.spent == pytest.approx((50 * 1.0 + 10 * 2.0) / 1e6)
    with pytest.raises(cost.BudgetExceeded):                     # 上界 = 10000 * 1 + 10000 * 2 = 0.03 > 上限：不发
        m.gate.run("m", 10_000, 10_000, lambda: (sent.append(1), (1, 1)))
    assert sent == [1] and m.gate.refused == 1


def test_a_call_that_raised_or_whose_usage_is_unknown_is_booked_at_the_bound_and_an_error_reply_is_not_charged():
    m = cost.CostMeter(PRICE, limit_usd=1.0)

    def boom():
        raise TimeoutError()
    with pytest.raises(TimeoutError):
        m.gate.run("m", 40, 10, boom)                            # 服务端可能已经算了：按上界记
    assert m.tokens["m"] == {"prompt": 40, "completion": 10, "calls": 1}
    m.gate.run("m", 40, 10, lambda: ("x", paid_gate.BOUND))      # 读不出用量：按上界记
    assert m.tokens["m"]["calls"] == 2 and m.tokens["m"]["prompt"] == 80
    m.gate.run("m", 40, 10, lambda: ("e", None))                 # 错误回复：不记
    assert m.tokens["m"]["calls"] == 2


def test_a_call_that_costs_more_than_its_bound_closes_the_gate_and_everything_after_it_is_refused():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    m.gate.run("m", 10, 10, lambda: ("x", (10_000, 10)))         # 比上界贵得多，但离上限还远
    assert m.gate.closed is True and m.spent < m.limit_usd / 10
    sent = []
    with pytest.raises(cost.BudgetExceeded):
        m.gate.run("m", 1, 1, lambda: (sent.append(1), (1, 1)))
    assert sent == []


def test_a_model_paid_by_quota_holds_no_room_its_tokens_are_counted_and_it_cannot_close_the_gate():
    m = cost.CostMeter(PRICE, limit_usd=0.0)
    m.gate.run("free", 10 ** 6, 10 ** 6, lambda: ("x", (30, 6)))
    assert m.tokens["free"] == {"prompt": 30, "completion": 6, "calls": 1} and m.spent == 0 and m.gate.closed is False


def test_the_extra_admit_check_runs_inside_the_gate_before_sending():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    sent = []
    with pytest.raises(cost.BudgetExceeded):
        m.gate.run("m", 1, 1, lambda: (sent.append(1), (1, 1)), admit=lambda: False)
    assert sent == []


# ---- 同一时间只有一个付费调用在外面 ----

def test_many_threads_asking_to_send_a_paid_call_at_once_only_ever_have_one_in_the_air():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    probe = Probe()

    def call():
        def fn():
            probe.enter()
            time.sleep(0.02)
            probe.leave()
            return "x", (1, 1)
        m.gate.run("m", 10, 10, fn)
    ts = [threading.Thread(target=call) for _ in range(12)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert probe.total == 12 and probe.max == 1


def test_when_the_limit_only_covers_one_call_exactly_one_is_sent_whoever_asks_first():
    one = (100 * 1.0 + 100 * 2.0) / 1e6
    m = cost.CostMeter(PRICE, limit_usd=one * 1.5)                # 只够一次
    sent, refused = [], []

    def call(i):
        try:
            m.gate.run("m", 100, 100, lambda: (sent.append(i) or time.sleep(0.05), (100, 100)))     # 这次实际花满上界
        except cost.BudgetExceeded:
            refused.append(i)
    ts = [threading.Thread(target=call, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert len(sent) == 1 and len(refused) == 1


def test_the_chat_calls_and_the_vector_calls_of_one_run_share_one_gate_and_never_overlap():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    probe = Probe()

    def chat_transport(url, headers, body, timeout):
        probe.enter()
        time.sleep(0.02)
        probe.leave()
        return ok(20, 5)

    def embed_transport(url, headers, body, timeout):
        probe.enter()
        time.sleep(0.02)
        probe.leave()
        return emb_reply(5)
    c = chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model="m", api_key_env="FAKE_CHAT_KEY", max_tokens=16), transport=chat_transport, sleep=lambda s: None)
    c.attach_meter(m, "m")
    em = embed_meter.EmbedMeter(embed_transport, cap_tokens=10 ** 6, cost_meter=m, price_key="emb")
    work = [lambda: c.complete("hello"), lambda: em.transport("u", {}, emb_body("abcdefgh"), 5)] * 6
    ts = [threading.Thread(target=w) for w in work]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert probe.total == 12 and probe.max == 1
    assert m.tokens["m"]["calls"] == 6 and m.tokens["emb"]["calls"] == 6


def test_after_the_gate_is_closed_every_kind_of_paid_call_is_refused():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    m.gate.close()
    sent = []
    c = chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model="m", api_key_env="FAKE_CHAT_KEY", max_tokens=16),
                        transport=lambda *a: (sent.append(1), ok(1, 1))[1], sleep=lambda s: None)
    c.attach_meter(m, "m")
    with pytest.raises(cost.BudgetExceeded):
        c.complete("hello")
    em = embed_meter.EmbedMeter(lambda *a: (sent.append(1), emb_reply(1))[1], cap_tokens=10 ** 6, cost_meter=m, price_key="emb")
    with pytest.raises(embed_meter.CapReached):
        em.transport("u", {}, emb_body("a"), 5)
    assert sent == [] and em.cap_reached.is_set()


# ---- 向量那一路：token 封顶也在门里 ----

def test_the_token_cap_is_checked_inside_the_gate_and_a_reply_above_the_bound_closes_the_whole_gate():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    real = lambda *a: emb_reply(500)                             # 上界 10，回来说 500：上界不成立
    em = embed_meter.EmbedMeter(real, cap_tokens=10 ** 6, cost_meter=m, price_key="emb")
    em.transport("u", {}, emb_body("abcdefgh"), 5)
    assert em.snapshot()["replies_above_bound"] == 1 and m.gate.closed is True and em.cap_reached.is_set()
    with pytest.raises(embed_meter.CapReached):
        em.transport("u", {}, emb_body("a"), 5)
    # token 上限本身：上界够不够，发之前在门里查
    m2 = cost.CostMeter(PRICE, limit_usd=1.0)
    sent = []
    em2 = embed_meter.EmbedMeter(lambda *a: (sent.append(1), emb_reply(10))[1], cap_tokens=15, cost_meter=m2, price_key="emb")
    em2.transport("u", {}, emb_body("abcdefgh"), 5)             # 上界 10 <= 15，实际 10
    with pytest.raises(embed_meter.CapReached):
        em2.transport("u", {}, emb_body("abcdefgh"), 5)         # 已用 10 + 上界 10 > 15
    assert len(sent) == 1 and em2.snapshot()["refused_after_cap"] == 1


def test_a_body_that_cannot_be_read_is_bounded_by_its_byte_length():
    """读不出的请求体：字节数仍然是 token 数的上界，所以按字节数核；不是 0（0 的话任何东西都放得进去）。"""
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    sent = []
    em = embed_meter.EmbedMeter(lambda *a: (sent.append(1), emb_reply(1))[1], cap_tokens=20, cost_meter=m, price_key="emb")
    with pytest.raises(embed_meter.CapReached):
        em.transport("u", {}, b"this is not json and it is longer than twenty bytes", 5)
    assert sent == []
    em.transport("u", {}, b"short body", 5)                                  # 10 个字节 <= 20
    assert len(sent) == 1


# ---- 记账出错（栖第三轮后一处 P1，柯 6379065831）：算不出价格也要关门、按上界记钱、账不留半截 ----

TIERED = {"m": cost.tiered((1000, 1.0, 1.0)), "emb": cost.tiered((1000, 1.0, 0.0))}      # 最后一档 1000 token


def test_usage_beyond_the_last_price_tier_closes_the_gate_books_the_bound_and_leaves_no_half_entry():
    m = cost.CostMeter(TIERED, limit_usd=1.0)
    bound = m.upper_cost("m", 500, 0)
    with pytest.raises(cost.BudgetExceeded):                                   # 回来说 5000：价格函数算不出来
        m.gate.run("m", 500, 0, lambda: ("x", (5000, 10)))
    assert m.gate.closed is True
    assert m.spent == pytest.approx(bound)                                     # 按这次的上界记了钱
    assert m.tokens["m"] == {"prompt": 500, "completion": 0, "calls": 1}        # 账里是上界那一笔，不是「加了用量、钱没加」的半截
    assert m.gate.booking_errors == 1
    sent = []
    with pytest.raises(cost.BudgetExceeded):                                   # 下一次被拒
        m.gate.run("m", 1, 0, lambda: (sent.append(1), (1, 0)))
    assert sent == []


def test_add_works_out_the_price_before_it_touches_the_ledger():
    m = cost.CostMeter(TIERED, limit_usd=1.0)
    with pytest.raises(ValueError):
        m.add("m", 5000, 10)
    assert m.tokens == {} and m.spent == 0                                      # 价格算不出来：一个数都没动


def test_any_booking_error_inside_the_gate_is_booked_at_the_bound_closes_the_gate_and_is_reported():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    bound = m.upper_cost("m", 40, 10)

    def broken_add(*a, **k):
        raise RuntimeError("the ledger broke")
    m.add = broken_add                                                         # 记账本身出错
    with pytest.raises(cost.BudgetExceeded):
        m.gate.run("m", 40, 10, lambda: ("x", (3, 3)))
    assert m.gate.closed is True and m.spent == pytest.approx(bound)
    assert m.tokens["m"] == {"prompt": 40, "completion": 10, "calls": 1}
    assert m.gate.booking_errors == 1


def test_when_even_booking_at_the_bound_fails_the_gate_is_still_closed_and_both_errors_are_carried():
    m = cost.CostMeter(PRICE, limit_usd=1.0)

    def broken_add(*a, **k):
        raise RuntimeError("add broke")

    def broken_book(*a, **k):
        raise OSError("book_at_bound broke")
    m.add, m.book_at_bound = broken_add, broken_book
    with pytest.raises(cost.BudgetExceeded) as ei:
        m.gate.run("m", 40, 10, lambda: ("x", (3, 3)))
    assert m.gate.closed is True and m.gate.booking_errors == 1
    assert isinstance(ei.value.__cause__, RuntimeError)                         # the first error (why it could not be booked)
    assert isinstance(ei.value.__context__, OSError)                            # and the second (why the bound could not be booked)
    sent = []
    with pytest.raises(cost.BudgetExceeded):                                   # the next call is refused
        m.gate.run("m", 1, 0, lambda: (sent.append(1), (1, 0)))
    assert sent == []


def test_a_call_that_raised_and_then_cannot_be_booked_still_reports_its_own_error_and_closes_the_gate():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    bound = m.upper_cost("m", 40, 10)
    m.add = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the ledger broke"))

    def boom():
        raise ConnectionError("the network went away")
    with pytest.raises(ConnectionError):                                       # the call's own error, not the booking one
        m.gate.run("m", 40, 10, boom)
    assert m.gate.closed is True and m.spent == pytest.approx(bound)
    assert m.tokens["m"] == {"prompt": 40, "completion": 10, "calls": 1}


def test_a_chat_reply_with_usage_beyond_the_last_tier_closes_the_gate_the_same_way():
    m = cost.CostMeter(TIERED, limit_usd=1.0)
    c = chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model="m", api_key_env="FAKE_CHAT_KEY", max_tokens=16),
                        transport=lambda *a: ok(5000, 5), sleep=lambda s: None)
    c.attach_meter(m, "m")
    with pytest.raises(cost.BudgetExceeded):
        c.complete("hello")
    assert m.gate.closed is True and m.tokens["m"]["calls"] == 1
    assert m.tokens["m"]["prompt"] == len(c._body("hello")) and m.tokens["m"]["completion"] == 16


def test_an_embedding_reply_beyond_the_last_tier_closes_the_gate_and_is_reported_as_the_cap_being_reached():
    m = cost.CostMeter(TIERED, limit_usd=1.0)
    em = embed_meter.EmbedMeter(lambda *a: emb_reply(5000), cap_tokens=10 ** 6, cost_meter=m, price_key="emb")
    with pytest.raises(embed_meter.CapReached):
        em.transport("u", {}, emb_body("abcdefgh"), 5)
    assert m.gate.closed is True and em.cap_reached.is_set()
    assert m.tokens["emb"] == {"prompt": 10, "completion": 0, "calls": 1}


# ---- 预留那一整套已经删掉 ----

def test_the_reservation_machinery_is_gone_from_the_cost_meter_the_chat_client_and_the_embed_meter():
    for name in ("reserve", "settle", "release", "reserved", "halted", "over_reservation", "_release"):
        assert not hasattr(cost.CostMeter, name) and not hasattr(cost.CostMeter({}, 1.0), name), name
    assert not hasattr(cost, "Reservation")
    here = os.path.join(os.path.dirname(__file__), "..", "eval", "ruler")
    for fname in os.listdir(here):
        if fname.endswith(".py"):
            src = open(os.path.join(here, fname), encoding="utf-8").read()
            for word in ("Reservation", ".reserve(", ".settle(", "reservation", "_inflight_tokens"):
                if fname == "vector_cache.py" and word == "_inflight_tokens":
                    continue
                assert word not in src, f"{fname} still has {word}"
