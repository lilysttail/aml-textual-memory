"""答题 / 判分那条线也做成硬封顶，用的是同一道付费门（见 test_ruler_paid_gate.py）：
每次尝试（重试也算）在门里发——先看「已花 + 请求体字节数和 max_tokens 算出来的上界」够不够，发，记实际；读不出用量、坏 200、网络错误按上界记；
错误回复不记。这里钉的是聊天客户端这一头。假接口、假钥匙。"""
import json

import pytest

from eval.ruler import chat, corpus, cost, qa_main
from eval.ruler.testing import write_mini_corpus

PRICE = {"m": (1.0, 2.0), "free": None}                 # 每百万 token 的美元：输入 1、输出 2


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_CHAT_KEY", "test-key-NOT-A-REAL-KEY")


def cfg(max_tokens=None, model="m"):
    return chat.ChatConfig(base_url="https://x.invalid/v1", model=model, api_key_env="FAKE_CHAT_KEY", max_tokens=max_tokens, max_retries=2)


def ok(prompt_tokens, completion_tokens, text="hi"):
    return 200, json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "model": "m",
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}).encode("utf-8")


class Transport:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(transport, meter, price_key="m", max_tokens=100):
    c = chat.ChatClient(cfg(max_tokens=max_tokens), transport=transport, sleep=lambda s: None)
    c.attach_meter(meter, price_key)
    return c


def test_the_upper_bound_is_the_request_bytes_for_the_prompt_plus_the_most_it_may_write():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    c = client(Transport(ok(10, 5)), m, max_tokens=100)
    body_len = len(c._body("hello"))
    seen = {}
    real = m.upper_cost

    def spy(model, p, q=0):
        seen["args"] = (model, p, q)
        return real(model, p, q)
    m.upper_cost = spy
    c.complete("hello")
    assert seen["args"] == ("m", body_len, 100)
    c2 = chat.ChatClient(cfg(max_tokens=None), transport=Transport(ok(1, 1)), sleep=lambda s: None)      # 官方判分那种没设 max_tokens 的形状
    c2.attach_meter(m, "m")
    c2.complete("hello")
    assert seen["args"][2] == chat.ASSUMED_COMPLETION_UPPER_BOUND


def test_a_reply_is_booked_by_what_it_reports_and_the_result_is_marked_metered():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    r = client(Transport(ok(40, 10)), m).complete("hello")
    assert r.metered is True
    assert m.tokens["m"] == {"prompt": 40, "completion": 10, "calls": 1}
    assert m.spent == pytest.approx((40 * 1.0 + 10 * 2.0) / 1e6)


def test_when_the_limit_only_covers_one_call_the_first_is_sent_and_the_second_is_refused():
    c0 = chat.ChatClient(cfg(max_tokens=100), transport=Transport(), sleep=lambda s: None)
    n = len(c0._body("hello"))
    one_call = (n * 1.0 + 100 * 2.0) / 1e6                                              # 一次调用的上界
    m = cost.CostMeter(PRICE, limit_usd=one_call * 1.5)                                 # 只够一次
    t = Transport(ok(n, 100), ok(5, 5))                                                 # 第一次实际花满上界
    c = client(t, m)
    c.complete("hello")
    with pytest.raises(cost.BudgetExceeded):
        c.complete("hello")
    assert t.calls == 1


def test_a_call_that_would_pass_the_limit_is_refused_before_anything_is_sent():
    m = cost.CostMeter(PRICE, limit_usd=0.00001)
    t = Transport(ok(1, 1))
    with pytest.raises(cost.BudgetExceeded):
        client(t, m, max_tokens=100).complete("hello " * 200)
    assert t.calls == 0 and m.spent == 0


def test_a_retried_call_goes_through_the_gate_per_attempt_and_an_error_reply_is_not_charged():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    t = Transport((429, b"{}"), (503, b"{}"), ok(20, 4))
    r = client(t, m).complete("hello")
    assert r.attempts == 3 and t.calls == 3 and m.gate.calls == 3
    assert m.tokens["m"] == {"prompt": 20, "completion": 4, "calls": 1}                # 只有成功的那次记账


def test_a_network_error_may_still_have_been_charged_so_it_is_booked_at_the_bound():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    c = client(Transport(TimeoutError(), ok(8, 2)), m)
    c.complete("hello")
    assert m.tokens["m"]["calls"] == 2 and m.tokens["m"]["completion"] == 100 + 2       # 超时那次按上界记，成功那次按回报记


def test_a_reply_without_usage_is_booked_at_the_bound_not_as_free_and_does_not_crash():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    c = client(Transport((200, json.dumps({"choices": [{"message": {"content": "x"}}]}).encode("utf-8"))), m)
    r = c.complete("hello")
    assert r.metered is True and r.prompt_tokens is None
    assert m.tokens["m"]["completion"] == 100 and m.tokens["m"]["prompt"] > 0


def test_a_200_reply_that_cannot_be_read_is_booked_at_the_bound_and_the_error_is_raised():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    c = client(Transport((200, b"this is not json at all")), m)
    with pytest.raises(chat.ChatError):
        c.complete("hello")
    assert m.tokens["m"]["calls"] == 1 and m.tokens["m"]["completion"] == 100 and m.tokens["m"]["prompt"] == len(c._body("hello"))
    m2 = cost.CostMeter(PRICE, limit_usd=1.0)
    c2 = client(Transport((200, json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 3}}).encode("utf-8"))), m2)
    with pytest.raises(chat.ChatError):
        c2.complete("hello")
    assert m2.tokens["m"]["calls"] == 1


def test_a_reply_that_costs_more_than_the_bound_closes_the_gate_and_every_later_call_is_refused():
    m = cost.CostMeter(PRICE, limit_usd=1.0)
    client(Transport(ok(10_000, 10)), m).complete("hello")                              # 比上界贵得多，但还远没到上限（0.01 美元 < 1）
    assert m.spent < m.limit_usd / 10 and m.gate.closed is True
    t = Transport(ok(1, 1))
    with pytest.raises(cost.BudgetExceeded):
        client(t, m).complete("hello")
    assert t.calls == 0


def test_a_model_paid_by_quota_holds_no_room_but_its_tokens_are_counted_once():
    m = cost.CostMeter(PRICE, limit_usd=0.0)
    client(Transport(ok(30, 6)), m, price_key="free").complete("hello")
    assert m.tokens["free"] == {"prompt": 30, "completion": 6, "calls": 1} and m.spent == 0 and m.gate.closed is False


def test_a_client_without_a_meter_works_as_before_and_is_not_marked_metered():
    r = chat.ChatClient(cfg(), transport=Transport(ok(3, 3)), sleep=lambda s: None).complete("hello")
    assert r.metered is False


# ---- 整个运行：执行器把表挂到答题和判分的客户端上，计数不重复 ----

@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def test_a_run_attaches_its_meter_to_the_answer_and_judge_clients_and_counts_each_call_once(tmp_path, mini):
    prices = {"ans": (1.0, 2.0), "jdg": (1.0, 2.0)}
    ans_client = chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model="ans", api_key_env="FAKE_CHAT_KEY", max_tokens=16),
                                 transport=Transport(*[ok(50, 3, "a puppy") for _ in range(8)]), sleep=lambda s: None)
    jdg_client = chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model="jdg", api_key_env="FAKE_CHAT_KEY"),
                                 transport=Transport(*[ok(60, 2, '{"label": "CORRECT"}') for _ in range(8)]), sleep=lambda s: None)
    ask = qa_main._model_ask(ans_client)
    rows = qa_main.execute_run(mini, str(tmp_path / "out"), "main", ["G2"], "main", 1.0, ask=ask, price_key="ans", judge_client=jdg_client,
                               judge_price_key="jdg", prices=prices, conv_ids=["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    spend = json.loads((tmp_path / "out" / "spend.json").read_text(encoding="utf-8"))
    assert spend["tokens"]["ans"]["calls"] == len(rows) == 4                              # 每次调用记一次，不是两次
    assert spend["tokens"]["ans"]["prompt"] == 4 * 50 and spend["tokens"]["jdg"]["calls"] == sum(r["judge_calls"] for r in rows)


def test_a_run_whose_call_is_refused_by_the_wall_stops_and_says_so(tmp_path, mini):
    prices = {"ans": (1.0, 2.0), "jdg": (1.0, 2.0)}
    ans_client = chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model="ans", api_key_env="FAKE_CHAT_KEY", max_tokens=16),
                                 transport=Transport(*[ok(50, 3) for _ in range(8)]), sleep=lambda s: None)
    ask = qa_main._model_ask(ans_client)
    with pytest.raises(cost.BudgetExceeded, match="stopped at the spending limit"):
        qa_main.execute_run(mini, str(tmp_path / "out"), "main", ["G2"], "main", 1e-9, ask=ask, price_key="ans", judge_client=None,
                            judge_price_key="jdg", prices=prices, conv_ids=["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    assert ans_client._transport.calls == 0


def test_the_command_line_builds_metered_clients_for_the_main_and_kimi_models_and_the_judge():
    ask, key = qa_main._ask_for("main", None)
    assert getattr(ask, "client", None) is not None and hasattr(ask.client, "attach_meter") and key == qa_main.ANSWER_MODEL
    ask_k, key_k = qa_main._ask_for("kimi", None)
    assert hasattr(ask_k.client, "attach_meter") and key_k == qa_main.KIMI_MODEL
