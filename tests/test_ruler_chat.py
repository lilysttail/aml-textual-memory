"""第二步的接模型那一层：OpenAI 兼容的聊天接口客户端（传输可替换）、用量、花费上限。全部用假接口，不联网、不用钥匙。"""
import json

import pytest

from eval.ruler import chat, cost

FAKE_KEY = "sk-fake-0123456789abcdef"


class FakeTransport:
    """记录每次请求，按顺序吐预设回应。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body.decode("utf-8")), "timeout": timeout})
        status, payload = self.replies.pop(0)
        return status, (payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8"))


def ok(text="Paris", p=11, c=2, model="m-1"):
    return 200, {"model": model, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": p, "completion_tokens": c}}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", FAKE_KEY)


def make(cfg_kw=None, *replies):
    cfg = chat.ChatConfig(base_url="https://example.invalid/v1", model="m-1", api_key_env="FAKE_KEY_ENV",
                          temperature=0.0, max_tokens=128, extra_body={"enable_thinking": False}, user_agent="ua-test",
                          **(cfg_kw or {}))
    t = FakeTransport(*replies)
    return chat.ChatClient(cfg, transport=t, sleep=lambda s: None), t


def test_the_request_is_one_user_message_with_the_pinned_settings():
    c, t = make(None, ok())
    r = c.complete("hello")
    call = t.calls[0]
    assert call["url"] == "https://example.invalid/v1/chat/completions"
    assert call["body"] == {"model": "m-1", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.0,
                            "max_tokens": 128, "enable_thinking": False}
    assert call["headers"]["Authorization"] == "Bearer " + FAKE_KEY and call["headers"]["User-Agent"] == "ua-test"
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.attempts, r.model) == ("Paris", 11, 2, 1, "m-1")


def test_a_model_that_refuses_a_temperature_gets_none_in_the_request():
    c, t = make(None, ok())
    c.cfg = chat.ChatConfig(**{**c.cfg.__dict__, "temperature": None, "max_tokens": None, "extra_body": {"thinking": {"type": "disabled"}}})
    c.complete("x")
    assert t.calls[0]["body"] == {"model": "m-1", "messages": [{"role": "user", "content": "x"}], "thinking": {"type": "disabled"}}


def test_rate_limits_and_server_errors_are_retried_then_given_up():
    c, t = make(None, (429, {"error": {"code": "rate"}}), (503, {}), ok("fine"))
    r = c.complete("x")
    assert r.text == "fine" and r.attempts == 3 and len(t.calls) == 3
    c, t = make({"max_retries": 2}, (500, {}), (500, {}), (500, {}))
    with pytest.raises(chat.ChatError) as e:
        c.complete("x")
    assert e.value.status == 500 and len(t.calls) == 3          # 第一次 + 2 次重试


def test_a_client_error_is_not_retried_and_carries_no_key_and_no_body_text():
    c, t = make(None, (401, {"error": {"message": "bad key " + FAKE_KEY, "code": "invalid_api_key"}}))
    with pytest.raises(chat.ChatError) as e:
        c.complete("x")
    assert e.value.status == 401 and e.value.code == "invalid_api_key" and len(t.calls) == 1
    assert FAKE_KEY not in str(e.value) and "bad key" not in str(e.value)


def test_a_missing_key_stops_before_anything_is_sent(monkeypatch):
    monkeypatch.delenv("FAKE_KEY_ENV")
    c, t = make(None, ok())
    with pytest.raises(chat.ChatError):
        c.complete("x")
    assert t.calls == []


def test_the_key_never_shows_up_in_a_repr():
    c, _ = make(None, ok())
    assert FAKE_KEY not in repr(c) and FAKE_KEY not in repr(c.cfg)


def test_an_unreadable_answer_is_an_error_not_an_empty_text():
    c, _ = make({"max_retries": 0}, (200, b"not json"))
    with pytest.raises(chat.ChatError):
        c.complete("x")
    c, _ = make({"max_retries": 0}, (200, {"choices": []}))
    with pytest.raises(chat.ChatError):
        c.complete("x")


def test_usage_missing_from_the_reply_is_a_visible_none_not_a_zero():
    c, _ = make(None, (200, {"model": "m-1", "choices": [{"message": {"content": "a"}}]}))
    r = c.complete("x")
    assert r.prompt_tokens is None and r.completion_tokens is None


# ---- 花费 ----

def test_cost_is_computed_from_the_reported_tokens_and_the_price_table():
    m = cost.CostMeter({"m-1": (0.5, 2.0)}, limit_usd=1.0)         # 美元 / 百万 token（输入, 输出）
    m.add("m-1", 1_000_000, 100_000)
    assert m.spent == pytest.approx(0.5 + 0.2)
    assert m.tokens["m-1"] == {"prompt": 1_000_000, "completion": 100_000, "calls": 1}


def test_an_unpriced_model_is_counted_in_tokens_and_costs_nothing_by_the_meter():
    m = cost.CostMeter({"quota-model": None}, limit_usd=1.0)         # 会员额度：不按价
    m.add("quota-model", 10, 5)
    assert m.spent == 0 and m.tokens["quota-model"]["prompt"] == 10


def test_an_unknown_model_is_refused_so_nothing_is_spent_unnoticed():
    m = cost.CostMeter({"m-1": (1.0, 1.0)}, limit_usd=1.0)
    with pytest.raises(KeyError):
        m.add("other", 1, 1)


def test_the_limit_stops_the_next_call_not_the_one_in_flight():
    m = cost.CostMeter({"m-1": (1000.0, 1000.0)}, limit_usd=0.5)
    m.check()
    m.add("m-1", 400, 200)                                         # 0.6 美元
    with pytest.raises(cost.BudgetExceeded):
        m.check()


def test_usage_that_the_reply_did_not_report_stops_the_run_rather_than_being_guessed():
    m = cost.CostMeter({"m-1": (1.0, 1.0)}, limit_usd=1.0)
    with pytest.raises(cost.UsageMissing):
        m.add("m-1", None, None)


def test_a_tiered_price_list_prices_the_whole_request_at_its_tier():
    price = cost.tiered((32_000, 0.03, 0.13), (256_000, 0.10, 0.40))
    m = cost.CostMeter({"flash": price}, limit_usd=1.0)
    m.add("flash", 10_000, 100)                                   # 第一档
    m.add("flash", 100_000, 100)                                  # 第二档：整条请求按第二档
    assert m.spent == pytest.approx((10_000 * 0.03 + 100 * 0.13 + 100_000 * 0.10 + 100 * 0.40) / 1e6)
    assert m.spent_by_model() == {"flash": m.spent}
    with pytest.raises(ValueError):
        m.add("flash", 300_000, 1)


def test_the_spending_carries_over_from_one_run_to_the_next():
    m = cost.CostMeter({"m-1": (1000.0, 1000.0)}, limit_usd=1.0)
    m.add("m-1", 300, 100)
    n = cost.CostMeter({"m-1": (1000.0, 1000.0)}, limit_usd=1.0)
    n.restore(m.state())
    assert n.spent == pytest.approx(0.4) and n.tokens["m-1"]["calls"] == 1 and n.spent_by_model() == {"m-1": pytest.approx(0.4)}
    n.add("m-1", 400, 200)                                         # 又 0.6 → 共 1.0
    with pytest.raises(cost.BudgetExceeded):
        n.check()


def test_a_request_of_exactly_a_tiers_size_is_priced_at_that_tier():
    price = cost.tiered((32_000, 0.03, 0.13), (256_000, 0.10, 0.40))
    assert price(32_000) == (0.03, 0.13) and price(32_001) == (0.10, 0.40) and price(256_000) == (0.10, 0.40)
