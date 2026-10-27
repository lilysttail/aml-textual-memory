"""multi-model preparation: the ruler talks to the OpenCode Go gateway (five small answering models), the judge request has a length
cap that cannot cut a verdict, and every new model is priced in the spending table. Fake transport only: no network, no key, no money."""
import json

import pytest

from eval.ruler import chat, cost, judge, qa_main

FAKE_KEY = "sk-fake-0123456789abcdef"
MODELS = ("hy3", "qwen3.8-flash", "minimax-m3.1-flash", "mimo-v2.6-flash", "glm-5.3-flash")
# the longest single judge reply in every result the ruler has kept so far: 357 judge calls over 7 result files, 12 or 17 completion tokens
OBSERVED_LONGEST_JUDGE_REPLY_TOKENS = 17


class Fake:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body.decode("utf-8")), "raw": body})
        status, payload = self.replies.pop(0)
        return status, (payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8"))


def openai_ok(text="a puppy", p=120, c=4, finish="stop"):
    return 200, {"model": "m", "choices": [{"message": {"content": text}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": p, "completion_tokens": c}}


def anthropic_ok(blocks=None, p=120, c=4, stop="end_turn"):
    return 200, {"model": "m", "type": "message", "role": "assistant", "stop_reason": stop,
                 "content": blocks if blocks is not None else [{"type": "text", "text": "a puppy"}],
                 "usage": {"input_tokens": p, "output_tokens": c}}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", FAKE_KEY)
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


# ---- the five models are configured the way the gateway needs ----

def test_the_five_models_are_listed_and_each_has_one_config():
    assert set(qa_main.OPENCODE_MODELS) == set(MODELS)
    for m in MODELS:
        c = qa_main.opencode_config(m)
        assert c.model == m and c.api_key_env == "OPENCODE_GO_API_KEY"
        assert c.temperature == 0.0 and c.max_tokens == 128
        assert c.base_url.startswith("https://opencode.ai/")


def test_an_unknown_model_is_refused_not_guessed():
    with pytest.raises(ValueError):
        qa_main.opencode_config("gpt-9-turbo")


def test_the_gateway_headers_are_there_honest_and_stable_for_one_config():
    c = qa_main.opencode_config("hy3")
    sid = c.headers["x-opencode-session"]
    assert sid.startswith("aml-ruler-") and len(sid) > len("aml-ruler-") + 8        # a stable id for this run
    assert c.user_agent == qa_main.OPENCODE_USER_AGENT and "aml-ruler" in c.user_agent and "Claude" not in c.user_agent
    assert qa_main.opencode_config("hy3").headers["x-opencode-session"] != sid                         # another config, another run


def test_the_four_openai_style_models_turn_reasoning_off_the_way_the_home_gateway_calls_do():
    for m in ("hy3", "minimax-m3.1-flash", "mimo-v2.6-flash", "glm-5.3-flash"):
        c = qa_main.opencode_config(m)
        assert c.api_style == "openai" and c.extra_body == {"reasoning_effort": "none"}
        assert c.base_url == "https://opencode.ai/zen/go/v1"


def test_qwen38_flash_is_the_one_the_gateway_docs_put_on_the_messages_endpoint():
    c = qa_main.opencode_config("qwen3.8-flash")
    assert c.api_style == "anthropic" and c.base_url == "https://opencode.ai/inference/go/anthropic/v1"


# ---- the client: extra headers, and the messages style ----

def test_extra_headers_go_out_but_cannot_replace_the_key_or_the_content_type():
    cfg = chat.ChatConfig(base_url="https://x.invalid/v1", model="m", api_key_env="OPENCODE_GO_API_KEY", max_tokens=128,  # gate:allow-secret
                          headers={"x-opencode-session": "sid-1", "Authorization": "Bearer EVIL", "content-type": "text/plain"})
    f = Fake(openai_ok())
    chat.ChatClient(cfg, transport=f, sleep=lambda s: None).complete("hi")
    h = f.calls[0]["headers"]
    assert h["x-opencode-session"] == "sid-1"
    assert h["Authorization"] == "Bearer " + FAKE_KEY and h["Content-Type"] == "application/json"
    assert "EVIL" not in json.dumps(h) and "text/plain" not in json.dumps(h)


def test_the_session_header_is_the_same_on_a_retry():
    c = qa_main.opencode_config("hy3")
    f = Fake((503, {"error": {"code": "unavailable"}}), openai_ok())
    chat.ChatClient(c, transport=f, sleep=lambda s: None).complete("hi")
    assert len(f.calls) == 2 and f.calls[0]["headers"]["x-opencode-session"] == f.calls[1]["headers"]["x-opencode-session"]


def test_the_messages_style_sends_a_messages_request_and_reads_the_reply():
    c = qa_main.opencode_config("qwen3.8-flash")
    f = Fake(anthropic_ok([{"type": "text", "text": "a puppy"}], p=321, c=6, stop="end_turn"))
    r = chat.ChatClient(c, transport=f, sleep=lambda s: None).complete("What did she adopt?")
    call = f.calls[0]
    assert call["url"] == "https://opencode.ai/inference/go/anthropic/v1/messages"
    assert call["body"] == {"model": "qwen3.8-flash", "max_tokens": 128, "temperature": 0.0,
                            "messages": [{"role": "user", "content": "What did she adopt?"}]}
    assert call["headers"]["anthropic-version"] == "2023-06-01" and call["headers"]["x-opencode-session"].startswith("aml-ruler-")
    assert call["headers"]["x-api-key"] == FAKE_KEY and call["headers"]["Authorization"] == "Bearer " + FAKE_KEY       # the docs do not say which one the gateway reads
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.finish_reason) == ("a puppy", 321, 6, "end_turn")


def test_the_openai_style_sends_no_messages_only_headers_and_the_messages_style_always_sets_max_tokens():
    f = Fake(openai_ok())
    chat.ChatClient(qa_main.opencode_config("hy3"), transport=f, sleep=lambda s: None).complete("q")
    assert "x-api-key" not in f.calls[0]["headers"] and "anthropic-version" not in f.calls[0]["headers"]
    assert f.calls[0]["url"] == "https://opencode.ai/zen/go/v1/chat/completions"
    cfg = chat.ChatConfig(base_url="https://x.invalid/v1", model="m", api_key_env="OPENCODE_GO_API_KEY", api_style="anthropic", max_tokens=None)  # gate:allow-secret
    f2 = Fake(anthropic_ok())
    chat.ChatClient(cfg, transport=f2, sleep=lambda s: None).complete("q")
    assert f2.calls[0]["body"]["max_tokens"] == chat.ASSUMED_COMPLETION_UPPER_BOUND                   # the messages API insists on one


def test_the_messages_style_keeps_only_text_blocks_and_joins_them():
    c = qa_main.opencode_config("qwen3.8-flash")
    f = Fake(anthropic_ok([{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "a "}, {"type": "text", "text": "puppy"}]))
    assert chat.ChatClient(c, transport=f, sleep=lambda s: None).complete("q").text == "a puppy"


def test_a_messages_reply_that_cannot_be_read_is_an_error_and_charged_at_the_bound():
    c = qa_main.opencode_config("qwen3.8-flash")
    m = cost.CostMeter(qa_main.PRICES, limit_usd=1.0)
    f = Fake((200, {"content": "not a list", "usage": {}}))
    cl = chat.ChatClient(c, transport=f, sleep=lambda s: None)
    cl.attach_meter(m, "qwen3.8-flash")
    with pytest.raises(chat.ChatError):
        cl.complete("q")
    assert m.tokens["qwen3.8-flash"]["calls"] == 1 and m.spent > 0                      # the provider may have charged it


def test_the_spending_bound_of_a_messages_call_is_the_body_bytes_plus_max_tokens():
    c = qa_main.opencode_config("qwen3.8-flash")
    m = cost.CostMeter(qa_main.PRICES, limit_usd=1.0)
    f = Fake(anthropic_ok(p=10, c=3))
    cl = chat.ChatClient(c, transport=f, sleep=lambda s: None)
    cl.attach_meter(m, "qwen3.8-flash")
    cl.complete("q")
    assert m.tokens["qwen3.8-flash"] == {"prompt": 10, "completion": 3, "calls": 1}
    bound = m.upper_cost("qwen3.8-flash", len(f.calls[0]["raw"]), 128)
    assert 0 < bound < 0.001


# ---- prices ----

def test_every_new_model_is_priced_from_the_go_price_list_and_goes_through_the_gate():
    expected = {"hy3": (0.14, 0.58), "qwen3.8-flash": (0.15, 0.47), "minimax-m3.1-flash": (0.15, 0.60),
                "mimo-v2.6-flash": (0.14, 0.28), "glm-5.3-flash": (0.15, 0.50)}
    for m, price in expected.items():
        assert qa_main.PRICES[m] == price
    m = cost.CostMeter(qa_main.PRICES, limit_usd=0.0)
    for name in MODELS:                                               # a zero limit refuses every one: the plan's models are counted at list price
        with pytest.raises(cost.BudgetExceeded):
            m.gate.run(name, 1000, 128, lambda: ("x", None))


def test_who_accepts_the_five_ids_and_the_old_three_and_nothing_else():
    ap = qa_main.build_parser()
    for who in ("main", "kimi", "haiku") + MODELS:
        ap.parse_args(["run", "--out", "o", "--data", "d", "--who", who, "--groups", "G2"])
    with pytest.raises(SystemExit):
        ap.parse_args(["run", "--out", "o", "--data", "d", "--who", "gpt-9", "--groups", "G2"])


def test_ask_for_builds_a_metered_asker_priced_under_the_model_id():
    for m in MODELS:
        ask, price_key = qa_main._ask_for(m, None)
        assert price_key == m and ask.client.cfg.model == m


# ---- the judge's length cap ----

def test_the_judge_cap_leaves_room_for_the_longest_verdict_seen_many_times_over():
    assert qa_main.JUDGE_MAX_TOKENS >= 4 * OBSERVED_LONGEST_JUDGE_REPLY_TOKENS
    for c in (qa_main.judge_config(), qa_main.judge_config(qa_main.SECOND_JUDGE_MODEL)):
        assert c.max_tokens == qa_main.JUDGE_MAX_TOKENS and c.extra_body == {"enable_thinking": False} and c.temperature == 0.0


def test_a_verdict_of_the_longest_length_seen_is_read_under_the_cap():
    fenced = '```json\n{\n    "label": "CORRECT"\n}\n```'
    cfg = qa_main.judge_config()
    f = Fake(openai_ok(fenced, c=OBSERVED_LONGEST_JUDGE_REPLY_TOKENS, finish="stop"))
    j = judge.RefinedJudge(chat.ChatClient(cfg, transport=f, sleep=lambda s: None))
    assert j.score("q", "a puppy", "a dog").score == 1
    assert f.calls[0]["body"]["max_tokens"] == qa_main.JUDGE_MAX_TOKENS


def test_a_verdict_cut_by_the_cap_is_an_error_at_once_never_a_wrong_answer_and_never_asked_again():
    cfg = qa_main.judge_config()
    f = Fake(openai_ok('```json\n{\n    "lab', c=qa_main.JUDGE_MAX_TOKENS, finish="length"), openai_ok('{"label": "CORRECT"}'))
    j = judge.RefinedJudge(chat.ChatClient(cfg, transport=f, sleep=lambda s: None))
    with pytest.raises(judge.JudgeError) as e:
        j.score("q", "a puppy", "a dog")
    assert "cut" in str(e.value) and len(f.calls) == 1


def test_a_cut_reply_that_already_holds_a_whole_label_is_still_read():
    cfg = qa_main.judge_config()
    f = Fake(openai_ok('{"label": "WRONG"} and then some more words', c=qa_main.JUDGE_MAX_TOKENS, finish="length"))
    j = judge.RefinedJudge(chat.ChatClient(cfg, transport=f, sleep=lambda s: None))
    assert j.score("q", "a puppy", "a cat").score == 0


def test_the_gate_reserve_for_a_judge_call_is_the_cap_not_four_thousand_tokens():
    cfg = qa_main.judge_config()
    f = Fake(openai_ok('{"label": "CORRECT"}', p=900, c=17))
    cl = chat.ChatClient(cfg, transport=f, sleep=lambda s: None)
    probe = chat.ChatClient(cfg, transport=Fake(openai_ok()), sleep=lambda s: None)
    body_len = len(probe._body("x" * 3000))
    reserve = (body_len * 0.3 + qa_main.JUDGE_MAX_TOKENS * 2.4) / 1_000_000
    m = cost.CostMeter(qa_main.PRICES, limit_usd=reserve + 1e-6)             # room for exactly that reserve: with 4096 tokens it would be refused
    cl.attach_meter(m, qa_main.JUDGE_MODEL)
    assert cl.complete("x" * 3000).text.startswith("{")
