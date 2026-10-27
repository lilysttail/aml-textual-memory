"""判分一致性那条命令（consistency-prepare）也走同一道硬封顶——两个判分客户端挂同一张表，删掉事后 meter.add，
被拒就停；同一份 spend.json 同一个上限，只有一种记账方式。假接口、假钥匙。"""
import json
import os

import pytest

from eval.ruler import chat, corpus, cost, qa_main
from eval.ruler.testing import write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_CHAT_KEY", "test-key-NOT-A-REAL-KEY")


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def reply(prompt_tokens, completion_tokens, label="CORRECT"):
    return 200, json.dumps({"choices": [{"message": {"content": json.dumps({"label": label})}, "finish_reason": "stop"}], "model": "j",
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}).encode("utf-8")


class Transport:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        return self.fn(body)


def judge_client(model, transport):
    return chat.ChatClient(chat.ChatConfig(base_url="https://x.invalid/v1", model=model, api_key_env="FAKE_CHAT_KEY"), transport=transport, sleep=lambda s: None)


def rows_of(mini):
    # 回答全是错的，判分必须真的发请求（答对了会走「字面相等」不发请求）
    return [{"group": "G2", "model_key": "main", "qa_id": q["qa_id"], "answer": "a completely different thing", "outcome": "made_up"}
            for q in mini.questions[:3]]


def prices():
    return {qa_main.JUDGE_MODEL: (0.3, 2.4), qa_main.SECOND_JUDGE_MODEL: (0.3, 2.4)}


def test_both_judges_are_metered_by_the_client_and_each_call_is_counted_once(tmp_path, mini, monkeypatch):
    monkeypatch.setattr(qa_main, "PRICES", dict(qa_main.PRICES, **prices()))
    out = str(tmp_path / "out")
    t1 = Transport(lambda b: reply(80, 5))
    t2 = Transport(lambda b: reply(90, 6))
    res = qa_main.prepare_consistency(mini, out, rows_of(mini), judge_client(qa_main.JUDGE_MODEL, t1), judge_client(qa_main.SECOND_JUDGE_MODEL, t2))
    spend = json.loads(open(os.path.join(out, "spend.json"), encoding="utf-8").read())
    assert spend["tokens"][qa_main.JUDGE_MODEL] == {"prompt": 80 * t1.calls, "completion": 5 * t1.calls, "calls": t1.calls}
    assert spend["tokens"][qa_main.SECOND_JUDGE_MODEL]["calls"] == t2.calls and t1.calls > 0 and t2.calls > 0
    assert res["sample"] == 3
    assert os.path.exists(os.path.join(out, "consistency", "ours.json")) and os.path.exists(os.path.join(out, "consistency", "ours_second_judge.json"))


def test_a_judge_call_the_wall_refuses_is_not_sent_and_the_command_stops_without_half_written_scores(tmp_path, mini, monkeypatch):
    monkeypatch.setattr(qa_main, "PRICES", dict(qa_main.PRICES, **prices()))
    out = str(tmp_path / "out")
    t1 = Transport(lambda b: reply(80, 5))
    t2 = Transport(lambda b: reply(80, 5))
    with pytest.raises(cost.BudgetExceeded):
        qa_main.prepare_consistency(mini, out, rows_of(mini), judge_client(qa_main.JUDGE_MODEL, t1), judge_client(qa_main.SECOND_JUDGE_MODEL, t2),
                                    limit_usd=1e-9)
    assert t1.calls == 0 and t2.calls == 0                       # 一次都没发
    assert not os.path.exists(os.path.join(out, "consistency", "ours.json")), "没跑完的分数不落盘"
    assert json.loads(open(os.path.join(out, "spend.json"), encoding="utf-8").read())["spent"] == 0


def test_a_refusal_on_the_first_judge_stops_the_command_before_the_second_judge_is_asked(tmp_path, mini, monkeypatch):
    """第一个判分被墙拒了，第二个判分本来还够额度（按计划额度付费、不占额度）：命令也得在那里停，第二个一次都不问、分数不落盘。"""
    monkeypatch.setattr(qa_main, "PRICES", dict(qa_main.PRICES, **{qa_main.JUDGE_MODEL: (1000.0, 1000.0), qa_main.SECOND_JUDGE_MODEL: None}))
    out = str(tmp_path / "out")
    t1 = Transport(lambda b: reply(80, 5))
    t2 = Transport(lambda b: reply(80, 5))
    with pytest.raises(cost.BudgetExceeded):
        qa_main.prepare_consistency(mini, out, rows_of(mini), judge_client(qa_main.JUDGE_MODEL, t1), judge_client(qa_main.SECOND_JUDGE_MODEL, t2), limit_usd=0.001)
    assert t1.calls == 0 and t2.calls == 0
    assert not os.path.exists(os.path.join(out, "consistency", "ours.json"))


def test_when_the_limit_only_covers_one_call_only_one_is_sent_and_the_next_is_refused(tmp_path, mini, monkeypatch):
    monkeypatch.setattr(qa_main, "PRICES", dict(qa_main.PRICES, **prices()))
    seen = {}

    def whole_bound(body):                                         # 这次调用实际花满了预留的上界：提示 = 请求字节数，输出 = 固定上界
        seen["bytes"] = len(body)
        return reply(len(body), chat.ASSUMED_COMPLETION_UPPER_BOUND)
    t1 = Transport(whole_bound)
    one_call = (700 * 0.3 + chat.ASSUMED_COMPLETION_UPPER_BOUND * 2.4) / 1e6       # 上界的量级（请求体几百字节）
    out = str(tmp_path / "out")
    with pytest.raises(cost.BudgetExceeded):
        qa_main.prepare_consistency(mini, out, rows_of(mini), judge_client(qa_main.JUDGE_MODEL, t1), judge_client(qa_main.SECOND_JUDGE_MODEL, Transport(whole_bound)),
                                    limit_usd=one_call * 1.4)
    assert t1.calls == 1, "上限只够一次：发出一次，第二次被拒"
    spend = json.loads(open(os.path.join(out, "spend.json"), encoding="utf-8").read())
    assert spend["tokens"][qa_main.JUDGE_MODEL]["calls"] == 1     # 发出去的那次照实记了，没发的没记


def test_the_command_has_no_after_the_fact_booking_left():
    import inspect
    src = inspect.getsource(qa_main)
    assert "meter.add(JUDGE_MODEL" not in src and "meter.add(SECOND_JUDGE_MODEL" not in src
    assert "prepare_consistency(" in src
