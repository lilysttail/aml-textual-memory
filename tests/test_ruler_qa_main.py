"""第二步的总控：写死的模型和设置、抽题落盘、跨几次运行累计花费、只许碰「调」的那 5 段、出表落盘。假接口，合成小题库。"""
import json
import os

import pytest

from eval.ruler import chat, corpus, cost, judge, qa_main, split
from eval.ruler.testing import write_mini_corpus


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


# ---- 写死的设置 ----

def test_the_main_answering_model_and_its_settings_are_pinned():
    c = qa_main.main_config()
    assert (c.model, c.base_url, c.api_key_env) == ("qwen3.7-flash-2026-07-15", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY")
    assert (c.temperature, c.max_tokens, c.extra_body) == (0.0, 128, {"enable_thinking": False})


def test_the_judge_is_a_stand_in_with_the_official_request_shape_plus_a_length_cap_because_the_official_model_is_not_served():
    c = qa_main.judge_config()
    assert (c.model, c.temperature, c.max_tokens, c.extra_body) == ("qwen3.5-27b", 0.0, qa_main.JUDGE_MAX_TOKENS, {"enable_thinking": False})
    assert c.model != judge.OFFICIAL_JUDGE_MODEL == "qwen3-14b"
    assert qa_main.judge_config(qa_main.SECOND_JUDGE_MODEL).model == "qwen3.6-27b"


def test_kimi_gets_no_temperature_and_thinking_off_and_an_honest_user_agent():
    c = qa_main.kimi_config()
    assert c.base_url == "https://api.kimi.com/coding/v1" and c.api_key_env == "KIMI_CODE_API_KEY"
    assert c.temperature is None and c.extra_body == {"thinking": {"type": "disabled"}} and c.max_tokens == 128
    assert "Claude" not in c.user_agent and c.model in ("k3-256k", "k3")


def test_every_model_in_use_has_a_price_entry_and_the_flash_tiers_are_the_list_prices():
    for m in (qa_main.ANSWER_MODEL, qa_main.JUDGE_MODEL, qa_main.SECOND_JUDGE_MODEL, qa_main.KIMI_MODEL, qa_main.HAIKU_MODEL):
        assert m in qa_main.PRICES
    f = qa_main.PRICES[qa_main.ANSWER_MODEL]
    assert f(10_000) == (0.03, 0.13) and f(100_000) == (0.10, 0.40)
    assert qa_main.PRICES[qa_main.KIMI_MODEL] is None and qa_main.PRICES[qa_main.HAIKU_MODEL] is None


def test_the_run_only_ever_uses_the_tuning_conversations():
    assert qa_main.CONVERSATIONS == split.TUNE and not set(qa_main.CONVERSATIONS) & set(split.LOCK)


# ---- 总控 ----

def fake_ask(q, memories):
    return chat.ChatResult(text="a puppy", prompt_tokens=100, completion_tokens=5, model="ans-m")


class JudgeChat:
    def complete(self, prompt):
        return chat.ChatResult(text='{"label": "CORRECT"}', prompt_tokens=100, completion_tokens=3, model="jdg-m")


PR = {"ans-m": (1.0, 2.0), "jdg-m": (1.0, 2.0)}


def execute(tmp_path, mini, **kw):
    args = dict(corpus=mini, out_dir=str(tmp_path / "out"), who="main", groups=["G2"], subset="main", limit_usd=1.0,
                ask=fake_ask, price_key="ans-m", judge_client=JudgeChat(), judge_price_key="jdg-m", prices=PR,
                conv_ids=["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    args.update(kw)
    return qa_main.execute_run(**args)


def test_a_run_writes_the_picks_the_rows_and_the_spending(tmp_path, mini):
    rows = execute(tmp_path, mini)
    out = tmp_path / "out"
    picks = json.loads((out / "picks.json").read_text(encoding="utf-8"))
    assert sorted(picks) == ["cross", "main"] and len(picks["main"]) == 4 and len(rows) == 4
    assert len((out / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    spend = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    assert spend["spent"] > 0 and spend["tokens"]["ans-m"]["calls"] == 4


def test_the_picks_are_made_once_and_kept(tmp_path, mini):
    execute(tmp_path, mini)
    first = (tmp_path / "out" / "picks.json").read_text(encoding="utf-8")
    execute(tmp_path, mini)                                   # 同样的参数再跑：用落盘的那份，文件没变
    assert (tmp_path / "out" / "picks.json").read_text(encoding="utf-8") == first
    with pytest.raises(ValueError):                           # 参数变了（每类 3 题）：落盘的不是这份，整轮拒掉，也不重抽、不改文件
        execute(tmp_path, mini, per_category=3)
    assert (tmp_path / "out" / "picks.json").read_text(encoding="utf-8") == first


def test_the_second_run_starts_from_what_the_first_one_spent(tmp_path, mini):
    execute(tmp_path, mini)
    first = json.loads((tmp_path / "out" / "spend.json").read_text(encoding="utf-8"))["spent"]
    execute(tmp_path, mini, groups=["G3"])
    second = json.loads((tmp_path / "out" / "spend.json").read_text(encoding="utf-8"))["spent"]
    assert second > first
    with pytest.raises(cost.BudgetExceeded):
        execute(tmp_path, mini, groups=["G1"], limit_usd=second)             # 累计已经到上限：这一轮一题都不开始


def test_the_subset_decides_which_questions_are_asked(tmp_path, mini):
    rows = execute(tmp_path, mini, subset="cross")
    assert len(rows) == 4                                                   # 每类 1 题，交叉核是主集的前 1 题/类 → 也是 4


def test_the_locked_conversations_are_refused_before_anything_is_written(tmp_path, mini):
    with pytest.raises(ValueError):
        execute(tmp_path, mini, conv_ids=["conv-26", "mini-1"])
    assert not (tmp_path / "out").exists()


def test_the_report_is_written_from_the_rows_and_has_no_text(tmp_path, mini):
    execute(tmp_path, mini)
    rep = qa_main.write_report(str(tmp_path / "out"))
    out = tmp_path / "out"
    assert rep["outcomes"]["G2"]["main"]["questions"] == 4
    assert (out / "report.json").exists() and (out / "report.md").exists()
    text = (out / "report.json").read_text(encoding="utf-8") + (out / "report.md").read_text(encoding="utf-8")
    assert "puppy" not in text and "Alice" not in text


def test_the_compare_command_needs_no_corpus_and_writes_the_agreement(tmp_path):
    d = tmp_path / "out" / "consistency"
    d.mkdir(parents=True)
    (d / "ours.json").write_text('{"q1": 1, "q2": 0}', encoding="utf-8")
    (d / "ours_second_judge.json").write_text('{"q1": 1, "q2": 1}', encoding="utf-8")
    official = tmp_path / "scored.jsonl"
    official.write_text('{"qa_id": "q1", "llm_score": 1.0}\n{"qa_id": "q2", "llm_score": 0.0}\n', encoding="utf-8")
    assert qa_main.main(["consistency-compare", "--out", str(tmp_path / "out"), "--official", str(official)]) == 0
    res = json.loads((d / "agreement.json").read_text(encoding="utf-8"))
    assert res["ours_vs_official_script"]["differ"] == 0 and res["two_stand_in_judges"]["differing_ids"] == ["q2"]


# ---- picks.json 读回来以后再核一遍 ----

def _write_picks(tmp_path, main, cross=None):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "picks.json").write_text(json.dumps({"main": main, "cross": cross if cross is not None else main[:1]}), encoding="utf-8")


def test_a_hand_edited_picks_file_with_a_locked_question_is_refused_and_nothing_is_asked(tmp_path, mini):
    for q in mini.questions:
        if q["qa_id"] == "q5":
            q["sample_id"] = "conv-26"                       # 把一道题放进锁着的对话
    _write_picks(tmp_path, ["q1", "q5"])
    calls = []

    def ask(q, memories):
        calls.append(q)
        return fake_ask(q, memories)

    with pytest.raises(ValueError):
        execute(tmp_path, mini, ask=ask)
    assert calls == [] and not (tmp_path / "out" / "rows.jsonl").exists() and not (tmp_path / "out" / "spend.json").exists()


def test_a_picks_question_outside_the_named_conversations_is_refused(tmp_path, mini):
    _write_picks(tmp_path, ["q1", "q4"])                     # q4 属于 mini-2，这次只许 mini-1
    with pytest.raises(ValueError):
        execute(tmp_path, mini, conv_ids=["mini-1"])
    assert not (tmp_path / "out" / "rows.jsonl").exists()


def test_a_picks_id_that_is_not_in_the_corpus_is_refused(tmp_path, mini):
    _write_picks(tmp_path, ["q1", "no-such-question"])
    with pytest.raises(ValueError):
        execute(tmp_path, mini)


def test_the_crosscheck_set_must_sit_inside_the_main_set_and_the_file_must_have_both_lists(tmp_path, mini):
    _write_picks(tmp_path, ["q1", "q2"], cross=["q3"])
    with pytest.raises(ValueError):
        execute(tmp_path, mini)
    (tmp_path / "out" / "picks.json").write_text(json.dumps({"main": ["q1"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        execute(tmp_path, mini)


def test_a_clean_picks_file_still_runs(tmp_path, mini):
    _write_picks(tmp_path, ["q3", "q2", "q5", "q1"], cross=["q3", "q2", "q5", "q1"])      # 每类 1 题，cross 每类 1 题
    assert len(execute(tmp_path, mini)) == 4


def test_the_picks_are_checked_when_they_are_made_too(tmp_path, mini):
    execute(tmp_path, mini)                                   # 新抽的也过同一道核
    picks = json.loads((tmp_path / "out" / "picks.json").read_text(encoding="utf-8"))
    by = {q["qa_id"]: q["sample_id"] for q in mini.questions}
    assert all(by[i] in ("mini-1", "mini-2") for i in picks["main"] + picks["cross"])


def test_check_picks_alone_refuses_a_locked_conversation_even_if_the_caller_listed_it(mini):
    from eval.ruler import sampling
    for q in mini.questions:
        if q["qa_id"] == "q5":
            q["sample_id"] = "conv-26"
    with pytest.raises(ValueError):
        sampling.check_picks(mini, {"main": ["q5"], "cross": ["q5"]}, ["mini-1", "conv-26"])


def test_check_picks_refuses_the_same_question_twice(mini):
    from eval.ruler import sampling
    with pytest.raises(ValueError):
        sampling.check_picks(mini, {"main": ["q1", "q1"], "cross": ["q1"]}, ["mini-1"])
