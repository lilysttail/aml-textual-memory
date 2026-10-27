"""第二步的抽题、取记忆、跑题、记结果、出表（全部用合成小题库 + 假接口，不联网）。"""
import json

import pytest

from eval.ruler import answerer, chat, corpus, cost, judge, qa_report, qa_run, sampling, target
from eval.ruler.testing import write_mini_corpus


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


# ---- 抽题：固定、可重复 ----

def test_the_pick_is_fixed_by_the_question_ids_and_does_not_depend_on_the_order_they_come_in(mini):
    a = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    mini.questions.reverse()
    b = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    assert a == b
    assert len(a["main"]) == 4 and {q.split("#")[0] for q in a["main"]} <= {"q1", "q2", "q3", "q4", "q5"}


def test_one_per_category_and_the_crosscheck_set_is_inside_the_main_set(mini):
    p = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    cat = {q["qa_id"]: str(q["category"]) for q in mini.questions}
    assert sorted(cat[i] for i in p["main"]) == ["1", "2", "3", "4"]
    assert set(p["cross"]) <= set(p["main"]) and len(p["cross"]) == 4


def test_only_the_named_conversations_are_used(mini):
    p = sampling.pick_questions(mini, ["mini-2"], per_category=1, cross_per_category=1, categories=("3", "4"))
    assert sorted(p["main"]) == ["q4", "q5"]
    with pytest.raises(ValueError):                       # 第 1 类在 mini-2 里一道都没有
        sampling.pick_questions(mini, ["mini-2"], per_category=1, cross_per_category=1)


def test_a_question_without_evidence_or_gold_answer_is_never_picked(mini):
    for q in mini.questions:
        if q["qa_id"] == "q3":
            q["evidence"] = []
    with pytest.raises(ValueError):
        sampling.pick_questions(mini, ["mini-1"], per_category=1, cross_per_category=1, categories=("1",))


def test_asking_for_more_than_there_is_is_an_error_not_a_shorter_list(mini):
    with pytest.raises(ValueError):
        sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=2, cross_per_category=1)


def test_the_groups_are_the_five_in_the_dispatch():
    g = sampling.GROUPS
    assert [(k, v.shape, v.mode, v.limit) for k, v in g.items()] == [
        ("G1", "a", "hits", 100), ("G2", "c", "hits", 100), ("G3", "c", "hits", 20),
        ("G4", "a", "evidence", None), ("G5", "c", "evidence", None)]


# ---- 取记忆 ----

def test_the_contexts_hold_what_the_service_hands_back_and_the_evidence_lines(mini):
    ctx = sampling.build_contexts(mini, ["q1", "q4"], shape="c", top_k=100)
    c = ctx["q1"]
    assert c.hits and c.n_gold == 1 and c.found_100 in (0, 1)
    assert len(c.evidence_lines) == 1 and c.evidence_lines[0].startswith("[2")
    assert all(isinstance(h.content, str) for h in c.hits)
    assert ctx["q4"].n_gold == 1


def test_found_in_the_first_20_is_never_more_than_found_in_100(mini):
    ctx = sampling.build_contexts(mini, [q["qa_id"] for q in mini.questions], shape="a", top_k=100)
    assert all(c.found_20 <= c.found_100 <= c.n_gold for c in ctx.values())


def test_the_evidence_lines_use_the_shape_of_the_group(mini):
    a = sampling.build_contexts(mini, ["q1"], shape="a")["q1"].evidence_lines[0]
    c = sampling.build_contexts(mini, ["q1"], shape="c")["q1"].evidence_lines[0]
    assert a.startswith("[user] ") and not a.startswith("[2")
    assert c.startswith("[2") and "Alice: " in c


def test_the_stored_line_is_what_the_service_really_stores(mini):
    from eval.ruler import unanswerable
    for shape in ("a", "b", "c", "d"):
        conv = mini.conversations[0]
        assert unanswerable.stored_texts_rule(conv, shape) == unanswerable.stored_texts_service(conv, shape)


def test_memories_for_a_group_are_the_first_n_hits_or_the_evidence_lines(mini):
    ctx = sampling.build_contexts(mini, ["q3"], shape="c")["q3"]
    assert sampling.memories_for(sampling.GROUPS["G2"], ctx) == [h.content for h in ctx.hits[:100]]
    assert sampling.memories_for(sampling.GROUPS["G3"], ctx) == [h.content for h in ctx.hits[:20]]
    assert sampling.memories_for(sampling.GROUPS["G5"], ctx) == ctx.evidence_lines


# ---- 错在哪一环 ----

@pytest.mark.parametrize("group,f_prompt,f100,f20,n_gold,expect", [
    ("G2", 0, 0, 0, 1, "not_returned"),
    ("G2", 1, 1, 1, 1, "evidence_in_prompt"),
    ("G3", 0, 1, 0, 1, "returned_not_in_prompt"),
    ("G3", 0, 0, 0, 1, "not_returned"),
    ("G3", 1, 1, 1, 2, "evidence_in_prompt"),
    ("G4", 1, 0, 0, 1, "evidence_in_prompt"),
    ("G5", 1, 0, 0, 1, "evidence_in_prompt"),
])
def test_where_the_evidence_was_lost(group, f_prompt, f100, f20, n_gold, expect):
    c = sampling.Context(qa_id="q", hits=[], evidence_lines=[], n_gold=n_gold, found_100=f100, found_20=f20)
    assert qa_run.stage_of(sampling.GROUPS[group], c) == expect


# ---- 跑题 ----

class FakeAnswerer:
    def __init__(self, replies):
        self.replies = dict(replies)           # question text -> answer text
        self.calls = []

    def __call__(self, question, memories):
        self.calls.append((question, list(memories)))
        return chat.ChatResult(text=self.replies.get(question, "Paris"), prompt_tokens=50, completion_tokens=3, model="ans-m", attempts=1)


class FakeJudgeChat:
    def __init__(self, label="CORRECT"):
        self.label, self.calls = label, 0

    def complete(self, prompt):
        self.calls += 1
        return chat.ChatResult(text='{"label": "%s"}' % self.label, prompt_tokens=200, completion_tokens=4, model="jdg-m")


def make_runner(tmp_path, mini, label="CORRECT", limit=1.0, qa_ids=("q1", "q3")):
    meter = cost.CostMeter({"ans-m": (1.0, 2.0), "jdg-m": (0.5, 1.0)}, limit_usd=limit)
    jc = FakeJudgeChat(label)
    ctx = {s: sampling.build_contexts(mini, list(qa_ids), shape=s) for s in ("a", "c")}
    r = qa_run.Runner(corpus=mini, contexts=ctx, meter=meter, judge=judge.RefinedJudge(jc), judge_price_key="jdg-m",
                      results_path=str(tmp_path / "rows.jsonl"))
    return r, meter, jc


def test_a_correct_answer_is_judged_and_the_usage_goes_into_the_meter(tmp_path, mini):
    r, meter, jc = make_runner(tmp_path, mini)
    fa = FakeAnswerer({"What did Alice adopt?": "a puppy and a bit more"})
    rows = r.run_model("main", fa, "ans-m", ["G2"], ["q1"])
    row = rows[0]
    assert (row["outcome"], row["score"], row["group"], row["model_key"], row["qa_id"], row["category"]) == ("correct", 1, "G2", "main", "q1", "4")
    assert (row["prompt_tokens"], row["completion_tokens"], row["judge_prompt_tokens"], row["judge_calls"]) == (50, 3, 200, 1)
    assert meter.tokens["ans-m"]["calls"] == 1 and meter.tokens["jdg-m"]["calls"] == 1
    assert fa.calls[0][1] == sampling.memories_for(sampling.GROUPS["G2"], r.contexts["c"]["q1"])


def test_an_exact_gold_answer_costs_no_judge_call(tmp_path, mini):
    r, meter, jc = make_runner(tmp_path, mini)
    rows = r.run_model("main", FakeAnswerer({"What did Alice adopt?": "a puppy"}), "ans-m", ["G2"], ["q1"])
    assert rows[0]["outcome"] == "correct" and rows[0]["shortcut"] == "exact" and jc.calls == 0


def test_i_dont_know_is_not_found_and_is_not_sent_to_the_judge(tmp_path, mini):
    r, meter, jc = make_runner(tmp_path, mini)
    rows = r.run_model("main", FakeAnswerer({"What did Alice adopt?": "I don't know"}), "ans-m", ["G2"], ["q1"])
    assert (rows[0]["outcome"], rows[0]["score"]) == ("not_found", None) and jc.calls == 0


def test_a_wrong_answer_that_is_not_i_dont_know_is_made_up(tmp_path, mini):
    r, meter, jc = make_runner(tmp_path, mini, label="WRONG")
    rows = r.run_model("main", FakeAnswerer({"What did Alice adopt?": "a cat"}), "ans-m", ["G2"], ["q1"])
    assert (rows[0]["outcome"], rows[0]["score"]) == ("made_up", 0)


def test_every_group_asks_with_its_own_memories(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)
    fa = FakeAnswerer({})
    r.run_model("main", fa, "ans-m", ["G1", "G2", "G3", "G4", "G5"], ["q1"])
    assert len(fa.calls) == 5
    g4, g5 = fa.calls[3][1], fa.calls[4][1]
    assert g4 == r.contexts["a"]["q1"].evidence_lines and g5 == r.contexts["c"]["q1"].evidence_lines and g4 != g5


def test_finished_rows_are_not_asked_again_after_a_restart(tmp_path, mini):
    r, meter, _ = make_runner(tmp_path, mini)
    fa = FakeAnswerer({})
    r.run_model("main", fa, "ans-m", ["G2"], ["q1", "q3"])
    r2, meter2, _ = make_runner(tmp_path, mini)
    fa2 = FakeAnswerer({})
    rows = r2.run_model("main", fa2, "ans-m", ["G2"], ["q1", "q3"])
    assert fa2.calls == [] and len(rows) == 2 and meter2.tokens == {}
    assert len((tmp_path / "rows.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2


def test_the_spending_limit_stops_the_run_and_what_was_done_is_kept(tmp_path, mini):
    r, meter, _ = make_runner(tmp_path, mini, limit=0.00001)
    with pytest.raises(cost.BudgetExceeded):
        r.run_model("main", FakeAnswerer({}), "ans-m", ["G2"], ["q1", "q3"])
    kept = (tmp_path / "rows.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(kept) == 1                                  # 第一题做完了（超限是做完后才发现的），第二题没开始


def test_a_failed_model_call_is_a_row_of_its_own_kind_not_a_wrong_answer(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)

    def boom(question, memories):
        raise chat.ChatError("HTTP 500", status=500, code="server_error")

    rows = r.run_model("main", boom, "ans-m", ["G2"], ["q1"])
    assert (rows[0]["outcome"], rows[0]["error_status"]) == ("api_error", 500) and rows[0]["answer"] == ""


def test_an_answer_with_tool_calls_is_void(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)

    def tooly(question, memories):
        return chat.ChatResult(text="Paris", prompt_tokens=5, completion_tokens=1, model="ans-m", extra={"tool_calls": 2})

    rows = r.run_model("haiku", tooly, "ans-m", ["G3"], ["q1"])
    assert rows[0]["outcome"] == "void" and rows[0]["tool_calls"] == 2


def test_the_judge_that_cannot_be_read_is_judge_error_not_wrong(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)
    r.judge = judge.RefinedJudge(type("C", (), {"complete": lambda self, p: chat.ChatResult("garbage", 1, 1, "jdg-m")})(), parse_retries=1)
    rows = r.run_model("main", FakeAnswerer({"What did Alice adopt?": "a cat"}), "ans-m", ["G2"], ["q1"])
    assert rows[0]["outcome"] == "judge_error"


def test_rows_hold_ids_and_numbers_and_the_local_answer_but_never_the_question_or_the_memories(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)
    r.run_model("main", FakeAnswerer({}), "ans-m", ["G2"], ["q1"])
    raw = (tmp_path / "rows.jsonl").read_text(encoding="utf-8")
    assert "Alice adopt" not in raw and "lighthouse" not in raw
    row = json.loads(raw.splitlines()[0])
    assert "question" not in row and "memories" not in row


# ---- 出表：只有数字和题号 ----

def row(group, model, qa, cat, outcome, stage="evidence_in_prompt", found=1, n_gold=1, **kw):
    return dict({"group": group, "model_key": model, "qa_id": qa, "category": cat, "outcome": outcome, "stage": stage,
                 "found_in_prompt": found, "n_gold": n_gold, "prompt_tokens": 10, "completion_tokens": 2,
                 "judge_prompt_tokens": 20, "judge_completion_tokens": 3, "judge_calls": 1}, **kw)


ROWS = [
    row("G2", "main", "q1", "1", "correct"), row("G2", "main", "q2", "2", "made_up"),
    row("G2", "main", "q3", "3", "not_found", stage="not_returned", found=0),
    row("G2", "main", "q4", "4", "correct", stage="not_returned", found=0),
    row("G3", "main", "q1", "1", "correct"), row("G3", "main", "q2", "2", "correct"),
    row("G3", "main", "q3", "3", "made_up", stage="returned_not_in_prompt", found=0),
    row("G3", "main", "q4", "4", "void", tool_calls=1),
]


def test_the_outcome_table_counts_per_group_and_model_and_leaves_out_void_rows():
    rep = qa_report.build_report(ROWS)
    t = rep["outcomes"]["G2"]["main"]
    assert t == {"questions": 4, "correct": 2, "not_found": 1, "made_up": 1, "other": 0}
    assert rep["outcomes"]["G3"]["main"]["questions"] == 3 and rep["void"]["G3"]["main"] == ["q4"]


def test_the_table_by_category():
    rep = qa_report.build_report(ROWS)
    assert rep["by_category"]["G2"]["main"]["2"] == {"questions": 1, "correct": 0, "not_found": 0, "made_up": 1, "other": 0}


def test_where_the_wrong_ones_were_lost_and_the_right_ones_without_evidence():
    rep = qa_report.build_report(ROWS)
    s = rep["where_lost"]["G2"]["main"]
    assert s == {"wrong_not_returned": 1, "wrong_returned_not_in_prompt": 0, "wrong_evidence_in_prompt": 1, "correct_without_evidence": 1}
    assert rep["where_lost"]["G3"]["main"]["wrong_returned_not_in_prompt"] == 1


def test_g2_against_g3_per_model_lists_the_flips_by_id():
    f = qa_report.flip_table(ROWS, "G2", "G3", "main")
    assert f["both_correct"] == ["q1"] and f["only_first_correct"] == [] and f["only_second_correct"] == ["q2"]
    assert f["neither_correct"] == ["q3"] and f["compared"] == 3                 # q4 在 G3 里作废，不比


def test_agreement_of_two_judgements_lists_the_ids_that_differ():
    ours = {"q1": 1, "q2": 0, "q3": 1}
    official = {"q1": 1, "q2": 1, "q3": 1}
    a = qa_report.agreement(ours, official)
    assert a == {"compared": 3, "same": 2, "differ": 1, "differing_ids": ["q2"], "only_ours": [], "only_official": []}
    assert qa_report.agreement({"q1": 1}, {"q2": 1})["only_ours"] == ["q1"]


def test_the_report_has_no_text_in_it():
    rep = qa_report.build_report(ROWS)
    flat = json.dumps(rep)
    assert "answer" not in flat.replace("answerer", "")


# ---- 判分一致性：抽样、喂给官方脚本的文件、读官方的分 ----

def test_the_consistency_sample_is_fixed_and_comes_from_the_asked_group_and_model():
    from eval.ruler import consistency
    rows = [row("G2", "main", f"q{i}", "1", "correct") for i in range(40)] + [row("G3", "main", f"x{i}", "1", "correct") for i in range(5)]
    a = consistency.pick_sample(rows, "main", "G2", n=30)
    assert a == consistency.pick_sample(list(reversed(rows)), "main", "G2", n=30)
    assert len(a) == 30 and len(set(a)) == 30 and all(i.startswith("q") for i in a)
    assert consistency.pick_sample(rows[:3], "main", "G2", n=30) == sorted(r["qa_id"] for r in rows[:3])      # 不够就全取


def test_the_files_for_the_official_script_are_its_own_formats(tmp_path, mini):
    import json as _json
    from eval.ruler import consistency
    rows = [{"group": "G2", "model_key": "main", "qa_id": "q1", "answer": "a puppy"}, {"group": "G2", "model_key": "main", "qa_id": "q2", "answer": "I don't know"}]
    consistency.write_official_inputs(mini, rows, ["q1", "q2"], str(tmp_path))
    qs = [_json.loads(l) for l in (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    ps = [_json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [q["qa_id"] for q in qs] == ["q1", "q2"] and qs[0]["answer"] == ["a puppy", "Biscuit"] and "question" in qs[0]
    assert ps == [{"qa_id": "q1", "predicted_answer": "a puppy"}, {"qa_id": "q2", "predicted_answer": "I don't know"}]


def test_the_official_scores_are_read_from_its_scored_file(tmp_path):
    from eval.ruler import consistency
    p = tmp_path / "scored.jsonl"
    p.write_text('{"qa_id": "q1", "llm_score": 1.0}\n{"qa_id": "q2", "llm_score": 0.0}\n{"qa_id": "q3"}\n', encoding="utf-8")
    assert consistency.read_official_scores(str(p)) == {"q1": 1, "q2": 0}          # 没有分的不算进去，也不当 0


def test_our_scores_judge_every_answer_even_i_dont_know(mini):
    from eval.ruler import consistency
    rows = [{"group": "G2", "model_key": "main", "qa_id": "q1", "answer": "I don't know"}]
    jc = FakeJudgeChat("WRONG")
    scores, results = consistency.our_scores(judge.RefinedJudge(jc), mini, rows, ["q1"])
    assert scores == {"q1": 0} and jc.calls == len(mini.questions[0]["answer"]) and len(results) == jc.calls


def test_a_row_that_ended_in_an_error_is_asked_again_and_the_later_row_replaces_it(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)

    def boom(question, memories):
        raise chat.ChatError("HTTP 500", status=500)

    first = r.run_model("main", boom, "ans-m", ["G2"], ["q1"])
    assert first[0]["outcome"] == "api_error"
    again = r.run_model("main", FakeAnswerer({}), "ans-m", ["G2"], ["q1"])
    assert again[0]["outcome"] != "api_error"
    from eval.ruler import qa_main
    kept = qa_main.latest_rows(str(tmp_path / "rows.jsonl"))
    assert len(kept) == 1 and kept[0]["outcome"] != "api_error"


def test_a_finished_judge_error_row_is_also_asked_again(tmp_path, mini):
    r, _, _ = make_runner(tmp_path, mini)
    bad = type("C", (), {"complete": lambda self, p: chat.ChatResult("garbage", 1, 1, "jdg-m")})()
    r.judge = judge.RefinedJudge(bad, parse_retries=0)
    assert r.run_model("main", FakeAnswerer({"What did Alice adopt?": "a cat"}), "ans-m", ["G2"], ["q1"])[0]["outcome"] == "judge_error"
    r2, _, _ = make_runner(tmp_path, mini)
    assert r2.run_model("main", FakeAnswerer({"What did Alice adopt?": "a cat"}), "ans-m", ["G2"], ["q1"])[0]["outcome"] == "correct"


def test_the_cross_check_set_is_the_first_picks_of_each_category_in_pick_order(mini):
    one = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=1, cross_per_category=1)
    assert one["cross"] == one["main"]                    # 每类只抽 1 题时，交叉核就是那 1 题
    two = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=1, cross_per_category=1, categories=("4",))
    assert two["cross"] == two["main"][:1]


def test_the_first_20_are_cut_from_the_same_ordered_list_not_a_new_search():
    hits = [target.Hit(id=f"r#{i}", content=f"[x] m{i}") for i in range(30)]
    ctx = sampling.Context(qa_id="q", hits=hits, evidence_lines=["e1"], n_gold=1, found_100=1, found_20=1)
    assert sampling.memories_for(sampling.GROUPS["G3"], ctx) == [f"[x] m{i}" for i in range(20)]
    assert len(sampling.memories_for(sampling.GROUPS["G2"], ctx)) == 30


def test_the_crosscheck_takes_the_first_of_several_picks_not_the_last(mini):
    p = sampling.pick_questions(mini, ["mini-1", "mini-2"], per_category=2, cross_per_category=1, categories=("4",))
    assert len(p["main"]) == 2 and p["cross"] == [p["main"][0]]
