"""step 4, part 1: the three gates on rows of several models, and the one-time claim of the locked part for a whole plan
(models, order, gates, who-is-asked-first), resumable only with exactly the same plan. Synthetic rows and a tmp ledger only."""
import json
import os

import pytest

from eval.ruler import split

SEGMENTS = {"s1": 150, "s2": 150, "s3": 150, "s4": 150, "s5": 134}          # 734 questions
LOCK5 = tuple(SEGMENTS)
MODELS = ("main", "hy3", "haiku")


def rows(version, model, correct_by_segment):
    """correct_by_segment: {segment: number correct (the first k questions)} -> the per-question rows of one model, one version."""
    out = []
    for g, n in SEGMENTS.items():
        k = correct_by_segment[g]
        out += [{"item_id": f"{g}#{i:03d}", "group": g, "correct": i < k, "model": model, "version": version} for i in range(n)]
    return out


def flat(k):
    return {g: k for g in SEGMENTS}


def build(base_by_model, dense_by_model):
    base = [r for m, c in base_by_model.items() for r in rows("kw", m, c)]
    dense = [r for m, c in dense_by_model.items() for r in rows("dense", m, c)]
    return base, dense


def decide(base_by_model, dense_by_model, **kw):
    base, dense = build(base_by_model, dense_by_model)
    return split.decide_dense(base, dense, lock=LOCK5, **kw)


# ---- constants ----

def test_the_third_gate_is_written_down_next_to_the_other_two():
    assert split.DENSE_MIN_OVERALL_GAIN_POINTS == 4 and split.DENSE_MIN_SEGMENTS_BETTER == 3 and split.DENSE_MAX_MODEL_LOSS_POINTS == 3


# ---- gate 1: all models pooled, at least 4 percent of all rows ----

def test_pooled_gain_needs_89_of_2202_rows_so_88_is_not_enough_and_89_is():
    base = {m: flat(50) for m in MODELS}
    # 88 more correct answers in total (all in one segment of one model: 134 questions there, so spread over two segments)
    d88 = {"main": {"s1": 50 + 44, "s2": 50 + 44, "s3": 50, "s4": 50, "s5": 50}, "hy3": flat(50), "haiku": flat(50)}
    d89 = {"main": {"s1": 50 + 45, "s2": 50 + 44, "s3": 50, "s4": 50, "s5": 50}, "hy3": flat(50), "haiku": flat(50)}
    r88, r89 = decide(base, d88), decide(base, d89)
    assert (r88["overall_gain_questions"], r89["overall_gain_questions"]) == (88, 89)
    assert r88["questions"] == r89["questions"] == 2202
    assert r88["gate_overall_ok"] is False and r89["gate_overall_ok"] is True


def test_one_model_gaining_a_lot_cannot_hide_inside_the_pool_if_it_is_the_only_one():
    base = {m: flat(50) for m in MODELS}
    dense = {"main": flat(80), "hy3": flat(50), "haiku": flat(50)}                  # +150 questions of 2,202 = 6.8 points pooled
    r = decide(base, dense)
    assert r["gate_overall_ok"] is True and r["models"]["main"]["net"] == 150 and r["models"]["hy3"]["net"] == 0


# ---- gate 2: segments, the three models pooled inside each segment ----

def test_a_segment_counts_as_better_when_the_three_models_together_gain_in_it():
    base = {m: flat(50) for m in MODELS}
    # segment s1: main +3, hy3 -1, haiku -1 -> pooled +1 -> better; s2: pooled 0 -> not better; the others +10 -> better
    dense = {"main": {"s1": 53, "s2": 50, "s3": 60, "s4": 60, "s5": 60},
             "hy3": {"s1": 49, "s2": 50, "s3": 60, "s4": 60, "s5": 60},
             "haiku": {"s1": 49, "s2": 50, "s3": 60, "s4": 60, "s5": 60}}
    r = decide(base, dense)
    assert r["segments"]["s1"]["better"] is True and r["segments"]["s2"]["better"] is False
    assert r["segments_better"] == 4


def test_two_better_segments_are_not_enough_even_when_the_pooled_gain_is_large():
    base = {m: flat(50) for m in MODELS}
    dense = {m: {"s1": 150, "s2": 150, "s3": 50, "s4": 50, "s5": 50} for m in MODELS}
    r = decide(base, dense)
    assert r["gate_overall_ok"] is True and r["segments_better"] == 2 and r["dense_on"] is False


# ---- gate 3: no model may lose more than 3 points: at most 22 of 734 questions ----

def test_a_model_may_lose_22_questions_but_not_23_the_integer_boundary():
    base = {m: flat(100) for m in MODELS}
    big = {m: flat(160) for m in ("hy3", "haiku")}                                  # the other two models gain a lot: gates 1 and 2 pass
    lose22 = dict(big, main={"s1": 100 - 22, "s2": 100, "s3": 100, "s4": 100, "s5": 100})
    lose23 = dict(big, main={"s1": 100 - 23, "s2": 100, "s3": 100, "s4": 100, "s5": 100})
    r22, r23 = decide(base, lose22), decide(base, lose23)
    assert r22["models"]["main"]["net"] == -22 and r23["models"]["main"]["net"] == -23
    assert r22["gate_overall_ok"] and r22["segments_better"] >= 3 and r23["gate_overall_ok"] and r23["segments_better"] >= 3
    assert r22["gate_model_loss_ok"] is True and r22["dense_on"] is True
    assert r23["gate_model_loss_ok"] is False and r23["dense_on"] is False


def test_the_loss_gate_looks_at_each_model_alone_not_at_the_pool():
    base = {m: flat(100) for m in MODELS}
    dense = {"main": {"s1": 70, "s2": 100, "s3": 100, "s4": 100, "s5": 100},       # main -30: over the line
             "hy3": flat(170), "haiku": flat(170)}
    r = decide(base, dense)
    assert r["models"]["main"]["loss_ok"] is False and r["models"]["hy3"]["loss_ok"] is True
    assert r["dense_on"] is False


def test_a_model_that_only_gains_or_stays_flat_never_trips_the_loss_gate():
    base = {m: flat(100) for m in MODELS}
    r = decide(base, {m: flat(110) for m in MODELS})
    assert r["gate_model_loss_ok"] is True and all(v["loss_ok"] for v in r["models"].values())


# ---- the numbers fit together ----

def test_every_models_numbers_add_up_to_the_pooled_numbers_in_every_segment():
    base = {"main": flat(50), "hy3": flat(40), "haiku": flat(60)}
    dense = {"main": flat(58), "hy3": flat(41), "haiku": flat(70)}
    r = decide(base, dense)
    for g, s in r["segments"].items():
        assert s["base_correct"] == sum(s["by_model"][m]["base_correct"] for m in MODELS)
        assert s["dense_correct"] == sum(s["by_model"][m]["dense_correct"] for m in MODELS)
    assert r["overall_gain_questions"] == sum(r["models"][m]["net"] for m in MODELS)
    assert sum(r["models"][m]["dense_correct"] for m in MODELS) - sum(r["models"][m]["base_correct"] for m in MODELS) == r["overall_gain_questions"]


# ---- the old checks still hold with the model dimension ----

def test_the_two_versions_must_hold_the_same_questions_for_every_model():
    base, dense = build({m: flat(50) for m in MODELS}, {m: flat(60) for m in MODELS})
    with pytest.raises(ValueError):
        split.decide_dense(base, [r for r in dense if not (r["model"] == "hy3" and r["item_id"] == "s1#000")], lock=LOCK5)


def test_every_model_must_hold_the_same_questions_as_the_others():
    base, dense = build({m: flat(50) for m in MODELS}, {m: flat(60) for m in MODELS})
    base = [r for r in base if not (r["model"] == "haiku" and r["item_id"] == "s2#000")]
    dense = [r for r in dense if not (r["model"] == "haiku" and r["item_id"] == "s2#000")]
    with pytest.raises(ValueError):
        split.decide_dense(base, dense, lock=LOCK5)


def test_the_same_question_twice_for_one_model_is_refused_and_the_declared_models_must_match():
    base, dense = build({m: flat(50) for m in MODELS}, {m: flat(60) for m in MODELS})
    with pytest.raises(ValueError):
        split.decide_dense(base + [base[0]], dense, lock=LOCK5)
    with pytest.raises(ValueError):
        split.decide_dense(base, dense, lock=LOCK5, models=("main", "hy3"))             # a model that was not declared
    assert split.decide_dense(base, dense, lock=LOCK5, models=MODELS)["dense_on"] is True


def test_rows_without_a_model_are_one_model_as_before():
    base = [{"item_id": f"{g}#{i}", "group": g, "correct": i < 50} for g in LOCK5 for i in range(100)]
    dense = [{"item_id": f"{g}#{i}", "group": g, "correct": i < 60} for g in LOCK5 for i in range(100)]
    r = split.decide_dense(base, dense, lock=LOCK5)
    assert r["dense_on"] is True and r["questions"] == 500 and list(r["models"]) == [None]


# ---- the claim for a whole plan ----

PLAN = {"plan_sha16": "7b8379aca7a46d45", "versions": ["kw", "dense"],
        "models": [{"key": "main", "order": 0, "id": "qwen3.7-flash-2026-07-15"}, {"key": "hy3", "order": 1}, {"key": "haiku", "order": 2}],
        "gates": {"overall_points": 4, "segments_better": 3, "model_loss_points": 3},
        "first_version": {"s1#000": "kw", "s1#001": "dense"}}


def test_the_first_claim_writes_the_plan_its_hash_and_one_used_file_per_version(tmp_path):
    ledger = str(tmp_path / "ledger")
    res = split.claim_locked_models(ledger, PLAN)
    assert res["status"] == "claimed"
    assert sorted(os.listdir(ledger)) == ["claim.json", "claim.sha256", "used.dense", "used.kw"]
    text = open(os.path.join(ledger, "claim.json"), encoding="utf-8").read()
    assert json.loads(text) == PLAN
    assert open(os.path.join(ledger, "claim.sha256"), encoding="utf-8").read().strip() == res["claim_sha256"]


def test_the_same_plan_again_is_a_resume_not_a_new_claim(tmp_path):
    ledger = str(tmp_path / "ledger")
    split.claim_locked_models(ledger, PLAN)
    before = {n: open(os.path.join(ledger, n), encoding="utf-8").read() for n in os.listdir(ledger)}
    again = split.claim_locked_models(ledger, json.loads(json.dumps(PLAN)))
    assert again["status"] == "resumed"
    assert {n: open(os.path.join(ledger, n), encoding="utf-8").read() for n in os.listdir(ledger)} == before


@pytest.mark.parametrize("change", ["add_model", "swap_order", "other_gate", "other_first", "other_model_id", "other_versions"])
def test_any_change_of_the_plan_after_the_claim_is_refused(tmp_path, change):
    ledger = str(tmp_path / "ledger")
    split.claim_locked_models(ledger, PLAN)
    other = json.loads(json.dumps(PLAN))
    if change == "add_model":
        other["models"].append({"key": "mimo", "order": 3})
    elif change == "swap_order":
        other["models"][0]["order"], other["models"][1]["order"] = 1, 0
    elif change == "other_gate":
        other["gates"]["model_loss_points"] = 2
    elif change == "other_first":
        other["first_version"]["s1#000"] = "dense"
    elif change == "other_model_id":
        other["models"][0]["id"] = "qwen3.7-flash"
    else:
        other["versions"] = ["kw", "dense_rerank"]
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked_models(ledger, other)


def test_a_version_already_used_by_an_old_style_claim_refuses_a_new_plan(tmp_path):
    ledger = str(tmp_path / "ledger")
    split.claim_locked(ledger, "kw", ["kw", "dense"])               # the step-1 claim of one version
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked_models(ledger, PLAN)


def test_a_ledger_that_is_not_a_directory_is_refused(tmp_path):
    f = tmp_path / "ledger"
    f.write_text("x")
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked_models(str(f), PLAN)


def test_a_plan_must_name_two_distinct_plain_versions_and_at_least_one_model(tmp_path):
    for bad in ({**PLAN, "versions": ["kw"]}, {**PLAN, "versions": ["kw", "KW"]}, {**PLAN, "versions": ["kw", "../x"]}, {**PLAN, "models": []},
                {k: v for k, v in PLAN.items() if k != "versions"}):
        with pytest.raises(ValueError):
            split.claim_locked_models(str(tmp_path / "l"), bad)
    assert not os.path.exists(str(tmp_path / "l")) or os.listdir(str(tmp_path / "l")) == []     # nothing written for a bad plan


def test_the_claim_is_made_once_even_when_two_processes_try_at_the_same_moment(tmp_path):
    ledger = str(tmp_path / "ledger")
    results = []
    import threading

    def go():
        try:
            results.append(split.claim_locked_models(ledger, PLAN)["status"])
        except Exception as e:                                       # noqa
            results.append(type(e).__name__)
    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("claimed") == 1 and all(r in ("claimed", "resumed") for r in results), results


def test_the_run_lock_is_exclusive_and_must_be_released_by_the_run(tmp_path):
    ledger = str(tmp_path / "ledger")
    os.makedirs(ledger)
    lock = split.acquire_run_lock(ledger)
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.acquire_run_lock(ledger)
    split.release_run_lock(lock)
    split.release_run_lock(split.acquire_run_lock(ledger))            # free again after the run let go
