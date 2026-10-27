"""step 4, part 3: the locked-part mode — claim first, retrieval kept in files, then per model and per question the two versions
are asked one after the other and judged one after the other, the first one being fixed by the question id's hash; resumable only with the
same plan; a pair is written only when whole. Fake models, fake judge, a synthetic corpus whose two conversations stand in for the locked
ones, tmp directories as the ledger: no real ledger, no network, no key, no money."""
import json
import os

import pytest

from eval.ruler import chat, corpus, cost, locked_run, qa_main, sampling, split, target
from eval.ruler.testing import write_mini_corpus

LOCK = ("mini-1", "mini-2")
WHOS = ["main", "hy3", "haiku"]
DECLARED = ["kw", "dense"]


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def qids(mini):
    return [q["qa_id"] for q in mini.questions if q["sample_id"] in LOCK]


class Env:
    """Everything a run needs, with fakes, and a log of what happened in which order."""

    def __init__(self, mini, tmp_path):
        self.mini = mini
        self.out = str(tmp_path / "out")
        self.ledger = str(tmp_path / "ledger")
        self.events = []
        self.fail_asks = {}                    # (who, version, qa_id) -> how many times that ask raises ChatError
        self.budget_stop_at = None             # raise BudgetExceeded when the n-th ask of the whole run is made
        self.asks = 0
        self.dense_flags = {}                  # qa_id -> dense_used for the dense retrieval
        self.judge_says = lambda prompt: "CORRECT" if "ans-DENSE" in prompt else "WRONG"

    # the retrieval: synthetic memories that name their version
    def search_contexts(self, corpus_, ids, version, dense_cfg):
        out = {}
        for i in ids:
            tag = "KW" if version == "kw" else "DENSE"
            hits = [target.Hit(id=f"{i}-{k}", content=f"{tag}-MEM-{k}") for k in range(3)]
            out[i] = sampling.Context(qa_id=i, hits=hits, evidence_lines=[], n_gold=1, found_100=1 if version == "dense" else 0,
                                      found_20=1 if version == "dense" else 0,
                                      dense_used=(None if version == "kw" else self.dense_flags.get(i, True)))
        return out

    def make_ask(self, who, session_id=None, workdir=None):
        env = self

        def ask(question, memories):
            env.asks += 1
            if env.budget_stop_at is not None and env.asks == env.budget_stop_at:
                raise cost.BudgetExceeded("stopped for the test")
            version = "kw" if memories and memories[0].startswith("KW") else "dense"
            qid = next(q["qa_id"] for q in env.mini.questions if q["question"] == question)
            env.events.append((who, "ask", version, qid))
            key = (who, version, qid)
            if env.fail_asks.get(key, 0) > 0:
                env.fail_asks[key] -= 1
                raise chat.ChatError("HTTP 500", status=500)
            return chat.ChatResult(text=f"ans-{'KW' if version == 'kw' else 'DENSE'}", prompt_tokens=100, completion_tokens=5,
                                   model=f"{who}-returned", finish_reason="stop")
        price = {"main": qa_main.ANSWER_MODEL, "hy3": "hy3", "haiku": qa_main.HAIKU_MODEL}[who]
        return ask, price

    @property
    def judge_client(self):
        env = self

        class J:
            def complete(self, prompt):
                version = "DENSE" if "ans-DENSE" in prompt else "KW"
                env.events.append(("judge", version))
                return chat.ChatResult(text='{"label": "%s"}' % env.judge_says(prompt), prompt_tokens=50, completion_tokens=3, model="j")
        return J()

    def identity(self, who, prior=None, session_id=None):
        d = {"key": who, "provider": "fake", "id": f"{who}-id", "pinned": who == "main"}
        if who == "hy3":
            d["session_header"] = (prior or {}).get("session_header") or "aml-ruler-test-session"
        return d

    def run(self, **kw):
        args = dict(corpus=self.mini, out_dir=self.out, ledger_dir=self.ledger, whos=list(WHOS), declared=list(DECLARED), candidate="dense",
                    limit_usd=10.0, lock=LOCK, expect_questions=None, make_ask=self.make_ask, judge_client=self.judge_client,
                    dense_cfg_factory=lambda meter: None, identity_for=self.identity)
        args.update(kw)
        return locked_run.run_locked(**args)


@pytest.fixture()
def env(mini, tmp_path, monkeypatch):
    e = Env(mini, tmp_path)
    monkeypatch.setattr(locked_run, "search_contexts", e.search_contexts)
    return e


def rows_of(env, who):
    path = os.path.join(env.out, f"rows_locked_{who}.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


# ---- the arguments ----

def test_first_version_is_fixed_by_the_hash_of_the_question_id_even_odd():
    for i in ("conv-26#q0001", "conv-41#q0007", "q5", "q1", "q2", "q3"):
        even = int(sampling._rank(i), 16) % 2 == 0
        assert locked_run.first_version(i) == ("kw" if even else "dense")
    assert {locked_run.first_version(f"x#{k}") for k in range(40)} == {"kw", "dense"}


def test_the_command_refuses_a_run_that_lacks_any_of_the_four():
    ap = qa_main.build_parser()
    base = ["run-locked", "--data", "d", "--out", "o", "--version", "dense", "--declared", "kw,dense", "--ledger", "l", "--models", "main,hy3,haiku"]
    ap.parse_args(base)
    for flag in ("--version", "--declared", "--ledger", "--models"):
        args = list(base)
        i = args.index(flag)
        del args[i:i + 2]
        with pytest.raises(SystemExit):
            ap.parse_args(args)


@pytest.mark.parametrize("bad", [dict(whos=[]), dict(whos=["main", "main"]), dict(whos=["main", "gpt-9"]), dict(declared=["kw"]),
                                 dict(declared=["kw", "kw"]), dict(declared=["kw", "../x"]), dict(candidate="dense_rerank"), dict(limit_usd=0)])
def test_bad_arguments_are_refused_before_anything_is_written(env, bad):
    with pytest.raises((ValueError, SystemExit)):
        env.run(**bad)
    assert not os.path.exists(env.out) and not os.path.exists(env.ledger) and env.events == []


def test_the_locked_questions_must_be_the_expected_number(env):
    with pytest.raises(ValueError):
        env.run(expect_questions=len(qids(env.mini)) + 1)
    assert not os.path.exists(env.ledger) and env.events == []


# ---- the claim comes first ----

def test_the_claim_exists_before_the_first_retrieval_and_before_anything_else_is_written(env, monkeypatch):
    seen = {}
    real = env.search_contexts

    def spy(corpus_, ids, version, dense_cfg):
        seen.setdefault("claim_there", os.path.exists(os.path.join(env.ledger, "claim.json")))
        seen.setdefault("out_files", sorted(os.listdir(env.out)) if os.path.isdir(env.out) else [])
        return real(corpus_, ids, version, dense_cfg)
    monkeypatch.setattr(locked_run, "search_contexts", spy)
    env.run()
    assert seen["claim_there"] is True and "rows_locked_main.jsonl" not in seen["out_files"]


def test_the_claim_holds_the_whole_plan(env):
    env.run()
    plan = json.load(open(os.path.join(env.ledger, "claim.json"), encoding="utf-8"))
    assert plan["plan_sha16"] == locked_run.PLAN_SHA16 and plan["versions"] == DECLARED and plan["candidate"] == "dense"
    assert [m["key"] for m in plan["models"]] == WHOS and [m["order"] for m in plan["models"]] == [0, 1, 2]
    assert plan["gates"] == {"min_overall_gain_points": 4, "min_segments_better": 3, "max_model_loss_points": 3}
    assert plan["partition"]["lock"] == list(LOCK) and plan["partition"]["questions"] == len(qids(env.mini))
    assert set(plan["first_version"]) == set(qids(env.mini)) and all(plan["first_version"][i] == locked_run.first_version(i) for i in plan["first_version"])
    assert plan["models"][1]["session_header"] == "aml-ruler-test-session" and "judge" in plan and "answer_prompt_sha256" in plan


def test_a_changed_plan_after_the_claim_is_refused_and_nothing_is_asked(env):
    env.run(whos=["main"], one_model=True)
    n_events = len(env.events)
    with pytest.raises(split.LockedSetAlreadyUsed):
        env.run(whos=["main", "hy3"])                       # a model added
    with pytest.raises(split.LockedSetAlreadyUsed):
        env.run(whos=["hy3", "main"])                       # another order
    assert len(env.events) == n_events


# ---- pairs, interleaved ----

def test_each_question_is_asked_v1_asked_v2_judged_v1_judged_v2_with_v1_fixed_by_the_hash(env):
    env.run(whos=["main"])
    seq = [e for e in env.events if e[0] in ("main", "judge")]
    # a question with several gold answers is judged once per gold answer (until one says CORRECT): collapse repeats of the same judged version
    collapsed = []
    for e in seq:
        item = ("ask", e[2], e[3]) if e[1] == "ask" else ("judge", e[1])
        if not collapsed or collapsed[-1] != item:
            collapsed.append(item)
    asks = [c for c in collapsed if c[0] == "ask"]
    assert len(asks) == 2 * len(qids(env.mini))
    pos = 0
    for qid in qids(env.mini):
        first = locked_run.first_version(qid)
        second = "dense" if first == "kw" else "kw"
        tag = lambda v: "KW" if v == "kw" else "DENSE"
        assert collapsed[pos:pos + 2] == [("ask", first, qid), ("ask", second, qid)]
        j = []
        pos += 2
        while pos < len(collapsed) and collapsed[pos][0] == "judge":
            j.append(collapsed[pos][1])
            pos += 1
        assert j[:2] == [tag(first), tag(second)] and set(j) <= {"KW", "DENSE"}      # judging follows both asks; first version judged first
    assert pos == len(collapsed)


def test_both_rows_of_a_pair_are_written_together_with_the_pairing_fields(env):
    env.run(whos=["main"])
    rows = rows_of(env, "main")
    ids = qids(env.mini)
    assert len(rows) == 2 * len(ids)
    claim_sha = open(os.path.join(env.ledger, "claim.sha256"), encoding="utf-8").read().strip()
    for k, qid in enumerate(ids):
        pair = rows[2 * k:2 * k + 2]
        assert {r["version"] for r in pair} == {"kw", "dense"} and all(r["qa_id"] == qid and r["pair_index"] == k for r in pair)
        assert [r["pair_pos"] for r in pair] == [0, 1] and pair[0]["version"] == locked_run.first_version(qid)
        for r in pair:
            assert r["model_key"] == "main" and r["claim_sha"] == claim_sha and r["segment"] in LOCK and r["asked_at"] <= r["judged_at"]
            assert r["outcome"] in ("correct", "made_up") and r["model_returned"] == "main-returned"
    assert all(r["outcome"] == ("correct" if r["version"] == "dense" else "made_up") for r in rows)


def test_the_three_models_run_one_after_the_other_in_the_claimed_order_each_in_its_own_file(env):
    env.run()
    order = [e[0] for e in env.events if e[0] in WHOS]
    n = 2 * len(qids(env.mini))
    assert order == ["main"] * n + ["hy3"] * n + ["haiku"] * n
    assert all(len(rows_of(env, w)) == n for w in WHOS)


def test_one_model_at_a_time_runs_only_the_next_unfinished_model(env):
    env.run(one_model=True)
    assert len(rows_of(env, "main")) > 0 and rows_of(env, "hy3") == []
    env.run(one_model=True)
    assert len(rows_of(env, "hy3")) > 0 and rows_of(env, "haiku") == []
    env.run(one_model=True)
    assert len(rows_of(env, "haiku")) > 0


# ---- resume, errors, the spending stop ----

def test_a_spending_stop_in_the_middle_is_resumed_on_the_same_claim_and_nothing_is_asked_twice(env):
    ids = qids(env.mini)
    env.budget_stop_at = 5                                    # the 5th ask of the run: two whole pairs done, the third pair stopped
    with pytest.raises(cost.BudgetExceeded):
        env.run(whos=["main"])
    assert len(rows_of(env, "main")) == 4                    # only whole pairs were written
    asks_before = sum(1 for e in env.events if e[0] == "main" and e[1] == "ask")
    env.budget_stop_at = None
    res = env.run(whos=["main"])
    assert res["claim"] == "resumed" and res["complete"] is True
    rows = rows_of(env, "main")
    assert len(rows) == 2 * len(ids) and sorted({r["qa_id"] for r in rows}) == sorted(ids)
    assert len({(r["qa_id"], r["version"]) for r in rows}) == len(rows)                  # no pair twice
    asks_after = sum(1 for e in env.events if e[0] == "main" and e[1] == "ask")
    assert asks_after - asks_before == 2 * (len(ids) - 2)                                # only the pairs that were not whole


def test_a_pair_with_an_error_is_not_written_and_is_asked_again_as_a_whole_pair_after_the_pass(env):
    ids = qids(env.mini)
    bad = ids[1]
    first = locked_run.first_version(bad)
    env.fail_asks[("main", "dense" if first == "kw" else "kw", bad)] = 1           # the SECOND ask of that pair fails once
    env.run(whos=["main"])
    rows = rows_of(env, "main")
    assert len(rows) == 2 * len(ids) and not any(r["outcome"] == "api_error" for r in rows)
    # the pair of `bad` is the last one written (retry after the pass) and was asked both versions again, in order
    pair = [r for r in rows if r["qa_id"] == bad]
    assert [r["pair_pos"] for r in pair] == [0, 1] and rows[-2:] == pair
    asks = [e for e in env.events if e[0] == "main" and e[1] == "ask" and e[3] == bad]
    assert [a[2] for a in asks] == [first, "dense" if first == "kw" else "kw", first, "dense" if first == "kw" else "kw"]


def test_a_pair_that_keeps_failing_stops_the_run_unfinished_and_the_next_model_is_not_started(env):
    ids = qids(env.mini)
    bad = ids[0]
    env.fail_asks[("main", "kw", bad)] = 99
    with pytest.raises(locked_run.LockedRunIncomplete) as e:
        env.run()
    assert bad in str(e.value)
    assert rows_of(env, "hy3") == [] and not any(ev[0] == "hy3" for ev in env.events)
    assert {r["qa_id"] for r in rows_of(env, "main")} == set(ids) - {bad}
    prog = json.load(open(os.path.join(env.out, "progress.json"), encoding="utf-8"))
    assert prog["models"]["main"]["incomplete"] == [bad] and prog["models"]["main"].get("ended") is None


def test_a_judge_error_makes_the_pair_incomplete_too(env, monkeypatch):
    ids = qids(env.mini)
    calls = {"n": 0}
    real = env.judge_client.complete

    class Flaky:
        def complete(self, prompt):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise chat.ChatError("HTTP 500", status=500)
            return env.judge_client.complete(prompt)
    res = env.run(whos=["main"], judge_client=Flaky())
    rows = rows_of(env, "main")
    assert res["complete"] is True and len(rows) == 2 * len(ids) and all(r["outcome"] in ("correct", "made_up") for r in rows)


def test_a_complete_model_is_not_asked_again_on_a_resume(env):
    env.run(whos=["main"])
    n = len(env.events)
    res = env.run(whos=["main"])
    assert res["claim"] == "resumed" and len(env.events) == n


def test_the_run_lock_is_held_during_the_run_and_released_after_even_after_a_stop(env):
    ids = qids(env.mini)
    env.budget_stop_at = 2
    with pytest.raises(cost.BudgetExceeded):
        env.run(whos=["main"])
    assert not os.path.exists(os.path.join(env.ledger, "run.lock"))
    os.makedirs(env.ledger, exist_ok=True)
    held = split.acquire_run_lock(env.ledger)
    env.budget_stop_at = None
    with pytest.raises(split.LockedSetAlreadyUsed):
        env.run(whos=["main"])
    split.release_run_lock(held)
    assert env.run(whos=["main"])["complete"] is True and len(ids) > 0


def test_a_retrieval_file_changed_after_it_was_made_is_refused_on_resume(env):
    env.run(whos=["main"], one_model=True)
    path = os.path.join(env.out, "retrieval_dense.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write("{}\n")
    with pytest.raises(ValueError):
        env.run(whos=["main"])


# ---- the spending ----

def test_every_ask_and_judge_goes_through_the_one_spending_table_and_the_limit_stops_the_run(env):
    res = env.run(whos=["main"], limit_usd=10.0)
    spend = json.load(open(os.path.join(env.out, "spend.json"), encoding="utf-8"))
    n = 2 * len(qids(env.mini))
    judge_calls = sum(r["judge_calls"] for r in rows_of(env, "main"))
    assert spend["tokens"][qa_main.ANSWER_MODEL]["calls"] == n and spend["tokens"][qa_main.JUDGE_MODEL]["calls"] == judge_calls >= n and res["complete"]


def test_a_tiny_limit_stops_with_budget_exceeded_and_leaves_no_half_pair(env):
    with pytest.raises(cost.BudgetExceeded):
        env.run(whos=["main"], limit_usd=0.0000001)
    assert len(rows_of(env, "main")) % 2 == 0


# ---- the dense use ----

def test_the_dense_retrieval_records_per_question_whether_the_dense_path_ran(env):
    ids = qids(env.mini)
    env.dense_flags[ids[0]] = False
    env.run(whos=["main"], one_model=True)
    lines = [json.loads(l) for l in open(os.path.join(env.out, "retrieval_dense.jsonl"), encoding="utf-8") if l.strip()]
    assert {l["qa_id"]: l["dense_used"] for l in lines}[ids[0]] is False and all(l["dense_used"] is True for l in lines if l["qa_id"] != ids[0])
    kw = [json.loads(l) for l in open(os.path.join(env.out, "retrieval_kw.jsonl"), encoding="utf-8") if l.strip()]
    assert all("dense_used" in l and l["dense_used"] is None for l in kw) and all(len(l["memories"]) == 3 for l in kw)


# ---- the report ----

def test_the_report_gives_each_model_alone_and_the_decision_only_when_all_models_are_whole(env):
    env.run(one_model=True)
    part = locked_run.locked_report(env.out, env.ledger)
    assert part["complete_models"] == ["main"] and part["decision"] is None
    assert part["models"]["main"]["kw"]["correct"] + part["models"]["main"]["kw"]["made_up"] + part["models"]["main"]["kw"]["not_found"] == len(qids(env.mini))
    env.run()
    full = locked_run.locked_report(env.out, env.ledger)
    assert full["complete_models"] == WHOS and full["decision"] is not None
    rows = []
    for w in WHOS:
        for r in rows_of(env, w):
            rows.append({"item_id": r["qa_id"], "group": r["segment"], "correct": r["outcome"] == "correct", "model": w, "version": r["version"]})
    base = [r for r in rows if r["version"] == "kw"]
    dense = [r for r in rows if r["version"] == "dense"]
    want = split.decide_dense(base, dense, lock=LOCK, models=WHOS)
    assert full["decision"]["dense_on"] == want["dense_on"] and full["decision"]["overall_gain_questions"] == want["overall_gain_questions"]
    assert full["decision"]["models"] == want["models"]


def test_the_report_has_per_model_per_segment_numbers_that_add_up_and_the_dense_use_counts(env):
    env.dense_flags[qids(env.mini)[0]] = False
    env.run()
    rep = locked_run.locked_report(env.out, env.ledger)
    d = rep["decision"]
    for g, s in d["segments"].items():
        assert s["dense_correct"] == sum(m["dense_correct"] for m in s["by_model"].values())
    assert rep["dense_use"] == {"questions": len(qids(env.mini)), "used_dense": len(qids(env.mini)) - 1, "skipped_dense": 1}
    text = json.dumps(rep, ensure_ascii=False)
    assert "ans-" not in text and "MEM" not in text                    # numbers and ids only: no answer text, no memory text


# ---- the real retrieval through the service (fake embedder), the real identities, the command line ----

def test_the_retrieval_files_come_from_the_real_service_search_and_say_whether_dense_ran(mini, tmp_path, monkeypatch):
    from eval.ruler import dense
    from eval.ruler.testing import FAKE_KEY, FakeEmbedApi
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)
    e = Env(mini, tmp_path)
    api = FakeEmbedApi()
    res = e.run(whos=["main"], dense_cfg_factory=lambda meter: dense.DenseConfig(transport=api, cost_meter=meter))
    assert res["complete"] is True and api.calls > 0
    kw = [json.loads(l) for l in open(os.path.join(e.out, "retrieval_kw.jsonl"), encoding="utf-8") if l.strip()]
    dn = [json.loads(l) for l in open(os.path.join(e.out, "retrieval_dense.jsonl"), encoding="utf-8") if l.strip()]
    assert [r["qa_id"] for r in kw] == [r["qa_id"] for r in dn] == qids(mini)
    assert all(r["dense_used"] is None for r in kw) and all(r["dense_used"] is True for r in dn)
    assert all(isinstance(m, str) and m for r in kw for m in r["memories"])


def test_the_default_identities_say_what_is_pinned_and_what_is_only_an_alias(monkeypatch):
    monkeypatch.setattr(locked_run, "_claude_version", lambda: "2.1.308 (Claude Code)")
    main = locked_run.model_identity("main")
    assert main["pinned"] is True and main["id"] == qa_main.ANSWER_MODEL and main["provider"] == "dashscope"
    hy3 = locked_run.model_identity("hy3")
    assert hy3["pinned"] is False and hy3["alias"] == "hy3" and hy3["gateway"] == "https://opencode.ai/zen/go/v1"
    assert hy3["session_header"].startswith("aml-ruler-")
    again = locked_run.model_identity("hy3", prior=hy3)                      # a resume keeps the session header of the claim
    assert again["session_header"] == hy3["session_header"] and locked_run.model_identity("hy3")["session_header"] != hy3["session_header"]
    haiku = locked_run.model_identity("haiku")
    assert haiku["cli_version"] == "2.1.308 (Claude Code)" and haiku["pinned"] == "half" and haiku["model_arg"] == qa_main.HAIKU_MODEL
    with pytest.raises(ValueError):
        locked_run.model_identity("kimi")


def test_a_run_with_a_cli_version_that_changed_since_the_claim_is_refused(env, monkeypatch):
    monkeypatch.setattr(locked_run, "_claude_version", lambda: "1.0")
    env.run(whos=["haiku"], identity_for=locked_run.model_identity, one_model=True)
    monkeypatch.setattr(locked_run, "_claude_version", lambda: "2.0")
    with pytest.raises(split.LockedSetAlreadyUsed):
        env.run(whos=["haiku"], identity_for=locked_run.model_identity)


def test_the_command_refuses_the_synthetic_corpus_because_it_is_not_the_locked_part_and_writes_nothing(mini, tmp_path):
    data = write_mini_corpus(str(tmp_path / "copy2"))
    out, ledger = str(tmp_path / "o"), str(tmp_path / "l")
    with pytest.raises(SystemExit) as e:
        qa_main.main(["run-locked", "--data", data, "--no-pin", "--out", out, "--version", "dense", "--declared", "kw,dense", "--ledger", ledger,
                      "--models", "main,hy3,haiku"])
    assert "refused" in str(e.value) and not os.path.exists(out) and not os.path.exists(ledger)


def test_report_locked_prints_and_saves_the_numbers(env, capsys):
    env.run()
    assert qa_main.main(["report-locked", "--out", env.out, "--ledger", env.ledger]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.load(open(os.path.join(env.out, "report_locked.json"), encoding="utf-8"))
    assert printed == saved and printed["complete_models"] == WHOS
    with pytest.raises(SystemExit):
        qa_main.main(["report-locked", "--out", env.out, "--ledger", env.ledger + "_none"])


def test_the_second_step_still_refuses_the_locked_conversations():
    with pytest.raises(ValueError):
        qa_main.execute_run(None, "x", "main", ["G2"], "main", 1.0, lambda q, m: None, "m", None, conv_ids=["conv-26"])


def test_a_pair_is_whole_only_when_both_versions_are_on_disk_with_a_real_outcome(tmp_path):
    path = str(tmp_path / "rows.jsonl")
    lines = [{"qa_id": "a", "version": "kw", "outcome": "correct"}, {"qa_id": "a", "version": "dense", "outcome": "made_up"},       # whole
             {"qa_id": "b", "version": "kw", "outcome": "correct"},                                                                   # torn: one version only
             {"qa_id": "c", "version": "kw", "outcome": "correct"}, {"qa_id": "c", "version": "dense", "outcome": "api_error"},     # an error row
             {"qa_id": "d", "version": "kw", "outcome": "judge_error"}, {"qa_id": "d", "version": "dense", "outcome": "correct"},
             {"qa_id": "e", "version": "kw", "outcome": "void"}, {"qa_id": "e", "version": "dense", "outcome": "correct"}]
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in lines))
    assert locked_run._whole_pairs(path, ["a", "b", "c", "d", "e", "f"]) == {"a"}


def test_the_version_check_of_the_claude_program_also_runs_with_the_auto_updater_off(monkeypatch):
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, " 2.1.308 (Claude Code)\n", ""

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env")
        return Done()
    monkeypatch.setattr(locked_run.subprocess, "run", fake_run)
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "0")
    assert locked_run._claude_version() == "2.1.308 (Claude Code)"
    assert seen["cmd"][-1] == "--version" and seen["env"] is not None and seen["env"]["DISABLE_AUTOUPDATER"] == "1"
