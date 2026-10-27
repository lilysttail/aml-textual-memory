"""step 4, part 4: the check before deploying a "dense on" version — under many simultaneous searches every question REALLY used the
meaning-based path, the answer is the same as the serial one, and what is on is not a keyword search in disguise. Synthetic corpus, fake
embedder, tuning-style conversations only (the locked ones are refused)."""
import json
import os

import pytest

from eval.ruler import corpus, gate_check, load_measure, qa_main
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)
    # The search path's embed gate never waits: a search that finds no free place drops its meaning-based half. With the default 2
    # places and 4 workers, a busy machine made a concurrent question skip now and then, and the counts below were off by one. The
    # places are given enough here (the deployed image does the same), so these tests do not depend on timing.
    monkeypatch.setenv("EMBED_MAX_CONCURRENCY", "64")


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


CONVS = ("mini-1", "mini-2")


class DownApi(FakeEmbedApi):
    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        return 500, b"{}"


def run(mini, tmp_path, transport=None, measure_fn=None, **kw):
    args = dict(corpus=mini, out_dir=str(tmp_path / "gate"), vector_cache=str(tmp_path / "vec.sqlite"), conv_ids=CONVS, expect_questions=None,
                workers=4, transport=transport or FakeEmbedApi())
    if measure_fn is not None:
        args["measure_fn"] = measure_fn
    args.update(kw)
    return gate_check.run_gate_check(**args)


def test_a_healthy_dense_service_passes_every_condition_and_the_two_baselines_differ(mini, tmp_path):
    rep = run(mini, tmp_path)
    assert rep["passed"] is True, json.dumps(rep["conditions"])
    n = rep["questions"]
    assert n == len([q for q in mini.questions if q["sample_id"] in CONVS])
    assert all(rep["conditions"].values()) and len(rep["conditions"]) == 6
    assert rep["used_dense"] == n and rep["skipped_dense"] == 0 and rep["http_503"] == 0 and rep["identical_to_dense_baseline"] == n
    assert rep["config_readback"]["recalls"]["dense"] is True and rep["config_readback"]["embed_provider"] == "dashscope"
    assert rep["baselines_differ_questions"] >= 1 and rep["query_vectors"]["expected"] == rep["query_vectors"]["seen"]


def test_the_two_baselines_and_the_concurrent_run_are_kept_as_files_of_ids(mini, tmp_path):
    run(mini, tmp_path)
    d = str(tmp_path / "gate")
    for name in ("gate_baseline_dense.jsonl", "gate_baseline_keyword.jsonl", "gate_concurrent_dense.jsonl", "gate_report.json"):
        assert os.path.exists(os.path.join(d, name)), name
    rows = [json.loads(l) for l in open(os.path.join(d, "gate_baseline_dense.jsonl"), encoding="utf-8") if l.strip()]
    assert all(isinstance(r["ids"], list) and r["ids"] for r in rows)


def test_a_dense_that_silently_became_keyword_fails_used_dense_and_the_baselines_are_the_same(mini, tmp_path):
    rep = run(mini, tmp_path, transport=DownApi())
    assert rep["passed"] is False
    assert rep["conditions"]["used_dense_every_question"] is False and rep["conditions"]["baselines_not_identical"] is False
    assert rep["used_dense"] == 0 and rep["baselines_differ_questions"] == 0


def test_a_concurrent_answer_that_differs_from_the_dense_baseline_fails(mini, tmp_path):
    def tamper(corpus_, out_dir, **kw):
        res = load_measure.measure(corpus_, out_dir, **kw)
        if kw.get("workers", 1) > 1 and kw.get("dense_cfg") is not None:
            p = os.path.join(out_dir, "rows_dense_cold" + kw["tag_suffix"] + ".jsonl")
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            rows[0]["ids"] = list(reversed(rows[0]["ids"])) if len(rows[0]["ids"]) > 1 else rows[0]["ids"] + ["x"]
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
        return res
    rep = run(mini, tmp_path, measure_fn=tamper)
    assert rep["passed"] is False and rep["conditions"]["concurrent_equals_dense_baseline"] is False and rep["identical_to_dense_baseline"] == rep["questions"] - 1


def test_a_503_in_the_concurrent_run_fails(mini, tmp_path):
    def with_503(corpus_, out_dir, **kw):
        res = load_measure.measure(corpus_, out_dir, **kw)
        if kw.get("workers", 1) > 1 and kw.get("dense_cfg") is not None:
            res["search_errors"] = {"503": 2}
        return res
    rep = run(mini, tmp_path, measure_fn=with_503)
    assert rep["passed"] is False and rep["conditions"]["no_503"] is False and rep["http_503"] == 2


def test_a_service_whose_readback_says_dense_is_off_fails_even_if_the_counts_look_right(mini, tmp_path):
    def lying(corpus_, out_dir, **kw):
        res = load_measure.measure(corpus_, out_dir, **kw)
        if kw.get("dense_cfg") is not None:
            res["config_readback"] = dict(res["config_readback"], recalls={"keyword": True, "dense": False})
        return res
    rep = run(mini, tmp_path, measure_fn=lying)
    assert rep["passed"] is False and rep["conditions"]["config_readback_dense_on"] is False


def test_a_provider_that_is_not_the_one_the_rules_fix_fails_the_readback(mini, tmp_path):
    def other(corpus_, out_dir, **kw):
        res = load_measure.measure(corpus_, out_dir, **kw)
        if kw.get("dense_cfg") is not None:
            res["config_readback"] = dict(res["config_readback"], embed_provider="local")
        return res
    rep = run(mini, tmp_path, measure_fn=other)
    assert rep["passed"] is False and rep["conditions"]["config_readback_dense_on"] is False


def test_the_locked_conversations_and_a_wrong_question_count_are_refused_before_anything_runs(mini, tmp_path):
    def boom(*a, **k):
        raise AssertionError("must not run")
    with pytest.raises(ValueError):
        run(mini, tmp_path, measure_fn=boom, conv_ids=("conv-26",))
    with pytest.raises(ValueError):
        run(mini, tmp_path, measure_fn=boom, expect_questions=10_000)
    assert not os.path.exists(str(tmp_path / "gate"))


def test_the_command_exists_and_needs_the_vector_file():
    ap = qa_main.build_parser()
    ap.parse_args(["gate-check", "--data", "d", "--out", "o", "--vector-cache", "v"])
    with pytest.raises(SystemExit):
        ap.parse_args(["gate-check", "--data", "d", "--out", "o"])


def test_the_command_refuses_a_corpus_that_is_not_the_tuning_part_and_writes_nothing(tmp_path):
    data = write_mini_corpus(str(tmp_path / "copy3"))
    out = str(tmp_path / "gate_out")
    with pytest.raises(SystemExit) as e:
        qa_main.main(["gate-check", "--data", data, "--no-pin", "--out", out, "--vector-cache", str(tmp_path / "v.sqlite")])
    assert "refused" in str(e.value) and not os.path.exists(out)


def test_query_vectors_that_do_not_match_the_distinct_question_texts_fail(mini, tmp_path):
    def extra(corpus_, out_dir, **kw):
        res = load_measure.measure(corpus_, out_dir, **kw)
        if kw.get("workers", 1) > 1 and kw.get("dense_cfg") is not None:
            res["embedding_calls_by_phase"]["during_questions"]["embed_texts"] += 3
        return res
    rep = run(mini, tmp_path, measure_fn=extra)
    assert rep["passed"] is False and rep["conditions"]["query_vectors_explained"] is False
    assert rep["query_vectors"]["seen"] == rep["query_vectors"]["expected"] + 3
