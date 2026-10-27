"""measure_dense 的整库量法并进尺子：HTTP + 并发 + 封顶 + 数向量接口调用次数。合成小题库 + 假向量接口，不联网。"""
import json

import pytest

from eval.ruler import corpus, dense, load_measure
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def test_a_keyword_measurement_over_http_writes_asks_and_scores(tmp_path, mini):
    r = load_measure.measure(mini, str(tmp_path / "out"), shape="a", workers=3, probe_n=0)
    assert r["label"] == "keyword" and r["embedding"] is None
    assert r["questions_asked"] == 5 and r["questions_answered_ok"] == 5 and r["search_errors"] == {} and r["add_errors"] == {}
    assert r["blocks_added"] == 3 and r["health_status_at_start"] == 200 and r["answers_over_top_k"] == 0
    assert 0.0 <= r["evidence_hit_rate"] <= 1.0 and r["evidence_hits"] == round(r["evidence_hit_rate"] * 5)
    assert set(r["by_category"]) == {"1", "2", "3", "4"}
    assert r["search_latency_s"] and r["add_latency_s"] and r["protocol"].startswith("cold")


def test_the_hit_rate_is_the_same_as_the_in_process_ruler_gives(tmp_path, mini):
    from eval.ruler import run
    root = write_mini_corpus(str(tmp_path / "copy2"))
    a = run.run(root, str(tmp_path / "o"), shape="a", neighbours=1, require_pin=False)
    b = load_measure.measure(mini, str(tmp_path / "out"), shape="a", workers=2, probe_n=0)
    assert b["evidence_hit_rate"] == a["hit_rate"]


def test_a_dense_measurement_counts_the_provider_calls_by_phase_and_prices_the_tokens(tmp_path, mini):
    api = FakeEmbedApi()
    r = load_measure.measure(mini, str(tmp_path / "out"), shape="c", dense_cfg=dense.DenseConfig(transport=api), workers=2, probe_n=3)
    e = r["embedding"]
    assert r["label"] == "dense" and e["calls"] == api.calls and e["tokens_used"] == e["tokens_by_replies"] > 0
    assert e["cost_usd"] == pytest.approx(e["tokens_used"] * 0.07 / 1e6, abs=1e-6)
    ph = r["embedding_calls_by_phase"]
    assert ph["during_writes"]["http_calls"] >= 1 and ph["during_questions"]["embed_invocations"] >= 1
    assert ph["per_question_asked"]["embed_invocations"] > 0
    probe = r["sequential_probe"]
    assert probe["searches_summary"]["n"] == 3 and probe["adds_summary"]["n"] >= 1
    assert set(probe["searches_summary"]["status"]) == {200}


def test_the_token_cap_stops_the_run_locally_and_says_so(tmp_path, mini):
    api = FakeEmbedApi()
    r = load_measure.measure(mini, str(tmp_path / "out"), shape="c", dense_cfg=dense.DenseConfig(transport=api, cap_tokens=3), workers=1, probe_n=0)
    assert r["stopped_at_token_cap"] is True and r["token_cap"] == 3
    assert api.calls <= 1 and r["embedding"]["refused_after_cap"] >= 1
    assert r["questions_skipped_after_cap"] + r["questions_answered_ok"] + sum(r["search_errors"].values()) == r["questions_asked"]
    assert r["questions_skipped_after_cap"] == r["questions_asked"]            # 写的时候就到顶了：一道题也没发给服务


def test_warm_means_one_search_per_conversation_first(tmp_path, mini):
    r = load_measure.measure(mini, str(tmp_path / "out"), shape="a", workers=2, warmup=True, probe_n=0)
    assert r["warmup"]["conversations"] == 2 and r["warmup"]["errors"] == {} and r["protocol"].startswith("warm")


def test_only_the_named_conversations_are_used(tmp_path, mini):
    r = load_measure.measure(mini, str(tmp_path / "out"), shape="a", conv_ids=["mini-2"], probe_n=0)
    assert r["questions_asked"] == 2 and r["blocks_added"] == 1


def test_the_result_files_hold_numbers_and_ids_only(tmp_path, mini):
    load_measure.measure(mini, str(tmp_path / "out"), shape="c", dense_cfg=dense.DenseConfig(transport=FakeEmbedApi()), probe_n=1)
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == ["group_dense_cold.json", "rows_dense_cold.jsonl"]
    text = "".join((tmp_path / "out" / n).read_text(encoding="utf-8") for n in names)
    for word in ("puppy", "lighthouse", "pottery", "Alice", FAKE_KEY):
        assert word not in text
    rows = [json.loads(l) for l in (tmp_path / "out" / "rows_dense_cold.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {set(r) == {"qa_id", "category", "code", "hit", "n", "chars", "dt"} for r in rows} == {True}


def test_a_second_measurement_with_the_same_vector_file_embeds_nothing(tmp_path, mini):
    cache = str(tmp_path / "vectors.sqlite")
    api1 = FakeEmbedApi()
    load_measure.measure(mini, str(tmp_path / "o1"), shape="c", dense_cfg=dense.DenseConfig(transport=api1, cache_path=cache), workers=1, probe_n=0)
    api2 = FakeEmbedApi()
    r2 = load_measure.measure(mini, str(tmp_path / "o2"), shape="c", dense_cfg=dense.DenseConfig(transport=api2, cache_path=cache), workers=1, probe_n=0)
    assert api1.calls >= 1 and api2.calls == 0 and r2["embedding"]["tokens_used"] == 0
    assert r2["vector_store"]["misses"] == 0 and r2["vector_store"]["hits"] > 0


def test_a_search_whose_meaning_path_was_skipped_by_the_embed_gate_is_counted():
    import logging
    c = load_measure.SkipCounter()
    log = logging.getLogger("core.recall")
    log.addHandler(c)
    try:
        log.warning("dense query vector skipped: embed concurrency gate unavailable")
        log.warning("something else")
    finally:
        log.removeHandler(c)
    assert c.count == 1


def test_the_result_says_how_many_searches_ran_without_their_meaning_path(tmp_path, mini):
    r = load_measure.measure(mini, str(tmp_path / "out"), shape="c", dense_cfg=dense.DenseConfig(transport=FakeEmbedApi()), workers=1, probe_n=0)
    assert r["dense_query_skipped"] == 0
    k = load_measure.measure(mini, str(tmp_path / "out2"), shape="a", workers=1, probe_n=0)
    assert k["dense_query_skipped"] is None                   # 关键词那种没有这回事
