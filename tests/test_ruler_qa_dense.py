"""第二步那套答题流程接上按意思找（只用假的向量接口和假的模型接口）：行写进单独的文件、向量调用进花费表、第二次不重算。"""
import json

import pytest

from eval.ruler import chat, corpus, dense, qa_main
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def fake_ask(q, memories):
    return chat.ChatResult(text="a puppy", prompt_tokens=100, completion_tokens=5, model="ans-m")


class JudgeChat:
    def complete(self, prompt):
        return chat.ChatResult(text='{"label": "CORRECT"}', prompt_tokens=100, completion_tokens=3, model="jdg-m")


PR = {"ans-m": (1.0, 2.0), "jdg-m": (1.0, 2.0), "text-embedding-v4": (0.07, 0.0)}


def execute(tmp_path, mini, out="out", **kw):
    args = dict(corpus=mini, out_dir=str(tmp_path / out), who="main", groups=["G2"], subset="main", limit_usd=1.0, ask=fake_ask,
                price_key="ans-m", judge_client=JudgeChat(), judge_price_key="jdg-m", prices=PR, conv_ids=["mini-1", "mini-2"],
                per_category=1, cross_per_category=1)
    args.update(kw)
    return qa_main.execute_run(**args)


def test_a_dense_run_keeps_its_rows_apart_and_prices_the_vector_calls(tmp_path, mini):
    api = FakeEmbedApi()
    rows = execute(tmp_path, mini, variant="dense", dense=dense.DenseConfig(transport=api))
    out = tmp_path / "out"
    assert len(rows) == 4 and (out / "rows_dense.jsonl").exists() and not (out / "rows.jsonl").exists()
    assert all(r["search"] == "dense" for r in rows)
    spend = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    assert spend["tokens"]["text-embedding-v4"]["prompt"] > 0 and spend["tokens"]["text-embedding-v4"]["calls"] == api.calls
    emb = json.loads((out / "embedding_dense.json").read_text(encoding="utf-8"))
    assert emb["c"]["embedding"]["calls"] == api.calls and "vector_store" in emb["c"]


def test_the_keyword_run_is_unchanged_and_has_no_embedding_file(tmp_path, mini):
    rows = execute(tmp_path, mini)
    assert (tmp_path / "out" / "rows.jsonl").exists() and not (tmp_path / "out" / "embedding_dense.json").exists()
    assert all(r["search"] == "keyword" for r in rows)


def test_the_same_data_a_second_time_embeds_nothing(tmp_path, mini):
    cache = str(tmp_path / "vectors.sqlite")
    api1 = FakeEmbedApi()
    execute(tmp_path, mini, out="o1", variant="dense", dense=dense.DenseConfig(transport=api1, cache_path=cache))
    api2 = FakeEmbedApi()
    execute(tmp_path, mini, out="o2", variant="dense", dense=dense.DenseConfig(transport=api2, cache_path=cache))
    assert api1.calls >= 1 and api2.calls == 0


def test_the_spending_limit_covers_the_vector_calls_too(tmp_path, mini):
    api = FakeEmbedApi()
    calls = []

    def ask(q, m):
        calls.append(q)
        return fake_ask(q, m)

    with pytest.raises(Exception):
        execute(tmp_path, mini, variant="dense", ask=ask, limit_usd=1e-12, dense=dense.DenseConfig(transport=api))
    assert calls == []                                   # 向量这一头先超了，一道题也没问模型


def test_dense_without_a_key_stops_and_asks_no_model(tmp_path, mini, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    calls = []
    with pytest.raises(RuntimeError):
        execute(tmp_path, mini, variant="dense", ask=lambda q, m: calls.append(q), dense=dense.DenseConfig(transport=FakeEmbedApi()))
    assert calls == []


def test_the_report_of_a_variant_reads_its_own_rows(tmp_path, mini):
    execute(tmp_path, mini)
    execute(tmp_path, mini, variant="dense", dense=dense.DenseConfig(transport=FakeEmbedApi()))
    rep = qa_main.write_report(str(tmp_path / "out"), variant="dense")
    assert rep["outcomes"]["G2"]["main"]["questions"] == 4
    assert (tmp_path / "out" / "report_dense.json").exists() and (tmp_path / "out" / "report.json").exists() is False


def test_the_embedding_model_has_a_price_in_the_real_table():
    assert qa_main.PRICES["text-embedding-v4"] == (0.07, 0.0)


# ---- 命令行 ----

def test_the_load_command_measures_over_http_and_refuses_locked_conversations(tmp_path, mini):
    root = write_mini_corpus(str(tmp_path / "copy"))
    out = tmp_path / "load"
    assert qa_main.main(["load", "--data", root, "--no-pin", "--out", str(out), "--conv-ids", "mini-1,mini-2", "--search", "keyword",
                         "--workers", "2", "--probe-n", "0"]) == 0
    assert (out / "group_keyword.json").exists()
    with pytest.raises(SystemExit):
        qa_main.main(["load", "--data", root, "--no-pin", "--out", str(tmp_path / "bad"), "--conv-ids", "conv-26", "--search", "keyword"])
    assert not (tmp_path / "bad").exists()


def test_the_load_command_with_dense_needs_a_key(tmp_path, mini, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    root = write_mini_corpus(str(tmp_path / "copy"))
    with pytest.raises(SystemExit):
        qa_main.main(["load", "--data", root, "--no-pin", "--out", str(tmp_path / "o"), "--conv-ids", "mini-1", "--search", "dense"])


def test_the_run_command_knows_the_search_and_vector_options():
    import argparse
    ap_help = qa_main.build_parser().format_help()
    assert "load" in ap_help
    args = qa_main.build_parser().parse_args(["run", "--data", "d", "--out", "o", "--who", "main", "--groups", "G2", "--search", "dense",
                                              "--vector-cache", "v.sqlite", "--embed-cap-tokens", "1000"])
    assert (args.search, args.vector_cache, args.embed_cap_tokens) == ("dense", "v.sqlite", 1000)
    assert qa_main.build_parser().parse_args(["report", "--out", "o", "--variant", "dense"]).variant == "dense"
