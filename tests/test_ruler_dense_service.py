"""尺子的服务路径能开按意思找：开关、向量接口的计数 / 花费 / 封顶、固定的本地向量库。向量接口是假的，钥匙是假的，不联网。"""
import os

import pytest

from eval.ruler import corpus, cost, dense, shapes, target
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus

@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def fill(t, mini, sample="mini-1", shape="c"):
    for conv in mini.conversations:
        if conv.sample_id != sample:
            continue
        for b in corpus.chunk_conversation(conv):
            assert t.add(b.user_id, b.session_id, b.request_id, [shapes.render(shape, m, s) for m, s in b.pairs()]) == 200


def test_dense_search_finds_a_record_the_words_cannot(tmp_path, mini):
    """问句里只有“wedding”，记忆里只有“married”：按字找不到，按意思找（假向量里这两个词同格）找得到。"""
    conv = mini.conversations[0]
    conv.sessions[0].messages[0] = type(conv.sessions[0].messages[0])(**{**conv.sessions[0].messages[0].__dict__, "text": "We got married yesterday"})
    user = corpus.user_id_of("mini-1")
    kw = target.ServiceTarget(str(tmp_path / "k.sqlite"), neighbours=False)
    fill(kw, mini)
    assert not any("married" in h.content for h in kw.search(user, "wedding ceremony", 5))
    kw.close()
    api = FakeEmbedApi()
    d = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=api))
    fill(d, mini)
    assert any("married" in h.content for h in d.search(user, "wedding ceremony", 5))
    d.close()


def test_the_calls_and_tokens_are_counted_and_go_into_the_spending_table(tmp_path, mini):
    api = FakeEmbedApi()
    cm = cost.CostMeter({"text-embedding-v4": (0.07, 0.0)}, limit_usd=1.0)
    cfg = dense.DenseConfig(transport=api, cost_meter=cm)
    d = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=cfg)
    fill(d, mini)
    d.search(corpus.user_id_of("mini-1"), "wedding ceremony", 5)
    snap = d.embed_meter.snapshot()
    assert snap["calls"] == api.calls >= 1 and snap["tokens_by_replies"] > 0
    assert cm.tokens["text-embedding-v4"]["prompt"] == snap["tokens_by_replies"] and cm.spent == pytest.approx(snap["tokens_by_replies"] * 0.07 / 1e6)
    d.close()


def test_the_second_run_over_the_same_data_asks_the_provider_for_nothing(tmp_path, mini):
    cache = str(tmp_path / "vectors.sqlite")
    api1 = FakeEmbedApi()
    d1 = target.ServiceTarget(str(tmp_path / "d1.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=api1, cache_path=cache))
    fill(d1, mini)
    d1.search(corpus.user_id_of("mini-1"), "wedding ceremony", 5)
    d1.close()
    assert api1.calls >= 1
    api2 = FakeEmbedApi()
    d2 = target.ServiceTarget(str(tmp_path / "d2.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=api2, cache_path=cache))
    fill(d2, mini)                                                    # 全新的服务库，同一份数据、同一个问题
    out = d2.search(corpus.user_id_of("mini-1"), "wedding ceremony", 5)
    assert out and api2.calls == 0 and d2.vector_stats()["hits"] > 0 and d2.vector_stats()["misses"] == 0
    d2.close()


def test_a_cap_stops_the_provider_calls_locally(tmp_path, mini):
    api = FakeEmbedApi()
    d = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=api, cap_tokens=1))
    try:
        fill(d, mini)
    except Exception:
        pass
    assert api.calls <= 1 and d.embed_meter.snapshot()["refused_after_cap"] >= 1
    d.close()


def test_the_default_target_never_needs_or_sees_a_key(tmp_path, mini, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    t = target.ServiceTarget(str(tmp_path / "k.sqlite"), neighbours=False)
    fill(t, mini)
    assert t.search(corpus.user_id_of("mini-1"), "puppy", 5) is not None
    assert t.embed_meter is None
    t.close()


def test_dense_without_a_key_stops_before_building_anything(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    with pytest.raises(RuntimeError):
        target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=FakeEmbedApi()))


def test_the_process_environment_and_the_app_module_are_left_as_they_were(tmp_path, mini):
    import app.main as m
    before = {k: os.environ.get(k) for k in ("RECALL_PATHS", "EMBED_PROVIDER", "RERANK_ENABLED", "DB_PATH", "API_KEY")}
    builder = m._build_embedder
    d = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=FakeEmbedApi()))
    d.close()
    assert {k: os.environ.get(k) for k in before} == before and m._build_embedder is builder


def test_a_keyword_target_built_after_a_dense_one_is_keyword_only_again(tmp_path, mini):
    d = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=FakeEmbedApi()))
    d.close()
    k = target.ServiceTarget(str(tmp_path / "k.sqlite"), neighbours=False)
    assert k.embed_meter is None and tuple(k._app.state.core.config.paths) == ("keyword",)
    k.close()


def test_the_dense_switches_are_the_ones_in_the_dispatch(tmp_path):
    cfg = dense.DenseConfig(transport=FakeEmbedApi())
    assert cfg.switches() == {"RECALL_PATHS": "keyword,dense", "RERANK_ENABLED": "0", "EMBED_PROVIDER": "dashscope"}
    assert dense.DenseConfig(transport=FakeEmbedApi(), rerank=True).switches()["RERANK_ENABLED"] == "1"


def test_the_keyword_path_never_has_the_key_in_its_environment_even_when_the_process_has_one():
    assert os.environ["DASHSCOPE_API_KEY"] == FAKE_KEY
    with dense.service_env("x.sqlite", False, None):
        assert "DASHSCOPE_API_KEY" not in os.environ
    assert os.environ["DASHSCOPE_API_KEY"] == FAKE_KEY                 # 出来以后原样放回
