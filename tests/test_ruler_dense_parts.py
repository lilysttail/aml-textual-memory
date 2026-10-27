"""第三步（二）准备的零件：向量的本地固定库、向量接口的计数和封顶（从 measure_dense 并进来）。假接口，不联网。"""
import json
import threading

import pytest

from core.embedder import EmbedUnavailable
from eval.ruler import cost, embed_meter, vector_cache

class FakeInner:
    """顶替 embedder：向量由文本决定（同一文本同一向量），记下每次收到什么。"""
    provider = "dashscope"
    model_name = "text-embedding-v4:8"
    dim = 8

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.tokens_used = 0

    def embed(self, texts, timeout_s=None):
        self.calls.append(list(texts))
        if self.fail:
            raise EmbedUnavailable("down")
        return [[float(len(t)), float(sum(map(ord, t)) % 97)] + [0.0] * 6 for t in texts]


# ---- 本地固定库 ----

def test_the_disk_cache_keeps_vectors_between_runs_and_keys_them_by_model_and_text(tmp_path):
    p = str(tmp_path / "vectors.sqlite")
    c = vector_cache.DiskVectorCache(p)
    c.put_many("m:8", {"hello": [0.1, 0.2, 0.30000000000000004]})
    assert c.get_many("m:8", ["hello", "nope"]) == {"hello": [0.1, 0.2, 0.30000000000000004]}      # 浮点原样回来
    assert c.get_many("other:8", ["hello"]) == {}                                                   # 换模型 = 不命中
    c.close()
    again = vector_cache.DiskVectorCache(p)
    assert again.get_many("m:8", ["hello"]) == {"hello": [0.1, 0.2, 0.30000000000000004]}
    assert again.count() == 1


def test_the_cache_file_holds_hashes_and_numbers_not_text(tmp_path):
    p = tmp_path / "vectors.sqlite"
    c = vector_cache.DiskVectorCache(str(p))
    c.put_many("m:8", {"SECRET-CORPUS-SENTENCE": [1.0, 2.0]})
    c.close()
    assert b"SECRET-CORPUS-SENTENCE" not in p.read_bytes()


# ---- 带缓存的 embedder ----

def test_the_caching_embedder_asks_the_inner_one_only_for_what_it_has_not_seen(tmp_path):
    inner = FakeInner()
    e = vector_cache.CachingEmbedder(inner, vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite")))
    out = e.embed(["a", "bb", "a"])
    assert inner.calls == [["a", "bb"]] and len(out) == 3 and out[0] == out[2] and out[0] != out[1]
    e.embed(["a", "bb"])
    assert len(inner.calls) == 1                                          # 全命中：一次都不问
    e.embed(["a", "ccc"])
    assert inner.calls[-1] == ["ccc"]
    assert e.stats() == {"hits": 4, "misses": 3, "inner_calls": 2}


def test_the_second_run_computes_nothing(tmp_path):
    path = str(tmp_path / "v.sqlite")
    texts = ["x1", "x2", "x3"]
    first = vector_cache.CachingEmbedder(FakeInner(), vector_cache.DiskVectorCache(path)).embed(texts)
    inner2 = FakeInner()
    second = vector_cache.CachingEmbedder(inner2, vector_cache.DiskVectorCache(path)).embed(texts)
    assert inner2.calls == [] and second == first


def test_the_caching_embedder_looks_like_the_embedder_it_wraps(tmp_path):
    e = vector_cache.CachingEmbedder(FakeInner(), vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite")))
    assert (e.provider, e.model_name, e.dim) == ("dashscope", "text-embedding-v4:8", 8)
    assert e.embed([]) == []


def test_a_failing_inner_embedder_caches_nothing_and_the_error_goes_on(tmp_path):
    cache = vector_cache.DiskVectorCache(str(tmp_path / "v.sqlite"))
    e = vector_cache.CachingEmbedder(FakeInner(fail=True), cache)
    with pytest.raises(EmbedUnavailable):
        e.embed(["a"])
    assert cache.count() == 0


# ---- 向量接口的计数和封顶 ----

def reply(tokens, n=1):
    return 200, json.dumps({"usage": {"total_tokens": tokens}, "data": [{"index": i, "embedding": [1.0]} for i in range(n)]}).encode("utf-8")


class Real:
    def __init__(self, *replies, raises=None):
        self.replies, self.raises, self.calls = list(replies), raises, 0

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.replies.pop(0)


def test_the_meter_counts_calls_tokens_and_statuses_and_hands_the_reply_on_unchanged():
    real = Real(reply(8), (429, b"{}"))
    m = embed_meter.EmbedMeter(real, cap_tokens=1000)
    body = b'{"model": "m", "input": ["abcdefgh"]}'                 # 上界 10 token
    assert m.transport("u", {}, body, 5) == reply(8)
    assert m.transport("u", {}, body, 5) == (429, b"{}")
    s = m.snapshot()
    assert (s["calls"], s["tokens_by_replies"], s["http_status"]) == (2, 8, {"200": 1, "429": 1})


def test_the_cap_refuses_locally_and_never_touches_the_network():
    real = Real(reply(60), reply(60))
    m = embed_meter.EmbedMeter(real, cap_tokens=50)
    m.transport("u", {}, b"", 5)
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, b"", 5)
    assert real.calls == 1 and m.cap_reached.is_set() and m.snapshot()["refused_after_cap"] == 1


def test_transport_exceptions_are_counted_by_type_and_raised_again():
    m = embed_meter.EmbedMeter(Real(raises=TimeoutError()), cap_tokens=10)
    with pytest.raises(TimeoutError):
        m.transport("u", {}, b"", 5)
    assert m.snapshot()["transport_exceptions"] == {"TimeoutError": 1}


def test_the_embedder_invocations_and_batch_sizes_are_counted():
    inner = FakeInner()
    m = embed_meter.EmbedMeter(Real(), cap_tokens=10)
    e = m.count_invocations(inner)
    e.embed(["a", "b", "c"])
    e.embed(["d"])
    s = m.snapshot()
    assert (s["embed_invocations"], s["embed_texts"], s["embed_invocation_sizes"]) == (2, 4, {3: 1, 1: 1})
    inner.fail = True
    with pytest.raises(EmbedUnavailable):
        e.embed(["x"])
    assert m.snapshot()["embed_unavailable_raised"] == 1


def test_each_reply_goes_into_the_spending_table_and_the_limit_stops_the_next_call():
    cm = cost.CostMeter({"text-embedding-v4": (0.07, 0.0)}, limit_usd=0.0001)
    real = Real(reply(1_000_000), reply(10))
    m = embed_meter.EmbedMeter(real, cap_tokens=10 ** 9, cost_meter=cm, price_key="text-embedding-v4")  # gate:allow-secret
    body = b'{"model": "m", "input": ["abcdefgh"]}'              # 上界 10 token：先按上界占额度，回来按实际记
    m.transport("u", {}, body, 5)
    assert cm.spent == pytest.approx(0.07) and cm.tokens["text-embedding-v4"] == {"prompt": 1_000_000, "completion": 0, "calls": 1}
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, body, 5)
    assert real.calls == 1


def test_a_successful_reply_without_usage_is_kept_at_the_reserved_amount_and_an_error_reply_is_not_charged():
    m = embed_meter.EmbedMeter(Real((200, b'{"data": []}'), (429, b"{}")), cap_tokens=100)
    body = b'{"model": "m", "input": ["abcdefgh"]}'              # 上界 10
    m.transport("u", {}, body, 5)
    m.transport("u", {}, body, 5)
    s = m.snapshot()
    assert s["tokens_by_replies"] == 10 and s["replies_without_usage"] == 2 and s["http_status"] == {"200": 1, "429": 1}


def test_the_meter_is_safe_to_call_from_many_threads():
    real = Real(*[reply(1) for _ in range(200)])
    m = embed_meter.EmbedMeter(real, cap_tokens=10 ** 6)
    body = b'{"model": "m", "input": ["abcdefgh"]}'
    ts = [threading.Thread(target=lambda: [m.transport("u", {}, body, 5) for _ in range(25)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert m.snapshot()["calls"] == 200 and m.snapshot()["tokens_by_replies"] == 200


def test_a_reply_that_lands_exactly_on_the_cap_closes_the_door():
    real = Real(reply(50), reply(1))
    m = embed_meter.EmbedMeter(real, cap_tokens=50)
    m.transport("u", {}, b"", 5)
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, b"", 5)
    assert real.calls == 1
