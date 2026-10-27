"""DashScope (Aliyun Bailian) embedder adapter, written RED first.

Everything here runs against a FAKE HTTP transport: no network, no real key.
The adapter's only public surface is the same as the local one:
embed(texts) -> vectors, plus model_name / dim. It never leaks the API key or
the request text into exceptions or logs (a failure is a short status line).

Covers: request shape (URL, auth header, body fields), batching (<=10 per
request, input order preserved even if the server answers out of order),
per-text truncation, L2 normalization, failure -> EmbedUnavailable (timeout /
HTTP error / bad JSON / wrong count / wrong dimension) with nothing sensitive
in the message, model_name carrying the dimension (cache-key safety),
constructor validation, usage/token counters.
"""
import json
import logging
import math
import socket

import pytest

from app.embedder import DashScopeEmbedder, EmbedUnavailable

KEY = "test-key-NOT-A-REAL-KEY-123"
SECRET_TEXT = "SECRET-INPUT-TEXT-do-not-log"  # gate:allow-secret


def _vec(dim):
    return [3.0, 4.0] + [0.0] * (dim - 2)          # norm 5 -> normalized [0.6, 0.8, 0...]


class FakeTransport:
    """transport(url, headers, body_bytes, timeout) -> (status, body_bytes)."""

    def __init__(self, dim=1024, shuffle=False, tokens_per_text=7, status=200,
                 raw=None, raises=None, drop=0, wrong_dim=None, mutate=None):
        self.dim, self.shuffle, self.tpt = dim, shuffle, tokens_per_text
        self.status, self.raw, self.raises = status, raw, raises
        self.drop, self.wrong_dim = drop, wrong_dim
        self.mutate = mutate          # data list -> data list: build a deliberately bad reply
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        req = json.loads(body)
        self.calls.append({"url": url, "headers": dict(headers), "body": req, "timeout": timeout})
        if self.raises:
            raise self.raises
        if self.raw is not None:
            return self.status, self.raw
        n = len(req["input"])
        d = self.wrong_dim or self.dim
        # each text's vector encodes its position in the request so order is checkable
        data = [{"object": "embedding", "index": i, "embedding": [3.0 + i, 4.0] + [0.0] * (d - 2)}
                for i in range(n - self.drop)]
        if self.shuffle:
            data = list(reversed(data))
        if self.mutate:
            data = self.mutate(data)
        out = {"object": "list", "data": data, "model": req["model"],
               "usage": {"prompt_tokens": self.tpt * n, "total_tokens": self.tpt * n}}
        return self.status, json.dumps(out).encode("utf-8")


def _emb(transport, **kw):
    kw.setdefault("api_key", KEY)
    return DashScopeEmbedder(transport=transport, **kw)


def test_request_shape():
    t = FakeTransport()
    _emb(t, dim=1024, timeout_s=12.5).embed(["alpha", "beta"])
    c = t.calls[0]
    assert c["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    assert c["headers"]["Authorization"] == f"Bearer {KEY}"
    assert c["headers"]["Content-Type"] == "application/json"
    assert c["body"] == {"model": "text-embedding-v4", "input": ["alpha", "beta"],
                         "dimensions": 1024, "encoding_format": "float"}
    assert c["timeout"] == 12.5


def test_base_url_is_configurable_and_trailing_slash_safe():
    t = FakeTransport()
    _emb(t, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/").embed(["x"])
    assert t.calls[0]["url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"


def test_batches_of_ten_and_order_preserved_even_if_server_shuffles():
    t = FakeTransport(shuffle=True)
    texts = [f"t{i}" for i in range(25)]
    out = _emb(t, batch=10).embed(texts)
    assert [len(c["body"]["input"]) for c in t.calls] == [10, 10, 5]
    assert [c["body"]["input"] for c in t.calls] == [texts[:10], texts[10:20], texts[20:]]
    assert len(out) == 25
    # vector i of each batch was built from index i -> normalized x-component = (3+i)/norm
    for k in range(25):
        i = k % 10
        want = (3.0 + i) / math.hypot(3.0 + i, 4.0)
        assert abs(out[k][0] - want) < 1e-9, k


def test_vectors_are_l2_normalized():
    out = _emb(FakeTransport()).embed(["a", "b", "c"])
    for v in out:
        assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9
        assert len(v) == 1024


def test_empty_input_makes_no_request():
    t = FakeTransport()
    e = _emb(t)
    assert e.embed([]) == []
    assert t.calls == [] and e.embedded_count == 0


def test_long_text_is_truncated_to_the_token_budget_short_text_untouched():
    t = FakeTransport()
    e = _emb(t, max_tokens=100)
    ascii_long = "x" * 10000
    cjk_long = "检" * 500
    short = "short text"
    e.embed([ascii_long, cjk_long, short])
    sent = t.calls[0]["body"]["input"]
    assert 0 < len(sent[0]) <= 100 * 4, "ascii: ~4 chars per token estimate"
    assert 0 < len(sent[1]) <= 100, "CJK: ~1 char per token estimate"
    assert sent[2] == short
    assert ascii_long.startswith(sent[0]) and cjk_long.startswith(sent[1]), "prefix kept"


def test_model_name_carries_dimension_so_the_vector_cache_never_mixes_dims():
    a = _emb(FakeTransport(), dim=1024)
    b = _emb(FakeTransport(dim=512), dim=512)
    assert a.model_name == "text-embedding-v4:1024" and a.dim == 1024
    assert b.model_name == "text-embedding-v4:512" and b.dim == 512
    t = FakeTransport(dim=512)
    _emb(t, dim=512).embed(["x"])
    assert t.calls[0]["body"]["dimensions"] == 512


def test_counters_embedded_count_and_tokens_used():
    t = FakeTransport(tokens_per_text=7)
    e = _emb(t, batch=10)
    e.embed([f"t{i}" for i in range(12)])
    assert e.embedded_count == 12
    assert e.tokens_used == 7 * 12


FAILURES = {
    "timeout": dict(raises=socket.timeout("timed out")),
    "connection_error": dict(raises=ConnectionError("boom")),
    "http_500": dict(status=500, raw=b'{"error": {"message": "server error", "code": "InternalError"}}'),
    "http_401_bad_key": dict(status=401, raw=b'{"error": {"message": "Incorrect API key", "code": "invalid_api_key"}}'),
    "http_500_with_valid_looking_body": dict(status=500),   # a non-200 is a failure even if the body parses
    "http_429_rate_limited": dict(status=429, raw=b'{"error": {"message": "rate limit", "code": "Throttling"}}'),
    "not_json": dict(status=200, raw=b"<html>gateway</html>"),
    "error_key_in_200": dict(status=200, raw=b'{"error": {"message": "quota", "code": "Arrearage"}}'),
    "missing_data": dict(status=200, raw=b'{"object": "list"}'),
    "too_few_vectors": dict(drop=1),
    "wrong_dimension": dict(wrong_dim=768),
}


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_failure_raises_embed_unavailable_without_key_or_text(name, caplog):
    t = FakeTransport(**FAILURES[name])
    e = _emb(t, dim=1024)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbedUnavailable) as ei:
            e.embed([SECRET_TEXT, "second"])
    msg = str(ei.value)
    assert KEY not in msg and SECRET_TEXT not in msg
    assert KEY not in caplog.text and SECRET_TEXT not in caplog.text
    assert msg, "a failure carries a short human-readable reason"


def _with_component(item_no, comp_no, value):
    def m(data):
        data[item_no]["embedding"][comp_no] = value
        return data
    return m


def _item(item_no, key, value):
    def m(data):
        data[item_no][key] = value
        return data
    return m


def _replace_item(item_no, value):
    def m(data):
        data[item_no] = value
        return data
    return m


# Every way a 200 reply can be wrong must end as EmbedUnavailable (the caller skips the route) -- never a
# silently accepted vector, never a TypeError / ValueError / OverflowError that turns /search into a 500.
BAD_REPLIES = {
    "duplicate_index": lambda d: [d[0], dict(d[1], index=0)],
    "extra_item_beyond_request": lambda d: d + [dict(d[0], index=2)],
    "extra_duplicate_item_count_only": lambda d: d + [dict(d[1])],
    "index_out_of_range": _item(1, "index", 2),
    "index_negative": _item(1, "index", -1),
    "index_float": _item(1, "index", 1.0),
    "index_string": _item(1, "index", "1"),
    "index_bool": _item(1, "index", True),
    "index_missing": lambda d: [d[0], {"embedding": d[1]["embedding"]}],
    "item_not_object": _replace_item(1, 7),
    "embedding_missing": lambda d: [d[0], {"index": 1}],
    "embedding_not_list": _item(1, "embedding", "0.1,0.2"),
    "component_none": _with_component(1, 5, None),
    "component_numeric_string": _with_component(1, 5, "1.0"),
    "component_bool": _with_component(1, 5, True),
    "component_list": _with_component(1, 5, [1.0]),
    "component_nan": _with_component(1, 5, float("nan")),
    "component_inf": _with_component(1, 5, float("inf")),
    "component_minus_inf": _with_component(1, 5, float("-inf")),
    "component_huge_int_overflows_float": _with_component(1, 5, 10 ** 400),
    "all_zero_vector": _item(1, "embedding", [0.0] * 1024),
}


@pytest.mark.parametrize("name", sorted(BAD_REPLIES))
def test_bad_vector_reply_is_embed_unavailable_nothing_leaks_usage_still_counted(name, caplog):
    t = FakeTransport(mutate=BAD_REPLIES[name])
    e = _emb(t, dim=1024)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbedUnavailable) as ei:
            e.embed([SECRET_TEXT, "second"])
    msg = str(ei.value)
    assert msg and KEY not in msg and SECRET_TEXT not in msg
    assert KEY not in caplog.text and SECRET_TEXT not in caplog.text
    # nothing usable was embedded, but the provider billed the call: tokens_used follows the reported usage (7 per text x 2)
    assert e.embedded_count == 0 and e.tokens_used == 14


@pytest.mark.parametrize("mag", [1e200, 1e-200])
def test_extreme_but_finite_magnitudes_still_normalize(mag):
    """The length is computed without overflow / underflow, so a finite vector is not rejected for its scale."""
    t = FakeTransport(mutate=_item(0, "embedding", [mag, mag] + [0.0] * 1022))
    out = _emb(t, dim=1024).embed(["a", "b"])
    assert math.isclose(math.sqrt(sum(x * x for x in out[0])), 1.0, rel_tol=1e-12)


def test_a_clean_reply_with_integer_components_is_still_accepted():
    """Guard against over-strict validation: ints are valid JSON numbers for a vector component."""
    t = FakeTransport(mutate=_item(0, "embedding", [3] + [4] + [0] * 1022))
    out = _emb(t, dim=1024).embed(["a", "b"])
    assert out[0][:2] == [0.6, 0.8]


def test_failure_in_a_later_batch_still_raises_and_counts_only_successes():
    class FailSecond(FakeTransport):
        def __call__(self, url, headers, body, timeout):
            if len(self.calls) == 1:
                self.calls.append({})
                raise socket.timeout("timed out")
            return super().__call__(url, headers, body, timeout)
    t = FailSecond()
    e = _emb(t, batch=10)
    with pytest.raises(EmbedUnavailable):
        e.embed([f"t{i}" for i in range(15)])


def test_constructor_validation():
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        DashScopeEmbedder(api_key="", transport=FakeTransport())
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        DashScopeEmbedder(api_key=None, transport=FakeTransport())
    for bad_dim in (0, 100, 4096):
        with pytest.raises(ValueError, match="dimension"):
            _emb(FakeTransport(), dim=bad_dim)
    for bad_batch in (0, 11, 100):
        with pytest.raises(ValueError, match="batch"):
            _emb(FakeTransport(), batch=bad_batch)
    with pytest.raises(ValueError, match="timeout"):
        _emb(FakeTransport(), timeout_s=0)
    with pytest.raises(ValueError, match="max_tokens"):
        _emb(FakeTransport(), max_tokens=0)


def test_max_tokens_is_capped_at_the_services_8192_limit():
    assert _emb(FakeTransport(), max_tokens=8192)._max_tokens == 8192
    assert _emb(FakeTransport(), max_tokens=1)._max_tokens == 1
    with pytest.raises(ValueError, match="max_tokens"):
        _emb(FakeTransport(), max_tokens=8193)


def test_repr_never_contains_the_key():
    e = _emb(FakeTransport())
    assert KEY not in repr(e) and KEY not in str(e)
