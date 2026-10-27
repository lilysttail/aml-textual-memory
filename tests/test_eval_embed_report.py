"""follow-up (柯's ruling): the ruler prints the real token count at the end and records
provider / model / dimension in the results JSON. Written RED first.

eval/embed_report.py reads the embedder behind the app's VectorCache and returns a small dict;
score.py / score_supplement.py put it in their result JSON as "embedder" and print one line.
"""
import os
from types import SimpleNamespace

from app.embedder import DashScopeEmbedder, LocalEmbedder
from eval.embed_report import embed_report, format_line

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _transport(url, headers, body, timeout):
    import json
    n = len(json.loads(body)["input"])
    data = [{"index": i, "embedding": [3.0, 4.0] + [0.0] * 1022} for i in range(n)]
    return 200, json.dumps({"data": data, "usage": {"total_tokens": 7 * n}}).encode()


def test_dashscope_report_has_provider_model_dimension_and_real_tokens():
    emb = DashScopeEmbedder(api_key="test-key-NOT-A-REAL-KEY-123", transport=_transport)
    emb.embed(["a", "b", "c"])
    rep = embed_report(SimpleNamespace(_emb=emb))
    assert rep == {"provider": "dashscope", "model": "text-embedding-v4:1024", "dimension": 1024,
                   "vectors_embedded": 3, "tokens_used": 21}


def test_local_report_has_no_token_count():
    class FakeLocal:
        provider, model_name, dim, embedded_count = "local", "bge-small-en-v1.5", 384, 5
    rep = embed_report(SimpleNamespace(_emb=FakeLocal()))
    assert rep == {"provider": "local", "model": "bge-small-en-v1.5", "dimension": 384,
                   "vectors_embedded": 5, "tokens_used": None}


def test_no_vector_cache_keyword_only_run():
    assert embed_report(None) == {"provider": None, "model": None, "dimension": None,
                                  "vectors_embedded": None, "tokens_used": None}


def test_format_line():
    rep = {"provider": "dashscope", "model": "text-embedding-v4:1024", "dimension": 1024,
           "vectors_embedded": 3, "tokens_used": 21}
    line = format_line(rep)
    for needle in ("provider=dashscope", "model=text-embedding-v4:1024", "dim=1024", "vectors_embedded=3", "tokens_used=21"):
        assert needle in line, line
    # the number is what the provider reported, billed even for replies we rejected -- the line says so
    assert "as reported by the provider's usage" in line, line
    assert "tokens_used=n/a" in format_line(dict(rep, provider="local", tokens_used=None))
    assert "reported" not in format_line(dict(rep, provider="local", tokens_used=None))
    assert "keyword-only" in format_line(embed_report(None))


def test_embedders_declare_their_provider():
    assert LocalEmbedder.provider == "local"
    assert DashScopeEmbedder(api_key="k" * 8, transport=_transport).provider == "dashscope"


def test_the_ruler_records_the_embedder_it_used():
    # The text-track ruler (eval/ruler) runs the in-process service on the
    # keyword path by default: no embedder, no key. The meaning-based path is
    # switched on only by a DenseConfig, and its calls are counted by the
    # ruler's own meter (eval/ruler/embed_meter.py).
    src = _read("eval/ruler/dense.py")
    assert "the keyword path never sees a key" in src and 'pop("DASHSCOPE_API_KEY"' in src
    assert "EmbedMeter" in src


def test_the_operations_doc_says_truncation_is_an_estimate_to_calibrate_against_usage():
    readme = _read("OPERATIONS.md")
    assert "calibrate" in readme and "usage" in readme
