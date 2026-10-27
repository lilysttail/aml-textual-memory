"""the ONE real-network smoke for the DashScope adapter. OFF by default.

Runs only when RUN_DASHSCOPE_SMOKE=1 AND DASHSCOPE_API_KEY is in the environment
(set in the shell or by a secrets manager). It embeds two short texts and prints how
many tokens the call cost. A skip is not a pass: the fake-transport tests in
test_dashscope.py are what protect the adapter every day.

    python -m pytest tests/test_dashscope_smoke.py -q -s
    (with RUN_DASHSCOPE_SMOKE=1 and DASHSCOPE_API_KEY set in that shell)
"""
import math
import os

import pytest

from app.config import DASHSCOPE_DEFAULT_BASE_URL
from app.embedder import DashScopeEmbedder

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DASHSCOPE_SMOKE") != "1" or not os.environ.get("DASHSCOPE_API_KEY"),
    reason="real-network smoke: set RUN_DASHSCOPE_SMOKE=1 and DASHSCOPE_API_KEY")


def test_real_call_two_texts():
    # same endpoint the app would use: DASHSCOPE_BASE_URL if set, else the config default (international host)
    kw = {"base_url": os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_DEFAULT_BASE_URL}
    emb = DashScopeEmbedder(api_key=os.environ["DASHSCOPE_API_KEY"], **kw)
    vecs = emb.embed(["connection pool exhausted", "the database ran out of connections"])
    assert len(vecs) == 2 and all(len(v) == emb.dim for v in vecs)
    assert all(abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6 for v in vecs)
    cos = sum(a * b for a, b in zip(*vecs))
    print(f"[dashscope smoke] model={emb.model_name} tokens_used={emb.tokens_used} cos(similar pair)={cos:.3f}")
    assert emb.tokens_used > 0 and cos > 0.3
