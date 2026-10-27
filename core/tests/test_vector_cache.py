"""向量缓存的 single-flight：认领之后任何异常都要清理 flight（栖 P1-3）。

认领（把 hash 放进 _inflight）之后的任何一次读失败，都必须给 flight 设上
error、发 event、把它从 _inflight 里拿走。否则这个 hash 永远没人释放，
后来的请求只能等到超时。
"""
import pytest

from core.vector_cache import VectorCache


class _Embedder:
    model_name = "fake:4"
    dim = 4

    def embed(self, texts, timeout_s=None):
        return [[0.0] * 4 for _ in texts]


class _FailingStore:
    """The second cache read fails — the case that used to leak a flight."""

    def __init__(self):
        self.calls = 0

    def get_vectors(self, model, hashes):
        self.calls += 1
        # The read that follows the claim (the even-numbered one) always fails.
        if self.calls % 2 == 0:
            raise RuntimeError("cache read failed")
        return {}

    def put_vectors(self, model, dim, rows):
        return None


def test_a_failing_cache_read_clears_the_flight():
    vc = VectorCache(_FailingStore(), _Embedder())
    with pytest.raises(RuntimeError):
        vc.get_or_compute(["一段文本"])
    assert vc._inflight == {}, "the claimed flight must not stay behind"


def test_the_next_caller_fails_fast_instead_of_waiting_for_a_timeout():
    vc = VectorCache(_FailingStore(), _Embedder(), wait_timeout_s=1.0)
    with pytest.raises(RuntimeError):
        vc.get_or_compute(["一段文本"])
    # The flight is gone, so the next caller claims it again and fails the same
    # way — instead of waiting on a flight nobody will ever release.
    with pytest.raises(RuntimeError):
        vc.get_or_compute(["一段文本"])
