"""Test doubles that live inside the core package (no real model, no network).

FakeEmbedder: a deterministic embedder — the same text always gets the same
vector, and texts that share characters land near each other. Enough to
exercise the dense route and the rerank without a provider.
"""
import hashlib
import math


class FakeEmbedder:
    """Deterministic stand-in for a real embedding provider.

    model_name carries the dimension, like the real adapters: "fake:16".
    """

    provider = "fake"

    def __init__(self, dim=16, model="fake"):
        self.dim = dim
        self._model = model
        self.model_name = f"{model}:{dim}"
        self.embedded_count = 0
        self.tokens_used = 0

    def embed(self, texts, timeout_s=None):
        """Deterministic fake vectors. timeout_s accepted for signature parity
        with the real adapters (the fake never times out)."""
        if not texts:
            return []
        self.embedded_count += len(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        vec = []
        for i in range(self.dim):
            h = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
            vec.append((h[0] / 255.0) * 2.0 - 1.0)
        n = math.hypot(*vec) or 1.0
        return [x / n for x in vec]
