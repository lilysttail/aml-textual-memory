"""Embedder plug: embed(texts) -> vectors, plus model_name / dim.

Two adapters behind the same shape (the upper layers know nothing about a
provider's protocol):
  - LocalEmbedder     bge-small onnx, development / internal reference only;
  - DashScopeEmbedder Aliyun Bailian text-embedding-v4 over plain HTTP.

The dev-only deps (onnxruntime, tokenizers, numpy) and the model files are
loaded lazily. If anything is missing, LocalEmbedder raises loudly at
construction — the dense path never silently falls back to something else.

DashScopeEmbedder talks to the OpenAI-compatible endpoint with the standard
library only (no new dependency, nothing added to the deploy image). Its
request shape follows LiteLLM's DashScope embedding transformation (MIT,
BerriAI/litellm, litellm/llms/dashscope/embed/transformation.py), rewritten
without the framework layer — see THIRD_PARTY_NOTICES.md. A timeout / HTTP
error / malformed answer raises EmbedUnavailable, which the search layer
treats as "skip this route", never as a failed Search.
"""
import glob
import json
import math
import os
import re
import threading
import urllib.error
import urllib.request

# One exception for the whole service: the core catches ITS copy when a
# provider is down ("skip this route"), so the shell's provider must raise the
# very same class.
from core.embedder import EmbedUnavailable  # noqa: F401

_DEFAULT_GLOB = os.path.join(
    os.path.dirname(__file__), "..", "..", "bench_downloads", "hf",
    "models--Qdrant--bge-small-en-v1.5-onnx-Q", "snapshots", "*")
_MAX_TOKENS = 512


class LocalEmbedder:
    """bge-small-en-v1.5 via onnxruntime. CLS pooling + L2 normalize, 512-token
    window (the model's real limit; the benchmark's v4 sees 8192, so local
    scores are a low-biased internal reference only)."""
    provider = "local"
    model_name = "bge-small-en-v1.5"
    dim = 384

    def __init__(self, model_dir=None, max_tokens=_MAX_TOKENS):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(
                "LocalEmbedder needs dev deps (onnxruntime, tokenizers, numpy); "
                "install requirements-dev.txt") from e
        self._np = np
        d = model_dir
        if not d:
            matches = sorted(glob.glob(_DEFAULT_GLOB))
            d = matches[0] if matches else None
        if not d or not os.path.isdir(d):
            raise RuntimeError(
                "bge-small model dir not found (pass model_dir or place it "
                "under bench_downloads/...): looked at {!r}".format(d))
        onnx_path = os.path.join(d, "model_optimized.onnx")
        tok_path = os.path.join(d, "tokenizer.json")
        for p in (onnx_path, tok_path):
            if not os.path.exists(p):
                raise RuntimeError(f"missing model file: {p}")
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=max_tokens)
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._in_names = [i.name for i in self._sess.get_inputs()]
        self.embedded_count = 0  # how many texts were actually embedded (cache misses)

    def embed(self, texts, timeout_s=None):
        """bge-small local inference; timeout_s accepted for signature parity
        (local compute has no network timeout)."""
        if not texts:
            return []
        self.embedded_count += len(texts)
        np = self._np
        encs = [self._tok.encode(t) for t in texts]
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.attention_mask)] = e.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._in_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self._sess.run(None, feed)[0]   # [B, L, 384]
        cls = out[:, 0]                        # CLS pooling (bge official)
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (cls / norms).astype(float).tolist()



# ---------------------------------------------------------------------------
# DashScope (Aliyun Bailian) adapter
# ---------------------------------------------------------------------------

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_MODEL = "text-embedding-v4"
# Documented limits of text-embedding-v4: at most 10 texts per request; output
# dimension one of these (default 1024).
DASHSCOPE_DIMS = (64, 128, 256, 512, 768, 1024, 1536, 2048)
DASHSCOPE_MAX_BATCH = 10
DASHSCOPE_MAX_TOKENS_LIMIT = 8192   # the service's own per-text input limit; a larger setting could only fail at run time
_SAFE_CODE = re.compile(r"[^A-Za-z0-9_.-]")


def estimate_tokens(text):
    """Cheap token estimate used for truncation: ~4 ASCII chars per token,
    1 token per non-ASCII char (CJK). The real tokenizer lives server-side; the
    configured limit is a soft cap well under the API's own 8192, so an
    estimate is enough."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return math.ceil(ascii_chars / 4) + (len(text) - ascii_chars)


def _truncate_to_tokens(text, max_tokens):
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def _urllib_transport(url, headers, body, timeout):
    """(url, headers, body bytes, timeout) -> (status, body bytes). Network
    errors propagate; the adapter turns them into EmbedUnavailable."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class DashScopeEmbedder:
    """text-embedding-v4 via the OpenAI-compatible endpoint, direct HTTP.

    model_name carries the dimension ("text-embedding-v4:1024") so the vector
    cache (keyed by model name) can never mix vectors of different sizes.
    Vectors are L2-normalized here, so the dense path's dot product is cosine
    no matter what the service returns."""
    provider = "dashscope"

    def __init__(self, api_key, dim=1024, batch=DASHSCOPE_MAX_BATCH, timeout_s=20.0,
                 max_tokens=2000, base_url=_DASHSCOPE_BASE_URL, model=_DASHSCOPE_MODEL,
                 transport=None):
        if not api_key:
            raise ValueError("DashScopeEmbedder needs the DASHSCOPE_API_KEY environment variable "
                             "(read by app.config; the adapter itself never reads the environment)")
        if dim not in DASHSCOPE_DIMS:
            raise ValueError(f"unsupported dimension {dim!r} (allowed: {list(DASHSCOPE_DIMS)})")
        if not (1 <= batch <= DASHSCOPE_MAX_BATCH):
            raise ValueError(f"batch must be 1..{DASHSCOPE_MAX_BATCH} (the API takes at most "
                             f"{DASHSCOPE_MAX_BATCH} texts per request), got {batch!r}")
        if not timeout_s > 0:
            raise ValueError(f"timeout must be > 0 seconds, got {timeout_s!r}")
        if not (1 <= max_tokens <= DASHSCOPE_MAX_TOKENS_LIMIT):
            raise ValueError(f"max_tokens must be 1..{DASHSCOPE_MAX_TOKENS_LIMIT} (the service's input limit), "
                             f"got {max_tokens!r}")
        self._api_key = api_key
        self.dim = dim
        self._batch = batch
        self._timeout = timeout_s
        self._max_tokens = max_tokens
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._transport = transport or _urllib_transport
        self.model_name = f"{model}:{dim}"
        self.embedded_count = 0   # texts embedded (cache misses), same meaning as LocalEmbedder
        self.tokens_used = 0      # sum of the usage the service reported (billed, even for replies we rejected)
        self._lock = threading.Lock()

    def __repr__(self):
        return f"DashScopeEmbedder(model={self.model_name!r}, batch={self._batch})"

    def embed(self, texts, timeout_s=None):
        """Embed *texts*. timeout_s overrides the per-request timeout when
        given (the search path passes the remaining backfill budget so a slow
        provider call gives up at the deadline instead of after it)."""
        if not texts:
            return []
        clipped = [_truncate_to_tokens(t, self._max_tokens) for t in texts]
        out = []
        for i in range(0, len(clipped), self._batch):
            out.extend(self._embed_batch(clipped[i:i + self._batch],
                                         timeout_s=timeout_s))
        with self._lock:
            self.embedded_count += len(texts)
        return out

    def _embed_batch(self, batch, timeout_s=None):
        body = json.dumps({"model": self._model, "input": batch, "dimensions": self.dim,
                           "encoding_format": "float"}).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            status, raw = self._transport(
                self._url, headers, body,
                timeout_s if timeout_s is not None else self._timeout)
        except Exception as e:                       # timeout, DNS, connection reset, ...
            raise EmbedUnavailable(f"DashScope request failed: {type(e).__name__}") from None
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            raise EmbedUnavailable(f"DashScope HTTP {status}: reply is not JSON") from None
        if not isinstance(obj, dict):
            raise EmbedUnavailable(f"DashScope HTTP {status}: unexpected reply shape")
        if status != 200 or "error" in obj:
            raise EmbedUnavailable(f"DashScope HTTP {status}{self._error_code(obj)}")
        usage = obj.get("usage")                         # the call was billed whatever its vectors look like, so the
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None   # count follows the provider's reported usage
        if isinstance(tokens, int):
            with self._lock:
                self.tokens_used += tokens
        data = obj.get("data")
        if not isinstance(data, list):
            raise EmbedUnavailable("DashScope reply has no data list")
        if len(data) != len(batch):                      # raw count first: duplicates / extras can't hide behind a dict
            raise EmbedUnavailable(f"DashScope reply has {len(data)} vectors for {len(batch)} texts")
        vectors = [None] * len(batch)
        for item in data:
            idx = item.get("index") if isinstance(item, dict) else None
            if type(idx) is not int or not 0 <= idx < len(batch):    # type(): a bool is not an index
                raise EmbedUnavailable("DashScope reply has a missing or out-of-range vector index")
            if vectors[idx] is not None:
                raise EmbedUnavailable("DashScope reply repeats a vector index")
            vectors[idx] = self._unit_vector(item.get("embedding"))
        return vectors

    @staticmethod
    def _error_code(obj):
        err = obj.get("error")
        code = err.get("code") if isinstance(err, dict) else obj.get("code")
        return f" ({_SAFE_CODE.sub('', str(code))[:40]})" if code else ""

    def _unit_vector(self, vec):
        """One reply vector -> L2-normalized floats. Anything that is not exactly `dim` finite numbers with a
        non-zero length is a bad reply (EmbedUnavailable), never a TypeError / NaN downstream."""
        if not isinstance(vec, list) or len(vec) != self.dim:
            raise EmbedUnavailable(f"DashScope reply vector malformed (expected dimension {self.dim})")
        try:
            v = [float(x) for x in vec if type(x) in (int, float)]   # bool / str / None / list are not numbers
        except OverflowError:                                          # an int too big for a float
            raise EmbedUnavailable("DashScope reply vector has a component out of range") from None
        if len(v) != len(vec):
            raise EmbedUnavailable("DashScope reply vector has a non-numeric component")
        n = math.hypot(*v)               # no overflow / underflow on extreme magnitudes; NaN / Inf in any component -> NaN / Inf here
        if not (math.isfinite(n) and n > 0):
            raise EmbedUnavailable("DashScope reply vector cannot be normalized (zero length or non-finite)")
        return [x / n for x in v]
