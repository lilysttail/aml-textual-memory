"""The embedder plug: embed(texts) -> vectors, plus model_name / dim.

复制自本项目早期服务层的 app/embedder.py（本仓库不含该文件；2026-10-21，小助手），改了：
  - 供应商专名去掉：适配器是通用的 OpenAI 兼容 /embeddings 客户端
    （HttpEmbedder），base_url / model / dim / api_key 全部由外面传，核心不写死
    任何一个供应商或模型名；本地 bge-small 那套留在比赛壳里；
  - 回包校验一点没松：条数、index 范围、重复、维度、非有限值、零向量；
  - 仍然是 EmbedUnavailable 语义：超时 / HTTP 错 / 回包坏 -> 这一路跳过，
    Search 不失败。

调用方自己决定用哪个模型：比赛那边锁学术榜规定的那个，别的使用方可以用本机模型
模型（2b）。核心只认协议。
"""
import json
import math
import re
import threading
import urllib.error
import urllib.request

_SAFE_CODE = re.compile(r"[^A-Za-z0-9_.-]")


class EmbedUnavailable(Exception):
    """The embedding provider could not answer (timeout, HTTP error, bad reply).

    The message is a short status line only — never the request text, never
    the API key. Callers skip the route that needed the vectors."""


def estimate_tokens(text):
    """Cheap token estimate used for truncation: ~4 ASCII chars per token,
    1 token per non-ASCII char (CJK). The real tokenizer lives server-side."""
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


class HttpEmbedder:
    """Any OpenAI-compatible /embeddings endpoint, direct HTTP.

    model_name carries the dimension ("<model>:<dim>") so the vector cache
    (keyed by model name) can never mix vectors of different sizes. Vectors
    are L2-normalized here, so the dense path's dot product is cosine no
    matter what the service returns.
    """

    provider = "http"

    def __init__(self, model, dim, api_key=None, base_url=None, batch=10,
                 timeout_s=20.0, max_tokens=2000, transport=None):
        if not model:
            raise ValueError("HttpEmbedder needs a model name")
        if not base_url:
            raise ValueError("HttpEmbedder needs the endpoint's base_url")
        if not isinstance(dim, int) or dim < 1:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if not (1 <= batch <= 64):
            raise ValueError(f"batch must be 1..64, got {batch!r}")
        if not timeout_s > 0:
            raise ValueError(f"timeout must be > 0 seconds, got {timeout_s!r}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}")
        self._api_key = api_key
        self.dim = dim
        self._batch = batch
        self._timeout = timeout_s
        self._max_tokens = max_tokens
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._transport = transport or _urllib_transport
        self.model_name = f"{model}:{dim}"
        self.embedded_count = 0   # texts embedded (cache misses)
        self.tokens_used = 0      # sum of the usage the service reported
        self._lock = threading.Lock()

    def __repr__(self):
        return f"HttpEmbedder(model={self.model_name!r}, batch={self._batch})"

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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            status, raw = self._transport(
                self._url, headers, body,
                timeout_s if timeout_s is not None else self._timeout)
        except Exception as e:                       # timeout, DNS, connection reset, ...
            raise EmbedUnavailable(f"embedding request failed: {type(e).__name__}") from None
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            raise EmbedUnavailable(f"embedding HTTP {status}: reply is not JSON") from None
        if not isinstance(obj, dict):
            raise EmbedUnavailable(f"embedding HTTP {status}: unexpected reply shape")
        if status != 200 or "error" in obj:
            raise EmbedUnavailable(f"embedding HTTP {status}{self._error_code(obj)}")
        usage = obj.get("usage")                         # the call was billed whatever its vectors look like
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, int):
            with self._lock:
                self.tokens_used += tokens
        data = obj.get("data")
        if not isinstance(data, list):
            raise EmbedUnavailable("embedding reply has no data list")
        if len(data) != len(batch):                      # raw count first: duplicates / extras can't hide behind a dict
            raise EmbedUnavailable(f"embedding reply has {len(data)} vectors for {len(batch)} texts")
        vectors = [None] * len(batch)
        for item in data:
            idx = item.get("index") if isinstance(item, dict) else None
            if type(idx) is not int or not 0 <= idx < len(batch):    # type(): a bool is not an index
                raise EmbedUnavailable("embedding reply has a missing or out-of-range vector index")
            if vectors[idx] is not None:
                raise EmbedUnavailable("embedding reply repeats a vector index")
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
            raise EmbedUnavailable(f"embedding reply vector malformed (expected dimension {self.dim})")
        try:
            v = [float(x) for x in vec if type(x) in (int, float)]   # bool / str / None / list are not numbers
        except OverflowError:                                          # an int too big for a float
            raise EmbedUnavailable("embedding reply vector has a component out of range") from None
        if len(v) != len(vec):
            raise EmbedUnavailable("embedding reply vector has a non-numeric component")
        n = math.hypot(*v)               # no overflow / underflow on extreme magnitudes; NaN / Inf -> NaN / Inf here
        if not (math.isfinite(n) and n > 0):
            raise EmbedUnavailable("embedding reply vector cannot be normalized (zero length or non-finite)")
        return [x / n for x in v]
