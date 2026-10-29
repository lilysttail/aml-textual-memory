"""Settings — the ONLY module that reads environment variables.

Parsed once at startup; invalid values fail fast with a clear error instead of
failing halfway through a request. Everything else receives a Settings object
(or values from it) and never touches os.environ directly.

The service is the textual-memory track: the shared core (`core/`) does the
storage and the retrieval, this shell reads the environment and translates it
into a CoreConfig. The two-level recall and the coding-track switches are gone
with the code track (single 3).

Env names: DB_PATH, API_KEY, REQUIRE_REAL_API_KEY, RECALL_PATHS, RERANK_ENABLED, RERANK_N,
STORE_BUSY_TIMEOUT_MS, STORE_WRITE_LOCK_TIMEOUT_MS, EMBED_*, BGE_SMALL_DIR,
and for the embedding provider EMBED_PROVIDER (local | dashscope) +
DASHSCOPE_* (the API key is read ONLY by its variable name DASHSCOPE_API_KEY,
never from a file, and never shown in repr). CONTENT_MAX_CHARS and
NEIGHBOUR_CANDIDATES belong to the shell's rendering.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# DashScope endpoint default: the INTERNATIONAL (Singapore) host, because the owner uses that console and an
# international key only works there. A China-site key needs DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# (mixing them is an HTTP 401 invalid_api_key). One definition; Settings and load() both use it.
DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# The placeholder values a development run falls back to (or that the old image carried). With
# REQUIRE_REAL_API_KEY on, none of them counts as a key.
DEV_API_KEYS = frozenset({"dev-key-change-me", "change-me"})


def _get(name: str, default: str):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _float(name: str, default: float) -> float:
    raw = _get(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")


def _bool(name: str, default: bool) -> bool:
    raw = _get(name, "1" if default else "0")
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (1/0/true/false), got {raw!r}")


def _paths(name: str, default: str) -> Tuple[str, ...]:
    raw = _get(name, default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one recall path, got {raw!r}")
    return tuple(parts)


@dataclass(frozen=True)
class Settings:
    db_path: str
    api_key: str = field(repr=False)   # never in repr: a logged Settings must not carry the service's key
    recall_paths: Tuple[str, ...]
    rerank_enabled: bool          # retrieval-layer rerank on the candidate pool
    rerank_n: int                 # candidate-pool size fed to rerank
    store_busy_timeout_ms: int
    store_write_lock_timeout_ms: int
    embed_max_concurrency: int
    embed_batch: int
    embed_backfill_wait_s: float
    embed_acquire_s: float
    bge_small_dir: Optional[str]
    # Embedding provider. Defaults keep old Settings(...) constructions working
    # and keep the local adapter the default.
    embed_provider: str = "local"
    dashscope_api_key: Optional[str] = field(default=None, repr=False)   # never in repr
    dashscope_dim: int = 1024
    dashscope_batch: int = 10
    dashscope_timeout_s: float = 20.0
    dashscope_max_tokens: int = 2000
    dashscope_base_url: str = DASHSCOPE_DEFAULT_BASE_URL
    # How many characters a hit's content may grow to when the core adds the
    # neighbouring turns. The hit's own text is never cut; neighbours are
    # dropped farthest-first.
    content_max_chars: int = 4000
    # Neighbours as candidates of their own. ON by default: the version we
    # ship is the version we submit, and the ruler says the neighbours bring
    # back 3.98 points more questions with an evidence turn (+10.3% characters).
    # The environment can still turn it off (NEIGHBOUR_CANDIDATES=0).
    # When on, every neighbour becomes its own entry with its own id, the
    # de-dup is on the underlying record, and the total never exceeds top_k.
    neighbour_candidates: bool = True
    # Search-path backfill limits.
    dense_backfill_m: int = 8
    dense_backfill_t: float = 2.0
    # The query vector's own deadline (seconds; waiting for the embedding gate included).
    dense_query_t: float = 3.0
    # The in-memory dense index (core/dense_index.py): how many builds / similarity passes run at once
    # (0 = the CPUs this process may use, from the container's quota), how long a search waits for a slot
    # before it answers 503, and the memory budget of the shared vector pool + the spaces' indexes.
    search_heavy_concurrency: int = 0
    search_heavy_wait_s: float = 15.0
    search_cache_mib: int = 640
    # English stemming for the built-in keyword tokenizer. OFF by
    # default; whether the competition turns it on is decided by the answer
    # ruler, not by this code. Changing it changes the index format — an
    # existing DB refuses to start until its index is rebuilt
    # (core.engine.rebuild_fts_index).
    stem_english: bool = False
    # Set by the deployment image (Dockerfile ENV), never by a local run: the service then refuses to
    # start unless API_KEY is a real key — missing, blank or still the development placeholder stops it,
    # naming the variable. A local run or the test suite leaves it off and needs no key.
    require_real_api_key: bool = False

    @property
    def enabled_paths(self) -> List[str]:
        return list(self.recall_paths)

    def validate(self) -> "Settings":
        known = {"keyword", "dense"}
        for p in self.recall_paths:
            if p not in known:
                raise ValueError(
                    f"unknown recall path {p!r} (known: {sorted(known)})")
        for name, v in (("STORE_BUSY_TIMEOUT_MS", self.store_busy_timeout_ms),
                        ("STORE_WRITE_LOCK_TIMEOUT_MS", self.store_write_lock_timeout_ms),
                        ("EMBED_BACKFILL_WAIT_S", self.embed_backfill_wait_s),
                        ("EMBED_ACQUIRE_S", self.embed_acquire_s)):
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.embed_max_concurrency < 1:
            raise ValueError("EMBED_MAX_CONCURRENCY must be >= 1")
        if self.embed_batch < 1:
            raise ValueError("EMBED_BATCH must be >= 1")
        if self.dense_backfill_m < 0:
            raise ValueError("DENSE_BACKFILL_M must be >= 0")
        if self.dense_backfill_t < 0:
            raise ValueError("DENSE_BACKFILL_T must be >= 0")
        if not self.dense_query_t > 0:
            raise ValueError("DENSE_QUERY_T must be > 0")
        if self.search_heavy_concurrency < 0:
            raise ValueError("SEARCH_HEAVY_CONCURRENCY must be >= 0 (0 = the CPUs available)")
        if not self.search_heavy_wait_s > 0:
            raise ValueError("SEARCH_HEAVY_WAIT_S must be > 0")
        if self.search_cache_mib < 1:
            raise ValueError("SEARCH_CACHE_MIB must be >= 1")
        if self.rerank_n < 1:
            raise ValueError("RERANK_N must be >= 1")
        if self.embed_provider not in ("local", "dashscope"):
            raise ValueError(f"EMBED_PROVIDER must be 'local' or 'dashscope', got {self.embed_provider!r}")
        from .embedder import DASHSCOPE_DIMS, DASHSCOPE_MAX_BATCH, DASHSCOPE_MAX_TOKENS_LIMIT
        if self.dashscope_dim not in DASHSCOPE_DIMS:
            raise ValueError(f"DASHSCOPE_DIM must be one of {list(DASHSCOPE_DIMS)}, got {self.dashscope_dim!r}")
        if not (1 <= self.dashscope_batch <= DASHSCOPE_MAX_BATCH):
            raise ValueError(f"DASHSCOPE_BATCH must be 1..{DASHSCOPE_MAX_BATCH} "
                             f"(the API takes at most {DASHSCOPE_MAX_BATCH} texts per request)")
        if not self.dashscope_timeout_s > 0:
            raise ValueError("DASHSCOPE_TIMEOUT_S must be > 0")
        if not (1 <= self.dashscope_max_tokens <= DASHSCOPE_MAX_TOKENS_LIMIT):
            raise ValueError(f"DASHSCOPE_MAX_TOKENS must be 1..{DASHSCOPE_MAX_TOKENS_LIMIT} "
                             f"(the service's input limit)")
        if self.content_max_chars < 1:
            raise ValueError(f"CONTENT_MAX_CHARS must be >= 1, got {self.content_max_chars!r}")
        if self.require_real_api_key and (not self.api_key.strip() or self.api_key.strip() in DEV_API_KEYS):
            raise ValueError("REQUIRE_REAL_API_KEY is on (the deployment image): the API_KEY environment variable "
                             "must be set to a real key — it is missing, blank or still a development placeholder "
                             "(set it on the platform's environment page; the image never carries it)")
        if self.embed_provider == "dashscope" and not (self.dashscope_api_key or "").strip():
            raise ValueError("EMBED_PROVIDER=dashscope needs the DASHSCOPE_API_KEY environment variable "
                             "(set it in the process environment — a shell variable locally, the "
                             "hosting platform's environment page when deployed; the app never reads it from a file)")
        return self


def load() -> Settings:
    s = Settings(
        db_path=_get("DB_PATH", os.path.join(
            os.path.dirname(__file__), "..", "data", "memory.db")),
        api_key=_get("API_KEY", "dev-key-change-me"),
        recall_paths=_paths("RECALL_PATHS", "keyword"),
        rerank_enabled=_bool("RERANK_ENABLED", False),
        rerank_n=_int("RERANK_N", 200),
        store_busy_timeout_ms=_int("STORE_BUSY_TIMEOUT_MS", 5000),
        store_write_lock_timeout_ms=_int("STORE_WRITE_LOCK_TIMEOUT_MS", 10000),
        embed_max_concurrency=_int("EMBED_MAX_CONCURRENCY", 2),
        embed_batch=_int("EMBED_BATCH", 32),
        embed_backfill_wait_s=_float("EMBED_BACKFILL_WAIT_S", 60.0),
        embed_acquire_s=_float("EMBED_ACQUIRE_S", 30.0),
        bge_small_dir=(lambda v: v or None)(_get("BGE_SMALL_DIR", "")),
        embed_provider=_get("EMBED_PROVIDER", "local").strip().lower(),
        dashscope_api_key=(lambda v: v or None)(_get("DASHSCOPE_API_KEY", "")),
        dashscope_dim=_int("DASHSCOPE_DIM", 1024),
        dashscope_batch=_int("DASHSCOPE_BATCH", 10),
        dashscope_timeout_s=_float("DASHSCOPE_TIMEOUT_S", 20.0),
        dashscope_max_tokens=_int("DASHSCOPE_MAX_TOKENS", 2000),
        dashscope_base_url=_get("DASHSCOPE_BASE_URL", DASHSCOPE_DEFAULT_BASE_URL),
        content_max_chars=_int("CONTENT_MAX_CHARS", 4000),
        neighbour_candidates=_bool("NEIGHBOUR_CANDIDATES", True),
        stem_english=_bool("STEM_ENGLISH", False),
        dense_backfill_m=_int("DENSE_BACKFILL_M", 8),
        dense_backfill_t=_float("DENSE_BACKFILL_T", 2.0),
        dense_query_t=_float("DENSE_QUERY_T", 3.0),
        search_heavy_concurrency=_int("SEARCH_HEAVY_CONCURRENCY", 0),
        search_heavy_wait_s=_float("SEARCH_HEAVY_WAIT_S", 15.0),
        search_cache_mib=_int("SEARCH_CACHE_MIB", 640),
        require_real_api_key=_bool("REQUIRE_REAL_API_KEY", False),
    )
    return s.validate()
