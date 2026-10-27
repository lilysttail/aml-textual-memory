"""EMBED_PROVIDER / DASHSCOPE_* settings, written RED first.

config.py is the ONLY module that reads environment variables. The key is read
only by its env-var NAME (DASHSCOPE_API_KEY); a missing key with
EMBED_PROVIDER=dashscope fails at startup with the variable name in the error.
The key never shows up in repr(Settings).
"""
import pytest

from app import config
from app.config import Settings

KEY = "test-key-NOT-A-REAL-KEY-123"

ENV_NAMES = ("EMBED_PROVIDER", "DASHSCOPE_API_KEY", "DASHSCOPE_DIM", "DASHSCOPE_BATCH",
             "DASHSCOPE_TIMEOUT_S", "DASHSCOPE_MAX_TOKENS", "DASHSCOPE_BASE_URL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for n in ENV_NAMES:
        monkeypatch.delenv(n, raising=False)


def test_defaults_keep_local_provider_and_documented_dashscope_values():
    s = config.load()
    assert s.embed_provider == "local"
    assert s.dashscope_dim == 1024 and s.dashscope_batch == 10
    assert s.dashscope_timeout_s == 20.0 and s.dashscope_max_tokens == 2000
    # the owner uses the international (Singapore) console, so that is the default; the China host is an override
    assert s.dashscope_base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert s.dashscope_api_key is None


def test_dashscope_without_key_fails_at_startup_naming_the_variable(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "dashscope")
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        config.load()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        config.load()


def test_dashscope_with_key_loads_and_key_is_not_in_repr(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", KEY)
    s = config.load()
    assert s.embed_provider == "dashscope" and s.dashscope_api_key == KEY
    assert KEY not in repr(s) and KEY not in str(s)


def test_local_provider_does_not_need_or_require_a_key(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "local")
    assert config.load().embed_provider == "local"


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    with pytest.raises(ValueError, match="EMBED_PROVIDER"):
        config.load()


@pytest.mark.parametrize("name,val,needle", [
    ("DASHSCOPE_DIM", "100", "DASHSCOPE_DIM"),
    ("DASHSCOPE_DIM", "abc", "DASHSCOPE_DIM"),
    ("DASHSCOPE_BATCH", "0", "DASHSCOPE_BATCH"),
    ("DASHSCOPE_BATCH", "11", "DASHSCOPE_BATCH"),
    ("DASHSCOPE_TIMEOUT_S", "0", "DASHSCOPE_TIMEOUT_S"),
    ("DASHSCOPE_TIMEOUT_S", "-1", "DASHSCOPE_TIMEOUT_S"),
    ("DASHSCOPE_MAX_TOKENS", "0", "DASHSCOPE_MAX_TOKENS"),
])
def test_bad_dashscope_numbers_fail_fast(monkeypatch, name, val, needle):
    monkeypatch.setenv(name, val)
    with pytest.raises(ValueError, match=needle):
        config.load()


def test_valid_overrides_are_honoured(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_DIM", "512")
    monkeypatch.setenv("DASHSCOPE_BATCH", "5")
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_S", "7.5")
    monkeypatch.setenv("DASHSCOPE_MAX_TOKENS", "1500")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")   # the China-site host
    s = config.load()
    assert (s.dashscope_dim, s.dashscope_batch, s.dashscope_timeout_s, s.dashscope_max_tokens) == (512, 5, 7.5, 1500)
    assert s.dashscope_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_max_tokens_is_capped_at_8192_the_services_own_limit(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_MAX_TOKENS", "8192")
    assert config.load().dashscope_max_tokens == 8192
    monkeypatch.setenv("DASHSCOPE_MAX_TOKENS", "8193")
    with pytest.raises(ValueError, match="DASHSCOPE_MAX_TOKENS"):
        config.load()
    monkeypatch.setenv("DASHSCOPE_MAX_TOKENS", "9000")
    with pytest.raises(ValueError, match="DASHSCOPE_MAX_TOKENS"):
        config.load()


def test_old_style_settings_construction_still_works():
    """Existing tests build Settings(...) without the new fields; they keep defaults."""
    s = Settings(db_path="x.db", api_key="k", recall_paths=("keyword",), rerank_enabled=False,
                 rerank_n=200, store_busy_timeout_ms=5000,
                 store_write_lock_timeout_ms=10000, embed_max_concurrency=2, embed_batch=32,
                 embed_backfill_wait_s=60.0, embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    assert s.validate() is s and s.embed_provider == "local"
