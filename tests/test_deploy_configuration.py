"""The deployed configuration is the configuration the answer ruler decided on.

The ruler's last locked run (step 4) started the service with: RECALL_PATHS=keyword,dense,
EMBED_PROVIDER=dashscope, RERANK_ENABLED=0, STEM_ENGLISH=0, NEIGHBOUR_CANDIDATES=1, and every
other setting at its code default (eval/ruler/dense.py `service_env`). Three things must hold
for the image built from this repository:

  1. the Dockerfile carries exactly that configuration (so a missed variable on the host
     cannot quietly turn the meaning-based path off);
  2. the image holds no key and refuses to start without a real API_KEY / DASHSCOPE_API_KEY,
     naming the variable — while local runs and the test suite are untouched (no key needed,
     keyword + local embedder by default);
  3. GET /health reports the running configuration (names only, never a key), so a read-back
     after a deployment can prove the meaning-based path is on.
"""
import os
import re
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config, main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_API_KEY = "deploy-api-key-NOT-A-REAL-KEY-456"
FAKE_DASHSCOPE_KEY = "deploy-dashscope-NOT-A-REAL-KEY-123"

# What the locked run measured. If one of these has to change, the ruler has to be run again.
CONCLUSION = {
    "recall_paths": ("keyword", "dense"),
    "embed_provider": "dashscope",
    "rerank_enabled": False,
    "stem_english": False,
    "neighbour_candidates": True,
    "dashscope_dim": 1024,
    "dashscope_batch": 10,
    "dashscope_timeout_s": 20.0,
    "dashscope_max_tokens": 2000,
    # The one value that is NOT the measured default (2). The search path's vector gate is non-blocking: a search
    # that finds no free place gives up its meaning-based half at once. At 16 / 32 simultaneous searches with the
    # default 2 places, 610 / 602 of 648 searches did (measured cold, real provider calls). Places only decide
    # whether a search is skipped, never how results are ranked, so the serial locked-run conclusion still holds.
    "embed_max_concurrency": 64,
    "embed_batch": 32,
    "embed_backfill_wait_s": 60.0,
    "embed_acquire_s": 30.0,
    "dense_backfill_m": 8,
    "dense_backfill_t": 2.0,
    "content_max_chars": 4000,
}


def _config_env_names():
    """Every environment variable app/config.py reads (found in its source, so a new one is covered too)."""
    with open(os.path.join(ROOT, "app", "config.py"), encoding="utf-8") as f:
        src = f.read()
    return sorted(set(re.findall(r'_(?:get|int|float|bool|paths)\(\s*"([A-Z_0-9]+)"', src)))


def _dockerfile_env():
    """The variables the Dockerfile's ENV lines set (continuation lines joined)."""
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read().replace("\\\r\n", " ").replace("\\\n", " ")
    env = {}
    for m in re.finditer(r"^ENV\s+(.+)$", text, re.MULTILINE):
        for pair in m.group(1).split():
            name, sep, value = pair.partition("=")
            assert sep, f"unsupported ENV form in the Dockerfile: {pair!r}"
            env[name] = value.strip('"')
    return env


@pytest.fixture
def clean_env(monkeypatch):
    for n in _config_env_names():
        monkeypatch.delenv(n, raising=False)
    return monkeypatch


def _as_deployed(monkeypatch, api_key=FAKE_API_KEY, dashscope_key=FAKE_DASHSCOPE_KEY, drop=()):
    """The environment inside the image, plus what the host adds (the two keys)."""
    for k, v in _dockerfile_env().items():
        if k not in drop:
            monkeypatch.setenv(k, v)
    if api_key is not None:
        monkeypatch.setenv("API_KEY", api_key)
    if dashscope_key is not None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", dashscope_key)


# ---- 1. the image carries the decided configuration ----

def test_the_image_configuration_is_the_configuration_the_ruler_decided_on(clean_env):
    _as_deployed(clean_env)
    s = config.load()
    got = {k: getattr(s, k) for k in CONCLUSION}
    assert got == CONCLUSION
    assert s.require_real_api_key is True


def test_the_dockerfile_sets_the_switches_itself_not_through_code_defaults(clean_env):
    """Each decided switch is written in the Dockerfile: a changed code default cannot change the deployment silently."""
    env = _dockerfile_env()
    for name, value in (("RECALL_PATHS", "keyword,dense"), ("EMBED_PROVIDER", "dashscope"), ("RERANK_ENABLED", "0"),
                        ("STEM_ENGLISH", "0"), ("NEIGHBOUR_CANDIDATES", "1"), ("REQUIRE_REAL_API_KEY", "1"),
                        ("EMBED_MAX_CONCURRENCY", "64")):
        assert env.get(name) == value, f"Dockerfile ENV {name} should be {value!r}, found {env.get(name)!r}"


# ---- 2. no key in the image; a missing key stops the start and names the variable ----

def test_the_image_holds_no_key():
    env = _dockerfile_env()
    assert "API_KEY" not in env and "DASHSCOPE_API_KEY" not in env, \
        "a key (or a placeholder for one) in the image is the hole this closes"
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        code = [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#")]
    assert not re.search(r"(^|\s)(DASHSCOPE_)?API_KEY\b", "\n".join(code)), \
        "the Dockerfile's instructions must not mention a key variable (comments may)"


def test_the_image_without_a_dashscope_key_refuses_to_start_naming_it(clean_env):
    _as_deployed(clean_env, dashscope_key=None)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        config.load()


@pytest.mark.parametrize("bad", [None, "", "   ", "change-me", "dev-key-change-me"])
def test_the_image_without_a_real_api_key_refuses_to_start_naming_it(clean_env, bad):
    _as_deployed(clean_env, api_key=bad)
    if bad is not None:
        clean_env.setenv("API_KEY", bad)
    with pytest.raises(ValueError, match=r"\bAPI_KEY\b"):
        config.load()


def test_the_image_with_a_real_api_key_starts_and_the_key_is_not_in_repr(clean_env):
    _as_deployed(clean_env)
    s = config.load()
    assert s.api_key == FAKE_API_KEY
    assert FAKE_API_KEY not in repr(s) and FAKE_DASHSCOPE_KEY not in repr(s)


# ---- local runs and the suite are untouched ----

def test_local_defaults_need_no_key_and_stay_keyword_plus_local(clean_env):
    s = config.load()
    assert s.recall_paths == ("keyword",)
    assert s.embed_provider == "local"
    assert s.rerank_enabled is False and s.stem_english is False
    assert s.api_key == "dev-key-change-me"
    assert s.require_real_api_key is False
    assert s.embed_max_concurrency == 2          # the raised gate is the image's, a local run keeps the small one


def test_the_development_key_is_fine_when_the_flag_is_off(clean_env):
    clean_env.setenv("API_KEY", "change-me")
    assert config.load().api_key == "change-me"


def test_the_flag_is_a_strict_boolean(clean_env):
    clean_env.setenv("REQUIRE_REAL_API_KEY", "maybe")
    with pytest.raises(ValueError, match="REQUIRE_REAL_API_KEY"):
        config.load()


# ---- 3. /health reports the running configuration ----

def _settings(**ov):
    d = dict(db_path=os.path.join(_TMPROOT, f"dc_{uuid.uuid4().hex}.db"),
             api_key=FAKE_API_KEY, recall_paths=("keyword", "dense"),
             rerank_enabled=False, rerank_n=200, store_busy_timeout_ms=5000,
             store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32,
             embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
             bge_small_dir=None, neighbour_candidates=True,
             dense_backfill_m=8, dense_backfill_t=2.0,
             embed_provider="dashscope", dashscope_api_key=FAKE_DASHSCOPE_KEY)
    d.update(ov)
    return Settings(**d)


class _NullEmbedder:
    model_name, dim = "null-v1", 8

    def embed(self, texts, timeout_s=None):
        return [[0.0] * 8 for _ in texts]


@pytest.fixture
def null_embedder(monkeypatch):
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: _NullEmbedder())


def test_health_reports_the_running_configuration(null_embedder):
    with TestClient(app_main.make_app(_settings())) as client:
        body = client.get("/health").json()
    assert body["ok"] is True and isinstance(body["pending_vectors"], int)
    assert body["config"] == {
        "recall_paths": ["keyword", "dense"], "embed_provider": "dashscope", "rerank_enabled": False,
        "stem_english": False, "neighbour_candidates": True, "dashscope_dim": 1024, "embed_max_concurrency": 2}


def test_health_reports_the_gate_places_the_core_really_runs_with(null_embedder):
    """The number comes from the running core, so a platform variable that overrides the image shows up in the read-back."""
    for places in (64, 2):
        with TestClient(app_main.make_app(_settings(embed_max_concurrency=places))) as client:
            assert client.get("/health").json()["config"]["embed_max_concurrency"] == places


@pytest.mark.parametrize("blank", ["   ", "\t \t"])
def test_a_blank_dashscope_key_refuses_to_start_naming_it_and_never_echoing_it(clean_env, blank):
    _as_deployed(clean_env, dashscope_key=blank)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY") as e:
        config.load()
    assert "\t" not in str(e.value) and blank not in str(e.value)


def test_a_blank_dashscope_key_is_refused_by_the_settings_check_itself(null_embedder):
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _settings(dashscope_api_key="  ").validate()


def test_health_tells_keyword_only_from_keyword_plus_dense(null_embedder):
    """The read-back is only worth something if the answer differs between the two."""
    ov = dict(recall_paths=("keyword",), embed_provider="local", dashscope_api_key=None)
    with TestClient(app_main.make_app(_settings(**ov))) as client:
        cfg = client.get("/health").json()["config"]
    assert cfg["recall_paths"] == ["keyword"] and cfg["embed_provider"] == "local"


def test_health_never_shows_a_key_or_an_address(null_embedder):
    with TestClient(app_main.make_app(_settings())) as client:
        text = client.get("/health").text
    for secret in (FAKE_API_KEY, FAKE_DASHSCOPE_KEY, "dashscope-intl", "http"):
        assert secret not in text, f"/health leaked {secret!r}"
