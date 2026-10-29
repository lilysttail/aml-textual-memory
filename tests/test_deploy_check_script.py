"""scripts/deploy_check.py — the read-back that proves the meaning-based path is on after a deployment.

The check is only worth something if it can tell the two cases apart, so the tests pin both the logic
(with canned answers) and the premise it rests on (a keyword-only service really returns nothing for the
smoke question, against the real keyword index).
"""
import importlib.util
import os
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("deploy_check", os.path.join(ROOT, "scripts", "deploy_check.py"))
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

KEY = "check-api-key-NOT-A-REAL-KEY-789"  # gate:allow-secret
GOOD_CONFIG = {"recall_paths": ["keyword", "dense"], "embed_provider": "dashscope", "rerank_enabled": False,
               "stem_english": False, "neighbour_candidates": True, "dashscope_dim": 1024, "embed_max_concurrency": 64}


def _health(**ov):
    cfg = dict(GOOD_CONFIG)
    cfg.update(ov)
    return {"ok": True, "config": cfg, "build": "0123456789ab"}


# ---- check_health ----

def test_health_with_the_decided_configuration_has_no_problem():
    assert dc.check_health(_health()) == []


@pytest.mark.parametrize("ov,needle", [
    ({"recall_paths": ["keyword"]}, "dense"),
    ({"embed_provider": "local"}, "embed_provider"),
    ({"rerank_enabled": True}, "rerank_enabled"),
    ({"stem_english": True}, "stem_english"),
    ({"embed_max_concurrency": 2}, "embed_max_concurrency"),         # a host variable that put the gate back to the code default
    ({"embed_max_concurrency": 128}, "embed_max_concurrency"),       # 64 is the acceptance value, not a minimum
])
def test_health_with_a_wrong_setting_names_it(ov, needle):
    problems = dc.check_health(_health(**ov))
    assert problems and any(needle in p for p in problems), problems


@pytest.mark.parametrize("build", [None, "", "short", "0123456789AB", "0123456789abc", 123456789012])
def test_a_service_without_a_usable_build_fingerprint_is_a_problem(build):
    body = _health()
    body["build"] = build
    problems = dc.check_health(body)
    assert problems and any("build" in p for p in problems), problems


def test_a_service_with_no_build_key_at_all_is_a_problem():
    body = _health()
    del body["build"]
    assert any("build" in p for p in dc.check_health(body))


def test_an_old_service_without_a_config_section_is_a_problem():
    problems = dc.check_health({"ok": True})
    assert problems and "older version" in problems[0]


def test_a_health_that_is_not_ok_is_a_problem():
    assert dc.check_health({"ok": False})
    assert dc.check_health("<html>")


# ---- check_search ----

def test_search_with_the_dog_memory_first_passes():
    assert dc.check_search([{"content": "[2026-10-26 09:00] user: I adopted a rescue greyhound named Biscuit last spring."},
                            {"content": "[2026-10-26 09:00] user: The lemon cake needs three eggs."}]) == []


def test_search_with_nothing_says_the_meaning_path_is_not_contributing():
    problems = dc.check_search([])
    assert problems and "meaning-based" in problems[0]


def test_search_with_another_memory_first_fails():
    problems = dc.check_search([{"content": "[x] user: The lemon cake needs three eggs."},
                                {"content": "[x] user: I adopted a rescue greyhound."}])
    assert problems and "not the memory about the dog" in problems[0]


# ---- run() end to end against a canned service ----

def _canned(health, hits, add_status=200, search_status=200):
    seen = []

    def http(method, path, body, key):
        seen.append((method, path, body, key))
        if path == "/health":
            return 200, health
        if path == "/add":
            return add_status, {"success": True}
        return search_status, {"data": hits}
    http.seen = seen
    return http


def test_run_passes_on_a_good_service_and_asks_with_top_k_100():
    hits = [{"content": "[x] user: I adopted a rescue greyhound named Biscuit last spring."}]
    http = _canned(_health(), hits)
    problems, lines = dc.run("https://x.invalid", KEY, http)
    assert problems == []
    search = [s for s in http.seen if s[1] == "/search"][0]
    assert search[2]["top_k"] == 100 and search[2]["query"] == dc.QUERY and search[3] == KEY
    add = [s for s in http.seen if s[1] == "/add"][0]
    assert len(add[2]["messages"]) == 4 and add[2]["user_id"].startswith("deploy-check-")


def test_run_prints_the_dense_index_settings_without_judging_them():
    """The dense index's read-back values are printed (one line) and never make a problem: CPUs differ by host."""
    hits = [{"content": "[x] user: I adopted a rescue greyhound named Biscuit last spring."}]
    odd = {"search_heavy_concurrency": 7, "search_heavy_wait_s": 1.0, "dense_query_t": 0.5, "search_cache_mib": 1,
           "numpy_threads": {"OPENBLAS_NUM_THREADS": "28"}}
    problems, lines = dc.run("https://x.invalid", KEY, _canned(_health(**odd), hits))
    assert problems == []
    line = [l for l in lines if "dense index (as it took effect)" in l]
    assert len(line) == 1 and "search_heavy_concurrency=7" in line[0] and '"OPENBLAS_NUM_THREADS": "28"' in line[0], lines


def test_run_stops_before_writing_when_the_configuration_is_wrong():
    http = _canned(_health(recall_paths=["keyword"]), [])
    problems, _ = dc.run("https://x.invalid", KEY, http)
    assert problems and not [s for s in http.seen if s[1] in ("/add", "/search")]


def test_run_reports_http_errors_instead_of_crashing():
    problems, _ = dc.run("https://x.invalid", KEY, _canned(_health(), [], add_status=503))
    assert problems and "503" in problems[0]
    problems, _ = dc.run("https://x.invalid", KEY, _canned(_health(), [], search_status=500))
    assert problems and "500" in problems[0]


def test_run_never_prints_the_key():
    hits = [{"content": "[x] user: I adopted a rescue greyhound named Biscuit last spring."}]
    problems, lines = dc.run("https://x.invalid", KEY, _canned(_health(), hits))
    assert KEY not in "\n".join(lines + problems)


# ---- the premise, against the real keyword index ----

def _settings(**ov):
    d = dict(db_path=os.path.join(_TMPROOT, f"dk_{uuid.uuid4().hex}.db"),
             api_key=KEY, recall_paths=("keyword",),
             rerank_enabled=False, rerank_n=200, store_busy_timeout_ms=5000,
             store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32,
             embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
             bge_small_dir=None, neighbour_candidates=True,
             dense_backfill_m=8, dense_backfill_t=2.0)
    d.update(ov)
    return Settings(**d)


def _through(client):
    def http(method, path, body, key):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = client.request(method, path, json=body, headers=headers)
        return r.status_code, r.json()
    return http


def test_a_keyword_only_service_is_caught_by_the_configuration_check():
    with TestClient(app_main.make_app(_settings())) as client:
        problems, lines = dc.run("http://localhost", KEY, _through(client))
    assert problems and any("dense" in p for p in problems)


def test_keywords_alone_find_nothing_for_the_smoke_question():
    """The premise: the question shares no content word with the memory it needs, so an empty answer really means
    the meaning-based path contributed nothing."""
    with TestClient(app_main.make_app(_settings())) as client:
        http = _through(client)
        user = "premise-user"
        status, _ = http("POST", "/add", {
            "request_id": "p1", "user_id": user, "session_id": "s1",
            "messages": [{"role": "user", "timestamp": 1728000000000 + i, "content": t}
                         for i, t in enumerate(dc.MEMORIES)]}, KEY)
        assert status == 200
        status, body = http("POST", "/search", {"query": dc.QUERY, "user_id": user, "top_k": 100}, KEY)
        assert status == 200 and body["data"] == []


# ---- the command line ----

def test_main_refuses_a_plain_http_address_and_a_missing_key(monkeypatch, capsys):
    monkeypatch.delenv(dc.KEY_ENV, raising=False)
    assert dc.main(["--base", "http://example.com"]) == 1
    assert "https" in capsys.readouterr().out
    assert dc.main(["--base", "https://example.com"]) == 1
    assert dc.KEY_ENV in capsys.readouterr().out


def test_main_does_not_take_the_key_from_an_argument(capsys):
    with pytest.raises(SystemExit):
        dc.main(["--base", "https://example.com", "--key", "x"])


# ---- every network failure becomes a readable FAIL with a reason code, never a traceback ----

import socket  # noqa: E402
import ssl  # noqa: E402
import urllib.error  # noqa: E402


class _Reply:
    def __init__(self, status, body):
        self.status, self._body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _urlopen_raising(exc):
    def fake(req, timeout=None):
        raise exc
    return fake


@pytest.mark.parametrize("exc,code", [
    (urllib.error.URLError("name resolution failed: secret-host.example"), "network_unreachable"),
    (urllib.error.URLError(ssl.SSLError("handshake failure")), "tls_error"),
    (ssl.SSLError("handshake failure"), "tls_error"),
    (TimeoutError("timed out"), "timeout"),
    (socket.timeout("timed out"), "timeout"),
    (urllib.error.URLError(socket.timeout("timed out")), "timeout"),
    (ConnectionRefusedError("refused"), "network_unreachable"),
    (ConnectionResetError("reset"), "network_unreachable"),
    (OSError("something else"), "network_unreachable"),
])
def test_a_network_failure_becomes_a_reason_code_without_the_details(monkeypatch, exc, code):
    monkeypatch.setattr(dc.urllib.request, "urlopen", _urlopen_raising(exc))
    with pytest.raises(dc.TransportFailure) as e:
        dc._urllib_http("https://example.invalid")("GET", "/health", None, None)
    assert e.value.code == code
    assert "secret-host" not in str(e.value) and "example.invalid" not in str(e.value)


@pytest.mark.parametrize("body", [b"<html>bad gateway</html>", b"", b"\xff\xfe\x00"])
def test_a_reply_that_is_not_json_becomes_a_reason_code(monkeypatch, body):
    monkeypatch.setattr(dc.urllib.request, "urlopen", lambda req, timeout=None: _Reply(200, body))
    with pytest.raises(dc.TransportFailure) as e:
        dc._urllib_http("https://example.invalid")("GET", "/health", None, None)
    assert e.value.code == "not_json"


def test_an_http_error_with_a_non_json_body_is_still_returned_as_a_status(monkeypatch):
    import io
    err = urllib.error.HTTPError("https://example.invalid/x", 502, "Bad Gateway", {}, io.BytesIO(b"<html>bad gateway</html>"))
    monkeypatch.setattr(dc.urllib.request, "urlopen", _urlopen_raising(err))
    status, body = dc._urllib_http("https://example.invalid")("GET", "/health", None, None)
    assert status == 502 and isinstance(body, str)


@pytest.mark.parametrize("fail_at", ["/health", "/add", "/search"])
def test_run_turns_a_transport_failure_at_any_step_into_a_problem_with_the_code(fail_at):
    hits = [{"content": "[x] user: I adopted a rescue greyhound named Biscuit last spring."}]
    inner = _canned(_health(), hits)

    def http(method, path, body, key):
        if path == fail_at:
            raise dc.TransportFailure("timeout")
        return inner(method, path, body, key)
    problems, lines = dc.run("https://x.invalid", KEY, http)
    assert problems and "timeout" in problems[0] and KEY not in "\n".join(problems + lines)


def test_main_prints_fail_with_the_code_and_exits_non_zero_instead_of_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv(dc.KEY_ENV, KEY)
    monkeypatch.setattr(dc.urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("dns")))
    assert dc.main(["--base", "https://example.invalid"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "network_unreachable" in out and "Traceback" not in out and KEY not in out


def test_any_other_exception_is_still_a_readable_fail_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv(dc.KEY_ENV, KEY)
    monkeypatch.setattr(dc, "_urllib_http", lambda base: (_ for _ in ()).throw(RuntimeError("boom with a detail")))
    assert dc.main(["--base", "https://example.invalid"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "RuntimeError" in out and "boom with a detail" not in out and "Traceback" not in out


# ---- the gate places are part of the acceptance read-back ----

def test_a_real_service_whose_gate_was_put_back_to_the_default_fails_the_read_back(monkeypatch):
    class _Null:
        model_name, dim = "null-v1", 8

        def embed(self, texts, timeout_s=None):
            return [[0.0] * 8 for _ in texts]
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: _Null())
    ov = dict(recall_paths=("keyword", "dense"), embed_provider="dashscope", dashscope_api_key="x" * 12, embed_max_concurrency=2)
    with TestClient(app_main.make_app(_settings(**ov))) as client:
        problems, _ = dc.run("http://localhost", KEY, _through(client))
    assert problems and any("embed_max_concurrency" in p for p in problems)
