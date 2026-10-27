#!/usr/bin/env python3
"""Read-back after a deployment: is the meaning-based (dense) path really on?

"The service answers" proves nothing: with the dense path off it answers too, from keywords only. Two checks:

  1. GET /health reports the configuration the service is RUNNING with (names only, never a key):
     the recall paths must include `dense`, the embedder must be `dashscope`, rerank and stemming off.
  2. A search that keywords cannot answer: four memories are written under a throw-away user, then one
     question that shares no content word with the memory it needs ("Which dog did I take in from a
     shelter?" vs. "I adopted a rescue greyhound ..."). Keywords find nothing; the meaning-based path
     ranks that memory first.

Usage (the key is read from the environment, never from an argument, and is never printed):

    set AML_API_KEY=<the service's API key>            (Windows cmd)
    $env:AML_API_KEY = "<the service's API key>"       (PowerShell)
    python scripts/deploy_check.py --base https://<the deployed address>

Exit code 0 = both checks passed; 1 = something is wrong (the lines above it say what).
The four memories stay in the live database under the throw-away user (`deploy-check-...`); they belong to
no evaluation user. `scripts/wipe.py` clears the whole database after the evaluation anyway.
"""
import argparse
import json
import os
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

KEY_ENV = "AML_API_KEY"  # gate:allow-secret

MEMORIES = [
    "I adopted a rescue greyhound named Biscuit last spring.",
    "I filed my tax return two weeks before the deadline.",
    "We hiked the ridge trail on Saturday and it rained the whole time.",
    "The lemon cake needs three eggs and a spoonful of baking soda.",
]
QUERY = "Which dog did I take in from a shelter?"
EXPECTED_WORD = "greyhound"
# `embed_max_concurrency` is the acceptance value of this deployment, not a performance hint: a search never waits for a place at the
# embedding gate, so with fewer places (a host variable can override the image's) simultaneous searches silently lose the meaning-based
# half while the single smoke question below still passes. Read back from the running core, so an override shows up here.
WANT_CONFIG = {"embed_provider": "dashscope", "rerank_enabled": False, "stem_english": False, "embed_max_concurrency": 64}


class TransportFailure(Exception):
    """The service could not be reached, or its reply could not be read. Carries a reason CODE only (never the exception text, which
    can hold an address or a header): network_unreachable / timeout / tls_error / not_json."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def check_health(body):
    """The problems found in a /health body (empty list = the running configuration is the decided one)."""
    if not isinstance(body, dict) or body.get("ok") is not True:
        return [f"/health did not answer ok: {body!r}"]
    cfg = body.get("config")
    if not isinstance(cfg, dict):
        return ["/health has no `config` section: an older version of the service is deployed"]
    problems = []
    paths = cfg.get("recall_paths")
    if not isinstance(paths, list) or "dense" not in paths:
        problems.append(f"recall_paths is {paths!r}: the meaning-based path (`dense`) is NOT on")
    for name, want in WANT_CONFIG.items():
        if cfg.get(name) != want:
            problems.append(f"{name} is {cfg.get(name)!r}, should be {want!r}")
    build = body.get("build")
    if not (isinstance(build, str) and re.fullmatch(r"[0-9a-f]{12}", build)):
        problems.append("`build` is missing or is not a 12-character fingerprint: this image carries no content fingerprint "
                        "(an older version, or not built from the Dockerfile), so scripts/verify_deploy.py cannot check it")
    return problems


def check_search(data):
    """The problems found in the `data` list of the smoke search."""
    if not data:
        return ["the search returned nothing: keywords cannot answer this question, so the "
                "meaning-based path is not contributing (not running, or its embedding calls fail)"]
    first = data[0].get("content", "")
    if EXPECTED_WORD not in first.lower():
        return [f"the first hit is not the memory about the dog: {first!r} — the meaning-based path "
                "ran but did not rank it first; look at the service log for embedding errors"]
    return []


def run(base, key, http):
    """Run both checks through `http(method, path, body, key) -> (status, json)`. Returns (problems, lines).
    A request that cannot be made or read (`TransportFailure`) ends the run with a problem that names the reason code."""
    lines, problems = [], []
    try:
        _steps(key, http, lines, problems)
    except TransportFailure as e:
        problems.append(f"could not reach the service or read its reply ({e.code}); nothing after this point was checked")
    return problems, lines


def _steps(key, http, lines, problems):
    status, health = http("GET", "/health", None, None)
    lines.append(f"GET /health -> {status}")
    if status != 200:
        problems.append(f"/health answered HTTP {status}")
    else:
        found = check_health(health)
        problems += found
        if isinstance(health, dict) and isinstance(health.get("config"), dict):
            lines.append("  running configuration: " + json.dumps(health["config"], sort_keys=True))
        if found:
            return                                       # the memories below would prove nothing on a wrong config

    user = f"deploy-check-{uuid.uuid4().hex[:12]}"
    now_ms = int(time.time() * 1000)
    status, body = http("POST", "/add", {
        "request_id": f"{user}-add", "user_id": user, "session_id": f"{user}-s",
        "messages": [{"role": "user", "timestamp": now_ms + i, "content": text}
                     for i, text in enumerate(MEMORIES)]}, key)
    lines.append(f"POST /add (4 memories, user {user}) -> {status}")
    if status != 200:
        problems.append(f"/add answered HTTP {status}: {body!r}")
        return

    status, body = http("POST", "/search", {"query": QUERY, "user_id": user, "top_k": 100}, key)
    lines.append(f"POST /search (top_k 100) -> {status}")
    if status != 200:
        problems.append(f"/search answered HTTP {status}: {body!r}")
        return
    data = body.get("data", []) if isinstance(body, dict) else []
    lines.append(f"  hits: {len(data)}; first: {data[0].get('content') if data else None!r}")
    problems += check_search(data)

    status, health = http("GET", "/health", None, None)
    if status == 200 and isinstance(health, dict):
        lines.append(f"  pending_vectors after the search: {health.get('pending_vectors')} (information only)")
    return


def _urllib_http(base):
    def http(method, path, body, key):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(base + path, method=method, headers=headers,
                                     data=None if body is None else json.dumps(body).encode("utf-8"))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:                 # an HTTP status is an answer, however the body looks
            try:
                raw = e.read().decode("utf-8", "replace")
            except OSError:
                raw = ""
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw[:200]
        except (TimeoutError, socket.timeout):
            raise TransportFailure("timeout") from None
        except ssl.SSLError:
            raise TransportFailure("tls_error") from None
        except urllib.error.URLError as e:                  # DNS, refused, unreachable... the reason may hide a timeout or a TLS error
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise TransportFailure("timeout") from None
            if isinstance(e.reason, ssl.SSLError):
                raise TransportFailure("tls_error") from None
            raise TransportFailure("network_unreachable") from None
        except OSError:                                     # refused / reset / aborted while connecting or reading
            raise TransportFailure("network_unreachable") from None
        except ValueError:                                  # a 200 whose body is not JSON (or not UTF-8)
            raise TransportFailure("not_json") from None
    return http


def _check_base(base):
    u = urlparse(base)
    if u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1")):
        return None
    return f"--base must be an https address (the API key travels in the request); got {base!r}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="the deployed address, e.g. https://example.app")
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    bad = _check_base(base)
    if bad:
        print("FAIL " + bad)
        return 1
    key = os.environ.get(KEY_ENV, "")
    if not key:
        print(f"FAIL set the environment variable {KEY_ENV} to the service's API key first (it is never taken from an argument)")
        return 1
    try:
        problems, lines = run(base, key, _urllib_http(base))
    except Exception as e:                                  # a script bug must still end as a readable FAIL, never a traceback
        print(f"FAIL the check itself broke ({type(e).__name__}); nothing was concluded")
        return 1
    print("\n".join(lines))
    if problems:
        print("\nFAIL")
        for p in problems:
            print("  - " + p)
        return 1
    print("\nPASS: the running configuration is the decided one, and the meaning-based path answered a question keywords cannot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
