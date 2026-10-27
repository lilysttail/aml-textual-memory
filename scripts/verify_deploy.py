#!/usr/bin/env python3
"""Does the running service carry the published version?

The image stores a content fingerprint (scripts/tree_digest.py) and /health reports it as `build`. This script clones the published
tag, computes the fingerprint with the clone's own scripts/tree_digest.py (the code the image ran when it was built), asks the live
/health for `build`, and compares. One line of output:

    MATCH build=<12 hex> tag=<tag>
    MISMATCH live=<12 hex> tag_build=<12 hex> tag=<tag>
    FAIL <reason code> tag=<tag>

Reason codes: clone_failed, digest_failed, health_not_ok, no_build_field, network_unreachable, timeout, tls_error, not_json.
Nothing from the network or from git is echoed. Exit code 0 only for MATCH. It reads only: it clones into a temporary directory
(removed afterwards) and sends one GET to /health; no key is needed.

    python scripts/verify_deploy.py --base https://<the deployed address> [--repo URL] [--tag submission-1]
"""
import argparse
import json
import os
import re
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlparse

DEFAULT_REPO = "https://github.com/lilysttail/aml-textual-memory"
DEFAULT_TAG = "submission-1"
_HEX12 = re.compile(r"[0-9a-f]{12}")


class TransportFailure(Exception):
    """The live service could not be reached or its reply could not be read. Carries a reason CODE only."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


class _Fail(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _clone(repo_url, tag, dest):
    r = subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", tag, repo_url, dest], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    if r.returncode != 0:
        raise _Fail("clone_failed")


def _digest(clone_dir):
    script = os.path.join(clone_dir, "scripts", "tree_digest.py")
    if not os.path.isfile(script):
        raise _Fail("digest_failed")
    r = subprocess.run([sys.executable, script, "--root", clone_dir], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not _HEX12.fullmatch(out):
        raise _Fail("digest_failed")
    return out


def _rmtree(path):
    def onerror(func, p, _exc):                        # git's object files are read-only on Windows
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=onerror)


def verify(base, repo_url, tag, http):
    """Returns (verdict, line). `http(method, path, body, key) -> (status, json)`."""
    dest = tempfile.mkdtemp(prefix="verify_deploy_")
    try:
        try:
            _clone(repo_url, tag, os.path.join(dest, "clone"))
            built = _digest(os.path.join(dest, "clone"))
            status, body = http("GET", "/health", None, None)
        except _Fail as e:
            return "FAIL", f"FAIL {e.code} tag={tag}"
        except TransportFailure as e:
            return "FAIL", f"FAIL {e.code} tag={tag}"
        if status != 200 or not isinstance(body, dict) or body.get("ok") is not True:
            return "FAIL", f"FAIL health_not_ok tag={tag}"
        live = body.get("build")
        if not isinstance(live, str) or not _HEX12.fullmatch(live):
            return "FAIL", f"FAIL no_build_field tag={tag}"
        if live == built:
            return "MATCH", f"MATCH build={built} tag={tag}"
        return "MISMATCH", f"MISMATCH live={live} tag_build={built} tag={tag}"
    finally:
        _rmtree(dest)


def _urllib_get(base):
    def http(method, path, body, key):
        req = urllib.request.Request(base + path, method=method, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:               # an HTTP status is an answer, however the body looks
            try:
                return e.code, json.loads(e.read().decode("utf-8", "replace"))
            except (ValueError, OSError):
                return e.code, None
        except (TimeoutError, socket.timeout):
            raise TransportFailure("timeout") from None
        except ssl.SSLError:
            raise TransportFailure("tls_error") from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise TransportFailure("timeout") from None
            if isinstance(e.reason, ssl.SSLError):
                raise TransportFailure("tls_error") from None
            raise TransportFailure("network_unreachable") from None
        except OSError:
            raise TransportFailure("network_unreachable") from None
        except ValueError:
            raise TransportFailure("not_json") from None
    return http


def _base_ok(base):
    u = urlparse(base)
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="the deployed address, e.g. https://example.app")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="the published repository (URL or path)")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    if not _base_ok(base):
        print("FAIL base_must_be_https")
        return 1
    try:
        verdict, line = verify(base, args.repo, args.tag, _urllib_get(base))
    except Exception as e:                                 # a script bug must still end as a readable line, never a traceback
        print(f"FAIL unexpected_{type(e).__name__} tag={args.tag}")
        return 1
    print(line)
    return 0 if verdict == "MATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
