"""scripts/verify_deploy.py — does the running service carry the published version?

It clones the published tag, works out the content fingerprint (with the clone's own scripts/tree_digest.py — the code the image
ran at build time), asks the live /health for `build`, and says MATCH / MISMATCH. Any failure to do so ends as a one-line FAIL with a
reason code, never a traceback; nothing from the network or from git is echoed.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("verify_deploy", os.path.join(ROOT, "scripts", "verify_deploy.py"))
vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vd)
_spec2 = importlib.util.spec_from_file_location("tree_digest_t", os.path.join(ROOT, "scripts", "tree_digest.py"))
td = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(td)


def _git(cwd, *args):
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args], cwd=cwd, capture_output=True,
                       text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _published_repo(tmp_path, eol="\n", tag="submission-1"):
    """A local repository that stands in for the published one: the files the digest covers (including its own tree_digest.py)."""
    repo = tmp_path / "published"
    files = {"Dockerfile": "FROM python:3.11-slim\n", "requirements.txt": "fastapi>=0.110\n", "app/main.py": "print('app')\n",
             "core/engine.py": "print('core')\n", "README.md": "# readme\n"}
    with open(os.path.join(ROOT, "scripts", "tree_digest.py"), "rb") as f:
        digest_script = f.read()
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.replace("\n", eol).encode("utf-8"))
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "scripts" / "tree_digest.py").write_bytes(digest_script)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    if tag:
        _git(repo, "tag", tag)
    return repo


def _http(build, status=200, body=None):
    def http(method, path, payload, key):
        assert method == "GET" and path == "/health"
        return status, body if body is not None else {"ok": True, "build": build}
    return http


def _digest_of(repo):
    return subprocess.run([sys.executable, str(repo / "scripts" / "tree_digest.py"), "--root", str(repo)], capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def test_match_when_the_live_build_is_the_tag_s_fingerprint(tmp_path):
    repo = _published_repo(tmp_path)
    verdict, line = vd.verify("https://x.invalid", str(repo), "submission-1", _http(_digest_of(repo)))
    assert verdict == "MATCH" and _digest_of(repo) in line and "submission-1" in line


def test_mismatch_when_the_live_build_differs(tmp_path):
    repo = _published_repo(tmp_path)
    verdict, line = vd.verify("https://x.invalid", str(repo), "submission-1", _http("000000000000"))
    assert verdict == "MISMATCH" and "000000000000" in line and _digest_of(repo) in line


def test_a_clone_with_crlf_line_endings_still_matches(tmp_path):
    """The deployed image's files and a Windows clone differ only in line endings: that is not a mismatch."""
    lf = _published_repo(tmp_path / "lf")
    crlf = _published_repo(tmp_path / "crlf", eol="\r\n")
    verdict, _ = vd.verify("https://x.invalid", str(crlf), "submission-1", _http(_digest_of(lf)))
    assert verdict == "MATCH"


def test_the_clone_is_made_from_the_tag_not_from_the_default_branch(tmp_path):
    repo = _published_repo(tmp_path)
    tagged = _digest_of(repo)
    (repo / "app" / "main.py").write_text("print('moved on after the tag')\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "later")
    verdict, _ = vd.verify("https://x.invalid", str(repo), "submission-1", _http(tagged))
    assert verdict == "MATCH"


@pytest.mark.parametrize("body,code", [({"ok": True}, "no_build_field"), ({"ok": True, "build": None}, "no_build_field"),
                                       ({"ok": True, "build": "short"}, "no_build_field"), ("<html>", "health_not_ok")])
def test_a_live_service_without_a_usable_build_field_is_a_fail_with_a_code(tmp_path, body, code):
    repo = _published_repo(tmp_path)
    verdict, line = vd.verify("https://x.invalid", str(repo), "submission-1", _http(None, body=body))
    assert verdict == "FAIL" and code in line


def test_a_health_that_is_not_200_is_a_fail(tmp_path):
    repo = _published_repo(tmp_path)
    verdict, line = vd.verify("https://x.invalid", str(repo), "submission-1", _http(None, status=503, body={"detail": "x"}))
    assert verdict == "FAIL" and "health_not_ok" in line


def test_a_tag_that_does_not_exist_is_a_clone_failure_not_a_traceback(tmp_path):
    repo = _published_repo(tmp_path)
    verdict, line = vd.verify("https://x.invalid", str(repo), "no-such-tag", _http("000000000000"))
    assert verdict == "FAIL" and "clone_failed" in line and "no-such-tag" not in line.replace("tag=no-such-tag", "")


def test_a_repository_without_the_digest_script_is_a_fail(tmp_path):
    repo = _published_repo(tmp_path)
    os.remove(repo / "scripts" / "tree_digest.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "drop")
    _git(repo, "tag", "-f", "submission-1")
    verdict, line = vd.verify("https://x.invalid", str(repo), "submission-1", _http("000000000000"))
    assert verdict == "FAIL" and "digest_failed" in line


def test_a_transport_failure_while_asking_the_live_service_is_a_fail_with_the_code(tmp_path):
    repo = _published_repo(tmp_path)

    def broken(method, path, payload, key):
        raise vd.TransportFailure("timeout")
    verdict, line = vd.verify("https://x.invalid", str(repo), "submission-1", broken)
    assert verdict == "FAIL" and "timeout" in line


def test_the_clone_directory_is_removed_afterwards(tmp_path, monkeypatch):
    repo = _published_repo(tmp_path)
    seen = []
    real = vd._clone

    def spy(repo_url, tag, dest):
        seen.append(dest)
        return real(repo_url, tag, dest)
    monkeypatch.setattr(vd, "_clone", spy)
    vd.verify("https://x.invalid", str(repo), "submission-1", _http("000000000000"))
    assert seen and not os.path.exists(seen[0])


def test_main_prints_one_line_and_exits_non_zero_on_anything_but_a_match(tmp_path, monkeypatch, capsys):
    repo = _published_repo(tmp_path)
    monkeypatch.setattr(vd, "_urllib_get", lambda base: _http("000000000000"))
    assert vd.main(["--base", "https://x.invalid", "--repo", str(repo)]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("MISMATCH")
    monkeypatch.setattr(vd, "_urllib_get", lambda base: _http(_digest_of(repo)))
    assert vd.main(["--base", "https://x.invalid", "--repo", str(repo)]) == 0
    assert capsys.readouterr().out.strip().startswith("MATCH")


def test_main_refuses_a_plain_http_address(capsys):
    assert vd.main(["--base", "http://example.com", "--repo", "x"]) == 1
    assert "https" in capsys.readouterr().out


def test_main_turns_an_unexpected_error_into_a_readable_fail(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vd, "verify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret detail")))
    assert vd.main(["--base", "https://x.invalid", "--repo", "x"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "RuntimeError" in out and "secret detail" not in out and "Traceback" not in out
