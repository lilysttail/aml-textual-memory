"""The image must hold every package the DEPLOYED configuration imports.

The deployed configuration (Dockerfile ENV: keyword + dense recall, DashScope embeddings) reaches code the default
configuration never touches: writing a record computes its vector, searching ranks with them, and both import numpy.
The image installs requirements.txt only; numpy sat in requirements-dev.txt, the image had none, and the first
/add after the first real deployment answered 500 (ModuleNotFoundError). Nothing in the suite saw it: the development
environment has numpy installed, and test_docker_image.py checked the default (keyword-only) path.

Two checks, both offline:

  1. static: every third-party package imported anywhere in app/, core/ and scripts/ (also inside functions, which is where the
     lazy imports hide) is listed in requirements.txt — or is on the short, explained list of development-only imports;
  2. behavioural: a fresh Python process whose import system lets through ONLY the standard library, our own app/ and
     core/, and what pip would install for requirements.txt (the listed packages and their dependencies, worked out from the
     installed package metadata) starts the app with the Dockerfile's ENV and walks the deployed path — write, search,
     /health — against a fake embedding service. A package missing from requirements.txt fails there with the same
     ModuleNotFoundError the platform logged.
"""
import ast
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import tempfile

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# third-party import name -> the distribution that provides it (what requirements.txt must list)
PROVIDED_BY = {"fastapi": "fastapi", "pydantic": "pydantic", "numpy": "numpy", "snowballstemmer": "snowballstemmer"}
# imports that are NOT on the deployed path, with the only files allowed to import them
DEV_ONLY = {
    "onnxruntime": {"app/embedder.py"},      # LocalEmbedder, EMBED_PROVIDER=local (development reference); the image uses dashscope
    "tokenizers": {"app/embedder.py"},       # same
    "requests": {"scripts/smoke_local.py"},  # a developer's smoke script against a local server
}
FAKE_API_KEY = "image-api-key-NOT-A-REAL-KEY-456"
FAKE_DASHSCOPE_KEY = "image-dashscope-NOT-A-REAL-KEY-123"


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements():
    """name -> Requirement for every real line of requirements.txt (comments and blanks dropped)."""
    out = {}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                r = Requirement(line)
                out[_norm(r.name)] = r
    return out


def _dockerfile_env():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read().replace("\\\r\n", " ").replace("\\\n", " ")
    env = {}
    for m in re.finditer(r"^ENV\s+(.+)$", text, re.MULTILINE):
        for pair in m.group(1).split():
            name, sep, value = pair.partition("=")
            assert sep, f"unsupported ENV form in the Dockerfile: {pair!r}"
            env[name] = value.strip('"')
    return env


def _dockerfile_python():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        m = re.search(r"FROM\s+python:(\d+)\.(\d+)", f.read())
    return f"{m.group(1)}.{m.group(2)}"


def _third_party_imports():
    """import name -> set of files (repo-relative) importing it, over app/, core/ and scripts/, nested imports included."""
    std = set(sys.stdlib_module_names)
    found = {}
    for pkg in ("app", "core", "scripts"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, pkg)):
            dn[:] = [d for d in dn if d not in ("__pycache__", "tests")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                with open(p, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=p)
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                        names = [node.module.split(".")[0]]
                    for n in names:
                        if n not in std and n not in ("app", "core"):
                            found.setdefault(n, set()).add(os.path.relpath(p, ROOT).replace(os.sep, "/"))
    return found


def _top_level_modules(dist):
    txt = dist.read_text("top_level.txt")
    if txt:
        return {ln.strip() for ln in txt.splitlines() if ln.strip()}
    tops = set()
    for f in dist.files or []:
        first = str(f).replace("\\", "/").split("/")[0]
        if first.endswith(".py"):
            tops.add(first[:-3])
        elif not first.endswith((".dist-info", ".data", "__pycache__")):
            tops.add(first)
    return tops


def _installed_with_requirements_txt():
    """The top-level module names a `pip install -r requirements.txt` provides: the listed distributions and, transitively, the
    ones they depend on (default markers, no extras) — read from the metadata of what is installed here."""
    seen, modules, todo = set(), set(), list(_requirements().values())
    while todo:
        r = todo.pop()
        name = _norm(r.name)
        if name in seen:
            continue
        if r.marker is not None and not r.marker.evaluate({"extra": ""}):
            continue
        seen.add(name)
        try:
            dist = md.distribution(r.name)
        except md.PackageNotFoundError:
            pytest.fail(f"{r.name} is in requirements.txt but not installed in this environment: the check cannot be made")
        modules |= _top_level_modules(dist)
        todo.extend(Requirement(x) for x in (dist.requires or []))
    return modules


# ---- 1. static ----

def test_every_third_party_import_on_the_deployed_path_is_listed_in_requirements_txt():
    listed = set(_requirements())
    problems = []
    for imp, files in sorted(_third_party_imports().items()):
        if imp in DEV_ONLY:
            extra = files - DEV_ONLY[imp]
            if extra:
                problems.append(f"{imp} is development-only but is also imported by {sorted(extra)}")
        elif imp not in PROVIDED_BY:
            problems.append(f"{imp} (imported by {sorted(files)}) is neither mapped to a distribution nor marked development-only")
        elif _norm(PROVIDED_BY[imp]) not in listed:
            problems.append(f"{imp} is imported by {sorted(files)} but {PROVIDED_BY[imp]} is not in requirements.txt")
    assert not problems, "\n".join(problems)


def test_the_dockerfile_installs_requirements_txt_and_starts_uvicorn_which_it_lists():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read()
    assert re.search(r"pip install[^\n]*-r requirements\.txt", text)
    assert "uvicorn" in _requirements(), "the image's CMD runs uvicorn"


def test_every_pinned_package_installs_on_the_python_the_image_runs():
    py = _dockerfile_python()
    bad = []
    for name, r in _requirements().items():
        if not any(s.operator == "==" for s in r.specifier):
            continue
        need = md.metadata(r.name)["Requires-Python"]
        if need and py not in SpecifierSet(need):
            bad.append(f"{r.name} needs Python {need}, the image runs {py}")
    assert not bad, "\n".join(bad)


def test_numpy_is_pinned_to_the_version_the_suite_and_the_measurements_ran_on():
    reqs = _requirements()
    assert "numpy" in reqs, "numpy is imported by the dense path the image runs"
    pins = [s.version for s in reqs["numpy"].specifier if s.operator == "=="]
    assert pins == [md.version("numpy")], f"requirements.txt must pin numpy exactly to the tested version, found {reqs['numpy']}"


def test_the_dev_requirements_header_no_longer_says_production_needs_none_of_it():
    with open(os.path.join(ROOT, "requirements-dev.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "does NOT need these" not in text, "the old header claimed production needs none of the file; numpy proved it wrong"
    assert "numpy" not in {_norm(re.split(r"[<>=!~ ]", ln.split("#", 1)[0].strip())[0])
                           for ln in text.splitlines() if ln.split("#", 1)[0].strip()}, \
        "numpy is a runtime dependency now (requirements.txt); the dev file must not list it a second time"


# ---- 2. behavioural: only what requirements.txt provides, the Dockerfile's ENV, the dense path walked ----

CHILD = r'''
import importlib.abc, json, os, sys
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["root"])
allowed = set(cfg["allowed"]) | set(sys.builtin_module_names) | {"app", "core"}

class _OnlyWhatTheImageHas(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in allowed:
            return None
        raise ModuleNotFoundError(f"No module named {top!r}", name=top)

sys.meta_path.insert(0, _OnlyWhatTheImageHas())
os.environ.update(cfg["env"])

calls = {"n": 0}
def fake_transport(url, headers, body, timeout):
    calls["n"] += 1
    req = json.loads(body)
    dim = req["dimensions"]
    data = []
    for i, text in enumerate(req["input"]):
        vec = [0.0] * dim
        vec[sum(map(ord, text)) % dim] = 1.0
        data.append({"index": i, "embedding": vec})
    return 200, json.dumps({"data": data, "usage": {"total_tokens": 7}}).encode()

import app.embedder as emb
emb._urllib_transport = fake_transport
from fastapi.testclient import TestClient
import app.main as m

auth = {"Authorization": "Bearer " + cfg["env"]["API_KEY"]}
out = {}
with TestClient(m.app) as c:
    out["health_before"] = c.get("/health").json()
    r = c.post("/add", headers=auth, json={"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": [
        {"role": "user", "timestamp": 1728000000000, "content": "I adopted a rescue greyhound named Biscuit last spring."},
        {"role": "user", "timestamp": 1728000001000, "content": "The lemon cake needs three eggs."}]})
    out["add"] = r.status_code
    r = c.post("/search", headers=auth, json={"query": "Which dog did I take in from a shelter?", "user_id": "u1", "top_k": 100})
    out["search"] = r.status_code
    out["hits"] = len(r.json().get("data", [])) if r.status_code == 200 else None
    out["health_after"] = c.get("/health").json()
out["embedding_calls"] = calls["n"]
print("RESULT " + json.dumps(out))
'''


def _walk_the_deployed_path(tmp_path):
    env = dict(_dockerfile_env())
    env.update({"API_KEY": FAKE_API_KEY, "DASHSCOPE_API_KEY": FAKE_DASHSCOPE_KEY,
                "DB_PATH": str(tmp_path / "memory.db")})
    cfg = {"root": ROOT, "env": env,
           "allowed": sorted(set(sys.stdlib_module_names) | _installed_with_requirements_txt())}
    script = tmp_path / "walk_deployed_path.py"
    script.write_text(CHILD, encoding="utf-8")
    p = subprocess.run([sys.executable, "-I", str(script), json.dumps(cfg)], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    return p


def test_the_deployed_path_runs_with_only_what_requirements_txt_provides(tmp_path):
    p = _walk_the_deployed_path(tmp_path)
    assert p.returncode == 0, "the deployed path needs a package requirements.txt does not provide:\n" + p.stderr[-1500:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    out = json.loads(line[len("RESULT "):])
    assert out["add"] == 200 and out["search"] == 200 and out["hits"] >= 1, out
    assert "dense" in out["health_after"]["config"]["recall_paths"] and out["health_after"]["config"]["embed_provider"] == "dashscope"
    assert out["embedding_calls"] >= 2, f"the meaning-based path never reached the embedding service: {out}"
