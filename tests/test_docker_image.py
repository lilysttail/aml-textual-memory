"""The image must contain the core: the shell imports it (review P1-1).

The Dockerfile used to copy only app/, so an image built from it could not
even import app.main. There is no Docker on this machine, so the check is
static plus a live one:

  - static: every top-level import of the files the image copies (app/ and
    core/) must be inside the image too, or be a third-party package that
    requirements.txt installs. That is exactly the condition the image has
    to satisfy.
  - live: the app imports and serves /health in a fresh process, with only
    the repository on the path (what the image holds).
"""
import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dockerfile_copies():
    """(source, destination) of every COPY in the Dockerfile."""
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read()
    out = []
    for m in re.finditer(r"^COPY\s+(\S+)\s+(\S+)\s*$", text, re.MULTILINE):
        src, dst = m.group(1), m.group(2)
        if src.startswith("--"):
            continue
        out.append((src, dst))
    return out


def _in_image():
    return {src.rstrip("/") for src, _ in _dockerfile_copies()}


def _imports_of(path):
    """The top-level module names a file imports (relative imports excluded)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # a relative import: stays inside the package
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_the_dockerfile_copies_the_core():
    copies = dict(_dockerfile_copies())
    assert "core" in copies, f"the image would have no core: {copies}"


def test_the_imports_of_the_copied_files_stay_inside_the_image():
    in_image = _in_image()
    problems = []
    for pkg in sorted(in_image):
        d = os.path.join(ROOT, pkg)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(d, name)
            for top in _imports_of(path):
                # our own packages must be in the image; anything else is a
                # third-party package that requirements.txt installs
                if top in ("app", "core") and top not in in_image:
                    problems.append(f"{pkg}/{name} imports {top}, which the image does not copy")
    assert not problems, "\n".join(problems)


def test_the_copied_files_can_import_the_app():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    code = "import app.main as m; print('imported', callable(m.make_app))"
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "imported" in p.stdout, p.stdout


def test_the_copied_files_can_serve():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["DB_PATH"] = os.path.join(ROOT, ".image_tmp", "data", "memory.db")
    os.makedirs(os.path.join(ROOT, ".image_tmp", "data"), exist_ok=True)
    code = ("from fastapi.testclient import TestClient; import app.main as m;"
            "c = TestClient(m.make_app(m.load()));"
            "r = c.get('/health'); print('health', r.status_code)")
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    finally:
        import shutil
        shutil.rmtree(os.path.join(ROOT, ".image_tmp"), ignore_errors=True)
    assert p.returncode == 0, p.stderr
    assert "health 200" in p.stdout, p.stdout