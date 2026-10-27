"""Nothing the repository ships points at a file the repository does not contain.

A document, a comment or an error message that names `scripts/some_tool.py` sends a reader to a file that has to be there. When the
published tree is a subset of the development tree (some scripts are not exported), a reference to a file that was left out is a
dead end for whoever follows it — and an error message like that is what a user sees at the worst moment.

Scanned: the Markdown files, and the Python files of app/, core/ (not its tests), scripts/ and eval/, and the Dockerfile. A
path-like name counts as a reference when it starts with scripts/, eval/, app/, core/ or tests/ and ends with a file extension.
References that are deliberately not files of this repository are listed in ALLOWED with the reason.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# deliberate non-files: the project's earlier service layer (carried-over code names it in provenance headers and notices),
# and files of the platform's own challenge repository that the text points at
ALLOWED = {
    "app/db.py": "provenance: the earlier service layer",
    "app/memory_store.py": "provenance: the earlier service layer",
    "app/recall.py": "provenance: the earlier service layer",
    "app/retrieval.py": "provenance: the earlier service layer",
    "app/vector_store.py": "provenance: the earlier service layer",
    "app/vector_cache.py": "provenance: the earlier service layer",
    "scripts/run_eval.sh": "the official evaluation script of the challenge repository",
}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".pytest_tmp", "node_modules", ".venv", "data"}
REF = re.compile(r"(?<![A-Za-z0-9_./-])((?:scripts|eval|app|core|tests)/[A-Za-z0-9_./-]*[A-Za-z0-9_]\.(?:py|ps1|sh|md|txt|json))\b")


def _scanned_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        for name in filenames:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if name == "Dockerfile" or name.endswith(".md") or name == "requirements.txt" or name == "requirements-dev.txt":
                yield rel
            elif name.endswith(".py") and rel.split("/")[0] in ("app", "scripts", "eval", "core") and not rel.startswith("core/tests/"):
                yield rel


def _dangling():
    bad = []
    for rel in _scanned_files():
        with open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                for m in REF.finditer(line):
                    ref = m.group(1)
                    if ref in ALLOWED or os.path.exists(os.path.join(ROOT, *ref.split("/"))):
                        continue
                    bad.append(f"{rel}:{n}: {ref}")
    return bad


def test_every_file_the_repository_names_is_in_the_repository():
    bad = _dangling()
    assert not bad, "references to files that are not in the repository:\n" + "\n".join(bad)


def test_the_allowed_non_files_really_are_not_here_and_are_still_mentioned():
    """An exception that nobody uses any more, or that now exists, is dead weight: keep the list honest."""
    text = ""
    for rel in _scanned_files():
        with open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8", errors="replace") as f:
            text += f.read() + "\n"
    stale = [ref for ref in ALLOWED if ref not in text]
    assert not stale, f"no longer mentioned anywhere, drop from ALLOWED: {stale}"
