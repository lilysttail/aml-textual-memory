"""scripts/tree_digest.py — the content fingerprint of what the image carries.

The image stores it (`BUILD_DIGEST`) and /health reports it as `build`; `scripts/verify_deploy.py` recomputes it from a clone of
the published tag and compares. It only proves anything if (a) it covers exactly what the Dockerfile copies in, (b) the same
content gives the same fingerprint wherever it is checked out (a Windows clone turns LF into CRLF), and (c) any change to a
covered file changes it.
"""
import importlib.util
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("tree_digest", os.path.join(ROOT, "scripts", "tree_digest.py"))
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)


def _write(root, rel, data):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def _tree(root, **over):
    files = {"Dockerfile": "FROM python:3.11-slim\n", "requirements.txt": "fastapi>=0.110\n", "app/main.py": "print('app')\n",
             "core/engine.py": "print('core')\n", "scripts/wipe.py": "print('wipe')\n",
             "tests/test_x.py": "assert True\n", "README.md": "# readme\n"}
    files.update(over)
    for rel, data in files.items():
        if data is not None:
            _write(root, rel, data)
    return root


def test_the_digest_is_twelve_lowercase_hex_characters(tmp_path):
    assert re.fullmatch(r"[0-9a-f]{12}", td.tree_digest(_tree(str(tmp_path))))


def test_the_same_content_gives_the_same_digest_wherever_it_lives(tmp_path):
    a, b = _tree(str(tmp_path / "one")), _tree(str(tmp_path / "two" / "deeper"))
    assert td.tree_digest(a) == td.tree_digest(b)


@pytest.mark.parametrize("rel", ["Dockerfile", "requirements.txt", "app/main.py", "core/engine.py", "scripts/wipe.py"])
def test_a_change_in_any_covered_file_changes_the_digest(tmp_path, rel):
    root = _tree(str(tmp_path))
    before = td.tree_digest(root)
    _write(root, rel, "changed\n")
    assert td.tree_digest(root) != before


@pytest.mark.parametrize("rel", ["app/new.py", "core/new.py", "scripts/new.py"])
def test_an_added_file_in_a_covered_directory_changes_the_digest(tmp_path, rel):
    root = _tree(str(tmp_path))
    before = td.tree_digest(root)
    _write(root, rel, "x\n")
    assert td.tree_digest(root) != before


def test_a_removed_file_changes_the_digest(tmp_path):
    root = _tree(str(tmp_path))
    before = td.tree_digest(root)
    os.remove(os.path.join(root, "scripts", "wipe.py"))
    assert td.tree_digest(root) != before


def test_a_renamed_file_changes_the_digest(tmp_path):
    root = _tree(str(tmp_path))
    before = td.tree_digest(root)
    os.rename(os.path.join(root, "core", "engine.py"), os.path.join(root, "core", "engine2.py"))
    assert td.tree_digest(root) != before


def test_files_the_image_does_not_carry_do_not_change_it(tmp_path):
    root = _tree(str(tmp_path))
    before = td.tree_digest(root)
    _write(root, "tests/test_x.py", "assert False\n")
    _write(root, "README.md", "# other\n")
    _write(root, "app/__pycache__/main.cpython-311.pyc", b"\x00\x01compiled")
    _write(root, "BUILD_DIGEST", "0123456789ab\n")
    _write(root, "app/main.pyc", b"\x00junk")
    _write(root, "core/__pycache__/anything.txt", "whatever the interpreter left here\n")     # the directory is skipped, not just *.pyc
    assert td.tree_digest(root) == before


def test_line_endings_do_not_matter_for_text_files(tmp_path):
    """A clone on Windows (autocrlf) has CRLF where the image's checkout has LF: the fingerprints must still agree."""
    lf = _tree(str(tmp_path / "lf"))
    crlf = _tree(str(tmp_path / "crlf"), **{"app/main.py": "print('app')\r\n", "requirements.txt": "fastapi>=0.110\r\n",
                                            "Dockerfile": "FROM python:3.11-slim\r\n"})
    assert td.tree_digest(lf) == td.tree_digest(crlf)


def test_a_binary_file_is_hashed_as_it_is(tmp_path):
    a = _tree(str(tmp_path / "a"), **{"scripts/blob.bin": b"\x00\r\n\x00A"})
    b = _tree(str(tmp_path / "b"), **{"scripts/blob.bin": b"\x00\n\x00A"})
    assert td.tree_digest(a) != td.tree_digest(b)


def test_what_it_covers_is_exactly_what_the_dockerfile_copies():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read()
    copied = {m.group(1).rstrip("/") for m in re.finditer(r"^COPY\s+(?!--)(\S+)\s+\S+\s*$", text, re.MULTILINE)}
    assert copied == set(td.INCLUDE), f"the digest covers {sorted(td.INCLUDE)}, the Dockerfile copies {sorted(copied)}"


def test_the_dockerfile_stores_the_digest_where_health_reads_it():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read()
    assert re.search(r"^RUN python scripts/tree_digest\.py > BUILD_DIGEST\s*$", text, re.MULTILINE)
    # it must run AFTER everything it covers has been copied in
    last_copy = max(m.start() for m in re.finditer(r"^COPY\s", text, re.MULTILINE))
    assert text.index("RUN python scripts/tree_digest.py") > last_copy


def test_the_command_line_prints_the_digest_and_can_list_the_files(tmp_path):
    root = _tree(str(tmp_path))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "tree_digest.py"), "--root", root, *a],
                                    capture_output=True, text=True, encoding="utf-8")
    out = run()
    assert out.returncode == 0 and out.stdout.strip() == td.tree_digest(root)
    listing = run("--list")
    assert listing.returncode == 0 and "app/main.py" in listing.stdout and "tests/test_x.py" not in listing.stdout


def test_a_missing_covered_path_is_an_error_not_a_digest_of_less(tmp_path):
    root = _tree(str(tmp_path), **{"core/engine.py": None})        # no core/ directory at all
    assert not os.path.exists(os.path.join(root, "core"))
    with pytest.raises(FileNotFoundError):
        td.tree_digest(root)
