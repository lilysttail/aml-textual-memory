#!/usr/bin/env python3
"""Content fingerprint of what the deployed image carries (12 hex characters). Standard library only.

The image build runs it once and stores the result (`BUILD_DIGEST`); the service reports it as `build` on /health;
`scripts/verify_deploy.py` recomputes it from a clone of the published tag and compares. So "what runs is the published
version" is something anyone can check, whatever the hosting platform shows.

It covers what the Dockerfile copies into the image (INCLUDE below — a test keeps the two lists equal): the Dockerfile itself
(the deployed configuration lives in its ENV lines), requirements.txt, and the app/, core/ and scripts/ directories. Tests,
documentation, compiled files and the stored digest itself are not covered.

Text files are hashed with CRLF read as LF, so a clone made on a machine that converts line endings gives the same answer as
the image's own checkout. A file with a NUL byte is treated as binary and hashed as it is.

    python scripts/tree_digest.py [--root DIR] [--list]
"""
import argparse
import hashlib
import os
import sys

INCLUDE = ("Dockerfile", "requirements.txt", "app", "core", "scripts")
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = (".pyc", ".pyo")


def _default_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _files(root):
    """Sorted (posix relative path, absolute path) of every covered file. A covered path that does not exist is an error."""
    found = []
    for item in INCLUDE:
        top = os.path.join(root, item)
        if os.path.isfile(top):
            found.append((item, top))
        elif os.path.isdir(top):
            for dirpath, dirnames, filenames in os.walk(top):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for name in sorted(filenames):
                    if name.endswith(SKIP_SUFFIXES):
                        continue
                    full = os.path.join(dirpath, name)
                    found.append((os.path.relpath(full, root).replace(os.sep, "/"), full))
        else:
            raise FileNotFoundError(f"covered path is missing: {item}")
    return sorted(found)


def _file_hash(path):
    with open(path, "rb") as f:
        data = f.read()
    if b"\x00" not in data[:8192]:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def file_hashes(root=None):
    root = root or _default_root()
    return [(rel, _file_hash(full)) for rel, full in _files(root)]


def tree_digest(root=None):
    h = hashlib.sha256()
    for rel, digest in file_hashes(root):
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return h.hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="the repository root (default: the one this script lives in)")
    ap.add_argument("--list", action="store_true", help="print every covered file with its hash instead of the digest")
    args = ap.parse_args(argv)
    try:
        if args.list:
            for rel, digest in file_hashes(args.root):
                print(f"{digest[:12]}  {rel}")
        else:
            print(tree_digest(args.root))
    except FileNotFoundError as e:
        print(f"tree_digest: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
