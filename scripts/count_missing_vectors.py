"""How many records have no vector yet — exact, read only. Run it by hand (e.g. inside the container after a
deployment); /health does not carry this number because it needs a full read of every space.

    python scripts/count_missing_vectors.py <path to the database> [--model MODEL] [--by-space]

MODEL is the vector model key as stored (e.g. "text-embedding-v4:1024"); without it the key is taken from the
service's own settings (the same environment the service runs with). The judgment of "missing" is the core's
own (core.engine.find_missing_vectors: valid records of the dense lines whose text has no vector for the
model) — the same one the backfill script uses.

The database is opened read only (SQLite's mode=ro): no table or row is created or changed and the main database
file is not written. Opening a database in WAL mode read only may still make SQLite create or use its side files
(`-wal`, `-shm`) next to it, as any reader does. A database without the core's tables is an error (exit code 2),
not "0 missing".

Prints, last line, one JSON object: {"missing_vectors": N, "spaces": S, "spaces_with_missing": K, "model": ...}.
With --by-space, one line per space that has missing records comes first ("<space>\\t<count>").
"""
import argparse
import json
import os
import pathlib
import sqlite3
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.config import CoreConfig  # noqa: E402
from core.db import table_exists  # noqa: E402
from core.engine import find_missing_vectors  # noqa: E402
from core.store import Store  # noqa: E402

NEEDED_TABLES = ("core_records", "core_vectors")


class ReadOnlyDatabase:
    """The one thing Store and find_missing_vectors need from a database: connect(). Read only, never creates."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._uri = pathlib.Path(self.path).as_uri() + "?mode=ro"

    def connect(self):
        return sqlite3.connect(self._uri, uri=True)


def _service_model():
    from app.config import load
    from app.main import _build_embedder
    return _build_embedder(load()).model_name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count records without a vector (read only).")
    ap.add_argument("db_path")
    ap.add_argument("--model", help="vector model key as stored; default: the service's configured model")
    ap.add_argument("--by-space", action="store_true", help="also print one line per space with missing records")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.db_path):
        print(f"no database at {args.db_path}", file=sys.stderr)
        return 2
    db = ReadOnlyDatabase(args.db_path)
    conn = db.connect()
    try:
        absent = [t for t in NEEDED_TABLES if not table_exists(conn, t)]
        spaces = [] if absent else [r[0] for r in conn.execute("SELECT DISTINCT space_id FROM core_records")]
    finally:
        conn.close()
    if absent:
        print(f"not a core database (missing tables: {', '.join(absent)})", file=sys.stderr)
        return 2

    model = args.model or _service_model()
    store = Store(db)
    exclude = CoreConfig().dense_exclude_lines
    total, with_missing = 0, 0
    for sid in spaces:
        n = len(find_missing_vectors(store, db, model, exclude, sid))
        if n:
            with_missing += 1
            total += n
            if args.by_space:
                print(f"{sid}\t{n}")
    print(json.dumps({"missing_vectors": total, "spaces": len(spaces), "spaces_with_missing": with_missing,
                      "model": model}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
