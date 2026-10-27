"""The core's own write-time settings — the database remembers how it was written.

The FTS index is built from the terms the tokenizer produced at write time. If
the tokenizer changes later, the old index still answers with the old terms:
fewer hits, no error, nothing says so. So the core records the settings its
data was written with, and refuses to start on a database that holds data
written under different ones (the fix is to rebuild the index from a fresh
snapshot).

This is the core's own mechanism, separate from the shell's: the core must be
able to stand on its own after it is moved home.
"""
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

from .db import table_exists

# The version of the way the core writes its rows / builds its index. Bump it
# when a write changes in a way the old index would not understand.
WRITE_FORMAT_VERSION = "2"  # v2: event time (numeric date + English month spellings) added to the FTS index

SETTINGS = ("tokenizer", "write_format", "stem")


@dataclass(frozen=True)
class Diff:
    name: str
    recorded: Optional[str]       # None = the record has no value for this item
    now: str


class WriteSettingsMismatch(RuntimeError):
    """The database holds data written under other write-time settings."""

    def __init__(self, diffs: List[Diff], had_record: bool):
        self.diffs = diffs
        self.had_record = had_record
        lines = ["refusing to start: this database already holds data, but it was not "
                 "written with the current write-time settings."]
        if not had_record:
            lines.append("  There is no record of the settings it was written with.")
        for d in diffs:
            rec = "(not recorded)" if d.recorded is None else repr(d.recorded)
            lines.append(f"  {d.name}: database has {rec}, now {d.now!r}")
        lines.append("These settings decide what the index contains; old rows would "
                     "answer with the wrong terms. Rebuild the index (write a fresh "
                     "snapshot) or clear the database first.")
        super().__init__("\n".join(lines))


def current(config) -> Dict[str, str]:
    """The effective value of every write-time setting under this config."""
    if config.tokenizer_key:
        tokenizer = config.tokenizer_key
    elif config.tokenizer is None:
        tokenizer = "builtin-default"  # gate:allow-secret
    else:
        tokenizer = type(config.tokenizer).__name__
    stem = "0"
    if config.stem_english:
        # Record WHICH stemmer and version, not just "on": switching
        # implementation or version later must refuse old databases and
        # force a rebuild, never silently mix stems.
        try:
            from importlib.metadata import version as _pkg_version
            stem = f"snowball-english-{_pkg_version('snowballstemmer')}"
        except Exception as e:
            raise RuntimeError(
                "stem_english is on but snowballstemmer is not importable: "
                f"{type(e).__name__}: {e}")
    return {"tokenizer": tokenizer, "write_format": WRITE_FORMAT_VERSION,
            "stem": stem}


def recorded(db) -> Dict[str, str]:
    """What the database says it was built with ({} = no record).

    A database from before this table existed has no record — that is not an
    error, it just means nothing has been written under a known setting.
    """
    conn = db.connect()
    try:
        if not table_exists(conn, "core_write_settings"):
            return {}
        return dict(conn.execute("SELECT name, value FROM core_write_settings").fetchall())
    finally:
        conn.close()


def record(db, values: Dict[str, str]) -> None:
    """Replace the record with `values` (one transaction)."""
    with db.writing() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS core_write_settings("
                         "name TEXT PRIMARY KEY, value TEXT)")
            conn.execute("DELETE FROM core_write_settings")
            conn.executemany("INSERT INTO core_write_settings(name, value) VALUES (?, ?)",
                              sorted(values.items()))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def has_data(db) -> bool:
    """True when anything was ever written (a row or a batch)."""
    conn = db.connect()
    try:
        for t in ("core_records", "core_batches"):
            if not table_exists(conn, t):
                continue
            if conn.execute(f"SELECT 1 FROM {t} LIMIT 1").fetchone():
                return True
        return False
    finally:
        conn.close()


def check_or_record(db, config) -> None:
    """The startup step. Empty database: record the current values. Database with
    data: they must equal the record, or raise."""
    now = current(config)
    rec = recorded(db)
    if not has_data(db):
        if rec != now:
            record(db, now)
        return
    diffs = [Diff(name, rec.get(name), now[name]) for name in SETTINGS
             if rec.get(name) != now[name]]
    if diffs:
        raise WriteSettingsMismatch(diffs, had_record=bool(rec))
