"""The wipe: it clears the core's tables and the retired ones, through the
real entry point (scripts/wipe.py).

WHAT THIS IS
    A wipe has to leave nothing of the evaluation behind — not in the core's
    tables, and not in the tables the old service left in the same file. The
    test drives the REAL script (the same one an operator runs), so it also
    covers the list of retired tables and the output.

HOW TO RUN
        py -m pytest tests/test_core_wipe.py -q
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid

from core.db import TABLES as CORE_TABLES
from scripts.wipe import RETIRED_TABLES

# One row for every table of the core's registry. The list of tables comes
# from the registry (core.db.TABLES); this dict only says how to fill one.
CORE_ROWS = {
    "core_records": "INSERT INTO core_records(space_id, record_id, session_id, text)"
                    " VALUES ('sp','r','s','t')",
    "core_fts": "INSERT INTO core_fts(space_id, record_id, session_id, speaker, body)"
                " VALUES ('sp','r','s','user','body')",
    "core_vectors": "INSERT INTO core_vectors(model, content_hash, dim, vec)"
                    " VALUES ('m','h',4,x'00')",
    "core_batches": "INSERT INTO core_batches(space_id, batch_id) VALUES ('sp','b')",
    "core_sessions": "INSERT INTO core_sessions(space_id, session_id, next_turn_index)"
                     " VALUES ('sp','s',1)",
    "core_spaces": "INSERT INTO core_spaces(space_id, snapshot_id, updated_at)"
                   " VALUES ('sp','v','t')",
    "core_write_settings": "INSERT INTO core_write_settings(name, value)"
                           " VALUES ('k','v')",
    "core_pending_vectors": "INSERT INTO core_pending_vectors(space_id, record_id, content_hash, reason, retry_count, last_attempt_at)"
                            " VALUES ('sp','r','h','timeout',1,'t')",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIPE_SCRIPT = os.path.join(ROOT, "scripts", "wipe.py")

# The nine tables the old service registered (the old service's table list).
OLD_TABLES = [
    ("memories", "CREATE TABLE memories(user_id TEXT, mem_id TEXT, session_id TEXT,"
                 " content TEXT, created_at TEXT, task_key TEXT,"
                 " PRIMARY KEY(user_id, mem_id))",
     "INSERT INTO memories VALUES ('u','m','s','c','t',null)"),
    ("mem_fts", "CREATE VIRTUAL TABLE mem_fts USING fts5(mem_id UNINDEXED,"
                " user_id UNINDEXED, content)",
     "INSERT INTO mem_fts VALUES ('m','u','c')"),
    ("processed_requests", "CREATE TABLE processed_requests(user_id TEXT,"
                           " request_id TEXT, PRIMARY KEY(user_id, request_id))",
     "INSERT INTO processed_requests VALUES ('u','r')"),
    ("vectors", "CREATE TABLE vectors(model TEXT, content_hash TEXT, dim INTEGER,"
                " vec BLOB, PRIMARY KEY(model, content_hash))",
     "INSERT INTO vectors VALUES ('m','h',4,x'00')"),
    ("fingerprints", "CREATE TABLE fingerprints(user_id TEXT, mem_id TEXT,"
                     " fp_type TEXT, fp_value TEXT,"
                     " PRIMARY KEY(user_id, mem_id, fp_type, fp_value))",
     "INSERT INTO fingerprints VALUES ('u','m','error','e')"),
    ("tasks", "CREATE TABLE tasks(user_id TEXT, task_key TEXT, head_text TEXT,"
              " head_source_request_id TEXT, head_mem_id TEXT, head_ts TEXT,"
              " first_ts TEXT, n_msgs INTEGER, repo_hint TEXT,"
              " PRIMARY KEY(user_id, task_key))",
     "INSERT INTO tasks VALUES ('u','t','h','r','m','ts','ts',1,null)"),
    ("tasks_fts", "CREATE VIRTUAL TABLE tasks_fts USING fts5(task_key UNINDEXED,"
                  " user_id UNINDEXED, head_text)",
     "INSERT INTO tasks_fts VALUES ('t','u','h')"),
    ("write_settings", "CREATE TABLE write_settings(name TEXT PRIMARY KEY, value TEXT)",
     "INSERT INTO write_settings VALUES ('k','v')"),
    ("task_fingerprints", "CREATE TABLE task_fingerprints(user_id TEXT, task_key TEXT,"
                          " fp_type TEXT, fp_value TEXT,"
                          " PRIMARY KEY(user_id, task_key, fp_type, fp_value))",
     "INSERT INTO task_fingerprints VALUES ('u','t','error','e')"),
]


def _db_path():
    d = os.path.join(tempfile.gettempdir(), "wipe-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "wipe.db")


def _seed(path):
    """The core's tables (one row in EVERY one of them) plus one row in every
    old table. The list of core tables comes from the registry, not from a
    second copy kept here."""
    from core.db import Database
    db = Database(path)
    db.create_schema()
    conn = sqlite3.connect(path)
    for _name, ddl, insert in OLD_TABLES:
        conn.execute(ddl)
        conn.execute(insert)
    conn.commit()
    conn.close()
    with db.writing() as conn:
        for name, insert in CORE_ROWS.items():
            conn.execute(insert)
        conn.commit()


def test_wipe_clears_the_core_tables():
    path = _db_path()
    _seed(path)
    # every core table has a row before the wipe (a table that was never
    # seeded would let a wipe that skips it pass)
    conn = sqlite3.connect(path)
    try:
        for name in CORE_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            assert n >= 1, f"{name} was not seeded"
    finally:
        conn.close()
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    conn = sqlite3.connect(path)
    try:
        for name in CORE_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            assert n == 0, f"{name} still has {n} rows"
    finally:
        conn.close()


def test_wipe_clears_the_retired_tables_of_the_old_service_too():
    """Every old table has a row before the wipe and none after it — and the
    tables are still there (a wipe clears rows, it does not change the shape
    of the database)."""
    path = _db_path()
    _seed(path)
    # the list the script uses is the old service's registration; the test
    # does not keep a second copy of it
    assert set(RETIRED_TABLES) == {name for name, _d, _i in OLD_TABLES}, RETIRED_TABLES

    conn = sqlite3.connect(path)
    try:
        for name, _ddl, _insert in OLD_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            assert n >= 1, f"{name} was not seeded"
    finally:
        conn.close()

    r = subprocess.run([sys.executable, WIPE_SCRIPT, path],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)

    conn = sqlite3.connect(path)
    try:
        for name, _ddl, _insert in OLD_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            assert n == 0, f"{name} still has {n} rows after the wipe"
            still_there = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
            assert still_there == 1, f"{name} was dropped — a wipe only clears rows"
        assert conn.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 0
    finally:
        conn.close()
    assert "memories" in r.stdout, r.stdout


def test_wipe_works_on_an_empty_database_file():
    path = _db_path()
    with open(path, "wb"):
        pass
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)


def test_wipe_leaves_tables_it_does_not_know_about_alone():
    path = _db_path()
    _seed(path)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE something_else(a TEXT)")
    conn.execute("INSERT INTO something_else VALUES ('keep me')")
    conn.commit()
    conn.close()
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM something_else").fetchone()[0] == 1
    finally:
        conn.close()
