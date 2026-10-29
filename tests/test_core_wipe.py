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

import pytest

from core.db import TABLES as CORE_TABLES
from scripts.wipe import RETIRED_TABLES

# The core's tables that a wipe leaves alone because they are not evaluation data: the record of which write
# settings the data was written under (core/write_settings.py checks it at start-up only), and the change
# counters (core_generations) — a wipe bumps them, it never clears them (they must only grow).
KEPT_BY_A_WIPE = {"core_write_settings", "core_generations"}

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
    "core_generations": "INSERT INTO core_generations(key, gen) VALUES ('space:sp', 3)",
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
            if name == "core_generations":
                gens = dict(conn.execute("SELECT key, gen FROM core_generations").fetchall())
                assert gens.get("space:sp") == 4 and gens.get("epoch", 0) >= 1,                     f"a wipe must bump every counter and the epoch, never clear them: {gens}"
            elif name in KEPT_BY_A_WIPE:
                assert n == 1, f"{name} is not evaluation data: a wipe must leave it alone ({n} rows)"
            else:
                assert n == 0, f"{name} still has {n} rows"
    finally:
        conn.close()
    cleared = [l for l in r.stdout.splitlines() if l.startswith("cleared:")][0]
    for name in KEPT_BY_A_WIPE:
        assert name not in cleared, cleared


def test_the_registry_and_the_wipe_agree_on_what_is_evaluation_data():
    """The wipe clears exactly the registry's evaluation-data tables — and the tables that are not evaluation data
    (the record of the write settings, the change counters) are named here, so moving one into the data set is a
    visible change."""
    from core.db import business_tables
    assert KEPT_BY_A_WIPE == {"core_write_settings", "core_generations"}
    assert set(business_tables()) == set(CORE_TABLES) - KEPT_BY_A_WIPE


def test_wipe_then_write_without_a_restart_then_restart_still_starts():
    """The operator's path: wipe the live database, the running service is NOT restarted and keeps taking
    writes, and the host restarts the service later. Wiping the record of the write settings along with the data
    left data without a record, and that restart refused to start (WriteSettingsMismatch)."""
    from core import CoreConfig, Record, build
    path = _db_path()
    cfg = lambda: CoreConfig(db_path=path, paths=("keyword",))
    running = build(cfg())
    running.write([Record(record_id="r1", session_id="s", text="giraffe before the wipe", speaker="user")],
                  space_id="u", batch_id="b1")
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    # no restart: the same running core takes a write after the wipe
    running.write([Record(record_id="r2", session_id="s", text="zebra after the wipe", speaker="user")],
                  space_id="u", batch_id="b2")
    restarted = build(cfg())          # must not raise
    hits = restarted.search("zebra", space_id="u", top_k=5)
    assert [h.record_id for h in hits] == ["r2"], hits
    assert restarted.search("giraffe", space_id="u", top_k=5) == []   # the wiped data is gone


def test_wipe_then_start_with_new_settings_records_the_new_settings():
    """What always worked has to keep working: data written under the old settings, a change of settings, a wipe,
    a start — the start records the NEW settings (the kept old record does not block it), and from then on
    the database belongs to the new settings."""
    from core import CoreConfig, Record, build
    from core.db import Database
    from core.write_settings import WriteSettingsMismatch, recorded
    path = _db_path()
    old = build(CoreConfig(db_path=path, paths=("keyword",), stem_english=False))
    old.write([Record(record_id="r1", session_id="s", text="the cats slept", speaker="user")],
              space_id="u", batch_id="b1")
    assert recorded(Database(path))["stem"] == "0"
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    new_cfg = lambda: CoreConfig(db_path=path, paths=("keyword",), stem_english=True)
    new = build(new_cfg())            # the database is empty: no refusal, the new settings are recorded
    assert recorded(Database(path))["stem"].startswith("snowball-english"), recorded(Database(path))
    new.write([Record(record_id="r2", session_id="s", text="the cats slept", speaker="user")],
              space_id="u", batch_id="b2")
    build(new_cfg())                  # restart under the new settings: starts
    with pytest.raises(WriteSettingsMismatch):   # and the old settings are now the wrong ones for this data
        build(CoreConfig(db_path=path, paths=("keyword",), stem_english=False))


# --- the wipe and a database that is open somewhere else (the running service) ------------------------------------

def _seed_core(path, n=16, **kw):
    """A real database with `n` evaluation rows and the three write-settings rows, made the way the service makes it."""
    from core.db import Database
    db = Database(path, **kw)
    db.create_schema()
    with db.writing() as conn:
        for i in range(n):
            conn.execute("INSERT INTO core_records(space_id, record_id, session_id, text) VALUES ('u', ?, 's', 'hello')", (f"r{i}",))
            conn.execute("INSERT INTO core_fts(space_id, record_id, session_id, speaker, body) VALUES ('u', ?, 's', 'user', 'hello')", (f"r{i}",))
        from core import CoreConfig
        from core.write_settings import current
        conn.executemany("INSERT INTO core_write_settings(name, value) VALUES (?, ?)",
                         sorted(current(CoreConfig(db_path=path, paths=("keyword",))).items()))   # the settings a real start records
        conn.commit()
    return db


def _counts(path):
    conn = sqlite3.connect(path)
    try:
        return (conn.execute("SELECT COUNT(*) FROM core_records").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM core_write_settings").fetchone()[0])
    finally:
        conn.close()


def test_the_wipe_script_never_deletes_the_side_files_of_a_database(monkeypatch):
    """The running service holds the database open while the operator wipes it. Deleting its `-wal` / `-shm` from under it
    drops what the wipe just committed (it is still in the `-wal`) and breaks every later connection ("disk I/O error"; the
    service's /health answers 500). On Linux a file can be deleted while it is open; on Windows the delete fails, so a test
    that only watched the outcome could not see the bug there. So this one watches the calls: no deletion of any file of
    that database, whatever the platform."""
    import pathlib
    import runpy
    path = _db_path()
    db = _seed_core(path)
    held = db.connect()                       # the service's connection: while it is open the side files exist
    held.execute("SELECT COUNT(*) FROM core_records").fetchone()
    assert os.path.exists(path + "-wal") and os.path.exists(path + "-shm"), "the side files must exist for this test to mean anything"
    asked = []
    real_remove, real_unlink, real_path_unlink = os.remove, os.unlink, pathlib.Path.unlink

    def spy(real):
        def inner(target, *a, **k):
            asked.append(os.fspath(target))
            return real(target, *a, **k)
        return inner

    monkeypatch.setattr(os, "remove", spy(real_remove))
    monkeypatch.setattr(os, "unlink", spy(real_unlink))
    monkeypatch.setattr(pathlib.Path, "unlink", lambda self, *a, **k: (asked.append(str(self)), real_path_unlink(self, *a, **k))[1])
    monkeypatch.setattr(sys, "argv", [WIPE_SCRIPT, path])
    runpy.run_path(WIPE_SCRIPT, run_name="__main__")
    held.close()
    mine = [a for a in asked if os.path.abspath(a).startswith(os.path.abspath(path))]
    assert mine == [], f"the wipe deleted files of the database: {mine}"


def test_wipe_while_another_connection_holds_the_database_open():
    """The operator's wipe on the live service (the Zeabur Command), end to end: one connection stays open the way the
    service's does, the REAL script wipes in another process. Afterwards a NEW connection (every /health is one) must
    work and see the wiped database, the open connection must read and write, the database must be sound, and a restart
    must start. (On Windows the old behaviour is hidden — the delete just fails — so the test above is the one that
    fails there; on Linux this one fails too: "disk I/O error".)"""
    from core import CoreConfig, build
    from core.db import Database
    path = _db_path()
    db = _seed_core(path)
    held = db.connect()
    assert held.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 16
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert _counts(path) == (0, 3), "a new connection must see the wiped data and the kept write-settings record"
    assert held.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 0   # the service's own connection
    held.execute("INSERT INTO core_records(space_id, record_id, session_id, text) VALUES ('u','after','s','written after the wipe')")
    held.commit()
    assert _counts(path) == (1, 3)
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        check.close()
    held.close()
    build(CoreConfig(db_path=path, paths=("keyword",)))   # a restart of the service: must start (the record was kept)


def test_wipe_reports_what_the_checkpoint_did():
    """`checkpoint:` is the last line of the script's output: `truncated` when nobody else was reading (the log is folded
    into the database file and emptied), `partial` when another connection pins a read (the wipe is committed and visible
    anyway; SQLite folds the rest in later)."""
    path = _db_path()
    db = _seed_core(path)
    r = subprocess.run([sys.executable, WIPE_SCRIPT, path], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert r.stdout.strip().splitlines()[-1] == "checkpoint: truncated", r.stdout
    assert not os.path.exists(path + "-wal") or os.path.getsize(path + "-wal") == 0

    path2 = _db_path()
    db2 = _seed_core(path2, busy_timeout_ms=200)   # the checkpoint waits for readers at most this long
    reader = db2.connect()
    reader.execute("BEGIN")
    assert reader.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 16   # a read that stays open: it pins the log
    result = db2.wipe()
    assert result["checkpoint"] == "partial", result
    assert _counts(path2) == (0, 3)                                                   # committed and visible to everyone
    assert reader.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 16   # the open reader keeps its own snapshot
    reader.rollback()
    assert reader.execute("SELECT COUNT(*) FROM core_records").fetchone()[0] == 0
    reader.close()


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
