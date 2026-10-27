"""Wipe the memory DB after evaluation (plan_v3 section E: evaluation data is
deleted when done). Single implementation: db.wipe() (the table registry
decides what gets cleared). Also removes the WAL/SHM sidecar files.

The record of which write settings the data was written under (core_write_settings) is not evaluation data and
stays: the service checks it only at start-up, so a wipe that deleted it would leave a running service that
writes new data with no record and refuses to start at its next restart. With no data in the database the
start-up replaces the record with the current settings, so keeping it never blocks a change of settings.

The service writes through the shared core, so the core's tables are the ones
that hold the evaluation data. A database from an older version may not have
every table: those are skipped (and named in the output), never created — a
wipe clears what is there and does not change the shape of the database. The
tables of the retired code track (memories, tasks, ...) are cleared too when
they are still there, so nothing of the old service survives a wipe.

Usage:
    python scripts/wipe.py [db_path]

db_path defaults to $DB_PATH, then app's default (../data/memory.db).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db import Database as CoreDatabase  # noqa: E402

# The tables of the retired code track, in the order the old service registered
# them: the old service's table list had nine (memories, mem_fts,
# processed_requests, vectors, fingerprints, tasks, tasks_fts, write_settings,
# task_fingerprints). A database created by the old service still has them; a
# wipe clears them too (the evaluation data has to go), but it never creates
# them and never drops them. The list lives here, in the shell's script, not
# in the core — the core does not know these names.
RETIRED_TABLES = ("memories", "mem_fts", "processed_requests", "vectors",
                  "fingerprints", "tasks", "tasks_fts", "write_settings",
                  "task_fingerprints")


def main():
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = os.environ.get(
            "DB_PATH",
            os.path.join(os.path.dirname(__file__), "..", "data", "memory.db"),
        )
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        print(f"no db at {db_path}, nothing to wipe")
        return
    result = CoreDatabase(db_path).wipe(extra_tables=RETIRED_TABLES)
    # WAL checkpoint files are safe to remove once the DB is closed.
    for suffix in ("-wal", "-shm"):
        side = db_path + suffix
        if os.path.exists(side):
            try:
                os.remove(side)
            except OSError:
                pass
    print(f"wiped {db_path}")
    if result["cleared"]:
        print("cleared: " + ", ".join(sorted(set(result["cleared"]))))
    if result["skipped"]:
        print("skipped (not in this database): " + ", ".join(sorted(set(result["skipped"]))))


if __name__ == "__main__":
    main()