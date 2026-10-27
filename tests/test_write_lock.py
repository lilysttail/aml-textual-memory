"""E concurrency fix — 64-way writes must not starve on the SQLite write lock.

Root cause (located on jin/e-locked-flake): 64 threads all hitting BEGIN
IMMEDIATE at once make SQLite's busy-handler back-off thrash; a round of 64
writes takes ~4-5s (right at busy_timeout) and occasionally throws StoreBusy.

Fix: an in-process write lock serializes writers (FIFO) so they no longer all
pile onto BEGIN IMMEDIATE. These tests lock that in (against the core's
storage, which is the one the service writes through):
  - 0 StoreBusy across many rounds, same-key and distinct-key
  - a per-round time budget well under busy_timeout (5s)
  - distinct-key writes all land (no lost writes)
"""
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)

from core.db import Database, StoreBusy  # noqa: E402
from core.models import Record  # noqa: E402
from core.store import Store  # noqa: E402

# Must be well under busy_timeout (5s). The serialized fix runs sub-second;
# 2.0s leaves headroom for CI jitter while still failing the ~4-5s pre-fix.
ROUND_BUDGET_S = 2.0
WORKERS = 64
ROUNDS = 5


def _fresh():
    db = Database(os.path.join(_TMPROOT, f"wl_{uuid.uuid4().hex}.db"))
    db.create_schema()
    store = Store(db)
    return db, store


def _run(store, same_key=True):
    def w(i):
        rid = "rX" if same_key else f"r{i}"
        rec = [Record(record_id=f"{rid}#0", session_id="s", text="same request replay")]
        try:
            store.write(rec, space_id="u", batch_id=rid)
            return 0
        except StoreBusy:
            return 1
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        busy = sum(ex.map(w, range(WORKERS)))
    return time.time() - t0, busy


def test_concurrent_writes_no_busy_same_key():
    worst = 0.0
    for _ in range(ROUNDS):
        db, store = _fresh()
        dur, busy = _run(store, same_key=True)
        worst = max(worst, dur)
        assert busy == 0, f"same-key: {busy} StoreBusy in a round"
        assert store.count("u") == 1
    assert worst < ROUND_BUDGET_S, f"same-key worst round {worst:.2f}s >= {ROUND_BUDGET_S}s"


def test_concurrent_writes_no_busy_distinct_key():
    worst = 0.0
    for _ in range(ROUNDS):
        db, store = _fresh()
        dur, busy = _run(store, same_key=False)
        worst = max(worst, dur)
        assert busy == 0, f"distinct-key: {busy} StoreBusy in a round"
        assert store.count("u") == WORKERS, "lost writes under distinct-key concurrency"
    assert worst < ROUND_BUDGET_S, f"distinct-key worst round {worst:.2f}s >= {ROUND_BUDGET_S}s"


def test_connect_failure_releases_write_lock():
    """If connect() fails after the write lock is acquired, the lock must still
    be released — otherwise every later write starves and 503s."""
    db, store = _fresh()
    orig = db.connect

    def boom():
        raise sqlite3.OperationalError("simulated connect failure")

    db.connect = boom
    with pytest.raises(sqlite3.OperationalError):
        store.write([Record(record_id="r1#0", session_id="s", text="x")],
                    space_id="u", batch_id="r1")
    assert not db._write_lock.locked(), "write lock leaked after connect failure"

    # lock was released: the next write still works
    db.connect = orig
    res = store.write([Record(record_id="r2#0", session_id="s", text="ok")],
                      space_id="u", batch_id="r2")
    assert res["written"] == 1
    assert store.count("u") == 1
