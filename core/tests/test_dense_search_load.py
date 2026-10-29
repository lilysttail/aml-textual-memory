"""Dense search under load: what one search may cost, and what it must still return.

Written before the change (test-first). On the version before the change most of
these fail because the RESULT is wrong (a count is too high, the dense path came
back empty, no 503 was given), not because something is missing. The ones marked
"regression guard" pass before the change too: they pin behaviour the change
must keep (a cache must never hand back what was there before a snapshot or a
write). The same guards for the wipe and backfill scripts live with the service's
tests, next to those scripts.

How the costs are counted: the database's `connect` is wrapped so every
connection and every statement goes through a spy. A "full read of a space" is
a SELECT on core_records narrowed by the space only (not by record ids, not by
session, no LIMIT, no COUNT). The spy can also make chosen statements slow, to
stand in for a big space on a slow disk.

New settings these tests turn on (set on the CoreConfig before build; the
version before the change ignores them):
  - search_heavy_concurrency: at most this many searches doing the heavy part
    (building a space's in-memory index, computing similarities) at once;
  - search_heavy_wait_s: how long a search may wait for that; past it the
    search raises StoreBusy (the service answers 503, which the platform retries).
"""
import os
import re
import shutil
import tempfile
import threading
import time
import uuid

import pytest

from core import CoreConfig, Record, build
from core.db import StoreBusy
from core.embedder import EmbedUnavailable
from core.testing import FakeEmbedder

# --------------------------------------------------------------------- spies

_FULL_READ = re.compile(r"\bFROM\s+core_records\b", re.I)
_NARROWED = re.compile(r"record_id\s+IN|session_id\s*=|\bLIMIT\b|COUNT\s*\(", re.I)
_BULK_VECTORS = re.compile(r"\bFROM\s+core_vectors\b.*content_hash\s+IN", re.I | re.S)


def is_full_space_read(sql):
    s = " ".join(sql.split())
    return s.upper().startswith("SELECT") and bool(_FULL_READ.search(s)) and not _NARROWED.search(s)


def bulk_vector_hashes(sql, params):
    """How many content hashes one core_vectors read asks for (0 if it is not one)."""
    if not _BULK_VECTORS.search(" ".join(sql.split())):
        return 0
    return max(0, len(params or ()) - 1)          # first parameter is the model


class Spy:
    def __init__(self):
        self.lock = threading.Lock()
        self.connects = 0
        self.full_reads = 0
        self.bulk_vector_reads = []                # number of hashes per bulk read
        self.in_full_read = 0
        self.peak_full_read = 0
        self.full_read_delay = 0.0
        self.bulk_vector_delay = 0.0               # delay a bulk read asking for more than one hash
        self.in_bulk_read = 0
        self.peak_bulk_read = 0                    # bulk vector reads (more than one hash) at the same moment

    def reset_counts(self):
        with self.lock:
            self.connects = 0
            self.full_reads = 0
            self.bulk_vector_reads = []

    def run(self, sql, params, call):
        full = is_full_space_read(sql)
        nhash = bulk_vector_hashes(sql, params)
        if full:
            with self.lock:
                self.full_reads += 1
                self.in_full_read += 1
                self.peak_full_read = max(self.peak_full_read, self.in_full_read)
            try:
                if self.full_read_delay:
                    time.sleep(self.full_read_delay)
                return call()
            finally:
                with self.lock:
                    self.in_full_read -= 1
        if nhash > 1:
            with self.lock:
                self.bulk_vector_reads.append(nhash)
                self.in_bulk_read += 1
                self.peak_bulk_read = max(self.peak_bulk_read, self.in_bulk_read)
            try:
                if self.bulk_vector_delay:
                    time.sleep(self.bulk_vector_delay)
                return call()
            finally:
                with self.lock:
                    self.in_bulk_read -= 1
        if nhash:
            with self.lock:
                self.bulk_vector_reads.append(nhash)
        return call()


class _Conn:
    """A sqlite3 connection that reports to the spy; everything else is the real one."""

    def __init__(self, conn, spy):
        self._c = conn
        self._spy = spy

    def execute(self, sql, params=(), *rest):
        return self._spy.run(sql, params, lambda: self._c.execute(sql, params, *rest))

    def __enter__(self):
        self._c.__enter__()
        return self

    def __exit__(self, *exc):
        return self._c.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._c, name)


def spy_on(db):
    spy = Spy()
    real = db.connect

    def connect():
        with spy.lock:
            spy.connects += 1
        return _Conn(real(), spy)
    db.connect = connect
    return spy


# ------------------------------------------------------------------ embedders

class SwitchEmbedder(FakeEmbedder):
    """Fake vectors; can be switched off (raises), or made slow for chosen texts."""

    def __init__(self, dim=16):
        super().__init__(dim=dim)
        self.down = False
        self.slow_texts = set()
        self.slow_s = 0.0
        self.texts = []

    def embed(self, texts, timeout_s=None):
        if self.down:
            raise EmbedUnavailable("switched off (test)")
        if self.slow_s and any(t in self.slow_texts for t in texts):
            time.sleep(self.slow_s)
        self.texts.extend(texts)
        return super().embed(texts, timeout_s=timeout_s)


# ------------------------------------------------------------------- builders

@pytest.fixture()
def tmpdir_():
    d = os.path.join(tempfile.gettempdir(), "core-load-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_core(d, embedder=None, **extra):
    """A keyword + dense core. `extra` may carry settings the old version does not know."""
    known = {k: v for k, v in extra.items() if k in CoreConfig.__dataclass_fields__}
    unknown = {k: v for k, v in extra.items() if k not in CoreConfig.__dataclass_fields__}
    cfg = CoreConfig(db_path=os.path.join(d, "core.db"), paths=("keyword", "dense"),
                     embedder=embedder or SwitchEmbedder(), context_before=1, context_after=1,
                     embed_max_concurrency=4, **known)
    for k, v in unknown.items():
        setattr(cfg, k, v)
    return build(cfg)


def words(i, salt="w"):
    return f"{salt}{i % 7} item {i} note {salt} {i * 31 % 97} detail {i * 7 % 13}"


def records(n, prefix="b", salt="w", sessions=1):
    out = []
    for i in range(n):
        out.append(Record(record_id=f"{prefix}#{i}", session_id=f"s{i % sessions}", text=words(i, salt),
                          speaker="user", ts=f"2026-10-21T05:{i // 60:02d}:{i % 60:02d}.000Z"))
    return out


def dense(core, space, query, limit=50):
    return [(c.record_id, c.score) for c in core.recall_candidates(query, space, limit=limit, paths=("dense",))]


def top(core, space, query):
    got = dense(core, space, query)
    return got[0] if got else (None, None)


def run_threads(n, fn):
    out, errs = [None] * n, [None] * n
    start = threading.Barrier(n)

    def one(i):
        try:
            start.wait()
            out[i] = fn(i)
        except Exception as e:                    # noqa: BLE001  (kept per thread, checked by the test)
            errs[i] = e
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    return out, errs


# ------------------------------------------------------- the cost of a search

def test_second_search_does_not_read_the_whole_space_again(tmpdir_):
    """Two searches in one space read the space's records once in total (before: once per search)."""
    core = make_core(tmpdir_)
    core.write(records(30), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    core.search("item 3 note", space_id="u1", top_k=5)
    core.search("detail 9 w2", space_id="u1", top_k=5)
    assert spy.full_reads == 1, f"full reads of the space over two searches: {spy.full_reads} (want 1)"


def test_second_search_does_not_fetch_the_space_vectors_again(tmpdir_):
    """The second search fetches no stored vectors in bulk (before: all of the space's, every search)."""
    core = make_core(tmpdir_)
    core.write(records(30), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    core.search("item 3 note", space_id="u1", top_k=5)
    spy.reset_counts()
    core.search("detail 9 w2", space_id="u1", top_k=5)
    bulk = [n for n in spy.bulk_vector_reads if n > 1]
    assert bulk == [], f"bulk vector reads in the second search (hashes asked each time): {bulk} (want none)"


def test_one_search_opens_the_same_number_of_connections_whatever_top_k(tmpdir_):
    """Neighbours are read over one connection: 40 hits cost no more connections than 5."""
    core = make_core(tmpdir_)
    core.write(records(90, sessions=3), space_id="u1", batch_id="b")
    core.search("item 1 note", space_id="u1", top_k=5)            # warm whatever is kept in memory
    spy = spy_on(core.db)
    core.search("item 2 note", space_id="u1", top_k=5)
    small = spy.connects
    spy.reset_counts()
    hits = core.search("item 4 note", space_id="u1", top_k=40)
    big = spy.connects
    assert len(hits) >= 30, f"the test needs many hits, got {len(hits)}"
    assert big == small, f"connections for top_k=5: {small}, for top_k=40: {big} (want the same)"


def test_heavy_part_runs_at_most_n_at_once(tmpdir_):
    """8 searches on 8 cold spaces at once, limit 2: at most 2 spaces are being read at any moment."""
    core = make_core(tmpdir_, search_heavy_concurrency=2, search_heavy_wait_s=60.0)
    for s in range(8):
        core.write(records(20, prefix=f"b{s}", salt=f"x{s}"), space_id=f"u{s}", batch_id=f"b{s}")
    spy = spy_on(core.db)
    spy.full_read_delay = 0.3
    out, errs = run_threads(8, lambda i: core.search(f"item {i} note", space_id=f"u{i}", top_k=5))
    assert errs == [None] * 8, f"every search must answer while waiting its turn: {errs}"
    assert spy.peak_full_read <= 2, f"spaces being read at the same moment: {spy.peak_full_read} (want <= 2)"


def test_waiting_past_the_limit_raises_store_busy(tmpdir_):
    """Limit 1, wait 0.2 s, the heavy part takes 1 s: of two searches at once, one gets StoreBusy (-> 503)."""
    core = make_core(tmpdir_, search_heavy_concurrency=1, search_heavy_wait_s=0.2)
    for s in range(2):
        core.write(records(20, prefix=f"b{s}", salt=f"x{s}"), space_id=f"u{s}", batch_id=f"b{s}")
    spy = spy_on(core.db)
    spy.full_read_delay = 1.0
    out, errs = run_threads(2, lambda i: core.search(f"item {i} note", space_id=f"u{i}", top_k=5))
    busy = [e for e in errs if isinstance(e, StoreBusy)]
    assert len(busy) == 1, f"want exactly one StoreBusy, got errors {errs}"


def test_cold_space_hit_by_32_searches_is_built_once(tmpdir_):
    """32 searches arrive at one cold space together: the space is read once, all 32 get the same dense order."""
    core = make_core(tmpdir_, search_heavy_concurrency=2, search_heavy_wait_s=60.0)
    core.write(records(40), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    spy.full_read_delay = 0.3
    out, errs = run_threads(32, lambda i: [rid for rid, _ in dense(core, "u1", "item 5 note detail")][:10])
    assert errs == [None] * 32, f"errors: {[e for e in errs if e]}"
    assert spy.full_reads == 1, f"full reads of the space: {spy.full_reads} (want 1)"
    assert all(o == out[0] and o for o in out), "every search must see the same non-empty dense order"


# ------------------------------------------- the query vector is not squeezed out

def test_query_vector_waits_for_a_briefly_full_gate(tmpdir_):
    """Every embedding slot is busy for 0.5 s: the query vector waits for one, the dense path still answers.

    Before: the query vector gives up at once when the gate is full, and dense comes back empty."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")           # vectors are computed at write time
    gate = core._vc._sem
    held = 0
    for _ in range(4):
        assert gate.acquire(timeout=1)
        held += 1
    threading.Timer(0.5, lambda: [gate.release() for _ in range(held)]).start()
    got = dense(core, "u1", "a query nobody asked before 1")
    assert got, "the dense path was skipped while the gate was full for only 0.5 s"


def test_query_vector_survives_a_slow_backfill(tmpdir_):
    """Records without vectors, the provider takes 2.5 s on them: the query vector is still computed.

    Before: the backfill and the query vector share one 2 s deadline; the backfill eats it, dense is empty."""
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb)
    emb.down = True
    core.write(records(20), space_id="u1", batch_id="b")           # stored without vectors
    emb.down = False
    emb.slow_texts = {r.text for r in records(20)}
    emb.slow_s = 2.5
    got = dense(core, "u1", "a query nobody asked before 2")
    assert got, "the dense path was skipped because the backfill used up the query vector's time"


def test_query_vector_survives_slow_reads_before_it(tmpdir_):
    """Reading the space's stored vectors takes 3.5 s (longer than the query vector's own 3 s): the query vector is still computed.

    Before: the 2 s deadline starts before that read, so it has run out when the query vector is asked for."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    spy.bulk_vector_delay = 3.5
    got = dense(core, "u1", "a query nobody asked before 3")
    assert got, "the dense path was skipped because the reads before the query vector took longer than 2 s"


# ------------------------------- regression guards: no stale answer after a change

def _exact(core, space, text):
    """The best dense score for a query equal to `text` (1.0 when that text's own vector is used)."""
    rid, score = top(core, space, text)
    return rid, (score or 0.0)


def test_replace_snapshot_is_seen_by_the_next_search(tmpdir_):
    """Regression guard. Search, then replace the space's snapshot with other text under the same ids."""
    core = make_core(tmpdir_)
    old = records(20, salt="alpha")
    new = records(20, salt="omega")
    core.write(old, space_id="u1", batch_id="b")
    assert _exact(core, "u1", old[5].text)[1] > 0.999
    core.replace_space_snapshot("u1", new, snapshot_id="snap-2")
    rid, score = _exact(core, "u1", new[5].text)
    assert rid == "b#5" and score > 0.999, f"after the snapshot: top {rid} score {score}"
    assert _exact(core, "u1", old[5].text)[1] < 0.999


def test_new_write_is_seen_by_the_next_search(tmpdir_):
    """Regression guard. Search, write one more batch through Core.write: it is found by meaning."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    dense(core, "u1", "warm up")
    extra = records(5, prefix="c", salt="zulu")
    core.write(extra, space_id="u1", batch_id="c")
    rid, score = _exact(core, "u1", extra[2].text)
    assert rid == "c#2" and score > 0.999


def test_write_in_a_caller_transaction_is_seen_by_the_next_search(tmpdir_):
    """Regression guard. Search, then write through write_in_tx inside the caller's own transaction."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    dense(core, "u1", "warm up")
    extra = records(5, prefix="t", salt="tango")
    with core.db.writing() as conn:
        conn.execute("BEGIN IMMEDIATE")
        core.write_in_tx(conn, extra, space_id="u1", batch_id="t")
        conn.commit()
    rid, score = _exact(core, "u1", extra[1].text)
    assert rid == "t#1" and score > 0.999


def test_write_during_a_slow_build_is_seen_afterwards(tmpdir_):
    """Regression guard. While a search is still reading the space (slow), a new batch is written.
    A search after both have finished finds the new batch by meaning."""
    core = make_core(tmpdir_, search_heavy_concurrency=2, search_heavy_wait_s=60.0)
    core.write(records(20), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    spy.full_read_delay = 0.8
    t = threading.Thread(target=lambda: dense(core, "u1", "slow first search"))
    t.start()
    time.sleep(0.2)
    extra = records(5, prefix="d", salt="delta")
    core.write(extra, space_id="u1", batch_id="d")
    t.join(30)
    spy.full_read_delay = 0.0
    rid, score = _exact(core, "u1", extra[3].text)
    assert rid == "d#3" and score > 0.999, f"the batch written during the build: top {rid} score {score}"


# --------------------------------- the kept index: counters, builds, memory budget

def _gens(core, *keys):
    conn = core.db.connect()
    try:
        got = dict(conn.execute("SELECT key, gen FROM core_generations").fetchall())
    finally:
        conn.close()
    return tuple(got.get(k, 0) for k in keys)


def test_counters_only_grow_through_writes_and_a_wipe(tmpdir_):
    """Every write bumps its space's counter; a wipe bumps every counter and the epoch — never back to 0."""
    from core.db import Database
    core = make_core(tmpdir_)
    core.write(records(5), space_id="u1", batch_id="b1")
    a = _gens(core, "epoch", "space:u1", "vectors:fake:16")
    core.write(records(5, prefix="c", salt="new"), space_id="u1", batch_id="c1")
    b = _gens(core, "epoch", "space:u1", "vectors:fake:16")
    Database(core.db.path).wipe()
    c = _gens(core, "epoch", "space:u1", "vectors:fake:16")
    core.write(records(5), space_id="u1", batch_id="b1")
    d = _gens(core, "epoch", "space:u1", "vectors:fake:16")
    assert b[1] > a[1] and b[2] > a[2], (a, b)
    assert c[0] > b[0] and c[1] > b[1] and c[2] > b[2], f"the wipe must bump every counter: {b} -> {c}"
    assert d[1] > c[1] and d[2] > c[2], (c, d)


def test_an_index_built_while_the_space_changed_is_not_kept(tmpdir_):
    """A write lands while the space is being read: that build is used once, not kept; the next search reads again
    and sees the write."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    spy.full_read_delay = 0.6
    t = threading.Thread(target=lambda: dense(core, "u1", "first"))
    t.start()
    time.sleep(0.2)
    extra = records(3, prefix="e", salt="echo")
    core.write(extra, space_id="u1", batch_id="e")
    t.join(30)
    spy.full_read_delay = 0.0
    stats = core.recalls[1].index.stats()
    assert stats["builds_not_kept"] == 1, stats
    before = spy.full_reads
    rid, score = _exact(core, "u1", extra[0].text)
    assert spy.full_reads == before + 1 and rid == "e#0" and score > 0.999


def test_a_failed_build_is_not_kept(tmpdir_):
    """The first build fails (a database error): the search raises it, nothing is kept, the next search builds."""
    import sqlite3
    core = make_core(tmpdir_)
    core.write(records(10), space_id="u1", batch_id="b")
    store = core.store
    real = store.iter_records
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("disk I/O error (test)")
        return real(*a, **k)
    store.iter_records = flaky
    with pytest.raises(sqlite3.OperationalError):
        dense(core, "u1", "first")
    assert core.recalls[1].index.stats()["spaces"] == 0
    rid, score = _exact(core, "u1", records(10)[2].text)
    assert rid == "b#2" and score > 0.999 and calls["n"] == 2


def test_new_vectors_elsewhere_do_not_reread_a_complete_space(tmpdir_):
    """Another space gets new vectors: a kept space with every vector is reused as is, no read at all."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    dense(core, "u1", "warm")
    spy = spy_on(core.db)
    core.write(records(5, prefix="o", salt="other"), space_id="u2", batch_id="o")   # bumps vectors:<model>
    spy.reset_counts()
    dense(core, "u1", "again")
    assert spy.full_reads == 0 and [n for n in spy.bulk_vector_reads if n > 1] == []


def test_a_space_waiting_for_vectors_looks_up_only_those(tmpdir_):
    """A kept space with 6 records still without vectors; the backfill script's way (another writer) adds them:
    the next search looks up those 6 (no full read of the space) and ranks them."""
    from core.vector_cache import VectorCache
    from core.vector_store import VectorStore
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb, dense_backfill_m=0)
    core.write(records(20), space_id="u1", batch_id="b")
    emb.down = True
    late = records(6, prefix="l", salt="lima")
    core.write(late, space_id="u1", batch_id="l")
    emb.down = False
    assert _exact(core, "u1", late[1].text)[0] != "l#1"
    VectorCache(VectorStore(core.db), SwitchEmbedder()).get_or_compute([r.text for r in late])
    spy = spy_on(core.db)
    rid, score = _exact(core, "u1", late[1].text)
    assert rid == "l#1" and score > 0.999
    assert spy.full_reads == 0 and sorted(n for n in spy.bulk_vector_reads if n > 1) == [6], spy.bulk_vector_reads


def test_shared_texts_are_pooled_once(tmpdir_):
    """Two spaces with the same 30 texts: the pool holds 30 vectors, not 60."""
    core = make_core(tmpdir_)
    core.write(records(30), space_id="u1", batch_id="b")
    core.write(records(30), space_id="u2", batch_id="b")
    dense(core, "u1", "q1")
    dense(core, "u2", "q2")
    st = core.recalls[1].index.stats()
    assert st["spaces"] == 2 and st["pool_vectors"] == 30, st


def test_the_memory_budget_holds_and_evicted_vectors_leave_the_pool(tmpdir_):
    """Budget for about one space: after three spaces are searched the kept bytes are within the budget, the
    least recently used spaces are gone, and the pool holds only the kept space's vectors. Evicted spaces still
    answer (they are rebuilt)."""
    dim = 64
    core = make_core(tmpdir_, embedder=SwitchEmbedder(dim=dim), search_cache_bytes=30000)   # one space ~23 KB counted, two do not fit
    for s in range(3):
        core.write(records(40, prefix=f"b{s}", salt=f"s{s}x"), space_id=f"u{s}", batch_id=f"b{s}")
    for s in range(3):
        assert dense(core, f"u{s}", f"query {s}")
    st = core.recalls[1].index.stats()
    assert st["pool_bytes"] + st["index_bytes"] <= st["budget_bytes"], st
    assert st["spaces"] == 1 and st["pool_vectors"] == 40 and st["evictions"] == 2, st
    rid, score = _exact(core, "u0", records(40, prefix="b0", salt="s0x")[7].text)
    assert rid == "b0#7" and score > 0.999


def test_a_wipe_drops_what_was_kept_even_for_data_without_a_counter(tmpdir_):
    """Data written before the counters existed has no "space:<id>" row. A wipe still drops what was kept for
    it (the epoch), and the pool is emptied: the next search sees what was written after the wipe."""
    from core.db import Database
    core = make_core(tmpdir_)

    def forget_counter():
        conn = core.db.connect()
        conn.execute("DELETE FROM core_generations WHERE key='space:u1'")  # gate:allow-secret
        conn.commit()
        conn.close()
    old, new = records(10, salt="alpha"), records(10, salt="omega")
    core.write(old, space_id="u1", batch_id="b")
    forget_counter()
    assert _exact(core, "u1", old[3].text)[1] > 0.999
    Database(core.db.path).wipe()
    core.write(new, space_id="u1", batch_id="b")
    forget_counter()
    rid, score = _exact(core, "u1", new[3].text)
    assert rid == "b#3" and score > 0.999, f"after the wipe: top {rid} score {score}"
    st = core.recalls[1].index.stats()
    assert st["pool_vectors"] == 10, st


# ------------------------- waiting, refreshing and leases (the code review's three points)

def test_a_search_waiting_for_another_s_build_gives_up_at_the_wait_limit(tmpdir_):
    """Two searches arrive at a cold space; the build takes 1 s, the wait limit is 0.3 s. The one that builds
    answers; the one that waits for it raises SearchBusy (503) at the limit — not 300 s later."""
    from core.dense_index import SearchBusy
    core = make_core(tmpdir_, search_heavy_concurrency=2, search_heavy_wait_s=0.3)
    core.write(records(20), space_id="u1", batch_id="b")
    spy = spy_on(core.db)
    spy.full_read_delay = 1.0
    started = time.monotonic()
    out, errs = run_threads(2, lambda i: dense(core, "u1", f"cold {i}"))
    elapsed = time.monotonic() - started
    busy = [e for e in errs if isinstance(e, SearchBusy)]
    assert len(busy) == 1 and sum(1 for o in out if o) == 1, (out, errs)
    assert spy.full_reads == 1 and elapsed < 5, (spy.full_reads, elapsed)
    assert core.recalls[1].index.stats()["waits_timed_out"] == 1


def test_new_vectors_for_a_space_with_many_missing_are_looked_up_once_under_the_gate(tmpdir_):
    """A kept space with 2000 records still without vectors; the backfill (another writer) adds them; 32
    searches arrive at once. The 2000 are looked up ONCE (one refresh), never more refresh reads at a time than
    the gate allows, and every search ranks them."""
    from core.vector_cache import VectorCache
    from core.vector_store import VectorStore
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb, dense_backfill_m=0, search_heavy_concurrency=1, search_heavy_wait_s=60.0)
    core.write(records(30), space_id="u1", batch_id="b")
    emb.down = True
    late = records(2000, prefix="m", salt="mike")
    core.write(late, space_id="u1", batch_id="m")
    emb.down = False
    dense(core, "u1", "warm")
    VectorCache(VectorStore(core.db), SwitchEmbedder()).get_or_compute([r.text for r in late])
    spy = spy_on(core.db)
    spy.bulk_vector_delay = 0.2
    before = core.recalls[1].index.stats()["refreshes"]
    out, errs = run_threads(32, lambda i: _exact(core, "u1", late[i * 7].text))
    assert errs == [None] * 32, [e for e in errs if e]
    assert all(rid == f"m#{i * 7}" and score > 0.999 for i, (rid, score) in enumerate(out)), out[:3]
    assert core.recalls[1].index.stats()["refreshes"] - before == 1
    assert sum(spy.bulk_vector_reads[k] for k in range(len(spy.bulk_vector_reads))
               if spy.bulk_vector_reads[k] > 1) == len({r.text for r in late}), spy.bulk_vector_reads
    assert spy.peak_bulk_read <= 1 and spy.full_reads == 0, (spy.peak_bulk_read, spy.full_reads)


def _pause_scores_for(core, space_id):
    """While a search of *space_id* is inside scores(), stop it there until released (it holds its lease)."""
    index = core.recalls[1].index
    real = index.scores
    entered, release = threading.Event(), threading.Event()

    def paused(idx, q):
        out = real(idx, q)
        if idx.space_id == space_id:
            entered.set()
            release.wait(20)
        return out
    index.scores = paused
    return entered, release


def test_eviction_never_releases_an_index_a_search_is_reading(tmpdir_):
    """Budget for one space. A search of u0 is inside scores(); a search of u1 needs the room. u0 is the least
    recently used, but it is being read: it is not released (it stays kept and counted). The room is made by
    evicting what nobody reads (u1, once its own search is done); the budget holds throughout once that search
    ends, and u0 is still there after its reader."""
    dim = 64
    core = make_core(tmpdir_, embedder=SwitchEmbedder(dim=dim), search_cache_bytes=30000)
    for s in range(2):
        core.write(records(40, prefix=f"b{s}", salt=f"s{s}x"), space_id=f"u{s}", batch_id=f"b{s}")
    index = core.recalls[1].index
    entered, release = _pause_scores_for(core, "u0")
    t = threading.Thread(target=lambda: dense(core, "u0", "reader"))
    t.start()
    assert entered.wait(10)
    assert dense(core, "u1", "the other space")
    during = index.stats()
    assert list(index._spaces) == ["u0"], f"u0 must stay kept while it is read: {list(index._spaces)}"
    assert during["pool_vectors"] == 40 and during["evictions"] == 1 and during["leased"] == 1, during
    assert during["pool_bytes"] + during["index_bytes"] <= during["budget_bytes"], during
    release.set()
    t.join(20)
    after = index.stats()
    assert list(index._spaces) == ["u0"] and after["pool_vectors"] == 40 and after["leased"] == 0, after
    assert after["pool_bytes"] + after["index_bytes"] <= after["budget_bytes"], after


def test_an_index_replaced_while_it_is_read_stays_counted_until_its_reader_is_done(tmpdir_):
    """A search of u1 is reading; u1 changes and the next search builds a new index. The old one is retired
    but stays counted (its vectors are still in use) until the first search is done, then it is released."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    index = core.recalls[1].index
    entered, release = _pause_scores_for(core, "u1")
    t = threading.Thread(target=lambda: dense(core, "u1", "reader"))
    t.start()
    assert entered.wait(10)
    index.scores = index.__class__.scores.__get__(index)              # later searches score normally
    extra = records(5, prefix="n", salt="november")
    core.write(extra, space_id="u1", batch_id="n")
    assert _exact(core, "u1", extra[1].text)[0] == "n#1"
    during = index.stats()
    assert during["retired_waiting"] == 1, during
    counted_during = during["index_bytes"]
    release.set()
    t.join(20)
    after = index.stats()
    assert after["retired_waiting"] == 0 and after["leased"] == 0 and after["index_bytes"] < counted_during, after


def test_the_index_size_counts_its_strings_and_lists(tmpdir_):
    """A space's counted index bytes include every id and hash string and the lists that hold them."""
    import sys
    core = make_core(tmpdir_)
    core.write(records(50), space_id="u1", batch_id="b")
    index = core.recalls[1].index
    idx = index.get("u1")
    try:
        floor = (sum(sys.getsizeof(i) for i in idx.ids) + sum(sys.getsizeof(h) for h in idx.hashes)
                 + sys.getsizeof(idx.ids) + sys.getsizeof(idx.hashes) + sys.getsizeof(idx.vecs))
        assert idx.index_bytes >= floor, (idx.index_bytes, floor)
    finally:
        index.done(idx)


def test_a_snapshot_replace_drops_the_old_pending_vector_rows(tmpdir_):
    """Records written while the provider was down leave pending rows (why they have no vector). A snapshot
    replace of the space removes them with the old rows: they explained texts that are gone."""
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb)
    emb.down = True
    core.write(records(6, salt="old"), space_id="u1", batch_id="b")
    core.write(records(3, prefix="o", salt="other"), space_id="u2", batch_id="o")

    def pending(space):
        conn = core.db.connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM core_pending_vectors WHERE space_id=?", (space,)).fetchone()[0]
        finally:
            conn.close()
    assert pending("u1") == 6 and pending("u2") == 3
    core.replace_space_snapshot("u1", records(4, salt="new"), snapshot_id="s2")
    assert pending("u1") == 0 and pending("u2") == 3


def test_refreshes_of_several_spaces_take_turns_at_the_gate(tmpdir_):
    """Four kept spaces each wait for 300 vectors; the backfill adds them all; one search per space arrives at
    once with one heavy slot. The four refreshes run one at a time (the gate), each once."""
    from core.vector_cache import VectorCache
    from core.vector_store import VectorStore
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb, dense_backfill_m=0, search_heavy_concurrency=1, search_heavy_wait_s=60.0)
    lates = {}
    for s in range(4):
        core.write(records(10, prefix=f"b{s}", salt=f"k{s}"), space_id=f"u{s}", batch_id=f"b{s}")
        emb.down = True
        lates[s] = records(300, prefix=f"m{s}", salt=f"miss{s}x")
        core.write(lates[s], space_id=f"u{s}", batch_id=f"m{s}")
        emb.down = False
        dense(core, f"u{s}", "warm")
    VectorCache(VectorStore(core.db), SwitchEmbedder()).get_or_compute([r.text for s in lates for r in lates[s]])
    spy = spy_on(core.db)
    spy.bulk_vector_delay = 0.3
    before = core.recalls[1].index.stats()["refreshes"]
    out, errs = run_threads(4, lambda i: _exact(core, f"u{i}", lates[i][5].text))
    assert errs == [None] * 4 and [rid for rid, _ in out] == [f"m{i}#5" for i in range(4)], (out, errs)
    assert core.recalls[1].index.stats()["refreshes"] - before == 4
    assert spy.peak_bulk_read == 1, f"refreshes running at the same moment: {spy.peak_bulk_read} (gate of 1)"


def test_a_write_landing_before_a_refresh_takes_its_slot_is_seen_afterwards(tmpdir_):
    """A kept space waits for vectors; the backfill adds them; a search decides to refresh, and before it takes
    its heavy-work slot (so before the refresh opens its snapshot) a new batch is written to the space. The
    refresh sees the space changed and reads it in full (counted as a build, not a refresh); the new batch is
    found. (A write landing AFTER the snapshot is the next test.)"""
    from core.vector_cache import VectorCache
    from core.vector_store import VectorStore
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb, dense_backfill_m=0)
    core.write(records(20), space_id="u1", batch_id="b")
    emb.down = True
    late = records(5, prefix="l", salt="lima")
    core.write(late, space_id="u1", batch_id="l")
    emb.down = False
    dense(core, "u1", "warm")                                        # kept, 5 records still without a vector
    VectorCache(VectorStore(core.db), SwitchEmbedder()).get_or_compute([r.text for r in late])
    index = core.recalls[1].index
    real_heavy = index._heavy
    extra = records(3, prefix="z", salt="zulu")
    landed = []

    def heavy(what):
        if what.startswith("refreshing") and not landed:
            core.write(extra, space_id="u1", batch_id="z")             # lands between the counters read and the refresh
            landed.append(1)
        return real_heavy(what)
    index._heavy = heavy
    before = index.stats()
    dense(core, "u1", "the refreshing search")
    assert landed, "the test needs the refresh path"
    index._heavy = real_heavy
    after = index.stats()
    assert (after["builds"] - before["builds"], after["refreshes"] - before["refreshes"]) == (1, 0), (before, after)
    rid, score = _exact(core, "u1", extra[1].text)
    assert rid == "z#1" and score > 0.999, f"the batch written during the refresh wait: top {rid} score {score}"
    rid, score = _exact(core, "u1", late[2].text)
    assert rid == "l#2" and score > 0.999


def test_an_index_used_but_not_kept_is_counted_as_leased_while_read(tmpdir_):
    """The counters moved during a build, so the index is used once and not kept; while its search reads it,
    stats() counts it as leased; when the search is done, nothing is leased."""
    core = make_core(tmpdir_)
    core.write(records(20), space_id="u1", batch_id="b")
    index = core.recalls[1].index
    real = index._current_gens
    calls = {"n": 0}

    def moving(space_id):
        g = real(space_id)
        calls["n"] += 1
        return g if calls["n"] == 1 else (g[0], g[1] + 1, g[2])       # "after the build" the space looks changed
    index._current_gens = moving
    entered, release = _pause_scores_for(core, "u1")
    t = threading.Thread(target=lambda: dense(core, "u1", "reader"))
    t.start()
    assert entered.wait(10)
    during = index.stats()
    assert during["spaces"] == 0 and during["builds_not_kept"] == 1 and during["leased"] == 1, during
    release.set()
    t.join(20)
    assert index.stats()["leased"] == 0


def test_a_write_landing_after_a_refresh_read_its_snapshot_is_not_kept_and_is_seen_next(tmpdir_):
    """A kept space waits for vectors; the backfill adds them; a search refreshes, and AFTER the refresh has read
    its snapshot (before the "did the counters move?" check) a new batch is written. That refreshed index is used
    for its search but not kept; the next search reads the space again and finds the new batch."""
    from core.vector_cache import VectorCache
    from core.vector_store import VectorStore
    emb = SwitchEmbedder()
    core = make_core(tmpdir_, embedder=emb, dense_backfill_m=0)
    core.write(records(20), space_id="u1", batch_id="b")
    emb.down = True
    late = records(5, prefix="l", salt="lima")
    core.write(late, space_id="u1", batch_id="l")
    emb.down = False
    dense(core, "u1", "warm")
    VectorCache(VectorStore(core.db), SwitchEmbedder()).get_or_compute([r.text for r in late])
    index = core.recalls[1].index
    real_refresh = index._refresh_missing
    extra = records(3, prefix="y", salt="yankee")

    def refresh_then_write(base, space_id):
        out = real_refresh(base, space_id)                           # the snapshot has been read and closed
        core.write(extra, space_id=space_id, batch_id="y")
        return out
    index._refresh_missing = refresh_then_write
    before = index.stats()
    dense(core, "u1", "the refreshing search")
    index._refresh_missing = real_refresh
    after = index.stats()
    assert after["refreshes"] - before["refreshes"] == 1 and after["builds_not_kept"] - before["builds_not_kept"] == 1, \
        (before, after)
    rid, score = _exact(core, "u1", extra[2].text)
    assert rid == "y#2" and score > 0.999, f"the batch written after the refresh snapshot: top {rid} score {score}"
