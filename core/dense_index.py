"""The in-memory dense index: each space's "record -> vector" table, kept between searches.

Why: the dense path used to re-read a whole space on every search (all its texts, a sha256 of each, every
vector fetched one SQLite row at a time) before it could rank anything. With 32 searches at once on a big space
that cost was paid 32 times over, on 2 CPUs.

What it keeps
  - a shared vector pool: one numpy vector per (model, content hash), with a reference count. A text that
    appears in many spaces is held once.
  - per space: the ids of its dense-eligible records and a reference to each one's pooled vector, plus the
    records that have no vector yet (with their text, for the search-time backfill).

When it is rebuilt (the counters live in core_generations; see core/db.py)
  - every use first reads three counters with one indexed statement: "epoch" (a wipe), "space:<id>" (a change
    to that space's records) and "vectors:<model>" (new vectors). The counters are bumped in the same
    transaction as the change, by every writer: Core.write / write_in_tx, replace_space_snapshot, the vector
    store (write-time vectors, search-time backfill, the backfill script) and the wipe.
  - epoch changed -> everything is dropped (the pool too: a wiped database must not lend old vectors);
    the space's counter changed -> the space is rebuilt; only the vectors counter changed -> a space that
    still waits for vectors looks those up (only those records' vectors, no full read); a space without
    missing vectors is unaffected.
  - an index is built (or refreshed) from one read snapshot and tagged with the counters read in that
    snapshot; if the counters moved meanwhile it is used for that search but not kept.

How much work runs at once
  - building or refreshing an index and computing similarities ("the heavy part") run under one gate of
    `heavy_concurrency` slots; a search that cannot get a slot within `heavy_wait_s` raises SearchBusy (the
    service answers 503, which the platform retries).
  - searches that arrive at a space that needs a build OR a refresh wait for ONE such job per (space,
    counters); a failed job is not kept. Waiting for another search's job is bounded by the same
    `heavy_wait_s` (past it: SearchBusy), so no search waits longer than that on another's behalf.

Memory
  - a byte budget covers the pool (each vector's bytes plus the array's own overhead) and every space's
    index (its id, hash and vector lists, the strings in them, and the texts of records still without a
    vector); after a build is kept, least-recently-used spaces are dropped until the total fits, and their
    vectors leave the pool when no other space refers to them.
  - a search holds a lease on the index it uses (get() .. done()). An index with an active lease is never
    released: eviction skips it, and an index that is replaced or dropped (a newer build, a wipe) while
    leased stays counted until its last reader is done, then it is released. So the counted bytes are never
    below what the kept and leased indexes really hold; while every kept index is leased the total may stand
    above the budget until those readers finish, then the budget is restored.
  - outside the budget, bounded by the gate: the similarity scratch (CHUNK vectors at a time) and the data
    read during a build or refresh, for at most `heavy_concurrency` searches at once; and an index that is
    used but not kept (counters moved during its build) lives only as long as its searches.
"""
import contextlib
import logging
import os
import sys
import threading
from collections import OrderedDict

from .db import (EPOCH_KEY, GENERATIONS, StoreBusy, space_generation_key,
                 vectors_generation_key)
from .vector_helper import content_hash as _content_hash

log = logging.getLogger("core.dense_index")

CHUNK = 2048                      # vectors stacked at a time when scoring
ARRAY_OVERHEAD = 112              # a numpy array object's own bytes, on top of its data
DEFAULT_HEAVY_WAIT_S = 15.0
DEFAULT_BUDGET_BYTES = 640 * 2 ** 20


class SearchBusy(StoreBusy):
    """No slot for the heavy part within the wait limit. Unlike other StoreBusy on the dense path (which only
    drop the dense half of one search), this fails the search: the caller answers 503 and the client retries."""


def effective_cpus():
    """CPUs this process may use: the cgroup quota when there is one (a container), else the machine's count."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return max(1, -(-int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, os.cpu_count() or 1)


def _index_bytes(ids, hashes, vecs, missing):
    """What one space's index holds besides the pooled vector data: the three parallel lists, every id and
    hash string, and each missing record's tuple with its id, hash and text."""
    size = sys.getsizeof
    total = size(ids) + size(hashes) + size(vecs) + size(missing)
    total += sum(size(i) for i in ids) + sum(size(h) for h in hashes)
    total += sum(size(m) + size(m[0]) + size(m[1]) + size(m[2]) for m in missing)
    return total + 256                # the SpaceIndex object itself and its fields


class SpaceIndex:
    """One space: dense-eligible record ids with their pooled vectors, and the ones still without a vector."""
    __slots__ = ("space_id", "gens", "ids", "hashes", "vecs", "missing", "index_bytes",
                 "readers", "pool", "retired")

    def __init__(self, space_id, gens, ids, hashes, vecs, missing):
        self.space_id = space_id
        self.gens = gens                  # (epoch, space, vectors) counters this index reflects
        self.ids = ids                    # record ids that have a vector ...
        self.hashes = hashes              # ... their content hashes ...
        self.vecs = vecs                  # ... and their vectors (same order)
        self.missing = missing            # [(record_id, content_hash, text)] without a vector yet
        self.index_bytes = _index_bytes(ids, hashes, vecs, missing)
        self.readers = 0                  # searches holding a lease on it (get() .. done())
        self.pool = None                  # the pool its vectors are counted in, while it is counted
        self.retired = False              # no longer the kept index of its space; released at its last done()


class _Flight:
    __slots__ = ("gens", "event", "result", "error")

    def __init__(self, gens):
        self.gens = gens
        self.event = threading.Event()
        self.result = None
        self.error = None


class DenseIndex:
    def __init__(self, db, store, vector_store, model, exclude_lines=(), heavy_concurrency=None,
                 heavy_wait_s=None, budget_bytes=None):
        self._db = db
        self._store = store
        self._vec = vector_store
        self._model = model
        self._exclude = tuple(exclude_lines or ())
        n = heavy_concurrency if heavy_concurrency else effective_cpus()
        self.heavy_concurrency = max(1, int(n))
        self.heavy_wait_s = DEFAULT_HEAVY_WAIT_S if heavy_wait_s is None else float(heavy_wait_s)
        self.budget_bytes = DEFAULT_BUDGET_BYTES if budget_bytes is None else int(budget_bytes)
        self._gate = threading.BoundedSemaphore(self.heavy_concurrency)
        self._lock = threading.Lock()
        self._epoch = None
        self._spaces = OrderedDict()      # space_id -> SpaceIndex, least recently used first
        self._pool = {}                   # content_hash -> [vector, references] (the current epoch's pool)
        self._pool_bytes = 0              # every counted vector: the current pool, and old ones still leased
        self._index_bytes = 0
        self._retired = set()             # replaced / dropped indexes still leased: counted until released
        self._leased = set()              # every index with an active reader (kept, retired, or used but not kept)
        self._flights = {}                # space_id -> _Flight (a build or refresh in progress)
        self._counts = {"builds": 0, "builds_not_kept": 0, "evictions": 0, "refreshes": 0, "busy": 0,
                        "waits_timed_out": 0}

    # ---------------------------------------------------------------- the gate

    @contextlib.contextmanager
    def _heavy(self, what):
        """One slot of the heavy-work gate for the block, or SearchBusy past the wait limit."""
        if not self._gate.acquire(timeout=self.heavy_wait_s):
            with self._lock:
                self._counts["busy"] += 1
            raise SearchBusy(f"no slot for {what} within {self.heavy_wait_s:g}s "
                             f"({self.heavy_concurrency} at a time)")
        try:
            yield
        finally:
            self._gate.release()

    # -------------------------------------------------------------- counters

    def _read_gens(self, conn, space_id):
        keys = (EPOCH_KEY, space_generation_key(space_id), vectors_generation_key(self._model))
        got = dict(conn.execute(f"SELECT key, gen FROM {GENERATIONS} WHERE key IN (?,?,?)", keys).fetchall())
        return tuple(got.get(k, 0) for k in keys)

    def _current_gens(self, space_id):
        conn = self._db.connect()
        try:
            return self._read_gens(conn, space_id)
        finally:
            conn.close()

    # ------------------------------------------------------- the pool (lock held)

    def _keep(self, idx):
        """Make *idx* the kept index of its space: its vectors join the pool, the old one is retired."""
        vecs = idx.vecs
        for i, h in enumerate(idx.hashes):
            slot = self._pool.get(h)
            if slot is None:
                self._pool[h] = [vecs[i], 1]
                self._pool_bytes += vecs[i].nbytes + ARRAY_OVERHEAD
            else:
                slot[1] += 1
                vecs[i] = slot[0]         # share the pooled copy
        idx.pool = self._pool
        self._index_bytes += idx.index_bytes
        old = self._spaces.pop(idx.space_id, None)
        self._spaces[idx.space_id] = idx
        if old is not None:
            self._retire(old)

    def _retire(self, idx):
        """*idx* stops being kept: released now, or at its last done() while a search still reads it."""
        if idx.readers > 0:
            idx.retired = True
            self._retired.add(idx)
        else:
            self._release(idx)

    def _release(self, idx):
        """Give back one counted index (idempotent): its references leave its pool, its bytes leave the count."""
        pool = idx.pool
        if pool is None:
            return
        for h in idx.hashes:
            slot = pool.get(h)
            if slot is None:
                continue
            slot[1] -= 1
            if slot[1] <= 0:
                del pool[h]
                self._pool_bytes -= slot[0].nbytes + ARRAY_OVERHEAD
        self._index_bytes -= idx.index_bytes
        idx.pool = None
        self._retired.discard(idx)

    def _fit_budget(self):
        """Evict least-recently-used kept indexes that nobody is reading until the count fits the budget."""
        for space_id in list(self._spaces):
            if self._pool_bytes + self._index_bytes <= self.budget_bytes:
                return
            idx = self._spaces[space_id]
            if idx.readers > 0:
                continue                  # in use: never released under a reader
            del self._spaces[space_id]
            self._release(idx)
            self._counts["evictions"] += 1

    def _drop_all(self):
        """A wipe: nothing kept is valid. Leased indexes stay counted (in the old pool) until their last reader."""
        for idx in list(self._spaces.values()):
            self._retire(idx)
        self._spaces.clear()
        self._pool = {}                   # a new pool: no vector of before the wipe is lent to a new build

    def _lease(self, idx):
        """One more reader of *idx* (lock held)."""
        idx.readers += 1
        self._leased.add(idx)

    def done(self, idx):
        """End one search's lease on *idx* (call it after every get(), also when the search failed)."""
        with self._lock:
            idx.readers -= 1
            if idx.readers <= 0:
                self._leased.discard(idx)
                if idx.retired:
                    self._release(idx)
            self._fit_budget()

    def stats(self):
        with self._lock:
            return dict(self._counts, spaces=len(self._spaces), pool_vectors=len(self._pool),
                        pool_bytes=self._pool_bytes, index_bytes=self._index_bytes,
                        budget_bytes=self.budget_bytes, heavy_concurrency=self.heavy_concurrency,
                        leased=len(self._leased),
                        retired_waiting=len(self._retired))

    # ------------------------------------------------------------ build / refresh

    def _pooled(self, hashes, epoch):
        """Vectors the current pool already holds for *hashes* (none when *epoch* is not the current one)."""
        with self._lock:
            if epoch != self._epoch:
                return {}
            return {h: self._pool[h][0] for h in hashes if h in self._pool}

    def _build(self, space_id):
        """Read one space in one snapshot: its records, their vectors (pooled ones are not fetched again)."""
        import numpy as np
        conn = self._db.connect()
        try:
            conn.execute("BEGIN")                        # one read snapshot for counters, records and vectors
            gens = self._read_gens(conn, space_id)
            items = self._store.iter_records(space_id, exclude_lines=self._exclude, conn=conn)
            hashes = [_content_hash(t) for _, t in items]
            known = self._pooled(set(hashes), gens[0])
            need = [h for h in dict.fromkeys(hashes) if h not in known]
            got = self._vec.get_vectors(self._model, need, conn=conn) if need else {}
            conn.execute("COMMIT")
        finally:
            conn.close()
        local = dict(known)
        for h, b in got.items():
            local[h] = np.frombuffer(b, dtype=np.float32)
        ids, hs, vecs, missing = [], [], [], []
        for (rid, text), h in zip(items, hashes):
            v = local.get(h)
            if v is None:
                missing.append((rid, h, text))
            else:
                ids.append(rid)
                hs.append(h)
                vecs.append(v)
        return SpaceIndex(space_id, gens, ids, hs, vecs, missing)

    def _refresh_missing(self, base, space_id):
        """New vectors exist and *base* still waits for some: look up only those (no full read of the space),
        in one snapshot tagged with the counters read in it.

        The rest of the new index is copied from *base*, so that copy is only right while the space itself is
        what *base* was built from. If the snapshot shows the space (or the epoch) has changed since — a write
        landed while this search waited for the gate — the copy would carry the new counters with the old
        records and be kept for good; the space is read in full instead (a build in a snapshot of its own)."""
        import numpy as np
        conn = self._db.connect()
        try:
            conn.execute("BEGIN")
            gens = self._read_gens(conn, space_id)
            if gens[:2] != base.gens[:2]:
                conn.execute("COMMIT")
                conn.close()
                conn = None
                return self._build(space_id)
            hashes = list(dict.fromkeys(h for _, h, _ in base.missing))
            known = self._pooled(set(hashes), gens[0])
            need = [h for h in hashes if h not in known]
            got = self._vec.get_vectors(self._model, need, conn=conn) if need else {}
            conn.execute("COMMIT")
        finally:
            if conn is not None:
                conn.close()
        local = dict(known)
        for h, b in got.items():
            local[h] = np.frombuffer(b, dtype=np.float32)
        ids, hs, vecs = list(base.ids), list(base.hashes), list(base.vecs)
        missing = []
        for rid, h, text in base.missing:
            v = local.get(h)
            if v is None:
                missing.append((rid, h, text))
            else:
                ids.append(rid)
                hs.append(h)
                vecs.append(v)
        return SpaceIndex(space_id, gens, ids, hs, vecs, missing)

    # ------------------------------------------------------------------ get

    def get(self, space_id):
        """The current index of one space (built, refreshed or reused), LEASED to the caller: call done(idx)
        when finished with it. Raises SearchBusy past the wait limit."""
        gens = self._current_gens(space_id)
        with self._lock:
            if gens[0] != self._epoch:                     # a wipe (or the first use): nothing kept is valid
                self._drop_all()
                self._epoch = gens[0]
            idx = self._spaces.get(space_id)
            base = None
            if idx is not None and idx.gens[:2] == gens[:2]:
                if idx.gens[2] == gens[2] or not idx.missing:
                    idx.gens = gens
                    self._spaces.move_to_end(space_id)
                    self._lease(idx)
                    return idx
                base = idx                                 # only new vectors: refresh the missing ones
            fl = self._flights.get(space_id)
            owner = fl is None or fl.gens != gens
            if owner:
                fl = _Flight(gens)
                self._flights[space_id] = fl
        if not owner:
            # Waiting for another search's build / refresh is the heavy part too: the same limit applies.
            if not fl.event.wait(self.heavy_wait_s):
                with self._lock:
                    self._counts["waits_timed_out"] += 1
                raise SearchBusy(f"another search's index job for this space did not finish within "
                                 f"{self.heavy_wait_s:g}s")
            if fl.error is not None:
                raise fl.error
            with self._lock:
                self._lease(fl.result)
            return fl.result
        try:
            with self._heavy("refreshing a space index" if base is not None else "building a space index"):
                new = self._refresh_missing(base, space_id) if base is not None else self._build(space_id)
            after = self._current_gens(space_id)
            with self._lock:
                # a refresh that found its space changed read the space in full: count it as the build it was
                refreshed = base is not None and new.gens[:2] == base.gens[:2]
                self._counts["refreshes" if refreshed else "builds"] += 1
                self._lease(new)                           # the owner's lease, before anything can evict it
                if after == new.gens and new.gens[0] == self._epoch:
                    self._keep(new)
                    self._fit_budget()
                else:
                    self._counts["builds_not_kept"] += 1
            fl.result = new
            return new
        except BaseException as e:
            fl.error = e
            raise
        finally:
            with self._lock:
                if self._flights.get(space_id) is fl:
                    del self._flights[space_id]
            fl.event.set()

    # ----------------------------------------------------------------- score

    def scores(self, idx, query_vector):
        """Cosine-style scores (dot products) of every vector of *idx* against the query, in idx order.
        The caller holds a lease on *idx* (from get())."""
        import numpy as np
        n = len(idx.vecs)
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        q = np.asarray(query_vector, dtype=np.float32)
        with self._heavy("computing similarities"):
            for i in range(0, n, CHUNK):
                out[i:i + CHUNK] = np.stack(idx.vecs[i:i + CHUNK]) @ q
        return out
