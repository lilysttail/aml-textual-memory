"""The service under search load: /health, /search, the write edge cases, and the offline count.

Written before the change (test-first). Before the change most of these fail
because the result is wrong; the ones marked "regression guard" pass before the
change too and pin behaviour the change must keep.

  - /health does a fixed amount of work, however many spaces there are and
    however many vectors are missing; it no longer carries `pending_vectors`
    (that exact count has no cheap source); a missing table is still a 500.
  - /search no longer counts the space's missing vectors (it was only used for
    a response header nobody reads).
  - An empty user_id is an ordinary space name: what was added under it is
    found under it.
  - A batch whose messages all had no text still claims its request id: the
    same id sent again with content is the same write (a replay), not a new one.
  - The exact number of missing vectors comes from a read-only script, run by
    hand inside the container: scripts/count_missing_vectors.py <db> [--model M]
    prints one JSON line {"missing_vectors": N, ...}; it opens the database read
    only (no table or row is created or changed, the main file is not written;
    SQLite may use the WAL side files as any reader does), and fails (exit code
    not 0) on a database without the tables instead of reporting 0.
  - The wipe and backfill scripts, run while the service is up, are seen by the
    next search (regression guards).
"""
import gc
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)
os.environ["API_KEY"] = "test-key"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embedder import EmbedUnavailable  # noqa: E402
from core import Record  # noqa: E402
from core.testing import FakeEmbedder  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUTH = {"Authorization": "Bearer test-key"}


class SwitchEmbedder(FakeEmbedder):
    """Fake vectors (model "fake:16"); `down` makes every call fail like a provider outage."""

    def __init__(self):
        super().__init__(dim=16)
        self.down = False

    def embed(self, texts, timeout_s=None):
        if self.down:
            raise EmbedUnavailable("switched off (test)")
        return super().embed(texts, timeout_s=timeout_s)


_MADE = []


@pytest.fixture(autouse=True)
def _remove_databases():
    """Every database a test made is removed afterwards (with SQLite's side files)."""
    yield
    gc.collect()                       # connections the code under test left to the garbage collector hold the file open
    while _MADE:
        p = _MADE.pop()
        for f in (p, p + "-wal", p + "-shm"):
            try:
                os.remove(f)
            except OSError:
                pass


def _new_db_path(kind):
    p = os.path.join(_TMPROOT, f"{kind}_{uuid.uuid4().hex}.db")
    _MADE.append(p)
    return p


def _settings(**ov):
    d = dict(db_path=_new_db_path("sl"),
             api_key="test-key", recall_paths=("keyword", "dense"),
             rerank_enabled=False, rerank_n=200, store_busy_timeout_ms=5000,
             store_write_lock_timeout_ms=10000,
             embed_max_concurrency=4, embed_batch=32,
             embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
             bge_small_dir=None, neighbour_candidates=True,
             dense_backfill_m=8, dense_backfill_t=2.0)
    d.update(ov)
    return Settings(**d)


@pytest.fixture()
def emb(monkeypatch):
    e = SwitchEmbedder()
    monkeypatch.setattr(app_main, "_build_embedder", lambda s: e)
    return e


def _client(**ov):
    return TestClient(app_main.make_app(_settings(**ov)), raise_server_exceptions=False)


def _add(client, user, request_id, texts, session="s1"):
    r = client.post("/add", headers=AUTH, json={
        "request_id": request_id, "user_id": user, "session_id": session,
        "messages": [{"role": "user", "timestamp": 1728000000000 + i, "content": t} for i, t in enumerate(texts)]})
    assert r.status_code == 200, f"/add {request_id}: {r.status_code} {r.text[:200]}"


def _records(n, prefix, salt):
    return [Record(record_id=f"{prefix}#{i}", session_id="s1", text=f"{salt} line {i} about {salt}{i % 5}",
                   speaker="user", ts=f"2026-10-21T05:00:{i:02d}.000Z") for i in range(n)]


class StatementCount:
    """Counts statements run through one Database (every connection it hands out)."""

    def __init__(self, db):
        self.n = 0
        real = db.connect

        def connect():
            conn = real()
            conn.set_trace_callback(self._one)
            return conn
        db.connect = connect

    def _one(self, sql):
        self.n += 1


def _fill_spaces(core, n_spaces, per_space=3):
    for s in range(n_spaces):
        core.write(_records(per_space, prefix=f"b{s}", salt=f"sp{s}x"), space_id=f"u{s}", batch_id=f"b{s}")


# ------------------------------------------------------------------- /health

def test_health_work_does_not_grow_with_the_number_of_spaces(emb):
    """10 spaces or 300, every vector missing (the provider was down at write time), no search yet:
    one /health runs the same number of statements. Before: it re-reads every space."""
    emb.down = True
    counts = {}
    for n in (10, 300):
        with _client() as client:
            core = client.app.state.core
            _fill_spaces(core, n)
            spy = StatementCount(core.db)
            r = client.get("/health")
            assert r.status_code == 200, f"/health with {n} spaces: {r.status_code}"
            counts[n] = spy.n
    assert counts[300] == counts[10], f"statements in one /health: {counts} (want the same for 10 and 300 spaces)"


def test_health_no_longer_reports_pending_vectors(emb):
    """/health answers ok, build and config; the exact missing-vector count moved to the offline script."""
    with _client() as client:
        _add(client, "u1", "r1", ["connection pool exhausted in the database client"])
        body = client.get("/health").json()
        assert body.get("ok") is True
        assert "pending_vectors" not in body, f"/health still carries pending_vectors: {body.get('pending_vectors')!r}"


@pytest.mark.parametrize("table", ["core_vectors", "core_records", "core_generations"])
def test_health_is_500_when_a_needed_table_is_gone(emb, table):
    """Regression guard. A space nobody searched yet, then a needed table disappears: /health is 500, not ok."""
    with _client() as client:
        _add(client, "u1", "r1", ["connection pool exhausted in the database client"])
        conn = sqlite3.connect(client.app.state.core.db.path)
        conn.execute(f"DROP TABLE {table}")
        conn.commit()
        conn.close()
        r = client.get("/health")
        assert r.status_code == 500, f"/health with {table} gone: {r.status_code}"


# ------------------------------------------------------------------- /search

def test_search_does_not_count_missing_vectors(emb, monkeypatch):
    """A space with missing vectors: /search neither counts them nor sends X-Core-Pending-Vectors."""
    emb.down = True
    with _client() as client:
        _add(client, "u1", "r1", [f"note {i} about the release train" for i in range(5)])
        emb.down = False
        core = client.app.state.core
        calls = []
        for name in ("pending_vectors", "missing_vector_records"):     # the per-space count and its full read
            real = getattr(core, name, None)
            if real is not None:
                monkeypatch.setattr(core, name, lambda *a, _r=real, _n=name, **k: calls.append(_n) or _r(*a, **k))
        r = client.post("/search", headers=AUTH, json={"query": "release train", "user_id": "u1", "top_k": 5})
        assert r.status_code == 200
        assert calls == [] and "x-core-pending-vectors" not in {k.lower() for k in r.headers}, \
            f"missing-vector counting during /search: {calls}, header: {r.headers.get('X-Core-Pending-Vectors')!r}"


def test_search_answers_503_when_the_heavy_part_has_no_slot(emb, monkeypatch):
    """The dense index could not get a slot in time (SearchBusy): /search answers 503 (retried by the client),
    not 200 with the dense half silently missing."""
    from core.dense_index import SearchBusy
    with _client() as client:
        _add(client, "u1", "r1", ["the release train leaves on thursday"])
        index = client.app.state.core.recalls[1].index

        def busy(space_id):
            raise SearchBusy("no slot (test)")
        monkeypatch.setattr(index, "get", busy)
        r = client.post("/search", headers=AUTH, json={"query": "release train", "user_id": "u1", "top_k": 5})
        assert r.status_code == 503, (r.status_code, r.text[:200])


def test_the_service_passes_its_settings_to_the_dense_index(emb):
    """SEARCH_HEAVY_CONCURRENCY / SEARCH_HEAVY_WAIT_S / SEARCH_CACHE_MIB reach the running index."""
    with _client(search_heavy_concurrency=3, search_heavy_wait_s=4.5, search_cache_mib=77) as client:
        st = client.app.state.core.recalls[1].index.stats()
        assert (st["heavy_concurrency"], st["budget_bytes"]) == (3, 77 * 2 ** 20), st
        assert client.app.state.core.recalls[1].index.heavy_wait_s == 4.5


# ----------------------------------------------------------- write edge cases

def test_empty_user_id_is_an_ordinary_space(emb):
    """Regression guard. Added under user_id "" -> found under user_id ""."""
    with _client() as client:
        _add(client, "", "r1", ["the quarterly budget review moved to friday"])
        r = client.post("/search", headers=AUTH, json={"query": "quarterly budget review", "user_id": "", "top_k": 5})
        assert r.status_code == 200
        assert any("quarterly budget review" in d["content"] for d in r.json()["data"]), r.json()


def test_batch_without_text_still_claims_its_request_id(emb):
    """request_id "r1" with no text in any message, then "r1" again with text: the second is the same
    write sent again (a replay), so nothing new is stored. Before: the second one was written."""
    with _client() as client:
        _add(client, "u1", "r1", ["", "   "])
        _add(client, "u1", "r1", ["the vault code is kept in the blue folder"])
        r = client.post("/search", headers=AUTH, json={"query": "vault code blue folder", "user_id": "u1", "top_k": 5})
        assert r.status_code == 200
        assert r.json()["data"] == [], f"a replayed request id stored new rows: {r.json()['data']}"


# ------------------------------------------- scripts run while the service is up

def _dense_top(core, space, text):
    got = core.recall_candidates(text, space, limit=10, paths=("dense",))
    return (got[0].record_id, got[0].score) if got else (None, 0.0)


def test_wipe_script_then_same_ids_rewritten_are_searched_fresh(emb):
    """Regression guard. Search (fills any cache), run scripts/wipe.py on the live database, write the SAME
    ids with other text: the next search matches the new text with its own vector, nothing of the old."""
    with _client() as client:
        core = client.app.state.core
        old, new = _records(10, "b", "alpha"), _records(10, "b", "omega")
        core.write(old, space_id="u1", batch_id="b")
        assert _dense_top(core, "u1", old[3].text)[1] > 0.999
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wipe.py"), core.db.path],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-500:]
        core.write(new, space_id="u1", batch_id="b")
        rid, score = _dense_top(core, "u1", new[3].text)
        assert rid == "b#3" and score > 0.999, f"after wipe + rewrite: top {rid} score {score}"
        assert _dense_top(core, "u1", old[3].text)[1] < 0.999, "an old row's vector is still matched"


def test_backfill_script_vectors_are_used_by_the_next_search(emb, monkeypatch):
    """Regression guard. Stored without vectors and no search-time backfill: dense finds nothing.
    scripts/backfill_vectors.py adds them on the live database: the next search uses them."""
    with _client(dense_backfill_m=0) as client:
        core = client.app.state.core
        rows = _records(10, "b", "kilo")
        emb.down = True
        core.write(rows, space_id="u1", batch_id="b")
        emb.down = False
        assert _dense_top(core, "u1", rows[4].text)[0] is None, "the test needs dense to be empty first"
        sys.path.insert(0, ROOT)
        try:
            from scripts import backfill_vectors as bf
            monkeypatch.setattr(bf, "load", lambda: None)
            monkeypatch.setattr(bf, "_build_embedder", lambda settings: SwitchEmbedder())
            monkeypatch.setattr(sys, "argv", ["backfill_vectors.py", core.db.path])
            bf.main()
        finally:
            sys.path.remove(ROOT)
        rid, score = _dense_top(core, "u1", rows[4].text)
        assert rid == "b#4" and score > 0.999, f"after the backfill: top {rid} score {score}"


# ---------------------------------------------------- the offline missing count

def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


COUNT_SCRIPT = os.path.join(ROOT, "scripts", "count_missing_vectors.py")


def _count_script(*args):
    assert os.path.exists(COUNT_SCRIPT), "scripts/count_missing_vectors.py does not exist yet"
    return subprocess.run([sys.executable, COUNT_SCRIPT, *args],
                          capture_output=True, text=True, timeout=120)


def test_count_script_reports_the_exact_missing_number_read_only(emb):
    """7 of 12 records were stored while the provider was down: the script prints missing_vectors 7,
    and the database file is byte for byte the same afterwards."""
    with _client() as client:
        core = client.app.state.core
        core.write(_records(5, "a", "ok"), space_id="u1", batch_id="a")
        emb.down = True
        core.write(_records(4, "b", "gone"), space_id="u1", batch_id="b")
        core.write(_records(3, "c", "lost"), space_id="u2", batch_id="c")
        path = core.db.path
    before = _sha(path)
    r = _count_script(path, "--model", "fake:16")
    assert r.returncode == 0, f"exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}"
    last = json.loads(r.stdout.strip().splitlines()[-1])
    assert last.get("missing_vectors") == 7, last
    assert _sha(path) == before, "the count script changed the database"


def test_count_script_fails_on_a_database_without_the_tables():
    """An empty database file: the script exits non-zero (it does not report 0), and creates no table."""
    path = _new_db_path("empty")
    sqlite3.connect(path).close()
    r = _count_script(path, "--model", "fake:16")
    conn = sqlite3.connect(path)
    tables = [t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    conn.close()
    assert r.returncode != 0 and not re.search(r'"missing_vectors"\s*:\s*0\b', r.stdout), r.stdout[-300:]
    assert tables == [], f"the count script created tables: {tables}"


# ------------------------------------------------------- the linear-algebra threads

def _thread_settings_after_importing_app(preset=None):
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if preset is not None:
        env["OPENBLAS_NUM_THREADS"] = preset
    code = ("import os, app, numpy; "
            "print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout.split()


def test_the_service_gives_numpy_one_thread_unless_told_otherwise():
    """Importing the service package (before numpy) sets the linear-algebra libraries to one thread: in a
    container they would otherwise start one thread per HOST cpu and stall the container's cpu quota."""
    assert _thread_settings_after_importing_app() == ["1", "1", "1"]
    assert _thread_settings_after_importing_app(preset="4")[0] == "4", "a value from the environment is kept"


def test_a_search_waiting_past_the_limit_for_another_s_build_answers_503(emb):
    """Two searches reach a cold space at once; building its index takes 1 s, SEARCH_HEAVY_WAIT_S is 0.3 s.
    The search that builds answers 200; the one waiting for that build answers 503 at the limit."""
    import threading
    import time
    with _client(search_heavy_wait_s=0.3) as client:
        _add(client, "u1", "r1", [f"note {i} about the release train" for i in range(5)])
        db = client.app.state.core.db
        real = db.connect

        class Slow:
            def __init__(self, conn):
                self._c = conn

            def execute(self, sql, *a):
                if "FROM core_records WHERE space_id=? AND valid=1" in " ".join(sql.split()):
                    time.sleep(1.0)
                return self._c.execute(sql, *a)

            def __getattr__(self, name):
                return getattr(self._c, name)
        db.connect = lambda: Slow(real())
        codes = []
        start = threading.Barrier(2)

        def one(i):
            start.wait()
            r = client.post("/search", headers=AUTH, json={"query": f"release train {i}", "user_id": "u1", "top_k": 5})
            codes.append(r.status_code)
        ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        db.connect = real
        assert sorted(codes) == [200, 503], codes
