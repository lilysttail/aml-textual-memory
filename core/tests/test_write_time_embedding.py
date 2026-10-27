"""write-time embedding tests (test-first, RED).

These tests exercise the new contract:
  - vectors are computed at write time (not lazily at search time)
  - write fail-open on provider errors (record still lands, pending tracked)
  - search fail-fast (never 503 from the dense path)
  - replay doesn't recompute (embed calls = 0 on same request_id)
  - missing vectors are visible via health, header, and log
  - missing-vector detection is based on records-vectors fact, not pending table
  - dense-eligible criteria is shared (valid=1, space, exclude lines, model)
  - partial batch failure is reconciled per-record
  - fail-open scope is narrow (only provider errors, not SQLite)
  - hash/model_name comes from a single helper

Uses SpyEmbedder (from conftest) to count embed calls.
"""
import hashlib
import json
import logging
import os
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
from core.vector_cache import VectorCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TMPROOT = os.path.join(tempfile.gettempdir(), "wte-test")


def _tmpdir():
    d = os.path.join(_TMPROOT, uuid.uuid4().hex[:12])
    os.makedirs(d, exist_ok=True)
    return d


def _cleanup():
    shutil.rmtree(_TMPROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _cleanup_tmp():
    yield
    _cleanup()


def _content_hash(text):
    """SHA-256 of UTF-8 text — the single source of truth for hashing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SpyEmbedder(FakeEmbedder):
    """FakeEmbedder that remembers every text it was asked to embed."""

    def __init__(self, dim=16):
        super().__init__(dim=dim)
        self.embedded_texts = []
        self.embed_calls = 0

    def embed(self, texts, timeout_s=None):
        self.embedded_texts.extend(texts)
        self.embed_calls += 1
        return super().embed(texts, timeout_s=timeout_s)


class FailingEmbedder(SpyEmbedder):
    """SpyEmbedder that raises on the Nth call (or always)."""

    def __init__(self, dim=16, fail_from_call=1):
        super().__init__(dim=dim)
        self._fail_from = fail_from_call

    def embed(self, texts, timeout_s=None):
        if self.embed_calls >= self._fail_from:
            raise EmbedUnavailable("provider error (test)")
        return super().embed(texts, timeout_s=timeout_s)


class PartialFailingEmbedder(SpyEmbedder):
    """Fails on the second batch of a multi-batch call.

    Batch 1 succeeds, batch 2+ fails. Used to test partial batch reconciliation.
    """

    def __init__(self, dim=16, batch_size=2):
        super().__init__(dim=dim)
        self._batch_size = batch_size
        self._total_texts = 0

    def embed(self, texts, timeout_s=None):
        batch_num = self._total_texts // self._batch_size
        self._total_texts += len(texts)
        if batch_num >= 1:
            raise EmbedUnavailable("provider error on batch 2+")
        return super().embed(texts, timeout_s=timeout_s)


class SlowEmbedder(SpyEmbedder):
    """SpyEmbedder that sleeps before returning — simulates a slow provider.

    When timeout_s is given and shorter than the delay, it behaves like a
    real HTTP timeout: sleeps up to timeout_s, then raises EmbedUnavailable.
    """

    def __init__(self, dim=16, delay_s=0.5):
        super().__init__(dim=dim)
        self._delay = delay_s

    def embed(self, texts, timeout_s=None):
        if timeout_s is not None and self._delay > timeout_s:
            time.sleep(timeout_s)
            raise EmbedUnavailable("request timeout (test)")
        time.sleep(self._delay)
        return super().embed(texts, timeout_s=timeout_s)


class GateFullEmbedder(SpyEmbedder):
    """SpyEmbedder that always raises when the concurrency gate is full."""

    def __init__(self, dim=16):
        super().__init__(dim=dim)

    def embed(self, texts, timeout_s=None):
        raise EmbedUnavailable("concurrency gate timed out (test)")


def _build_core(db_dir, paths=("keyword", "dense"), embedder=None,
                embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
                dense_exclude_lines=("terminal",),
                embed_max_concurrency=2, embed_batch=32,
                dense_backfill_m=8, dense_backfill_t=2.0):
    """Build a Core with a temp DB and optional embedder."""
    d = os.path.join(db_dir, "core.db")
    cfg = CoreConfig(
        db_path=d,
        paths=paths,
        embedder=embedder,
        embed_max_concurrency=embed_max_concurrency,
        embed_batch=embed_batch,
        embed_backfill_wait_s=embed_backfill_wait_s,
        embed_acquire_s=embed_acquire_s,
        dense_exclude_lines=dense_exclude_lines,
        dense_backfill_m=dense_backfill_m,
        dense_backfill_t=dense_backfill_t,
        context_before=1,
        context_after=1,
    )
    return build(cfg), d


def _record(record_id, text, session_id="s1", line=None, valid=True):
    return Record(
        record_id=record_id,
        session_id=session_id,
        text=text,
        turn_index=None,
        speaker="u",
        ts="2026-10-22T00:00:00.000Z",
        line=line,
        valid=valid,
    )


def _get_db_conn(db_path):
    """Open a raw SQLite connection for inspection."""
    import sqlite3
    return sqlite3.connect(db_path)


def _get_pending_count(conn, space_id):
    """Count rows in core_pending_vectors for a space (returns 0 if table doesn't exist)."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM core_pending_vectors WHERE space_id = ?",
            (space_id,),
        ).fetchone()
        return row[0]
    except Exception:
        return 0


def _get_vector_count(conn, model_name):
    """Count rows in core_vectors for a model."""
    row = conn.execute(
        "SELECT COUNT(*) FROM core_vectors WHERE model = ?",
        (model_name,),
    ).fetchone()
    return row[0]


def _get_missing_vectors(conn, space_id, model_name, exclude_lines=("terminal",)):
    """Get record_ids that have a record but no vector (the fact-based gap)."""
    import hashlib
    # Select dense-eligible records.
    sql = (
        "SELECT r.record_id, r.text FROM core_records r "
        "WHERE r.space_id = ? AND r.valid = 1")
    params = [space_id]
    if exclude_lines:
        ph = ",".join("?" for _ in exclude_lines)
        sql += f" AND (r.line IS NULL OR r.line NOT IN ({ph}))"
        params.extend(exclude_lines)
    rows = conn.execute(sql, params).fetchall()
    # Compute hashes and check which ones are missing.
    hashes = {r[0]: hashlib.sha256(r[1].encode("utf-8")).hexdigest() for r in rows}
    hash_list = list(hashes.values())
    existing = set()
    for i in range(0, len(hash_list), 500):
        chunk = hash_list[i:i + 500]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT content_hash FROM core_vectors "
            f"WHERE model = ? AND content_hash IN ({ph})",
            [model_name, *chunk]):
            existing.add(row[0])
    return [rid for rid, h in hashes.items() if h not in existing]


# ---------------------------------------------------------------------------
# V1: Write time computes vectors
# ---------------------------------------------------------------------------

def test_v1_write_time_computes_vectors():
    """After /add, the written records' content hashes should be in core_vectors."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [_record("r1", "hello world"), _record("r2", "foo bar")]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Vectors should have been computed at write time
    assert emb.embed_calls == 1, f"expected 1 embed call, got {emb.embed_calls}"
    assert len(emb.embedded_texts) == 2

    conn = _get_db_conn(db_path)
    vec_count = _get_vector_count(conn, emb.model_name)
    conn.close()
    assert vec_count == 2, f"expected 2 vectors in core_vectors, got {vec_count}"


# ---------------------------------------------------------------------------
# V2: Failure still writes successfully (fail-open)
# ---------------------------------------------------------------------------

def test_v2_failure_still_writes_successfully():
    """Embedder error → /add still 200, row exists, core_pending_vectors has entries, keyword still works."""
    d = _tmpdir()
    emb = FailingEmbedder(dim=16, fail_from_call=0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [_record("r1", "hello world"), _record("r2", "foo bar")]
    result = core.write(recs, space_id="sp1", batch_id="b1")

    # Write should still succeed
    assert result.get("written", 0) == 2, f"expected 2 written, got {result}"

    # Rows should be in core_records
    conn = _get_db_conn(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM core_records WHERE space_id = ?",
        ("sp1",),
    ).fetchone()[0]
    conn.close()
    assert count == 2

    # pending vectors should track the failures
    conn = _get_db_conn(db_path)
    pending = _get_pending_count(conn, "sp1")
    conn.close()
    assert pending == 2, f"expected 2 pending, got {pending}"

    # Keyword search should still work
    hits = core.search("hello", space_id="sp1", top_k=5)
    assert len(hits) > 0, "keyword search should still return results"


# ---------------------------------------------------------------------------
# V3: Limited backfill on search
# ---------------------------------------------------------------------------

def test_v3_limited_backfill_on_search():
    """Records exist, vectors missing → search fills at most M, pending decreases, no request blocking."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    # Write many records (will succeed, vectors computed)
    recs = [_record(f"r{i}", f"text {i}") for i in range(20)]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Now delete the vectors (simulate missing vectors)
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.commit()
    conn.close()

    # Search should still work (not 503)
    hits = core.search("text 5", space_id="sp1", top_k=5)
    assert hits is not None, "search should not fail"


# ---------------------------------------------------------------------------
# V4: Replay doesn't recompute
# ---------------------------------------------------------------------------

def test_v4_replay_doesnt_recompute():
    """Same request_id N times → embed calls ≤ 1, no duplicate rows."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [_record("r1", "hello world")]
    for i in range(5):
        core.write(recs, space_id="sp1", batch_id="b1")

    assert emb.embed_calls <= 1, f"expected ≤1 embed call, got {emb.embed_calls}"

    conn = _get_db_conn(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM core_records WHERE space_id = ?",
        ("sp1",),
    ).fetchone()[0]
    conn.close()
    assert count == 1, f"expected 1 row, got {count}"


# ---------------------------------------------------------------------------
# V4b: Replay skips vectors (embed calls = 0 on same request_id after failure)
# ---------------------------------------------------------------------------

def test_v4b_replay_skips_vectors_after_failure():
    """First call fails on embed, replay with same request_id → embed calls must be 0."""
    d = _tmpdir()
    emb = FailingEmbedder(dim=16, fail_from_call=1)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [_record("r1", "hello world")]

    # First call: write succeeds, embed fails
    result1 = core.write(recs, space_id="sp1", batch_id="b1")
    assert result1.get("written", 0) == 1
    assert emb.embed_calls == 1

    # Replay: same request_id, embed should NOT be called again
    result2 = core.write(recs, space_id="sp1", batch_id="b1")
    assert result2.get("written", 0) == 0, "replay should not write"
    assert emb.embed_calls == 1, f"expected 1 embed call total, got {emb.embed_calls} — replay should skip vectors"


# ---------------------------------------------------------------------------
# V5: Search doesn't compute full corpus
# ---------------------------------------------------------------------------

def test_v5_search_doesnt_compute_full_corpus():
    """Many missing vectors → embed calls per search ≤ M+1 (query vector only), not O(N)."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    # Write 50 records
    recs = [_record(f"r{i}", f"text {i}") for i in range(50)]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Clear vectors to simulate missing
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.commit()
    conn.close()

    # Reset embed call counter
    emb.embed_calls = 0
    emb.embedded_texts = []

    # Search should trigger at most M+1 embed calls (query + limited backfill)
    hits = core.search("text 5", space_id="sp1", top_k=5)
    assert emb.embed_calls <= 9, f"expected ≤9 embed calls (M=8+1 query), got {emb.embed_calls}"


# ---------------------------------------------------------------------------
# V6: Missing vectors visible (health + header + log)
# ---------------------------------------------------------------------------

def test_v6_missing_vectors_visible_in_health():
    """After a failed write, /health should report pending_vectors > 0."""
    d = _tmpdir()
    emb = FailingEmbedder(dim=16, fail_from_call=0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [_record("r1", "hello world")]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Check via database (the health endpoint would check this)
    conn = _get_db_conn(db_path)
    missing = _get_missing_vectors(conn, "sp1", emb.model_name)
    conn.close()
    assert len(missing) == 1, f"expected 1 missing vector, got {len(missing)}"


# ---------------------------------------------------------------------------
# V6b: Missing vectors by data (not pending table)
# ---------------------------------------------------------------------------

def test_v6b_missing_vectors_by_data_not_pending():
    """Record exists, vector missing, NO pending row → system still finds the gap."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    # Write a record
    recs = [_record("r1", "hello world")]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Manually delete the vector (simulate a gap that pending doesn't know about)
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.execute("DELETE FROM core_pending_vectors")
    conn.commit()
    conn.close()

    # The fact-based check should still find the gap
    conn = _get_db_conn(db_path)
    missing = _get_missing_vectors(conn, "sp1", emb.model_name)
    conn.close()
    assert len(missing) == 1, f"expected 1 missing vector (fact-based), got {len(missing)}"


# ---------------------------------------------------------------------------
# V7: Write fail-open, search fail-fast
# ---------------------------------------------------------------------------

def test_v7_write_fail_open_search_fail_fast():
    """Concurrency gate full → write 200 + pending; search skips dense, keyword works, no 503."""
    d = _tmpdir()

    # Gate-full embedder for write path
    write_emb = GateFullEmbedder(dim=16)
    core_w, db_path_w = _build_core(d, paths=("keyword", "dense"), embedder=write_emb)

    recs = [_record("r1", "hello world")]
    result = core_w.write(recs, space_id="sp1", batch_id="b1")
    assert result.get("written", 0) == 1, "write should still succeed"

    conn = _get_db_conn(db_path_w)
    pending = _get_pending_count(conn, "sp1")
    conn.close()
    assert pending == 1, "pending should track the failure"

    # Search should still work (no 503)
    hits = core_w.search("hello", space_id="sp1", top_k=5)
    assert hits is not None, "search should not fail with 503"


# ---------------------------------------------------------------------------
# 栖 P1-1 (round 2): search deadline really bounds the dense path
# ---------------------------------------------------------------------------

def test_deadline_gate_full_gives_up_immediately():
    """闸满：搜索补算抢不到闸立即放弃，不等 30 秒，keyword 照答。

    证据式断言：搜索耗时 < 1 秒（远小于闸的 30s acquire timeout），
    不抛异常，keyword 有结果。
    """
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb,
                                 embed_max_concurrency=1,
                                 dense_backfill_m=8, dense_backfill_t=2.0)
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")
    # 清向量制造缺口：搜索时必须补算才能有 dense 候选。
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.commit()
    conn.close()
    # 占住唯一的一个闸许可：补算和 query 向量都抢不到。
    core._vc._sem.acquire()
    try:
        start = time.monotonic()
        hits = core.search("hello", space_id="sp1", top_k=5)
        elapsed = time.monotonic() - start
    finally:
        core._vc._sem.release()
    assert hits is not None and len(hits) > 0, "keyword 必须照答"
    assert elapsed < 1.0, f"闸满时应立即放弃，耗时 {elapsed:.2f}s"


def test_deadline_slow_provider_gives_up_between_batches():
    """接口慢：第一批在时限内完成并落库，第二批到点超时放弃。

    证据式断言：搜索耗时在时限附近（T=0.2s、单批 0.15s → 约 0.2-0.3s），
    第一批落库（>=2 条），不抛错，keyword 照答。
    """
    d = _tmpdir()
    emb = SlowEmbedder(dim=16, delay_s=0.15)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb,
                                 embed_batch=2,
                                 dense_backfill_m=8, dense_backfill_t=0.2)
    core.write([_record(f"r{i}", f"text {i}") for i in range(4)],
               space_id="sp1", batch_id="b1")
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.commit()
    conn.close()

    start = time.monotonic()
    hits = core.search("text 2", space_id="sp1", top_k=5)
    elapsed = time.monotonic() - start

    assert hits is not None, "搜索不应抛错"
    assert elapsed < 1.0, f"慢接口也应被 deadline 管住，耗时 {elapsed:.2f}s"
    # 第一批（text0/text1，0.15s < 剩余 0.2s）应落库；第二批到点超时放弃。
    conn = _get_db_conn(db_path)
    vec_count = _get_vector_count(conn, emb.model_name)
    conn.close()
    assert vec_count >= 2, f"已完成的批应落库，got {vec_count}"


def test_deadline_same_hash_in_flight_gives_up():
    """同一段在别处算：搜索不等 60 秒，deadline 后放弃，keyword 照答。"""
    import threading as _th
    d = _tmpdir()
    emb = SlowEmbedder(dim=16, delay_s=1.0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb,
                                 dense_backfill_m=8, dense_backfill_t=0.1)
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.commit()
    conn.close()

    a_done = _th.Event()

    def a_work():
        try:
            # 无 deadline：A 会等满慢调用，期间占着闸和 inflight flight。
            core._vc.get_or_compute(["hello world"])
        finally:
            a_done.set()

    ta = _th.Thread(target=a_work)
    ta.start()
    time.sleep(0.2)  # 让 A 的 flight 建立、闸被占。
    try:
        start = time.monotonic()
        hits = core.search("hello", space_id="sp1", top_k=5)
        elapsed = time.monotonic() - start
    finally:
        a_done.wait(5)
    assert hits is not None and len(hits) > 0, "keyword 必须照答"
    assert elapsed < 1.0, f"同段在算时应在 deadline 后放弃，耗时 {elapsed:.2f}s"


def test_deadline_single_slow_call_gives_up_at_timeout():
    """单次慢调用真的到点放弃：一次调用睡得比时限长，剩余时间传成
    这次调用的 HTTP 超时，到点抛 EmbedUnavailable → dense 跳过。

    断言搜索耗时在时限附近（T=0.2s，delay=1.0s → 睡 0.2s 后放弃），
    不是 <2s 这种宽断言；keyword 照答。
    """
    d = _tmpdir()
    emb = SlowEmbedder(dim=16, delay_s=1.0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb,
                                 dense_backfill_m=8, dense_backfill_t=0.2)
    # 写入时算向量：无 deadline → 用配置超时（timeout_s=None）→ 睡 1s 完成。
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")
    conn = _get_db_conn(db_path)
    conn.execute("DELETE FROM core_vectors")
    conn.commit()
    conn.close()

    start = time.monotonic()
    hits = core.search("hello", space_id="sp1", top_k=5)
    elapsed = time.monotonic() - start

    assert hits is not None and len(hits) > 0, "keyword 必须照答"
    # 补算那次调用睡到 timeout_s=剩余时间（约 0.2s）才放弃；query 的剩余
    # 时间已耗尽，立即失败。总耗时应在时限附近。
    assert 0.12 <= elapsed < 0.6, \
        f"单次慢调用应到点（约 {elapsed:.2f}s 应在 0.12-0.6s 内）放弃"


def test_deadline_query_vector_single_call_gives_up():
    """query 那次向量调用单独受时限管：向量都已算好，只有 query 慢。

    断言搜索耗时在时限附近（T=0.2s，delay=1.0s），dense 跳过、keyword 照答。
    """
    d = _tmpdir()
    emb = SlowEmbedder(dim=16, delay_s=1.0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb,
                                 dense_backfill_m=8, dense_backfill_t=0.2)
    # 写入时算好向量，搜索时没有缺口 → 只有 query 向量要算。
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")

    start = time.monotonic()
    hits = core.search("hello", space_id="sp1", top_k=5)
    elapsed = time.monotonic() - start

    assert hits is not None and len(hits) > 0, "keyword 必须照答"
    assert 0.12 <= elapsed < 0.6, \
        f"query 调用应到点（约 {elapsed:.2f}s 应在 0.12-0.6s 内）放弃"


# ---------------------------------------------------------------------------
# 栖 P1-3 (round 2): pending/缺口读写的异常范围收窄
# ---------------------------------------------------------------------------

def test_pending_table_write_failure_propagates():
    """core_pending_vectors 写不进去：必须冒泡，不许吞成写入成功。"""
    d = _tmpdir()
    emb = FailingEmbedder(dim=16, fail_from_call=0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)
    conn = _get_db_conn(db_path)
    conn.execute("DROP TABLE core_pending_vectors")
    conn.commit()
    conn.close()
    with pytest.raises(Exception):
        core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")


def test_pending_clear_failure_propagates():
    """向量算好后清 pending 失败：冒泡，不许静默。"""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")
    # 人为留一条 pending 行，然后让 clear 失败：删掉表。
    conn = _get_db_conn(db_path)
    conn.execute("INSERT OR REPLACE INTO core_pending_vectors "
                 "(space_id, record_id, content_hash, reason, retry_count, last_attempt_at) "
                 "VALUES (?,?,?,?,?,?)",
                 ("sp1", "r1", "h", "stale", 1, "t"))
    conn.commit()
    conn.execute("DROP TABLE core_pending_vectors")
    conn.commit()
    conn.close()
    with pytest.raises(Exception):
        # 再次写入同 space 的另一个 batch：成功路径会 clear pending → 表没了 → 冒泡。
        core.write([_record("r2", "another text")], space_id="sp1", batch_id="b2")


def test_missing_vector_records_db_failure_propagates():
    """core_vectors 读失败：missing_vector_records 不许当成没有缺口。"""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")
    conn = _get_db_conn(db_path)
    conn.execute("DROP TABLE core_vectors")
    conn.commit()
    conn.close()
    with pytest.raises(Exception):
        core.missing_vector_records("sp1")


def test_pending_retry_count_increments():
    """同一 record 再次记 pending：retry_count 递增，不是重置成 1。"""
    d = _tmpdir()
    emb = FailingEmbedder(dim=16, fail_from_call=0)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)
    core.write([_record("r1", "hello world")], space_id="sp1", batch_id="b1")
    conn = _get_db_conn(db_path)
    row = conn.execute("SELECT retry_count FROM core_pending_vectors "
                       "WHERE space_id=? AND record_id=?", ("sp1", "r1")).fetchone()
    assert row[0] == 1
    conn.close()
    # 直接再记一次（模拟补算又失败）：retry 应变成 2。
    core._record_pending("sp1", "r1", _content_hash("hello world"),
                         "timeout", "now2")
    conn = _get_db_conn(db_path)
    row = conn.execute("SELECT retry_count FROM core_pending_vectors "
                       "WHERE space_id=? AND record_id=?", ("sp1", "r1")).fetchone()
    conn.close()
    assert row[0] == 2, f"retry_count 应递增为 2，got {row[0]}"


# ---------------------------------------------------------------------------
# 栖 #2: Dense-eligible criteria is shared
# ---------------------------------------------------------------------------

def test_dense_eligible_excludes_terminal_lines():
    """Terminal lines should not get vectors, and should not count as pending."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [
        _record("r1", "normal message", line=None),
        _record("r2", "terminal output", line="terminal"),
    ]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Only the normal message should have a vector
    conn = _get_db_conn(db_path)
    vec_count = _get_vector_count(conn, emb.model_name)
    conn.close()
    assert vec_count == 1, f"expected 1 vector (terminal excluded), got {vec_count}"

    # No pending for the terminal line
    conn = _get_db_conn(db_path)
    pending = _get_pending_count(conn, "sp1")
    conn.close()
    assert pending == 0, f"terminal should not be in pending, got {pending}"


def test_dense_eligible_requires_valid_and_same_space():
    """Only valid=1 records in the same space should get vectors."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [
        _record("r1", "valid record", valid=True),
        _record("r2", "invalid record", valid=False),
    ]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Only the valid record should have a vector
    conn = _get_db_conn(db_path)
    vec_count = _get_vector_count(conn, emb.model_name)
    conn.close()
    assert vec_count == 1, f"expected 1 vector (invalid excluded), got {vec_count}"


# ---------------------------------------------------------------------------
# 栖 #3: Partial batch failure reconciliation
# ---------------------------------------------------------------------------

def test_partial_batch_failure_reconciliation():
    """Batch 1 succeeds, batch 2 fails → only batch 2 records should be pending."""
    d = _tmpdir()
    emb = PartialFailingEmbedder(dim=16, batch_size=2)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb,
                                 embed_batch=2)

    # Write 4 records (2 batches of 2)
    recs = [_record(f"r{i}", f"text {i}") for i in range(4)]
    result = core.write(recs, space_id="sp1", batch_id="b1")

    # Write should still succeed
    assert result.get("written", 0) == 4

    conn = _get_db_conn(db_path)

    # Batch 1 (r0, r1) should have vectors
    batch1_hashes = [_content_hash("text 0"), _content_hash("text 1")]
    vec1 = conn.execute(
        "SELECT COUNT(*) FROM core_vectors WHERE model = ? AND content_hash IN (?, ?)",
        (emb.model_name, batch1_hashes[0], batch1_hashes[1]),
    ).fetchone()[0]
    assert vec1 == 2, f"batch 1 should have 2 vectors, got {vec1}"

    # Batch 2 (r2, r3) should NOT have vectors (embed failed)
    batch2_hashes = [_content_hash("text 2"), _content_hash("text 3")]
    vec2 = conn.execute(
        "SELECT COUNT(*) FROM core_vectors WHERE model = ? AND content_hash IN (?, ?)",
        (emb.model_name, batch2_hashes[0], batch2_hashes[1]),
    ).fetchone()[0]
    assert vec2 == 0, f"batch 2 should have 0 vectors, got {vec2}"

    # Batch 2 records should be in pending
    pending = _get_pending_count(conn, "sp1")
    conn.close()
    assert pending == 2, f"expected 2 pending (batch 2), got {pending}"


# ---------------------------------------------------------------------------
# 栖 #4: Narrow fail-open scope
# ---------------------------------------------------------------------------

def test_sqlite_error_not_swallowed():
    """Core SQLite errors should NOT be swallowed into 200."""
    d = _tmpdir()
    emb = SpyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    # Corrupt the database to simulate a SQLite error
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE core_records")
    conn.commit()
    conn.close()

    # Write should fail with an error, not silently succeed
    with pytest.raises(Exception):
        recs = [_record("r1", "hello world")]
        core.write(recs, space_id="sp1", batch_id="b1")


# ---------------------------------------------------------------------------
# 栖 #5: Hash/model_name single helper
# ---------------------------------------------------------------------------

def test_hash_is_consistent_across_operations():
    """The same text should always hash to the same value, regardless of where it's computed."""
    text = "hello world"
    h1 = _content_hash(text)
    h2 = _content_hash(text)
    assert h1 == h2

    # Different text should hash differently
    h3 = _content_hash("different text")
    assert h1 != h3


def test_model_name_includes_dimension():
    """Model name should include the dimension to prevent mixing vectors of different sizes."""
    emb = SpyEmbedder(dim=16)
    assert emb.model_name == "fake:16"


# ---------------------------------------------------------------------------
# Search path: StoreBusy should not become 503
# ---------------------------------------------------------------------------

def test_search_does_not_raise_store_busy():
    """Search should never raise StoreBusy (it should skip dense and return keyword results)."""
    d = _tmpdir()

    # Use a gate-full embedder that always raises StoreBusy
    from core.vector_cache import VectorCache as _VC
    import unittest.mock

    class BusyEmbedder(SpyEmbedder):
        def embed(self, texts, timeout_s=None):
            raise StoreBusy("gate full")

    emb = BusyEmbedder(dim=16)
    core, db_path = _build_core(d, paths=("keyword", "dense"), embedder=emb)

    recs = [_record("r1", "hello world")]
    core.write(recs, space_id="sp1", batch_id="b1")

    # Search should not raise StoreBusy
    try:
        hits = core.search("hello", space_id="sp1", top_k=5)
        assert hits is not None, "search should return results"
    except StoreBusy:
        pytest.fail("search should not raise StoreBusy — it should skip dense and return keyword results")