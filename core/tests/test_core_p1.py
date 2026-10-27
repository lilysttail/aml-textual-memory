"""栖代码审第二轮（6 条 P1 + 3 条次要）的核心侧自测。

每条都对着一个能复现坏行为的断言：拿掉修复，这些测试必须红。
"""
import os
import tempfile
import uuid

import pytest

from core import CoreConfig, Record, build
from core.db import Database
from core.embedder import EmbedUnavailable
from core.tests.conftest import rec


def _db_path(name):
    d = os.path.join(tempfile.gettempdir(), "core-p1-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


class _DownEmbedder:
    """An embedder whose provider is down (timeout / bad reply)."""

    model_name = "down:8"
    dim = 8

    def embed(self, texts, timeout_s=None):
        raise EmbedUnavailable("provider down")


# ── P1-5: the core keeps its own write-time settings ─────────────────────────

def test_P1_5_the_core_records_its_write_time_settings():
    path = _db_path("settings.db")
    c = build(CoreConfig(db_path=path, paths=("keyword",), tokenizer_key="cut-v1"))
    c.write([rec("r1", "一句话")], space_id="home")
    conn = Database(path).connect()
    try:
        recorded = dict(conn.execute(
            "SELECT name, value FROM core_write_settings").fetchall())
    finally:
        conn.close()
    assert recorded, "the core must remember how it wrote its data"
    assert "cut-v1" in recorded.values()


def test_P1_5_a_changed_tokenizer_refuses_to_start_on_a_database_with_data():
    path = _db_path("settings2.db")
    build(CoreConfig(db_path=path, paths=("keyword",), tokenizer_key="cut-v1"))
    c = build(CoreConfig(db_path=path, paths=("keyword",), tokenizer_key="cut-v1"))
    c.write([rec("r1", "一句话")], space_id="home")
    # same settings -> starts fine
    build(CoreConfig(db_path=path, paths=("keyword",), tokenizer_key="cut-v1"))
    # a different tokenizer -> the old index would answer with the wrong terms
    with pytest.raises(Exception) as ei:
        build(CoreConfig(db_path=path, paths=("keyword",), tokenizer_key="cut-v2"))
    assert "setting" in str(ei.value).lower()


# ── P1-6: a dense path that is down does not break the search ───────────────

def test_P1_6_a_dense_path_that_is_down_does_not_break_the_search():
    path = _db_path("dense.db")
    c = build(CoreConfig(db_path=path, paths=("keyword", "dense"),
                         embedder=_DownEmbedder()))
    c.write([rec("r1", "一句话 alpha")], space_id="home")
    hits = c.search("alpha", space_id="home", top_k=5)
    assert [h.record_id for h in hits] == ["r1"]


# ── minor: turn indexes follow the event time, not the arrival order ────────

def test_minor_turn_indexes_follow_the_event_time_not_the_arrival_order():
    path = _db_path("ts.db")
    c = build(CoreConfig(db_path=path, paths=("keyword",)))
    c.write([
        rec("r_late", "晚的那句", ts="2026-10-21T06:00:00.000Z"),
        rec("r_early", "早的那句", ts="2026-10-21T05:00:00.000Z"),
    ], space_id="home")
    rows = c.store.get_rows("home", ["r_late", "r_early"])
    assert rows["r_early"]["turn_index"] < rows["r_late"]["turn_index"], rows


# ── minor: the search hits carry the per-path scores ───────────────────────

def test_minor_search_hits_carry_the_path_scores():
    path = _db_path("scores.db")
    c = build(CoreConfig(db_path=path, paths=("keyword",)))
    c.write([rec("r1", "一句话 alpha")], space_id="home")
    hit = c.search("alpha", space_id="home", top_k=5)[0]
    assert hit.path_scores, "the scheme promises the per-path scores"
    assert "keyword" in hit.path_scores


# ── minor: a snapshot with duplicate ids is refused, not silently shortened ──

def test_minor_a_snapshot_with_duplicate_record_ids_is_refused():
    path = _db_path("dup.db")
    c = build(CoreConfig(db_path=path, paths=("keyword",)))
    with pytest.raises(ValueError):
        c.replace_space_snapshot("home", [rec("dup", "第一条"),
                                          rec("dup", "第二条")],
                                 snapshot_id="v1")
    assert c.store.count("home") == 0, "a refused snapshot must write nothing"
