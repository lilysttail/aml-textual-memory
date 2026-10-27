"""The shape a second consumer of the core brings: a made-up export (window, date, line, direction, name, source id, state, flag,
text, time) fed into the core can be searched back with who said it, when, which window and the lines around it.

Only the keyword route is used here, so the assertions search for literal words. No real database is attached: this only shows
that the core takes this shape in without knowing where the data came from.
"""
import os
import shutil
import tempfile
import uuid

from core import CoreConfig, Record, build

# The frozen columns of the made-up export: window id, date, line, direction, a name looked up at read time, a source id, a
# verdict state, a sensitivity flag, plus the text and the time.
VIEW_ROWS = [
    {"window": "w1", "date": "2026-10-20", "line": "chat", "direction": "in",
     "name": "alice", "source_id": "src-1", "state": "kept", "sec_flag": False,
     "text": "remember this one: alpha", "ts": "2026-10-20T12:00:00.000Z", "seq": 0},
    {"window": "w1", "date": "2026-10-20", "line": "chat", "direction": "out",
     "name": "bob", "source_id": "src-2", "state": "kept", "sec_flag": False,
     "text": "got it, this alpha one is saved", "ts": "2026-10-20T12:01:00.000Z", "seq": 1},
    {"window": "w1", "date": "2026-10-20", "line": "chat", "direction": "in",
     "name": "alice", "source_id": "src-3", "state": "kept", "sec_flag": False,
     "text": "one more alpha line to keep", "ts": "2026-10-20T12:02:00.000Z", "seq": 2},
]


def _to_records(rows):
    out = []
    for r in rows:
        out.append(Record(record_id=r["source_id"],
                           session_id=r["window"],
                           text=r["text"],
                           turn_index=r["seq"],
                           speaker=r["name"],
                           direction=r["direction"],
                           line=r["line"],
                           ts=r["ts"],
                           sec_flag=r["sec_flag"],
                           state=r["state"]))
    return out


def _core():
    d = os.path.join(tempfile.gettempdir(), "second-consumer-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    cfg = CoreConfig(db_path=os.path.join(d, "second.db"), paths=("keyword",),
                     context_before=1, context_after=1)
    return build(cfg), d


def test_the_export_shape_is_searchable_with_its_people_and_neighbours():
    core, d = _core()
    try:
        core.replace_space_snapshot("space-1", _to_records(VIEW_ROWS), snapshot_id="snap-1")
        hits = core.search("alpha", space_id="space-1", top_k=5)
        assert [h.record_id for h in hits] == ["src-1", "src-2", "src-3"]
        # who said it, when, which window
        assert hits[0].speaker == "alice"
        assert hits[0].ts == "2026-10-20T12:00:00.000Z"
        assert hits[0].session_id == "w1"
        # the lines around: one before and one after the middle hit
        middle = hits[1]
        assert [r.record_id for r in middle.neighbors_before] == ["src-1"]
        assert [r.record_id for r in middle.neighbors_after] == ["src-3"]
        # in time order
        assert [h.ts for h in hits] == sorted(h.ts for h in hits)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_export_shape_survives_a_judgement_change():
    """A record is judged deleted at the source -> the next snapshot does not have it -> it cannot be found, and the neighbours change."""
    core, d = _core()
    try:
        core.replace_space_snapshot("space-1", _to_records(VIEW_ROWS), snapshot_id="snap-1")
        assert len(core.search("alpha", space_id="space-1", top_k=5)) == 3
        after = [r for r in VIEW_ROWS if r["source_id"] != "src-2"]
        core.replace_space_snapshot("space-1", _to_records(after), snapshot_id="snap-2")
        hits = core.search("alpha", space_id="space-1", top_k=5)
        assert [h.record_id for h in hits] == ["src-1", "src-3"]
        assert [r.record_id for r in hits[0].neighbors_after] == ["src-3"]
    finally:
        shutil.rmtree(d, ignore_errors=True)
