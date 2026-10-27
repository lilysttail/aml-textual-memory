"""The core's face: build one, write into it, search it.

Core is the whole public surface of this package:

    core = build(CoreConfig(db_path="..."))
    core.write(records, space_id="home", batch_id="b1")
    core.replace_space_snapshot("home", snapshot_records)   # 整份重建 + 原子换代
    hits = core.search("那句话", space_id="home", top_k=5)

What it does NOT do: generate answers, rewrite text, or know where the data
came from. Conflict resolution ("newest wins"), list precision and time
phrasing are the answering side's business — the core hands over the evidence
and lets the caller decide.
"""
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import rerank as fusion
from . import write_settings
from .config import CoreConfig
from .db import Database, StoreBusy, is_locked
from .embedder import EmbedUnavailable
from .models import Record, SearchHit
from .recall import DenseRecall, KeywordRecall
from .store import Store
from .vector_cache import VectorCache
from .vector_store import VectorStore
from .vector_helper import content_hash as _ch, is_dense_eligible as _dense_eligible
from .vector_helper import model_key

log = logging.getLogger("core.engine")

# Default backfill limits for the search path: at most M texts per
# search, bounded by T seconds. The provider's own timeout applies on top.
DEFAULT_BACKFILL_M = 8
DEFAULT_BACKFILL_T = 2.0


def rebuild_fts_index(config: CoreConfig) -> None:
    """Re-index every record with the current write-time settings.

    Upgrade path for a database written under older settings (e.g. English
    stemming off): `Core.build` refuses to start on it (WriteSettingsMismatch),
    because the old index answers with the old terms. Running this re-writes
    the FTS index from core_records with the current tokenizer/stem settings
    and records the new settings, so the next start succeeds.

    Fresh databases never need it (build records settings at first write).
    """
    db = config.db if config.db is not None else Database(
        config.db_path,
        busy_timeout_ms=config.busy_timeout_ms,
        write_lock_timeout_ms=config.write_lock_timeout_ms)
    db.create_schema()
    store = Store(db, tokenizer=config.tokenizer,
                  stem_english=config.stem_english)
    with db.writing() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            # core_fts is an FTS5 table with no unique key — REPLACE would
            # insert a second row, not replace. Clear the whole index in the
            # same transaction, then insert one row per record; a failure in
            # the middle rolls back to the old index, never a half one.
            conn.execute("DELETE FROM core_fts")
            rows = conn.execute(
                "SELECT space_id, record_id, session_id, speaker, ts, text "
                "FROM core_records").fetchall()
            for space_id, record_id, session_id, speaker, ts, text in rows:
                conn.execute(
                    "INSERT INTO core_fts"
                    "(space_id, record_id, session_id, speaker, body) "
                    "VALUES (?,?,?,?,?)",
                    (space_id, record_id, session_id, speaker,
                     store._index_body(text, ts=ts)))
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
    write_settings.record(db, write_settings.current(config))


@dataclass
class Candidate:
    """One candidate of a recall path (for the ruler and for debugging)."""
    record_id: str
    score: float
    path_scores: Dict[str, float] = field(default_factory=dict)


class Core:
    def __init__(self, store, config, recalls=(), vector_cache=None, db=None):
        self._store = store
        self._config = config
        self._recalls = list(recalls)
        self._vc = vector_cache
        self._db = db

    # ----------------------------------------------------------------- build

    @classmethod
    def build(cls, config: CoreConfig) -> "Core":
        if config.db is not None:
            db = config.db
        else:
            if not config.db_path:
                raise ValueError("CoreConfig needs a db_path (or an injected db)")
            db = Database(config.db_path,
                          busy_timeout_ms=config.busy_timeout_ms,
                          write_lock_timeout_ms=config.write_lock_timeout_ms)
        db.create_schema()
        store = Store(db, tokenizer=config.tokenizer,
                      stem_english=config.stem_english)
        # The database remembers how the data was written (which tokenizer, which
        # write format). Data written under other settings stops the start here.
        write_settings.check_or_record(db, config)

        vc = config.vector_cache
        if vc is None and config.embedder is not None:
            vc = VectorCache(
                VectorStore(db), config.embedder,
                max_concurrency=config.embed_max_concurrency,
                batch_size=config.embed_batch,
                wait_timeout_s=config.embed_backfill_wait_s,
                acquire_timeout_s=config.embed_acquire_s)

        recalls = []
        for name in config.paths:
            if name == "keyword":
                recalls.append(KeywordRecall(store))
            elif name == "dense":
                if vc is None:
                    raise ValueError("the dense path needs an embedder (or a vector_cache)")
                recalls.append(DenseRecall(store, vc,
                                           exclude_lines=config.dense_exclude_lines,
                                           backfill_m=config.dense_backfill_m,
                                           backfill_t=config.dense_backfill_t))
            else:
                raise ValueError(f"unknown recall path {name!r} (known: keyword, dense)")
        return cls(store, config, recalls, vc, db=db)

    # ----------------------------------------------------------------- write

    def write(self, records, space_id: str, batch_id=None):
        """Write records into one space. Idempotent on (space_id, batch_id).

        After writing, computes vectors for the newly written records
        (fail-open: a provider error does NOT fail the write).
        On replay (same batch_id), vectors are NOT recomputed.
        """
        result = self._store.write(records, space_id, batch_id)
        # Replay: same (space_id, batch_id) was already written; skip vectors.
        if result.get("replay"):
            return result
        written = result.get("written", 0)
        if written == 0 or self._vc is None:
            return result
        # Compute vectors for the newly written records (fail-open).
        self._compute_vectors_for(records, space_id)
        return result

    def _compute_vectors_for(self, records, space_id):
        """Compute vectors for *records* at write time.

        Fail-open: provider errors are caught, pending rows are written,
        and the caller's write is NOT affected. Only dense-eligible records
        (valid, not in exclude_lines) get vectors.
        """
        from datetime import datetime, timezone
        exclude = self._config.dense_exclude_lines
        model = model_key(self._vc._emb.model_name)
        eligible = []  # (record, content_hash) pairs
        for r in records:
            row = r.to_row()
            if not _dense_eligible(row, exclude):
                continue
            ch = _ch(r.text)
            eligible.append((r, ch))
        if not eligible:
            return
        now = datetime.now(timezone.utc).isoformat()
        # Deduplicate by content hash — same text should only be embedded once.
        seen_hashes = set()
        unique_texts = []
        for r, ch in eligible:
            if ch not in seen_hashes:
                seen_hashes.add(ch)
                unique_texts.append(r.text)
        try:
            # get_or_compute computes missing vectors and stores them.
            self._vc.get_or_compute(unique_texts)
        except EmbedUnavailable as e:
            self._reconcile_pending(
                eligible, model, exclude, space_id,
                reason=f"provider_error: {type(e).__name__}", now=now)
            return
        except StoreBusy as e:
            self._reconcile_pending(
                eligible, model, exclude, space_id,
                reason=f"gate: {type(e).__name__}", now=now)
            return
        # Other exceptions (real bugs, SQLite errors, etc.) propagate —
        # they must not be swallowed into a 200.
        # Success path: clear any stale pending rows for the eligible records.
        for rec, ch in eligible:
            self._clear_pending(space_id, rec.record_id)

    def _reconcile_pending(self, eligible, model, exclude, space_id,
                           reason, now):
        """After a partial embed failure, re-read the cache and write pending
        rows only for the records that don't have a vector yet.

        A failure here (e.g. the vector-store read) must propagate, not be
        treated as "nothing is missing" — a SQLite problem is not a gap of
        zero.
        """
        hashes = [ch for _, ch in eligible]
        existing = self._vc._vec.get_vectors(model, hashes)
        log.debug("reconcile: eligible=%d existing=%d reason=%s",
                 len(eligible), len(existing), reason)
        for rec, ch in eligible:
            if ch in existing:
                self._clear_pending(space_id, rec.record_id)
            else:
                self._record_pending(space_id, rec.record_id, ch, reason, now)

    def _record_pending(self, space_id, record_id, content_hash, reason, now):
        """Insert or bump a row in core_pending_vectors.

        retry_count increments on each attempt, never resets to 1. A failure
        to write this row must propagate: a lost pending row is a silently
        missing vector nobody can see.
        """
        with self._db.writing() as conn:
            conn.execute(
                "INSERT INTO core_pending_vectors "
                "(space_id, record_id, content_hash, reason, retry_count, last_attempt_at) "
                "VALUES (?,?,?,?,1,?) "
                "ON CONFLICT(space_id, record_id) DO UPDATE SET "
                "content_hash=excluded.content_hash, reason=excluded.reason, "
                "retry_count=core_pending_vectors.retry_count+1, "
                "last_attempt_at=excluded.last_attempt_at",
                (space_id, record_id, content_hash, reason, now))
            conn.commit()

    def _clear_pending(self, space_id, record_id):
        """Delete a row from core_pending_vectors (vector was computed).

        A failure here propagates too — a stale pending row hides a real
        state mismatch.
        """
        with self._db.writing() as conn:
            conn.execute(
                "DELETE FROM core_pending_vectors WHERE space_id=? AND record_id=?",
                (space_id, record_id))
            conn.commit()

    def pending_vectors(self, space_id: str) -> int:
        """Count records that are missing vectors (fact-based, not from pending table).

        A record is missing a vector when:
          - it is in core_records with valid=1
          - its content_hash is NOT in core_vectors for the current model
          - its line is NOT in dense_exclude_lines
        """
        records = self.missing_vector_records(space_id, limit=0)
        return len(records)

    def missing_vector_records(self, space_id: str, limit=0) -> list:
        """Get records that are missing vectors (for backfill).

        Returns [(record_id, content_hash, text), ...] for dense-eligible
        records that don't have a vector in core_vectors.

        The dense-eligibility SQL lives in exactly one place: Store.iter_records
        (valid=1, space, exclude_lines). This method reuses it and computes the
        records-vectors gap in Python, so health / backfill / search can never
        drift into a second SQL copy of "which records should have vectors".
        A database failure here propagates: it must not be reported as
        "no gaps".
        """
        if self._vc is None:
            return []
        model = model_key(self._vc._emb.model_name)
        exclude = self._config.dense_exclude_lines
        items = self._store.iter_records(space_id, exclude_lines=exclude)
        if not items:
            return []
        with self._db.connect() as conn:
            hashes = {rid: _ch(text) for rid, text in items}
            hash_list = list(hashes.values())
            existing = set()
            for i in range(0, len(hash_list), 500):
                chunk = hash_list[i:i + 500]
                ph = ",".join("?" for _ in chunk)
                for row in conn.execute(
                    f"SELECT content_hash FROM core_vectors "
                    f"WHERE model = ? AND content_hash IN ({ph})",
                    [model, *chunk]):
                    existing.add(row[0])
            result = [(rid, hashes[rid], text) for rid, text in items
                      if hashes[rid] not in existing]
            if limit > 0:
                result = result[:limit]
            return result

    def write_in_tx(self, conn, records, space_id: str, batch_id=None):
        """Write inside a transaction the caller owns (the shell writes its own
        tables and the core's in one go)."""
        return self._store.write(records, space_id, batch_id, conn=conn)

    def replace_space_snapshot(self, space_id: str, records, snapshot_id=None):
        """Rebuild one space from a snapshot, atomically (see Store)."""
        return self._store.replace_snapshot(space_id, records, snapshot_id)

    # ------------------------------------------------------------------ read

    def keyword_candidates(self, space_id, query, options=(), limit=50,
                           include_ids=None):
        """Keyword candidates of one space: [(record_id, score)]."""
        rows = self._store.keyword_search(space_id, query, limit=limit,
                                          include_ids=include_ids)
        return [(r[0], float(r[1])) for r in rows]

    def recall_candidates(self, query, space_id, limit=50, paths=None):
        """The candidate pool with every path's own score — the ruler's door.

        No fusion, no rerank, no top_k cut, no neighbours: just what each path
        found and how it scored it.
        """
        out = []
        for rc in self._recalls:
            if paths is not None and rc.name not in paths:
                continue
            if not rc.enabled:
                continue
            try:
                got = rc.candidates(space_id, query, (), limit)
            except EmbedUnavailable as e:
                # The provider behind this path is down: skip the path, the
                # others still answer.
                log.warning("embedding route skipped: path=%s reason=%s", rc.name, e)
                continue
            for rid, score in got:
                out.append(Candidate(record_id=rid, score=float(score),
                                     path_scores={rc.name: float(score)}))
        return out

    def search(self, query, space_id, top_k=None, context_before=None,
               context_after=None) -> List[SearchHit]:
        """Search one space: recall -> fuse -> rerank -> cut -> neighbours -> order.

        Returns evidence only: the text is the text that was written.
        """
        top_k = self._config.top_k if top_k is None else top_k
        before = self._config.context_before if context_before is None else context_before
        after = self._config.context_after if context_after is None else context_after
        over = max(top_k * 5, 50)

        paths = []
        path_scores = {}
        for rc in self._recalls:
            if not rc.enabled:
                continue
            try:
                got = rc.candidates(space_id, query, (), over)
            except (EmbedUnavailable, StoreBusy) as e:
                # The provider behind this path is down (timeout / HTTP error /
                # bad reply / gate full). The scheme promises this path is
                # skipped and the others still answer — a failed search would
                # be worse.
                log.warning("embedding route skipped: path=%s reason=%s", rc.name, e)
                continue
            paths.append(got)
            for rid, score in got:
                path_scores.setdefault(rid, {})[rc.name] = float(score)
        pool = fusion.merge_rrf(paths)
        if self._config.rerank_enabled and self._vc is not None:
            try:
                pool = fusion.rerank(self._store, space_id, query, pool,
                                      lambda: self._vc, top_n=self._config.rerank_n)
            except EmbedUnavailable as e:
                log.warning("embedding route skipped: rerank reason=%s", e)
        ordered = fusion.rank(pool, top_k)
        return self.enrich(space_id, ordered, before, after, path_scores=path_scores)

    def enrich(self, space_id, ordered, context_before=0, context_after=0,
               path_scores=None) -> List[SearchHit]:
        """Turn ranked ids into hits with their rows and their neighbours."""
        ordered = list(ordered)
        ids = [i for i, _ in ordered]
        rows = self._store.get_rows(space_id, ids)
        score_of = dict(ordered)
        hits = []
        for rid in ids:
            row = rows.get(rid)
            if not row:
                continue
            rec = Record.from_row(row)
            before, after = ([], [])
            if context_before or context_after:
                before, after = self._store.neighbors(
                    space_id, rec.session_id, rec.turn_index,
                    before=context_before, after=context_after)
            hits.append(SearchHit(
                record_id=rec.record_id, space_id=space_id,
                session_id=rec.session_id, text=rec.text,
                score=float(score_of.get(rid, 0.0)),
                turn_index=rec.turn_index, speaker=rec.speaker,
                direction=rec.direction, line=rec.line, ts=rec.ts,
                created_at=rec.created_at,
                path_scores=dict((path_scores or {}).get(rid, {})),
                neighbors_before=before, neighbors_after=after))
        if self._config.group_by_session:
            hits = self._group_by_session(hits)
        return hits

    @staticmethod
    def _group_by_session(hits):
        """Inside one session: by time. Between sessions: by the group's best score.

        hits arrive score-descending, so a session's first appearance carries
        that session's best score.
        """
        groups = {}
        order = []
        for h in hits:
            if h.session_id not in groups:
                groups[h.session_id] = []
                order.append(h.session_id)
            groups[h.session_id].append(h)
        out = []
        for sid in order:
            out.extend(sorted(groups[sid],
                              key=lambda h: (h.ts or "", h.turn_index if h.turn_index is not None else 0)))
        return out

    # --------------------------------------------------------------- getters

    @property
    def store(self) -> Store:
        return self._store

    @property
    def config(self) -> CoreConfig:
        return self._config

    @property
    def recalls(self) -> Sequence[Any]:
        return self._recalls

    @property
    def db(self):
        """The Database this core was built on (the shell reads /health state
        through it).
        """
        return self._db
