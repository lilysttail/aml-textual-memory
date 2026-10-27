"""核心的存储层：结构化行、FTS 索引、邻居、快照换代。

复制自本项目早期服务层的 app/memory_store.py（本仓库不含该文件；2026-10-21，小助手），
改了：
  - 单列 content 拆成结构化列（说话人 / 时间 / 会话 / 轮次 / 线 / 方向 /
    判决状态 / 敏感标记），主键改成 (space_id, record_id) 复合键；
  - 新增：按 (space, session) 原子发号（core_sessions 游标）；
  - 新增：邻居取回（当前有效窗口顺序，valid=false 不露出）；
  - 新增：整份快照重建 + 原子换代（replace_snapshot）；
  - 切词器变成插头（默认仍是骨架那套英文切词；中文切词是 2b 的事）。
没改：FTS5 + BM25 的排序方向（bm25 越负越好，映射成越大越好的分数）、
回表时同时比 space_id 与 record_id、查询词数上限。

核心不认识数据来源：字段名是 space_id / session_id / record_id 这一套，
平台的名字由壳映射。
"""
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .db import StoreBusy, is_locked
from .models import Record

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Generic English stopwords (function words only). Not tuned to any dataset —
# the point is to stop boilerplate function words from crowding out content
# terms. Content words (identifiers, error names, domain terms) never appear.
_STOPWORDS = frozenset("""
a an the this that these those
i you he she it we they me him her us them my your his its our their mine yours
is am are was were be been being
do does did doing done
have has had having
will would shall should can could might must
and or but nor so yet for
if then else when while because as until although though
of in on at by to from with about into through during before after
above below up down out off over under again further
not no only own same than too very
here there where why how what which who whom whose
all any both each few more most other some such
s t re ve ll d m o
""".split())

# Month / weekday names must NEVER be stopwords: a query's "May" is usually
# the fifth month, not the modal verb, and the two cannot be told apart in
# English — keeping it indexed costs little and dropping it loses real hits.
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]

# Cap query width so a huge issue body can't produce a pathologically wide
# OR query. 128 distinct content terms is plenty; BM25 weights them by IDF,
# so rare terms (identifiers, error names) dominate the score and common
# boilerplate words contribute little.
_MAX_QUERY_TERMS = 128


def default_tokens(text: str, stem: bool = False):
    """The default tokenizer: word-ish tokens minus English function words.

    stem: apply light English stemming (plural / tense) to each kept term
    (index and query must use the same setting).
    """
    seen = set()
    kept = []
    for t in _TOKEN_RE.findall(text.lower()):
        if len(t) <= 1 or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        if stem:
            t = _stem_en(t)
        kept.append(t)
        if len(kept) >= _MAX_QUERY_TERMS:
            break
    return kept


_STEMMER = None


def _ensure_stemmer():
    """Load the Snowball English stemmer (Porter2). Fails loudly at startup
    when stemming is on but the package is missing — never silently treat
    "on" as "off".
    """
    global _STEMMER
    if _STEMMER is None:
        import snowballstemmer
        _STEMMER = snowballstemmer.stemmer("english")
    return _STEMMER


def _stem_en(word: str) -> str:
    """English stemmer: Snowball's official Porter2 implementation.

    Delegates to the snowballstemmer package (the `english` algorithm = the
    complete Porter2). Hand-written rules and deviations are gone — same
    word forms land on the same root the way the published algorithm does
    (visit/visited/visiting/visits → visit; change/changed/changing → chang;
    house/houses → hous). Known algorithm over-stems are accepted and
    documented in THIRD_PARTY_NOTICES / the keyword delivery; no exceptions
    are special-cased. Index and query run the same function, so stems
    always line up.
    """
    return _ensure_stemmer().stemWord(word)


def _ts_index_words(ts: str):
    """ISO event time -> index terms: numeric date, '8 may 2023', 'may 2023'.

    These go into the FTS index next to the text terms so a query's English
    month name (or the plain date) can find the record. The stored text and
    the rendered output are untouched — this is index-only.
    """
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return []
    mon = _MONTHS[dt.month - 1]
    return [f"{dt.year}-{dt.month:02d}-{dt.day:02d}",
            f"{dt.day} {mon} {dt.year}",
            f"{mon} {dt.year}"]


def fts_query(query: str, tokenizer=None, stem: bool = False) -> str:
    """Sanitize free text into a safe FTS5 OR query.

    tokenizer(text) -> list[str] is the plug (2b wires a Chinese tokenizer in
    here); without one the default English tokenizer is used.

    The default tokenizer drops one-letter terms and English function words
    (the scaffold's rule). A tokenizer plug decides for itself what a term
    is — a Chinese tokenizer cuts into words, and a per-character cut keeps
    single characters, which the default rule would throw away.
    stem applies only to the built-in English tokenizer (ignored when a plug
    is given); index and query must use the same value.
    """
    if tokenizer is not None:
        terms = [str(t) for t in tokenizer(query.lower()) if t]
    else:
        terms = default_tokens(query, stem=stem)
    seen = set()
    kept = []
    for t in terms:
        t = str(t)
        if not t or t in seen:
            continue
        seen.add(t)
        kept.append(t)
        if len(kept) >= _MAX_QUERY_TERMS:
            break
    if not kept:
        return '""'
    return " OR ".join(f'"{t}"' for t in kept)


class Store:
    """Everything the core knows about storage."""

    # The columns of core_records, in the order the SELECTs use them.
    _COLS = ("space_id", "record_id", "session_id", "turn_index", "speaker",
             "direction", "line", "ts", "created_at", "text", "sec_flag",
             "valid", "state", "meta")

    def __init__(self, db, tokenizer=None, stem_english: bool = False):
        self._db = db
        self._tokenizer = tokenizer
        self._stem_english = stem_english
        if stem_english:
            _ensure_stemmer()  # fails loudly at startup, not mid-search

    def _index_body(self, text: str, ts: str = None) -> str:
        """What goes into the FTS index: the text cut into terms.

        ts: the record's event time, if any — its numeric date and English
        month spellings are added to the index so date/month queries match
        (the stored text itself is never changed; this only touches the index).

        The same tokenizer is used when writing and when querying, so the
        index and the query always speak the same language. Without a
        tokenizer plug the default one keeps a whole CJK run as one term
        (exact-run matching); a Chinese tokenizer (2b) will cut it into
        words. The text itself is stored verbatim in core_records.
        """
        if self._tokenizer is not None:
            terms = [str(t) for t in self._tokenizer(text) if t]
        else:
            parts = [text]
            if ts:
                parts.extend(_ts_index_words(ts))
            terms = default_tokens(" ".join(parts), stem=self._stem_english)
        return " ".join(terms)

    @classmethod
    def _row_to_dict(cls, row):
        return dict(zip(cls._COLS, row))

    # ------------------------------------------------------------------ write

    def write(self, records, space_id: str, batch_id=None, conn=None):
        """Write records into one space. Idempotent on (space_id, batch_id).

        Returns {"written": n, "replay": bool}. A replay (same batch id)
        writes nothing and says so.

        conn: write inside a transaction the caller owns (the shell writes its
        own tables and the core's in one go). Without it the write gets its
        own transaction.
        """
        records = list(records)
        if not records:
            return {"written": 0, "replay": False}
        records = self._in_event_order(records)
        if conn is not None:
            return self._write_records(conn, records, space_id, batch_id)
        with self._db.writing() as c:
            try:
                c.execute("BEGIN IMMEDIATE")
                res = self._write_records(c, records, space_id, batch_id)
                if res.get("replay"):
                    c.rollback()
                    return res
                c.commit()
                return res
            except sqlite3.OperationalError as e:
                c.rollback()
                if is_locked(e):
                    raise StoreBusy(str(e)) from e
                raise

    def _write_records(self, conn, records, space_id: str, batch_id):
        """The write itself, inside whatever transaction the caller has open."""
        if batch_id is not None:
            cur = conn.execute(
                "INSERT OR IGNORE INTO core_batches(space_id, batch_id) "
                "VALUES (?,?)", (space_id, batch_id))
            if cur.rowcount == 0:
                return {"written": 0, "replay": True}
        written = 0
        for r in records:
            if self._insert(conn, r, space_id, batch_id):
                written += 1
        return {"written": written, "replay": False}

    def replace_snapshot(self, space_id: str, records, snapshot_id=None):
        """Rebuild one space's whole index from a snapshot, atomically.

        The old rows go away together with their neighbours (neighbours are
        taken from the currently valid set, so a deleted row's neighbours are
        not left behind). Re-running the same snapshot writes the same rows.

        A snapshot with two rows of the same id is refused: writing one of them
        and reporting success would silently shorten the snapshot.
        """
        records = list(records)
        seen = set()
        for r in records:
            if r.record_id in seen:
                raise ValueError(
                    f"the snapshot for space {space_id!r} has two rows with the same "
                    f"record_id {r.record_id!r} — nothing was written")
            seen.add(r.record_id)
        records = self._in_event_order(records)
        with self._db.writing() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM core_fts WHERE space_id=?", (space_id,))
                conn.execute("DELETE FROM core_records WHERE space_id=?", (space_id,))
                conn.execute("DELETE FROM core_sessions WHERE space_id=?", (space_id,))
                written = 0
                for r in records:
                    if self._insert(conn, r, space_id, None):
                        written += 1
                conn.execute(
                    "INSERT OR REPLACE INTO core_spaces(space_id, snapshot_id, updated_at) "
                    "VALUES (?,?,?)",
                    (space_id, snapshot_id or "",
                     datetime.now(timezone.utc).isoformat()))
                conn.commit()
                return {"written": written}
            except sqlite3.OperationalError as e:
                conn.rollback()
                if is_locked(e):
                    raise StoreBusy(str(e)) from e
                raise

    @staticmethod
    def _in_event_order(records):
        """Records in event order.

        Rows that know when they happened come first, by their event time;
        rows that do not follow, in arrival order. The turn indexes follow
        this order, so a request whose messages arrive out of order still
        gets its neighbours in time order, and a request without timestamps
        gets its turn indexes in arrival order.
        """
        with_ts = [r for r in records if r.ts]
        without_ts = [r for r in records if not r.ts]
        with_ts.sort(key=lambda r: r.ts)
        return with_ts + without_ts

    def _insert(self, conn, r: Record, space_id: str, batch_id) -> bool:
        """One row + its FTS entry. Returns True when the row was new."""
        row = r.to_row()
        turn = row["turn_index"]
        if turn is None:
            turn = self._next_turn_index(conn, space_id, row["session_id"])
            row["turn_index"] = turn
        else:
            # An explicit turn index moves the cursor forward, in the same
            # transaction: the next auto turn must not be handed the same
            # number (and the neighbours of a session must not collide).
            self._bump_turn_index(conn, space_id, row["session_id"], turn)
        cur = conn.execute(
            "INSERT OR IGNORE INTO core_records("
            "space_id, record_id, session_id, turn_index, speaker, direction, line, "
            "ts, created_at, text, sec_flag, valid, state, meta, batch_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (space_id, row["record_id"], row["session_id"], row["turn_index"],
             row["speaker"], row["direction"], row["line"], row["ts"],
             row["created_at"], row["text"], row["sec_flag"], row["valid"],
             row["state"], row["meta"], batch_id))
        if cur.rowcount == 0:
            return False
        conn.execute(
            "INSERT OR REPLACE INTO core_fts(space_id, record_id, session_id, speaker, body) "
            "VALUES (?,?,?,?,?)",
            (space_id, row["record_id"], row["session_id"], row["speaker"],
             self._index_body(row["text"], ts=row["ts"])))
        return True

    def _bump_turn_index(self, conn, space_id: str, session_id: str, turn) -> None:
        """Move the session cursor to at least turn + 1 (same transaction)."""
        row = conn.execute(
            "SELECT next_turn_index FROM core_sessions "
            "WHERE space_id=? AND session_id=?", (space_id, session_id)).fetchone()
        current = int(row[0]) if row is not None else 0
        nxt = max(current, int(turn) + 1)
        conn.execute(
            "INSERT INTO core_sessions(space_id, session_id, next_turn_index) "
            "VALUES (?,?,?) ON CONFLICT(space_id, session_id) DO UPDATE SET "
            "next_turn_index=excluded.next_turn_index",
            (space_id, session_id, nxt))

    def _next_turn_index(self, conn, space_id: str, session_id: str) -> int:
        """Hand out the next turn index of one (space, session).

        Same transaction as the insert that uses it, and the session row is
        updated in the same transaction — so two writers of the same session
        can never be handed the same number.
        """
        row = conn.execute(
            "SELECT next_turn_index FROM core_sessions "
            "WHERE space_id=? AND session_id=?", (space_id, session_id)).fetchone()
        nxt = int(row[0]) if row is not None else 0
        conn.execute(
            "INSERT INTO core_sessions(space_id, session_id, next_turn_index) "
            "VALUES (?,?,?) ON CONFLICT(space_id, session_id) DO UPDATE SET "
            "next_turn_index=excluded.next_turn_index",
            (space_id, session_id, nxt + 1))
        return nxt

    # ------------------------------------------------------------------- read

    def keyword_search(self, space_id: str, query: str, limit: int = 50,
                      include_ids=None):
        """FTS5 BM25 candidates, space-partitioned.

        Returns [(record_id, score)]; better match = higher score.

        include_ids: restrict the candidates to these ids (the shell uses it to
        keep a scoped search inside a scope; the core does not know what a
        scope is).
        """
        fq = fts_query(query, self._tokenizer, stem=self._stem_english)
        want = None if include_ids is None else set(include_ids)
        conn = self._db.connect()
        try:
            if want is not None and not want:
                return []
            rows = []
            if want is None:
                rows = conn.execute(
                    """SELECT r.record_id, bm25(core_fts)
                       FROM core_fts
                       JOIN core_records r ON r.record_id = core_fts.record_id
                                          AND r.space_id = core_fts.space_id
                       WHERE core_fts MATCH ? AND core_fts.space_id = ?
                         AND r.valid = 1
                       ORDER BY bm25(core_fts) LIMIT ?""",
                    (fq, space_id, limit)).fetchall()
            else:
                ids = list(want)
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    ph = ",".join("?" * len(chunk))
                    rows.extend(conn.execute(
                        f"""SELECT r.record_id, bm25(core_fts)
                            FROM core_fts
                            JOIN core_records r ON r.record_id = core_fts.record_id
                                               AND r.space_id = core_fts.space_id
                            WHERE core_fts MATCH ? AND core_fts.space_id = ?
                              AND r.valid = 1 AND r.record_id IN ({ph})
                            ORDER BY bm25(core_fts) LIMIT ?""",
                        (fq, space_id, *chunk, limit)).fetchall())
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()
        out = []
        for record_id, bm25 in rows:
            # FTS5 bm25() <= 0, more negative = better match. Map to [0,1) so a
            # BETTER match gets a HIGHER score: bm25=0->0, -0.5->0.33, -10->0.91.
            score = -bm25 / (1.0 - bm25)
            out.append((record_id, float(score)))
        out.sort(key=lambda r: r[1], reverse=True)
        return out[:limit]

    def get_rows(self, space_id, record_ids):
        """{record_id: row dict} for the given ids (one space only)."""
        ids = list(record_ids)
        if not ids:
            return {}
        conn = self._db.connect()
        try:
            out = {}
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                ph = ",".join("?" * len(chunk))
                for row in conn.execute(
                    f"SELECT space_id, record_id, session_id, turn_index, speaker, "
                    f"direction, line, ts, created_at, text, sec_flag, valid, state, meta "
                    f"FROM core_records WHERE space_id=? AND record_id IN ({ph})",
                        (space_id, *chunk)):
                    d = self._row_to_dict(row)
                    out[d["record_id"]] = d
            return out
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()

    def get_record(self, space_id: str, record_id: str):
        rows = self.get_rows(space_id, [record_id])
        row = rows.get(record_id)
        return Record.from_row(row) if row else None

    def iter_records(self, space_id: str, exclude_lines=()):
        """[(record_id, text)] over one space (dense full mode).

        exclude_lines: those `line` values are skipped — a line that must not
        reach the vector route (e.g. terminal lines) still reaches the keyword
        index, it is only left out of the vectors.
        """
        excl = [l for l in (exclude_lines or ()) if l]
        conn = self._db.connect()
        try:
            if excl:
                ph = ",".join("?" * len(excl))
                return list(conn.execute(
                    f"SELECT record_id, text FROM core_records "
                    f"WHERE space_id=? AND valid=1 AND (line IS NULL OR line NOT IN ({ph}))",
                    (space_id, *excl)))
            return list(conn.execute(
                "SELECT record_id, text FROM core_records WHERE space_id=? AND valid=1",
                (space_id,)))
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()

    def count(self, space_id: str) -> int:
        conn = self._db.connect()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM core_records WHERE space_id=?", (space_id,)).fetchone()[0]
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()

    def neighbors(self, space_id: str, session_id: str, turn_index,
                  before: int = 0, after: int = 0):
        """The current-valid neighbours of one record.

        Neighbours come from the CURRENT valid set in window order — a deleted
        row is not skipped over to a further one, it is simply gone (and its
        neighbours are the rows that are still valid around it). Rows with
        valid = 0 never show up here.
        """
        before_rows, after_rows = [], []
        if turn_index is None:
            return before_rows, after_rows
        conn = self._db.connect()
        try:
            cols = ("space_id, record_id, session_id, turn_index, speaker, direction, "
                    "line, ts, created_at, text, sec_flag, valid, state, meta")
            if before > 0:
                before_rows = [Record.from_row(self._row_to_dict(r)) for r in conn.execute(
                    f"SELECT {cols} FROM core_records "
                    f"WHERE space_id=? AND session_id=? AND valid=1 AND turn_index < ? "
                    f"ORDER BY turn_index DESC LIMIT ?",
                    (space_id, session_id, turn_index, before)).fetchall()]
                before_rows.reverse()          # chronological
            if after > 0:
                after_rows = [Record.from_row(self._row_to_dict(r)) for r in conn.execute(
                    f"SELECT {cols} FROM core_records "
                    f"WHERE space_id=? AND session_id=? AND valid=1 AND turn_index > ? "
                    f"ORDER BY turn_index ASC LIMIT ?",
                    (space_id, session_id, turn_index, after)).fetchall()]
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()
        return before_rows, after_rows

    def session_records(self, space_id: str, session_id: str):
        """Every valid row of one session, in window order (tests / tools)."""
        conn = self._db.connect()
        try:
            cols = ("space_id, record_id, session_id, turn_index, speaker, direction, "
                    "line, ts, created_at, text, sec_flag, valid, state, meta")
            return [Record.from_row(self._row_to_dict(r)) for r in conn.execute(
                f"SELECT {cols} FROM core_records "
                f"WHERE space_id=? AND session_id=? AND valid=1 "
                f"ORDER BY turn_index ASC, ts ASC", (space_id, session_id)).fetchall()]
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()
