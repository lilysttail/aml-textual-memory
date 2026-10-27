"""Database: connections, the write lock, the table registry, and wipe.

复制自本项目早期服务层的 app/db.py（本仓库不含该文件；2026-10-21，小助手）。改了：
  - 表注册表换成核心自己的八张表（core_*）；比赛服务用的那几张表一张都不进核心；
  - 去掉为老列做兼容迁移的那段 ALTER（核心没有那一列）；
  - 索引换成核心的三条。
没改：连接方式、写锁（单写者 + 有界等待）、busy_timeout、StoreBusy 语义、
wipe 的做法。骨架那套写锁是骨架里最该白捡的一块：写者串行、读不设卡。
"""
import os
import sqlite3
import threading
from contextlib import contextmanager

# name -> (DDL, is_evaluation_data). is_data=True tables are wiped by wipe().
# A table that is NOT evaluation data (is_data=False) is never cleared by a wipe: it describes how the database is to
# be read, not what the evaluation put in it (core_write_settings is the one: see its entry).
TABLES = {
    # 一条记录 = 一条消息。主键是 (space_id, record_id)：两个空间用同一个
    # record_id 各存各的，谁也抢不到谁。
    "core_records": ("""CREATE TABLE IF NOT EXISTS core_records(
                     space_id TEXT NOT NULL, record_id TEXT NOT NULL,
                     session_id TEXT NOT NULL,
                     turn_index INTEGER, speaker TEXT, direction TEXT, line TEXT,
                     ts TEXT, created_at TEXT, text TEXT NOT NULL, sec_flag INTEGER,
                     valid INTEGER, state TEXT, meta TEXT, batch_id TEXT,
                     PRIMARY KEY(space_id, record_id))""", True),
    "core_fts": ("CREATE VIRTUAL TABLE IF NOT EXISTS core_fts "
                 "USING fts5(space_id UNINDEXED, record_id UNINDEXED, "
                 "session_id UNINDEXED, speaker UNINDEXED, body)", True),
    "core_vectors": ("""CREATE TABLE IF NOT EXISTS core_vectors(
                     model TEXT, content_hash TEXT, dim INTEGER, vec BLOB,
                     PRIMARY KEY(model, content_hash))""", True),
    # 幂等键：同 (space_id, batch_id) 写两次不重复落。
    "core_batches": ("""CREATE TABLE IF NOT EXISTS core_batches(
                     space_id TEXT, batch_id TEXT,
                     PRIMARY KEY(space_id, batch_id))""", True),
    # 会话内的轮次游标：turn_index 在同一事务里按 (space, session) 原子发号。
    "core_sessions": ("""CREATE TABLE IF NOT EXISTS core_sessions(
                     space_id TEXT, session_id TEXT, next_turn_index INTEGER,
                     PRIMARY KEY(space_id, session_id))""", True),
    # 这个空间当前这份快照的版本号（整份重建 + 原子换代用）。
    "core_spaces": ("""CREATE TABLE IF NOT EXISTS core_spaces(
                     space_id TEXT PRIMARY KEY, snapshot_id TEXT, updated_at TEXT)""", True),
    # 核心自己的写入设置（core/write_settings.py）：数据是按哪些设置写进去的。
    # 换切词器之后旧索引会用错的词回答，所以对不上就拒启动。
    # 不是评测数据（is_data=False），清库不动它：启动时只核 / 记一次（core/write_settings.py check_or_record），
    # 写数据时不回写；清库把它一起清掉的话，清库后不重启就接着写数据，数据在、记录没了，下一次重启就拒启动
    # 。留着它不会挡路：库里没有数据时启动会把它换成当前设置。
    "core_write_settings": ("""CREATE TABLE IF NOT EXISTS core_write_settings(
                     name TEXT PRIMARY KEY, value TEXT)""", False),
    # 写入时向量计算失败的记录。这张表只解释「为什么这次没算成」，
    # 不参与「缺不缺」的判断——缺不缺以 core_records + core_vectors 的事实差集为准。
    "core_pending_vectors": ("""CREATE TABLE IF NOT EXISTS core_pending_vectors(
                     space_id TEXT NOT NULL,
                     record_id TEXT NOT NULL,
                     content_hash TEXT NOT NULL,
                     reason TEXT,
                     retry_count INTEGER NOT NULL DEFAULT 0,
                     last_attempt_at TEXT,
                     PRIMARY KEY (space_id, record_id))""", True),
}

_INDEX_DDL = [
    # 取邻居按这个走。
    "CREATE INDEX IF NOT EXISTS idx_core_session "
    "ON core_records(space_id, session_id, turn_index)",
    # 按时间排。
    "CREATE INDEX IF NOT EXISTS idx_core_ts "
    "ON core_records(space_id, ts)",
    # 只捞当前有效的行。
    "CREATE INDEX IF NOT EXISTS idx_core_valid "
    "ON core_records(space_id, valid, session_id, turn_index)",
]


class StoreBusy(Exception):
    """Lock wait exceeded busy_timeout; caller should answer 503 (retryable)."""


def is_locked(err: sqlite3.OperationalError) -> bool:
    msg = str(err).lower()
    return "database is locked" in msg or "database is busy" in msg


class Database:
    def __init__(self, path: str, busy_timeout_ms: int = 5000,
                 write_lock_timeout_ms: int = 10000, lock=None):
        self._path = os.path.abspath(path)
        self._busy_timeout_ms = busy_timeout_ms
        self._write_lock_timeout_ms = write_lock_timeout_ms
        # Serializes all writers in this process (deployment is single-worker).
        # Reads are not gated: they open their own connection and WAL lets them
        # run concurrently with the single writer.
        # lock: share one lock with another Database on the same file (the
        # shell and the core can point at the same sqlite file; without a
        # shared lock they would each think they are the only writer).
        self._write_lock = lock or threading.Lock()
        self.write_lock = self._write_lock
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=self._busy_timeout_ms / 1000.0)
        try:
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        except Exception:
            # don't leak a half-open connection if the PRAGMA fails
            conn.close()
            raise
        return conn

    @contextmanager
    def writing(self):
        """Acquire the write lock (bounded wait) and yield a fresh connection.

        The lock makes writers FIFO so only one is ever at BEGIN IMMEDIATE. A
        fresh short-lived connection per write keeps this simple. Exceed the
        lock wait budget -> StoreBusy -> 503, never an unbounded wait. A failure
        inside releases the lock (the conn is closed in finally).
        """
        if not self._write_lock.acquire(timeout=self._write_lock_timeout_ms / 1000.0):
            raise StoreBusy("write lock wait exceeded")
        conn = None
        try:
            conn = self.connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()
            self._write_lock.release()

    def create_schema(self) -> None:
        """One-time setup: WAL + all registered tables + their indexes."""
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            for ddl, _ in TABLES.values():
                conn.execute(ddl)
            for ddl in _INDEX_DDL:
                conn.execute(ddl)
            conn.commit()
        finally:
            conn.close()

    def wipe(self, extra_tables=()) -> dict:
        """Clear all evaluation-data tables (see scripts/wipe.py). The table
        set comes from the TABLES registry — no separate list to keep in sync.

        A database from an older version does not have every table in the
        registry: the missing ones are SKIPPED (and reported), not created.
        `extra_tables` names tables this core does not know about (the ones a
        retired service left behind): they are cleared when they are there,
        never created and never dropped. The core does not care what they are
        called — the caller decides.

        The wipe never touches the database's side files (`-wal`, `-shm`): while another process holds the
        database open (the running service does), those files are theirs too, and deleting them from under it
        loses what was just committed and breaks every later connection ("disk I/O error"). After the commit
        it asks SQLite to fold the log into the database file (a checkpoint); SQLite removes the side files
        itself when the last connection closes.

        Returns {"cleared": [...], "skipped": [...], "checkpoint": one of "truncated" / "partial" /
        "not_wal" / "unavailable"}: "truncated" = the log is folded into the database file and emptied;
        "partial" = another connection was reading, SQLite folds the rest in later (the wipe is committed and
        visible to everyone either way)."""
        cleared = []
        skipped = []
        checkpoint = "unavailable"
        with self.writing() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for name, (_, is_data) in TABLES.items():
                    if not is_data:
                        continue
                    if not table_exists(conn, name):
                        skipped.append(name)
                        continue
                    conn.execute(f"DELETE FROM {name}")
                    cleared.append(name)
                for name in extra_tables:
                    if not table_exists(conn, name):
                        continue
                    conn.execute(f"DELETE FROM {name}")
                    cleared.append(name)
                conn.commit()
            except sqlite3.OperationalError as e:
                conn.rollback()
                if is_locked(e):
                    raise StoreBusy(str(e)) from e
                raise
            checkpoint = _checkpoint(conn)
        return {"cleared": cleared, "skipped": skipped, "checkpoint": checkpoint}


def _checkpoint(conn) -> str:
    """Best effort: fold the write-ahead log into the database file. Never raises and never deletes a file: the
    wipe has already committed, a checkpoint that cannot finish only means SQLite does it later."""
    try:
        busy, log, _done = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error:
        return "unavailable"
    if log == -1:
        return "not_wal"          # the database is not in WAL mode: there is no log to fold in
    return "partial" if busy else "truncated"


def table_exists(conn, name: str) -> bool:
    """Is this table in this database? (an older database has fewer tables)"""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,)).fetchone()
    return row is not None


def business_tables():
    """Names of all evaluation-data tables (the ones wipe() clears)."""
    return [n for n, (_, is_data) in TABLES.items() if is_data]
