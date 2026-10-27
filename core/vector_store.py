"""Vector cache store: owns ONLY the core_vectors table.

复制自本项目早期服务层的 app/vector_store.py（本仓库不含该文件；2026-10-21，小助手），
改了：表名 vectors -> core_vectors。没改：读开自己的连接、写走 db 的写锁
（不绕开写序列化）、幂等插入、锁忙时抛 StoreBusy。
"""
import sqlite3

from .db import StoreBusy, is_locked


class VectorStore:
    def __init__(self, db):
        self._db = db

    def get_vectors(self, model, hashes):
        """{content_hash: vec_bytes} for any cached vectors."""
        hs = list(hashes)
        if not hs:
            return {}
        conn = self._db.connect()
        try:
            out = {}
            for i in range(0, len(hs), 500):
                chunk = hs[i:i + 500]
                ph = ",".join("?" * len(chunk))
                for h, b in conn.execute(
                    f"SELECT content_hash, vec FROM core_vectors "
                    f"WHERE model=? AND content_hash IN ({ph})", (model, *chunk)):
                    out[h] = bytes(b)
        except sqlite3.OperationalError as e:
            if is_locked(e):
                raise StoreBusy(str(e)) from e
            raise
        finally:
            conn.close()
        return out

    def put_vectors(self, model, dim, rows):
        """Cache vectors; rows = [(content_hash, vec_bytes)]. Idempotent; goes
        through the db write lock, never bypassing write serialization."""
        if not rows:
            return
        with self._db.writing() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(
                    "INSERT OR IGNORE INTO core_vectors(model, content_hash, dim, vec) "
                    "VALUES (?,?,?,?)",
                    [(model, h, dim, b) for h, b in rows])
                conn.commit()
            except sqlite3.OperationalError as e:
                conn.rollback()
                if is_locked(e):
                    raise StoreBusy(str(e)) from e
                raise
