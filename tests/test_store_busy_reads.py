"""读存储时遇到「数据库被锁」要翻成 StoreBusy（HTTP 503，可重试），不是别的错。

核心的读方法都写了 `except sqlite3.OperationalError` + `is_locked` 的翻译：真遇到
锁时抛的应该是 StoreBusy（让壳回 503、平台重试），而不是别的错把真错误盖住。
这里用一个一执行就报「database is locked」的假连接把每个读方法都走一遍。
"""
import os
import sqlite3
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)

import pytest  # noqa: E402

from core.db import Database, StoreBusy  # noqa: E402
from core.store import Store  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402


class _Failing:
    def __init__(self, msg):
        self._msg = msg

    def execute(self, *a, **k):
        raise sqlite3.OperationalError(self._msg)

    def close(self):
        pass


def _db(msg):
    db = Database(os.path.join(_TMPROOT, f"busy_{uuid.uuid4().hex}.db"))
    db.create_schema()
    db.connect = lambda: _Failing(msg)
    return db


CALLS = {
    "store.keyword_search": lambda db: Store(db).keyword_search("u", "hello world"),
    "store.get_rows": lambda db: Store(db).get_rows("u", ["r1"]),
    "store.count": lambda db: Store(db).count("u"),
    "store.iter_records": lambda db: Store(db).iter_records("u"),
    "store.neighbors": lambda db: Store(db).neighbors("u", "s", 1, before=1, after=1),
    "store.session_records": lambda db: Store(db).session_records("u", "s"),
    "vector.get_vectors": lambda db: VectorStore(db).get_vectors("m", ["h"]),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_locked_database_becomes_store_busy(name):
    with pytest.raises(StoreBusy):
        CALLS[name](_db("database is locked"))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_any_other_operational_error_is_left_alone(name):
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        CALLS[name](_db("no such table: nope"))
