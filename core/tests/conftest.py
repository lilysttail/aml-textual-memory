"""核心自测的夹具：一个临时库、一个核心、几行假数据。

临时库用 os.makedirs 自建目录（不是 pytest 的 tmp_path、也不是
tempfile.mkdtemp）：在这台机器上，那两种方式建出来的目录会被锁权限，
sqlite 在里面开不了库。
"""
import os
import shutil
import tempfile
import uuid

import pytest

from core import CoreConfig, Record, build
from core.testing import FakeEmbedder


def _tmpdir():
    base = tempfile.gettempdir()
    d = os.path.join(base, "core-test-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture()
def core():
    """一个只有关键词路的核心（不碰网络、不碰真模型）。"""
    d = _tmpdir()
    try:
        cfg = CoreConfig(db_path=os.path.join(d, "core.db"),
                         paths=("keyword",),
                         context_before=1, context_after=1)
        yield build(cfg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


class SpyEmbedder(FakeEmbedder):
    """A fake embedder that remembers which texts it was asked to embed."""

    def __init__(self, dim=16):
        super().__init__(dim=dim)
        self.embedded_texts = []

    def embed(self, texts, timeout_s=None):
        self.embedded_texts.extend(texts)
        return super().embed(texts, timeout_s=timeout_s)


@pytest.fixture()
def dense_core():
    """带一路假向量的核心（FakeEmbedder，确定性，不联网）。"""
    d = _tmpdir()
    try:
        cfg = CoreConfig(db_path=os.path.join(d, "dense.db"),
                         paths=("keyword", "dense"),
                         embedder=SpyEmbedder(dim=16),
                         dense_exclude_lines=("terminal",),
                         context_before=1, context_after=1)
        yield build(cfg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def rec(record_id, text, session_id="s1", turn_index=None, speaker="alice",
        ts="2026-10-21T05:00:00.000Z", line=None, valid=True):
    """一行假记录。"""
    return Record(record_id=record_id, session_id=session_id, text=text,
                  turn_index=turn_index, speaker=speaker, ts=ts, line=line,
                  valid=valid)
