"""核心存储层自测：写入、隔离、幂等、事务、切词器插头。

对应方案 6.1 的 T1 / T2 / T5 / T6 / T7 / T9 / T10 / T11。

注意：这一单只有「按字找」这一路（语义召回是 2b），所以断言搜的是
「整段字面」或「一个完整的词」；中文子串能不能命中，取决于切词器插头
（T10 就是验这个），不写「换个说法也能搜到」。
"""
import os
import shutil
import sqlite3
import tempfile
import uuid

import pytest

from core import CoreConfig, Record, build
from core.tests.conftest import rec


def test_T1_roundtrip_keeps_the_text_verbatim(core):
    core.write([rec("r1", "第一行\n第二行  空格")], space_id="home")
    row = core.store.get_rows("home", ["r1"])["r1"]
    assert row["text"] == "第一行\n第二行  空格"


def test_T2_speaker_and_time_are_columns_not_guessed(core):
    core.write([rec("r1", "一句原话", speaker="bob", ts="2026-10-20T12:34:56.000Z")],
               space_id="home")
    hit = core.search("一句原话", space_id="home", top_k=1)[0]
    assert hit.speaker == "bob"
    assert hit.ts == "2026-10-20T12:34:56.000Z"
    assert hit.session_id == "s1"


def test_T5_returned_text_is_the_written_text(core):
    text = "带  空格\n和换行的一行"
    core.write([rec("r1", text)], space_id="home")
    row = core.store.get_rows("home", ["r1"])["r1"]
    assert row["text"] == text
    hit = core.search("和换行的一行", space_id="home", top_k=1)[0]
    assert hit.text == text


def test_T6_one_space_cannot_see_another(core):
    core.write([rec("r1", "只属于甲的一句话")], space_id="jia")
    core.write([rec("r2", "只属于乙的一句话")], space_id="yi")
    jia = [h.record_id for h in core.search("只属于甲的一句话", space_id="jia", top_k=10)]
    yi = [h.record_id for h in core.search("只属于乙的一句话", space_id="yi", top_k=10)]
    assert jia == ["r1"]
    assert yi == ["r2"]


def test_T7_same_batch_twice_writes_once(core):
    core.write([rec("r1", "同一批")], space_id="home", batch_id="b1")
    again = core.write([rec("r1", "同一批")], space_id="home", batch_id="b1")
    assert again["replay"] is True
    assert core.store.count("home") == 1


def test_T9_a_failing_write_leaves_nothing_behind(core):
    """写一半失败：一条都留不下（事务回滚）。"""
    bad = Record(record_id="r2", session_id="s1", text="坏的", meta={"x": {1, 2}})
    with pytest.raises(TypeError):
        core.write([rec("r1", "好的"), bad], space_id="home", batch_id="b1")
    assert core.store.count("home") == 0
    # 约束不满足的行会被拒掉，其它行照写，不留半条
    res = core.write([rec("r3", "好的"),
                      Record(record_id="r4", session_id="s1", text=None)],
                     space_id="home", batch_id="b2")
    assert res["written"] == 1
    assert core.store.count("home") == 1


def test_T10_the_tokenizer_is_a_plug(core):
    """带一个假中文切词器（按字切）：子串也能命中。

    这是切词器插头的作用，不是语义召回——它只决定「字怎么切」。
    """
    calls = []

    def tokenizer(text):
        calls.append(text)
        return list(text)

    d = os.path.join(tempfile.gettempdir(), "core-tok-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    try:
        cfg = CoreConfig(db_path=os.path.join(d, "tok.db"), paths=("keyword",),
                         tokenizer=tokenizer)
        c = build(cfg)
        c.write([rec("r1", "记忆库设计")], space_id="home")
        hits = c.search("记忆库", space_id="home", top_k=5)
        assert [h.record_id for h in hits] == ["r1"]
        assert calls, "the tokenizer plug was never called"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_T11_two_spaces_same_record_id_and_same_batch_id(core):
    core.write([rec("same-id", "甲的正文")], space_id="jia", batch_id="b1")
    core.write([rec("same-id", "乙的正文")], space_id="yi", batch_id="b1")
    assert core.store.count("jia") == 1
    assert core.store.count("yi") == 1
    jia = [h.text for h in core.search("甲的正文", space_id="jia", top_k=5)]
    yi = [h.text for h in core.search("乙的正文", space_id="yi", top_k=5)]
    assert jia == ["甲的正文"]
    assert yi == ["乙的正文"]
    # each space replays its own batch idempotently
    assert core.write([rec("same-id", "甲的正文")], space_id="jia", batch_id="b1")["replay"]
    assert core.write([rec("same-id", "乙的正文")], space_id="yi", batch_id="b1")["replay"]
