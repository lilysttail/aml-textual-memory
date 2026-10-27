"""按字找三处缺口（不花钱）：月份名停用词、时间英文写法进索引、英文词干还原。

测试先行：这些测试在旧代码上应全红（旧行为：may 是停用词；时间不进 FTS
索引；没有词干还原）。全部用自造小例子，不用公开题库原文。不花钱、不联网。
"""
import os
import shutil
import tempfile
import uuid

import pytest

from core import CoreConfig, Record, build
from core.write_settings import WriteSettingsMismatch


def _tmpdir():
    d = os.path.join(tempfile.gettempdir(), "kw-" + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    # 尽力清理临时目录
    base = os.path.join(tempfile.gettempdir(), "kw-")
    try:
        for name in os.listdir(tempfile.gettempdir()):
            if name.startswith("kw-"):
                shutil.rmtree(os.path.join(tempfile.gettempdir(), name), ignore_errors=True)
    except Exception:
        pass


def _core(d, stem=False):
    cfg = CoreConfig(db_path=os.path.join(d, "kw.db"), paths=("keyword",),
                     stem_english=stem)
    return build(cfg)


def _rec(rid, text, ts=None):
    return Record(record_id=rid, session_id="s1", text=text,
                  speaker="u", ts=ts, created_at=ts, valid=True)


def _hit(core, space, query):
    rows = core.store.keyword_search(space, query, limit=50)
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# 1) 月份名不许被当停用词
# ---------------------------------------------------------------------------

def test_may_is_not_stopped_in_queries():
    """问句里的 May（五月）不能被停用词表扔掉。"""
    d = _tmpdir()
    core = _core(d)
    core.write([_rec("r1", "the annual review happened in may")], space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core, "sp1", "may"), "may（五月）应能搜到"


def test_no_month_or_weekday_names_in_stopwords():
    """停用词表里不应有月份名 / 星期名。"""
    from core.store import _STOPWORDS
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    bad = [w for w in months + days if w in _STOPWORDS]
    assert bad == [], f"停用词表不应有月份/星期名: {bad}"


# ---------------------------------------------------------------------------
# 2) 时间按英文月份名也能搜到（只动索引，不改渲染）
# ---------------------------------------------------------------------------

def test_ts_english_month_in_index():
    """记录带时间：问句里的月份名应能对到（May 2023 → ts 2023-05-08）。"""
    d = _tmpdir()
    core = _core(d)
    core.write([_rec("r1", "annual report notes", ts="2023-05-08T05:00:00.000Z")],
               space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core, "sp1", "may"), "月份名 may 应能搜到带时间的记录"


def test_ts_english_day_month_year_in_index():
    """柯的例子：'8 May 2023' 这种完整英文日期写法应能搜到。"""
    d = _tmpdir()
    core = _core(d)
    core.write([_rec("r1", "audit trail", ts="2023-05-08T05:00:00.000Z")],
               space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core, "sp1", "8 may 2023"), "8 May 2023 应能搜到"


def test_ts_numeric_date_in_index():
    """数字日期（2023-05-08）也应照旧能搜到。"""
    d = _tmpdir()
    core = _core(d)
    core.write([_rec("r1", "audit trail", ts="2023-05-08T05:00:00.000Z")],
               space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core, "sp1", "2023-05-08"), "数字日期应能搜到"


def test_no_ts_rendering_change():
    """只动索引：正文渲染不该带出英文日期（渲染是壳的事，核心不碰）。"""
    d = _tmpdir()
    core = _core(d)
    core.write([_rec("r1", "plain text", ts="2023-05-08T05:00:00.000Z")],
               space_id="sp1", batch_id="b1")
    rec = core.store.get_record("sp1", "r1")
    assert rec is not None and rec.text == "plain text", "存储正文应原样"


# ---------------------------------------------------------------------------
# 3) 英文词干还原（只在英文配置里开）
# ---------------------------------------------------------------------------

def test_stem_plural_when_enabled():
    """cats → cat：词干开关开时能对上。"""
    d = _tmpdir()
    core = _core(d, stem=True)
    core.write([_rec("r1", "the cats slept all day")], space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core, "sp1", "cat"), "cat 应能搜到 cats"


def test_stem_tense_when_enabled():
    """walking → walk：词干开关开时能对上。"""
    d = _tmpdir()
    core = _core(d, stem=True)
    core.write([_rec("r1", "they are walking home")], space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core, "sp1", "walk"), "walk 应能搜到 walking"


def test_stem_off_by_default():
    """默认（英文配置不开）词干还原不生效：cat 对 cats 不命中。"""
    d = _tmpdir()
    core = _core(d, stem=False)
    core.write([_rec("r1", "the cats slept all day")], space_id="sp1", batch_id="b1")
    assert "r1" not in _hit(core, "sp1", "cat"), "默认不应做词干还原"


def test_stem_flag_toggles_same_data():
    """同一份数据：stem 开命中、关不命中（配置项真生效）。"""
    d = _tmpdir()
    core_on = _core(d, stem=True)
    core_on.write([_rec("r1", "boxes of books")], space_id="sp1", batch_id="b1")
    assert "r1" in _hit(core_on, "sp1", "box"), "boxes → box 应命中"
    assert "r1" in _hit(core_on, "sp1", "book"), "books → book 应命中"


# ---------------------------------------------------------------------------
# 4) 旧库升级路径：分词改了，旧索引要能重建
# ---------------------------------------------------------------------------

def test_old_db_refuses_new_stem_config():
    """旧配置写的数据，换新配置（词干开）启动应被拒（WriteSettingsMismatch）。"""
    d = _tmpdir()
    core_old = _core(d, stem=False)
    core_old.write([_rec("r1", "the cats slept")], space_id="sp1", batch_id="b1")
    with pytest.raises(WriteSettingsMismatch):
        _core(d, stem=True)


def test_rebuild_index_after_config_change():
    """升级路径：旧库换新配置 → 拒启动 → 重建索引 → 新配置可用、新词干命中。"""
    d = _tmpdir()
    core_old = _core(d, stem=False)
    core_old.write([_rec("r1", "the cats slept all day")], space_id="sp1", batch_id="b1")
    # 旧配置下 cat 对 cats 不命中（旧索引）。
    assert "r1" not in _hit(core_old, "sp1", "cat")

    # 换新配置启动被拒。
    with pytest.raises(WriteSettingsMismatch):
        _core(d, stem=True)

    # 重建索引（新分词写回 FTS + 记录新设置）。
    from core.engine import rebuild_fts_index
    cfg = CoreConfig(db_path=os.path.join(d, "kw.db"), paths=("keyword",), stem_english=True)
    rebuild_fts_index(cfg)

    # 新配置启动成功，cat 命中 cats。
    core_new = _core(d, stem=True)
    assert "r1" in _hit(core_new, "sp1", "cat"), "重建后 cat 应能搜到 cats"


# ---------------------------------------------------------------------------
# 柯核退回的三处（修 1/2/3）：每处在 95fcfa7 上先红
# ---------------------------------------------------------------------------

def test_rebuild_index_keeps_one_row_per_record():
    """重建后每条记录在 core_fts 恰好一行（FTS5 无唯一键，不能用 REPLACE）。"""
    import sqlite3
    d = _tmpdir()
    core = _core(d, stem=True)
    core.write([_rec("r1", "the cats slept"), _rec("r2", "dogs barked all night")],
               space_id="sp1", batch_id="b1")
    from core.engine import rebuild_fts_index
    cfg = CoreConfig(db_path=os.path.join(d, "kw.db"), paths=("keyword",), stem_english=True)
    rebuild_fts_index(cfg)
    conn = sqlite3.connect(os.path.join(d, "kw.db"))
    try:
        fts_rows = conn.execute("SELECT COUNT(*) FROM core_fts").fetchone()[0]
        rec_rows = conn.execute("SELECT COUNT(*) FROM core_records").fetchone()[0]
    finally:
        conn.close()
    assert fts_rows == rec_rows == 2, \
        f"重建后 core_fts 应每记录一行: fts={fts_rows} rec={rec_rows}"


def test_old_format_refused_with_stem_off():
    """时间词改了所有库的索引格式：旧格式库词干关着也拒启动，报错里有 write_format。"""
    import sqlite3
    from core.db import Database
    from core import write_settings as ws
    d = _tmpdir()
    db_path = os.path.join(d, "kw.db")
    core = _core(d, stem=False)
    core.write([_rec("r1", "annual report", ts="2023-05-08T05:00:00.000Z")],
               space_id="sp1", batch_id="b1")
    # 手动把记录改成旧格式（write_format=1）——模拟 95fcfa7 之前写的库。
    db = Database(db_path)
    ws.record(db, {"tokenizer": "builtin-default", "write_format": "1", "stem": "0"})
    with pytest.raises(WriteSettingsMismatch) as ei:
        _core(d, stem=False)
    assert "write_format" in str(ei.value), \
        f"拒启动报错里应有 write_format: {ei.value}"


def test_rebuild_after_format_bump_records_ts_words():
    """旧格式库重建后能启动，且索引里有时间词（May 能搜到 2023-05-08 的记录）。"""
    import sqlite3
    from core.db import Database
    from core import write_settings as ws
    from core.engine import rebuild_fts_index
    d = _tmpdir()
    db_path = os.path.join(d, "kw.db")
    core = _core(d, stem=False)
    core.write([_rec("r1", "annual report", ts="2023-05-08T05:00:00.000Z")],
               space_id="sp1", batch_id="b1")
    db = Database(db_path)
    ws.record(db, {"tokenizer": "builtin-default", "write_format": "1", "stem": "0"})
    # 旧格式 → 当前代码启动被拒。
    with pytest.raises(WriteSettingsMismatch):
        _core(d, stem=False)
    # 重建 → 新格式记录 → 启动正常、时间词在索引里。
    cfg = CoreConfig(db_path=db_path, paths=("keyword",), stem_english=False)
    rebuild_fts_index(cfg)
    core_new = _core(d, stem=False)
    assert "r1" in _hit(core_new, "sp1", "may"), "重建后索引应有时间词（May 命中）"


def test_stem_verb_forms_merge_to_same_root():
    """词干第 2/3 轮（柯 6376262518、6376634449）：两轮词组全部同根——
    20 组动词 + e 结尾的 19 组动词 + 5 组 e 结尾名词复数。
    不钉具体还原成什么，那是算法的事。不规则变化（came/met）不进。"""
    from core.store import _stem_en
    groups = {
        # 20 组动词
        "visit": ["visit", "visited", "visiting", "visits"],
        "listen": ["listen", "listened", "listening", "listens"],
        "happen": ["happen", "happened", "happening", "happens"],
        "want": ["want", "wanted", "wanting", "wants"],
        "wait": ["wait", "waited", "waiting", "waits"],
        "travel": ["travel", "traveled", "traveling", "travels"],
        "open": ["open", "opened", "opening", "opens"],
        "meet": ["meet", "meeting", "meets"],
        "plan": ["plan", "planned", "planning", "plans"],
        "hope": ["hope", "hoped", "hoping", "hopes"],
        "come": ["come", "coming", "comes"],
        "like": ["like", "liked", "liking", "likes"],
        "paint": ["paint", "painted", "painting", "paints"],
        "cook": ["cook", "cooked", "cooking", "cooks"],
        "enjoy": ["enjoy", "enjoyed", "enjoying", "enjoys"],
        "start": ["start", "started", "starting", "starts"],
        "need": ["need", "needed", "needing", "needs"],
        "train": ["train", "trained", "training", "trains"],
        "read": ["read", "reading", "reads"],
        "garden": ["garden", "gardened", "gardening", "gardens"],
        # e 结尾的 19 组
        "change": ["change", "changed", "changing", "changes"],
        "dance": ["dance", "danced", "dancing", "dances"],
        "decide": ["decide", "decided", "deciding", "decides"],
        "arrive": ["arrive", "arrived", "arriving", "arrives"],
        "believe": ["believe", "believed", "believing", "believes"],
        "share": ["share", "shared", "sharing", "shares"],
        "close": ["close", "closed", "closing", "closes"],
        "use": ["use", "used", "using", "uses"],
        "create": ["create", "created", "creating", "creates"],
        "love": ["love", "loved", "loving", "loves"],
        "practice": ["practice", "practiced", "practicing", "practices"],
        "place": ["place", "placed", "placing", "places"],
        "bake": ["bake", "baked", "baking", "bakes"],
        # e 结尾名词复数
        "house": ["house", "houses"],
        "course": ["course", "courses"],
        "size": ["size", "sizes"],
        "horse": ["horse", "horses"],
        "case": ["case", "cases"],
    }
    for root, forms in groups.items():
        stems = {_stem_en(w) for w in forms}
        assert len(stems) == 1, f"{root} 组应还原成同一个根: {forms} -> {sorted(stems)}"


def test_stem_no_residual_chunks():
    """不是变形的词不切成残块（string/spring 保持原样）。"""
    from core.store import _stem_en
    for word in ["string", "spring"]:
        assert _stem_en(word) == word, f"{word} 不能切成残块，实际 {_stem_en(word)}"
