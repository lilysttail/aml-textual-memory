"""核心的邻居、排序、快照换代、并发发号、终端行自测。

对应方案 6.1 的 T3 / T4 / T8 / T12 / T13 / T14。
"""
from concurrent.futures import ThreadPoolExecutor

from core.tests.conftest import rec


def test_T3_neighbours_come_from_the_same_session(core):
    core.write([
        rec("r0", "第一句", turn_index=0, ts="2026-10-21T05:00:00.000Z"),
        rec("r1", "第二句", turn_index=1, ts="2026-10-21T05:01:00.000Z"),
        rec("r2", "第三句", turn_index=2, ts="2026-10-21T05:02:00.000Z"),
    ], space_id="home")
    hit = [h for h in core.search("第二句", space_id="home", top_k=5)][0]
    assert [r.text for r in hit.neighbors_before] == ["第一句"]
    assert [r.text for r in hit.neighbors_after] == ["第三句"]


def test_T4_hits_of_one_session_come_back_in_time_order(core):
    core.write([
        rec("r0", "同一扇窗里先说的话 alpha", session_id="s1", turn_index=0,
            ts="2026-10-21T05:00:00.000Z"),
        rec("r1", "同一扇窗里后说的话 alpha", session_id="s1", turn_index=1,
            ts="2026-10-21T05:01:00.000Z"),
    ], space_id="home")
    hits = core.search("alpha", space_id="home", top_k=5)
    assert [h.record_id for h in hits] == ["r0", "r1"]
    assert [h.ts for h in hits] == sorted(h.ts for h in hits)


def test_T8_an_invalid_row_is_neither_found_nor_a_neighbour(core):
    core.write([
        rec("r0", "有效的一句", turn_index=0),
        rec("r1", "被改判的一句", turn_index=1, valid=False),
        rec("r2", "有效的另一句", turn_index=2),
    ], space_id="home")
    found = [h.record_id for h in core.search("被改判的一句", space_id="home", top_k=5)]
    assert found == []
    hit = [h for h in core.search("有效的一句", space_id="home", top_k=5)][0]
    assert [r.record_id for r in hit.neighbors_after] == ["r2"]


def test_T12_snapshot_replacement_takes_the_old_rows_away(core):
    first = [
        rec("r0", "收进来的一句话 alpha", session_id="s1", turn_index=0),
        rec("r1", "中间那一句 alpha", session_id="s1", turn_index=1),
        rec("r2", "最后一句 alpha", session_id="s1", turn_index=2),
    ]
    core.replace_space_snapshot("home", first, snapshot_id="v1")
    assert [h.record_id for h in core.search("alpha", space_id="home", top_k=5)] == ["r0", "r1", "r2"]

    # 判删中间那条 -> 它和它的邻居都换人
    second = [r for r in first if r.record_id != "r1"]
    core.replace_space_snapshot("home", second, snapshot_id="v2")
    found = [h.record_id for h in core.search("alpha", space_id="home", top_k=5)]
    assert "r1" not in found
    hit = [h for h in core.search("收进来的一句话 alpha", space_id="home", top_k=5)][0]
    assert [r.record_id for r in hit.neighbors_after] == ["r2"]

    # 收回 -> 又能搜到；重跑同一份快照不重复
    core.replace_space_snapshot("home", first, snapshot_id="v3")
    assert len(core.search("alpha", space_id="home", top_k=5)) == 3
    core.replace_space_snapshot("home", first, snapshot_id="v3")
    assert core.store.count("home") == 3


def test_T12b_a_row_inserted_in_the_middle_still_finds_its_neighbours(core):
    base = [
        rec("r0", "开头一句", session_id="s1", turn_index=0),
        rec("r2", "结尾一句", session_id="s1", turn_index=2),
    ]
    core.replace_space_snapshot("home", base, snapshot_id="v1")
    # 中间补进一条旧消息（窗口顺序变了）
    grown = base + [rec("r1", "补进来的一句", session_id="s1", turn_index=1)]
    core.replace_space_snapshot("home", grown, snapshot_id="v2")
    hit = [h for h in core.search("补进来的一句", space_id="home", top_k=5)][0]
    assert [r.record_id for r in hit.neighbors_before] == ["r0"]
    assert [r.record_id for r in hit.neighbors_after] == ["r2"]


def test_T13_terminal_lines_stay_out_of_the_vector_route(dense_core):
    """终端行只进关键词索引：按字还能搜到，但不会被向量化。"""
    dense_core.write([
        rec("r1", "终端里的一行 line", line="terminal"),
        rec("r2", "网页上的一行 line", line="web"),
    ], space_id="home")
    hits = dense_core.search("line", space_id="home", top_k=5)
    assert [h.record_id for h in hits] == ["r1", "r2"]      # 按字找得到
    spy = dense_core.config.embedder
    assert "网页上的一行 line" in spy.embedded_texts      # 进了向量路
    assert "终端里的一行 line" not in spy.embedded_texts  # 没进向量路


def test_T14b_an_explicit_turn_index_moves_the_cursor_forward(core):
    """A snapshot that says turn 5 must not leave the next auto turn at 0.

    Otherwise the next increment write would be handed 0 again and the
    neighbours of a session would collide (栖 P1-2).
    """
    core.replace_space_snapshot("home", [
        rec("r0", "快照里的一条", session_id="s1", turn_index=5),
    ], snapshot_id="v1")
    core.write([rec("r1", "快照之后写的一条", session_id="s1")], space_id="home")
    row = core.store.get_rows("home", ["r1"])["r1"]
    assert row["turn_index"] == 6, row


def test_T14_turn_indexes_are_handed_out_without_collisions(core):
    def one(i):
        core.write([rec(f"r{i}", f"第 {i} 句")], space_id="home", batch_id=f"b{i}")
        return core.store.get_rows("home", [f"r{i}"])[f"r{i}"]["turn_index"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        turns = list(pool.map(one, range(20)))
    assert len(set(turns)) == 20
    assert sorted(turns) == list(range(20))
