"""给尺子留的口子：候选池（每路自己的分数）+ 假嵌入器。

单 3 的尺子要按「证据轮次」算召回，得能看到候选池和每条命中在第几位；
走 HTTP 拿不到这些，走核心函数能拿到。
"""
from core.testing import FakeEmbedder
from core.tests.conftest import rec


def test_recall_candidates_give_the_pool_and_each_path_score(core):
    core.write([
        rec("r1", "第一段 alpha 内容"),
        rec("r2", "第二段 beta 内容"),
    ], space_id="home")
    cands = core.recall_candidates("alpha", space_id="home", limit=10)
    assert [c.record_id for c in cands] == ["r1"]
    assert cands[0].path_scores == {"keyword": cands[0].score}
    assert cands[0].score > 0


def test_recall_candidates_do_not_cut_or_neighbour(dense_core):
    """候选池不受 top_k 影响，也不带邻居——尺子要自己算。"""
    dense_core.write([
        rec("r0", "第一段 alpha", turn_index=0),
        rec("r1", "第二段 alpha", turn_index=1),
    ], space_id="home")
    cands = dense_core.recall_candidates("alpha", space_id="home", limit=50)
    assert {c.record_id for c in cands} == {"r0", "r1"}
    assert all(not hasattr(c, "neighbors_before") for c in cands)


def test_the_fake_embedder_is_deterministic():
    a = FakeEmbedder(dim=16)
    b = FakeEmbedder(dim=16)
    assert a.embed(["一句同样的话"])[0] == b.embed(["一句同样的话"])[0]
    assert a.model_name == "fake:16"
    assert len(a.embed(["x"])[0]) == 16
