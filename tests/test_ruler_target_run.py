"""新尺子的被测找法（进程内的服务）、证据诊断、一次完整的跑法。合成题库，不花钱，不调模型。"""
import json
import os

import pytest

from eval.ruler import corpus, diagnostics, items, run, shapes, target
from eval.ruler.testing import mini_questions, write_mini_corpus


@pytest.fixture()
def mini(tmp_path):
    root = write_mini_corpus(str(tmp_path / "copy"))
    return root, corpus.load_corpus(root, require_pin=False)


# ---- 被测找法：进程内的比赛服务，走真的 /add /search 合同 ----

def test_service_target_adds_and_searches_through_the_real_contract(tmp_path, mini):
    _, c = mini
    t = target.ServiceTarget(str(tmp_path / "db1.sqlite"), neighbours=False)
    try:
        conv = c.conversations[0]
        for b in corpus.chunk_conversation(conv):
            status = t.add(b.user_id, b.session_id, b.request_id, [shapes.render("b", m, s) for m, s in b.pairs()])
            assert status == 200
        hits = t.search(corpus.chunk_conversation(conv)[0].user_id, "puppy named Biscuit", 10)
        assert hits and "Biscuit" in hits[0].content
        assert hits[0].content.startswith("[2023-05-08 13:56] user: ")      # 带时间时服务这样渲染
    finally:
        t.close()


def test_a_time_text_in_timestamp_is_taken_without_a_time_by_the_service(tmp_path):
    """旧尺子走 HTTP 时的写法（题库的时间文字塞进毫秒字段）：以前整块被拒（422），2026-10-29 起服务照收、当没有时间——
    时间就丢了，所以形状 b 仍要换算成毫秒。"""
    t = target.ServiceTarget(str(tmp_path / "db2.sqlite"), neighbours=False)
    try:
        status = t.add("u", "s", "r1", [{"role": "user", "content": "hello Biscuit", "timestamp": "1:56 pm on 8 May, 2023"}])
        assert status == 200
        hits = t.search("u", "Biscuit", 10)
        assert [h.content for h in hits] == ["[user] hello Biscuit"], "时间文字读不成毫秒：当没有时间"
    finally:
        t.close()


def test_neighbours_switch_changes_what_the_service_hands_back(tmp_path, mini):
    _, c = mini
    conv = c.conversations[0]
    blocks = corpus.chunk_conversation(conv)
    results = {}
    for flag in (False, True):
        t = target.ServiceTarget(str(tmp_path / ("n%d.sqlite" % flag)), neighbours=flag)
        try:
            for b in blocks:
                t.add(b.user_id, b.session_id, b.request_id, [shapes.render("a", m, s) for m, s in b.pairs()])
            results[flag] = [h.id for h in t.search(blocks[0].user_id, "puppy Biscuit", 10)]
        finally:
            t.close()
    assert not any("#nb" in i for i in results[False])
    assert any("#nb" in i for i in results[True])


def test_the_stored_text_dump_is_what_the_answer_model_would_read(tmp_path, mini):
    _, c = mini
    t = target.ServiceTarget(str(tmp_path / "db3.sqlite"), neighbours=False)
    try:
        conv = c.conversations[0]
        for b in corpus.chunk_conversation(conv):
            t.add(b.user_id, b.session_id, b.request_id, [shapes.render("c", m, s) for m, s in b.pairs()])
        user = corpus.chunk_conversation(conv)[0].user_id
        texts = t.stored_texts(user)
        assert len(texts) == 5
        assert all(x.startswith("[20") for x in texts)                     # 形状 c 带时间
        assert any("user: Alice: I adopted a puppy" in x for x in texts)   # 渲染 = [时间] role: 正文，正文里带人名
        # 跟 /search 吐出来的内容是同一个渲染（防两边漂开）
        hit = t.search(user, "adopted puppy Biscuit", 1)[0]
        assert hit.content in texts
    finally:
        t.close()


# ---- 证据诊断 ----

def test_evidence_map_resolves_plain_and_neighbour_ids():
    em = diagnostics.EvidenceMap()
    em.register("req#0", "D1:1")
    em.register("req#1", "D1:2")
    assert em.dia_of("req#0") == "D1:1"
    assert em.dia_of("req#0#nbreq#1") == "D1:2"        # 邻居的 id：命中 id + #nb + 邻居自己的 id → 认邻居自己
    assert em.dia_of("nope#9") is None


def test_diagnose_reports_how_many_gold_came_back_and_where():
    em = diagnostics.EvidenceMap()
    for i, d in enumerate(["D1:1", "D1:2", "D1:3", "D1:4"]):
        em.register("r#%d" % i, d)
    it = items.Item(item_id="q", input="x", expected_evidence=["D1:2", "D1:4", "D9:9"], category="1", group="g")
    hits = [target.Hit("r#0", "aa"), target.Hit("r#3", "bbb"), target.Hit("r#1", "c")]
    row = diagnostics.diagnose(it, hits, em)
    assert row["n_gold"] == 3 and row["found_n"] == 2
    assert row["ranks"] == {"D1:4": 2, "D1:2": 3}
    assert row["first_rank"] == 2 and row["hit"] is True
    assert row["returned"] == 3 and row["chars"] == 6
    assert "input" not in row and "question" not in row           # 输出行里没有题目文字


def test_diagnose_without_any_gold_hit_has_no_first_rank():
    em = diagnostics.EvidenceMap()
    em.register("r#0", "D1:1")
    it = items.Item(item_id="q", input="x", expected_evidence=["D2:2"])
    row = diagnostics.diagnose(it, [target.Hit("r#0", "a")], em)
    assert row["hit"] is False and row["first_rank"] is None and row["found_n"] == 0


def test_summarise_matches_the_old_rulers_definitions():
    rows = [
        {"category": "1", "hit": True, "first_rank": 1, "found_n": 1, "n_gold": 1, "chars": 10},
        {"category": "1", "hit": True, "first_rank": 4, "found_n": 1, "n_gold": 2, "chars": 20},
        {"category": "2", "hit": False, "first_rank": None, "found_n": 0, "n_gold": 1, "chars": 30},
        {"category": "2", "hit": True, "first_rank": 50, "found_n": 1, "n_gold": 1, "chars": 40},
    ]
    s = diagnostics.summarise(rows)
    assert s["questions"] == 4 and s["hit_rate"] == 0.75
    assert s["recall@1"] == 0.25 and s["recall@5"] == 0.5 and s["recall@10"] == 0.5 and s["recall@100"] == 0.75
    assert abs(s["mrr"] - (1 + 0.25 + 0.02) / 3) < 1e-9
    assert abs(s["mrr_all"] - (1 + 0.25 + 0.02) / 4) < 1e-9
    assert s["chars_total"] == 100
    assert abs(s["gold_found_fraction_mean"] - (1 + 0.5 + 0 + 1) / 4) < 1e-9
    assert s["by_category"]["1"]["hit_rate"] == 1.0 and s["by_category"]["2"]["hit_rate"] == 0.5


# ---- 题的形状不锁死成问答 ----

def test_an_item_without_a_question_or_an_answer_fits_and_goes_through_the_diagnosis():
    """另一个使用方「自动浮现」的题：没有问题、没有标准答案，只有「该冒出来的那几条记忆」。"""
    it = items.Item(item_id="surface-1", kind="surface", input=None, expected_answers=None,
                    expected_evidence=["m1"], scope=3)
    row = items.to_row(it)
    assert row["kind"] == "surface" and row["input"] is None and row["expected_answers"] is None
    em = diagnostics.EvidenceMap()
    em.register("x#0", "m1")
    d = diagnostics.diagnose(it, [target.Hit("x#0", "text")], em)
    assert d["hit"] is True and d["first_rank"] == 1


def test_the_answerer_and_the_judge_are_interfaces_only_and_refuse_to_run():
    from eval.ruler import answerer, judge
    with pytest.raises(NotImplementedError):
        answerer.NotConnectedAnswerer().answer(items.Item(item_id="q", input="x"), [])
    with pytest.raises(NotImplementedError):
        judge.NotConnectedJudge().judge(items.Item(item_id="q", input="x"), "an answer")
    assert {v.value for v in judge.Verdict} == {"correct", "not_found", "made_up"}      # 对 / 说没找到 / 编了


# ---- 一次完整的跑法 ----

def test_a_run_writes_numbers_and_ids_only(tmp_path, mini):
    root, _ = mini
    out = str(tmp_path / "out")
    s = run.run(root, out, shape="b", neighbours=0, require_pin=False)
    assert s["questions"] == 5 and 0.0 <= s["hit_rate"] <= 1.0
    assert s["shape"] == "b" and s["neighbours"] == 0 and s["pinned"] is False
    files = sorted(os.listdir(out))
    assert any(f.startswith("rows_") for f in files) and any(f.startswith("summary_") for f in files)
    blob = "".join(open(os.path.join(out, f), encoding="utf-8").read() for f in files)
    for q in mini_questions():                     # 题目和对话文字一个字都不在输出里
        assert q["question"] not in blob
    assert "Biscuit" not in blob and "lighthouse" not in blob
    rows = [json.loads(l) for l in open(os.path.join(out, "rows_shape_b_neighbours_0.jsonl"), encoding="utf-8")]
    assert {r["item_id"] for r in rows} == {"q1", "q2", "q3", "q4", "q5"}


@pytest.mark.parametrize("shape", ["a", "b", "c", "d", "legacy"])
def test_every_shape_runs_through_the_service(tmp_path, mini, shape):
    root, _ = mini
    s = run.run(root, str(tmp_path / ("o-" + shape)), shape=shape, neighbours=1, require_pin=False)
    assert s["questions"] == 5 and s["shape"] == shape


def test_the_shape_changes_what_is_stored(tmp_path, mini):
    root, c = mini
    stored = {}
    for shape in ("a", "b", "c", "d"):
        t = target.ServiceTarget(str(tmp_path / ("s-%s.sqlite" % shape)), neighbours=False)
        try:
            conv = c.conversations[0]
            for b in corpus.chunk_conversation(conv):
                t.add(b.user_id, b.session_id, b.request_id, [shapes.render(shape, m, s) for m, s in b.pairs()])
            stored[shape] = " ".join(t.stored_texts(corpus.chunk_conversation(conv)[0].user_id))
        finally:
            t.close()
    assert "[2023-05-08" not in stored["a"] and "[2023-05-08" in stored["b"]
    assert "Alice:" not in stored["b"] and "Alice:" in stored["c"]
    assert "tulip" not in stored["c"] and "tulip" in stored["d"]


def test_limit_asks_only_the_first_questions(tmp_path, mini):
    root, _ = mini
    s = run.run(root, str(tmp_path / "lim"), shape="a", neighbours=0, limit=2, require_pin=False)
    assert s["questions"] == 2


# 旧尺子（已删）在合成小题库上跑出的数字，原样钉在这里。真题库上的对照（1380 题）写在 eval/ruler/README.md。
OLD_RULER_ON_MINI = {"hit_rate": 0.6, "recall@1": 0.6, "recall@5": 0.6, "recall@10": 0.6, "recall@100": 0.6,
                     "mrr": 1.0, "mrr_all": 0.6, "questions": 5}


@pytest.mark.parametrize("neighbours", [0, 1])
def test_the_core_engine_reproduces_the_old_ruler_exactly(tmp_path, mini, neighbours):
    """复现用：旧尺子的写法（核心直写，带块内 turn_index）跑出的数，跟旧尺子留下的钉死值一字不差。
    CoreTarget 只为这条复现留着；量平台真碰到的东西，一律走 ServiceTarget。"""
    root, _ = mini
    new = run.run(root, str(tmp_path / "new"), shape="legacy", neighbours=neighbours, require_pin=False, engine="core")
    for k, v in OLD_RULER_ON_MINI.items():
        assert new[k] == v, k
