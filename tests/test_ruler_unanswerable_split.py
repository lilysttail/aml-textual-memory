"""「根本答不出来」的三条定义（U1 点名而库里没有这个名字 / U2 问时间而库里没有一条带时间 / U3 答案只在图片说明里）
和「按对话切调 / 考」的算法。合成题库，不碰真题库。"""
import pytest

from eval.ruler import corpus, split, unanswerable
from eval.ruler.testing import write_mini_corpus


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def ids(result, shape):
    """某形状下被判「根本答不出来」的题号 → 原因集合。"""
    return {q: set(r) for q, r in result[shape]["questions"].items() if r}


def test_u1_a_named_speaker_that_is_nowhere_in_the_stored_texts(mini):
    res = unanswerable.classify(mini, stored="service")
    # q1 点了 Alice，会话 1 里没有任何一条正文提到 Alice：形状 a、b 答不出来（库里只有 [user] / [assistant] 标签）
    for shape in ("a", "b"):
        assert ids(res, shape)["q1"] >= {"U1"}
    # 形状 c、d 每条正文前都带说话人名字，U1 恒为 0
    for shape in ("c", "d"):
        assert "U1" not in {r for rs in ids(res, shape).values() for r in rs}


def test_u1_needs_every_named_speaker_to_be_missing_and_a_mention_in_any_text_saves_it(mini):
    # 会话 2 的 q4 点了 Cara；会话 2 的正文里没有 Cara（"We visited..." 是 Cara 说的，但名字不在正文里）
    res = unanswerable.classify(mini, stored="service")
    assert "U1" in ids(res, "a")["q4"]
    # 把一条正文改成提到 Cara：不再算
    conv = mini.conversations[1]
    conv.sessions[0].messages[1] = type(conv.sessions[0].messages[1])(
        **{**conv.sessions[0].messages[1].__dict__, "text": "Nice Cara, I love lighthouses."})
    res2 = unanswerable.classify(mini, stored="service")
    assert "q4" not in ids(res2, "a") or "U1" not in ids(res2, "a")["q4"]


def test_u2_a_time_question_when_no_stored_record_carries_a_time(mini):
    res = unanswerable.classify(mini, stored="service")
    assert "U2" in ids(res, "a")["q2"]                 # 形状 a 没有时间
    for shape in ("b", "c", "d"):
        assert "q2" not in ids(res, shape) or "U2" not in ids(res, shape)["q2"]
    # 时间题 = 第 2 类：别的题不会落在 U2 里
    assert all("U2" not in r for q, r in ids(res, "a").items() if q != "q2")


def test_u3_answer_only_in_the_caption_is_lost_unless_the_caption_is_stored(mini):
    res = unanswerable.classify(mini, stored="service")
    for shape in ("a", "b", "c"):
        assert "U3" in ids(res, shape)["q3"]
    assert "q3" not in ids(res, "d") or "U3" not in ids(res, "d")["q3"]


def test_u3_wording_at_least_one_candidate_in_the_evidence_caption_and_no_candidate_in_any_text(mini):
    """答案是几个都算对的候选：只要有一个候选出现在任何一条正文里，就不算「只在说明里」。"""
    qs = {q["qa_id"]: q for q in mini.questions}
    # q3 的候选是 ["red tulip", "tulips"]：说明里有 "red tulip"，正文里一个候选都没有 → U3
    assert unanswerable.answer_only_in_caption(mini, qs["q3"]) is True
    # 把候选换成「一个在说明里、一个在别的正文里」→ 不算
    q = dict(qs["q3"], answer=["red tulip", "puppy"])
    assert unanswerable.answer_only_in_caption(mini, q) is False
    # 说明里一个候选都没有 → 不算
    assert unanswerable.answer_only_in_caption(mini, dict(qs["q3"], answer=["blue whale"])) is False


def test_counts_per_shape_and_category_and_the_union(mini):
    res = unanswerable.classify(mini, stored="service")
    table = unanswerable.table(mini, res)
    assert set(table) == {"a", "b", "c", "d"}
    assert table["a"]["total"]["union"] >= table["a"]["total"]["U1"]
    assert table["d"]["total"]["U3"] == 0 and table["c"]["total"]["U1"] == 0
    # 每个类别里的并集 ≤ 该类题数
    per_cat = unanswerable.questions_per_category(mini)
    for shape in table:
        for cat, row in table[shape]["by_category"].items():
            assert row["union"] <= per_cat[cat]


def test_the_time_wording_table_by_category(mini):
    t = unanswerable.time_wording_by_category(mini)
    assert t["2"]["with_time_wording"] == 1 and t["2"]["total"] == 1       # q2 "When did ..."
    assert sum(v["with_time_wording"] for k, v in t.items() if k != "2") == 0


def test_u2_is_counted_by_category_2_and_the_wording_variant_is_only_a_sensitivity_list(mini):
    """U2 的定义按第 2 类数。别的类里带时间字样的题另列一张对照：形状里没有时间时，按字样数最多再多这几道答不出。"""
    other = next(q for q in mini.questions if str(q["category"]) == "4")
    other["question"] = "When did Alice visit the lake?"
    s = unanswerable.u2_wording_sensitivity(mini)
    assert s["category_2_questions"] == 1 and s["category_2_with_wording"] == 1 and s["category_2_without_wording"] == 0
    assert s["other_category_with_wording_ids"] == [str(other["qa_id"])] and s["other_category_with_wording"] == 1
    # 形状 a 没有时间：按字样数会多 1 道；形状 b/c/d 带时间，不多
    assert s["extra_if_counted_by_wording"] == {"a": 1, "b": 0, "c": 0, "d": 0}
    # 定义本身不变：U2 只落在第 2 类上
    res = unanswerable.classify(mini, stored="rule")
    assert str(other["qa_id"]) not in {q for q, r in res["a"]["questions"].items() if "U2" in r}


def test_the_sensitivity_list_holds_ids_only(mini):
    s = unanswerable.u2_wording_sensitivity(mini)
    assert all(isinstance(i, str) for i in s["other_category_with_wording_ids"])
    for v in s.values():
        assert isinstance(v, (int, list, dict)) and not isinstance(v, str)


def test_the_rule_based_stored_text_agrees_with_the_real_service(mini):
    """不起服务也能算（规则渲染）的口径，必须跟真服务存着的文字一致——不一致就是漂了。"""
    a = unanswerable.classify(mini, stored="service")
    b = unanswerable.classify(mini, stored="rule")
    assert a == b


# ---- 按对话切调 / 考 ----

def test_per_conversation_counts(mini):
    counts = split.per_conversation_counts(mini)
    assert counts["mini-1"] == {"total": 3, "1": 1, "2": 1, "3": 0, "4": 1}
    assert counts["mini-2"] == {"total": 2, "1": 0, "2": 0, "3": 1, "4": 1}


def test_the_split_is_by_conversation_and_deterministic():
    counts = {"c%d" % i: {"total": n, "1": n // 5, "2": n // 4, "3": n // 20, "4": n - n // 5 - n // 4 - n // 20}
              for i, n in enumerate([100, 120, 140, 160, 180, 200, 220, 240, 260, 280])}
    a = split.choose_split(counts, lock_size=4)
    b = split.choose_split(counts, lock_size=4)
    assert a == b
    assert len(a["lock"]) == 4 and len(a["tune"]) == 6
    assert set(a["lock"]).isdisjoint(a["tune"]) and set(a["lock"]) | set(a["tune"]) == set(counts)
    assert a["lock_questions"] == sum(counts[c]["total"] for c in a["lock"])
    # 锁着那份的类别比例尽量跟整体一致
    assert a["category_share_gap"] < 0.1


def test_minimum_detectable_difference_known_values():
    # 两个版本在同一批 n 题上独立比较、正确率约 0.5，80% 功效、双侧 5%：约 1.96+0.84 = 2.80 个标准误
    d = split.min_detectable_difference_unpaired(n=400, p=0.5)
    assert abs(d - 2.8016 * (2 * 0.25 / 400) ** 0.5) < 5e-4
    # 配对：两版只在 d_frac 比例的题上结论不同，可测的差 ≈ 2.8 * sqrt(d_frac / n)
    dp = split.min_detectable_difference_paired(n=400, discordant_fraction=0.1)
    assert abs(dp - 2.8016 * (0.1 / 400) ** 0.5) < 5e-4
    assert dp < d


def test_design_effect_grows_when_questions_of_one_conversation_move_together():
    flat = [{"group": g, "hit": i % 2 == 0} for g in "ab" for i in range(50)]            # 各组都是一半一半
    lumpy = [{"group": "a", "hit": True}] * 50 + [{"group": "b", "hit": False}] * 50       # 一组全中一组全不中
    assert split.design_effect(flat) < 1.2
    assert split.design_effect(lumpy) > 10


def test_naming_gap_counts_questions_whose_evidence_text_never_says_the_name(mini):
    g = unanswerable.naming_gap(mini)
    # q1 (cat 4, names Alice) and q4 (cat 4, names Cara): their evidence texts lack the name; q2 (cat 2, names Alice) too
    assert g["4"]["naming_own_speaker"] == 2 and g["4"]["none_of_named_in_evidence_text"] == 2
    assert g["2"]["naming_own_speaker"] == 1 and g["2"]["none_of_named_in_evidence_text"] == 1
    assert g["3"]["naming_own_speaker"] == 0 and g["1"]["naming_own_speaker"] == 0


def test_u1_any_variant_is_never_smaller_than_the_all_variant(mini):
    res = unanswerable.classify(mini, stored="rule")
    all_variant = sum(1 for r in res["a"]["questions"].values() if "U1" in r)
    assert unanswerable.u1_any_missing(mini, "a") >= all_variant
