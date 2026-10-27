"""切法写死（2026-10-23 定，选「丙」）：锁 5 段、调 5 段；锁着那份只用一次；「开不开按意思找」的门槛写死在代码里。
不碰真题库（用 eval/ruler/testing.py 的合成题库）。"""
import json
import os
import subprocess
import sys
import time

import pytest

from eval.ruler import corpus, run, split
from eval.ruler.testing import write_mini_corpus


# ---- 写死的切法 ----

def test_the_fixed_partition_is_five_locked_and_five_tuned_conversations():
    assert split.LOCK == ("conv-26", "conv-41", "conv-42", "conv-47", "conv-50")
    assert len(split.TUNE) == 5 and set(split.TUNE).isdisjoint(split.LOCK)
    assert set(split.LOCK) | set(split.TUNE) == {"conv-%d" % i for i in (26, 30, 41, 42, 43, 44, 47, 48, 49, 50)}


def test_the_partition_check_refuses_a_corpus_whose_counts_moved():
    counts = {c: {"total": 10} for c in split.LOCK + split.TUNE}
    split.check_partition(counts, lock_questions=50, tune_questions=50)           # 数对了就放行
    with pytest.raises(ValueError):
        split.check_partition(counts, lock_questions=734, tune_questions=648)     # 数不对：不是锁定时看到的那份题库
    with pytest.raises(ValueError):
        split.check_partition({"conv-26": {"total": 1}}, lock_questions=1, tune_questions=0)   # 少了对话


def test_the_real_corpus_numbers_are_pinned_in_the_module():
    assert (split.LOCK_QUESTIONS, split.TUNE_QUESTIONS) == (734, 648)


# ---- 开不开按意思找：写死的门槛 ----

def _rows(version, plan):
    """plan = {group: (n_questions, n_correct)} -> 逐题行 [{item_id, group, correct}]"""
    out = []
    for g, (n, k) in plan.items():
        out += [{"item_id": "%s#%03d" % (g, i), "group": g, "correct": i < k, "version": version} for i in range(n)]
    return out


LOCK3 = ("a", "b", "c", "d", "e")


def test_the_gate_is_written_down_as_numbers():
    assert split.DENSE_MIN_OVERALL_GAIN_POINTS == 4
    assert split.DENSE_MIN_SEGMENTS_BETTER == 3


def test_dense_goes_on_when_overall_gain_reaches_4_points_and_three_segments_are_better():
    base = _rows("base", {g: (100, 50) for g in LOCK3})
    dense = _rows("dense", {"a": (100, 56), "b": (100, 56), "c": (100, 56), "d": (100, 50), "e": (100, 50)})   # 总 +3.6，不够
    assert split.decide_dense(base, dense, lock=LOCK3)["dense_on"] is False
    dense = _rows("dense", {"a": (100, 58), "b": (100, 58), "c": (100, 50), "d": (100, 50), "e": (100, 50)})  # 总 +3.2
    assert split.decide_dense(base, dense, lock=LOCK3)["dense_on"] is False
    dense = _rows("dense", {"a": (100, 60), "b": (100, 55), "c": (100, 55), "d": (100, 50), "e": (100, 50)})  # 总 +4.0 且 3 段更好
    r = split.decide_dense(base, dense, lock=LOCK3)
    assert r["dense_on"] is True and r["segments_better"] == 3 and r["overall_gain_points"] == 4.0


def test_a_big_overall_gain_from_two_segments_alone_does_not_turn_it_on():
    base = _rows("base", {g: (100, 50) for g in LOCK3})
    dense = _rows("dense", {"a": (100, 80), "b": (100, 80), "c": (100, 50), "d": (100, 50), "e": (100, 50)})   # 总 +12，但只有 2 段更好
    r = split.decide_dense(base, dense, lock=LOCK3)
    assert r["overall_gain_points"] == 12.0 and r["segments_better"] == 2 and r["dense_on"] is False


def test_three_better_segments_do_not_turn_it_on_if_the_overall_gain_is_under_4_points():
    base = _rows("base", {g: (100, 50) for g in LOCK3})
    dense = _rows("dense", {"a": (100, 51), "b": (100, 51), "c": (100, 51), "d": (100, 50), "e": (100, 50)})
    r = split.decide_dense(base, dense, lock=LOCK3)
    assert r["segments_better"] == 3 and r["dense_on"] is False


def test_the_comparison_is_per_question_and_pooled_over_questions_not_averaged_over_segments():
    # 两段各 10 题、三段各 100 题：按题合并多对 14 题 / 320 题 = 4.375 个点；各段差的平均是 20.8 个点，两者不同
    base = _rows("base", {"a": (10, 5), "b": (10, 5), "c": (100, 50), "d": (100, 50), "e": (100, 50)})
    dense = _rows("dense", {"a": (10, 10), "b": (10, 10), "c": (100, 52), "d": (100, 52), "e": (100, 50)})
    r = split.decide_dense(base, dense, lock=LOCK3)
    assert r["overall_gain_points"] == 4.375
    assert r["questions"] == 320
    assert r["segments_better"] == 4


def test_a_gain_of_exactly_4_points_counts_with_no_rounding_slack():
    # 734 题时 4 个点 = 29.36 题：多 29 题（3.95）不够，多 30 题（4.09）够
    plan_b = {"a": (200, 100), "b": (200, 100), "c": (200, 100), "d": (67, 33), "e": (67, 33)}   # 734 题
    assert sum(n for n, _ in plan_b.values()) == 734
    base = _rows("base", plan_b)
    d29 = _rows("dense", {"a": (200, 110), "b": (200, 110), "c": (200, 109), "d": (67, 33), "e": (67, 33)})    # +29
    d30 = _rows("dense", {"a": (200, 110), "b": (200, 110), "c": (200, 110), "d": (67, 33), "e": (67, 33)})    # +30
    assert split.decide_dense(base, d29, lock=LOCK3)["dense_on"] is False
    assert split.decide_dense(base, d30, lock=LOCK3)["dense_on"] is True


def test_the_decision_refuses_rows_that_are_not_exactly_the_locked_conversations():
    base = _rows("base", {g: (10, 5) for g in LOCK3})
    with pytest.raises(ValueError):                                                   # 夹带了调的那一份
        split.decide_dense(base + _rows("base", {"z": (10, 5)}), base + _rows("dense", {"z": (10, 6)}), lock=LOCK3)
    with pytest.raises(ValueError):                                                   # 少一段
        split.decide_dense(base[:-10], base[:-10], lock=LOCK3)
    with pytest.raises(ValueError):                                                   # 两个版本问的不是同一批题
        split.decide_dense(base, base[:-1] + [dict(base[-1], item_id="other")], lock=LOCK3)


def test_the_gate_uses_the_real_locked_conversations_by_default():
    base = _rows("base", {g: (10, 5) for g in split.LOCK})
    r = split.decide_dense(base, base)
    assert r["dense_on"] is False and r["segments_better"] == 0 and set(r["segments"]) == set(split.LOCK)


# ---- 锁着那份只用一次（账本是一个目录：每个认领一个文件，独占创建） ----

def test_the_locked_set_can_be_claimed_once_per_declared_version(tmp_path):
    ledger = str(tmp_path / "locked_use")
    declared = ["baseline", "hardened"]
    split.claim_locked(ledger, "hardened", declared)
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked(ledger, "hardened", declared)               # 同一版再量一次：不行
    split.claim_locked(ledger, "baseline", declared)                   # 另一版各量一次：行
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked(ledger, "baseline", declared)
    with pytest.raises(ValueError):
        split.claim_locked(ledger, "third", declared)                  # 第三个版本：不在声明的两个里，不行


def test_the_declared_pair_is_fixed_by_the_first_claim(tmp_path):
    ledger = str(tmp_path / "locked_use")
    split.claim_locked(ledger, "baseline", ["baseline", "hardened"])
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked(ledger, "hardened_v2", ["baseline", "hardened_v2"])   # 事后换对手：不行
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked(ledger, "baseline", ["hardened", "baseline"])         # 同一对换个顺序：认得出是同一对，但这一版用过了
    split.claim_locked(ledger, "hardened", ["hardened", "baseline"])             # 顺序不同的同一对：认作同一对


def test_a_claim_must_declare_exactly_two_distinct_versions_and_name_one_of_them(tmp_path):
    ledger = str(tmp_path / "locked_use")
    for bad in (["only"], ["x", "x"], ["a", "b", "c"], [], ["Base", "base"]):     # 最后一个：大小写不同在 Windows 上是同一个文件名
        with pytest.raises(ValueError):
            split.claim_locked(ledger, bad[0] if bad else "a", bad)
    with pytest.raises(ValueError):
        split.claim_locked(ledger, "c", ["a", "b"])
    assert not (tmp_path / "locked_use").exists()                               # 被拒的申请不留痕


@pytest.mark.parametrize("label", ["", "a/b", "..", "a b", "x" * 65, "a:b", "con.txt\\x"])
def test_a_version_label_that_is_not_a_plain_file_name_is_refused(tmp_path, label):
    with pytest.raises(ValueError):
        split.claim_locked(str(tmp_path / "locked_use"), label, [label, "other"])
    assert not (tmp_path / "locked_use").exists()


def test_an_unreadable_declaration_or_a_ledger_that_is_not_a_directory_is_a_refusal_not_a_fresh_start(tmp_path):
    ledger = tmp_path / "locked_use"
    ledger.mkdir()
    (ledger / "declared.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked(str(ledger), "baseline", ["baseline", "hardened"])
    assert not (ledger / "used.baseline").exists()                              # 拒了就什么都不写
    old_style = tmp_path / "old_ledger.json"
    old_style.write_text('{"declared": ["baseline", "hardened"], "used": []}', encoding="utf-8")
    with pytest.raises(split.LockedSetAlreadyUsed):
        split.claim_locked(str(old_style), "baseline", ["baseline", "hardened"])


def test_the_ledger_is_one_declaration_file_and_one_marker_per_version(tmp_path):
    ledger = tmp_path / "locked_use"
    split.claim_locked(str(ledger), "baseline", ["baseline", "hardened"])
    assert sorted(p.name for p in ledger.iterdir()) == ["declared.json", "used.baseline"]       # 没有留下临时文件
    assert json.loads((ledger / "declared.json").read_text(encoding="utf-8")) == {"declared": ["baseline", "hardened"]}
    split.claim_locked(str(ledger), "hardened", ["baseline", "hardened"])
    assert sorted(p.name for p in ledger.iterdir()) == ["declared.json", "used.baseline", "used.hardened"]


_CLAIM = ("import sys, time\n"
          "from eval.ruler import split\n"
          "ledger, version, a, b, t0 = sys.argv[1:6]\n"
          "while time.time() < float(t0):\n"
          "    pass\n"
          "try:\n"
          "    split.claim_locked(ledger, version, [a, b])\n"
          "    print('ok')\n"
          "except split.LockedSetAlreadyUsed:\n"
          "    print('used')\n")


def _race(ledger, claims, start_in=6.0):
    """Several separate processes try to claim at the same wall-clock moment. claims = [(version, a, b), ...]"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time() + start_in
    env = dict(os.environ, PYTHONPATH=root)
    procs = [subprocess.Popen([sys.executable, "-c", _CLAIM, str(ledger), v, a, b, repr(t0)], cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for v, a, b in claims]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        outs.append(out.strip())
    return outs


def test_many_processes_claiming_the_same_version_at_once_only_one_wins(tmp_path):
    outs = _race(tmp_path / "locked_use", [("baseline", "baseline", "hardened")] * 6)
    assert sorted(outs) == ["ok"] + ["used"] * 5


def test_two_different_versions_claiming_for_the_first_time_at_once_both_are_kept(tmp_path):
    ledger = tmp_path / "locked_use"
    outs = _race(ledger, [("baseline", "baseline", "hardened")] * 3 + [("hardened", "baseline", "hardened")] * 3)
    assert sorted(outs) == ["ok", "ok"] + ["used"] * 4
    assert sorted(p.name for p in ledger.iterdir()) == ["declared.json", "used.baseline", "used.hardened"]


def test_two_different_pairs_declared_at_once_only_one_pair_stands(tmp_path):
    ledger = tmp_path / "locked_use"
    outs = _race(ledger, [("baseline", "baseline", "hardened")] * 3 + [("baseline", "baseline", "other")] * 3)
    assert sorted(outs) == ["ok"] + ["used"] * 5
    declared = json.loads((ledger / "declared.json").read_text(encoding="utf-8"))["declared"]
    assert declared in (["baseline", "hardened"], ["baseline", "other"])


# ---- 跑的时候按段拿题 ----

@pytest.fixture()
def mini(tmp_path):
    return write_mini_corpus(str(tmp_path / "copy"))


def test_a_run_on_the_tune_part_never_touches_a_locked_conversation(tmp_path, mini):
    s = run.run(mini, str(tmp_path / "t"), shape="a", neighbours=0, require_pin=False, partition="tune", lock=("mini-1",))
    rows = [json.loads(l) for l in open(tmp_path / "t" / "rows_shape_a_neighbours_0.jsonl", encoding="utf-8")]
    assert rows and {r["group"] for r in rows} == {"mini-2"} and s["partition"] == "tune"


def test_a_run_on_the_locked_part_needs_a_version_and_the_declared_pair_and_a_ledger(tmp_path, mini):
    kw = dict(shape="a", neighbours=0, require_pin=False, partition="locked", lock=("mini-1",))
    with pytest.raises(ValueError):
        run.run(mini, str(tmp_path / "o1"), **kw)                                                  # 没说是哪一版
    ledger = str(tmp_path / "locked_use")
    s = run.run(mini, str(tmp_path / "o2"), version="baseline", declared=["baseline", "hardened"], ledger=ledger, **kw)
    rows = [json.loads(l) for l in open(tmp_path / "o2" / "rows_shape_a_neighbours_0.jsonl", encoding="utf-8")]
    assert {r["group"] for r in rows} == {"mini-1"} and s["partition"] == "locked" and s["version"] == "baseline"
    with pytest.raises(split.LockedSetAlreadyUsed):                                                # 同一版第二次
        run.run(mini, str(tmp_path / "o3"), version="baseline", declared=["baseline", "hardened"], ledger=ledger, **kw)


def test_a_refused_locked_run_does_not_write_anything(tmp_path, mini):
    ledger = str(tmp_path / "locked_use")
    kw = dict(shape="a", neighbours=0, require_pin=False, partition="locked", lock=("mini-1",),
              version="baseline", declared=["baseline", "hardened"], ledger=ledger)
    run.run(mini, str(tmp_path / "o2"), **kw)
    with pytest.raises(split.LockedSetAlreadyUsed):
        run.run(mini, str(tmp_path / "o3"), **kw)
    assert not (tmp_path / "o3").exists() or not any((tmp_path / "o3").iterdir())


def test_without_a_partition_the_run_is_the_whole_set_and_says_so(tmp_path, mini):
    s = run.run(mini, str(tmp_path / "all"), shape="a", neighbours=0, require_pin=False)
    assert s["partition"] == "all" and s["questions"] == 5


def test_a_locked_run_that_fails_half_way_still_counts_as_used(tmp_path, mini, monkeypatch):
    from eval.ruler import target

    def boom(self, *a, **k):
        raise RuntimeError("half way")

    monkeypatch.setattr(target.ServiceTarget, "search", boom)
    kw = dict(shape="a", neighbours=0, require_pin=False, partition="locked", lock=("mini-1",), version="baseline",
              declared=["baseline", "hardened"], ledger=str(tmp_path / "locked_use"))
    with pytest.raises(RuntimeError):
        run.run(mini, str(tmp_path / "o1"), **kw)
    monkeypatch.undo()
    with pytest.raises(split.LockedSetAlreadyUsed):
        run.run(mini, str(tmp_path / "o2"), **kw)
