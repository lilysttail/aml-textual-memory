"""The three facts and the command line of the new ruler (synthetic copy)."""
import json

from eval.ruler import __main__ as cli
from eval.ruler import corpus, facts
from eval.ruler.testing import mini_conversations, write_mini_corpus


def test_facts_on_the_mini_copy(tmp_path):
    c = corpus.load_corpus(write_mini_corpus(str(tmp_path)), require_pin=False)
    f = facts.compute(c, mini_conversations())
    assert f["messages_total"] == 7 and f["sessions_total"] == 3
    assert f["time_on_message_own_key_date_time"] == 0                     # 时间不在消息上
    assert f["messages_whose_copy_equals_their_session_time"] == 7
    assert f["sessions_where_all_messages_share_one_time"] == 3
    assert f["session_times_matching_the_pattern"] == 3
    assert f["messages_with_nonempty_caption"] == 2
    assert f["questions_marked_multimodal"] == 1
    assert f["questions_naming_a_speaker_of_their_own_conversation"] == 3
    assert f["questions_naming_no_known_speaker"] == 2
    assert f["roles"] == {"user": 4, "assistant": 3}


def test_cli_prints_numbers_only(tmp_path, capsys):
    root = write_mini_corpus(str(tmp_path / "copy"))
    assert cli.main(["facts", "--data", root, "--no-pin"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["questions_total"] == 5
    assert "Biscuit" not in out and "lighthouse" not in out
    assert cli.main(["unanswerable", "--data", root, "--no-pin", "--stored", "rule"]) == 0
    assert "U1" in capsys.readouterr().out
    assert cli.main(["split", "--data", root, "--no-pin", "--sizes", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["options"][0]["lock"]
    assert cli.main(["run", "--data", root, "--no-pin", "--out", str(tmp_path / "o"), "--shape", "c", "--neighbours", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["shape"] == "c"


def test_cli_refuses_a_copy_that_is_not_the_pinned_one(tmp_path):
    import pytest
    root = write_mini_corpus(str(tmp_path / "copy"))
    with pytest.raises(corpus.PinMismatch):
        cli.main(["facts", "--data", root])


def test_cli_run_defaults_to_the_tuning_part_and_the_locked_part_needs_its_ledger(tmp_path, capsys):
    import pytest
    root = write_mini_corpus(str(tmp_path / "copy"))
    assert cli.main(["run", "--data", root, "--no-pin", "--out", str(tmp_path / "o1"), "--shape", "a"]) == 0
    assert json.loads(capsys.readouterr().out)["partition"] == "tune"
    with pytest.raises(SystemExit):                                    # 锁定那份不给版本 / 声明 / 记录文件：不跑
        cli.main(["run", "--data", root, "--no-pin", "--out", str(tmp_path / "o2"), "--shape", "a", "--partition", "locked"])
    assert not (tmp_path / "o2").exists()


def test_cli_split_prints_the_fixed_cut_and_the_gate(tmp_path, capsys):
    root = write_mini_corpus(str(tmp_path / "copy"))
    assert cli.main(["split", "--data", root, "--no-pin", "--sizes", "1"]) == 0
    fixed = json.loads(capsys.readouterr().out)["fixed"]
    assert fixed["lock"] == ["conv-26", "conv-41", "conv-42", "conv-47", "conv-50"] and fixed["lock_questions"] == 734
    assert fixed["dense_gate"] == {"min_overall_gain_points": 4, "min_segments_better": 3}
