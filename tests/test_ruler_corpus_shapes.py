"""新尺子（eval/ruler）的题库读取、切块、Add 四种形状。不碰真题库（用 eval/ruler/testing.py 的合成题库）。"""
import datetime as dt
import importlib

import pytest

from eval.ruler import corpus, shapes
from eval.ruler.testing import mini_conversations, write_mini_corpus


def _utc_ms(*a):
    return int(dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_session_time_text_becomes_epoch_milliseconds_in_utc():
    assert corpus.session_time_ms("1:56 pm on 8 May, 2023") == _utc_ms(2023, 5, 8, 13, 56)
    assert corpus.session_time_ms("12:05 am on 1 January, 2024") == _utc_ms(2024, 1, 1, 0, 5)
    assert corpus.session_time_ms("12:30 pm on 15 March, 2022") == _utc_ms(2022, 3, 15, 12, 30)


@pytest.mark.parametrize("bad", ["", "yesterday", "2023-05-08", "13:00 pm on 8 May, 2023", "1:56 pm on 8 Mayy, 2023"])
def test_unreadable_session_time_is_an_error_not_a_silent_none(bad):
    with pytest.raises(ValueError):
        corpus.session_time_ms(bad)


def test_the_pinned_constants_live_in_one_place_and_the_load_test_reads_them_from_there():
    """题库的钉死常量只写在 eval/ruler/corpus.py 一处；eval/local_load/test_full_corpus.py 读它，自己不再留一份。"""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "eval" / "local_load" / "test_full_corpus.py").read_text(encoding="utf-8")
    assert "EXPECTED_" not in src and "corp.PINNED_QUESTIONS" in src and "corp.load_corpus(" in src
    assert (corpus.PINNED_QUESTIONS, len(corpus.PINNED_QUESTIONS_SHA256), len(corpus.PINNED_CONVERSATIONS_SHA256)) == (1382, 64, 64)


def test_load_refuses_a_corpus_that_is_not_the_pinned_copy(tmp_path):
    root = write_mini_corpus(str(tmp_path))
    with pytest.raises(corpus.PinMismatch):
        corpus.load_corpus(root)
    c = corpus.load_corpus(root, require_pin=False)
    assert c.pinned is False


def test_load_reads_sessions_messages_and_session_time(tmp_path):
    c = corpus.load_corpus(write_mini_corpus(str(tmp_path)), require_pin=False)
    assert [x.sample_id for x in c.conversations] == ["mini-1", "mini-2"]
    first = c.conversations[0]
    assert (first.speaker_a, first.speaker_b) == ("Alice", "Bob")
    assert [s.index for s in first.sessions] == [1, 2]
    assert first.sessions[0].time_ms == _utc_ms(2023, 5, 8, 13, 56)
    m = first.sessions[0].messages[2]
    assert (m.dia_id, m.role, m.caption) == ("D1:3", "user", "a photo of a red tulip garden")
    assert len(c.questions) == 5


# 旧尺子（已删）在合成小题库上切出来的块，原样钉在这里：块的编号、会话编号、用户编号、块里是哪几条消息。
_R = "eval:local:locomo_refined:"
OLD_RULER_CHUNKS = {
    ("mini-1", 20): [(_R + "mini-1:1:0", "mini-1:1", ["D1:1", "D1:2", "D1:3"]),
                     (_R + "mini-1:2:0", "mini-1:2", ["D2:1", "D2:2"])],
    ("mini-1", 2): [(_R + "mini-1:1:0", "mini-1:1", ["D1:1", "D1:2"]),
                    (_R + "mini-1:1:1", "mini-1:1", ["D1:3"]),
                    (_R + "mini-1:2:0", "mini-1:2", ["D2:1", "D2:2"])],
    ("mini-2", 20): [(_R + "mini-2:1:0", "mini-2:1", ["D1:1", "D1:2"])],
    ("mini-2", 2): [(_R + "mini-2:1:0", "mini-2:1", ["D1:1", "D1:2"])],
}


def test_chunking_is_the_old_rulers_chunking(tmp_path):
    """同样的对话切出同样的块——对着旧尺子切法留下的钉死值比（旧尺子本身已删）。"""
    root = write_mini_corpus(str(tmp_path))
    c = corpus.load_corpus(root, require_pin=False)
    for conv in c.conversations:
        for max_messages in (20, 2):
            got = [(b.request_id, b.session_id, [m.dia_id for m in b.messages])
                   for b in corpus.chunk_conversation(conv, max_messages=max_messages)]
            assert got == OLD_RULER_CHUNKS[(conv.sample_id, max_messages)], (conv.sample_id, max_messages)
            assert all(b.user_id == _R + conv.sample_id for b in corpus.chunk_conversation(conv, max_messages=max_messages))


def test_word_limit_also_cuts_blocks():
    long_text = " ".join(["word"] * 1500)
    sess = corpus.Session(index=1, time_text="1:00 pm on 1 May, 2023", time_ms=1, messages=[
        corpus.Msg("D1:%d" % i, "A", "user", long_text, "", 1, i) for i in range(3)])
    conv = corpus.Conv("c", "A", "B", [sess])
    assert [len(b.messages) for b in corpus.chunk_conversation(conv)] == [1, 1, 1]


# ---- Add 的四种形状：一个地方定义 ----

def _first_session(tmp_path):
    c = corpus.load_corpus(write_mini_corpus(str(tmp_path)), require_pin=False)
    return c.conversations[0].sessions[0]


def test_the_four_shapes_are_defined_in_one_place():
    assert set(shapes.SHAPES) == {"a", "b", "c", "d"}
    assert set(shapes.ALL_SHAPES) == {"a", "b", "c", "d", "legacy", "legacy_t"}     # legacy* 只给复现旧数用，不是那四种


def test_shape_a_is_the_most_conservative(tmp_path):
    s = _first_session(tmp_path)
    for m in s.messages:
        out = shapes.render("a", m, s)
        assert set(out) == {"role", "content"}                      # 没有时间
        assert out["role"] in ("user", "assistant")                  # 只有 role，没有人名
        assert out["content"] == m.text                              # 没有人名前缀、没有图片说明


def test_shape_b_adds_a_millisecond_timestamp_taken_from_the_session(tmp_path):
    s = _first_session(tmp_path)
    outs = [shapes.render("b", m, s) for m in s.messages]
    assert all(isinstance(o["timestamp"], int) and not isinstance(o["timestamp"], bool) for o in outs)
    assert {o["timestamp"] for o in outs} == {s.time_ms}              # 时间只有会话一级：同一会话每条同一个毫秒
    assert [o["content"] for o in outs] == [m.text for m in s.messages]
    assert {o["role"] for o in outs} <= {"user", "assistant"}


def test_shape_c_puts_the_speaker_name_in_front_of_the_text(tmp_path):
    s = _first_session(tmp_path)
    out = shapes.render("c", s.messages[0], s)
    assert out["content"] == "Alice: " + s.messages[0].text
    assert out["timestamp"] == s.time_ms and out["role"] == "user"


def test_shape_d_adds_the_image_caption_only_where_there_is_one(tmp_path):
    s = _first_session(tmp_path)
    plain, with_caption = s.messages[0], s.messages[2]
    assert shapes.render("d", plain, s)["content"] == shapes.render("c", plain, s)["content"]
    d = shapes.render("d", with_caption, s)["content"]
    assert d.startswith("Alice: Look at this picture of the garden.")
    assert "a photo of a red tulip garden" in d
    assert "a photo of a red tulip garden" not in shapes.render("c", with_caption, s)["content"]


def test_each_shape_adds_to_the_one_before(tmp_path):
    """a ⊂ b ⊂ c ⊂ d：后一种包含前一种带的所有东西。"""
    s = _first_session(tmp_path)
    m = s.messages[2]
    a, b, c, d = (shapes.render(x, m, s) for x in "abcd")
    assert set(a) < set(b) == set(c) == set(d)
    assert m.text in c["content"] and c["content"] in d["content"]


def test_legacy_is_what_the_old_ruler_wrote(tmp_path):
    s = _first_session(tmp_path)
    out = shapes.render("legacy", s.messages[1], s)
    assert out == {"role": "Bob", "content": s.messages[1].text}


def test_unknown_shape_is_an_error(tmp_path):
    s = _first_session(tmp_path)
    with pytest.raises(KeyError):
        shapes.render("z", s.messages[0], s)


def test_the_full_corpus_load_test_sends_what_the_rulers_shape_a_defines():
    """「平台发什么样」全仓只定义一处（eval/ruler/shapes.py）：整库加载测试的 Add 负载用形状 a 生成，切块用尺子的切法，
    不自己再造一份角色 / 时间 / 切块。"""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "eval" / "local_load" / "test_full_corpus.py").read_text(encoding="utf-8")
    assert 'SHAPE = "a"' in src and "shapes.render(SHAPE" in src and "corp.chunk_conversation(" in src
    assert 'm.get("speaker")' not in src and 'm.get("date_time")' not in src and "_blocks(" not in src
    assert "ruler_reference_hit_rate" in src and "shape a" in src
