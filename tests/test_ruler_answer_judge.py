"""第二步的答题提示、说没找到的认法、官方 refined 判分（语义照抄官方，接口用假的）。"""
import hashlib

import pytest

from eval.ruler import answerer, chat, items, judge


class FakeChat:
    """顶替 ChatClient：按顺序吐文本，记下收到的提示。"""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        t = self.texts.pop(0)
        if isinstance(t, Exception):
            raise t
        return chat.ChatResult(text=t, prompt_tokens=100, completion_tokens=5, model="judge-m", attempts=1, finish_reason="stop")


# ---- 答题提示：原文写死 ----

def test_the_answer_prompt_is_the_one_in_the_dispatch_word_for_word():
    p = answerer.build_answer_prompt(["[a] one", "[b] two"], "Who?")
    assert p == ("You are answering a question about earlier conversations between two people.\n"
                 "Below are memory snippets retrieved from those conversations, in the order given. "
                 "Each snippet may show when it was said and who said it.\n"
                 "Use only these snippets. Give a short answer: a few words or one sentence.\n"
                 "If the snippets do not contain the answer, reply exactly: I don't know\n"
                 "\n"
                 "Memories:\n"
                 "[a] one\n"
                 "[b] two\n"
                 "\n"
                 "Question: Who?\n"
                 "Answer:")
    assert hashlib.sha256(answerer.ANSWER_PROMPT.encode("utf-8")).hexdigest() == answerer.ANSWER_PROMPT_SHA256


def test_a_memory_is_one_line_even_if_the_text_has_line_breaks():
    p = answerer.build_answer_prompt(["[a] first\nsecond\r\nthird"], "Q?")
    assert "[a] first second third\n" in p


def test_the_haiku_version_only_has_one_extra_first_line():
    base = answerer.build_answer_prompt(["[a] x"], "Q?")
    h = answerer.build_answer_prompt(["[a] x"], "Q?", prefix=answerer.HAIKU_TEST_LINE)
    assert h == answerer.HAIKU_TEST_LINE + "\n" + base
    assert answerer.HAIKU_TEST_LINE.startswith("这是一次测试。不要调用任何工具、不要读任何文件")


def test_no_memories_still_gives_a_well_formed_prompt():
    p = answerer.build_answer_prompt([], "Q?")
    assert "Memories:\n\n\nQuestion: Q?\nAnswer:" in p


# ---- 三种结局 ----

@pytest.mark.parametrize("text", ["I don't know", "I don't know.", " i don't know ", "I don’t know", "I DON'T KNOW!"])
def test_saying_i_dont_know_is_not_found(text):
    assert answerer.is_not_found(text) is True


@pytest.mark.parametrize("text", ["", "I don't know, maybe Paris", "Paris", "I do not know", "unknown"])
def test_anything_else_is_an_attempt(text):
    assert answerer.is_not_found(text) is False


def test_outcome_comes_from_the_answer_and_the_judge():
    assert answerer.outcome("I don't know", 0) == "not_found"
    assert answerer.outcome("Paris", 1) == "correct"
    assert answerer.outcome("Rome", 0) == "made_up"
    assert answerer.outcome("", 0) == "no_answer"
    assert answerer.outcome("Paris", None) == "judge_error"


def test_the_answerer_sends_exactly_the_built_prompt_and_returns_the_usage():
    c = FakeChat("Paris")
    r = answerer.ModelAnswerer(c).answer_with_usage("Where?", ["[x] a", "[y] b"])
    assert c.prompts == [answerer.build_answer_prompt(["[x] a", "[y] b"], "Where?")]
    assert (r.text, r.prompt_tokens) == ("Paris", 100)


# ---- 判分：官方 refined ----

def test_the_refined_prompt_is_the_official_text_byte_for_byte():
    assert hashlib.sha256(judge.REFINED_PROMPT.encode("utf-8")).hexdigest() == judge.REFINED_PROMPT_SHA256
    assert judge.REFINED_PROMPT_SHA256 == "899d035bd61813bc718812b2997f5908111c63ccd2ed76533e65f5dc3ca337a4"
    assert len(judge.REFINED_PROMPT) == 2346 and "Core principle — Inclusion + Non-contradiction" in judge.REFINED_PROMPT


def test_the_judge_fills_the_official_template_and_reads_the_label():
    c = FakeChat('```json\n{\n    "label": "CORRECT"\n}\n```')
    r = judge.RefinedJudge(c).score("Q?", ["Gold"], "Pred")
    assert r.score == 1 and r.shortcut is None and len(c.prompts) == 1
    assert c.prompts[0] == judge.REFINED_PROMPT.format(question="Q?", gold_answer="Gold", generated_answer="Pred")
    assert r.results[0].prompt_tokens == 100


def test_wrong_label_and_odd_case():
    assert judge.RefinedJudge(FakeChat('{"label": "WRONG"}')).score("Q", ["G"], "P").score == 0
    assert judge.RefinedJudge(FakeChat('{"label": " correct "}')).score("Q", ["G"], "P").score == 1


def test_an_exact_match_is_correct_without_asking_a_model():
    c = FakeChat()
    r = judge.RefinedJudge(c).score("Q", ["Paris", "France"], "Paris")
    assert (r.score, r.shortcut, c.prompts) == (1, "exact", [])


def test_an_empty_prediction_is_wrong_without_asking_a_model():
    c = FakeChat()
    r = judge.RefinedJudge(c).score("Q", ["Paris"], "")
    assert (r.score, r.shortcut, c.prompts) == (0, "empty", [])


def test_several_gold_answers_any_correct_wins_and_all_are_tried_when_none_is():
    c = FakeChat('{"label": "WRONG"}', '{"label": "CORRECT"}', '{"label": "WRONG"}')
    r = judge.RefinedJudge(c).score("Q", ["a", "b", "c"], "P")
    assert r.score == 1 and len(c.prompts) == 2                    # 第二个就对了，不再问第三个
    c = FakeChat('{"label": "WRONG"}', '{"label": "WRONG"}')
    assert judge.RefinedJudge(c).score("Q", ["a", "b"], "P").score == 0 and len(c.prompts) == 2


def test_the_official_prediction_cleaning_is_applied_before_judging():
    assert judge.normalize_prediction("  think </think> Paris ") == "Paris"
    assert judge.normalize_prediction("reasoning... Final Answer: Paris") == "Paris"
    assert judge.normalize_prediction("so \\boxed{Paris} ok") == "Paris"
    assert judge.normalize_prediction("plain") == "plain"
    c = FakeChat()
    assert judge.RefinedJudge(c).score("Q", ["Paris"], "Final answer: Paris").shortcut == "exact"


def test_an_unreadable_judge_reply_is_retried_then_an_error_never_a_silent_wrong():
    c = FakeChat("no json here", "still none", "nope")
    with pytest.raises(judge.JudgeError):
        judge.RefinedJudge(c, parse_retries=2).score("Q", ["G"], "P")
    assert len(c.prompts) == 3
    c = FakeChat("no json", '{"label": "CORRECT"}')
    assert judge.RefinedJudge(c, parse_retries=2).score("Q", ["G"], "P").score == 1


def test_the_judge_error_from_the_chat_layer_is_passed_on_as_a_judge_error():
    c = FakeChat(chat.ChatError("down", status=500))
    with pytest.raises(judge.JudgeError):
        judge.RefinedJudge(c).score("Q", ["G"], "P")


def test_the_json_object_is_found_inside_text_and_fences():
    assert judge.extract_json_object('blah {"label": "WRONG"} blah') == '{"label": "WRONG"}'
    assert judge.extract_json_object('```json\n{"label": "WRONG"}\n```') == '{"label": "WRONG"}'
    with pytest.raises(ValueError):
        judge.extract_json_object("nothing")


def test_the_judge_works_on_a_ruler_item_through_the_interface():
    it = items.Item(item_id="q", input="Q?", expected_answers=["Paris"], expected_evidence=["D1:1"])
    assert judge.RefinedJudge(FakeChat('{"label": "CORRECT"}')).judge(it, "The city of Paris") == judge.Verdict.CORRECT
    assert judge.RefinedJudge(FakeChat('{"label": "WRONG"}')).judge(it, "Rome") == judge.Verdict.MADE_UP
    assert judge.RefinedJudge(FakeChat()).judge(it, "I don't know") == judge.Verdict.NOT_FOUND
