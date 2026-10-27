"""The ruler's block ③ — whoever USES what was handed over (for the competition: the answering model).

Step 2 connects it: the answer prompt is written down here word for word (the same text for every model; the Haiku
sub-agents get one extra first line), and an answer is sorted into one of three outcomes:

    correct     the judge said CORRECT
    not_found   the answer is exactly "I don't know" (the prompt tells the model to say that when the memories lack the answer)
    made_up     anything else the judge said WRONG to

`no_answer` (an empty reply) and `judge_error` (the judge could not be read) are kept apart, never folded into "wrong".
"""
import hashlib
import re
from typing import Optional, Protocol, Sequence

ANSWER_PROMPT = """You are answering a question about earlier conversations between two people.
Below are memory snippets retrieved from those conversations, in the order given. Each snippet may show when it was said and who said it.
Use only these snippets. Give a short answer: a few words or one sentence.
If the snippets do not contain the answer, reply exactly: I don't know

Memories:
{memories}

Question: {question}
Answer:"""
ANSWER_PROMPT_SHA256 = hashlib.sha256(ANSWER_PROMPT.encode("utf-8")).hexdigest()

HAIKU_TEST_LINE = "这是一次测试。不要调用任何工具、不要读任何文件，只根据下面给你的内容作答。"

NOT_FOUND_TEXT = "i don't know"


class Answerer(Protocol):
    def answer(self, item, hits: Sequence) -> str:
        """Given the item and the ordered memories, produce what the user of the memories would say."""


class NotConnectedAnswerer:
    def answer(self, item, hits: Sequence) -> str:
        raise NotImplementedError("no answering model is connected")


def one_line(text: str) -> str:
    return re.sub(r"\s*[\r\n]+\s*", " ", text or "").strip()


def build_answer_prompt(memories: Sequence[str], question: str, prefix: str = "") -> str:
    """One memory per line, in the order given. `prefix` (the Haiku test line) goes on its own first line."""
    body = ANSWER_PROMPT.format(memories="\n".join(one_line(m) for m in memories), question=question)
    return prefix + "\n" + body if prefix else body


def is_not_found(answer: str) -> bool:
    t = (answer or "").replace("’", "'").strip().lower().rstrip(".!")
    return t == NOT_FOUND_TEXT


def outcome(answer: str, score: Optional[int]) -> str:
    if score is None and not is_not_found(answer) and (answer or "").strip():
        return "judge_error"
    if is_not_found(answer):
        return "not_found"
    if not (answer or "").strip():
        return "no_answer"
    return "correct" if score == 1 else "made_up"


class ModelAnswerer:
    """Asks one model through a chat client. `answer_with_usage` returns the whole reply (text + the reported usage)."""

    def __init__(self, client):
        self.client = client

    def answer_with_usage(self, question: str, memories: Sequence[str]):
        return self.client.complete(build_answer_prompt(memories, question))

    def answer(self, item, hits: Sequence) -> str:
        return self.answer_with_usage(item.input, [h.content for h in hits]).text
