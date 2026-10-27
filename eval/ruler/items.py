"""The ruler's block ①: what is asked, and what a good answer to it looks like.

The fields are not locked to "a question with a gold answer". An item may have no question (something the person said
that should make an old memory surface), and may have no gold answer at all (only the memories that should have come up).
That is what lets the same ruler be pointed at a different consumer later.
"""
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Item:
    item_id: str
    kind: str = "qa"                                   # "qa" | "surface" | whatever a later consumer needs
    input: Optional[str] = None                        # the question, or the utterance; may be None
    scope: int = 100                                   # how many results may be handed over
    expected_answers: Optional[List[str]] = None       # answers that count as right (several are all acceptable); may be None
    expected_evidence: Optional[List[str]] = None      # ids of the memories that should come up; may be None
    category: Optional[str] = None
    group: Optional[str] = None                        # which memory space (conversation) the item belongs to
    meta: dict = field(default_factory=dict)


def to_row(item: Item) -> dict:
    return asdict(item)


def items_from_corpus(corpus, scope: int = 100) -> List[Item]:
    out = []
    for q in corpus.questions:
        answers = q.get("answer")
        if isinstance(answers, str):
            answers = [answers]
        out.append(Item(item_id=str(q.get("qa_id")), kind="qa", input=q.get("question"), scope=scope,
                        expected_answers=[str(a) for a in answers] if answers else None,
                        expected_evidence=[str(e) for e in (q.get("evidence") or [])] or None,
                        category=str(q.get("category")), group=q.get("sample_id"),
                        meta={"multimodal": bool(q.get("is_multi_modality"))}))
    return out
