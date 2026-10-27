"""Which questions step 2 asks, and what each group hands to the answering model.

The pick is fixed and repeatable: within each category the eligible questions are ordered by a hash of their id and the first
N are taken, so it does not depend on the order the questions are stored in. Only the conversations named by the caller are
used (the tuning part; the locked part is never passed in). The cross-check set is the first few of each category's picks.

Five groups (the shape the memories were stored in, and what is handed over):
    G1  shape a, the 100 returned memories        G4  shape a, only the standard-evidence sentences (an upper bound)
    G2  shape c, the 100 returned memories        G5  shape c, only the standard-evidence sentences (an upper bound)
    G3  shape c, the first 20 of the same search
Search: the service path, keyword only, neighbours on, top_k 100.
"""
import hashlib
from collections import Counter
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import corpus as corp
from . import diagnostics, items, shapes, split, target

SEED = "454-step2-v1"
CATEGORIES = ("1", "2", "3", "4")


@dataclass(frozen=True)
class Group:
    name: str
    shape: str
    mode: str                       # "hits" (what the search returned) or "evidence" (only the standard evidence)
    limit: Optional[int]


GROUPS: Dict[str, Group] = {
    "G1": Group("G1", "a", "hits", 100),
    "G2": Group("G2", "c", "hits", 100),
    "G3": Group("G3", "c", "hits", 20),
    "G4": Group("G4", "a", "evidence", None),
    "G5": Group("G5", "c", "evidence", None),
}


@dataclass
class Context:
    qa_id: str
    hits: list
    evidence_lines: List[str]
    n_gold: int
    found_100: int
    found_20: int
    dense_used: Optional[bool] = None            # did the meaning-based path really run for this question (None = keyword-only, nothing to say)


def _rank(qa_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{qa_id}".encode("utf-8")).hexdigest()


def pick_questions(corpus, conv_ids: Sequence[str], per_category: int = 12, cross_per_category: int = 3,
                   categories: Sequence[str] = CATEGORIES) -> Dict[str, List[str]]:
    conv_ids = set(conv_ids)
    main: List[str] = []
    cross: List[str] = []
    for cat in categories:
        eligible = [q for q in corpus.questions
                    if q.get("sample_id") in conv_ids and str(q.get("category")) == cat and q.get("evidence") and q.get("answer")]
        if len(eligible) < per_category:
            raise ValueError(f"category {cat}: {len(eligible)} eligible questions, {per_category} asked for")
        picked = [q["qa_id"] for q in sorted(eligible, key=lambda q: _rank(q["qa_id"]))[:per_category]]
        main += picked
        cross += picked[:cross_per_category]
    return {"main": main, "cross": cross}


def check_picks(corpus, picks, conv_ids: Sequence[str], per_category: Optional[int] = None,
                cross_per_category: Optional[int] = None, categories: Sequence[str] = CATEGORIES) -> None:
    """The picks file is read back from disk, where a person may have edited it: every id must be a question of this corpus that
    belongs to one of `conv_ids` and not to a locked conversation; no id twice in either list; the cross-check set must sit inside
    the main set. Given the parameters the picks were made with (`per_category`, `cross_per_category`, `categories`), the counts must
    match too: the main set holds exactly `per_category` questions of every category, the cross-check set exactly
    `cross_per_category` of every category — a question dropped, added or swapped for one of another category is refused.
    Anything wrong refuses the whole run (before anything is asked or written)."""
    if not isinstance(picks, dict) or not isinstance(picks.get("main"), list) or not isinstance(picks.get("cross"), list):
        raise ValueError("picks.json must hold a \"main\" list and a \"cross\" list")
    sample_of = {q["qa_id"]: q.get("sample_id") for q in corpus.questions}
    category_of = {q["qa_id"]: str(q.get("category")) for q in corpus.questions}
    allowed = set(conv_ids) - set(split.LOCK)
    for qid in list(picks["main"]) + list(picks["cross"]):
        if qid not in sample_of:
            raise ValueError("picks.json names a question that is not in the corpus")
        if sample_of[qid] not in allowed:
            raise ValueError(f"picks.json names a question outside the allowed conversations ({qid})")
    if len(set(picks["main"])) != len(picks["main"]):
        raise ValueError("picks.json names a question twice")
    if len(set(picks["cross"])) != len(picks["cross"]):
        raise ValueError("the cross-check set names a question twice")
    if not set(picks["cross"]) <= set(picks["main"]):
        raise ValueError("the cross-check set is not inside the main set")
    for name, per in (("main", per_category), ("cross", cross_per_category)):
        if per is None:
            continue
        want = {c: per for c in categories}
        got = Counter(category_of[q] for q in picks[name])
        if len(picks[name]) != per * len(categories) or dict(got) != want:
            raise ValueError(f"the {name} set does not hold exactly {per} question(s) of each of the categories {list(categories)}")


def build_contexts(corpus, qa_ids: Sequence[str], shape: str, top_k: int = 100, dense=None,
                   stats_out: Optional[dict] = None, stem_english: bool = False) -> Dict[str, Context]:
    """Write the conversations the questions belong to through the service in `shape`, search each question, and keep the
    returned memories (in order, neighbours included) with how much of the standard evidence came back."""
    qs = {q["qa_id"]: q for q in corpus.questions}
    need = {qs[i]["sample_id"] for i in qa_ids}
    tmp = tempfile.mkdtemp(prefix="ruler-qa-")
    t = target.ServiceTarget(os.path.join(tmp, "s.sqlite"), neighbours=True, dense=dense, stem_english=stem_english)
    emap = diagnostics.EvidenceMap()
    lines_of: Dict[str, Dict[str, str]] = {}
    try:
        for conv in corpus.conversations:
            if conv.sample_id not in need:
                continue
            lines_of[conv.sample_id] = {}
            for b in corp.chunk_conversation(conv):
                msgs = [shapes.render(shape, m, s) for m, s in b.pairs()]
                if t.add(b.user_id, b.session_id, b.request_id, msgs) != 200:
                    raise RuntimeError("Add refused a block")
                for i, (m, s) in enumerate(b.pairs()):
                    emap.register(t.record_id(b.request_id, i), m.dia_id)
                    lines_of[conv.sample_id][m.dia_id] = target.stored_line(shape, m, s)
        out: Dict[str, Context] = {}
        for qid in qa_ids:
            q = qs[qid]
            before = t.dense_use.marks() if t.dense_use is not None else None
            hits = t.search(corp.user_id_of(q["sample_id"]), q["question"], top_k)
            dense_used = None if before is None else (t.dense_use.marks()[1] > before[1])       # searches run one at a time: the count is this question's
            it = items.Item(item_id=qid, input=q["question"], expected_evidence=[str(e) for e in q.get("evidence") or []])
            d100 = diagnostics.diagnose(it, hits, emap)
            d20 = diagnostics.diagnose(it, hits[:20], emap)
            gold = [str(e) for e in q.get("evidence") or []]
            lines = lines_of[q["sample_id"]]
            conv_order = [m.dia_id for c in corpus.conversations if c.sample_id == q["sample_id"] for s in c.sessions for m in s.messages]
            ev = [lines[d] for d in conv_order if d in gold and d in lines]
            out[qid] = Context(qa_id=qid, hits=hits, evidence_lines=ev, n_gold=d100["n_gold"], found_100=d100["found_n"],
                               found_20=d20["found_n"], dense_used=dense_used)
        return out
    finally:
        if stats_out is not None and t.embed_meter is not None:
            stats_out["embedding"] = t.embed_meter.snapshot()
            stats_out["vector_store"] = t.vector_stats() or None
        if stats_out is not None and getattr(t, "dense_use", None) is not None:
            stats_out["dense_use"] = t.dense_use.snapshot()
        t.close()
        shutil.rmtree(tmp, ignore_errors=True)


def memories_for(group: Group, ctx: Context) -> List[str]:
    if group.mode == "evidence":
        return list(ctx.evidence_lines)
    return [h.content for h in ctx.hits[:group.limit]]
