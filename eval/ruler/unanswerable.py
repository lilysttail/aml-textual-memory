"""Questions that cannot be answered AT ALL in a given storing shape — counted without any model (step 1).

"Cannot be answered at all" = what the answer needs is not in the store, whatever the finder does. Three causes, each its own
class (a question may fall into more than one; the union is de-duplicated). "In the store" means: the text of every record of
that conversation as the answering model would read it ("[time] role: content", read back from the service's own store).

  U1  the question names a speaker of its own conversation (whole-word, either of the two), and in ALL the names it names,
      no record of the conversation contains that name. A lower bound: a missing name costs real answers only some of the time.
  U2  the question asks for a time (category "2") and no record of the conversation carries a time.
  U3  at least one acceptable answer text is in the caption of an evidence message, and none of the acceptable answers is in
      the text of ANY message of the conversation, and the shape stores no caption — so nothing the model can read has it.
      (Only answers that match a caption word for word are caught; answers that need working out from a caption are not.)
"""
import re
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List

from . import corpus as corp
from . import shapes, target

TIME_WORDING = re.compile(r"\b(when|what (date|day|time|month|year)|which (date|day|month|year)|how long ago)\b", re.I)
_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]")


def _words(text: str) -> str:
    return " " + re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).strip() + " "


def _has_word(text: str, name: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", text or "", re.I) is not None


def _candidates(q: dict) -> List[str]:
    a = q.get("answer")
    a = [a] if isinstance(a, str) else (a or [])
    return [w for w in (_words(str(x)) for x in a) if w.strip()]


def _conv_of(corpus, sample_id):
    for c in corpus.conversations:
        if c.sample_id == sample_id:
            return c
    raise KeyError(sample_id)


def answer_only_in_caption(corpus, q: dict) -> bool:
    """U3's two shape-free conditions (柯's wording): at least one candidate answer is in the caption of an evidence message,
    and NONE of the candidates is in the text of any message of the conversation."""
    cands = _candidates(q)
    if not cands:
        return False
    conv = _conv_of(corpus, q.get("sample_id"))
    evidence = set(str(e) for e in (q.get("evidence") or []))
    msgs = [m for s in conv.sessions for m in s.messages]
    captions = " ".join(_words(m.caption) for m in msgs if m.dia_id in evidence and m.caption)
    if not any(c in captions for c in cands):
        return False
    texts = " ".join(_words(m.text) for m in msgs)
    return not any(c in texts for c in cands)


def stored_texts_rule(conv: corp.Conv, shape: str) -> List[str]:
    """What the service would store and hand back, computed by rule (a test keeps it equal to the real service)."""
    return [target.stored_line(shape, m, s) for b in corp.chunk_conversation(conv) for m, s in b.pairs()]


def stored_texts_service(conv: corp.Conv, shape: str) -> List[str]:
    """The same, read back from a real in-process service after writing the conversation through Add."""
    d = tempfile.mkdtemp(prefix="ruler-u-")
    t = target.ServiceTarget(d + "/u.sqlite", neighbours=False)
    try:
        for b in corp.chunk_conversation(conv):
            if t.add(b.user_id, b.session_id, b.request_id, [shapes.render(shape, m, s) for m, s in b.pairs()]) != 200:
                raise RuntimeError("Add refused a block")
        return t.stored_texts(corp.user_id_of(conv.sample_id))
    finally:
        t.close()


def classify(corpus, stored: str = "service") -> Dict[str, dict]:
    """{shape: {"questions": {qa_id: [reasons...]}}} for the four shapes."""
    get = stored_texts_service if stored == "service" else stored_texts_rule
    res = {}
    for shape in shapes.SHAPES:
        per_conv = {c.sample_id: get(c, shape) for c in corpus.conversations}
        out = {}
        for q in corpus.questions:
            texts = per_conv[q.get("sample_id")]
            blob = " ".join(texts)
            reasons = []
            names = [n for n in (q.get("speaker_a") or "", q.get("speaker_b") or "") if n and _has_word(q.get("question") or "", n)]
            if names and all(not _has_word(blob, n) for n in names):
                reasons.append("U1")
            if str(q.get("category")) == "2" and not any(_STAMP.match(t) for t in texts):
                reasons.append("U2")
            if answer_only_in_caption(corpus, q) and not any(c in _words(blob) for c in _candidates(q)):
                reasons.append("U3")
            out[str(q.get("qa_id"))] = reasons
        res[shape] = {"questions": out}
    return res


def questions_per_category(corpus) -> Dict[str, int]:
    return dict(Counter(str(q.get("category")) for q in corpus.questions))


def table(corpus, res: Dict[str, dict]) -> Dict[str, dict]:
    cat_of = {str(q.get("qa_id")): str(q.get("category")) for q in corpus.questions}
    out = {}
    for shape, r in res.items():
        total = Counter()
        by = defaultdict(Counter)
        for qid, reasons in r["questions"].items():
            cat = cat_of[qid]
            by[cat]["questions"] += 1
            for x in reasons:
                total[x] += 1
                by[cat][x] += 1
            if reasons:
                total["union"] += 1
                by[cat]["union"] += 1
        out[shape] = {"total": {k: total.get(k, 0) for k in ("U1", "U2", "U3", "union")},
                      "by_category": {c: {k: by[c].get(k, 0) for k in ("questions", "U1", "U2", "U3", "union")} for c in sorted(by)}}
    return out


def naming_gap(corpus) -> Dict[str, dict]:
    """NOT one of the three definitions — a sensitivity count for U1. Questions that name a speaker of their own conversation
    where NONE of the named speakers appears in the text of ANY evidence message. In shapes a and b (no name in front of the
    text) the model must work out who spoke from `role` alone; in c and d every message starts with the name, so the gap is 0.
    Also: how many questions have ANY named speaker missing from every stored record (the "any" variant of U1)."""
    out: Dict[str, dict] = {}
    for q in corpus.questions:
        cat = str(q.get("category"))
        row = out.setdefault(cat, {"questions": 0, "naming_own_speaker": 0, "none_of_named_in_evidence_text": 0})
        row["questions"] += 1
        names = [n for n in (q.get("speaker_a") or "", q.get("speaker_b") or "") if n and _has_word(q.get("question") or "", n)]
        if not names:
            continue
        row["naming_own_speaker"] += 1
        conv = _conv_of(corpus, q.get("sample_id"))
        evidence = set(str(e) for e in (q.get("evidence") or []))
        etext = " ".join(m.text for s in conv.sessions for m in s.messages if m.dia_id in evidence)
        if all(not _has_word(etext, n) for n in names):
            row["none_of_named_in_evidence_text"] += 1
    return out


def u1_any_missing(corpus, shape: str = "a") -> int:
    """U1 with "ANY named speaker is absent from every stored record" instead of "ALL" (sensitivity only)."""
    per_conv = {c.sample_id: " ".join(stored_texts_rule(c, shape)) for c in corpus.conversations}
    n = 0
    for q in corpus.questions:
        names = [x for x in (q.get("speaker_a") or "", q.get("speaker_b") or "") if x and _has_word(q.get("question") or "", x)]
        if names and any(not _has_word(per_conv[q.get("sample_id")], x) for x in names):
            n += 1
    return n


def time_wording_by_category(corpus) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for q in corpus.questions:
        row = out.setdefault(str(q.get("category")), {"with_time_wording": 0, "total": 0})
        row["total"] += 1
        row["with_time_wording"] += 1 if TIME_WORDING.search(q.get("question") or "") else 0
    return out


def u2_wording_sensitivity(corpus) -> Dict[str, object]:
    """NOT a definition — U2 is counted by category "2" (the corpus' own label). This lists the questions of OTHER categories whose
    wording asks for a time, to show how much more would be unanswerable in a shape that stores no time if U2 were counted by
    wording instead. Ids and counts only."""
    cat2 = [q for q in corpus.questions if str(q.get("category")) == "2"]
    asks = lambda q: TIME_WORDING.search(q.get("question") or "") is not None
    other = [q for q in corpus.questions if str(q.get("category")) != "2" and asks(q)]
    extra = {}
    for shape in shapes.SHAPES:
        no_time = {c.sample_id: not any(_STAMP.match(t) for t in stored_texts_rule(c, shape)) for c in corpus.conversations}
        extra[shape] = sum(1 for q in other if no_time[q.get("sample_id")])
    return {"category_2_questions": len(cat2),
            "category_2_with_wording": sum(1 for q in cat2 if asks(q)),
            "category_2_without_wording": sum(1 for q in cat2 if not asks(q)),
            "other_category_with_wording": len(other),
            "other_category_with_wording_ids": [str(q.get("qa_id")) for q in other],
            "extra_if_counted_by_wording": extra}
