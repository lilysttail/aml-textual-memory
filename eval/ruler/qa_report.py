"""The tables of step 2: numbers and question ids only — no question, no memory text, no answer text."""
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

OUTCOMES = ("correct", "not_found", "made_up")


def _count(rows: Sequence[dict]) -> dict:
    c = Counter(r["outcome"] for r in rows)
    return {"questions": len(rows), "correct": c["correct"], "not_found": c["not_found"], "made_up": c["made_up"],
            "other": len(rows) - c["correct"] - c["not_found"] - c["made_up"]}


def _nested(rows: Iterable[dict]) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows:
        out.setdefault(r["group"], {}).setdefault(r["model_key"], []).append(r)
    return out


def build_report(rows: Sequence[dict]) -> dict:
    valid = [r for r in rows if r["outcome"] != "void"]
    rep: dict = {"outcomes": {}, "by_category": {}, "other_detail": {}, "where_lost": {}, "evidence_complete": {}, "usage": {},
                 "void": {}}
    for g, models in _nested(rows).items():
        rep["void"].setdefault(g, {})
        for m, rs in models.items():
            rep["void"][g][m] = sorted(r["qa_id"] for r in rs if r["outcome"] == "void")
    for g, models in _nested(valid).items():
        for tab in ("outcomes", "by_category", "other_detail", "where_lost", "evidence_complete", "usage"):
            rep[tab].setdefault(g, {})
        for m, rs in models.items():
            rep["outcomes"][g][m] = _count(rs)
            cats = sorted({r["category"] for r in rs})
            rep["by_category"][g][m] = {c: _count([r for r in rs if r["category"] == c]) for c in cats}
            rep["other_detail"][g][m] = dict(Counter(r["outcome"] for r in rs if r["outcome"] not in OUTCOMES))
            wrong = [r for r in rs if r["outcome"] in ("made_up", "not_found")]
            rep["where_lost"][g][m] = {
                "wrong_not_returned": sum(1 for r in wrong if r["stage"] == "not_returned"),
                "wrong_returned_not_in_prompt": sum(1 for r in wrong if r["stage"] == "returned_not_in_prompt"),
                "wrong_evidence_in_prompt": sum(1 for r in wrong if r["stage"] == "evidence_in_prompt"),
                "correct_without_evidence": sum(1 for r in rs if r["outcome"] == "correct" and r["stage"] != "evidence_in_prompt"),
            }
            inp = [r for r in wrong if r["stage"] == "evidence_in_prompt"]
            rep["evidence_complete"][g][m] = {
                "wrong_with_all_evidence": sum(1 for r in inp if r["found_in_prompt"] >= r["n_gold"]),
                "wrong_with_part_of_the_evidence": sum(1 for r in inp if r["found_in_prompt"] < r["n_gold"]),
            }
            rep["usage"][g][m] = {k: sum(int(r.get(k) or 0) for r in rs) for k in
                                  ("prompt_tokens", "completion_tokens", "judge_prompt_tokens", "judge_completion_tokens", "judge_calls")}
    return rep


def flip_table(rows: Sequence[dict], first: str, second: str, model: str) -> dict:
    """The same questions in two groups for one model: which stayed right, which turned right or wrong (ids). Void rows are left out."""
    def ok(g):
        return {r["qa_id"]: r["outcome"] == "correct" for r in rows if r["group"] == g and r["model_key"] == model and r["outcome"] != "void"}
    a, b = ok(first), ok(second)
    both = sorted(set(a) & set(b))
    return {"compared": len(both),
            "both_correct": [i for i in both if a[i] and b[i]],
            "only_first_correct": [i for i in both if a[i] and not b[i]],
            "only_second_correct": [i for i in both if b[i] and not a[i]],
            "neither_correct": [i for i in both if not a[i] and not b[i]]}


def agreement(ours: Dict[str, int], official: Dict[str, int]) -> dict:
    both = sorted(set(ours) & set(official))
    differ = [i for i in both if int(ours[i]) != int(official[i])]
    return {"compared": len(both), "same": len(both) - len(differ), "differ": len(differ), "differing_ids": differ,
            "only_ours": sorted(set(ours) - set(official)), "only_official": sorted(set(official) - set(ours))}
