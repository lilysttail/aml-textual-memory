"""Evidence diagnosis: for each item, how many of the memories that should have come back did, and where the first one sits.

This is a CLUE FOR WHERE THINGS GO WRONG, not a score (the only score is "the answer was right"). It is the old ruler's
evidence matching (by id, never "does the text contain the answer"), moved here; the old entry point stays until this
reproduces its numbers.
"""
from typing import Dict, List, Optional, Sequence

KS = (1, 5, 10, 100)


class EvidenceMap:
    """record id -> the dataset's own id (dia_id). The service hands neighbours over as candidates of their own: their id is
    "<hit id>#nb<neighbour's own record id>", and what the neighbour IS is its own record, not the hit it hangs on."""

    def __init__(self):
        self._dia: Dict[str, str] = {}

    def register(self, record_id: str, dia_id: str) -> None:
        self._dia[record_id] = dia_id

    def dia_of(self, hit_id: str) -> Optional[str]:
        if "#nb" in hit_id:
            hit_id = hit_id.split("#nb", 1)[1]
        return self._dia.get(hit_id)


def diagnose(item, hits: Sequence, emap: EvidenceMap) -> dict:
    gold = set(item.expected_evidence or [])
    ranks: Dict[str, int] = {}
    for pos, h in enumerate(hits, start=1):
        dia = emap.dia_of(h.id)
        if dia is not None and dia in gold and dia not in ranks:
            ranks[dia] = pos
    first = min(ranks.values()) if ranks else None
    return {"item_id": item.item_id, "kind": item.kind, "category": item.category, "group": item.group,
            "n_gold": len(gold), "found_n": len(ranks), "ranks": ranks, "first_rank": first, "hit": bool(ranks),
            "returned": len(hits), "chars": sum(len(h.content) for h in hits)}


def _block(rows: Sequence[dict]) -> dict:
    n = len(rows)
    out: Dict[str, object] = {"questions": n}
    if not n:
        out.update({"hit_rate": 0.0, "mrr": 0.0, "mrr_all": 0.0, "chars_total": 0, "approx_tokens_total": 0,
                    "gold_found_fraction_mean": 0.0})
        for k in KS:
            out[f"recall@{k}"] = 0.0
        return out
    found = [r for r in rows if r.get("hit")]
    out["hit_rate"] = len(found) / n
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in rows if r.get("first_rank") is not None and r["first_rank"] <= k) / n
    out["mrr"] = (sum(1.0 / r["first_rank"] for r in found) / len(found)) if found else 0.0
    out["mrr_all"] = sum(1.0 / r["first_rank"] for r in found) / n
    out["chars_total"] = sum(int(r.get("chars", 0)) for r in rows)
    out["approx_tokens_total"] = sum(int(r.get("chars", 0)) // 4 for r in rows)
    out["gold_found_fraction_mean"] = sum((r["found_n"] / r["n_gold"]) if r.get("n_gold") else 0.0 for r in rows) / n
    return out


def summarise(rows: Sequence[dict]) -> dict:
    out = _block(rows)
    for key, name in (("category", "by_category"), ("group", "by_group")):
        buckets: Dict[str, List[dict]] = {}
        for r in rows:
            buckets.setdefault(str(r.get(key) or "unknown"), []).append(r)
        out[name] = {k: _block(v) for k, v in sorted(buckets.items())}
    return out
