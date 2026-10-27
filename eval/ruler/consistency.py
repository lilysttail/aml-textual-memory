"""Do our judge and the official `scripts/run_eval.sh` agree? (step 2)

A fixed sample of answers is judged twice: by our `RefinedJudge` (every answer, "I don't know" included — the official script
does not special-case it either) and by the official script, run on files in the official formats. The report lists how many
differ and which question ids; nothing else.
"""
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

from . import judge as judge_mod

SEED = "454-step2-consistency-v1"


def pick_sample(rows: Sequence[dict], model_key: str, group: str, n: int = 30) -> List[str]:
    ids = sorted({r["qa_id"] for r in rows if r["group"] == group and r["model_key"] == model_key and r.get("outcome") not in ("void", "api_error")})
    ids.sort(key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode("utf-8")).hexdigest())
    return sorted(ids[:n])


def write_official_inputs(corpus, rows: Sequence[dict], ids: Sequence[str], out_dir: str) -> None:
    """questions.jsonl (the original records of those questions) and predictions.jsonl ({qa_id, predicted_answer}), the
    official formats. Local files only."""
    os.makedirs(out_dir, exist_ok=True)
    qs = {q["qa_id"]: q for q in corpus.questions}
    ans = {r["qa_id"]: r["answer"] for r in rows}
    with open(os.path.join(out_dir, "questions.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps(qs[i], ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps({"qa_id": i, "predicted_answer": ans[i]}, ensure_ascii=False) + "\n")


def read_official_scores(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("llm_score") is not None:
                out[str(r["qa_id"])] = int(round(float(r["llm_score"])))
    return out


def our_scores(judge: judge_mod.RefinedJudge, corpus, rows: Sequence[dict], ids: Sequence[str]) -> Tuple[Dict[str, int], list]:
    qs = {q["qa_id"]: q for q in corpus.questions}
    ans = {r["qa_id"]: r["answer"] for r in rows}
    scores: Dict[str, int] = {}
    results: list = []
    for i in ids:
        res = judge.score(qs[i]["question"], qs[i].get("answer"), ans[i])
        scores[i] = res.score
        results += res.results
    return scores, results
