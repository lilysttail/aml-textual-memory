"""The check before a "meaning-based path on" version is deployed (step 4, plan v3 section 8).

An earlier concurrency gate once switched the meaning-based half of searches off silently (1,148 of 1,382 searches) and nothing said so.
Counting "skipped = 0" is not enough: with the path switched off by a wrong setting, skipped is 0 as well, concurrent and serial agree, and
nothing was ever used. So this check counts the positive thing too. On the TUNING conversations only (a locked one is refused):
  1. serial, dense: one question at a time, the answers (100 record ids each, in order) kept as the fixed baseline;
  2. serial, keyword only: kept as the other baseline;
  3. concurrent, dense (8 workers): every question at once.
It passes only when all of these hold:
  - config readback: the RUNNING service says its dense path is built and enabled and its embedder is the one the rules fix (`dashscope`);
  - `used_dense` = every question (the dense path returned something for each) and `dense_query_skipped` = 0;
  - the query vectors asked of the provider are the distinct question texts (nothing more, nothing less);
  - no 503;
  - the concurrent answer of every question equals the dense baseline, id for id, in order;
  - the dense baseline and the keyword baseline are NOT all the same (the dense version is not a keyword search in disguise).
How many questions the two baselines differ in is reported only; there is no threshold on it and no number from the locked part is used.
Files: `gate_baseline_dense.jsonl`, `gate_baseline_keyword.jsonl`, `gate_concurrent_dense.jsonl` (ids only), `gate_report.json`.
"""
import json
import os
import shutil
from typing import Callable, Dict, List, Optional, Sequence

from . import dense as dense_mod
from . import load_measure, split

REQUIRED_PROVIDER = "dashscope"            # the embedding provider the competition rules fix for the academic track


def _rows(path: str) -> Dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["qa_id"]] = r
    return out


def run_gate_check(corpus, out_dir: str, vector_cache: str, conv_ids: Sequence[str] = split.TUNE,
                   expect_questions: Optional[int] = split.TUNE_QUESTIONS, workers: int = 8, transport: Optional[Callable] = None,
                   cost_meter=None, embed_cap_tokens: int = 600_000, measure_fn: Callable = load_measure.measure) -> dict:
    if set(conv_ids) & set(split.LOCK):
        raise ValueError("the locked conversations are not touched by this check")
    questions = [q for q in corpus.questions if q.get("sample_id") in set(conv_ids)]
    if expect_questions is not None and len(questions) != expect_questions:
        raise ValueError(f"the conversations hold {len(questions)} questions, not the {expect_questions} expected")
    os.makedirs(out_dir, exist_ok=True)

    def cfg():
        return dense_mod.DenseConfig(transport=transport, cap_tokens=embed_cap_tokens, cache_path=vector_cache, cost_meter=cost_meter)

    common = dict(shape="c", warmup=False, probe_n=0, conv_ids=list(conv_ids), keep_ids=True, stem_english=False)
    serial_dense = measure_fn(corpus, out_dir, dense_cfg=cfg(), workers=1, tag_suffix="_gate_serial", **common)
    serial_kw = measure_fn(corpus, out_dir, dense_cfg=None, workers=1, tag_suffix="_gate_serial", **common)
    concurrent = measure_fn(corpus, out_dir, dense_cfg=cfg(), workers=workers, tag_suffix="_gate_concurrent", **common)

    for src, dst in ((f"rows_dense_cold_gate_serial.jsonl", "gate_baseline_dense.jsonl"), (f"rows_keyword_gate_serial.jsonl", "gate_baseline_keyword.jsonl"),
                     (f"rows_dense_cold_gate_concurrent.jsonl", "gate_concurrent_dense.jsonl")):
        shutil.copyfile(os.path.join(out_dir, src), os.path.join(out_dir, dst))
    base_dense = _rows(os.path.join(out_dir, "gate_baseline_dense.jsonl"))
    base_kw = _rows(os.path.join(out_dir, "gate_baseline_keyword.jsonl"))
    conc = _rows(os.path.join(out_dir, "gate_concurrent_dense.jsonl"))

    n = len(questions)
    qids = [q["qa_id"] for q in questions]
    identical = sum(1 for i in qids if i in conc and i in base_dense and conc[i]["ids"] == base_dense[i]["ids"])
    differ = sum(1 for i in qids if i in base_dense and i in base_kw and base_dense[i]["ids"] != base_kw[i]["ids"])
    distinct_texts = len({q.get("question") or "" for q in questions})
    rb = concurrent.get("config_readback") or {}
    rb_serial = serial_dense.get("config_readback") or {}
    use = concurrent.get("dense_use") or {}
    skipped = concurrent.get("dense_query_skipped")
    seen = ((concurrent.get("embedding_calls_by_phase") or {}).get("during_questions") or {}).get("embed_texts")
    errors = concurrent.get("search_errors") or {}
    http_503 = int(errors.get("503", 0)) + int((concurrent.get("add_errors") or {}).get("503", 0))

    def dense_on(readback):
        return (readback.get("recalls", {}).get("dense") is True and "dense" in (readback.get("paths") or [])
                and readback.get("embed_provider") == REQUIRED_PROVIDER and readback.get("rerank_enabled") is False)

    conditions = {
        "config_readback_dense_on": dense_on(rb) and dense_on(rb_serial),
        "used_dense_every_question": use.get("nonempty") == n and use.get("raised", 0) == 0 and skipped == 0,
        "query_vectors_explained": seen == distinct_texts,
        "no_503": http_503 == 0 and not errors,
        "concurrent_equals_dense_baseline": identical == n,
        "baselines_not_identical": differ >= 1,
    }
    rep = {"passed": all(conditions.values()), "conditions": conditions, "questions": n,
           "used_dense": use.get("nonempty"), "skipped_dense": skipped, "dense_search_calls": use.get("calls"), "http_503": http_503,
           "search_errors": errors, "identical_to_dense_baseline": identical, "baselines_differ_questions": differ,
           "config_readback": rb, "query_vectors": {"expected": distinct_texts, "seen": seen}, "workers": workers,
           "embedding_cost_usd": round(sum(float(((r.get("embedding") or {}).get("cost_usd")) or 0) for r in (serial_dense, concurrent)), 6),
           "seconds": {"serial_dense": (serial_dense.get("seconds") or {}).get("ask"), "concurrent_dense": (concurrent.get("seconds") or {}).get("ask")}}
    with open(os.path.join(out_dir, "gate_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep
