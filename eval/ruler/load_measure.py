"""One measurement of a corpus over real HTTP, in ONE process (step 3) — an earlier stand-alone script, moved into the ruler.

Ported from an earlier stand-alone measurement script: write every conversation
through /add (several conversations at once), optionally warm each conversation with one search, ask every question through
/search at the same time (`workers` threads), then send a few requests ONE AT A TIME to count exactly how many embedding calls
each makes. What changed in the move: the corpus, the blocks, the Add shape, the evidence matching and the pin check are the
ruler's own (`corpus.py`, `shapes.py`, `diagnostics.py`) instead of a second copy; the meaning-based path, its token cap, its
metering and the fixed local vector store come from `dense.py` / `embed_meter.py` / `vector_cache.py`; the target is
`HttpServiceTarget`. The result files carry numbers and question ids only.

Group label: "keyword" (no embedder, no key) or "dense" (+ "_rerank" when reranking is on); the file tag adds "_warm" / "_cold"
for the dense runs.
"""
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from . import corpus as corp
from . import dense as dense_mod
from . import diagnostics, shapes
from .http_target import HttpServiceTarget

EMBED_PRICE_PER_MILLION_TOKENS_USD = 0.07        # text-embedding-v4, international console list price


class SkipCounter(logging.Handler):
    """The service skips the meaning-based half of a search when the embed gate has no free place (it says so in the log, the
    search still answers from the keyword half). Under many simultaneous searches that silently turns the dense path off for
    some questions, so the measurement counts those log lines and reports them."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "dense query vector skipped" in record.getMessage():
            self.count += 1


def _percentiles(values):
    from eval.local_load.common import percentiles
    return percentiles(values)


def measure(corpus, out_dir: str, shape: str = "a", dense_cfg: Optional[dense_mod.DenseConfig] = None, workers: int = 8,
            warmup: bool = False, probe_n: int = 30, conv_ids: Optional[Sequence[str]] = None, top_k: int = 100,
            db_path: Optional[str] = None, neighbours: bool = True, stem_english: bool = False, keep_ids: bool = False,
            tag_suffix: str = "") -> dict:
    convs = [c for c in corpus.conversations if conv_ids is None or c.sample_id in conv_ids]
    keep = {c.sample_id for c in convs}
    questions = [q for q in corpus.questions if q.get("sample_id") in keep]
    label = "keyword" if dense_cfg is None else ("dense_rerank" if dense_cfg.rerank else "dense")
    tmp = None
    if db_path is None:
        tmp = tempfile.mkdtemp(prefix="ruler-load-")
        db_path = os.path.join(tmp, "memory.sqlite")
    skips = SkipCounter()
    logging.getLogger("core.recall").addHandler(skips)
    t = HttpServiceTarget(db_path, neighbours=neighbours, dense=dense_cfg, stem_english=stem_english)
    meter = t.embed_meter
    emap = diagnostics.EvidenceMap()
    lock = threading.Lock()
    started = time.time()
    add_times, add_errors, blocks = [], Counter(), 0

    def push(conv):
        out = []
        for b in corp.chunk_conversation(conv):
            if meter is not None and meter.cap_reached.is_set():
                break
            msgs = [shapes.render(shape, m, s) for m, s in b.pairs()]
            with lock:
                for i, m in enumerate(b.messages):
                    emap.register(t.record_id(b.request_id, i), m.dia_id)
            out.append(t.add_status(b.user_id, b.session_id, b.request_id, msgs))
        return out

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(push, convs):
                for code, dt in res:
                    blocks += 1
                    add_times.append(dt)
                    if code != "200":
                        add_errors[code] += 1
        write_seconds = round(time.time() - started, 1)
        marks_after_write = meter.marks() if meter else {"http_calls": 0, "embed_invocations": 0, "embed_texts": 0}

        warmup_info = None
        if warmup:
            first_q = {}
            for q in questions:
                first_q.setdefault(q.get("sample_id"), q.get("question") or "")

            def warm_one(item):
                sample, text = item
                code, _hits, dt = t.search_status(corp.user_id_of(sample), text, top_k, timeout=1800.0)
                return code, dt
            w0 = time.time()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                ws = list(pool.map(warm_one, first_q.items()))
            warmup_info = {"conversations": len(ws), "seconds_total": round(time.time() - w0, 1),
                           "first_search_seconds": sorted(round(d, 1) for _c, d in ws),
                           "errors": dict(Counter(c for c, _d in ws if c != "200"))}

        def ask(q):
            gold = {str(e) for e in (q.get("evidence") or [])}
            if meter is not None and meter.cap_reached.is_set():
                return {"qa_id": q["qa_id"], "category": q.get("category"), "code": "skipped_cap", "dt": 0.0, "n": 0, "chars": 0, "hit": False,
                        "has_gold": bool(gold)}
            code, hits, dt = t.search_status(corp.user_id_of(q["sample_id"]), q.get("question") or "", top_k)
            got = {emap.dia_of(h.id) for h in hits} - {None}
            return {"qa_id": q["qa_id"], "category": q.get("category"), "code": code, "dt": dt, "n": len(hits),
                    "chars": sum(len(h.content) for h in hits), "hit": bool(got & gold), "has_gold": bool(gold), "ids": [h.id for h in hits]}

        t1 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(ask, questions))
        ask_seconds = round(time.time() - t1, 1)
        marks_after_ask = meter.marks() if meter else dict(marks_after_write)

        probe = None
        if probe_n > 0 and meter is not None and not meter.cap_reached.is_set():
            probe = {"searches": [], "adds": []}
            for q in questions[:probe_n]:
                before = meter.marks()
                code, _h, _dt = t.search_status(corp.user_id_of(q["sample_id"]), q.get("question") or "", top_k)
                after = meter.marks()
                probe["searches"].append({"code": int(code) if code.isdigit() else code, **{k: after[k] - before[k] for k in after}})
            for b in list(corp.chunk_conversation(convs[0]))[:min(5, probe_n)]:
                msgs = [shapes.render(shape, m, s) for m, s in b.pairs()]
                before = meter.marks()
                code, _dt = t.add_status("eval:local:probe_user", f"probe:{b.session_id}", f"probe:{b.request_id}", msgs)
                after = meter.marks()
                probe["adds"].append({"code": int(code) if code.isdigit() else code, "messages": len(msgs), **{k: after[k] - before[k] for k in after}})
            for kind in ("searches", "adds"):
                rows = probe[kind]
                probe[kind + "_summary"] = {"n": len(rows), "http_calls_per_request": dict(Counter(x["http_calls"] for x in rows)),
                                            "embed_invocations_per_request": dict(Counter(x["embed_invocations"] for x in rows)),
                                            "status": dict(Counter(x["code"] for x in rows))}

        asked = len(answers)
        ok = [a for a in answers if a["code"] == "200"]
        skipped = [a for a in answers if a["code"] == "skipped_cap"]
        search_errors = Counter(a["code"] for a in answers if a["code"] not in ("200", "skipped_cap"))
        hits_n = sum(1 for a in ok if a["hit"])
        with_gold = [a for a in answers if a.get("has_gold")]
        by_cat = {}
        for a in answers:
            c = by_cat.setdefault(str(a.get("category")), {"asked": 0, "hit": 0})
            c["asked"] += 1
            c["hit"] += 1 if (a["code"] == "200" and a["hit"]) else 0
        snap = meter.snapshot() if meter else None
        tokens_embedder = getattr(getattr(t, "_caching", None), "tokens_used", None) if meter else None
        tokens = snap["tokens_by_replies"] if snap else 0
        tag = label + ("" if dense_cfg is None else ("_warm" if warmup else "_cold")) + tag_suffix
        result = {
            "label": label, "shape": shape, "switches": dense_cfg.switches() if dense_cfg else dense_mod.KEYWORD_SWITCHES,
            "questions_asked": asked, "questions_answered_ok": len(ok), "questions_skipped_after_cap": len(skipped),
            "stopped_at_token_cap": bool(meter and meter.cap_reached.is_set()), "token_cap": dense_cfg.cap_tokens if dense_cfg else None,
            "blocks_added": blocks, "add_errors": dict(add_errors), "add_latency_s": _percentiles(add_times),
            "search_errors": dict(search_errors), "storage_busy_reasons": dict(t.busy_reasons),
            "search_latency_s": _percentiles([a["dt"] for a in ok]),
            "evidence_hit_rate": round(hits_n / asked, 4) if asked else None, "evidence_hits": hits_n, "questions_with_evidence": len(with_gold),
            "evidence_hit_rate_questions_with_evidence": (round(sum(1 for a in with_gold if a["code"] == "200" and a["hit"]) / len(with_gold), 4)
                                                          if with_gold else None),
            "chars_per_answer_mean": round(sum(a["chars"] for a in ok) / len(ok), 1) if ok else None,
            "answers_over_top_k": sum(1 for a in ok if a["n"] > top_k),
            "by_category": {k: {**v, "rate": round(v["hit"] / v["asked"], 4)} for k, v in sorted(by_cat.items())},
            "protocol": ("warm: one search per conversation first, then all questions" if warmup else "cold: all questions at once"),
            "stem_english": stem_english, "workers": workers, "warmup": warmup_info, "health_status_at_start": t.health_status,
            "embedding_calls_by_phase": ({"during_writes": marks_after_write,
                                          "during_questions": {k: marks_after_ask[k] - marks_after_write[k] for k in marks_after_ask},
                                          "per_block_added": ({k: round(marks_after_write[k] / blocks, 3) for k in marks_after_write} if blocks else None),
                                          "per_question_asked": ({k: round((marks_after_ask[k] - marks_after_write[k]) / asked, 3) for k in marks_after_ask}
                                                                 if asked else None)} if meter else None),
            "sequential_probe": probe,
            "seconds": {"write": write_seconds, "ask": ask_seconds, "total": round(time.time() - started, 1)},
            "embedding": ({**snap, "tokens_used": tokens, "cost_usd": round(tokens * EMBED_PRICE_PER_MILLION_TOKENS_USD / 1_000_000, 6)}
                          if snap else None),
            "vector_store": t.vector_stats() or None,
            "dense_query_skipped": skips.count if dense_cfg is not None else None,
            "dense_use": t.dense_use.snapshot() if t.dense_use is not None else None,         # searches that really ran the dense path
            "config_readback": t.config_readback(),                                              # what the running service says it is set to
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"group_{tag}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"rows_{tag}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for a in answers:
                row = {"qa_id": a["qa_id"], "category": a["category"], "code": a["code"], "hit": a["hit"], "n": a["n"],
                       "chars": a["chars"], "dt": round(a["dt"], 4)}
                if keep_ids:
                    row["ids"] = a.get("ids", [])                    # the record ids of the answer, in order (local file; ids only, no text)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result
    finally:
        logging.getLogger("core.recall").removeHandler(skips)
        t.close()
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
