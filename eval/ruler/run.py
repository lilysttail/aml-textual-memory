"""One run of the ruler over the local copy: write every conversation through Add in one shape, ask every item once, diagnose.

    python -m eval.ruler run --data <copy> --out <dir> --shape a|b|c|d|legacy --neighbours 0|1 [--limit N]

Local, no network, no model, no money: the in-process service on the keyword path. Output is numbers and ids only — no question,
no conversation text.
"""
import json
import os
import tempfile
import time
from typing import Optional

from . import corpus as corp
from . import diagnostics, items, shapes, split, target


def run(data_dir: str, out_dir: str, shape: str = "a", neighbours: int = 0, limit: int = 0, top_k: int = 100,
        require_pin: bool = True, db_path: Optional[str] = None, engine: str = "service", partition: str = "all",
        lock=split.LOCK, version: Optional[str] = None, declared=None, ledger: Optional[str] = None) -> dict:
    """partition "tune" / "locked" keep only that part's conversations; "all" is the whole set and is for describing it only,
    never for choosing between versions. "locked" needs `version`, `declared` (the two versions of the one comparison) and a
    `ledger` (a directory): the claim is made before anything is written, and a second run of the same version is refused."""
    if partition not in ("all", "tune", "locked"):
        raise ValueError("partition is all, tune or locked")
    if partition == "locked" and not (version and declared and ledger):
        raise ValueError("a run on the locked part names its version, the declared pair and the ledger")
    c = corp.load_corpus(data_dir, require_pin=require_pin)
    if partition != "all":
        if tuple(lock) == split.LOCK and require_pin:
            split.check_partition(split.per_conversation_counts(c))     # the corpus is the one the cut was made on
        keep = set(lock) if partition == "locked" else {x.sample_id for x in c.conversations} - set(lock)
        c.conversations = [x for x in c.conversations if x.sample_id in keep]
        c.questions = [q for q in c.questions if q.get("sample_id") in keep]
    if partition == "locked":
        split.claim_locked(ledger, version, declared)                  # after the checks, before anything is written
    all_items = items.items_from_corpus(c, scope=top_k)
    if limit:
        all_items = all_items[:limit]
    os.makedirs(out_dir, exist_ok=True)
    own_db = db_path is None
    tmpdir = tempfile.mkdtemp(prefix="ruler-") if own_db else None
    db_path = db_path or os.path.join(tmpdir, "ruler.sqlite")
    started = time.time()
    t = (target.CoreTarget if engine == "core" else target.ServiceTarget)(db_path, neighbours=bool(neighbours))
    emap = diagnostics.EvidenceMap()
    written = 0
    try:
        for conv in c.conversations:
            for b in corp.chunk_conversation(conv):
                msgs = [shapes.render(shape, m, s) for m, s in b.pairs()]
                status = t.add(b.user_id, b.session_id, b.request_id, msgs)
                if status != 200:
                    raise RuntimeError(f"Add answered {status} for a block (shape {shape})")
                for i, m in enumerate(b.messages):
                    emap.register(t.record_id(b.request_id, i), m.dia_id)
                written += len(msgs)
        rows, skipped = [], 0
        for it in all_items:
            if not it.expected_evidence:
                skipped += 1
                continue
            if it.input is None:
                rows.append(dict(diagnostics.diagnose(it, [], emap), note="no_input"))
                continue
            hits = t.search(corp.user_id_of(it.group), it.input, it.scope)
            rows.append(diagnostics.diagnose(it, hits, emap))
    finally:
        t.close()
    summary = diagnostics.summarise(rows)
    summary.update({"engine": engine, "shape": shape, "shape_doc": shapes.ALL_SHAPES[shape].doc, "neighbours": int(neighbours),
                    "pinned": c.pinned, "partition": partition, "version": version, "items_total": len(all_items),
                    "items_without_evidence_skipped": skipped, "records_written": written, "seconds": round(time.time() - started, 1)})
    tag = f"shape_{shape}_neighbours_{int(neighbours)}" + ("" if engine == "service" else f"_{engine}")
    with open(os.path.join(out_dir, f"rows_{tag}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
