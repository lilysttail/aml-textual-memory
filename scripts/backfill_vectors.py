#!/usr/bin/env python3
"""Backfill missing vectors for dense-eligible records.

Usage:
    python scripts/backfill_vectors.py <db_path> [--space SPACE] [--batch N] [--dry-run]

For each dense-eligible record (valid=1, not in exclude_lines) that doesn't
have a vector in core_vectors, this script computes and stores the vector.
Records that fail are logged with their reason.

The script uses the same embedder configuration as the running service (read
from the environment). Run it with the same settings the service uses.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Make the core package importable from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import Settings, load  # noqa: E402
from app.embedder import DashScopeEmbedder, LocalEmbedder  # noqa: E402
from core import Core, CoreConfig  # noqa: E402
from core.db import Database, StoreBusy  # noqa: E402
from core.embedder import EmbedUnavailable  # noqa: E402
from core.vector_helper import content_hash as _ch  # noqa: E402
from core.vector_helper import model_key as _model_key  # noqa: E402

log = logging.getLogger("backfill")


def _build_embedder(settings):
    if settings.embed_provider == "dashscope":
        return DashScopeEmbedder(
            api_key=settings.dashscope_api_key,
            dim=settings.dashscope_dim,
            batch=settings.dashscope_batch,
            timeout_s=settings.dashscope_timeout_s,
            max_tokens=settings.dashscope_max_tokens,
            base_url=settings.dashscope_base_url,
        )
    return LocalEmbedder(model_dir=settings.bge_small_dir)


def main():
    parser = argparse.ArgumentParser(description="Backfill missing vectors")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("--space", help="Only backfill this space (default: all)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size for embed calls (default: 32)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done, but don't compute")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    settings = load()
    embedder = _build_embedder(settings)
    db = Database(args.db_path)
    db.create_schema()

    core = Core.build(CoreConfig(
        db=db,
        paths=("keyword", "dense"),
        embedder=embedder,
        embed_max_concurrency=1,
        embed_batch=args.batch,
        embed_backfill_wait_s=60.0,
        embed_acquire_s=30.0,
    ))

    # Get the list of spaces to process.
    if args.space:
        spaces = [args.space]
    else:
        with db.connect() as conn:
            spaces = [r[0] for r in conn.execute(
                "SELECT DISTINCT space_id FROM core_records").fetchall()]

    total_missing = 0
    total_computed = 0
    total_failed = 0

    for space_id in spaces:
        missing = core.missing_vector_records(space_id)
        if not missing:
            log.info("space %s: no missing vectors", space_id)
            continue

        total_missing += len(missing)
        log.info("space %s: %d missing vectors", space_id, len(missing))

        if args.dry_run:
            log.info("  (dry run) would compute %d vectors", len(missing))
            continue

        # Process in batches.
        model = _model_key(embedder.model_name)
        computed = 0
        failed = 0

        for i in range(0, len(missing), args.batch):
            batch = missing[i:i + args.batch]
            texts = [m[2] for m in batch]
            try:
                vectors = core._vc.get_or_compute(texts)
                # Clear pending rows for successfully computed vectors.
                for rid, ch, text in batch:
                    core._clear_pending(space_id, rid)
                computed += len(batch)
                log.info("  batch %d/%d: computed %d vectors",
                         i // args.batch + 1,
                         (len(missing) + args.batch - 1) // args.batch,
                         len(batch))
            except (EmbedUnavailable, StoreBusy) as e:
                # Provider / gate gave up: reconcile per-record from the
                # actual vector store, and only then mark what is really
                # missing. Nothing else is swallowed — a SQLite error must
                # propagate, not become "some embeddings failed".
                log.warning("  batch %d/%d failed: %s",
                            i // args.batch + 1,
                            (len(missing) + args.batch - 1) // args.batch,
                            type(e).__name__)
                hashes = [_ch(text) for _, _, text in batch]
                existing = core._vc._vec.get_vectors(model, hashes)
                now = datetime.now(timezone.utc).isoformat()
                for (rid, ch, text), _ in zip(batch, hashes):
                    if ch in existing:
                        core._clear_pending(space_id, rid)
                    else:
                        core._record_pending(
                            space_id, rid, ch,
                            f"backfill: {type(e).__name__}", now)
                failed += len(batch)

        total_computed += computed
        total_failed += failed
        log.info("space %s: computed %d, failed %d",
                 space_id, computed, failed)

    log.info("total: missing=%d computed=%d failed=%d",
             total_missing, total_computed, total_failed)


if __name__ == "__main__":
    main()