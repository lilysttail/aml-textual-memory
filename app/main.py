"""FastAPI app exposing the AML Add/Search contract.

The service is a THIN shell around the shared retrieval core (`core/`): the
core holds the records, does the recall, the ranking and the neighbours; this
module does the HTTP contract, the authentication, the platform field mapping
and the rendering. What the platform's answer model reads (`content`) is the
candidate itself; the neighbours are candidates of their own, behind the
NEIGHBOUR_CANDIDATES switch (on by default, the environment can turn it off).
"""
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from core import Core, CoreConfig, Record
from core.db import Database, StoreBusy

from .auth import make_require_api_key
from .config import Settings, load
from .models import AddRequest, AddResponse, SearchHit, SearchRequest, SearchResponse

log = logging.getLogger("app.main")

# The content digest the image build stored (scripts/tree_digest.py). /health reports it as `build`, so a clone of the
# published tag can prove it is what runs (scripts/verify_deploy.py). A local run has no such file: build is null.
BUILD_DIGEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BUILD_DIGEST")
_BUILD_RE = re.compile(r"[0-9a-f]{12}")


def _read_build():
    """The stored digest, or None. Anything that is not exactly twelve lowercase hex characters counts as none and is never echoed."""
    try:
        with open(BUILD_DIGEST_PATH, "rb") as f:
            text = f.read(256).decode("ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text if _BUILD_RE.fullmatch(text) else None

# The recall paths the core serves. The coding-track ones are gone with the
# code track (single 3); two-level recall is gone with them.
_CORE_PATHS = ("keyword", "dense")


def _build_embedder(settings: Settings):
    """The embedding provider the shell hands to the core.

    The core knows the protocol, not the provider: the service picks the model
    (the academic track fixes it), the core plugs it in.
    """
    if settings.embed_provider == "dashscope":
        from .embedder import DashScopeEmbedder
        return DashScopeEmbedder(
            api_key=settings.dashscope_api_key, dim=settings.dashscope_dim,
            batch=settings.dashscope_batch, timeout_s=settings.dashscope_timeout_s,
            max_tokens=settings.dashscope_max_tokens, base_url=settings.dashscope_base_url)
    from .embedder import LocalEmbedder
    return LocalEmbedder(model_dir=settings.bge_small_dir)  # raises loudly if missing


def make_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.db_path,
                      busy_timeout_ms=settings.store_busy_timeout_ms,
                      write_lock_timeout_ms=settings.store_write_lock_timeout_ms)
        paths = tuple(p for p in settings.recall_paths if p in _CORE_PATHS)
        needs_dense = ("dense" in paths) or settings.rerank_enabled
        core = Core.build(CoreConfig(
            db=db,
            paths=paths,
            rerank_enabled=settings.rerank_enabled,
            rerank_n=settings.rerank_n,
            context_before=1,
            context_after=1,
            embedder=_build_embedder(settings) if needs_dense else None,
            embed_max_concurrency=settings.embed_max_concurrency,
            embed_batch=settings.embed_batch,
            embed_backfill_wait_s=settings.embed_backfill_wait_s,
            embed_acquire_s=settings.embed_acquire_s,
            dense_backfill_m=settings.dense_backfill_m,
            dense_backfill_t=settings.dense_backfill_t,
            stem_english=settings.stem_english,
        ))
        app.state.core = core

        yield
        # teardown: drop the vector-cache reference so the onnx session can be
        # GC'd; nothing else needs explicit closing.

    app = FastAPI(title="agent-memory-challenge", lifespan=lifespan)
    _require_auth = make_require_api_key(settings.api_key)

    @app.get("/health")
    def health():
        # The lifespan sets app.state.core; if it hasn't run yet (or if the
        # app was created without a lifespan) we still answer 200 with the
        # minimum contract.
        core = getattr(app.state, "core", None)
        pending = 0
        if core is not None and getattr(core, "config", None) and "dense" in core.config.paths:
            # Sum across all spaces. A database failure here must propagate
            # (an HTTP 500), never masquerade as "no gaps".
            with core.db.connect() as conn:
                spaces = conn.execute(
                    "SELECT DISTINCT space_id FROM core_records"
                ).fetchall()
                for (sid,) in spaces:
                    pending += core.pending_vectors(sid)
        # The configuration the service is RUNNING with (the core's own, not a re-read of the environment),
        # names and switches only — no key, no address. A read-back after a deployment uses this to prove
        # the meaning-based path is on.
        running = core.config if core is not None else None
        return {"ok": True, "pending_vectors": pending, "build": _read_build(),
                "config": {
                    "recall_paths": list(running.paths) if running is not None else list(settings.recall_paths),
                    "embed_provider": settings.embed_provider,
                    "rerank_enabled": running.rerank_enabled if running is not None else settings.rerank_enabled,
                    "stem_english": running.stem_english if running is not None else settings.stem_english,
                    "neighbour_candidates": settings.neighbour_candidates,
                    "dashscope_dim": settings.dashscope_dim,
                    "embed_max_concurrency": (running.embed_max_concurrency if running is not None
                                              else settings.embed_max_concurrency)}}

    def _render(ts, speaker, text):
        """The contract's rendering of one message.

        With an event time: "[2026-10-21 05:00] user: text".
        Without one:      "[user] text" — no made-up date.
        """
        if ts:
            stamp = ts[:16].replace("T", " ")
            return f"[{stamp}] {speaker or 'unknown'}: {text}"
        return f"[{speaker or 'unknown'}] {text}"

    @app.post("/add", response_model=AddResponse)
    def add(req: AddRequest, request: Request, _=Depends(_require_auth)):
        state = request.app.state
        records = []
        for i, m in enumerate(req.messages):
            # The official contract makes `timestamp` optional. When it is
            # missing we do NOT know when the message was said: the event time
            # stays empty and the rendering does not invent a date. The
            # ingestion time is a different thing and is kept as such.
            ts = (datetime.fromtimestamp(m.timestamp / 1000, tz=timezone.utc)
                  if m.timestamp is not None else None)
            stored_at = (ts.isoformat() if ts is not None
                         else datetime.now(timezone.utc).isoformat())
            records.append(Record(record_id=f"{req.request_id}#{i}",
                                  session_id=req.session_id,
                                  text=m.content, speaker=m.role,
                                  ts=ts.isoformat() if ts is not None else None,
                                  created_at=stored_at))
        try:
            state.core.write(records, space_id=req.user_id, batch_id=req.request_id)
        except StoreBusy:
            # Lock wait exceeded; tell the platform to retry rather than 500.
            raise HTTPException(status_code=503, detail="storage busy, retry")
        return AddResponse(request_id=req.request_id,
                           user_id=req.user_id,
                           session_id=req.session_id)

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request, response: Response,
               _=Depends(_require_auth)):
        state = request.app.state
        context = 1 if settings.neighbour_candidates else 0
        try:
            hits = state.core.search(req.query, space_id=req.user_id,
                                     top_k=req.top_k,
                                     context_before=context, context_after=context)
        except StoreBusy:
            raise HTTPException(status_code=503, detail="storage busy, retry")
        # Attach the pending-vectors count for this space as a response header.
        if state.core is not None and "dense" in state.core.config.paths:
            pending = state.core.pending_vectors(req.user_id)
            if pending > 0:
                response.headers["X-Core-Pending-Vectors"] = str(pending)
        out = []
        seen = set()
        seen_records = set()
        for h in hits:
            content = _render(h.ts, h.speaker, h.text)
            out.append({"id": h.record_id, "content": content,
                        "score": float(h.score),
                        "created_at": h.created_at or h.ts or ""})
            seen.add(h.record_id)
            seen_records.add(h.record_id)
        if settings.neighbour_candidates:
            # The neighbours become candidates of their own: each with its own
            # id, nothing repeated, the total never above top_k. The de-dup is
            # on the UNDERLYING record: a neighbour that is already a hit is
            # not handed over twice, however it was reached — so the hits are
            # collected first, then the neighbours.
            for h in hits:
                for nb in list(h.neighbors_before) + list(h.neighbors_after):
                    if len(out) >= req.top_k:
                        break
                    if nb.record_id in seen_records:
                        continue
                    nb_id = f"{h.record_id}#nb{nb.record_id}"
                    if nb_id in seen:
                        continue
                    seen.add(nb_id)
                    seen_records.add(nb.record_id)
                    out.append({"id": nb_id,
                                "content": _render(nb.ts, nb.speaker, nb.text),
                                "score": 0.0,
                                "created_at": nb.created_at or nb.ts or ""})
        out = out[:req.top_k]
        return SearchResponse(data=[SearchHit(**h) for h in out])

    return app


# Production default (uvicorn entry). Tests/eval build their own via make_app.
app = make_app(load())
