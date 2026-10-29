"""FastAPI app exposing the AML Add/Search contract.

The service is a THIN shell around the shared retrieval core (`core/`): the
core holds the records, does the recall, the ranking and the neighbours; this
module does the HTTP contract, the authentication, the platform field mapping
and the rendering. What the platform's answer model reads (`content`) is the
candidate itself; the neighbours are candidates of their own, behind the
NEIGHBOUR_CANDIDATES switch (on by default, the environment can turn it off).
"""
import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from core import Core, CoreConfig, Record
from core.db import Database, StoreBusy

from .auth import make_require_api_key
from .config import Settings, load
from .models import AddRequest, AddResponse, SearchHit, SearchRequest, SearchResponse

log = logging.getLogger("app.main")

# 2026-10-29: the shell takes an /add batch as the official contract allows and makes each message usable here; a message it
# cannot use is skipped (logged as a count, never its content), the batch is never refused for it. See app/models.py.
_MAX_TIMESTAMP_MS = 4102444800000          # 2100-01-01T00:00:00Z: a later time is "no time" (as before, but no longer a 422)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _content_text(content):
    """The text of a message's content, or None when it carries none (the message is then skipped).
    A string as given (blank -> None); a list of text parts (strings, or objects with a "text" string) joined with single spaces;
    a single such object; a number as its decimal text. Anything else -> None."""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, bool) or content is None:
        return None
    if isinstance(content, (int, float)):
        return str(content)
    parts = content if isinstance(content, list) else [content] if isinstance(content, dict) else []
    texts = []
    for p in parts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict) and isinstance(p.get("text"), str):
            texts.append(p["text"])
    joined = " ".join(t for t in texts if t.strip())
    return joined if joined.strip() else None


def _timestamp_ms(value):
    """Epoch milliseconds as an int, or None ("no time"): a number (a fraction is cut off), a string of digits / a decimal number;
    a negative one is a time before 1970; one after 2100, not finite, or unreadable is None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = int(text)
        except ValueError:
            try:
                value = float(text)
            except ValueError:
                return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        value = int(value)
    if not isinstance(value, int) or value > _MAX_TIMESTAMP_MS:
        return None
    return value


def _event_time(ms):
    """The event time of a message, or None. The same `fromtimestamp` as before for every time it can do (the stored text of an
    ordinary message is unchanged); a time it cannot do (before 1970 on Windows, far out of range) is computed from the epoch, and if
    even that cannot be represented the message has no time. Never raises: a bad time must not turn into a 500."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        pass
    try:
        return _EPOCH + timedelta(milliseconds=ms)
    except (OverflowError, OSError, ValueError):
        return None


def content_batch_key(user_id, session_id, kept):
    """The idempotency key of a batch that came with an empty request id: "anon-" + the first 24 hex of a sha256 over the user id,
    the session id and the messages as they will be stored (position, role, event time, text). The same batch sent again gives the
    same key; a batch that differs in anything stored gives another."""
    doc = {"user_id": user_id, "session_id": session_id,
           "messages": [[i, role, ts.isoformat() if ts is not None else None, text] for i, (role, ts, text) in kept]}
    raw = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "anon-" + hashlib.sha256(raw).hexdigest()[:24]


def normalize_message(item):
    """One /add message -> (role, event time or None, text), or None when it is skipped (not an object, or no text)."""
    if not isinstance(item, dict):
        return None
    text = _content_text(item.get("content"))
    if text is None:
        return None
    role = item.get("role")
    role = role if isinstance(role, str) else "user"
    return role, _event_time(_timestamp_ms(item.get("timestamp"))), text

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

    @app.exception_handler(RequestValidationError)
    async def _log_refusal(request: Request, exc: RequestValidationError):
        # Since 2026-10-29: what is still refused (a field the contract requires is missing, or not of its type) is logged so the next refusal
        # can be read in the service log: one line, which field (loc), what kind (type), the message (msg) — those three ONLY.
        # pydantic's errors() also carries `input` (the request's own text, e.g. a message's content) and `url`: never logged.
        try:
            parts = [f"loc={'.'.join(str(x) for x in e.get('loc', ()))} type={e.get('type')} msg={e.get('msg')}" for e in exc.errors()]
            log.warning("refused 422 %s %s: %s", request.method, request.url.path, " | ".join(parts)[:1000])
        except Exception:                                       # the log line must never change the answer
            pass
        return await request_validation_exception_handler(request, exc)

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
        kept = []
        for i, item in enumerate(req.messages):
            norm = normalize_message(item)
            if norm is not None:
                kept.append((i, norm))               # no text to keep: skipped, the rest of the batch is stored
        skipped = len(req.messages) - len(kept)
        # The batch key: the request id (writes are idempotent per (user_id, batch key)). An empty request id would make every such
        # batch of one user "the same batch" and silently drop all but the first, so it is replaced by a key made of the batch's
        # own content: sent again unchanged it is still recognised; any other batch gets another key. The reply keeps the empty id.
        batch_key = req.request_id or content_batch_key(req.user_id, req.session_id, kept)
        # An empty session id: the batch's own key is its session key, so batches without a session never become one
        # session (and each other's neighbours).
        session_key = req.session_id if req.session_id else batch_key
        for i, (role, ts, text) in kept:
            # The official contract makes `timestamp` optional. When it is
            # missing we do NOT know when the message was said: the event time
            # stays empty and the rendering does not invent a date. The
            # ingestion time is a different thing and is kept as such.
            stored_at = (ts.isoformat() if ts is not None
                         else datetime.now(timezone.utc).isoformat())
            # record id keeps the message's position in the request as sent (a skipped message leaves a gap, nothing is renumbered)
            records.append(Record(record_id=f"{batch_key}#{i}",
                                  session_id=session_key,
                                  text=text, speaker=role,
                                  ts=ts.isoformat() if ts is not None else None,
                                  created_at=stored_at))
        if skipped:
            log.warning("add: %d of %d message(s) skipped (no text)", skipped, len(req.messages))
        try:
            state.core.write(records, space_id=req.user_id, batch_id=batch_key)
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
        # Since 2026-10-29: a blank query or a top_k below 1 is answered, not refused: 200 with no results
        if not req.query.strip() or req.top_k < 1:
            return SearchResponse(data=[])
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
