"""The ruler's block ②: the thing being measured — "given an input, hand back an ordered list of memories".

`ServiceTarget` is the competition service itself, in this process (FastAPI's test client, no network, no docker), built by the
very same `app.main.make_app`, and spoken to through the real /add and /search contract. Switches that are not secrets are set
here; nothing that needs a key is touched (the local keyword path only — no embedder, no model).
"""
from dataclasses import dataclass
from typing import List, Optional



@dataclass(frozen=True)
class Hit:
    id: str
    content: str


def render_like_service(ts: Optional[str], speaker: str, text: str) -> str:
    """The candidate text the service hands over (app/main.py `_render`): "[YYYY-MM-DD HH:MM] role: text", or
    "[role] text" when the event time is not known. A test keeps this in step with the real service."""
    if ts:
        stamp = ts[:16].replace("T", " ")
        return f"[{stamp}] {speaker or 'unknown'}: {text}"
    return f"[{speaker or 'unknown'}] {text}"


def stored_line(shape: str, msg, session) -> str:
    """What the service stores and hands back for one message written in `shape` (the rule; a test keeps it equal to the real
    service)."""
    import datetime as dt
    from . import shapes
    r = shapes.render(shape, msg, session)
    ts = dt.datetime.fromtimestamp(r["timestamp"] / 1000, tz=dt.timezone.utc).isoformat() if "timestamp" in r else None
    return render_like_service(ts, r["role"], r["content"])


class CoreTarget:
    """The same core WITHOUT the HTTP shell, written the way the old ruler wrote it (record ids "<request>:<i>", `turn_index`
    given, no rendering). Exists so the old numbers can be reproduced exactly; the service is what the platform talks to."""
    record_sep = ":"

    def __init__(self, db_path: str, neighbours: bool = False):
        from core import CoreConfig, build
        self._core = build(CoreConfig(db_path=db_path, paths=("keyword",)))
        self._neighbours = neighbours

    def record_id(self, request_id: str, i: int) -> str:
        return f"{request_id}:{i}"

    def add(self, user_id: str, session_id: str, request_id: str, messages: List[dict]) -> int:
        from core import Record
        recs = [Record(record_id=self.record_id(request_id, i), session_id=session_id, text=m["content"], speaker=m["role"],
                       ts=None, turn_index=i) for i, m in enumerate(messages)]
        self._core.write(recs, space_id=user_id, batch_id=request_id)
        return 200

    def search(self, user_id: str, query: str, top_k: int = 100) -> List[Hit]:
        ctx = 1 if self._neighbours else 0
        hits = self._core.search(query, space_id=user_id, top_k=top_k, context_before=ctx, context_after=ctx)
        out = [Hit(id=h.record_id, content=h.text) for h in hits]
        if self._neighbours:
            seen = {h.id for h in out}
            counted = set(seen)
            for h in hits:
                for nb in list(h.neighbors_before) + list(h.neighbors_after):
                    if len(out) >= top_k:
                        break
                    if nb.record_id in counted:
                        continue
                    nb_id = f"{h.record_id}#nb{nb.record_id}"
                    if nb_id in seen:
                        continue
                    seen.add(nb_id)
                    counted.add(nb.record_id)
                    out.append(Hit(id=nb_id, content=nb.text))
        return out[:top_k]

    def close(self) -> None:
        pass


class ServiceTarget:
    record_sep = "#"

    def record_id(self, request_id: str, i: int) -> str:
        return f"{request_id}#{i}"

    def __init__(self, db_path: str, neighbours: bool = False, dense=None, stem_english: bool = False):
        """`dense` (an `eval.ruler.dense.DenseConfig`) switches the service to keyword + dense; without it nothing needing a key
        is touched (the local keyword path only — no embedder, no model)."""
        from .dense import LOCAL_KEY as key, DenseUseCounter, service_env
        self.stem_english = stem_english
        with service_env(db_path, neighbours, dense, stem_english=stem_english) as (hooks, m):
            from fastapi.testclient import TestClient
            self._app = m.make_app(m.load())
            self._client = TestClient(self._app)
            self._client.__enter__()                  # runs the lifespan: builds the core (and, for dense, the embedder)
        self.embed_meter = hooks.embed_meter
        self._caching = hooks.caching
        self._cache = hooks.cache                  # the vector file: closed with the target (an open file cannot be deleted on Windows)
        self.dense_use = DenseUseCounter().install() if dense is not None else None      # counts the searches that really ran the dense path
        self._auth = {"Authorization": f"Bearer {key}"}

    def vector_stats(self) -> dict:
        return self._caching.stats() if self._caching is not None else {}

    def add(self, user_id: str, session_id: str, request_id: str, messages: List[dict]) -> int:
        """POST /add; returns the HTTP status (200 = stored)."""
        body = {"request_id": request_id, "user_id": user_id, "session_id": session_id, "messages": messages}
        return self._client.post("/add", json=body, headers=self._auth).status_code

    def search(self, user_id: str, query: str, top_k: int = 100) -> List[Hit]:
        r = self._client.post("/search", json={"query": query, "user_id": user_id, "top_k": top_k}, headers=self._auth)
        if r.status_code != 200:
            raise RuntimeError(f"search answered {r.status_code}")
        return [Hit(id=h["id"], content=h["content"]) for h in r.json().get("data", [])]

    def stored_texts(self, user_id: str) -> List[str]:
        """Every record of this user as the answering model would read it (read from the service's own store)."""
        core = self._app.state.core
        with core.db.connect() as conn:
            rows = conn.execute("SELECT speaker, ts, text FROM core_records WHERE space_id=? ORDER BY session_id, turn_index",
                                (user_id,)).fetchall()
        return [render_like_service(ts, speaker, text) for speaker, ts, text in rows]

    def close(self) -> None:
        try:
            self._client.__exit__(None, None, None)
        except Exception:
            pass
        if self.dense_use is not None:
            self.dense_use.uninstall()
        if self._cache is not None:
            self._cache.close()
