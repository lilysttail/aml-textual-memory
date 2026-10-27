"""A fixed local store for text vectors, so the same text is never embedded twice (step 3).

`DiskVectorCache` is a small SQLite file keyed by (model name, sha256 of the text): it holds hashes and numbers, never the text.
`CachingEmbedder` wraps the embedder the service uses: a text already in the file costs nothing, only the others go to the
provider, and what comes back is written down. Because the key is the text and not a record, a second run over the same data
(even through a brand-new service database) asks the provider for nothing. The wrapper looks like the embedder it wraps
(`provider`, `model_name`, `dim`, `embed`).
"""
import hashlib
import sqlite3
import struct
import threading
from typing import Dict, List, Optional, Sequence


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DiskVectorCache:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._lock = threading.Lock()
        self._closed = False
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS vectors(model TEXT, text_hash TEXT, dim INTEGER, vec BLOB, "
                               "PRIMARY KEY(model, text_hash))")

    def get_many(self, model: str, texts: Sequence[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        with self._lock:
            for t in set(texts):
                row = self._conn.execute("SELECT dim, vec FROM vectors WHERE model=? AND text_hash=?", (model, _key(t))).fetchone()
                if row:
                    out[t] = list(struct.unpack(f"<{row[0]}d", row[1]))
        return out

    def put_many(self, model: str, vectors: Dict[str, Sequence[float]]) -> None:
        with self._lock, self._conn:
            for t, v in vectors.items():
                self._conn.execute("INSERT OR REPLACE INTO vectors(model, text_hash, dim, vec) VALUES (?,?,?,?)",
                                   (model, _key(t), len(v), struct.pack(f"<{len(v)}d", *v)))

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        """Close the SQLite file (idempotent); an open connection keeps the file undeletable on Windows."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()


class CachingEmbedder:
    """One thread at a time asks the provider for a given (model, text): the others wait for it and read the answer from the file
    (a failed asker releases its claim, and a waiter then asks for itself). Within one call a repeated text is asked for once."""

    def __init__(self, inner, cache: DiskVectorCache):
        self._inner = inner
        self._cache = cache
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}   # text hash -> set when the asker is done (answered or failed)
        self._hits = self._misses = self._inner_calls = 0

    def __getattr__(self, name):                    # provider, model_name, dim, tokens_used, ...
        return getattr(self._inner, name)

    def embed(self, texts, timeout_s: Optional[float] = None):
        if not texts:
            return []
        model = self._inner.model_name
        have: Dict[str, List[float]] = {}
        asked_by_me = 0
        while True:
            have.update(self._cache.get_many(model, [t for t in texts if t not in have]))
            need = [t for t in dict.fromkeys(texts) if t not in have]
            if not need:
                break
            with self._lock:                           # claim what nobody is asking for; wait for what somebody is
                mine = [t for t in need if _key(t) not in self._inflight]
                for t in mine:
                    self._inflight[_key(t)] = threading.Event()
                waiting = [self._inflight[_key(t)] for t in need if _key(t) in self._inflight and t not in mine]
            if mine:
                try:
                    vecs = self._inner.embed(mine, timeout_s=timeout_s) if timeout_s is not None else self._inner.embed(mine)
                    fresh = dict(zip(mine, vecs))
                    self._cache.put_many(model, fresh)
                    have.update(fresh)
                    with self._lock:
                        self._inner_calls += 1
                finally:                               # answered or failed: release the claim either way
                    with self._lock:
                        events = [self._inflight.pop(_key(t), None) for t in mine]
                    for ev in events:
                        if ev is not None:
                            ev.set()
                asked_by_me += len(mine)
            for ev in waiting:
                ev.wait()
        with self._lock:
            self._hits += len(texts) - asked_by_me         # a repeat inside one call, or one somebody else asked for, is a hit
            self._misses += asked_by_me
        return [have[t] for t in texts]

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "inner_calls": self._inner_calls}
