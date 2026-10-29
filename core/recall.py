"""Recall paths — the core's layer 2.

复制自本项目早期服务层的 app/recall.py（本仓库不含该文件；2026-10-21，小助手），改了：
  - 只留 KeywordRecall 和 DenseRecall；代码赛道的指纹那一路不进核心；
  - user_id -> space_id，mem_id -> record_id；
  - DenseRecall 多一个 exclude_lines：某些线（终端行）不进向量路，但仍进
    关键词索引；
  - DenseRecall 的候选集来自内存里的空间索引（core/dense_index.py：建一次、留着、每次用前核代际），
    不再每次整份重读；query 向量有自己的期限（从要它那一刻算，可以在期限内等闸位），
    限量补算用之后剩下的时间、不等闸；到点立刻放弃，不等完再看表。
没改：每路只出 [(record_id, score)]、各路互不知情、融合在 rerank 层。

两层召回（任务层）不进核心：那是代码赛道的东西，留给比赛壳，单 3 清理。
"""
import math  # noqa: F401  (kept for parity with the source module)
import logging
import time

from .db import StoreBusy
from .embedder import EmbedUnavailable
from .vector_helper import model_key as _model_key

log = logging.getLogger("core.recall")

# Search-path backfill defaults: at most M texts per search,
# bounded by T seconds. If the gate is full or the provider is slow, we
# skip the backfill and use only the vectors already in the cache.
DEFAULT_BACKFILL_M = 8
DEFAULT_BACKFILL_T = 2.0
# The query vector's own deadline (seconds), from when it is asked for, waiting for the embedding gate included.
DEFAULT_QUERY_T = 3.0


class KeywordRecall:
    name = "keyword"

    def __init__(self, store, enabled=True):
        self._store = store
        self.enabled = enabled

    def candidates(self, space_id, query, options, limit, include_ids=None):
        rows = self._store.keyword_search(space_id, query, limit=limit,
                                          include_ids=include_ids)
        return [(r[0], float(r[1])) for r in rows]


class DenseRecall:
    """Semantic recall over vectors (the whole space, ranked by similarity).

    The space's "record -> vector" table comes from the in-memory DenseIndex (core/dense_index.py): built once,
    kept between searches, checked against the change counters on every use. A search then costs one query
    vector and one similarity pass — not a re-read of the space.

    Order inside one search:
      1. the space's index (may wait for a heavy-work slot; past the limit SearchBusy fails the search);
      2. the query vector, under its OWN deadline (`query_t`) that starts when it is asked for and includes
         waiting for a slot of the embedding gate — earlier work cannot use it up;
      3. similarities;
      4. with the time left in `backfill_t` (counted from here): at most M records that still have no vector
         get one and join this search's ranking (the gate is not waited for).
    Rerank lives in the retrieval layer.
    """

    name = "dense"

    def __init__(self, store, vector_cache, enabled=True, exclude_lines=(),
                 backfill_m=None, backfill_t=None, query_t=None, index=None):
        from .dense_index import DenseIndex
        self._store = store
        self._vc = vector_cache
        self.enabled = enabled
        self._exclude_lines = tuple(exclude_lines or ())
        self._backfill_m = DEFAULT_BACKFILL_M if backfill_m is None else backfill_m
        self._backfill_t = DEFAULT_BACKFILL_T if backfill_t is None else backfill_t
        self._query_t = DEFAULT_QUERY_T if query_t is None else query_t
        self._index = index if index is not None else DenseIndex(
            store._db, store, vector_cache._vec, _model_key(vector_cache._emb.model_name),
            exclude_lines=self._exclude_lines)

    @property
    def index(self):
        return self._index

    def candidates(self, space_id, query, options, limit):
        idx = self._index.get(space_id)              # SearchBusy propagates: the search answers 503
        try:
            return self._rank(idx, query, limit)
        finally:
            self._index.done(idx)                    # end the lease: the index may now be released / evicted

    def _rank(self, idx, query, limit):
        import numpy as np
        if not idx.ids and not idx.missing:
            return []

        try:
            deadline = time.monotonic() + self._query_t
            qv = np.asarray(self._vc.get_or_compute([query], deadline=deadline, wait_for_gate=True)[0],
                            dtype=np.float32)
        except (StoreBusy, EmbedUnavailable) as e:
            # No query vector in time: dense has nothing to rank against — skip the path (keyword still answers).
            log.warning("dense query vector skipped: %s", e)
            return []

        ids = list(idx.ids)
        sims = list(self._index.scores(idx, qv)) if idx.ids else []

        # Limited backfill with what is left of backfill_t: only vectors that came back join this ranking;
        # the rest wait for a later search or the backfill script. No bare except: a real bug must surface.
        if idx.missing and self._backfill_m > 0:
            batch = idx.missing[:self._backfill_m]
            try:
                new_vecs = self._vc.get_or_compute([t for _, _, t in batch],
                                                   deadline=time.monotonic() + self._backfill_t)
                for (rid, _, _), vec in zip(batch, new_vecs):
                    ids.append(rid)
                    sims.append(float(np.asarray(vec, dtype=np.float32) @ qv))
            except (StoreBusy, EmbedUnavailable) as e:
                log.warning("dense backfill skipped: %s", e)

        if not ids:
            return []
        sims = np.asarray(sims, dtype=np.float32)
        order = np.argsort(-sims)[:limit]          # same ordering call as before the in-memory index (ties included)
        return [(ids[k], float(sims[k])) for k in order]
