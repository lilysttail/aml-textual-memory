"""Recall paths — the core's layer 2.

复制自本项目早期服务层的 app/recall.py（本仓库不含该文件；2026-10-21，小助手），改了：
  - 只留 KeywordRecall 和 DenseRecall；代码赛道的指纹那一路不进核心；
  - user_id -> space_id，mem_id -> record_id；
  - DenseRecall 多一个 exclude_lines：某些线（终端行）不进向量路，但仍进
    关键词索引；
  - DenseRecall 的候选集只取已有向量的记录 + 限量补算：
    补算和 query 向量都带同一个 deadline，到点立刻放弃（不等完再看表），
    闸满/接口慢/同段在算都不会拖住搜索。
没改：每路只出 [(record_id, score)]、各路互不知情、融合在 rerank 层。

两层召回（任务层）不进核心：那是代码赛道的东西，留给比赛壳，单 3 清理。
"""
import math  # noqa: F401  (kept for parity with the source module)
import logging
import time

from .db import StoreBusy
from .embedder import EmbedUnavailable
from .vector_helper import content_hash as _content_hash
from .vector_helper import model_key as _model_key

log = logging.getLogger("core.recall")

# Search-path backfill defaults: at most M texts per search,
# bounded by T seconds. If the gate is full or the provider is slow, we
# skip the backfill and use only the vectors already in the cache.
DEFAULT_BACKFILL_M = 8
DEFAULT_BACKFILL_T = 2.0


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
    """Semantic recall over vectors (full pool cosine).

    Rerank lives in the retrieval layer: rerank() re-orders the candidate
    pool with vectors; DenseRecall itself only ranks the whole space.
    """

    name = "dense"

    def __init__(self, store, vector_cache, enabled=True, exclude_lines=(),
                 backfill_m=None, backfill_t=None):
        self._store = store
        self._vc = vector_cache
        self.enabled = enabled
        self._exclude_lines = tuple(exclude_lines or ())
        self._backfill_m = DEFAULT_BACKFILL_M if backfill_m is None else backfill_m
        self._backfill_t = DEFAULT_BACKFILL_T if backfill_t is None else backfill_t

    def candidates(self, space_id, query, options, limit):
        import numpy as np

        items = self._store.iter_records(space_id, exclude_lines=self._exclude_lines)
        if not items:
            return []
        ids = [i for i, _ in items]
        contents = [c for _, c in items]

        # One deadline covers the whole dense path of this search: the
        # limited backfill AND the query vector. When it runs out we give
        # up immediately — never "wait, then check the clock".
        deadline = time.monotonic() + self._backfill_t

        model = _model_key(self._vc._emb.model_name)
        hashes = [_content_hash(t) for t in contents]
        cached = self._vc._vec.get_vectors(model, hashes)

        kept_ids = []
        kept_vecs = []
        missing_texts = []
        missing_positions = []  # original index of each missing text

        for i, (h, t) in enumerate(zip(hashes, contents)):
            if h in cached:
                kept_ids.append(ids[i])
                kept_vecs.append(np.frombuffer(cached[h], dtype=np.float32))
            else:
                missing_texts.append(t)
                missing_positions.append(i)

        # Limited backfill: at most M texts, bounded by the deadline.
        # get_or_compute(deadline=...) gives up inside (gate, in-flight,
        # between batches) as soon as the time runs out. Only the vectors
        # that actually came back participate; the rest wait for the next
        # backfill pass.
        if missing_texts and self._backfill_m > 0:
            batch = missing_texts[:self._backfill_m]
            try:
                new_vecs = self._vc.get_or_compute(batch, deadline=deadline)
                for i, vec in enumerate(new_vecs):
                    orig_idx = missing_positions[i]
                    kept_ids.append(ids[orig_idx])
                    kept_vecs.append(np.asarray(vec, dtype=np.float32))
            except (StoreBusy, EmbedUnavailable) as e:
                # Backfill gave up (deadline / gate / provider). The vectors
                # already computed before the give-up are cached; the rest
                # wait for the next pass. Search proceeds with what we have.
                log.warning("dense backfill skipped: %s", e)
            # No bare except: a real bug (SQLite error, logic error) must
            # propagate and be visible, not masquerade as "dense down".

        if not kept_vecs:
            return []

        # The query vector is under the same deadline. If it can't be
        # computed in time, dense has nothing to rank against — skip the
        # path (keyword still answers).
        try:
            qv = np.asarray(self._vc.get_or_compute([query], deadline=deadline)[0])
        except (StoreBusy, EmbedUnavailable) as e:
            log.warning("dense query vector skipped: %s", e)
            return []

        mat = np.vstack(kept_vecs)
        sims = mat @ qv
        order = np.argsort(-sims)[:limit]
        return [(kept_ids[k], float(sims[k])) for k in order]
