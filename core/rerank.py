"""Ranking / fusion seam — the core's layer 3.

复制自本项目早期服务层的 app/retrieval.py（本仓库不含该文件；2026-10-21，小助手），
改了：user_id -> space_id，mem_id -> record_id，内容从核心的 store 取
（get_rows 返回 text，不是带前缀的 content）。没改：RRF 的 k=60、rank 的
排序与截断、重排用向量余弦。

★ 一条不许破的规矩：排序永远不生成答案 —— 它只重排候选。options 可以影响
检索，但不许合成答案。
"""
from typing import List, Tuple


def rank(pool: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
    """Final ordering: sort (id, score) desc, cut."""
    return sorted(pool, key=lambda t: t[1], reverse=True)[:top_k]


def merge_rrf(paths, k: int = 60):
    """Reciprocal-rank fusion (RRF, Cormack 2009; k=60 fixed weights).

    Each path's [(record_id, score)] is treated as a ranking (score desc). A
    candidate's fused score is sum over paths of 1/(k + its rank in that
    path). There is NO single-path special case: 1/(k+rank) is strictly
    monotonic, so for one path the fused order equals the input order — the
    formula handles it.
    """
    scores = {}
    for path in paths:
        for rank, (rid, _) in enumerate(path, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank(store, space_id, query, pool, get_vc, top_n=None):
    """Vector rerank of the candidate pool.

    pool: [(record_id, score)] from the enabled recall paths. Takes the first
    top_n candidates (default: all), embeds their text + the query via the
    shared VectorCache, and reorders by cosine. Same candidates, same cache,
    same model — scores reproduce.
    """
    import numpy as np
    cand = pool[:top_n] if top_n is not None else pool
    if not cand:
        return []
    ids = [i for i, _ in cand]
    rows = store.get_rows(space_id, ids)
    present = [i for i in ids if i in rows]
    if not present:
        return []
    texts = [rows[i]["text"] for i in present]
    vc = get_vc()
    mat = np.vstack(vc.get_or_compute(texts))
    qv = np.asarray(vc.get_or_compute([query])[0])
    sims = mat @ qv  # L2-normalized -> dot product is cosine
    order = np.argsort(-sims)
    return [(present[k], float(sims[k])) for k in order]
