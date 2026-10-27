"""Vector cache with single-flight de-dup, a concurrency gate, and batching.

复制自本项目早期服务层的 app/vector_cache.py（本仓库不含该文件；2026-10-21，小助手），
改了：
  - 内容哈希走 core/vector_helper.py 的唯一帮手，不再自带一份；
  - model 键也走 helper 的 model_key()；
  - get_or_compute 支持 deadline：给了 deadline 后，抢闸、
    等别人、批间检查全部用剩余时间，到点立刻放弃，不等完再看表。

没改逻辑：single-flight 去重、并发闸、分批、失败时同一个错误对象发给所有等待者。

Two correctness details (ke review):
  - claim window: between reading the cache and claiming, a previous owner may
    have just written the vector and left the in-flight table, so a new claimer
    could recompute it. _compute_mine re-reads the cache after claiming and
    skips anything already there — at most a no-op, never a wrong result.
  - the error travels on the in-flight object (_Flight), not a separate table.
    A waiter reads the error off the exact _Flight it waited on, so a later
    request reclaiming the same hash can't clear the error out from under it.
"""
import threading
import time

from .db import StoreBusy
from .vector_helper import content_hash as _content_hash
from .vector_helper import model_key


class _Flight:
    """One in-flight compute for a content_hash: an event + the owner's error."""
    __slots__ = ("event", "error")

    def __init__(self):
        self.event = threading.Event()
        self.error = None


def _remaining(deadline):
    """Seconds left until *deadline* (monotonic), or None when no deadline."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


class VectorCache:
    def __init__(self, vec, embedder, max_concurrency=None, batch_size=None,
                 wait_timeout_s=None, acquire_timeout_s=None):
        self._vec = vec
        self._emb = embedder
        mc = 2 if max_concurrency is None else max_concurrency
        self._sem = threading.BoundedSemaphore(max(1, mc))
        self._batch = 32 if batch_size is None else batch_size
        self._wait_to = 60.0 if wait_timeout_s is None else wait_timeout_s
        self._acq_to = 30.0 if acquire_timeout_s is None else acquire_timeout_s
        self._inflight = {}   # content_hash -> _Flight
        self._lock = threading.Lock()

    def get_or_compute(self, texts, deadline=None):
        """Vectors for *texts* (cached ones reused; missing ones computed).

        deadline: a time.monotonic() value. When given, every wait inside
        (gate, in-flight, between batches) uses the remaining time and gives
        up as soon as it runs out — the caller is not blocked past the
        deadline. Without it, the configured timeouts apply.
        """
        import numpy as np
        model = model_key(self._emb.model_name)
        order_hashes = [_content_hash(t) for t in texts]
        uniq = {}
        for h, t in zip(order_hashes, texts):
            uniq.setdefault(h, t)

        result = dict(self._vec.get_vectors(model, list(uniq.keys())))
        need = [h for h in uniq if h not in result]

        mine, waits = [], []
        with self._lock:
            for h in need:
                fl = self._inflight.get(h)
                if fl is None:
                    self._inflight[h] = _Flight()
                    mine.append(h)
                else:
                    waits.append((h, fl))

        if mine:
            self._compute_mine(model, mine, uniq, np, deadline)  # raises if our compute failed

        for h, fl in waits:
            if deadline is not None:
                if _remaining(deadline) <= 0:
                    raise StoreBusy(f"vector wait deadline exceeded for {h[:8]}")
                if not fl.event.wait(_remaining(deadline)):
                    raise StoreBusy(f"vector wait timed out for {h[:8]}")
            else:
                if not fl.event.wait(self._wait_to):
                    raise StoreBusy(f"vector wait timed out for {h[:8]}")
            if fl.error is not None:
                raise fl.error

        if need:
            got = self._vec.get_vectors(model, need)
            for h in need:
                if h not in got:
                    raise StoreBusy(f"vector unavailable after wait for {h[:8]}")
                result[h] = got[h]
        return [np.frombuffer(result[h], dtype=np.float32) for h in order_hashes]

    def _compute_mine(self, model, hashes, uniq, np, deadline=None):
        err = None
        pending = list(hashes)
        try:
            # Claim window: another owner may have written some of these between our
            # cache read and our claim. Re-read and release those without computing.
            already = self._vec.get_vectors(model, hashes)
            if already:
                with self._lock:
                    for h in list(already.keys()):
                        fl = self._inflight.pop(h, None)
                        if fl is not None:
                            fl.event.set()
            pending = [h for h in hashes if h not in already]
            for i in range(0, len(pending), self._batch):
                # With a deadline, stop as soon as the remaining time runs out
                # instead of starting another batch the caller can't use.
                rem = _remaining(deadline)
                if rem is not None and rem <= 0:
                    raise StoreBusy("embed backfill deadline exceeded")
                chunk = pending[i:i + self._batch]
                chunk_texts = [uniq[h] for h in chunk]
                # With a deadline (search path), the gate is non-blocking:
                # grab it or give up immediately. Without one (write path),
                # wait up to the configured acquire timeout.
                if not self._sem.acquire(
                        timeout=0 if rem is not None else self._acq_to):
                    raise StoreBusy("embed concurrency gate unavailable")
                try:
                    # The remaining budget becomes THIS call's HTTP timeout:
                    # a slow provider call gives up at the deadline instead of
                    # running past it (the adapter turns the transport timeout
                    # into EmbedUnavailable). Without a deadline the configured
                    # per-request timeout applies.
                    vecs = self._emb.embed(chunk_texts, timeout_s=rem)
                finally:
                    self._sem.release()
                rows = [(h, np.asarray(v, dtype=np.float32).tobytes())
                        for h, v in zip(chunk, vecs)]
                self._vec.put_vectors(model, self._emb.dim, rows)
        except Exception as e:
            err = e
        finally:
            with self._lock:
                for h in pending:
                    fl = self._inflight.pop(h, None)
                    if fl is not None:
                        fl.error = err          # error travels on the _Flight
                        fl.event.set()
        if err is not None:
            raise err
