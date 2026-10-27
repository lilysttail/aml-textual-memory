"""Counts what goes to / comes back from the embedding provider and stops it locally at a hard cap (step 3).

The cap is a wall. Every request goes through the run's paid gate (`paid_gate.py`): one lock, the whole request made inside it, so at
most one paid call is outside at any time. Inside the gate, before the request leaves: is "tokens used + an upper bound of this
request" within the cap, and "spent + the most it can cost" within the spending limit? The upper bound is
`sum(bytes of the texts) + 2 * texts` (a token covers at least one byte; each text carries two special tokens), or the body's byte length
when the body cannot be read. After it: a 2xx reply with usage is booked at its usage; a 2xx reply without usage, and a request that
raised, are booked at the upper bound (the provider may have charged them); an error reply (4xx / 5xx) is not processed and is not
booked, even if it carries a usage block. A 2xx reply that charges more than the bound means the bound was wrong: it is counted and the whole
gate closes, so everything after it (every kind of paid call) is refused.

Ported from an earlier stand-alone measurement script, unchanged in what it does:
it sees the request body only to hand it on, reads from the reply nothing but the status code and `usage.total_tokens`, never
looks at the headers, counts calls / tokens / statuses / transport exceptions, and once the cap is reached refuses every further
request before it reaches the network. New: every reply's tokens also go into the step's spending table (a `CostMeter`), and the
spending limit stops the next request the same way the token cap does.
"""
import json
import threading
from collections import Counter
from typing import Callable, Optional

from . import paid_gate
from .cost import BudgetExceeded


def estimate_request_tokens(body) -> Optional[int]:
    """An upper bound of the tokens an embedding request can be charged (see the module note), or None when the body cannot be read."""
    try:
        raw = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else str(body)
        texts = json.loads(raw)["input"]
        if isinstance(texts, str):
            texts = [texts]
        return sum(len(t.encode("utf-8")) for t in texts) + 2 * len(texts)
    except Exception:
        return None


class CapReached(Exception):
    """The token cap (or the spending limit) is used up: refuse locally, never touch the network."""


class EmbedMeter:
    def __init__(self, real_transport: Callable, cap_tokens: int, cost_meter=None, price_key: Optional[str] = None,
                 concurrent: bool = False):
        """`concurrent=True` (a concurrency measurement only): the requests are NOT queued behind the paid gate, so they overlap
        like a deployed service's would. The wall is a running sum: the upper bound of a request's tokens is added to
        `bounds_sent` under the lock before it leaves and is never taken off again, so the sum of the bounds of everything sent is at most the
        cap whatever the interleaving. A spending meter needs the serial gate, so it cannot be combined with this."""
        if concurrent and cost_meter is not None:
            raise ValueError("concurrent mode cannot be combined with a spending meter (it needs the serial paid gate)")
        self.cap = cap_tokens
        self._concurrent = concurrent
        self.bounds_sent = 0              # concurrent mode: the sum of the upper bounds of every request sent
        self._closed = False
        self._real = real_transport
        self._cost = cost_meter
        self._gate = cost_meter.gate if cost_meter is not None else paid_gate.PaidGate(None)       # one gate for every paid call of the run
        self._price_key = price_key
        self._lock = threading.Lock()
        self.calls = 0
        self.refused = 0
        self.tokens = 0                   # what the replies charged (or what a reply-less request was kept at)
        self.above_bound = 0              # replies that charged more than the upper bound (the bound was wrong)
        self.charged_without_reply = 0    # tokens kept for requests that raised
        self.without_usage = 0
        self.status = Counter()
        self.exceptions = Counter()
        self.embed_unavailable = 0
        self.embed_invocations = 0        # how many times the app called embedder.embed(...)
        self.embed_texts = 0              # how many texts those invocations carried in total
        self.embed_invocation_sizes = Counter()
        self.cap_reached = threading.Event()

    def transport(self, url, headers, body, timeout):
        est = estimate_request_tokens(body)
        if est is None:                                    # unreadable body: its byte length is still an upper bound of the tokens
            est = len(body) if isinstance(body, (bytes, bytearray)) else len(str(body))
        if self._concurrent:
            return self._transport_concurrent(url, headers, body, timeout, est)
        price_key = self._price_key if (self._cost is not None and self._price_key) else None

        def admit():
            return self.tokens + est <= self.cap

        def go():
            with self._lock:
                self.calls += 1
            try:
                status, raw = self._real(url, headers, body, timeout)
            except Exception as e:                         # timeout / DNS / reset: counted by type; the provider may have charged it
                with self._lock:
                    self.exceptions[type(e).__name__] += 1
                    self.tokens += est
                    self.charged_without_reply += est
                raise
            ok2xx = 200 <= status < 300
            used = None
            if ok2xx:                                      # only a successful reply counts its usage; an error reply was not processed,
                try:                                       # even if it carries a usage block
                    used = int(json.loads(raw)["usage"]["total_tokens"])
                except Exception:
                    used = None
            charged = used if used is not None else (est if ok2xx else 0)
            with self._lock:
                self.status[str(status)] += 1
                self.tokens += charged
                if used is None and ok2xx:
                    self.without_usage += 1
                elif used is None:
                    self.without_usage += 1
                if used is not None and used > est:
                    self.above_bound += 1
            if used is not None and used > est:            # the bound was wrong: close the whole gate
                self._gate.close()
                self.cap_reached.set()
            charge = None
            if ok2xx:
                charge = (used, 0) if used is not None else paid_gate.BOUND
            return (status, raw), charge

        try:
            return self._gate.run(price_key, est, 0, go, admit=admit)
        except BudgetExceeded:
            with self._lock:
                self.refused += 1
            self.cap_reached.set()
            raise CapReached()

    def _transport_concurrent(self, url, headers, body, timeout, est):
        """The concurrent path: add the bound to the running sum under the lock, send OUTSIDE it, book the reply under the lock."""
        with self._lock:
            if self._closed or self.bounds_sent + est > self.cap:
                self.refused += 1
                self.cap_reached.set()
                raise CapReached()
            self.bounds_sent += est
            self.calls += 1
        try:
            status, raw = self._real(url, headers, body, timeout)
        except Exception as e:                             # timeout / DNS / reset: counted by type; the provider may have charged it
            with self._lock:
                self.exceptions[type(e).__name__] += 1
                self.tokens += est
                self.charged_without_reply += est
            raise
        ok2xx = 200 <= status < 300
        used = None
        if ok2xx:                                          # only a successful reply counts its usage (an error reply was not processed)
            try:
                used = int(json.loads(raw)["usage"]["total_tokens"])
            except Exception:
                used = None
        with self._lock:
            self.status[str(status)] += 1
            self.tokens += used if used is not None else (est if ok2xx else 0)
            if used is None:
                self.without_usage += 1
            elif used > est:                               # the bound was wrong: nothing more is sent
                self.above_bound += 1
                self._closed = True
                self.cap_reached.set()
        return status, raw

    def count_invocations(self, embedder):
        """Wrap `embedder.embed` so every invocation, its batch size and an `EmbedUnavailable` is counted. Returns the embedder."""
        from core.embedder import EmbedUnavailable
        real_embed = embedder.embed

        def counting_embed(texts, *a, **kw):
            with self._lock:
                self.embed_invocations += 1
                self.embed_texts += len(texts)
                self.embed_invocation_sizes[len(texts)] += 1
            try:
                return real_embed(texts, *a, **kw)
            except EmbedUnavailable:
                with self._lock:
                    self.embed_unavailable += 1
                raise
        embedder.embed = counting_embed
        return embedder

    def marks(self) -> dict:
        with self._lock:
            return {"http_calls": self.calls, "embed_invocations": self.embed_invocations, "embed_texts": self.embed_texts}

    def snapshot(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "embed_invocations": self.embed_invocations, "embed_texts": self.embed_texts,
                    "embed_invocation_sizes": dict(self.embed_invocation_sizes), "refused_after_cap": self.refused,
                    "tokens_by_replies": self.tokens, "replies_without_usage": self.without_usage,
                    "replies_above_bound": self.above_bound, "tokens_charged_without_reply": self.charged_without_reply,
                    "http_status": dict(self.status), "transport_exceptions": dict(self.exceptions),
                    "embed_unavailable_raised": self.embed_unavailable}
