"""The one gate every paid call goes through: answering, judging, the judge-consistency check, the embedding calls.

One lock; the whole call is made inside it, so at any moment at most one paid call is outside. Inside the lock, in this order:
  1. is "spent + the most this call can cost" within the limit (and does an optional extra check — the embedding token cap — pass)?
     no → refused, nothing is sent;
  2. send it;
  3. book what it really cost (a call that raised, or whose usage cannot be read, is booked at the upper bound; an error reply that
     was not processed is not booked);
  4. if it cost more than its upper bound, the bound was wrong: the gate closes and everything after it is refused;
     if booking itself fails (usage beyond the last price tier, a model nobody priced, a ledger error), the call is booked at its
     upper bound instead, the gate closes, and BudgetExceeded goes up — the books never keep half an entry.
Because only one call is ever outside, no two calls compete for the room that is left, and nothing has to be held back in advance.
The other steps of a run (building contexts, searching, writing results) stay multi-threaded; only the paid step queues.
"""
import threading
from typing import Callable, Optional

from .cost import BudgetExceeded

BOUND = object()            # a charge meaning "book this call at its upper bound" (usage unreadable)


class PaidGate:
    def __init__(self, meter=None):
        self.meter = meter                      # a CostMeter (None = no price table: only the lock and the extra check)
        self._lock = threading.RLock()                # re-entrant: a call inside the gate may close the gate (its bound was wrong)
        self.closed = False
        self.calls = 0
        self.refused = 0
        self.over_bound = 0
        self.booking_errors = 0

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def run(self, price_key: Optional[str], prompt_upper: int, completion_upper: int, fn: Callable,
            admit: Optional[Callable[[], bool]] = None):
        """`fn()` makes the call and returns (value, charge) where charge is None (not charged), BOUND (book at the bound) or
        (prompt_tokens, completion_tokens). Raises BudgetExceeded without calling `fn` when the call is refused."""
        with self._lock:
            if self.closed:
                self.refused += 1
                raise BudgetExceeded("the gate is closed: a call cost more than its upper bound, nothing more is sent")
            bound = 0.0
            if self.meter is not None and price_key:
                bound = self.meter.upper_cost(price_key, prompt_upper, completion_upper)
                if self.meter.spent + bound > self.meter.limit_usd:
                    self.refused += 1
                    raise BudgetExceeded(f"{self.meter.spent:.6f} spent + {bound:.6f} the most this call can cost would pass the "
                                         f"{self.meter.limit_usd:.6f} US dollar limit")
            if admit is not None and not admit():
                self.refused += 1
                raise BudgetExceeded("refused by the extra check (the token cap)")
            self.calls += 1
            try:
                value, charge = fn()
            except BaseException:
                try:
                    self._book(price_key, prompt_upper, completion_upper, bound)   # it may have been processed: booked at the bound
                except BudgetExceeded:
                    pass                                                           # the call's own error is the one to report
                raise
            if charge is not None:
                if charge is BOUND:
                    self._book(price_key, prompt_upper, completion_upper, bound)
                else:
                    self._book(price_key, charge[0], charge[1], bound, bound_tokens=(prompt_upper, completion_upper))
            return value

    def _book(self, price_key, prompt_tokens, completion_tokens, bound_usd, bound_tokens=None) -> None:
        """Book the call. Whatever goes wrong while booking (price cannot be worked out, model not priced, ledger error), the call is
        booked at its upper bound instead, the gate closes, and BudgetExceeded goes up: the money has been spent and the books must say so."""
        if self.meter is None or not price_key:
            return
        try:
            usd = self.meter.add(price_key, prompt_tokens, completion_tokens)
        except Exception as e:
            self.closed = True                               # first: whatever happens next, nothing more is sent
            self.booking_errors += 1
            p_upper, c_upper = bound_tokens if bound_tokens is not None else (prompt_tokens, completion_tokens)
            try:
                self.meter.book_at_bound(price_key, p_upper, c_upper, bound_usd)
            except Exception as e2:                          # not even the bound could be booked: both errors go up with it
                raise BudgetExceeded(f"{price_key}: the call could not be booked from its reply ({type(e).__name__}), and not even at "
                                     f"its upper bound ({type(e2).__name__}); the gate is closed") from e
            raise BudgetExceeded(f"{price_key}: the call could not be booked from its reply ({type(e).__name__}); booked at its "
                                 f"upper bound and the gate is closed") from e
        if usd > bound_usd + 1e-12:                          # it cost more than its bound: the bound was wrong
            self.over_bound += 1
            self.closed = True
