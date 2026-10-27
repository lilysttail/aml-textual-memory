"""What the model calls cost, from the token counts the API itself reported (step 2).

Prices are US dollars per million tokens (input, output). A price may also be a function of the request's prompt
token count returning (input, output) — for the tiered price lists. A model whose price is `None` is paid for out of a plan's
quota, not by price: its tokens are counted and the meter charges nothing for it. Usage the reply did not report is an error — it is never
guessed, because the spending limit rests on these numbers.

Every paid call goes through `CostMeter.gate` (see `paid_gate.py`): one lock, the whole call made inside it, so the limit is a wall.
"""
import threading
from typing import Callable, Dict, Optional, Tuple, Union

Price = Union[None, Tuple[float, float], Callable[[int], Tuple[float, float]]]


def tiered(*tiers: Tuple[int, float, float]) -> Callable[[int], Tuple[float, float]]:
    """tiered((32_000, 0.03, 0.13), (256_000, 0.10, 0.40)): the whole request is priced at the first tier whose limit covers its
    prompt tokens (the way the list prices say it); a request above the last limit is an error."""
    def price(prompt_tokens: int) -> Tuple[float, float]:
        for limit, p_in, p_out in tiers:
            if prompt_tokens <= limit:
                return p_in, p_out
        raise ValueError("the request is larger than the largest priced tier")
    return price


class BudgetExceeded(Exception):
    pass


class UsageMissing(Exception):
    pass


class CostMeter:
    def __init__(self, prices: Dict[str, Price], limit_usd: float):
        self.prices = prices
        self.limit_usd = limit_usd
        self.spent = 0.0
        self.tokens: Dict[str, dict] = {}
        self._usd: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._gate = None

    @property
    def gate(self):
        """The one gate every paid call of this meter goes through (created once, shared by answering, judging and the vector calls)."""
        with self._lock:
            if self._gate is None:
                from .paid_gate import PaidGate
                self._gate = PaidGate(self)
            return self._gate

    def check(self) -> None:
        """Called before each question: refuses once the limit has been reached."""
        with self._lock:
            if self.spent >= self.limit_usd:
                raise BudgetExceeded(f"spent {self.spent:.4f} of the {self.limit_usd:.2f} US dollar limit")

    def upper_cost(self, model: str, prompt_tokens_upper: int, completion_tokens_upper: int = 0) -> float:
        """The most a call can cost, in dollars, from upper bounds of its token counts (tiered prices only go up with the count, so
        the bound's tier is the highest the call can land in). A model paid by plan quota costs nothing. A request above the
        largest priced tier is refused."""
        price = self.prices[model]                          # KeyError for a model nobody priced
        if price is None:
            return 0.0
        try:
            p_in, p_out = price(prompt_tokens_upper) if callable(price) else price
        except ValueError as e:
            raise BudgetExceeded(f"{model}: {e}") from e
        return (prompt_tokens_upper * p_in + completion_tokens_upper * p_out) / 1_000_000

    def add(self, model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> float:
        """Book one call; returns what it cost in dollars."""
        price = self.prices[model]                          # KeyError for a model nobody priced
        if prompt_tokens is None or completion_tokens is None:
            raise UsageMissing(f"{model}: the reply reported no token usage")
        usd = 0.0
        if price is not None:                               # work the price out first: if it cannot be, the ledger is not touched
            p_in, p_out = price(prompt_tokens) if callable(price) else price
            usd = (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000
        with self._lock:
            self._write(model, prompt_tokens, completion_tokens, usd)
        return usd

    def book_at_bound(self, model: str, prompt_tokens_upper: int, completion_tokens_upper: int, usd: float) -> None:
        """Book a call at its upper bound when what it really cost cannot be worked out. Computes nothing, so it cannot fail on a price."""
        with self._lock:
            self._write(model, prompt_tokens_upper, completion_tokens_upper, usd)

    def _write(self, model: str, prompt_tokens: int, completion_tokens: int, usd: float) -> None:
        row = self.tokens.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        row["prompt"] += prompt_tokens
        row["completion"] += completion_tokens
        row["calls"] += 1
        if self.prices.get(model) is not None:
            self.spent += usd
            self._usd[model] = self._usd.get(model, 0.0) + usd

    def spent_by_model(self) -> Dict[str, float]:
        return {m: self._usd.get(m, 0.0) for m in self.tokens}

    def state(self) -> dict:
        """What has been spent so far, to carry over to the next run (the limit is for the whole step, not per run)."""
        with self._lock:
            return {"spent": self.spent, "tokens": {m: dict(r) for m, r in self.tokens.items()}, "usd": dict(self._usd)}

    def restore(self, state: dict) -> None:
        with self._lock:
            self.spent = float(state.get("spent", 0.0))
            self.tokens = {m: dict(r) for m, r in state.get("tokens", {}).items()}
            self._usd = {m: float(v) for m, v in state.get("usd", {}).items()}
