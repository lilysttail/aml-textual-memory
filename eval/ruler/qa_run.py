"""Step 2 — ask a model, judge the answer, keep one row per (group, model, question) (local files; no text in the floor).

A row holds ids, numbers, the outcome and the model's own answer (the answer stays in the local results directory, never in the
floor and never in the repository). It does not hold the question or the memories. Rows already on disk are not asked again
after a restart. The spending limit is checked before each question; a question that has started is finished, so the limit can be
passed by at most one question's worth.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

from . import answerer, chat, cost, judge, sampling


RETRY_OUTCOMES = ("api_error", "judge_error")


def stage_of(group: sampling.Group, ctx: sampling.Context) -> str:
    """Where the standard evidence stands for the memories this group hands over."""
    if group.mode == "evidence":
        return "evidence_in_prompt"
    if ctx.found_100 == 0:
        return "not_returned"
    if group.limit is not None and group.limit < 100 and ctx.found_20 == 0:
        return "returned_not_in_prompt"
    return "evidence_in_prompt"


def _found_in_prompt(group: sampling.Group, ctx: sampling.Context) -> int:
    if group.mode == "evidence":
        return len(ctx.evidence_lines)
    return ctx.found_20 if group.limit is not None and group.limit < 100 else ctx.found_100


class Runner:
    def __init__(self, corpus, contexts: Dict[str, Dict[str, sampling.Context]], meter: cost.CostMeter,
                 judge: judge.RefinedJudge, judge_price_key: str, results_path: str, variant: str = "keyword", stem_english: bool = False):
        self.questions = {q["qa_id"]: q for q in corpus.questions}
        self.contexts = contexts                      # shape -> qa_id -> Context
        self.meter = meter
        self.judge = judge
        self.judge_price_key = judge_price_key
        self.results_path = results_path
        self.variant = variant                        # which search path the memories came from: keyword or dense
        self.stem_english = stem_english              # whether the keyword path matched English word forms in this run
        self._lock = threading.Lock()

    # ---- the rows on disk ----

    def _load(self) -> Dict[tuple, dict]:
        out = {}
        try:
            with open(self.results_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out[(r["group"], r["model_key"], r["qa_id"])] = r      # the later row replaces an earlier one
        except FileNotFoundError:
            pass
        for key in [k for k, r in out.items() if r.get("outcome") in RETRY_OUTCOMES]:
            del out[key]                                                       # these are asked again
        return out

    def _append(self, row: dict) -> None:
        with self._lock:
            with open(self.results_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- one question in one group by one model ----

    def _one(self, model_key: str, ask: Callable, price_key: str, group: sampling.Group, qa_id: str) -> dict:
        row, reply = self._ask(model_key, ask, price_key, group, qa_id)
        return row if reply is None else self._judge(row, qa_id, reply)

    def _ask(self, model_key: str, ask: Callable, price_key: str, group: sampling.Group, qa_id: str):
        """Ask the model one question. Returns (row, reply); reply is None when the row is already final (the call failed, or the model
        used a tool and the row is void). The step-4 locked run asks both versions of a question, then judges both, so asking and
        judging are separate calls (`_one` is the two in a row)."""
        q = self.questions[qa_id]
        ctx = self.contexts[group.shape][qa_id]
        memories = sampling.memories_for(group, ctx)
        row = {"group": group.name, "model_key": model_key, "qa_id": qa_id, "category": str(q.get("category")), "search": self.variant, "stem_english": self.stem_english,
               "memories_n": len(memories), "n_gold": ctx.n_gold, "found_100": ctx.found_100, "found_20": ctx.found_20,
               "found_in_prompt": _found_in_prompt(group, ctx), "stage": stage_of(group, ctx)}
        try:
            reply = ask(q["question"], memories)
        except chat.ChatError as e:
            return dict(row, answer="", outcome="api_error", score=None, error_status=e.status, error_code=e.code), None
        if not getattr(reply, "metered", False):          # a metered client has already booked this call (it held the room first)
            self.meter.add(price_key, reply.prompt_tokens, reply.completion_tokens)
        row.update(answer=reply.text, prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                   model_returned=reply.model, finish_reason=reply.finish_reason, attempts=reply.attempts)
        tool_calls = reply.extra.get("tool_calls")
        if reply.extra:
            row["extra"] = {k: v for k, v in reply.extra.items() if k != "text"}
        if tool_calls:
            return dict(row, outcome="void", score=None, tool_calls=tool_calls), None
        if tool_calls is not None:
            row["tool_calls"] = tool_calls
        return row, reply

    def _judge(self, row: dict, qa_id: str, reply) -> dict:
        """Judge the answer of a row `_ask` made; returns the finished row."""
        q = self.questions[qa_id]
        score, shortcut, jr = None, None, []
        if not answerer.is_not_found(reply.text) and reply.text.strip():
            try:
                res = self.judge.score(q["question"], q.get("answer"), reply.text)
                score, shortcut, jr = res.score, res.shortcut, res.results
            except judge.JudgeError as e:
                row["judge_error"] = str(e)[:80]
                jr = []
        jp = jc = 0
        for r in jr:
            if not getattr(r, "metered", False):
                self.meter.add(self.judge_price_key, r.prompt_tokens, r.completion_tokens)
            jp += r.prompt_tokens or 0
            jc += r.completion_tokens or 0
        row.update(score=score, shortcut=shortcut, judge_calls=len(jr), judge_prompt_tokens=jp, judge_completion_tokens=jc,
                   outcome=answerer.outcome(reply.text, score))
        return row

    def run_model(self, model_key: str, ask: Callable, price_key: str, group_names: Sequence[str], qa_ids: Sequence[str],
                  workers: int = 1) -> List[dict]:
        done = self._load()
        units = [(sampling.GROUPS[g], qid) for g in group_names for qid in qa_ids]
        stop = threading.Event()

        def work(unit):
            group, qid = unit
            key = (group.name, model_key, qid)
            if key in done:
                return done[key]
            if stop.is_set():
                return None
            try:
                self.meter.check()
            except cost.BudgetExceeded:
                stop.set()
                return None
            try:
                row = self._one(model_key, ask, price_key, group, qid)
            except cost.BudgetExceeded:                    # the wall refused a call: stop, and say how far the run got
                stop.set()
                return None
            self._append(row)
            return row

        if workers <= 1:
            rows = [work(u) for u in units]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(work, units))
        if stop.is_set():
            raise cost.BudgetExceeded(f"stopped at the spending limit after {sum(1 for r in rows if r)} of {len(units)}")
        return rows
