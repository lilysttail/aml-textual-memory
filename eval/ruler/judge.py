"""The ruler's block ④ — the judgement: right / said "not found" / made it up.

The judge is the official LoCoMo-Refined one, copied with its source:

    mem-eval-suite/LoCoMo_refined @ 887091190789e8d6760e70b9edd696539923dc4f  (CC BY-NC 4.0; a modified version of
    snap-research/locomo, see THIRD_PARTY_NOTICES.md and the repository's NOTICE)
      - the `refined` prompt            src/llm_judge.py      JUDGE_PROMPTS["refined"]  -> refined_judge_prompt.txt (byte for byte)
      - the request and the label read  src/llm_judge_runtime.py   one user message, temperature 0, thinking off, JSON "label",
                                        a list of gold answers: judged one by one, CORRECT if any is
      - the surrounding rules           src/evaluate.py       exact match with a gold answer = 1 without a model, an empty
                                        prediction = 0 without a model, the prediction cleaned first (\\boxed{}, "final answer:",
                                        "</think>")
Our version differs only in what carries it: the model call goes through `chat.ChatClient` (standard library, usage reported
back), a reply that cannot be read is retried and then an error — never a silent WRONG — and a list of gold answers stops at
the first CORRECT (same score, fewer tokens). The official `scripts/run_eval.sh` is run once on a sample to see that the two agree.
"""
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Sequence

from .answerer import is_not_found
from .chat import ChatError, ChatResult

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refined_judge_prompt.txt"), encoding="utf-8", newline="") as _f:
    REFINED_PROMPT = _f.read()
REFINED_PROMPT_SHA256 = "899d035bd61813bc718812b2997f5908111c63ccd2ed76533e65f5dc3ca337a4"

OFFICIAL_JUDGE_MODEL = "qwen3-14b"


class Verdict(Enum):
    CORRECT = "correct"
    NOT_FOUND = "not_found"
    MADE_UP = "made_up"


class Judge(Protocol):
    def judge(self, item, predicted: str) -> Verdict:
        """Compare what was said with what is expected."""


class JudgeError(Exception):
    """The judge could not produce a label (the call failed, or the reply had none after the retries)."""


@dataclass
class JudgeResult:
    score: int
    shortcut: Optional[str] = None                  # "exact" / "empty" when no model was asked
    results: List[ChatResult] = field(default_factory=list)     # every model reply, for the usage and the spending meter


# ---- official helpers (src/evaluate.py, src/llm_judge_runtime.py), copied ----

def _strip_code_fence(content: str) -> str:
    stripped = (content or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text: str) -> str:
    stripped = _strip_code_fence(text).strip()
    in_string = False
    escape = False
    depth = 0
    start: Optional[int] = None
    for index, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return stripped[start: index + 1]
    raise ValueError("failed to extract JSON object from judge response")


def _extract_braced_content(text: str, opening_brace_index: int) -> Optional[str]:
    if opening_brace_index < 0 or opening_brace_index >= len(text) or text[opening_brace_index] != "{":
        return None
    depth = 0
    content_start = opening_brace_index + 1
    for index in range(opening_brace_index, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
            continue
        if ch != "}":
            continue
        depth -= 1
        if depth == 0:
            return text[content_start:index]
    return None


def _extract_last_boxed_value(text: str) -> Optional[str]:
    last_value: Optional[str] = None
    for match in re.finditer(r"\\box(?:ed)?\{", text):
        extracted = _extract_braced_content(text, match.end() - 1)
        if extracted is not None:
            last_value = extracted.strip()
    return last_value


def normalize_prediction(text: str) -> str:
    cleaned = str(text or "").strip()
    boxed_value = _extract_last_boxed_value(cleaned)
    if boxed_value is not None:
        return boxed_value
    lower = cleaned.lower()
    if "final answer:" in lower:
        idx = lower.index("final answer:")
        cleaned = cleaned[idx + len("final answer:"):].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def normalize_candidates(raw) -> List[str]:
    cands: List[str] = []
    if isinstance(raw, (list, tuple)):
        cands.extend(str(c) for c in raw if c not in (None, ""))
    elif raw not in (None, ""):
        cands.append(str(raw))
    return cands or [""]


class RefinedJudge:
    def __init__(self, client, parse_retries: int = 2):
        self.client = client
        self.parse_retries = parse_retries

    def _label(self, question: str, gold: str, prediction: str, results: List[ChatResult]) -> int:
        prompt = REFINED_PROMPT.format(question=question, gold_answer=gold, generated_answer=prediction)
        last = None
        for _ in range(self.parse_retries + 1):
            try:
                r = self.client.complete(prompt)
            except ChatError as e:
                raise JudgeError(f"the judge call failed ({e})")
            results.append(r)
            try:
                payload = json.loads(extract_json_object(r.text))
                return 1 if str(payload.get("label", "")).strip().upper() == "CORRECT" else 0
            except (ValueError, AttributeError) as e:
                if r.finish_reason == "length":         # cut by the length cap before a label: asking again would be cut the same way
                    raise JudgeError("the judge reply was cut by the length cap before a label")
                last = e
        raise JudgeError(f"the judge reply had no readable label ({type(last).__name__})")

    def score(self, question: str, golds, predicted: str) -> JudgeResult:
        prediction = normalize_prediction(predicted)
        candidates = normalize_candidates(golds)
        if any(prediction == c for c in candidates):
            return JudgeResult(1, "exact")
        if prediction == "" and any(c != "" for c in candidates):
            return JudgeResult(0, "empty")
        if all(c == "" for c in candidates):
            return JudgeResult(0, "empty")
        results: List[ChatResult] = []
        for c in candidates:
            if self._label(question, c, prediction, results) == 1:
                return JudgeResult(1, None, results)
        return JudgeResult(0, None, results)

    def judge(self, item, predicted: str) -> Verdict:
        if is_not_found(predicted):
            return Verdict.NOT_FOUND
        return Verdict.CORRECT if self.score(item.input, item.expected_answers, predicted).score == 1 else Verdict.MADE_UP


class NotConnectedJudge:
    def judge(self, item, predicted: str) -> Verdict:
        raise NotImplementedError("no judge is connected")
