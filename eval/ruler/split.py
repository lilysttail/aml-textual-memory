"""Splitting the ten public conversations into a "tune" part and a "locked" part — by conversation, never by question
(questions of one conversation share one memory, so splitting by question would let tuning peek at the exam) — and how
large a difference the locked part can tell apart.
"""
import hashlib
import itertools
import json
import math
import os
import re
import uuid
from typing import Dict, List, Sequence

Z_ALPHA_2 = 1.959964      # two-sided 5%
Z_BETA = 0.841621         # 80% power
Z = Z_ALPHA_2 + Z_BETA


def per_conversation_counts(corpus) -> Dict[str, Dict[str, int]]:
    cats = sorted({str(q.get("category")) for q in corpus.questions})
    out = {c.sample_id: {"total": 0, **{k: 0 for k in cats}} for c in corpus.conversations}
    for q in corpus.questions:
        row = out[q.get("sample_id")]
        row["total"] += 1
        row[str(q.get("category"))] += 1
    return out


def _share(counts: Dict[str, Dict[str, int]], ids: Sequence[str], cats: Sequence[str]) -> Dict[str, float]:
    total = sum(counts[i]["total"] for i in ids) or 1
    return {c: sum(counts[i][c] for i in ids) / total for c in cats}


def choose_split(counts: Dict[str, Dict[str, int]], lock_size: int) -> dict:
    """The `lock_size` conversations whose category mix is closest to the whole set's. Deterministic: ties go to the
    alphabetically first set."""
    ids = sorted(counts)
    cats = sorted(k for k in counts[ids[0]] if k != "total")
    overall = _share(counts, ids, cats)
    best = None
    for combo in itertools.combinations(ids, lock_size):
        share = _share(counts, combo, cats)
        gap = max(abs(share[c] - overall[c]) for c in cats)
        if best is None or gap < best[0] - 1e-12:
            best = (gap, combo)
    gap, combo = best
    lock = list(combo)
    tune = [i for i in ids if i not in combo]
    return {"lock": lock, "tune": tune, "lock_questions": sum(counts[i]["total"] for i in lock),
            "tune_questions": sum(counts[i]["total"] for i in tune),
            "lock_by_category": {c: sum(counts[i][c] for i in lock) for c in cats},
            "category_share_gap": gap}


def min_detectable_difference_unpaired(n: int, p: float) -> float:
    """Two versions each answering the same n questions, compared as if independent (conservative): the difference in accuracy
    that is told apart from noise with 80% power at a two-sided 5% level."""
    return Z * math.sqrt(2 * p * (1 - p) / n)


def min_detectable_difference_paired(n: int, discordant_fraction: float) -> float:
    """Same questions, same conversations: only the questions where the two versions disagree carry information
    (McNemar). `discordant_fraction` = share of questions where exactly one version is right."""
    return Z * math.sqrt(discordant_fraction / n)


def design_effect(rows: Sequence[dict]) -> float:
    """1 + (m - 1) * ICC for a 0/1 outcome (`hit`) grouped by `group` (one-way ANOVA estimate, clamped to [0, 1]): how much the
    effective number of questions shrinks because questions of one conversation move together."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(str(r.get("group")), []).append(1.0 if r.get("hit") else 0.0)
    k = len(groups)
    n = sum(len(v) for v in groups.values())
    if k < 2 or n <= k:
        return 1.0
    grand = sum(sum(v) for v in groups.values()) / n
    ssb = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in groups.values())
    ssw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in groups.values())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - sum(len(v) ** 2 for v in groups.values()) / n) / (k - 1)
    denom = msb + (m0 - 1) * msw
    icc = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    return 1.0 + (n / k - 1) * icc


# ---- The fixed cut (decided 2026-10-23, option C) ----
#
# Five conversations are LOCKED and used once, at the very end; the other five are for tuning. Chosen by the category mix in
# `choose_split(counts, 5)`; written down here so nobody re-cuts it after looking at a result.

LOCK = ("conv-26", "conv-41", "conv-42", "conv-47", "conv-50")
TUNE = ("conv-30", "conv-43", "conv-44", "conv-48", "conv-49")
LOCK_QUESTIONS = 734
TUNE_QUESTIONS = 648

# Whether to turn the meaning-based (dense) path on is decided ONCE, on the locked part, question by question:
# turn it on only if the overall gain is at least 4 points AND at least 3 of the 5 locked conversations are better with it.
# Otherwise keyword-only. Fixed now; not to be changed before the locked part has been looked at.
DENSE_MIN_OVERALL_GAIN_POINTS = 4
DENSE_MIN_SEGMENTS_BETTER = 3
# Step 4 (plan v3): with several answering models, a third gate — no single model may do worse with the meaning-based path by more
# than 3 points (its questions' 3 percent: at most 22 of 734 fewer correct, 23 fewer fails; integer arithmetic, see decide_dense).
DENSE_MAX_MODEL_LOSS_POINTS = 3


def check_partition(counts: Dict[str, Dict[str, int]], lock_questions: int = LOCK_QUESTIONS,
                    tune_questions: int = TUNE_QUESTIONS) -> None:
    """Refuse a corpus that is not the one the cut was made on: a conversation missing or extra, or a question count that moved."""
    want = set(LOCK) | set(TUNE)
    if set(counts) != want:
        raise ValueError("the conversations are not the ten the cut was made on")
    got_lock = sum(counts[c]["total"] for c in LOCK)
    got_tune = sum(counts[c]["total"] for c in TUNE)
    if (got_lock, got_tune) != (lock_questions, tune_questions):
        raise ValueError("the question counts of the locked/tuned parts are not the ones the cut was made on")


def decide_dense(base: Sequence[dict], dense: Sequence[dict], lock: Sequence[str] = LOCK, models: Sequence[str] = None) -> dict:
    """The one-time decision on the locked part. `base` and `dense` are per-question rows {item_id, group, correct} of the two
    versions on exactly the locked conversations and exactly the same questions; a row may carry a `model` (the answering
    model; rows without one are a single model). Integer arithmetic throughout. Three gates, all must hold:
      1. all models pooled, the meaning-based version answers at least DENSE_MIN_OVERALL_GAIN_POINTS percent of all rows more correctly;
      2. at least DENSE_MIN_SEGMENTS_BETTER of the locked conversations are better, the models pooled inside each conversation;
      3. no single model answers more than DENSE_MAX_MODEL_LOSS_POINTS percent of its own questions fewer correctly.
    `models`, when given, is the declared list: the rows must hold exactly those models."""
    def index(rows, what):
        out = {}
        for r in rows:
            key = (r.get("model"), r["item_id"])
            if key in out:
                raise ValueError(f"{what}: the same question twice")
            out[key] = r
        return out

    b, d = index(base, "base"), index(dense, "dense")
    if set(b) != set(d):
        raise ValueError("the two versions were not asked the same questions")
    names = sorted({m for m, _ in b}, key=lambda m: (m is None, str(m)))
    if models is not None and set(names) != set(models):
        raise ValueError("the rows do not hold exactly the declared models")
    per_model_ids = {m: {i for mm, i in b if mm == m} for m in names}
    if len({frozenset(s) for s in per_model_ids.values()}) != 1:
        raise ValueError("the models were not asked the same questions")
    if {r["group"] for r in b.values()} != set(lock) or any(b[k]["group"] != d[k]["group"] for k in b):
        raise ValueError("the rows are not exactly the locked conversations")
    segments, better = {}, 0
    for g in lock:
        by_model, kb, kd, n_seg = {}, 0, 0, 0
        for m in names:
            ids = [i for i in per_model_ids[m] if b[(m, i)]["group"] == g]
            mb = sum(1 for i in ids if b[(m, i)]["correct"])
            md = sum(1 for i in ids if d[(m, i)]["correct"])
            by_model[m] = {"questions": len(ids), "base_correct": mb, "dense_correct": md}
            kb, kd, n_seg = kb + mb, kd + md, n_seg + len(ids)
        segments[g] = {"questions": n_seg, "base_correct": kb, "dense_correct": kd, "better": kd > kb, "by_model": by_model}
        better += 1 if kd > kb else 0
    models_out, loss_ok_all = {}, True
    for m in names:
        q = len(per_model_ids[m])
        mb = sum(1 for i in per_model_ids[m] if b[(m, i)]["correct"])
        md = sum(1 for i in per_model_ids[m] if d[(m, i)]["correct"])
        loss_ok = (mb - md) * 100 <= DENSE_MAX_MODEL_LOSS_POINTS * q            # fewer correct than this fails: 22 of 734 passes, 23 does not
        loss_ok_all = loss_ok_all and loss_ok
        models_out[m] = {"questions": q, "base_correct": mb, "dense_correct": md, "net": md - mb, "net_points": 100 * (md - mb) / q, "loss_ok": loss_ok}
    n = len(b)
    gain = sum(s["dense_correct"] - s["base_correct"] for s in segments.values())
    overall_ok = gain * 100 >= DENSE_MIN_OVERALL_GAIN_POINTS * n
    on = overall_ok and better >= DENSE_MIN_SEGMENTS_BETTER and loss_ok_all
    return {"dense_on": on, "questions": n, "overall_gain_questions": gain, "overall_gain_points": 100 * gain / n,
            "gate_overall_ok": overall_ok, "gate_segments_ok": better >= DENSE_MIN_SEGMENTS_BETTER, "gate_model_loss_ok": loss_ok_all,
            "segments_better": better, "segments": segments, "models": models_out,
            "rule": {"min_overall_gain_points": DENSE_MIN_OVERALL_GAIN_POINTS, "min_segments_better": DENSE_MIN_SEGMENTS_BETTER,
                     "max_model_loss_points": DENSE_MAX_MODEL_LOSS_POINTS}}


# ---- The locked part is used once ----

class LockedSetAlreadyUsed(Exception):
    """The locked part was already used for this version (or the pair differs, or the ledger cannot be read)."""


_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _create_exclusive(path: str, text: str = "") -> None:
    """Create `path` and fail with FileExistsError if it is already there — the one operation every claim rests on."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _declare(ledger_dir: str, declared: Sequence[str]) -> None:
    """Fix the pair of versions, once. The file appears complete or not at all (written aside, then hard-linked into place;
    the link fails if the name is taken), so a process that loses the race reads a whole declaration."""
    final = os.path.join(ledger_dir, "declared.json")
    tmp = os.path.join(ledger_dir, f".declared.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    _create_exclusive(tmp, json.dumps({"declared": sorted(declared)}))
    try:
        try:
            os.link(tmp, final)
            return
        except FileExistsError:
            pass
    finally:
        os.remove(tmp)
    try:
        with open(final, encoding="utf-8") as f:
            doc = json.load(f)
        same = sorted(doc["declared"]) == sorted(declared)
    except (OSError, ValueError, KeyError, TypeError):
        raise LockedSetAlreadyUsed("the declaration in the ledger is unreadable")
    if not same:
        raise LockedSetAlreadyUsed("the ledger already holds a different pair of versions")


def claim_locked(ledger_dir: str, version: str, declared: Sequence[str]) -> None:
    """Record that `version` is about to be run on the locked part. The ledger is a directory: `declared.json` (the pair of
    versions, made once by the first claim) and one `used.<version>` file per version, each made with an exclusive create, so
    two processes claiming at the same moment cannot both succeed and different versions cannot overwrite each other. A run
    that stops half-way still counts — looking at part of the locked set is looking. Only version labels are stored.
    Re-opening takes a person, in the floor thread."""
    declared = list(declared)
    if (len(declared) != 2 or version not in declared or not all(_LABEL.match(v or "") for v in declared)
            or len({v.lower() for v in declared}) != 2):
        raise ValueError("a claim declares exactly two distinct versions (plain file names) and names one of them")
    try:
        os.makedirs(ledger_dir, exist_ok=True)
    except (FileExistsError, NotADirectoryError):
        raise LockedSetAlreadyUsed("the ledger is not a directory")
    _declare(ledger_dir, declared)
    try:
        _create_exclusive(os.path.join(ledger_dir, f"used.{version.lower()}"), version)
    except FileExistsError:
        raise LockedSetAlreadyUsed(f"the locked part was already used for {version}")


# ---- The locked part is used once, by a whole plan (step 4: several models, a fixed order, three gates) ----

def _canonical(plan: dict) -> str:
    return json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


def claim_locked_models(ledger_dir: str, plan: dict) -> dict:
    """Claim the locked part for a whole plan: the two versions, the models and their identities, the run order, the gates and who is
    asked first on every question (`plan`, a JSON-able dict). `claim.json` is made once, the way `declared.json` is (written aside, then
    hard-linked into place; the link fails if the name is taken), `claim.sha256` and one `used.<version>` file per version follow. A plan
    exactly equal to the claimed one is a RESUME (status "resumed", nothing written); any difference — another model, another order, another
    gate, another first-version table — is refused. A version already used by an earlier claim refuses a new plan. The caller makes this
    call before it writes anything else. Re-opening takes a person, in the floor thread."""
    versions = plan.get("versions") if isinstance(plan, dict) else None
    if (not isinstance(versions, list) or len(versions) != 2 or not all(isinstance(v, str) and _LABEL.match(v) for v in versions)
            or len({v.lower() for v in versions}) != 2):
        raise ValueError("a plan names exactly two distinct versions (plain file names)")
    if not isinstance(plan.get("models"), list) or not plan["models"]:
        raise ValueError("a plan names at least one model")
    text = _canonical(plan)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        os.makedirs(ledger_dir, exist_ok=True)
    except (FileExistsError, NotADirectoryError):
        raise LockedSetAlreadyUsed("the ledger is not a directory")
    final = os.path.join(ledger_dir, "claim.json")
    if not os.path.exists(final):
        for v in versions:
            # `claim.json` is linked into place before any used.<version> is made, so a used file without a claim file is an older claim
            if os.path.exists(os.path.join(ledger_dir, f"used.{v.lower()}")) and not os.path.exists(final):
                raise LockedSetAlreadyUsed(f"the locked part was already used for {v}")
    if not os.path.exists(final):
        tmp = os.path.join(ledger_dir, f".claim.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        _create_exclusive(tmp, text)
        try:
            try:
                os.link(tmp, final)
                won = True
            except FileExistsError:
                won = False
        finally:
            os.remove(tmp)
        if won:
            for name, body in (("claim.sha256", digest + "\n"),) + tuple((f"used.{v.lower()}", v) for v in versions):
                try:
                    _create_exclusive(os.path.join(ledger_dir, name), body)
                except FileExistsError:
                    pass                                   # a racing process that read the winner's claim never writes these
            return {"status": "claimed", "claim_sha256": digest}
    try:
        with open(final, encoding="utf-8", newline="") as f:
            held = f.read()
    except OSError:
        raise LockedSetAlreadyUsed("the claim in the ledger is unreadable")
    if held != text:
        raise LockedSetAlreadyUsed("the ledger already holds a different plan")
    return {"status": "resumed", "claim_sha256": digest}


def acquire_run_lock(ledger_dir: str) -> str:
    """One run at a time on the locked part: an exclusive-create lock file with the process id. A run that crashed leaves it; a person
    looks (is that process really gone?) and removes it in the floor thread — the code never removes a lock it did not make."""
    path = os.path.join(ledger_dir, "run.lock")
    try:
        _create_exclusive(path, f"{os.getpid()} {uuid.uuid4().hex}")
    except FileExistsError:
        raise LockedSetAlreadyUsed("another run holds the lock on the locked part (or a crashed run left run.lock; a person decides)")
    return path


def release_run_lock(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
