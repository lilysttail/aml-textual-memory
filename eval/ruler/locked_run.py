"""Step 4 (plan v3): the one-time run on the locked conversations that decides whether the meaning-based path goes on.

What is here, in the order a run does it:
  1. the arguments are checked and the plan is worked out (nothing is written);
  2. the claim (`split.claim_locked_models`): models and their identities, run order, the three gates, and which version is asked first on
     every question — written once; the same plan again is a resume; any other plan is refused. Nothing else is written before it;
  3. both versions of the retrieval are made once and kept in files (`retrieval_kw.jsonl`, `retrieval_dense.jsonl`);
  4. per model, in the claimed order, per question: ask the first version, ask the second, judge the first, judge the second (the first one
     is fixed by the question id's hash), and write both rows together. A pair with any failed ask / judge / void row is not written; after the
     model's pass such pairs are asked again as whole pairs once; a pair that still fails stops the run as unfinished and the next model is
     not started. A stop at the spending limit leaves only whole pairs and is resumed on the same claim;
  5. the report: each model alone once it is whole, the decision (`split.decide_dense`, three gates) only when all models are whole.
Rows, retrieval and progress files stay in the local `--out` directory (answers and memories are text); the report has numbers and ids only.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Callable, Dict, List, Optional, Sequence

from . import answerer, chat, claude_cli, cost, judge, qa_run, sampling, split, target

PLAN_SHA16 = "7b8379aca7a46d45"        # sha256 (first 16) of the plan document v3 this code implements (the plan document, kept privately)
VERSIONS = ("kw", "dense")             # kw = keyword only, dense = keyword + meaning-based; both with neighbours, shape c, 100 memories, stemming off
CANDIDATE = "dense"                    # the version the gates ask about; the other is the base
GROUP = "G2"
GOOD_OUTCOMES = ("correct", "not_found", "made_up")


class LockedRunIncomplete(Exception):
    """A pair kept failing after its second try: the model is unfinished, the gates are not worked out, the next model is not started."""


def first_version(qa_id: str) -> str:
    """Which version is asked first on this question: fixed by the same hash step 2 used to pick its 48 questions (even -> kw first)."""
    return "kw" if int(sampling._rank(qa_id), 16) % 2 == 0 else "dense"


def locked_questions(corpus, lock: Sequence[str]) -> List[dict]:
    return [q for q in corpus.questions if q.get("sample_id") in set(lock)]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _save_json(path: str, doc) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---- who answers: the identity of each model, as far as it can be pinned ----

def _claude_version() -> str:
    exe = shutil.which("claude") or "claude"
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60, env=dict(os.environ, **claude_cli.CHILD_ENV))
        return (r.stdout or r.stderr).strip()[:80] or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def known_models() -> List[str]:
    from . import qa_main
    return ["main", "haiku"] + sorted(qa_main.OPENCODE_MODELS)


def model_identity(who: str, prior: Optional[dict] = None) -> dict:
    """What can be written down about this answering model: what is pinned, what is only an alias, which gateway / CLI, the settings."""
    from . import qa_main
    if who == "main":
        c = qa_main.main_config()
        return {"key": "main", "provider": "dashscope", "base_url": c.base_url, "id": c.model, "pinned": True,
                "settings": {"temperature": c.temperature, "max_tokens": c.max_tokens, "extra_body": c.extra_body}}
    if who in qa_main.OPENCODE_MODELS:
        c = qa_main.opencode_config(who, session_id=(prior or {}).get("session_header"))
        return {"key": who, "provider": "opencode-go", "gateway": c.base_url, "alias": who, "api_style": c.api_style, "pinned": False,
                "session_header": c.headers["x-opencode-session"], "user_agent": c.user_agent,
                "settings": {"temperature": c.temperature, "max_tokens": c.max_tokens, "extra_body": c.extra_body}}
    if who == "haiku":
        return {"key": "haiku", "provider": "claude-cli", "cli": "claude -p", "cli_version": _claude_version(), "model_arg": qa_main.HAIKU_MODEL,
                "pinned": "half", "settings": {"tools": "none", "settings_sources": "none", "session_persistence": False}}
    raise ValueError(f"{who!r} is not an answering model of this run: {known_models()}")


def build_plan(questions: Sequence[dict], lock: Sequence[str], models: Sequence[dict], declared: Sequence[str], candidate: str) -> dict:
    from . import qa_main
    per_segment: Dict[str, int] = {g: 0 for g in lock}
    for q in questions:
        per_segment[q["sample_id"]] += 1
    probe = answerer.build_answer_prompt(["<memory>"], "<question>")
    return {
        "plan_sha16": PLAN_SHA16, "versions": list(declared), "candidate": candidate,
        "version_defs": {"kw": {"recall_paths": "keyword", "stem_english": False, "neighbours": True, "top_k": sampling.GROUPS[GROUP].limit, "shape": "c",
                                "group": GROUP},
                         "dense": {"recall_paths": "keyword,dense", "embed_provider": "dashscope", "rerank": False, "stem_english": False,
                                   "neighbours": True, "top_k": sampling.GROUPS[GROUP].limit, "shape": "c", "group": GROUP}},
        "partition": {"lock": list(lock), "questions": len(questions), "per_segment": per_segment},
        "models": [dict(m) for m in models],
        "judge": {"model": qa_main.JUDGE_MODEL, "prompt_sha256": judge.REFINED_PROMPT_SHA256, "max_tokens": qa_main.JUDGE_MAX_TOKENS,
                  "temperature": 0.0, "pinned": False},
        "answer_prompt_sha256": hashlib.sha256(probe.encode("utf-8")).hexdigest(),
        "gates": {"min_overall_gain_points": split.DENSE_MIN_OVERALL_GAIN_POINTS, "min_segments_better": split.DENSE_MIN_SEGMENTS_BETTER,
                  "max_model_loss_points": split.DENSE_MAX_MODEL_LOSS_POINTS},
        "question_order": [q["qa_id"] for q in questions],
        "first_version": {q["qa_id"]: first_version(q["qa_id"]) for q in questions},
    }


# ---- the retrieval, once, in files ----

def search_contexts(corpus, qa_ids: Sequence[str], version: str, dense_cfg) -> Dict[str, sampling.Context]:
    """Write the conversations through the service in the version's setting and search every question (100 memories, shape c)."""
    if version == "dense" and dense_cfg is None:
        raise ValueError("the dense retrieval needs a dense configuration")
    return sampling.build_contexts(corpus, qa_ids, shape="c", top_k=sampling.GROUPS[GROUP].limit, dense=dense_cfg if version == "dense" else None,
                                   stem_english=False)


def _retrieval_path(out_dir: str, version: str) -> str:
    return os.path.join(out_dir, f"retrieval_{version}.jsonl")


def ensure_retrieval(corpus, out_dir: str, version: str, questions: Sequence[dict], dense_cfg, progress: dict) -> None:
    path = _retrieval_path(out_dir, version)
    if os.path.exists(path):
        have = _sha16(path)
        recorded = progress.setdefault("retrieval", {}).get(version)
        if recorded is not None and recorded != have:
            raise ValueError(f"retrieval_{version}.jsonl is not the file that was made (its hash changed)")
        progress["retrieval"][version] = have
        return
    ids = [q["qa_id"] for q in questions]
    ctxs = search_contexts(corpus, ids, version, dense_cfg)
    seg = {q["qa_id"]: q["sample_id"] for q in questions}
    cat = {q["qa_id"]: str(q.get("category")) for q in questions}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            c = ctxs[i]
            f.write(json.dumps({"qa_id": i, "segment": seg[i], "category": cat[i], "memories": [h.content for h in c.hits[:sampling.GROUPS[GROUP].limit]],
                                "n_gold": c.n_gold, "found_100": c.found_100, "found_20": c.found_20, "dense_used": c.dense_used},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    progress.setdefault("retrieval", {})[version] = _sha16(path)


def _load_contexts(out_dir: str, version: str) -> Dict[str, sampling.Context]:
    out = {}
    with open(_retrieval_path(out_dir, version), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["qa_id"]] = sampling.Context(qa_id=r["qa_id"], hits=[target.Hit(id="", content=m) for m in r["memories"]], evidence_lines=[],
                                                   n_gold=r["n_gold"], found_100=r["found_100"], found_20=r["found_20"], dense_used=r.get("dense_used"))
    return out


# ---- one model: pairs, interleaved ----

def _rows_path(out_dir: str, who: str) -> str:
    return os.path.join(out_dir, f"rows_locked_{who}.jsonl")


def _whole_pairs(path: str, question_ids: Sequence[str]) -> set:
    """The questions with both versions on disk with a real outcome (a torn or errored pair is not whole)."""
    seen: Dict[str, Dict[str, str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    seen.setdefault(r["qa_id"], {})[r["version"]] = r.get("outcome")
    except FileNotFoundError:
        pass
    return {i for i in question_ids if set(seen.get(i, {})) == set(VERSIONS) and all(o in GOOD_OUTCOMES for o in seen[i].values())}


def _run_pair(runners: Dict[str, qa_run.Runner], who: str, ask: Callable, price_key: str, qa_id: str, pair_index: int, segment: str,
              claim_sha: str) -> Optional[List[dict]]:
    """Ask the first version, ask the second, judge the first, judge the second. None = the pair is not whole (not written)."""
    order = [first_version(qa_id), "dense" if first_version(qa_id) == "kw" else "kw"]
    group = sampling.GROUPS[GROUP]
    rows, replies = {}, {}
    for pos, v in enumerate(order):
        asked_at = _now()
        row, reply = runners[v]._ask(who, ask, price_key, group, qa_id)
        row.update(version=v, segment=segment, pair_index=pair_index, pair_pos=pos, asked_at=asked_at, claim_sha=claim_sha)
        if reply is None:
            return None                                  # api_error or a void row: the pair stands unfinished, the second version is not asked
        rows[v], replies[v] = row, reply
    for v in order:
        runners[v]._judge(rows[v], qa_id, replies[v])
        rows[v]["judged_at"] = _now()
        if rows[v]["outcome"] not in GOOD_OUTCOMES:
            return None                                  # a judge error
    return [rows[v] for v in order]


def run_model_pairs(who: str, ask: Callable, price_key: str, runners: Dict[str, qa_run.Runner], questions: Sequence[dict], out_dir: str,
                    claim_sha: str, meter: cost.CostMeter, progress: dict, save_progress: Callable[[], None]) -> dict:
    ids = [q["qa_id"] for q in questions]
    path = _rows_path(out_dir, who)
    done = _whole_pairs(path, ids)
    info = progress.setdefault("models", {}).setdefault(who, {})
    info.setdefault("started", _now())
    info["incomplete"] = []
    info.pop("ended", None)

    def write(pair: List[dict]) -> None:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pair))
        done.add(pair[0]["qa_id"])
        info["pairs_done"] = len(done)
        save_progress()

    failed: List[int] = []
    for k, q in enumerate(questions):
        if q["qa_id"] in done:
            continue
        meter.check()                                    # the spending limit: raises BudgetExceeded, only whole pairs are on disk
        pair = _run_pair(runners, who, ask, price_key, q["qa_id"], k, q["sample_id"], claim_sha)
        if pair is None:
            failed.append(k)
        else:
            write(pair)
    still = []
    for k in failed:                                     # once more, as whole pairs, after the pass
        q = questions[k]
        meter.check()
        pair = _run_pair(runners, who, ask, price_key, q["qa_id"], k, q["sample_id"], claim_sha)
        if pair is None:
            still.append(q["qa_id"])
        else:
            write(pair)
    info["pairs_done"] = len(done)
    if still:
        info["incomplete"] = still
        save_progress()
        raise LockedRunIncomplete(f"{who}: {len(still)} pair(s) unfinished after the second try: {', '.join(still)}")
    info["ended"] = _now()
    save_progress()
    return {"pairs_done": len(done), "complete": True}


# ---- the whole run ----

def _validate(whos, declared, candidate, limit_usd) -> None:
    if not whos or len(set(whos)) != len(whos) or any(w not in known_models() for w in whos):
        raise ValueError(f"--models is a list of distinct answering models out of {known_models()}")
    if (not isinstance(declared, (list, tuple)) or sorted(declared) != sorted(VERSIONS)):
        raise ValueError(f"--declared names exactly the two versions {list(VERSIONS)}")
    if candidate != CANDIDATE:
        raise ValueError(f"--version is the version the gates ask about: {CANDIDATE!r}")
    if not (limit_usd and limit_usd > 0):
        raise ValueError("a spending limit above zero is required")


def run_locked(corpus, out_dir: str, ledger_dir: str, whos: Sequence[str], declared: Sequence[str], candidate: str, limit_usd: float,
               lock: Sequence[str] = split.LOCK, expect_questions: Optional[int] = split.LOCK_QUESTIONS, make_ask: Optional[Callable] = None,
               judge_client=None, dense_cfg_factory: Optional[Callable] = None, identity_for: Optional[Callable] = None,
               one_model: bool = False) -> dict:
    from . import qa_main
    whos, declared = list(whos), list(declared)
    _validate(whos, declared, candidate, limit_usd)
    questions = locked_questions(corpus, lock)
    if expect_questions is not None and len(questions) != expect_questions:
        raise ValueError(f"the locked conversations hold {len(questions)} questions, not the {expect_questions} the cut was made on")
    if tuple(lock) == split.LOCK:
        split.check_partition(split.per_conversation_counts(corpus))
    identity_for = identity_for or model_identity
    prior = _load_json(os.path.join(ledger_dir, "claim.json"), None)
    prior_models = {m.get("key"): m for m in (prior or {}).get("models", [])}
    models = [dict(identity_for(w, prior_models.get(w)), order=i) for i, w in enumerate(whos)]
    plan = build_plan(questions, lock, models, declared, candidate)
    claim = split.claim_locked_models(ledger_dir, plan)             # BEFORE anything else is written
    claim_sha = claim["claim_sha256"]
    run_lock = split.acquire_run_lock(ledger_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
        progress_path = os.path.join(out_dir, "progress.json")
        progress = _load_json(progress_path, {})
        progress["claim_sha256"] = claim_sha

        def save_progress():
            _save_json(progress_path, progress)
        meter = cost.CostMeter(qa_main.PRICES, limit_usd)
        spend_path = os.path.join(out_dir, "spend.json")
        meter.restore(_load_json(spend_path, {}))
        make_ask = make_ask or (lambda who, session_id=None, workdir=None: qa_main._ask_for(who, workdir, session_id))
        judge_client = judge_client if judge_client is not None else chat.ChatClient(qa_main.judge_config())
        if hasattr(judge_client, "attach_meter"):
            judge_client.attach_meter(meter, qa_main.JUDGE_MODEL)
        result = {"claim": claim["status"], "models": {}, "complete": False}
        workdir = None
        try:
            dense_cfg = dense_cfg_factory(meter) if dense_cfg_factory else None
            for v in VERSIONS:
                ensure_retrieval(corpus, out_dir, v, questions, dense_cfg, progress)
                save_progress()
            ctx = {v: _load_contexts(out_dir, v) for v in VERSIONS}
            runners = {v: qa_run.Runner(corpus=corpus, contexts={"c": ctx[v]}, meter=meter, judge=judge.RefinedJudge(judge_client),
                                        judge_price_key=qa_main.JUDGE_MODEL, results_path=os.devnull,
                                        variant="keyword" if v == "kw" else "dense", stem_english=False) for v in VERSIONS}
            ids = [q["qa_id"] for q in questions]
            ran_one = False
            for m in models:
                who = m["key"]
                if len(_whole_pairs(_rows_path(out_dir, who), ids)) == len(ids):
                    result["models"][who] = {"pairs_done": len(ids), "complete": True}
                    continue
                if one_model and ran_one:
                    result["models"][who] = {"pairs_done": len(_whole_pairs(_rows_path(out_dir, who), ids)), "complete": False}
                    continue
                if who == "haiku":
                    workdir = tempfile.mkdtemp(prefix="haiku-cwd-")
                ask, price_key = make_ask(who, m.get("session_header"), workdir)
                client = getattr(ask, "client", None)
                if client is not None and hasattr(client, "attach_meter"):
                    client.attach_meter(meter, price_key)
                result["models"][who] = run_model_pairs(who, ask, price_key, runners, questions, out_dir, claim_sha, meter, progress, save_progress)
                ran_one = True
            result["complete"] = all(result["models"].get(m["key"], {}).get("complete") for m in models)
        finally:
            _save_json(spend_path, meter.state())
            save_progress()
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)
        return result
    finally:
        split.release_run_lock(run_lock)


# ---- the report: numbers and ids only ----

def _latest_rows(path: str) -> Dict[tuple, dict]:
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[(r["qa_id"], r["version"])] = r
    except FileNotFoundError:
        pass
    return out


def locked_report(out_dir: str, ledger_dir: str) -> dict:
    plan = _load_json(os.path.join(ledger_dir, "claim.json"), None)
    if plan is None:
        raise ValueError("there is no claim in the ledger: nothing was run")
    order = plan["question_order"]
    base_v = next(v for v in plan["versions"] if v != plan["candidate"])
    cand_v = plan["candidate"]
    models, complete = {}, []
    all_rows: Dict[str, Dict[tuple, dict]] = {}
    for m in plan["models"]:
        who = m["key"]
        rows = _latest_rows(_rows_path(out_dir, who))
        whole = all(((i, v) in rows and rows[(i, v)]["outcome"] in GOOD_OUTCOMES) for i in order for v in plan["versions"])
        all_rows[who] = rows
        if not whole:
            models[who] = {"complete": False, "pairs_done": len(_whole_pairs(_rows_path(out_dir, who), order))}
            continue
        complete.append(who)
        entry = {"complete": True, "questions": len(order)}
        for v in plan["versions"]:
            rs = [rows[(i, v)] for i in order]
            entry[v] = {o: sum(1 for r in rs if r["outcome"] == o) for o in GOOD_OUTCOMES}
            entry[v]["by_segment"] = {g: {o: sum(1 for r in rs if r["segment"] == g and r["outcome"] == o) for o in GOOD_OUTCOMES} for g in plan["partition"]["lock"]}
        better = [i for i in order if rows[(i, cand_v)]["outcome"] == "correct" and rows[(i, base_v)]["outcome"] != "correct"]
        worse = [i for i in order if rows[(i, base_v)]["outcome"] == "correct" and rows[(i, cand_v)]["outcome"] != "correct"]
        entry["flips"] = {"better": len(better), "worse": len(worse), "net": len(better) - len(worse), "better_ids": better, "worse_ids": worse}
        models[who] = entry
    decision = None
    if len(complete) == len(plan["models"]):
        keyed = [m["key"] for m in plan["models"]]
        base_rows, cand_rows = [], []
        for who in keyed:
            for i in order:
                for v, dest in ((base_v, base_rows), (cand_v, cand_rows)):
                    r = all_rows[who][(i, v)]
                    dest.append({"item_id": i, "group": r["segment"], "correct": r["outcome"] == "correct", "model": who})
        decision = split.decide_dense(base_rows, cand_rows, lock=plan["partition"]["lock"], models=keyed)
    dense_use = None
    rpath = _retrieval_path(out_dir, cand_v)
    if os.path.exists(rpath):
        flags = [json.loads(l).get("dense_used") for l in open(rpath, encoding="utf-8") if l.strip()]
        dense_use = {"questions": len(flags), "used_dense": sum(1 for f in flags if f is True), "skipped_dense": sum(1 for f in flags if f is False)}
    spend = _load_json(os.path.join(out_dir, "spend.json"), {})
    return {"plan_sha16": plan["plan_sha16"], "claim_sha256": _load_json(os.path.join(out_dir, "progress.json"), {}).get("claim_sha256"),
            "complete_models": complete, "models": models, "decision": decision, "dense_use": dense_use,
            "spend": {"spent": spend.get("spent"), "tokens": spend.get("tokens"), "usd": spend.get("usd")},
            "timeline": {k: {"started": v.get("started"), "ended": v.get("ended")} for k, v in _load_json(os.path.join(out_dir, "progress.json"), {}).get("models", {}).items()}}
