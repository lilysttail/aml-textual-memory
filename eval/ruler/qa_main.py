"""Step 2, the driver: python -m eval.ruler.qa_main <command> ...

    pick     fix the 48 questions (12 per category, tuning conversations only) and the 12 for the cross-check; ids only
    run      ask one model (--who main|kimi|haiku|an OpenCode model id) for some groups, judge, keep rows; the spending carries over between runs
    report   tables from the rows: numbers and question ids only
    probe    list the model ids the Kimi plan offers (no question is sent)
    consistency-prepare / consistency-compare   our judge against the official scripts/run_eval.sh on a fixed sample of answers

Everything is local except the model calls; keys come from the environment (set by whoever starts the run), never from files.
The locked conversations are refused here: the named conversations are checked, and so is every question id of picks.json
after it is read back (a hand-edited file is refused as a whole), before anything is asked or written.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from typing import Callable, Dict, List, Optional, Sequence

from . import answerer, chat, claude_cli, consistency, cost, judge, qa_report, qa_run, sampling, split
from . import dense as dense_mod
from . import corpus as corp

INTL_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ANSWER_MODEL = "qwen3.7-flash-2026-07-15"
# The official judge model (qwen3-14b) is not served to this account on the international endpoint (HTTP 404; not in /models), so
# a same-family stand-in is used with the official prompt and request shape: judgements here are for "where does it go wrong"
# only and are not comparable with platform scores. A second stand-in judges the consistency sample too.
JUDGE_MODEL = "qwen3.5-27b"
SECOND_JUDGE_MODEL = "qwen3.6-27b"
KIMI_BASE = "https://api.kimi.com/coding/v1"
KIMI_MODEL = "k3-256k"
HAIKU_MODEL = "claude-haiku-5-5"
# The longest single judge reply in every result kept so far is 17 completion tokens (357 judge calls, 7 result files: 12 or 17); the cap
# is 128, more than seven times that. A reply that hits it before a label is an error (see judge.py), never a score. With the cap the
# paid gate reserves 128 tokens per judge call instead of an assumed 4,096 (the official request shape sets none).
JUDGE_MAX_TOKENS = 128
# Small answering models on the OpenCode Go gateway (a subscription with a monthly dollar limit per model, not pay as you go). Ids checked against
# the gateway's public model list on 2026-10-24. The first four speak the OpenAI chat style on the base the home code already uses
# (workspace pipeline_helpers.js, NIGHTLY_LLM_PROVIDERS.opencode); the gateway docs put qwen3.8-flash on the messages-style endpoint only.
OPENCODE_OPENAI_BASE = "https://opencode.ai/zen/go/v1"
OPENCODE_MESSAGES_BASE = "https://opencode.ai/inference/go/anthropic/v1"
OPENCODE_MODELS: Dict[str, str] = {"hy3": "openai", "qwen3.8-flash": "anthropic", "minimax-m3.1-flash": "openai",
                                   "mimo-v2.6-flash": "openai", "glm-5.3-flash": "openai"}
CONVERSATIONS = split.TUNE
USER_AGENT = "aml-ruler-step2/1.0"
OPENCODE_USER_AGENT = "aml-ruler/1.0"

# US dollars per million tokens. The answering model: Alibaba Cloud Model Studio international price list (page "model-pricing",
# read 2026-10-23, updated 2026-10-22), non-thinking, by the request's input size. The judge: the international list does not
# show qwen3-14b and the account cannot call it; the stand-ins are priced from the same list (non-thinking, up to 256K input).
# Kimi and Haiku run on plans (quota), not by price.
PRICES: Dict[str, cost.Price] = {
    ANSWER_MODEL: cost.tiered((32_000, 0.03, 0.13), (256_000, 0.10, 0.40), (1_000_000, 0.20, 0.80)),
    JUDGE_MODEL: (0.3, 2.4),
    SECOND_JUDGE_MODEL: (0.6, 3.6),
    KIMI_MODEL: None,
    HAIKU_MODEL: None,
    dense_mod.EMBED_PRICE_KEY: (0.07, 0.0),            # text-embedding-v4, international list price, per million input tokens
    # OpenCode Go price list (opencode.ai/docs/go, read 2026-10-24), per million tokens, input / output. The plan counts usage against each
    # model's monthly dollar limit at these prices, so they go in the table and through the paid gate like any priced model.
    "hy3": (0.14, 0.58),
    "qwen3.8-flash": (0.15, 0.47),
    "minimax-m3.1-flash": (0.15, 0.60),
    "mimo-v2.6-flash": (0.14, 0.28),
    "glm-5.3-flash": (0.15, 0.50),
}


def main_config() -> chat.ChatConfig:
    return chat.ChatConfig(base_url=INTL_BASE, model=ANSWER_MODEL, api_key_env="DASHSCOPE_API_KEY", temperature=0.0, max_tokens=128,  # gate:allow-secret
                           extra_body={"enable_thinking": False}, user_agent=USER_AGENT)


def judge_config(model: str = JUDGE_MODEL) -> chat.ChatConfig:
    # the official request (one user message, temperature 0, thinking off; src/llm_judge_runtime.py) plus a length cap that no verdict comes near
    return chat.ChatConfig(base_url=INTL_BASE, model=model, api_key_env="DASHSCOPE_API_KEY", temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,  # gate:allow-secret
                           extra_body={"enable_thinking": False}, user_agent=USER_AGENT)


def opencode_config(model: str, session_id: Optional[str] = None) -> chat.ChatConfig:
    """One of the OpenCode Go gateway's small models as an answering model. The gateway refuses a request without a session id (seen in our own probe), so one is made for this config and stays the same for every attempt of the run; the user agent says who is calling. Reasoning is
    turned off with `reasoning_effort: none`, the setting that took effect through this gateway in our own probe; whether each of
    these models honours it is checked by the first paid request, not assumed."""
    if model not in OPENCODE_MODELS:
        raise ValueError(f"{model!r} is not one of the OpenCode models of this step: {sorted(OPENCODE_MODELS)}")
    style = OPENCODE_MODELS[model]
    headers = {"x-opencode-session": session_id or "aml-ruler-" + str(uuid.uuid4())}
    return chat.ChatConfig(base_url=OPENCODE_MESSAGES_BASE if style == "anthropic" else OPENCODE_OPENAI_BASE, model=model,
                           api_key_env="OPENCODE_GO_API_KEY", temperature=0.0, max_tokens=128,  # gate:allow-secret
                           extra_body={} if style == "anthropic" else {"reasoning_effort": "none"}, headers=headers,
                           api_style=style, user_agent=OPENCODE_USER_AGENT)


def kimi_config() -> chat.ChatConfig:
    # k3 takes only its own default temperature (any other value is a 400): the request carries none
    return chat.ChatConfig(base_url=KIMI_BASE, model=KIMI_MODEL, api_key_env="KIMI_CODE_API_KEY", temperature=None, max_tokens=128,  # gate:allow-secret
                           extra_body={"thinking": {"type": "disabled"}}, user_agent=USER_AGENT)


def _load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _save_json(path: str, doc) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def execute_run(corpus, out_dir: str, who: str, groups: Sequence[str], subset: str, limit_usd: float, ask: Callable,
                price_key: str, judge_client, judge_price_key: str = JUDGE_MODEL, prices: Optional[dict] = None,
                conv_ids: Sequence[str] = CONVERSATIONS, per_category: int = 12, cross_per_category: int = 3,
                workers: int = 1, dense: Optional[dense_mod.DenseConfig] = None, variant: str = "keyword",
                stem_english: bool = False) -> List[dict]:
    if set(conv_ids) & set(split.LOCK):
        raise ValueError("the locked conversations are not touched in step 2")      # the named conversations; picks.json is checked below
    os.makedirs(out_dir, exist_ok=True)
    picks_path = os.path.join(out_dir, "picks.json")
    picks = _load_json(picks_path, None)
    if picks is None:
        picks = sampling.pick_questions(corpus, conv_ids, per_category=per_category, cross_per_category=cross_per_category)
        sampling.check_picks(corpus, picks, conv_ids, per_category=per_category, cross_per_category=cross_per_category)
        _save_json(picks_path, picks)
    else:
        sampling.check_picks(corpus, picks, conv_ids, per_category=per_category, cross_per_category=cross_per_category)
    shapes_needed = sorted({sampling.GROUPS[g].shape for g in groups})
    # the run's conditions, written before anything is built or asked: the setting is explicit, never left to the outer environment
    _save_json(os.path.join(out_dir, f"conditions_{variant}.json"), {"variant": variant, "stem_english": bool(stem_english),
                                                                    "per_category": per_category, "cross_per_category": cross_per_category})
    meter = cost.CostMeter(prices or PRICES, limit_usd)
    spend_path = os.path.join(out_dir, "spend.json")
    meter.restore(_load_json(spend_path, {}))
    # the answering and the judging clients hold room in the meter before every call (a hard wall, not a check between questions)
    for client, key in ((getattr(ask, "client", None), price_key), (judge_client, judge_price_key)):
        if client is not None and hasattr(client, "attach_meter"):
            client.attach_meter(meter, key)
    if dense is not None:
        dense.cost_meter = meter                        # the vector calls count against the same limit and go in the same table
    emb_stats: dict = {}
    try:
        contexts = {s: sampling.build_contexts(corpus, picks["main"], shape=s, dense=dense, stats_out=emb_stats.setdefault(s, {}),
                                         stem_english=stem_english)
                    for s in shapes_needed}
    finally:
        if dense is not None:
            _save_json(spend_path, meter.state())
            _save_json(os.path.join(out_dir, f"embedding_{variant}.json"), emb_stats)
    # the cap / the spending limit stopped vector calls while the contexts were built: the meaning-based half of those searches is missing,
    # and a run that carries on would label keyword-only results as "dense". Stop before any model is asked.
    for shape_stats in emb_stats.values():
        snap = shape_stats.get("embedding") or {}
        if snap.get("refused_after_cap") or snap.get("replies_above_bound"):
            raise cost.BudgetExceeded("the vector calls were stopped at the token cap / spending limit: the dense contexts are incomplete, "
                                      "no model was asked")
    runner = qa_run.Runner(corpus=corpus, contexts=contexts, meter=meter, judge=judge.RefinedJudge(judge_client),
                           judge_price_key=judge_price_key, results_path=os.path.join(out_dir, _rows_name(variant)), variant=variant,
                           stem_english=stem_english)
    try:
        return runner.run_model(who, ask, price_key, list(groups), picks[subset], workers=workers)
    finally:
        _save_json(spend_path, meter.state())


# ---- the report ----

def _rows_name(variant: str) -> str:
    return "rows.jsonl" if variant == "keyword" else f"rows_{variant}.jsonl"


def latest_rows(path: str) -> List[dict]:
    """One row per (group, model, question): the last one written."""
    out: Dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["group"], r["model_key"], r["qa_id"])] = r
    return list(out.values())


def _rows(out_dir: str, variant: str = "keyword") -> List[dict]:
    return latest_rows(os.path.join(out_dir, _rows_name(variant)))


def write_report(out_dir: str, variant: str = "keyword") -> dict:
    rows = _rows(out_dir, variant)
    rep = qa_report.build_report(rows)
    models = sorted({r["model_key"] for r in rows})
    rep["flips"] = {f"{a}_vs_{b}": {m: qa_report.flip_table(rows, a, b, m) for m in models}
                    for a, b in (("G2", "G3"), ("G4", "G5"), ("G1", "G2"))}
    rep["spend"] = _load_json(os.path.join(out_dir, "spend.json"), {})
    rep["conditions"] = _load_json(os.path.join(out_dir, f"conditions_{variant}.json"), {})
    rep["cli_reported_cost_usd"] = round(sum(float((r.get("extra") or {}).get("cost_usd") or 0) for r in rows if r["model_key"] == "haiku"), 6)
    rep["tool_calls_by_row"] = {m: {"rows": sum(1 for r in rows if r["model_key"] == m and "tool_calls" in r),
                                    "with_tool_calls": sum(1 for r in rows if r["model_key"] == m and r.get("tool_calls"))}
                                for m in models if m == "haiku"}
    suffix = "" if variant == "keyword" else f"_{variant}"
    _save_json(os.path.join(out_dir, f"report{suffix}.json"), rep)
    with open(os.path.join(out_dir, f"report{suffix}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(rep))
    return rep


def render_markdown(rep: dict) -> str:
    out = ["# step 2: numbers and question ids only", ""]
    out += ["## outcomes (questions / correct / said not found / made up / other)", "", "| group | model | n | correct | not found | made up | other |", "|---|---|---|---|---|---|---|"]
    for g in sorted(rep["outcomes"]):
        for m, t in sorted(rep["outcomes"][g].items()):
            out.append(f"| {g} | {m} | {t['questions']} | {t['correct']} | {t['not_found']} | {t['made_up']} | {t['other']} |")
    out += ["", "## where the evidence was lost", "", "| group | model | wrong, not returned | wrong, returned but not in prompt | wrong, evidence in prompt | correct without evidence |", "|---|---|---|---|---|---|"]
    for g in sorted(rep["where_lost"]):
        for m, t in sorted(rep["where_lost"][g].items()):
            out.append(f"| {g} | {m} | {t['wrong_not_returned']} | {t['wrong_returned_not_in_prompt']} | {t['wrong_evidence_in_prompt']} | {t['correct_without_evidence']} |")
    out += ["", "## by category (correct / questions)", "", "| group | model | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|---|"]
    for g in sorted(rep["by_category"]):
        for m, cats in sorted(rep["by_category"][g].items()):
            out.append(f"| {g} | {m} | " + " | ".join(f"{cats[c]['correct']}/{cats[c]['questions']}" if c in cats else "-" for c in "1234") + " |")
    out += ["", "## flips between two groups (counts; ids are in report.json)", "", "| pair | model | compared | both correct | only first | only second | neither |", "|---|---|---|---|---|---|---|"]
    for pair, per in rep.get("flips", {}).items():
        for m, f in sorted(per.items()):
            out.append(f"| {pair} | {m} | {f['compared']} | {len(f['both_correct'])} | {len(f['only_first_correct'])} | {len(f['only_second_correct'])} | {len(f['neither_correct'])} |")
    sp = rep.get("spend") or {}
    out += ["", "## spending", "", f"spent by the meter: {sp.get('spent', 0):.4f} US dollars", "", "| model | calls | prompt tokens | completion tokens |", "|---|---|---|---|"]
    for m, t in sorted((sp.get("tokens") or {}).items()):
        out.append(f"| {m} | {t['calls']} | {t['prompt']} | {t['completion']} |")
    return "\n".join(out) + "\n"


# ---- command line ----

def prepare_consistency(corpus, out_dir: str, rows: Sequence[dict], judge_client, second_judge_client, limit_usd: float = 1.0,
                        sample_n: int = 30) -> dict:
    """Judge a fixed sample of the main model's G2 answers with both stand-in judges and write the official-format inputs.
    The two judging clients hold room in the same spending meter as every other call (a hard wall, the same `spend.json`, the same limit):
    a call the wall refuses is not sent and the command stops with `BudgetExceeded`; the spend so far is saved either way, and the
    scores are written only when both judges finished."""
    ids = consistency.pick_sample(rows, "main", "G2", n=sample_n)
    d = os.path.join(out_dir, "consistency")
    os.makedirs(d, exist_ok=True)
    picked = [r for r in rows if r["group"] == "G2" and r["model_key"] == "main"]
    consistency.write_official_inputs(corpus, picked, ids, d)
    meter = cost.CostMeter(PRICES, limit_usd)
    spend_path = os.path.join(out_dir, "spend.json")
    meter.restore(_load_json(spend_path, {}))
    judge_client.attach_meter(meter, JUDGE_MODEL)
    second_judge_client.attach_meter(meter, SECOND_JUDGE_MODEL)
    try:
        scores, _ = consistency.our_scores(judge.RefinedJudge(judge_client), corpus, picked, ids)
        scores2, _ = consistency.our_scores(judge.RefinedJudge(second_judge_client), corpus, picked, ids)
    finally:
        _save_json(spend_path, meter.state())
    _save_json(os.path.join(d, "ours.json"), scores)
    _save_json(os.path.join(d, "ours_second_judge.json"), scores2)
    return {"sample": len(ids), "ours_correct": sum(scores.values())}


def _model_ask(client):
    """The answering function for one model through a chat client; `.client` lets the run attach its spending meter to the client."""
    a = answerer.ModelAnswerer(client)

    def ask(question, memories):
        return a.answer_with_usage(question, memories)
    ask.client = client
    return ask


def _ask_for(who: str, workdir: Optional[str], session_id: Optional[str] = None):
    if who == "main":
        return _model_ask(chat.ChatClient(main_config())), ANSWER_MODEL
    if who == "kimi":
        return _model_ask(chat.ChatClient(kimi_config())), KIMI_MODEL
    if who == "haiku":
        return claude_cli.ClaudeCliAnswerer(HAIKU_MODEL, workdir=workdir), HAIKU_MODEL
    if who in OPENCODE_MODELS:
        return _model_ask(chat.ChatClient(opencode_config(who, session_id))), who
    raise SystemExit("--who is main, kimi, haiku or one of " + ", ".join(sorted(OPENCODE_MODELS)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pick", "run", "report", "probe", "consistency-prepare", "consistency-compare", "load"):
        p = sub.add_parser(name)
        if name != "probe":
            p.add_argument("--out", required=True, help="results directory (outside the repository)")
        if name in ("pick", "run", "consistency-prepare", "load"):
            p.add_argument("--data", required=True, help="the commit copy of the public data")
            p.add_argument("--no-pin", action="store_true", help="do NOT refuse a copy that is not the pinned one (tests only)")
        if name in ("run", "load"):
            p.add_argument("--search", choices=["keyword", "dense"], default="keyword", help="dense = keyword + meaning-based path (needs DASHSCOPE_API_KEY)")
            p.add_argument("--vector-cache", help="the fixed local vector file: a text embedded once is never embedded again")
            p.add_argument("--stem-english", choices=["on", "off"], default="off", help="English word-form matching of the keyword path; set explicitly for every run and written into its conditions")
            p.add_argument("--embed-cap-tokens", type=int, default=600_000, help="stop calling the embedding provider after this many tokens")
        if name == "run":
            p.add_argument("--who", required=True, choices=["main", "kimi", "haiku"] + sorted(OPENCODE_MODELS))
            p.add_argument("--groups", required=True)
            p.add_argument("--subset", default="main", choices=["main", "cross"])
            p.add_argument("--limit-usd", type=float, default=1.0)
            p.add_argument("--workers", type=int, default=1)
        if name == "load":
            p.add_argument("--shape", default="a", choices=["a", "b", "c", "d"])
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--warmup", action="store_true")
            p.add_argument("--probe-n", type=int, default=30)
            p.add_argument("--concurrent-embed", action="store_true",
                           help="a concurrency measurement: the embedding requests overlap like a deployed service's instead of queueing behind "
                                "the paid gate. The cap becomes a running sum of request upper bounds (a looser wall: real tokens are far below it)")
            p.add_argument("--conv-ids", help="comma list; default = the tuning conversations. A locked conversation is refused.")
        if name == "report":
            p.add_argument("--variant", default="keyword")
        if name == "consistency-compare":
            p.add_argument("--official", required=True, help="the *_scored.jsonl the official script wrote")
    # step 4 (plan v3): the one-time run on the locked conversations. All four of --version / --declared / --ledger / --models are required.
    p = sub.add_parser("run-locked", help="the one-time paired run on the locked conversations (claims first; the same plan again resumes)")
    p.add_argument("--data", required=True, help="the commit copy of the public data")
    p.add_argument("--no-pin", action="store_true", help="do NOT refuse a copy that is not the pinned one (tests only)")
    p.add_argument("--out", required=True, help="results directory (outside the repository)")
    p.add_argument("--version", required=True, help="the version the gates ask about (dense)")
    p.add_argument("--declared", required=True, help="the two versions, comma separated (kw,dense)")
    p.add_argument("--ledger", required=True, help="the ledger directory: the claim of the locked part")
    p.add_argument("--models", required=True, help="the answering models in run order, comma separated (main,hy3,haiku)")
    p.add_argument("--limit-usd", type=float, default=1.2)
    p.add_argument("--vector-cache", help="the fixed local vector file")
    p.add_argument("--embed-cap-tokens", type=int, default=600_000)
    p.add_argument("--one-model", action="store_true", help="run only the next unfinished model, then stop (to post its numbers before the next)")
    p = sub.add_parser("report-locked", help="the locked run's numbers and, once every model is whole, the three gates")
    p.add_argument("--out", required=True)
    p.add_argument("--ledger", required=True)
    p = sub.add_parser("gate-check", help="before a dense-on deployment: every tuning question really used the dense path under concurrency (never the locked part)")
    p.add_argument("--data", required=True, help="the commit copy of the public data")
    p.add_argument("--no-pin", action="store_true", help="do NOT refuse a copy that is not the pinned one (tests only)")
    p.add_argument("--out", required=True, help="results directory (outside the repository)")
    p.add_argument("--vector-cache", required=True, help="the fixed local vector file")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-usd", type=float, default=0.05)
    p.add_argument("--embed-cap-tokens", type=int, default=600_000)
    return ap


def _dense_config(args) -> Optional[dense_mod.DenseConfig]:
    if args.search != "dense":
        return None
    return dense_mod.DenseConfig(cap_tokens=args.embed_cap_tokens, cache_path=args.vector_cache,
                                 concurrent_embed=getattr(args, "concurrent_embed", False))


def _run_locked(corpus, args) -> int:
    from . import locked_run, split as split_mod
    try:
        res = locked_run.run_locked(corpus, args.out, args.ledger, [m for m in args.models.split(",") if m], [v for v in args.declared.split(",") if v],
                                    args.version, args.limit_usd, one_model=args.one_model,
                                    dense_cfg_factory=lambda meter: dense_mod.DenseConfig(cap_tokens=args.embed_cap_tokens, cache_path=args.vector_cache,
                                                                                         cost_meter=meter))
    except split_mod.LockedSetAlreadyUsed as e:
        raise SystemExit(f"refused: {e}")
    except locked_run.LockedRunIncomplete as e:
        raise SystemExit(f"unfinished: {e}")
    except cost.BudgetExceeded as e:
        raise SystemExit(f"stopped at the spending limit (the same command resumes on the same claim): {e}")
    except ValueError as e:
        raise SystemExit(f"refused: {e}")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "report":
        rep = write_report(args.out, args.variant)
        print(json.dumps({"outcomes": rep["outcomes"], "spent_usd": (rep.get("spend") or {}).get("spent")}, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "probe":
        return _probe()
    if args.cmd == "consistency-compare":
        ours = _load_json(os.path.join(args.out, "consistency", "ours.json"), {})
        ours2 = _load_json(os.path.join(args.out, "consistency", "ours_second_judge.json"), {})
        res = {"ours_vs_official_script": qa_report.agreement(ours, consistency.read_official_scores(args.official)),
               "two_stand_in_judges": qa_report.agreement(ours, ours2)}
        _save_json(os.path.join(args.out, "consistency", "agreement.json"), res)
        print(json.dumps(res, indent=1))
        return 0
    if args.cmd == "report-locked":
        from . import locked_run
        try:
            rep = locked_run.locked_report(args.out, args.ledger)
        except ValueError as e:
            raise SystemExit(str(e))
        _save_json(os.path.join(args.out, "report_locked.json"), rep)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0
    corpus = corp.load_corpus(args.data, require_pin=not args.no_pin) if args.cmd != "consistency-compare" else None
    if args.cmd == "run-locked":
        return _run_locked(corpus, args)
    if args.cmd == "gate-check":
        from . import gate_check
        meter = cost.CostMeter(PRICES, args.limit_usd)
        try:
            rep = gate_check.run_gate_check(corpus, args.out, args.vector_cache, workers=args.workers, cost_meter=meter,
                                            embed_cap_tokens=args.embed_cap_tokens)
        except ValueError as e:
            raise SystemExit(f"refused: {e}")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if rep["passed"] else 1
    if args.cmd == "load":
        from . import load_measure
        conv_ids = [c for c in args.conv_ids.split(",") if c] if args.conv_ids else list(CONVERSATIONS)
        if set(conv_ids) & set(split.LOCK):
            raise SystemExit("the locked conversations are not touched")
        try:
            r = load_measure.measure(corpus, args.out, shape=args.shape, dense_cfg=_dense_config(args), workers=args.workers,
                                     warmup=args.warmup, probe_n=args.probe_n, conv_ids=conv_ids,
                                     stem_english=args.stem_english == "on")
        except RuntimeError as e:
            raise SystemExit(str(e))
        print(json.dumps({k: r[k] for k in ("label", "questions_asked", "evidence_hit_rate", "stopped_at_token_cap", "search_errors", "add_errors")},
                         ensure_ascii=False))
        return 0
    if args.cmd == "pick":
        picks = sampling.pick_questions(corpus, CONVERSATIONS)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "picks.json")
        if os.path.exists(path):
            raise SystemExit("picks.json already exists: the pick is made once")
        _save_json(path, picks)
        print(json.dumps(picks, indent=1))
        return 0
    if args.cmd == "run":
        workdir = tempfile.mkdtemp(prefix="haiku-cwd-") if args.who == "haiku" else None
        try:
            ask, price_key = _ask_for(args.who, workdir)
            rows = execute_run(corpus, args.out, args.who, [g for g in args.groups.replace(",", " ").split()], args.subset, args.limit_usd, ask, price_key,
                               chat.ChatClient(judge_config()), workers=args.workers, dense=_dense_config(args), variant=args.search,
                               stem_english=args.stem_english == "on")
        finally:
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)
        from collections import Counter
        print(json.dumps({"rows": len(rows), "outcomes": dict(Counter(r["outcome"] for r in rows))}, indent=1))
        return 0
    rows = _rows(args.out)
    if args.cmd == "consistency-prepare":
        try:
            res = prepare_consistency(corpus, args.out, rows, chat.ChatClient(judge_config()), chat.ChatClient(judge_config(SECOND_JUDGE_MODEL)))
        except cost.BudgetExceeded as e:
            raise SystemExit(f"stopped at the spending limit: {e}")
        print(json.dumps(res))
        return 0
    return 2


def _probe() -> int:
    import urllib.request
    key = os.environ.get("KIMI_CODE_API_KEY", "")
    if not key:
        raise SystemExit("KIMI_CODE_API_KEY is not set")
    req = urllib.request.Request(KIMI_BASE + "/models", headers={"Authorization": "Bearer " + key, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        doc = json.loads(r.read().decode("utf-8"))
    print(json.dumps(sorted(m.get("id") for m in doc.get("data", []))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
