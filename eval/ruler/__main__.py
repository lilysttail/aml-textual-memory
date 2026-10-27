"""python -m eval.ruler <command> --data <copy of the public data> [...]

    run           write the conversations through Add in one shape, ask every question, diagnose the evidence
    facts         the three facts about the copy (time level, picture descriptions, questions naming a person)
    unanswerable  questions that cannot be answered at all, per shape x category (needs the service; local, no model)
    split         per-conversation question counts, a tune/locked split, and how big a difference the locked part can tell

Local, no network, no model, no money. Output is numbers and ids only.
"""
import argparse
import json
import sys

from . import corpus as corp
from . import facts as facts_mod
from . import run as run_mod
from . import split as split_mod
from . import unanswerable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "facts", "unanswerable", "split"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="the commit copy of the public data (contains data/public/)")
        p.add_argument("--no-pin", action="store_true", help="do NOT refuse a copy that is not the pinned one (tests only)")
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--shape", required=True, choices=["a", "b", "c", "d", "legacy", "legacy_t"])
            p.add_argument("--neighbours", type=int, default=0, choices=(0, 1))
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--top-k", type=int, default=100)
            p.add_argument("--engine", choices=["service", "core"], default="service",
                           help="service = what the platform talks to (default); core = the old ruler's way of writing, "
                                "kept only to reproduce its numbers")
            p.add_argument("--partition", choices=["tune", "locked", "all"], default="tune",
                           help="tune (default) = the five conversations for tuning; locked = the five locked ones, used once "
                                "(needs --version, --declared, --ledger); all = describing the whole set, never for choosing")
            p.add_argument("--version", help="locked part only: which of the two declared versions this run is")
            p.add_argument("--declared", help="locked part only: the two versions of the one comparison, e.g. baseline,hardened")
            p.add_argument("--ledger", help="locked part only: the once-only ledger (a directory; one file per claim)")
        if name == "unanswerable":
            p.add_argument("--stored", choices=["service", "rule"], default="service")
        if name == "split":
            p.add_argument("--rows", help="rows_*.jsonl of a run: used to estimate how much questions of one conversation move together")
            p.add_argument("--sizes", default="3,4,5,6")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        if args.partition == "locked" and not (args.version and args.declared and args.ledger):
            ap.error("--partition locked needs --version, --declared and --ledger")
        s = run_mod.run(args.data, args.out, shape=args.shape, neighbours=args.neighbours, limit=args.limit,
                        top_k=args.top_k, require_pin=not args.no_pin, engine=args.engine, partition=args.partition,
                        version=args.version, declared=args.declared.split(",") if args.declared else None,
                        ledger=args.ledger)
        keep = {k: v for k, v in s.items() if k not in ("by_category", "by_group")}
        print(json.dumps(keep, ensure_ascii=False, indent=1))
        return 0
    c = corp.load_corpus(args.data, require_pin=not args.no_pin)
    if args.cmd == "facts":
        raw = corp._read_jsonl(corp.data_paths(args.data)["conversations"])
        print(json.dumps(facts_mod.compute(c, raw), ensure_ascii=False, indent=1))
    elif args.cmd == "unanswerable":
        res = unanswerable.classify(c, stored=args.stored)
        print(json.dumps({"questions_per_category": unanswerable.questions_per_category(c),
                          "time_wording_by_category": unanswerable.time_wording_by_category(c),
                          "u2_wording_sensitivity": unanswerable.u2_wording_sensitivity(c),
                          "table": unanswerable.table(c, res),
                          "sensitivity": {"naming_gap_by_category": unanswerable.naming_gap(c),
                                          "u1_any_named_missing_shape_a": unanswerable.u1_any_missing(c, "a")}},
                         ensure_ascii=False, indent=1))
    else:
        counts = split_mod.per_conversation_counts(c)
        out = {"per_conversation": counts, "options": [],
               "fixed": {"lock": list(split_mod.LOCK), "tune": list(split_mod.TUNE),
                         "lock_questions": split_mod.LOCK_QUESTIONS, "tune_questions": split_mod.TUNE_QUESTIONS,
                         "dense_gate": {"min_overall_gain_points": split_mod.DENSE_MIN_OVERALL_GAIN_POINTS,
                                        "min_segments_better": split_mod.DENSE_MIN_SEGMENTS_BETTER}}}
        if not args.no_pin:
            split_mod.check_partition(counts)
        rows = []
        if args.rows:
            with open(args.rows, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
        deff = split_mod.design_effect(rows) if rows else None
        out["design_effect_from_rows"] = deff
        for k in [int(x) for x in args.sizes.split(",") if x]:
            sp = split_mod.choose_split(counts, lock_size=k)
            n = sp["lock_questions"]
            opt = dict(sp, mdd_unpaired_p05=split_mod.min_detectable_difference_unpaired(n, 0.5),
                       mdd_paired_discordant_10pct=split_mod.min_detectable_difference_paired(n, 0.10),
                       mdd_paired_discordant_20pct=split_mod.min_detectable_difference_paired(n, 0.20))
            if deff:
                opt["effective_questions_after_conversation_grouping"] = round(n / deff, 1)
                opt["mdd_paired_discordant_10pct_grouped"] = split_mod.min_detectable_difference_paired(n / deff, 0.10)
            out["options"].append(opt)
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
