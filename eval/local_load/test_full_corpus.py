"""The whole public corpus over HTTP, the way the platform drives it.

WHAT THIS IS
    We take the public LoCoMo-Refined copy (the same one the local ruler
    uses), cut every conversation with the ruler's cut (eval/ruler/corpus.py),
    push every block through /add over HTTP in the ruler's shape a
    (eval/ruler/shapes.py: role user/assistant + the text, nothing else; what
    the platform sends is defined there and nowhere else), then ask all 1,380
    questions through /search with top_k=100. We record: the errors, whether
    any answer exceeded top_k, how much text each answer carried (and its
    approximate token count), the latency distribution, and the peak memory of
    the container. The evidence recall is compared with the ruler's number for
    shape a on the service path with neighbours on (0.7899, eval/ruler/README.md).

HOW TO RUN
        py -m eval.local_load.test_full_corpus
    The corpus copy lives outside the repository, so AML_CORPUS has to point at
    the commit copy (the directory that contains data/public/, the same path the
    ruler takes as --data) — there is no default (the path depends on where the
    copy was put):
        set AML_CORPUS=<local copy of the public set>
        py -m eval.local_load.test_full_corpus
"""
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from eval.local_load.common import (CONTAINER, add, docker, parse_memory,
                                    percentiles, report, search, start_container,
                                    stop_container)

CORPUS = os.environ.get("AML_CORPUS", "")
SHAPE = "a"        # what the platform sends: eval/ruler/shapes.py is the only place that defines it

# The fixed copy this test is written against: mem-eval-suite/LoCoMo_refined,
# commit 887091190789e8d6760e70b9edd696539923dc4f (its manifest.json says
# 1382 questions and 10 conversations). The numbers are pinned in ONE place,
# eval/ruler/corpus.py, so a truncated or swapped corpus cannot pass as "the
# whole corpus" here or in the ruler.
from eval.ruler import corpus as corp
from eval.ruler import shapes


def main():
    if not CORPUS:
        raise SystemExit("set AML_CORPUS to the commit copy of the public data "
                         "(the directory that contains data/public/)")
    c = corp.load_corpus(CORPUS, require_pin=False)
    convs, questions = c.conversations, c.questions
    # the corpus is pinned: a truncated or swapped copy must not pass
    corpus_ok = c.pinned
    start_container(fresh=True)
    peak = {"bytes": 0}
    stop_sampling = threading.Event()

    def sample_memory():
        while not stop_sampling.is_set():
            try:
                out = docker("stats", "--no-stream", "--format",
                             "{{.MemUsage}}", CONTAINER, check=False)
                used = parse_memory(out)
                if used:
                    peak["bytes"] = max(peak["bytes"], used)
            except Exception:
                pass
            time.sleep(1.0)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        add_errors = []
        add_times = []
        # record_id -> dia_id (NOT the other way round: the same dia_id such as
        # "D1:3" appears in several conversations, so a global dia->record map
        # would collide and under-count the hits)
        rec_to_dia = {}
        blocks = 0
        lock = threading.Lock()

        def push(conv):
            """Write one conversation's blocks; returns the per-block results."""
            out = []
            for b in corp.chunk_conversation(conv):
                messages = [shapes.render(SHAPE, m, sess) for m, sess in b.pairs()]
                with lock:
                    for i, m in enumerate(b.messages):
                        rec_to_dia[f"{b.request_id}#{i}"] = m.dia_id
                code, dt = add(b.request_id, b.user_id, b.session_id, messages)
                out.append((b.request_id, code, dt))
            return out

        with ThreadPoolExecutor(max_workers=8) as pool:
            for res in pool.map(push, convs):
                for request_id, code, dt in res:
                    add_times.append(dt)
                    blocks += 1
                    if code != 200:
                        add_errors.append({"request_id": request_id, "code": code})

        search_errors = []
        over_top_k = []
        search_times = []
        chars_total = 0
        hit_rate_num = 0
        asked = 0

        def ask(q):
            """One question over HTTP; returns what we need to score it."""
            gold = {str(d) for d in (q.get("evidence") or [])}
            user_id = f"eval:local:locomo_refined:{q.get('sample_id')}"
            code, dt, data = search(q.get("question") or "", user_id, top_k=100)
            chars = sum(len(h.get("content") or "") for h in data)
            got = set()
            for h in data:
                # the id is "<request_id>#<i>", a neighbour adds "#nb<record_id>"
                base = (h.get("id") or "").split("#nb")[0]
                dia = rec_to_dia.get(base)
                if dia:
                    got.add(dia)
            return {"qa_id": q.get("qa_id"), "code": code, "dt": dt, "n": len(data),
                    "chars": chars, "hit": bool(got & gold)}

        wanted = list(questions)
        asked = len(wanted)
        # the questions are asked in parallel (the platform does not wait for us
        # to be polite); the latency numbers are therefore "under load"
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(pool.map(ask, wanted))
        for a in answers:
            search_times.append(a["dt"])
            if a["code"] != 200:
                search_errors.append({"qa_id": a["qa_id"], "code": a["code"]})
                continue
            if a["n"] > 100:
                over_top_k.append({"qa_id": a["qa_id"], "n": a["n"]})
            chars_total += a["chars"]
            if a["hit"]:
                hit_rate_num += 1

        payload = {
            "corpus_matches_the_pinned_copy": corpus_ok,
            "corpus_questions": len(questions),
            "corpus_expected_questions": corp.PINNED_QUESTIONS,
            "blocks_added": blocks,
            "add_errors": len(add_errors),
            "add_latency_s": percentiles(add_times),
            "questions_in_the_corpus": len(questions),
            "questions_asked": asked,
            "search_errors": len(search_errors),
            "answers_over_top_k": len(over_top_k),
            "search_latency_s": percentiles(search_times),
            "chars_total": chars_total,
            "approx_tokens_total": chars_total // 4,
            "chars_per_answer_mean": round(chars_total / asked, 1) if asked else 0,
            # a measurement, not a pass/fail line
            "evidence_hit_rate": round(hit_rate_num / asked, 4) if asked else None,
            "ruler_reference_hit_rate": 0.7899,   # the ruler, shape a, service path, neighbours on
            "container_peak_memory_mib": round(peak["bytes"] / (1024 ** 2), 1),
            "zeabur_limit_mib": 4096,
        }
        report("the whole public corpus over HTTP", payload)
        # the corpus is the pinned one, every question of it was asked (none
        # skipped), something was written, no errors, and no answer longer
        # than top_k
        return 0 if (corpus_ok and blocks > 0 and not add_errors
                     and not search_errors and not over_top_k
                     and asked == len(questions) == corp.PINNED_QUESTIONS) else 1
    finally:
        stop_sampling.set()
        stop_container()


if __name__ == "__main__":
    raise SystemExit(main())