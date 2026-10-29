# Method disclosure

What this service is, how it works, what came from elsewhere, and what it
deliberately does not do. (The submitted version is the commit tagged `submission-1`; see section 6.)

## 1. What we built

A memory retrieval service for long multi-session conversations. It stores each
turn as a **structured record** — who said it, when (if the source knows), in
which session, which turn — and returns **evidence** for a question: ranked
turns, in the order we believe is most useful, with the text exactly as it
was written.

The service answers nothing itself. The platform's answer model reads the
evidence we hand over.

## 2. The design, and where it lives

| decision | where |
|---|---|
| One retrieval core, no platform inside it | `core/` (imports nothing from `app/`, reads no environment variable) |
| Records carry who / when / which session / which turn | `core/models.py` |
| Isolation by `space_id` with a composite primary key `(space_id, record_id)` | `core/db.py`, `core/store.py` |
| Keyword recall (FTS5 + BM25) and a dense (meaning-based) route, **on** in the submitted version | `core/recall.py`, `core/store.py`, `Dockerfile` |
| Fusion of several routes by RRF (k = 60), optional vector rerank | `core/rerank.py` |
| The turns around a hit (same session, still valid, in window order) | `core/store.py:neighbors` |
| Writes: synchronous, idempotent per `(space_id, batch_id)`, one transaction | `core/store.py:write` |
| A whole snapshot can be rebuilt atomically | `core/store.py:replace_snapshot` |
| The core remembers how it wrote its data (tokenizer, write format) and refuses to start on a mismatch | `core/write_settings.py` |
| The service is a thin shell: HTTP contract, API key, field mapping, rendering, provider choice | `app/` |
| A message without a timestamp: the event time stays empty, the rendering invents no date | `app/main.py` |
| The content handed to the platform is the candidate itself; the neighbouring turns are candidates of their own behind a switch that is **on** in the submitted version (the ruler says it is worth 3.98 points more questions with an evidence turn, for 10.3% more characters) | `app/main.py`, `app/config.py` |
| Input is checked no more strictly than the official contract: a missing required field is a 422 (logged by field name and kind only, never the request's text); everything the contract allows is taken — a message with no text is skipped (not the batch), a missing role is `user`, a timestamp that cannot be read as a date is "no time", an empty session id makes the batch its own session; a blank query or a `top_k` below 1 answers with no results (section 2a) | `app/models.py`, `app/main.py` |
| The image carries the core, not only the shell | `Dockerfile`, `tests/test_docker_image.py` |
| The evidence-recall ruler (public LoCoMo-Refined; no answering, no scoring, no cost) | `eval/ruler/` |

## 2a. Why the input check was loosened (2026-10-29)

The first formal evaluation stopped with `ADD_API_CONTRACT_MISMATCH · HTTP 422`: our `/add` input check was stricter than the
official contract (it refused, for example, blank content, a timestamp before 1970, after 2100 or with a fraction, an empty session
id, an empty message list), the platform does not retry a 422, and some batches of the evaluation data were refused. We did not log
why a request was refused, so which of these the data hit is not known. The check now asks only what the contract asks: a required
field must be present; everything else is taken and made usable message by message (the list in section 2). What is still refused is
logged — which field, what kind of problem — and never the request's text. Retrieval, the storage format and the configuration did
not change: an ordinary request is stored and answered exactly as before, and a batch sent again writes nothing twice (writes are
idempotent per `(space_id, batch_id)`).

## 3. What came from elsewhere

Everything borrowed is listed in `THIRD_PARTY_NOTICES.md` — author, source,
licence, what we took, what we changed. The short list: the FTS5 + BM25
scoring direction (SQLite documentation), reciprocal-rank fusion (Cormack,
2009), the single-flight vector cache, the OpenAI-compatible embedding request
shape, and the public LoCoMo-Refined benchmark (used by the local ruler only).

This repository starts at its first public commit. The project's earlier development history is kept
privately; code carried over from it is our own work and is marked as such in the provenance headers.

## 4. What we deliberately do not do

- We do not generate answers in `Search`; we hand over evidence.
- We do not hard-code anything for a known question or dataset.
- We do not let one `user_id` see another's memory.
- We do not train on anything, and we do not carry evaluation data out of the
  evaluation environment: the evaluation data and its copies are used for that
  evaluation only, and are deleted within 30 days of it finishing.
- We do not use a model to rerank or answer beyond what the academic track
  allows (embeddings: `text-embedding-v4`; LLM components: `gpt-4o-mini`;
  rerankers are unrestricted).

## 5. What the local ruler says (and what it does not say)

The ruler measures the retrieval side only: did the search bring back the
turns the dataset marks as the evidence for a question? It never answers a
question and never calls a model. Its numbers are in
`eval/ruler/README.md`. That ruler runs the keyword path only (no embedder);
the recall numbers below are therefore **keyword-only numbers** and were not
re-measured with the meaning-based path on. Whether to switch that path on was
decided by a different measurement, in section 5a.

In short: **the submitted version has the neighbouring turns switched on**
(the switch is `NEIGHBOUR_CANDIDATES`, on by default; the environment can turn
it off). With them, the search finds at least one gold evidence turn for
**78.8%** of the questions (74.9% without them, for 10.3% more characters);
the first gold turn sits at position 1 for 26.8% of the questions in the
candidate pool. The numbers come from the public LoCoMo-Refined set, commit
887091190789e8d6760e70b9edd696539923dc4f.

They are **not** a prediction of the leaderboard score: the platform's answer
model, its prompts and its scoring decide the rest.

## 5a. Why the meaning-based path is on

The submitted version runs keyword recall **and** the dense route (embeddings: `text-embedding-v4`; no rerank,
no stemming, neighbouring turns on). It was switched on by a pre-set decision rule, applied once to a part of
the public LoCoMo-Refined set that had been locked away, separate from the conversations used for tuning: five
conversations, 734 questions.

How it was measured: every question was answered twice by each of three stand-in answering models — once
from the keyword-only search results, once from the keyword + dense results (top_k 100 both times) — and each
answer was judged against the gold answer. The rule had three gates, all of which had to pass:

1. across the three models, at least 4% more questions answered correctly (at least 89 of 2,202 answers);
2. at least 3 of the 5 conversations better with the dense route;
3. no single model with more than 3% of its questions net fewer correct with the dense route (at most 22 of 734).

Result: **236 more correct answers out of 2,202 (+10.72 percentage points); all 5 conversations better** (186 → 251,
229 → 299, 305 → 336, 243 → 267, 244 → 290 correct); every model came out ahead (net +64, +75 and +97 questions of
734 each). Some individual questions got worse (24, 31 and 29 per model) — the figures above are net. Two of the
734 searches could not use the embedding service (a transient error) and ran on keywords alone, which can only make
the dense result look slightly worse. The run was one-time: the gates were fixed in advance and there was no
re-run.

What this is **not**: a leaderboard score. The answering models are **stand-in answering models**
(`qwen3.7-flash-2026-07-15` pinned; the other two are provider aliases that cannot be pinned) with a **stand-in
answer prompt**; the judge is a stand-in too. The platform's answer model, prompts and scoring decide the rest.

## 6. The submitted version

Submitted version: the commit tagged `submission-1` of this repository (the same commit is given on the application form).
