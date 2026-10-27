# Third-party notices

One line per borrowed thing: where it came from, licence, what we took, what we
changed.

This repository starts at its first public commit. The project's earlier development history is kept
privately; code carried over from it is our own work and is marked as such in the provenance headers,
with one exception: the first-cycle entry listed first below.

## Code

- **tlysanhuo/agent-memory-challenge** (https://github.com/tlysanhuo/agent-memory-challenge; GitHub account
  `tlysanhuo`) — the first-cycle textual-track entry for the Agent Memory Leaderboard, MIT licence. We used it
  as a reference for the first version of this service's HTTP shell, file by file:
  - app/models.py — the field names and structure of the Add / Search contract (request_id; messages with
    role, timestamp and content; a search hit whose content carries "[YYYY-MM-DD HH:MM] role: text") follow
    that entry, because the official API Guide could not be read at the time. We wrote the classes ourselves;
    whether any passage is word for word the same we cannot say without the original.
  - app/auth.py — the contract behaviour (the three accepted headers Bearer, Token and X-Api-Key; a public
    /health) was confirmed from that entry; the code is ordinary FastAPI written to that behaviour.
  - scripts/smoke_local.py — only the idea of having a local smoke script comes from it; the content is ours
    (our own payloads, idempotent replay, a per-user isolation check).
  - tests/test_contract.py — no code taken; written from the contract. It covers the same points as that
    entry's tests because the contract requires them.

  Idempotency and concurrency are our own: a unique key in the database plus one in-process write lock
  (core/db.py, tests/test_write_lock.py), not that entry's way of claiming a request before writing it.
  **Limits of this entry:** that repository was no longer publicly reachable when we checked on 2026-10-27
  (anonymous and signed-in GitHub requests both answer "not found"; no copy at Software Heritage or the
  Wayback Machine). We could not compare line by line, and we could not read its LICENSE file, so we cannot
  name the copyright holder of that licence. This entry is written from our own project records, not from a
  diff.

- **SQLite FTS5 + BM25 scoring** — the FTS5 module of SQLite (public domain).
  We took the facts its documentation states: the bm25 function returns a
  negative number where a *more* negative value means a better match, so we map
  it to a positive score with -bm25 / (1 - bm25) (bm25 = 0 -> 0, -0.5 -> 0.33,
  -10 -> 0.91) so that "higher score" means "more relevant" everywhere in the
  service. The index stores the terms a tokenizer produced (the same tokenizer
  at write time and at query time).

- **Reciprocal-rank fusion** — Cormack, Clarke, Buettcher, "Reciprocal Rank
  Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR
  2009). We use the standard formula with the usual fixed constant k = 60:
  a candidate's fused score is the sum over routes of 1 / (k + its rank in
  that route). The single-route case is not special-cased — the formula is
  strictly monotonic, so one route comes out in its input order.

- **Vector cache with single-flight de-duplication** — the shape used in this
  project's own earlier service (app/vector_cache.py of this project's earlier
  service, not included here) and in common Go/Java caching libraries: one caller computes a
  missing vector while the others wait on it, the result is cached by content
  hash, and a failure hands the same error to every waiter. We wrote the
  implementation; we fixed a leak in it (a flight claimed before a read could
  fail was never released, so later requests waited for the timeout).

- **OpenAI-compatible embedding request shape** — the protocol facts the
  OpenAI embeddings API defines, as used for the DashScope-compatible
  endpoint: POST base_url + /embeddings, an Authorization Bearer header, a JSON
  body of model + input (plus dimensions, encoding_format), and the
  data[].embedding / usage reply shape. We wrote the client (app/embedder.py,
  core/embedder.py): standard library only, no framework layer, no environment
  reads, our own batching, truncation, L2 normalisation, and a single
  EmbedUnavailable type (the core's copy) that the search layer turns into
  "skip this route". The batch-size limit, the dimension list and the
  input-token limit come from Alibaba's Model Studio embedding documentation.
  An earlier version of this client was written after reading the DashScope
  embedding transformation of BerriAI/litellm (MIT for everything outside the
  enterprise directory; that file is outside it) — we took only the protocol
  facts above, not the code.

- **Single-writer SQLite discipline** — the pattern used in this project's own
  earlier service: one in-process write lock (FIFO, with a bounded wait) in
  front of BEGIN IMMEDIATE, WAL journaling, and a "busy" error that the HTTP
  layer turns into 503 (retryable) rather than a 500. We carried the shape
  over into the core (core/db.py).

## Copied parts inside the core package

Every file in core/ that came from the project's earlier service layer carries a provenance header (the
source file, the date, and what we changed); that earlier layer is this project's own work and is not part of
this repository. core/README.md lists them all:

| in core/ | from | what we changed |
|---|---|---|
| db.py | app/db.py | the table registry is the core's own eight tables; the write lock can be shared with another Database on the same file |
| store.py | app/memory_store.py | one content column became structured columns; added the turn-number cursor, the neighbour lookup, the atomic snapshot rebuild, and the tokenizer plug |
| recall.py | app/recall.py | only the keyword and dense routes; the dense one gained exclude_lines |
| rerank.py | app/retrieval.py | user_id -> space_id, mem_id -> record_id, the text comes from the core's own rows |
| embedder.py | app/embedder.py | the provider name is gone: the adapter is parameterised (HttpEmbedder) and the model is chosen by the caller |
| vector_store.py | app/vector_store.py | the table is core_vectors |
| vector_cache.py | app/vector_cache.py | the import path; the flight leak was fixed |

- **snowballstemmer 3.1.1** — the Snowball project’s official Python package
  (PyPI snowballstemmer, BSD-3-Clause, no dependencies). We call its
  english algorithm (= the complete Porter2 stemmer) for the optional
  English stemming of the built-in keyword tokenizer; we changed nothing in
  the package. Project: https://github.com/snowballstem/snowball. Known
  over-stems of the algorithm (e.g. evening -> even) are accepted and not
  special-cased.

- **BAAI/bge-small-en-v1.5** (FlagEmbedding, BAAI) — MIT License. Used only for the optional local dense-recall path
  (`BGE_SMALL_DIR`, `app/embedder.py`, `requirements-dev.txt`); the model weights are not included in this repository.

## Data

- **LoCoMo-Refined** — mem-eval-suite/LoCoMo_refined, commit
  887091190789e8d6760e70b9edd696539923dc4f (2026-05-18), licence
  **CC BY-NC 4.0** (the LICENSE.txt in that repository; GitHub's own licence
  field reads NOASSERTION, the file is what counts). It is a **modified
  version** of the upstream LoCoMo release by Snap Research
  (snap-research/locomo, ACL 2024 / arXiv:2402.17753, also CC BY-NC 4.0);
  the attribution and the list of modifications are in that repository's
  NOTICE, and we keep it: the upstream project, the paper, the licence, and
  the fact that our copy is a modified version of the upstream release.
  We use it for the **local ruler only** (evidence recall,
  eval/ruler/): the copy lives outside the repository, under
  eval_data/locomo_refined/<commit>/, is never committed, never enters the
  service, and is deleted with the rest of the evaluation data.
  Non-commercial use; not part of the released code.

- **LoCoMo-Refined judge** — the `refined` judge prompt (`eval/ruler/refined_judge_prompt.txt`, byte for byte), and the label
  reading, the gold-answer rules and the prediction cleaning in `eval/ruler/judge.py` (adapted), from mem-eval-suite/LoCoMo_refined,
  commit 887091190789e8d6760e70b9edd696539923dc4f (`src/llm_judge.py`, `src/llm_judge_runtime.py`, `src/evaluate.py`), licence
  **CC BY-NC 4.0**; a modified version of snap-research/locomo (see the data entry above and that repository's NOTICE). Used only in the
  local ruler, non-commercial; not part of the released service.

- **The evaluation data the platform sends us** — the chunks, the
  user_id / session_id identifiers and the questions. Used for the current
  evaluation only: not for training or fine-tuning, not for product
  analytics, not for rebuilding a dataset, not forwarded; deleted within 30
  days of the evaluation finishing. We never receive the gold answers.