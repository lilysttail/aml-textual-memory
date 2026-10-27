# core — the shared retrieval core

A retrieval core that does not care where its data comes from. The competition service is a thin shell around it; another
consumer (a conversation-log library) can be a second shell. The whole `core/` directory is meant to be lifted out and
reused as it is.

It does five things: write, retrieve, add the lines around a hit, order by time, replace a whole snapshot in one step.
It does **not**: generate answers, rewrite the original text, cut results into a list, or know where the data came from.

## Boundary

| | The core | The shell (the competition service / a second consumer's thin shell) |
|---|---|---|
| What is stored | structured records (who said it, when, which window, which turn, the text) | mapping of the platform's fields, authentication, the HTTP contract |
| Rendering | none | the `[YYYY-MM-DD HH:MM] role: text` prefix, neighbours merged into content, a character budget |
| Models | only knows the `embed(texts)` protocol | which model to use (the competition side is locked to the one the academic track names), provider settings |
| Configuration | built by the caller and passed in | reads environment variables |
| Isolation | `space_id` (the first part of the composite primary key) | the platform's user id -> `space_id` |
| Idempotency | `batch_id` | the platform's request id -> `batch_id` |

The core imports nothing from `app.*` and reads no environment variable (both are watched by a test:
`tests/test_core_is_clean.py`).

## Usage

```python
from core import CoreConfig, Record, build

core = build(CoreConfig(db_path="..."))          # builds its own database
# a shell can also hand in its own Database (core.db.Database(path, lock=db.write_lock))

core.write(records, space_id="user-1", batch_id="b1")     # idempotent: the same batch twice lands once
core.replace_space_snapshot("user-1", snapshot_rows, snapshot_id="v2")   # rebuild a whole space + atomic generation switch
hits = core.search("that sentence", space_id="user-1", top_k=5)   # evidence (original text + metadata + score)
cands = core.recall_candidates("that sentence", space_id="user-1", limit=50)   # the candidate pool the ruler uses
```

Every hit carries `neighbors_before` / `neighbors_after`: the immediately adjacent records of the same space and the same
session that are **currently valid** (in the current window order, not in an old turn number that may have been reordered).
A row with `valid = false` can neither be found nor shown as a neighbour.

## Record

| Field | Type | Required | A second consumer's export | The competition `/add` |
|---|---|---|---|---|
| `record_id` | str | yes | source id | `f"{request_id}#{i}"` |
| `session_id` | str | yes | window id | `req.session_id` |
| `turn_index` | int | given or not | the export's sequence number | the core issues it atomically per (space, session) |
| `speaker` | str | yes | a name looked up at read time | `m.role` |
| `direction` | str | no | direction | empty |
| `line` | str | no | line (terminal / chat / web ...) | empty |
| `ts` | str | yes | date + time | `m.timestamp` |
| `text` | str | yes | the text (verbatim) | `m.content` (bare text, no prefix) |
| `sec_flag` | bool | no | `sec_flag` | empty |
| `valid` | bool | yes | verdict state | always true |
| `state` | str | no | verdict state | empty |
| `created_at` | str | no | empty | empty (defaults to `ts`) |
| `space_id` | str | yes (given at call time) | one space per consumer | `req.user_id` |
| `batch_id` | str | no (given at call time) | empty | `req.request_id` |
| `meta` | dict | no | empty | empty |

**The primary key is `(space_id, record_id)`**: two spaces can use the same `record_id` and each keeps its own. Look-ups,
neighbours and batch claims always carry `space_id`.

## Tables

```
core_records(space_id, record_id, session_id, turn_index, speaker, direction, line,
             ts, created_at, text, sec_flag, valid, state, meta, batch_id,
             PRIMARY KEY(space_id, record_id))
core_fts(space_id, record_id, session_id, speaker, body)      -- FTS5
core_vectors(model, content_hash, dim, vec)
core_batches(space_id, batch_id, PRIMARY KEY(space_id, batch_id))
core_sessions(space_id, session_id, next_turn_index, PRIMARY KEY(space_id, session_id))
core_spaces(space_id PRIMARY KEY, snapshot_id, updated_at)
core_write_settings(name PRIMARY KEY, value)   -- write-time settings (tokenizer / write-format version); a wipe leaves it
core_pending_vectors(space_id, record_id, content_hash, reason, retry_count, last_attempt_at,
                     PRIMARY KEY(space_id, record_id))   -- why a vector was not computed at write time (explains, does not decide)
```

The FTS `body` holds the **tokenized word sequence** (writing and querying use the same tokenizer). Without a plugged-in
tokenizer, the built-in English tokenizer treats a whole run of Chinese text as one word (the whole run matches, a substring
does not); a Chinese tokenizer is a plug-in, and plugging one in changes the index version.

## Switches (`CoreConfig`)

| Switch | Default | Meaning |
|---|---|---|
| `paths` | `("keyword",)` | which routes to run (keyword / dense) |
| `rerank_enabled` / `rerank_n` | False / 200 | vector rerank |
| `context_before` / `context_after` | 0 | number of neighbours (the competition service uses 1 each; another consumer may want 1-3) |
| `group_by_session` | True | inside a session, ascending by `ts`; between groups, by the group's best score |
| `top_k` | 10 | number of results |
| `dense_exclude_lines` | `("terminal",)` | these `line` values go only into the keyword index, not the vectors |
| `tokenizer` | None | tokenizer plug (for Chinese, say) |
| `tokenizer_key` | None | the tokenizer's version id (such as `jieba-v1`), written into `core_write_settings` |
| `embedder` / `vector_cache` | None | vector plug (`embed(texts) -> vectors`) |

## Write-time settings: changing the tokenizer changes the vocabulary

The FTS index is built with the tokenizer that was in use **at write time**. Change the tokenizer and the old index still
answers with the old words: fewer hits, silently, with no error. So the core records the settings its data was written with
(`core_write_settings`: the tokenizer id + the write-format version), and **refuses to start** when the database already
holds data and the settings do not match, saying how to rebuild (write a fresh snapshot).

```python
core = build(CoreConfig(db_path="...", tokenizer=my_cutter, tokenizer_key="jieba-v1"))  # gate:allow-secret
```

When you change the tokenizer, change `tokenizer_key` too (or rebuild the index); otherwise the start is stopped.

## Parts carried over (copies; each file header has a provenance paragraph)

These come from this project's own earlier service layer, which is not part of this repository.

| From | Used in | What changed |
|---|---|---|
| `app/db.py` | `core/db.py` | the table list is the core's own eight tables; the shared write lock can be injected |
| `app/memory_store.py` | `core/store.py` | the single content column is split into structured columns; neighbours, turn numbering and snapshot generation added; the tokenizer became a plug |
| `app/recall.py` | `core/recall.py` | only the keyword / dense routes are kept; dense takes an extra `exclude_lines` |
| `app/retrieval.py` | `core/rerank.py` | user_id -> space_id, mem_id -> record_id, the content is taken from `text` |
| `app/embedder.py` | `core/embedder.py` | provider-specific names removed, the adapter is parameterized (`HttpEmbedder`) |
| `app/vector_store.py` | `core/vector_store.py` | table name `vectors` -> `core_vectors` |
| `app/vector_cache.py` | `core/vector_cache.py` | only the imports changed |

Neither carried over nor rebuilt from scratch: the direction of the BM25 score (bm25 is better when more negative -> a
score where bigger is better), RRF with k=60, the write lock (a single writer with a bounded wait), the `StoreBusy`
semantics, and the single-flight of the vector cache.

## What the core does not contain (the acceptance list)

- The track name, the evaluation platform's fields, the competition's configuration, hard-coded model or provider names:
  none of them.
- Fingerprints (error / path / ident / test), two-level recall (a task layer), authentication, the competition's
  configuration: none of these are in the core.
- Training code, data-migration tools, evaluation data: none. The core moves code, not data.

## Self-test

```
py -m pytest core/tests -q
```
