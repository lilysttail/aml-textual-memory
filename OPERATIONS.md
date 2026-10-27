# Operations

How the service is put together, what every switch does, and how the ruler is run.

## Architecture

```
app/     the shell: HTTP contract, API key, platform field mapping,
         the rendering of one message, the choice of the embedding provider
core/    the retrieval core: records, recall, fusion, rerank, neighbours,
         isolation, idempotency, snapshot rebuild. Imports nothing from app/,
         reads no environment variable, carries no platform field name.
eval/    the ruler (eval/ruler) and the tools that are not track-specific
         (embed_report, mem_guard)
```

The service is a thin shell over the core: `POST /add` writes structured
records through the core, `POST /search` asks the core and renders what comes
back. The core is written so that it can be lifted out and reused (another
consumer can reuse it); `core/README.md` has the
interface, the field mapping and the provenance of every copied part.

## Configuration (see `app/config.py`)

| env | default | meaning |
|---|---|---|
| `DB_PATH` | `data/memory.db` | SQLite file (one file holds the core's tables) |
| `API_KEY` | `dev-key-change-me` | auth for /add and /search. Never shown in `repr(Settings)`. The image does not carry one: see `REQUIRE_REAL_API_KEY` |
| `REQUIRE_REAL_API_KEY` | `0` | set to `1` by the Dockerfile only. The service then refuses to start unless `API_KEY` is a real key — missing, blank, `change-me` or `dev-key-change-me` stops it, naming `API_KEY`. A local run leaves it off and needs no key |
| `STEM_ENGLISH` | `0` | English word-form matching of the keyword path. Changing it changes the index format (an existing database refuses to start until its index is rebuilt) |
| `RECALL_PATHS` | `keyword` | comma list of the routes to run: `keyword` / `dense` |
| `RERANK_ENABLED` | `0` | rerank the fused pool's top N with vectors |
| `RERANK_N` | `200` | candidate-pool size fed to rerank |
| `STORE_BUSY_TIMEOUT_MS` / `STORE_WRITE_LOCK_TIMEOUT_MS` | 5000 / 10000 | write-lock budgets |
| `EMBED_MAX_CONCURRENCY` / `EMBED_BATCH` / `EMBED_WAIT_S` / `EMBED_ACQUIRE_S` | 2 / 32 / 60 / 30 | vector-cache gate and batching |
| `BGE_SMALL_DIR` | — | local model directory (development only) |
| `EMBED_PROVIDER` | `local` | `local` (bge-small onnx, development reference) / `dashscope` (text-embedding-v4) |
| `DASHSCOPE_API_KEY` | — | **required** when `EMBED_PROVIDER=dashscope` (startup fails naming this variable). Read only by this name from the process environment — never from a file, never in logs or repr |
| `DASHSCOPE_DIM` | `1024` | output dimension: 64 / 128 / 256 / 512 / 768 / 1024 / 1536 / 2048 |
| `DASHSCOPE_BATCH` | `10` | texts per request (the API takes at most 10) |
| `DASHSCOPE_TIMEOUT_S` | `20` | per-request timeout; on timeout / error the dense route is skipped |
| `DASHSCOPE_MAX_TOKENS` | `2000` | 1–8192; each text is cut to about this many tokens before sending. The count is an **estimate by character count** (~4 ASCII chars or 1 CJK char per token), not the service's tokenizer — calibrate it against the `usage` the service reports after the first real call |
| `DASHSCOPE_BASE_URL` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | the international (Singapore) host; a China-site key needs `https://dashscope.aliyuncs.com/compatible-mode/v1` (a key only works on its own region's host) |
| `CONTENT_MAX_CHARS` | `4000` | budget for a candidate's rendered text |
| `NEIGHBOUR_CANDIDATES` | `1` | the neighbouring turns as candidates of their own. **On** in the submitted version: the ruler says +3.98 points of evidence recall for +10.3% characters. Set it to 0 to turn it off |

Negative timeouts, unknown recall paths and invalid numbers fail at startup.

### Write-time settings (the core remembers them; a mismatch stops the start)

The core's FTS index is built from the terms a tokenizer produced. Change the
tokenizer (or the write format) and the old index still answers with the old
terms — silently fewer hits. So the core records the values its data was
written with (table `core_write_settings`: the tokenizer's key and the write
format version) and compares them at startup:

- **empty database** → the current values are recorded;
- **data with the same values** → starts;
- **data with different values, or no record at all** → refuses to start, names
  every item that differs, and says to rebuild the index (write a fresh
  snapshot) or clear the database (`python scripts/wipe.py`). Nothing is
  changed, nothing is backfilled.

## The ruler (evidence recall, not answer accuracy)

```
py -m eval.ruler run --data <public copy> --out <dir> --shape a|b|c|d [--neighbours 0|1]
```

The public copy lives outside the repository
(`eval_data/locomo_refined/<commit>/`), is never committed and is deleted with
the rest of the evaluation data. The ruler writes the conversations into a
scratch core, asks one search per question and reports whether the gold
evidence turns came back. It never answers a question and never calls a
model. Definitions and the last numbers: `eval/ruler/README.md`.

## Operations

- **Wipe** — `python scripts/wipe.py [db_path]` clears the core's tables. A
  database from an older version (missing a table) is not an error: the
  missing tables are skipped and named in the output, and the wipe does not
  create anything. Run it before and after an evaluation.
- **Evaluation data** — the platform's data and any copy of it is used for
  that evaluation only and is deleted within 30 days of it finishing.
- **Concurrency** — writes are serialized by one in-process lock (FIFO, with a
  bounded wait); a wait that runs out answers `503` so the platform retries.
  Reads are not gated (WAL).
- **Deployment** — `Dockerfile` runs `uvicorn app.main:app` on `$PORT` (8080).
  The service must be reachable over public HTTPS; the API key comes from the
  environment, never from a file.

## Submission deployment

The image built from the `Dockerfile` is the submitted service. Its configuration is the one the answer
ruler decided on (below); it is written into the Dockerfile on purpose, not left to code defaults or to the
host's environment page, so a variable missed on the host cannot quietly turn the meaning-based path off.

**Deployed commit:** `<fill in: the commit the platform builds>`

### What the image carries

| env | value | why |
|---|---|---|
| `RECALL_PATHS` | `keyword,dense` | keyword + meaning-based recall, as measured |
| `EMBED_PROVIDER` | `dashscope` | the academic track's embedding model, `text-embedding-v4` |
| `RERANK_ENABLED` | `0` | not part of the measured version |
| `STEM_ENGLISH` | `0` | not part of the measured version |
| `NEIGHBOUR_CANDIDATES` | `1` | on in the measured version |
| `EMBED_MAX_CONCURRENCY` | `64` | places at the embedding gate (code default 2, used by the measured serial run). A search never waits for a place: with none free it drops its meaning-based half and answers from keywords alone, silently. At 2 places, 16 / 32 simultaneous searches lost it for 94% / 93% of the questions; at 64 places, 16 / 32 / 64 simultaneous searches lost it for none (`eval/ruler/README.md`, "Concurrency of the embedding path"). It only decides whether a search is skipped, never how results are ranked |
| `REQUIRE_REAL_API_KEY` | `1` | refuse to start without a real `API_KEY` |
| `DB_PATH` / `PORT` | `/data/memory.db` / `8080` | |

Everything else is at its code default, which is what the measured run used too (`DASHSCOPE_DIM` 1024,
`DASHSCOPE_BATCH` 10, `DENSE_BACKFILL_M` 8, `DENSE_BACKFILL_T` 2.0 s, …). `tests/test_deploy_configuration.py`
pins the switches and those measured defaults against the Dockerfile and the code, and fails if either side moves alone.

### What the host sets

| env | what | notes |
|---|---|---|
| `API_KEY` | the key the platform sends with every request | the same value the platform has registered. Missing, blank or a development placeholder: the service **does not start** and the error names `API_KEY` |
| `DASHSCOPE_API_KEY` | the embedding provider's key | missing or blank: the service **does not start** and the error names `DASHSCOPE_API_KEY` (never its value) |
| `DASHSCOPE_BASE_URL` | only if the key belongs to the China site | the default is the international host; a key only works on its own region's host (otherwise HTTP 401 `invalid_api_key`) |

No key is ever written into the image, a file or a log. Mount a persistent volume at `/data` if the host can,
so a restart does not drop what the platform has already written.

### Read-back after a deployment

"The service answers" proves nothing: with the meaning-based path off it answers too. Run, from any machine:

```
set AML_API_KEY=<the service's API key>
python scripts/deploy_check.py --base https://<the deployed address>
```

It does two things and exits 0 only if both pass: (1) `GET /health` must report the running configuration
(`config.recall_paths` contains `dense`, `embed_provider` is `dashscope`, `rerank_enabled` and `stem_english`
are `false`, `embed_max_concurrency` is `64` — the acceptance value, read back from the running core, so a host
variable that put the gate back to the default 2 fails the check); (2) it writes four memories under a throw-away user and asks a question that shares no content
word with the memory it needs — keywords find nothing, so the memory can only come back through the
meaning-based path. The key is read from the environment only and never printed. A request that cannot be made or
read ends as `FAIL` with a reason code (`network_unreachable`, `timeout`, `tls_error`, `not_json`), never a traceback;
exit code 1.

### Which version is running

`/health` also reports `build`: a 12-character content fingerprint of what the image carries, worked out by
`scripts/tree_digest.py` over the Dockerfile, `requirements.txt`, `app/`, `core/` and `scripts/` (line endings do not
matter) and stored in the image when it is built. `deploy_check.py` requires it to be there. To prove that the running
service is the published version:

```
python scripts/verify_deploy.py --base https://<the deployed address> [--repo <URL>] [--tag submission-1]
```

It clones the tag, computes the fingerprint with the clone's own `tree_digest.py`, and compares it with the live `build`:
one line, `MATCH` / `MISMATCH` / `FAIL <reason>`; exit code 0 only for `MATCH`. No key is needed, and nothing is written
except a temporary clone that is removed afterwards.

### Where the decision comes from

The answer ruler's one-time run on the locked conversations (see `METHODS.md`, section 5): the meaning-based
path was switched on because all three gates passed. Not a prediction of the leaderboard score.
