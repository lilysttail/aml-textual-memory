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
| `DENSE_QUERY_T` | `3.0` | seconds the query vector may take, counted from when it is asked for, waiting for a place at the embedding gate included. Past it the search answers from keywords alone (logged as `dense query vector skipped`) |
| `DENSE_BACKFILL_M` / `DENSE_BACKFILL_T` | 8 / 2.0 | after the query vector: at most M records still without a vector get one during the search, within T seconds; the gate is not waited for |
| `SEARCH_HEAVY_CONCURRENCY` | `0` | how many searches may build a space's in-memory index or compute similarities at once; `0` = the CPUs the process may use (the container's CPU quota, not the host's count) |
| `SEARCH_HEAVY_WAIT_S` | `15` | how long a search waits for that; past it the search answers `503` (the platform retries) |
| `SEARCH_CACHE_MIB` | `640` | memory budget of the in-memory dense index: the shared vector pool plus every space's index. Least recently used spaces leave when it is full |
| `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `1` | set by the service package before numpy loads (a value in the environment is kept). In a container numpy would otherwise start one thread per **host** CPU and stall the container's CPU quota |

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
  create anything. Run it before and after an evaluation. It clears the evaluation data only: the record of the
  write settings (`core_write_settings`, above) stays, so a running service that is not restarted after a wipe can
  keep writing and still starts at its next restart; with no data in the database the start-up replaces that record
  with the current settings. It can be run while the service is running: it never deletes the database's side files
  (`-wal`, `-shm`), which belong to every process that has the database open (deleting them from under the service
  loses the wipe and makes every new connection fail with "disk I/O error"). It ends by asking SQLite to fold its log
  into the database file and prints the outcome as the last line, `checkpoint: truncated` (done) or
  `checkpoint: partial` (another connection was reading; the wipe is committed and visible to everyone anyway,
  SQLite folds the rest in later).
- **Evaluation data** — the platform's data and any copy of it is used for
  that evaluation only and is deleted within 30 days of it finishing.
- **Concurrency** — writes are serialized by one in-process lock (FIFO, with a
  bounded wait); a wait that runs out answers `503` so the platform retries.
  Reads are not gated (WAL), except the heavy part of the meaning-based path (building or refreshing a space's
  in-memory index, computing similarities): at most `SEARCH_HEAVY_CONCURRENCY` at once, and a search that cannot
  get a place within `SEARCH_HEAVY_WAIT_S` answers `503`. A search waiting for another search's build or refresh
  of the same space waits at most `SEARCH_HEAVY_WAIT_S` too, then answers `503`.
- **The in-memory dense index** (`core/dense_index.py`) — the first search in a space reads the space once (its
  records, the sha256 of each text, their stored vectors) and keeps a "record → vector" table in memory; later
  searches only compute the query vector and one similarity pass. Searches that arrive at a cold space together
  share one read (and searches that arrive after new vectors share one look-up of the missing ones). A vector is
  held once however many spaces contain its text (a shared pool with reference counts); when the pool plus the
  spaces' tables (their id / hash / vector lists and strings) pass `SEARCH_CACHE_MIB`, the least recently used
  spaces that no search is reading at that moment are dropped and their vectors leave the pool unless another
  space still uses them (a dropped space is read again on its next search, with the same result). A space being
  read is never dropped under its search; one replaced or wiped while read stays counted until that search ends.
  So the total can stand above the budget only while every kept space is being read, and returns under it
  when those searches end.
  Every use first reads three change counters from `core_generations` (one indexed statement): `epoch`,
  `space:<user_id>` and `vectors:<model>`. Each writer bumps them **in the same transaction** as its change:
  `/add` (and any write through the core), a snapshot replace, every new stored vector (write time, search-time
  backfill, `scripts/backfill_vectors.py`), and `scripts/wipe.py`, which bumps every counter and `epoch` (it never
  clears them: they only grow). So a wipe, a re-write of the same ids after a wipe, or a backfill run while the
  service is up is seen by the next search without a restart: a wipe drops everything kept; a changed space is
  read again; new vectors are looked up only for the records that were waiting for one. A change made around
  these paths (editing the tables by hand) is **not** seen until the space changes again or the service restarts.
- **Records still without a vector** — `/health` does not count them (that needs a full read of every space). Run,
  inside the container: `python scripts/count_missing_vectors.py /data/memory.db [--by-space]` (read only: no table
  or row is created or changed and the main database file is not written — SQLite may still create or use the WAL
  side files `-wal` / `-shm` next to it, as any reader does; a database without the core's tables is an error,
  never "0"). The last line is one
  JSON object with `missing_vectors`. `scripts/backfill_vectors.py` computes them.
- **Deployment** — `Dockerfile` runs `uvicorn app.main:app` on `$PORT` (8080).
  The service must be reachable over public HTTPS; the API key comes from the
  environment, never from a file.

## Submission deployment

The image built from the `Dockerfile` is the submitted service. Its configuration is the one the answer
ruler decided on (below); it is written into the Dockerfile on purpose, not left to code defaults or to the
host's environment page, so a variable missed on the host cannot quietly turn the meaning-based path off.

**Deployed commit:** the commit the tag `submission-1` points to (the same commit is given on the application form).

### What the image carries

| env | value | why |
|---|---|---|
| `RECALL_PATHS` | `keyword,dense` | keyword + meaning-based recall, as measured |
| `EMBED_PROVIDER` | `dashscope` | the academic track's embedding model, `text-embedding-v4` |
| `RERANK_ENABLED` | `0` | not part of the measured version |
| `STEM_ENGLISH` | `0` | not part of the measured version |
| `NEIGHBOUR_CANDIDATES` | `1` | on in the measured version |
| `EMBED_MAX_CONCURRENCY` | `64` | places at the embedding gate (code default 2, used by the measured serial run). A search's query vector waits for a place up to `DENSE_QUERY_T`; with none free by then the search drops its meaning-based half and answers from keywords alone. (Measured before that wait existed: at 2 places, 16 / 32 simultaneous searches lost it for 94% / 93% of the questions; at 64 places, 16 / 32 / 64 simultaneous searches lost it for none — `eval/ruler/README.md`, "Concurrency of the embedding path".) It only decides whether a search is skipped, never how results are ranked |
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
| `DB_PATH` | optional: where the SQLite file lives | the image default is `/data/memory.db`; a value set on the host wins over it. A service that starts on a database written under other settings refuses to start (the start-up check described above), so a new file name is also how a host starts clean |

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
variable that put the gate back to the default 2 fails the check). It also prints, without judging them (the
number of CPUs differs between hosts), the in-memory dense index's settings as they took effect, which `/health`
reports under `config`: `search_heavy_concurrency` (the resolved number of slots — `0` in the settings shows here as
the CPUs the process may use; on a host without the cgroup v2 `cpu.max` file that is the CPU count the process sees,
which can be the host's, so set `SEARCH_HEAVY_CONCURRENCY` there), `search_heavy_wait_s`, `dense_query_t`,
`search_cache_mib` and `numpy_threads` (the three thread variables as the process has them; `"1"` unless the host
set them); (2) it writes four memories under a throw-away user and asks a question that shares no content
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
