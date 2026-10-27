# eval/ruler: the ruler (the only entry; it replaced the first ruler's separate entry on 2026-10-23)

What counts as a result is whether the answering model answers correctly with what the memory handed over. Everything this step
measures is a **clue about where things go wrong**, not a score: whether the memories that should have come back did, and what the
store simply cannot answer. Step 1 is local: no network, no model, no money. Nothing here answers a question or judges an answer
(`answerer.py` and `judge.py` are interfaces only).

```
python -m eval.ruler run          --data <copy> --out <dir> --shape a|b|c|d [--neighbours 0|1] [--limit N] [--partition tune|locked|all]
python -m eval.ruler facts        --data <copy>
python -m eval.ruler unanswerable --data <copy> [--stored service|rule]
python -m eval.ruler split        --data <copy> [--rows rows_*.jsonl] [--sizes 3,4,5,6]
```

`<copy>` is the commit copy of the public data (the directory that contains `data/public/`; it lives outside the repository under
`eval_data/locomo_refined/<commit>/`, is never committed, and is deleted with the rest of the evaluation data). Without `--no-pin` a
copy that is not the pinned one is refused. Output is numbers and ids only: no question and no conversation text. Run it with the
project's own interpreter (`.venv_scaffold`).

## What it measures, and how

1. Each conversation is cut into **write blocks** (20 messages or 2,000 words, a block never crosses a session). This is our own
   approximation: the platform's internal cut is not public, so the ruler does not pretend to be the official one.
2. Every block goes through the **competition service** (`ServiceTarget`: the real `/add` and `/search`, in this process, keyword
   path, no embedder) in one of four **shapes** (below).
3. Every question is asked once. Gold is judged by **id** (`dia_id`), never by "the text contains the answer". The neighbours the
   service hands over under another id are registered as aliases of their own turn; without that, "the gold only shows up as a
   neighbour" is missed (the first ruler's one real bug; its first-round conclusion "neighbours add no evidence recall" was wrong).

| metric | meaning |
|---|---|
| hit_rate | share of questions where at least one gold turn came back |
| recall@k | share of questions with a gold turn in the first k (k = 1 / 5 / 10 / 100) |
| mrr | over the questions that found a gold turn, the mean of 1 / rank of the first one (`mrr_all` divides by all questions) |
| chars_total | characters handed to the answering model (in return order; a record counts once) |

**Boundary.** The ruler answers general questions (which path is worth switching on, whether neighbours are on, whether the order
needs work). It never writes a special case for a particular question (the competition forbids it) and its numbers are not score
predictions.

## The baseline, and the old numbers

**The baseline is the service path** (`ServiceTarget`), the thing the platform actually talks to. 1,380 questions (2 of the 1,382
have no evidence), 5,882 records, shape `legacy` (what the first ruler wrote), corpus commit 8870911:

| | hit_rate | recall@1 | recall@5 | recall@10 | recall@100 | mrr |
|---|---|---|---|---|---|---|
| neighbours off | 0.7486 | 0.0964 | 0.4181 | 0.5529 | 0.7486 | 0.3145 |
| neighbours on | **0.7899** | 0.0964 | 0.4181 | 0.5536 | **0.7899** | 0.2989 |

The first ruler (core written directly, no HTTP shell) reported 0.7486 and **0.7884** (recall@1 0.0978, recall@5 0.4123, recall@10
0.5536 / 0.5543, mrr 0.3120 / 0.2971; candidate pool before fusion: recall@1 0.2681, i.e. the evidence is in the pool and the fusion
and ranking push it back). Those numbers are reproduced digit for digit by `--engine core` (see below) and are kept here only as history.

**Why 0.7884 and 0.7899 differ (two questions, traced to the cause).** The old ruler gave each record `turn_index` = its position
**inside the block**, restarting at 0 for every block. Of the 272 sessions, 123 are cut into more than one block, and 900 records
carry a `turn_index` that an earlier record of the same session already has. The neighbour assembly orders by `turn_index`, so
across blocks of one session it sometimes took the wrong neighbour. The service's `/add` does not pass `turn_index`; the core then
numbers the turns by one cursor per session (`core/store.py`, `_next_turn_index`), so a session's turns are consecutive and never
collide. Checked three ways, all on the real copy: (1) the old ruler with `turn_index` left out gives the service's numbers;
(2) the core written directly with a session-wide `turn_index` gives the service's numbers (hit_rate 0.7899, recall@5 0.4181, mrr
0.2989); (3) the record-id spelling and the time are not the cause (each tried). Net effect: 4 questions flip in hit_rate, 3 gained
and 1 lost, +2 overall. **The old number was the ruler's defect; the service path is right and is the baseline from now on.** The
service does not need a fix.

`CoreTarget` / `--engine core` is kept **only** so the reproduction test can show that the new ruler's cutting, evidence matching and
diagnosis give the old ruler's numbers digit for digit (the old ruler's numbers on the synthetic mini corpus are pinned in
`tests/test_ruler_target_run.py`). It copies the old block-local `turn_index` on purpose, including the collisions. Measure nothing
with it; use `ServiceTarget`.

## The four blocks

| block | file | in step 1 |
|---|---|---|
| ① item (what is asked, what a good answer looks like) | `items.py` | done; the fields also fit items with no question and no gold answer |
| ② the thing measured | `target.py` | `ServiceTarget`; `CoreTarget` (reproduction only) |
| ③ whoever uses the result | `answerer.py` | interface only |
| ④ the judgement (right / said "not found" / made it up) | `judge.py` | interface only |

## The four Add shapes (`shapes.py`, defined nowhere else)

`a` role + text · `b` a + millisecond timestamp (the data has a time only on the session, so every message of a session gets the
same millisecond, read as UTC) · `c` b + the speaker's name before the text · `d` c + the picture description. `legacy` /
`legacy_t` are what the first ruler wrote; they exist only to reproduce its numbers.

## "Cannot be answered at all" (`unanswerable.py`)

U1 the question names a speaker and no stored record has that name · U2 a time question and no stored record has a time ·
U3 an acceptable answer is only in the caption of an evidence message and the shape stores no caption. See the module docstring for
the exact wording and what each one does not catch.

- **U2 is counted by category "2"** (the corpus' own label for time reasoning): 299 questions. 243 of them also carry time wording
  (when / what date / what time / which year / how long ago); 56 do not. 36 questions of **other** categories carry such wording.
  `python -m eval.ruler unanswerable` prints that as `u2_wording_sensitivity`, with the 36 ids and how many more questions per shape
  would be unanswerable if U2 were counted by wording (shape `a`: 36 more; `b`, `c`, `d`: none). It is a sensitivity table, not a
  definition.
- U1 is only a lower bound (a speaker's name appears somewhere in the conversation's hundreds of records). Two sensitivity counts sit
  next to it (the evidence text never names the person; any named person missing rather than all).

## Evidence diagnosis (`diagnostics.py`)

Per item: how many of the memories that should have come back did, and the rank of the first one. Matching is by id; neighbours are
candidates of their own and are matched through their own record.

## Tune part and locked part (`split.py`, fixed 2026-10-23, option C)

Cut **by conversation**, never by question (questions of one conversation share one memory). **Locked: conv-26, 41, 42, 47, 50
(734 questions). Tune: conv-30, 43, 44, 48, 49 (648).** The code refuses a corpus whose conversations or counts differ
(`check_partition`).

- `run` defaults to `--partition tune`. `--partition all` describes the whole set and must never be used to choose between versions.
- **The locked part is used once**: the hardened version against the current baseline, nothing in between. `--partition locked` needs
  `--version`, `--declared a,b` (the two versions of the one comparison) and `--ledger <directory>`; each declared version can be
  run on the locked part once, the pair is fixed by the first claim, and a run that stops half-way still counts. A refused run
  writes nothing. The ledger is a directory with one file for the declared pair and one `used.<version>` file per version, each made
  with an exclusive create (it fails if the file is already there), so two processes claiming at the same moment cannot both win
  and different versions cannot overwrite each other. Re-opening the ledger takes a person, in the floor thread.
- **Whether to turn the meaning-based (dense) path on** is decided once, on the locked part, **question by question**
  (`decide_dense`): on only if the overall gain is **at least 4 points** (pooled over the 734 questions) **and at least 3 of the 5
  locked conversations are better** with it; otherwise keyword-only. The threshold is written in the code now
  (`DENSE_MIN_OVERALL_GAIN_POINTS`, `DENSE_MIN_SEGMENTS_BETTER`) and is not to be changed before the locked part has been looked at.
- Tune only for obvious big problems; do not fit the tune part closely.
- What the locked part can tell apart: about 7 points (unpaired 7.3; paired with one tenth of questions differing 3.3; allowing for
  questions of a conversation moving together, design effect 5.15 on the evidence-hit stand-in, about 143 effective questions). A
  difference of a few points cannot be settled on the locked part alone.

## Step 2: one small batch of real answers (`qa_main.py`, `qa_run.py`, `sampling.py`, `qa_report.py`, `chat.py`, `claude_cli.py`)

Only to see **where it goes wrong**, never a score and never compared with platform scores. 48 questions (12 per category) from
the **tuning** conversations only (the locked ones are refused in code), picked by a hash of the question id and written once to
`picks.json`; 12 of them (3 per category) for the cross-check. Search: service path, keyword only, neighbours on, top_k 100.

| group | stored as | handed to the answering model |
|---|---|---|
| G1 | shape a | the 100 returned memories |
| G2 | shape c | the 100 returned memories |
| G3 | shape c | the first 20 of the same search |
| G4 | shape a | only the standard-evidence sentences (an upper bound) |
| G5 | shape c | only the standard-evidence sentences (an upper bound) |

Models (pinned versions): answering `qwen3.7-flash-2026-07-15` (thinking off, temperature 0, at most 128 output tokens);
cross-check `k3-256k` through the Kimi coding API (no temperature in the request: k3 refuses any value but its own default; thinking
off) and `claude-haiku-5-5` as one `claude -p` process per question (no tool, no settings, no MCP, no session record, an empty work
directory, the question and memories on standard input, the first line says it is a test; a row with any tool call is void).
Judge: the official LoCoMo-Refined `refined` prompt, copied byte for byte (`refined_judge_prompt.txt`, sha256 checked by a test) with
the official request shape and rules (see `judge.py`). **The official judge model `qwen3-14b` is not served to the account on the
international endpoint (HTTP 404), so `qwen3.5-27b` stands in** (a second stand-in, `qwen3.6-27b`, judges the 30-answer consistency
sample, which is also run through the official `scripts/run_eval.sh`). The answer prompt is in `answerer.py`, word for word.

Three outcomes per answer: **correct**, **said "I don't know"** (not found), **made up** (judged wrong and not "I don't know").
Wrong answers are split by where the standard evidence stood: not returned / returned but not in the first 20 / in the prompt.
Rows (with the model's own answers) stay in a local results directory, never in the repository or the floor; the report has ids and
numbers only. A run resumes where it stopped, rows that ended in an API or judge error are asked again, and the spending limit
(1 US dollar for the whole step, carried over between runs in `spend.json`) is checked before each question.

```
python -m eval.ruler.qa_main pick|run|report|probe|consistency-prepare|consistency-compare --out <dir outside the repo> [--data <copy>] ...
```
(keys only through the environment of the process that runs the command)

## Step 3: the meaning-based path and the load measurement (`dense.py`, `embed_meter.py`, `vector_cache.py`, `http_target.py`, `load_measure.py`)

The service path can be switched to **keyword + dense** (`DenseConfig`; `RECALL_PATHS=keyword,dense`, `EMBED_PROVIDER=dashscope`,
reranking off unless asked). `ServiceTarget(..., dense=DenseConfig(...))` and `HttpServiceTarget(..., dense=...)` are both built through
`dense.service_env`, the one place that sets a service's environment and puts the embedder hooks in (and puts everything back on exit).
Around the embedder, from the outside in: the invocation counter, a **fixed local vector file** (`vector_cache.py`: SQLite keyed by model
and sha256 of the text, hashes and numbers only — a text embedded once is never embedded again, even through a brand-new service
database), and the metering transport (`embed_meter.py`: calls, tokens, statuses, exceptions, batch sizes; refuses locally at the token cap
and at the step's spending limit; every reply's tokens go into the same spending table as the model calls, key `text-embedding-v4`,
0.07 US dollars per million tokens). The key is only ever read from the process environment (`DASHSCOPE_API_KEY`, set by whoever starts
the run); the keyword path is built without it.

`qa_main run --search dense [--vector-cache <file>] [--embed-cap-tokens N]` runs the step-2 answering flow on the dense memories: the rows go
to `rows_dense.jsonl` (each row has a `search` field), the vector figures to `embedding_dense.json`, and `report --variant dense` makes
`report_dense.json/md`.

`qa_main load` is an earlier stand-alone measurement script, moved into the ruler: the service runs in this
process on a loopback port (`HttpServiceTarget`, real HTTP, real threads); all conversations are written at once, optionally warmed with one
search each (`--warmup`), all questions are asked at once (`--workers`), then a few requests are sent one at a time to count exactly how many
embedding calls each makes (`--probe-n`). It reports `dense_query_skipped`: under many simultaneous searches the service's embed gate
(2 places, no waiting) can be full, and then that search silently runs on the keyword half only (one log line). **Do not judge the meaning-based
path's retrieval quality from a run with several workers**; the answering flow asks one question at a time and is not affected. Only the tuning
conversations are used (`--conv-ids` refuses a locked one). The tests use fake embedding and model interfaces
(`testing.FakeEmbedApi`); the real embedding service has since been called by the step-4 runs and the concurrency measurement below.

### Concurrency of the embedding path (`qa_main load --concurrent-embed`)

Every request to the embedding provider normally goes through the run's paid gate (one lock, one request outside at a time). That is right
for answering and judging, but it makes a concurrency measurement of the embedding path meaningless: the searches queue behind one another
(about 1.1 s per request, 17 s per search at 16 workers) instead of sending requests in parallel like a deployed service.
`--concurrent-embed` lets the requests overlap. The wall becomes a running sum of the upper bounds of everything sent (checked before each request
leaves, never taken off again), so `--embed-cap-tokens` is then a looser wall: the real tokens are far below it. It cannot be combined with a
spending meter, and the tests (`tests/test_embed_meter_concurrent_mode.py`) pin both modes. Use it only for `load`; the answering flow keeps the serial gate.

Measured on the 648 tuning questions, cold (no vector cache, real provider calls, the conversations written first; shape c, top_k 100):

| searches at once | places at the embed gate | skipped | used dense | 503 | search time, median / max |
|---|---|---|---|---|---|
| 16 | 2 (default) | 610 / 648 | 38 | 0 | 0.9 s / 3.3 s (queued behind the paid gate) |
| 32 | 2 (default) | 602 / 648 | 46 | 0 | 2.3 s / 5.1 s (same) |
| 16 | 64, serial paid gate | 104 / 648 | 544 | 0 | 17.5 s / 18.3 s (same) |
| 16 | 64, `--concurrent-embed` | 0 | 648 | 0 | 1.3 s / 2.1 s |
| 32 | 64, `--concurrent-embed` | 0 | 648 | 0 | 2.5 s / 3.4 s |
| 64 | 64, `--concurrent-embed` | 0 | 648 | 0 | 5.0 s / 7.8 s |

The gate on the search path never waits: a search that finds no free place drops its meaning-based half at once. With the default 2 places,
simultaneous searches lose that half almost always, silently; the deployed image therefore sets `EMBED_MAX_CONCURRENCY=64`. The search time
grows about linearly with the number of searches at once (the service reads the whole space's vectors and ranks them inside each search in one
process: the reason is read from the code, not profiled).
The provider's documented limit for `text-embedding-v4` on the international console is 1,800 requests and 1,000,000 tokens per minute
(from the documentation page; not checked against this key in the console).

### Review fixes (the first review round of the step-3 code)

- **One asker per text.** `CachingEmbedder` lets only one thread ask the provider for a given (model, text hash) at a time; the others wait and
  read the answer from the file. If the asking thread fails it releases its claim and a waiter asks for itself.
- **Every paid call goes through one gate** (`paid_gate.py`; `CostMeter.gate`). One lock, and the whole call is made inside it, so at any
  moment at most one paid call — answering, judging, the judge-consistency check, an embedding request — is outside. Inside the lock, in
  order: (1) is "spent + the most this call can cost" within the spending limit (and, for an embedding request, "tokens used + the bound"
  within the token cap)? no → refused, nothing is sent; (2) send; (3) book what it really cost; (4) if it cost more than its bound, the
  bound was wrong: the gate closes and every later paid call is refused. The upper bound: an embedding request =
  `sum(bytes of the texts) + 2 * texts` (a token covers at least one byte; each text carries two special tokens; an unreadable body =
  its byte length); a chat attempt = `request body bytes` for the prompt plus `max_tokens` (or the assumed
  `ASSUMED_COMPLETION_UPPER_BOUND` = 4096 when the request sets none — the official judge shape), at the model's price. Booking: a reply with
  usage at its usage; a call that raised (network error / timeout), a reply whose usage is unreadable, a 200 reply that cannot be parsed:
  at the upper bound (the provider may have charged it); an error reply (4xx / 5xx — only a 2xx counts its usage, even if an error reply
  carries a usage block): not booked. Models paid by plan quota (Kimi, Haiku) cost nothing but their tokens are counted. The run attaches its
  meter to the answering and judging clients (`ChatClient.attach_meter`); `prepare_consistency` does the same for its two judges; a result
  marked `metered` is not booked a second time by the runner. A call the gate refuses stops the run (`stopped at the spending limit after N
  of M`); if the vector calls were refused while the contexts were built, the run stops before any model is asked. The other steps of a
  run (building contexts, searching, writing results) stay multi-threaded; only the paid step queues. (An earlier version reserved room
  for in-flight calls; that machinery was removed after three review rounds kept finding holes in it.)
- **The picks file is checked in full.** `check_picks` takes the parameters the picks were made with: the main set holds exactly `per_category`
  questions of every category, the cross-check set exactly `cross_per_category` of every category, neither names a question twice, the cross-check
  set sits inside the main set. A run asked for a different shape than the stored picks refuses (the pick is made once; it is not redone).
- **The vector file closes with the target** (`ServiceTarget.close()` and `HttpServiceTarget.close()`), so it can be deleted on Windows.
- **Stemming is set explicitly on every run.** `service_env(..., stem_english=)` sets `STEM_ENGLISH` itself (default off; the outer environment
  never counts); `--stem-english on|off` on `run` and `load`; the setting is in `conditions_<variant>.json`, in every row and in the report.

### More answering models (OpenCode Go gateway) and the judge's length cap

`qa_main run --who <id>` now also takes `hy3`, `qwen3.8-flash`, `minimax-m3.1-flash`, `mimo-v2.6-flash`, `glm-5.3-flash` (`OPENCODE_MODELS`; ids checked
against the gateway's public model list). They go through the same `ChatClient` and the same paid gate, under their own id in the price table
(`PRICES`, from opencode.ai/docs/go: the plan counts usage against a monthly dollar limit per model at those prices). The key is read from the
environment variable `OPENCODE_GO_API_KEY` at the moment of the call and never stored; the gateway needs a stable `x-opencode-session` header
(`ChatConfig.headers`, one id per config) and an honest user agent. Four models speak the OpenAI chat style on `https://opencode.ai/zen/go/v1`
with `reasoning_effort: none` (the setting that took effect through this gateway in the home's own probe); the gateway docs put `qwen3.8-flash`
on the messages-style endpoint only, so `ChatConfig.api_style = "anthropic"` sends `POST {base}/messages` (key in both `Authorization` and
`x-api-key`, `anthropic-version` set) and reads text blocks and `input_tokens` / `output_tokens`. **Nothing here has called the gateway for a
question yet**: which endpoint path, which header for the key, and whether each model really stops reasoning are settled by the first paid request.

The judge request now has `max_tokens = 128` (`JUDGE_MAX_TOKENS`). The longest single judge reply in all kept results is 17 tokens; a reply that
reaches the cap before a label is a `JudgeError` at once (never a score, never asked again), and the paid gate reserves 128 tokens per judge call
instead of the 4,096 it assumed for a request with no cap.

## Step 4: the one-time run on the locked conversations, and the check before a "dense on" deployment (`locked_run.py`, `gate_check.py`)

Plan: the plan document (kept privately; sha16 `7b8379aca7a46d45`, which is `locked_run.PLAN_SHA16`). The one-time run on the locked part has been done and its result is in `METHODS.md` section 5a; the tests only use synthetic conversations and a tmp directory as the ledger.

```
python -m eval.ruler.qa_main run-locked --data <copy> --out <dir> --version dense --declared kw,dense --ledger <dir> --models main,hy3,haiku [--limit-usd 1.2] [--vector-cache <file>] [--one-model]
python -m eval.ruler.qa_main report-locked --out <dir> --ledger <dir>
python -m eval.ruler.qa_main gate-check --data <copy> --out <dir> --vector-cache <file> [--workers 8]
```

- **`run-locked`** needs all of `--version` (the version the gates ask about: `dense`), `--declared` (the two versions `kw,dense`), `--ledger`, `--models` (the answering
  models in run order); any missing is refused. It asks all 734 locked questions, 100 memories each (shape c, stemming off), and judges as before (qwen3.5-27b + refined, 128-token
  cap). Order of things: arguments checked and the plan worked out (nothing written) → **the claim** (`split.claim_locked_models`: the two versions, every model's identity as far as
  it can be pinned — the pinned id for `main`, the alias + gateway + session header for `hy3`, the `claude --version` for `haiku` —, the run order, the three gates, and which version is
  asked first on each question; made once; the same plan again is a resume, any other plan is refused; one run at a time through `run.lock`) → both retrievals once into
  `retrieval_kw.jsonl` / `retrieval_dense.jsonl` (hash kept in `progress.json`) → per model, per question: ask the first version, ask the second, judge the first, judge the second
  (the first is fixed by the question id's hash, even → kw first), both rows written together to `rows_locked_<model>.jsonl`. A pair with any failed ask / judge / void row is not
  written; after the model's pass such pairs are asked again as whole pairs once; a pair that still fails stops the run as unfinished and the next model is not started. A stop at
  the spending limit leaves only whole pairs and is resumed by the same command.
- **The gates** (`split.decide_dense`, integer arithmetic), all three: the models pooled gain at least 4 percent of all rows; at least 3 of the 5 conversations are better (the models
  pooled inside each); no single model has more than 3 percent of its questions fewer correct (22 of 734 passes, 23 fails). `report-locked` gives each model alone once it is whole
  and the decision only when all are; every model's and every conversation's own numbers are in it. No fourth outcome.
- **`gate-check`** (tuning conversations only; a locked one is refused): serial dense and serial keyword-only baselines, then 8 concurrent dense; it passes only when the running
  service's own config says dense is on with the DashScope embedder, every question's dense search returned something (`used_dense`) and none was skipped, the query vectors asked of
  the provider are the distinct question texts, there is no 503, every concurrent answer equals the dense baseline id for id, and the two baselines are not all the same (how many
  differ is only reported).

## Tests

`tests/test_ruler_*.py` (synthetic mini corpus in `testing.py`; nothing real). The whole-corpus load test
(`eval/local_load/test_full_corpus.py`) sends its Add payload in shape `a` through `shapes.render`, cuts with `corpus.chunk_conversation`
and reads its pinned numbers from `corpus.py`: what the platform sends is defined in one place. Old results of the code track (`eval/results`, 29
files) were moved out of the repository on 2026-10-23 with their hashes checked; the repository history still has them.
