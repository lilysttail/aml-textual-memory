# Agent Memory Challenge — Textual Memory track

A memory retrieval service for long multi-session conversations: it stores what
was said (who said it, when, in which session, which turn) and gives back
**evidence** — ranked turns with their surrounding context — so a question
answering model can work from the memory instead of guessing.

The service exposes exactly three endpoints:

| | |
|---|---|
| `POST /add` | write a chunk of a conversation (synchronous, idempotent per `request_id`) |
| `POST /search` | return the ranked evidence for a question (at most `top_k` items) |
| `GET /health` | liveness (unauthenticated) |

The platform calls `Add` per chunk, then `Search` per question, and does the
answering and the scoring itself. This service never answers a question.

## The contract in one page

**`POST /add`**

```json
{
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "messages": [{"role": "user", "timestamp": 1704067200000, "content": "memory text"}],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

- `request_id` — required, echoed back byte for byte; a repeated `request_id`
  is a replay and does not write twice.
- `messages` — required, ordered; `role` and non-empty `content`; `timestamp`
  (Unix milliseconds) is **optional** — when it is missing the service does
  not know when the message was said and does not invent a date.
- `user_id` — required; the **only** retrieval-isolation field.
- `session_id` — required; used to group a session and to find the turns
  around a hit. It is not a search filter.
- The response is `200` only after the write is persisted **and searchable**.

**`POST /search`**

```json
{"query": "Which answer best matches the memory?", "options": ["A. ...", "B. ..."],
 "user_id": "eval:run_abc123:locomo:conv-0", "top_k": 100}
```

```json
{"data": [{"id": "mem_1", "content": "[2026-10-21 05:00] user: remembered fact text",
           "score": 0.87, "created_at": "2026-10-21T05:00:00Z"}]}
```

- The response never exceeds `top_k`; the order is ours (most relevant first).
- `content` is the **candidate itself** — the platform's answer model reads it
  verbatim. The turns around a candidate are handed over as candidates of their
  own (on by default; `NEIGHBOUR_CANDIDATES=0` turns it off): on the public
  LoCoMo-Refined set that is worth 3.98 points more questions with an evidence
  turn, for 10.3% more characters.
- `options` (multiple-choice items) is accepted and ignored: it never contains
  the gold answer, and searching "for" the options would be fitting the test.

**Authentication** — `Authorization: Bearer <key>`, `Authorization: Token <key>`
or `X-Api-Key: <key>`. `/health` needs none.

## Run it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
API_KEY=dev-key .venv\Scripts\uvicorn app.main:app --port 8000
```

A local run is keyword-only by default and needs no key. The `dense` recall
path and the rerank need an embedding provider: `EMBED_PROVIDER=dashscope`
with `DASHSCOPE_API_KEY` (the academic track fixes the model:
`text-embedding-v4`), or the development-only local model via `BGE_SMALL_DIR`.
The submitted (Docker) version runs keyword + dense with DashScope and refuses
to start without a real `API_KEY` and a `DASHSCOPE_API_KEY`.
`core/README.md` has the full interface; `OPERATIONS.md` has the configuration,
the submission deployment and its read-back (`scripts/deploy_check.py`).

## How it is built

- `core/` — the retrieval core: structured records (who said it, when, which
  session, which turn), keyword + dense recall, RRF fusion, optional vector
  rerank, the neighbouring turns, per-user isolation, idempotent writes. It
  imports nothing from the service, reads no environment variable, and knows
  nothing about the platform — the same directory is meant to travel.
- `app/` — the shell: the HTTP contract, the API key, the platform field
  mapping, the rendering of one message, and the choice of embedding provider.
- `eval/ruler/` — a ruler that measures **evidence recall** on the
  public LoCoMo-Refined set: did the retrieval bring back the turns the
  dataset marks as the evidence? It never answers a question and never calls a
  model.

```
py -m pytest tests/ -q         # the service's tests
py -m pytest core/tests -q      # the core's own tests
```

## Third-party work

Everything borrowed is listed in `THIRD_PARTY_NOTICES.md` with its author, its
source, its licence and what we changed. The method is described in
`METHODS.md`.

## Licence

The service code is MIT (`LICENSE` in the released repository). The public
benchmark used by the local ruler is CC BY-NC 4.0 and is **not** part of this
repository.
