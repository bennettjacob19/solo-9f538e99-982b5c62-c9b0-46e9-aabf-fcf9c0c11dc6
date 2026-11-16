# Embedding Batch Gateway

A small FastAPI service that accepts **1–4 short texts per HTTP call** and
merges concurrent calls into **single-flight batches of at most 8 texts** for a
single-slot CPU embedding backend. It ships with a controllable local mock
backend and a Docker Compose file that starts both.

```
client ──► gateway :8000 ──(one batch at a time)──► backend :9000
             │ batching, deadlines, ordering           │ configurable delay
             │ single-flight, error fan-out            │ error injection
             └ /docs, /stats                          └ /docs, /admin/*
```

## Quick start

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| Gateway API docs (interactive) | http://localhost:8000/docs |
| Gateway stats | http://localhost:8000/stats |
| Gateway health | http://localhost:8000/healthz |
| Mock backend docs | http://localhost:9000/docs |
| Mock backend stats (real concurrency) | http://localhost:9000/admin/stats |

No Docker? The same two apps run directly:

```bash
pip install -r requirements-dev.txt

uvicorn backend.app.main:app --port 9000 &
EMBEDDING_BACKEND_URL=http://127.0.0.1:9000/v1/embeddings \
  uvicorn gateway.app.main:app --port 8000 &
```

Run the tests and the live smoke test:

```bash
pytest -q                  # no network required
python scripts/smoke_test.py   # requires both services running
```

## API

### `POST /v1/embeddings`

```json
{
  "input": ["first text", "second text"],
  "timeout_ms": 1000,
  "model": "local-demo"
}
```

* `input` — a string or a list of 1–4 strings. `texts` is accepted as an alias.
* `timeout_ms` — end-to-end deadline in milliseconds, **measured from the
  moment the gateway accepts the request** (a middleware stamps it before the
  body is parsed). Defaults to `GATEWAY_DEFAULT_TIMEOUT_MS` (5000) when omitted.
* `model` — optional; forwarded to the backend, otherwise `EMBEDDING_MODEL`.

Response (indices are always `0..n-1` in the caller's original input order):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
  ],
  "model": "local-demo",
  "batch_id": "batch-7",
  "batch_size": 8
}
```

`batch_id`/`batch_size` describe the backend batch the caller was merged into
(useful for demos and tests).

| Status | `error.code` | Meaning |
| --- | --- | --- |
| 422 | `validation_error` | text count, `timeout_ms` or schema violation |
| 502 | `backend_error` | backend HTTP error, timeout, or invalid payload |
| 503 | `service_unavailable` | gateway is shutting down |
| 504 | `request_timeout` | deadline hit while queueing or waiting |

### Ops endpoints

* `GET /healthz` — liveness.
* `GET /stats` — counters (`accepted`, `completed`, `timed_out`,
  `disconnected`, `failed`, `batches_dispatched`, `backend_errors`),
  `pending_requests`, `in_flight`, and `last_batch`.

## Batching semantics

1. **Merge window.** When the backend is idle, the first pending request opens
   a `GATEWAY_BATCH_WINDOW_MS` (20 ms) window starting when it became pending.
   Requests arriving inside the window join the batch FIFO.
2. **Size limit.** A batch never exceeds `GATEWAY_MAX_TEXTS_PER_BATCH` (8)
   texts. Requests are never split; a request only joins if it fits whole.
3. **No re-arming.** Requests that piled up while the backend was busy already
   elapsed their window, so they are merged and dispatched immediately once the
   backend frees up — no extra 20 ms per batch.
4. **Single-flight.** Exactly one dispatcher task exists, so at most one
   backend batch is in flight. The next batch is not sent until the previous
   response *or error* has been fully received and processed — even if every
   caller of that batch already timed out or disconnected.
5. **Deadline/abandon.** A request whose deadline expires or whose client
   disconnects while **pending** is removed from the queue and never sent. A
   request abandoned while **in flight** keeps its slot occupied until the
   backend answers, but its late vectors are discarded, never delivered to the
   expired caller.
6. **Errors.** Any backend failure (non-2xx, timeout, malformed body, wrong
   count, duplicate/out-of-range index, non-numeric vector) fails **all**
   waiters of that batch with `502 backend_error`, releases the slot in a
   `finally` block, and leaves the gateway ready for the next request.

Disconnect detection polls `Request.is_disconnected()` every
`GATEWAY_DISCONNECT_POLL_INTERVAL_MS`. On ASGI servers that cannot expose the
receive channel, probing is disabled rather than treated as a disconnect; the
deadline still bounds the caller's wait.

## Configuration

Gateway (environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND_URL` | `http://127.0.0.1:9000/v1/embeddings` | backend endpoint |
| `EMBEDDING_MODEL` | `local-demo` | default model name |
| `EMBEDDING_API_KEY` | *unset* | sent as `Authorization: Bearer …` |
| `EMBEDDING_BACKEND_TIMEOUT_SECONDS` | `30` | whole-request backend timeout |
| `EMBEDDING_BACKEND_CONNECT_TIMEOUT_SECONDS` | `5` | connect timeout |
| `GATEWAY_BATCH_WINDOW_MS` | `20` | idle merge window |
| `GATEWAY_MAX_TEXTS_PER_REQUEST` | `4` | max texts per HTTP call |
| `GATEWAY_MAX_TEXTS_PER_BATCH` | `8` | max texts per backend batch |
| `GATEWAY_DEFAULT_TIMEOUT_MS` | `5000` | default deadline |
| `GATEWAY_MAX_TIMEOUT_MS` | `600000` | max accepted deadline |
| `GATEWAY_DISCONNECT_POLL_INTERVAL_MS` | `20` | disconnect poll cadence |

Mock backend (also editable live via `POST /admin/config`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_DELAY_MS` | `300` in compose | simulated inference latency |
| `MOCK_ERROR_RATE` | `0.0` | random failure probability |
| `MOCK_ERROR_STATUS` | `503` | status used for injected errors |
| `MOCK_MODE` | `ok` | `ok`, `error`, `malformed`, `short`, `duplicate_index`, `shuffle` |
| `MOCK_SHUFFLE_RESPONSE` | `false` | deliberately reverse `data` order |
| `MOCK_MAX_INPUTS` | `8` | returns 400 for larger batches |
| `MOCK_DIM` | `8` | embedding dimension |

The mock logs every request and tracks `max_concurrency`,
`concurrency_violations`, `oversized_batches`, `max_batch_seen` and a request
log at `GET /admin/stats`. Because it is single-slot in intent, any overlap
above 1 (or any batch > `max_inputs`) is recorded as a violation.

## Verifying the key paths

```bash
# Merge + order
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["a","b"],"timeout_ms":1000}' | jq
curl -s localhost:9000/admin/stats | jq '.max_concurrency, .max_batch_seen'

# Inject a slow backend and expire a request
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"delay_ms":800,"mode":"ok"}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' -d '{"input":["slow"],"timeout_ms":100}'

# Inject an error for the next batch
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"error"}'
```

# Verification summary

`pytest -q` covers merge sizing (never > 8), request-order preservation,
window timing, pending-expiry removal, single-flight after every caller left,
late-vector discard, error fan-out with recovery, malformed backend payloads,
and disconnect abandonment. `scripts/smoke_test.py` repeats the key paths
against real uvicorn processes and checks the mock backend's recorded
`max_concurrency`/`concurrency_violations`.

## Project layout

```
gateway/app/config.py    settings from env
gateway/app/batcher.py   single-flight queue, merge window, deadlines
gateway/app/main.py      FastAPI app, routes, acceptance-time middleware
gateway/app/schemas.py   request/response models
backend/app/main.py      controllable mock embedding backend
tests/                   batching, window, timeout, disconnect, error tests
scripts/smoke_test.py    live end-to-end checks against running services
docker-compose.yml       gateway + mock backend with healthchecks
```

## Known limitations

* The deadline is enforced by the gateway only; the in-flight backend call is
  not aborted on caller timeout because the batch must still be drained before
  the next one is sent.
* `batch_size`/`batch_id` are informational; clients should rely on `index`
  ordering only.
* Disconnect-abandonment is best effort and depends on the ASGI server
  surfacing `http.disconnect` (verified with uvicorn).
* Batching state is per process. Keep one uvicorn worker (as in the Docker
  image) unless each replica has its own backend slot; otherwise two workers
  can issue one batch each concurrently.
* The mock backend is for local verification only — no auth, no persistence.
