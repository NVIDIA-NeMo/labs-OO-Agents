---
name: nooa-trace-viewer
description: Run and use the NOOA trace viewer — the web UI + OTLP receiver for browsing agent traces and eval results. Use when starting the viewer, importing/exporting/deleting traces, querying the viewer's REST API, or wiring an agent run so traces show up at localhost:5001.
compatibility: nooa package with the [viewer] extra (fastapi, uvicorn); nooa-cli for the `nooa` commands
---

# Trace Viewer

The viewer (`src/nooa/viewer/`) is a single FastAPI app that is both the **OTLP receiver** agents send spans to and the **web UI** (React SPA) for browsing traces, LLM calls, and evaluation experiments. Traces persist in SQLite, so they survive restarts.

## Start it

```bash
nooa start-dev                                  # http://localhost:5001
nooa start-dev --port 5002                      # custom port
nooa start-dev --db /path/to/other.db           # separate trace store
nooa start-dev --host 127.0.0.1                 # default host is 0.0.0.0
```

Stop with Ctrl-C. If deps are missing: `uv sync --extra viewer`. If the port is busy, the command reports the offending PID.

- **Database resolution**: `--db` > `$NOOA_TRACE_DB` > `~/.config/nooa/traces.db`.
- Alternate entry point: `python -m nooa.viewer` (this path honors `$NOOA_TRACE_VIEWER_PORT`, default 5001; `start-dev` uses `--port` instead). DB default for the raw module is `./traces.db` — prefer `start-dev`.

## End-to-end workflow

```bash
nooa start-dev                                # terminal 1
uv run python my_agent.py                        # terminal 2 — auto-traced (see nooa-capturing-traces)
open http://localhost:5001/traces                # browse the session
```

Agents auto-detect the viewer at startup and stream spans; no code changes needed. If the agent ran on another host/port, set `OTLP_ENDPOINT=http://<host>:<port>/v1/traces` in the agent's environment.

## Using the UI

| Route | What you get |
|---|---|
| `/traces` | paginated session list — search, filter by experiment / batch ID, delete |
| `/traces/view?session_id=…` | trace detail: span-hierarchy event list, expand/collapse, text search |
| `/eval` | evaluation experiments with live status |
| `/eval/experiment/{id}` | experiment summary; drill into per-test traces |

Trace-detail features worth knowing:

- **Timeline** — dual-canvas zoomable timeline of all events; **filter sidebar** — toggle event types, filter by span/agent/LLM/execution ID.
- **LLM call view** — full reconstructed conversations per call (`/api/traces/{session_id}/calls` under the hood).
- **Annotations** — score/label/comment/tags on a trace; quick 👍/👎.
- **Playground** — re-run an LLM turn with a different model/temperature and diff the output.
- **Batch ID chip** in the header (copyable) — groups imported/eval runs; filter the list with `/traces?batch_id=…`.
- Keyboard: `j`/`k` navigate, `/` search, `?` shows all shortcuts. URL captures view state, so links are shareable; add `&embed=true` to embed.

## Viewer plugin metadata

The viewer selects a registered span renderer from the `nooa.viewer.plugin`
OpenTelemetry span attribute. Core NOOA tracing sets this automatically for
method, generation, code-execution, and tool-execution spans. Code that adds a
new renderer in `src/nooa/viewer/frontend-react/src/components/plugins/index.ts`
can select it for custom spans by setting the same attribute on those spans.

Older imported traces may still carry the pre-NOOA key
`nemo_oo_agents.viewer.plugin`; the viewer accepts that as a compatibility
fallback, but new span producers should write `nooa.viewer.plugin`.

## Import, export, delete

```bash
# Import OTLP .jsonl trace files (from exporters.jsonl(...) or a viewer export)
nooa import-traces ./traces/                          # dir (recursive) or single file
nooa import-traces run.jsonl --endpoint http://host:5001 --batch-id my-exp-v2

# Import a Harbor eval job directory
nooa import-harbor <job-dir>

# Export one session as .jsonl (round-trips through import-traces)
curl -o session.jsonl "http://localhost:5001/api/trace/export?session_id=<ID>"

# Delete
nooa delete-traces --batch-id my-exp-v2               # one batch via API
# DELETE /api/traces/{session_id}                        # one session
# DELETE /api/traces?confirm=true                        # everything

# Housekeeping for on-disk trace FILES (not the viewer DB)
nooa traces list
nooa traces delete --older-than 7
```

Import notes: only OTLP JSON lines (`{"resourceSpans": [...]}`) are accepted; the session ID is derived from the filename; already-imported sessions are skipped; omitted `--batch-id` auto-generates `import_<timestamp>_<hex>`.

## REST API (for scripts and agents)

Base: `http://localhost:5001`. Most useful endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/traces?page&limit&search&experiment&batch_id` | paginated session list |
| `GET /api/trace?session_id=&limit=&offset=` | a session's raw OTLP spans (paginate with limit/offset) |
| `GET /api/trace/export?session_id=` | download session as `.jsonl` |
| `GET /api/traces/{session_id}/calls` | reconstructed LLM calls with full messages |
| `GET /api/trace/resource?session_id=` | resource attrs (incl. `batch_id`) |
| `GET /api/experiments`, `GET /api/experiment/{id}/traces` | experiment grouping |
| `GET /api/eval/experiment/{id}/summary`, `.../tests` | eval results |
| `POST /v1/traces` | OTLP JSON ingest (what agents/import use) |
| `POST /v1/sync` | block until the ingest queue is drained (use before reading back a just-finished run) |
| `GET /api/explorer/*` | structured analysis endpoints — use via `TraceExplorerClient` (see `nooa-trace-explorer`) |

Ingestion is queued and written by a single background writer, so a `GET` immediately after a run may miss the tail — `POST /v1/sync` first when scripting.

## Pitfalls

- `nooa start-dev` resolves the DB and sets `NOOA_TRACE_DB` in its own environment before starting uvicorn (same process); a *separately launched* `python -m nooa.viewer` without that var uses `./traces.db` in the CWD — easy way to "lose" traces into a second DB.
- The viewer refuses to start (exit 1) if the SQLite DB is locked by another viewer instance — check for an already-running `start-dev`.
- Old docs mention a separate viewer package and a `nooa viewer` subcommand; the viewer now lives in `src/nooa/viewer/` and `nooa start-dev` is the canonical command.

## Related skills

- `nooa-capturing-traces` — how spans get produced and exported.
- `nooa-trace-explorer` — programmatic trace analysis (CLI + Python) on top of files or this viewer.
