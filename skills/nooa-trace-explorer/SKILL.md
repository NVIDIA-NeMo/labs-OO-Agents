---
name: nooa-trace-explorer
description: Analyze NVIDIA OO Agents execution traces programmatically with the trace-explorer CLI and Python API. Use when debugging why an agent run failed, inspecting LLM turns and code executions, diffing two runs, aggregating errors across an eval experiment, or root-causing behavior from a trace file or viewer session.
compatibility: nooa package (trace-explorer CLI ships with it); a running viewer only for --viewer / thin-client modes
---

# Trace Explorer

The trace explorer (`src/nooa/trace_explorer/`) is a CLI + Python library purpose-built for **agent-driven root-cause analysis** of traces. Where the viewer is a web UI for humans, the explorer emits LLM-friendly text (or JSON) and supports search, diffing, and experiment-wide aggregation. It is not itself an Agent — it's a tool you (the coding agent) drive.

Data sources: local OTLP `.jsonl` trace files, or a running viewer (per-session or per-experiment). It never opens `traces.db` directly — viewer data goes through the HTTP API.

## Exploration strategy: progressive disclosure

Start broad, drill in. Overview → errors → session → turn → search.

```bash
uv run trace-explorer trace.jsonl                    # 1. overview: call graph, sessions, pass/fail
uv run trace-explorer trace.jsonl --errors           # 2. all errors (or --first-error)
uv run trace-explorer trace.jsonl -s 278a10          # 3. one session's turns (IDs can be 6-char prefixes)
uv run trace-explorer trace.jsonl -s 278a10 -t 0     # 4. one turn: context window → LLM output → execution result
uv run trace-explorer trace.jsonl --search "Timeout" # 5. regex/text search across everything
```

Output includes navigation hints for the next drill-down. Other useful flags: `--timeline`, `--eval` (evaluation context), `--json` (structured output), `-v` (full detail), `-q` (suppress parser warnings), `--raw <span_id>` (raw span JSON), `--diff other.jsonl`, `--api-help` (prints the Python API guide).

## Against a running viewer

```bash
uv run trace-explorer --viewer http://localhost:5001 --session-id <ID>            # overview
uv run trace-explorer --viewer http://localhost:5001 --session-id <ID> --errors
uv run trace-explorer --viewer http://localhost:5001 --session-id <ID> -s <SID> -t 0
uv run trace-explorer --viewer http://localhost:5001 --session-id <ID> --span-id <SPAN>  # jump to a span
```

## Experiment-level analysis (eval runs)

```bash
uv run trace-explorer --viewer <URL> --experiment <ID>            # pass/fail rates + drill-in commands
uv run trace-explorer --viewer <URL> --experiment <ID> --errors   # Python exceptions across failed sessions
uv run trace-explorer --viewer <URL> --experiment <ID> --failures # wrong-answer eval failures (no exception)
uv run trace-explorer --viewer <URL> --experiment <ID> --search "pattern"
uv run trace-explorer --viewer <URL> --experiment <ID> --json     # machine-readable: all tests + session_ids
```

`--errors` = crashes; `--failures` = wrong answers. For a full sweep: get `--json` first, then iterate sessions with `--session-id`.

## Python API — full parse (`TraceExplorer`)

```python
from nooa.trace_explorer import TraceExplorer

trace = await TraceExplorer.from_file("traces/<session_id>.jsonl")
# or: trace = await TraceExplorer.from_viewer("http://localhost:5001", "<session-id>")
# or: explorers = await TraceExplorer.load_experiment_sessions(url, "<experiment>")  # {session_id: TraceExplorer}

print(await trace.get_overview())          # text for LLM context
print(await trace.get_errors())
print(await trace.get_session("278a10"))
print(await trace.get_turn("278a10", 0))
print(await trace.search("ValidationError"))
print(await trace.get_timeline())
print(await trace.find_first_error())
print(await TraceExplorer.diff(trace, other))   # first divergence, call-graph diff
```

Every text method has a structured twin returning dataclasses with `.to_dict()`: `get_overview_data()`, `get_session_data()`, `get_turn_data()`, `get_errors_data()`, `search_data()`, `get_timeline_data()`, `find_first_error_data()`, `get_eval_context_data()`, `compare_data()`. Stats helpers: `.agent_count`, `.max_agent_depth`, `get_method_counts()`, `get_recursion_pattern()`.

## Python API — thin client for huge traces (`TraceExplorerClient`)

For very large traces (>100k spans / GB-scale), don't download spans — delegate analysis to the viewer server (`/api/explorer/*`, results cached server-side):

```python
from nooa.trace_explorer import TraceExplorerClient

client = TraceExplorerClient("http://localhost:5001", "<session-id>")
await client.get_summary()          # instant DB query: span count, duration, error count
await client.get_agent_spans()      # agent-tree skeleton only
await client.get_error_spans()
await client.search_fast("query")   # SQLite FTS5 full-text search
await client.get_session_fast("abc123", span_id="<full_span_id>")
await client.get_turn_fast("abc123", span_id="<full_span_id>", turn_index=0)
await client.get_overview()         # server builds + caches the full explorer
```

Prefer the thin client when the trace is huge, when exploring interactively (cache makes repeats instant), or when you only need one session's subtree.

## Typical workflows

- **Failing eval run**: `--experiment <ID>` → note failed tests' `session_id`s → `--session-id <ID> --errors` → `-s <SID> -t <N>` to read the exact LLM turn that went wrong.
- **Agent behaved oddly, no exception**: overview → `get_session` to scan turns → `get_turn` to read the context window the model actually saw (bad prompts are usually visible right there).
- **Regression between two runs**: `trace-explorer a.jsonl --diff b.jsonl` → shows first divergence point and call-graph differences.
- **"Where did X come from?"**: `--search "X"` then `--raw <span_id>` for the exact span.

## Tips

- Session IDs abbreviate to 6 characters everywhere.
- `--install-skill` copies the bundled skill to `~/.claude/skills/trace-explorer/` (there is an in-repo copy at `src/nooa/trace_explorer/skill/SKILL.md`).
- Entry points: `trace-explorer` console script or `python -m nooa.trace_explorer` — there is no `nooa` subcommand for it.
- Eval-artifact files are rejected by `from_file` (`*.006eval.*` by filename; `*.noo-eval.jsonl` fails span validation) — point it at the session trace, not the eval sidecar.
- Reasoning content is shown by default; hide with `--no-reasoning`.

## Related skills

- `nooa-capturing-traces` — produce the `.jsonl` files this tool reads.
- `nooa-trace-viewer` — the web UI and the server backing `--viewer` / thin-client modes.
