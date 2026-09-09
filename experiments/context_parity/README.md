# Default-context parity

## Research question

Does the context-view refactor preserve the complete LLM request produced by NOOA's legacy default context assembly?

## Design

`run.py` exports two Git revisions into isolated temporary trees and runs the same `capture.py` against both. The capture uses `FakeLLMClient` with fixed responses and records every final message list, tool contract, output schema, and stable call option received by the client.

The scenarios cover Predict composition and precedence, event filtering, dynamic-expression failure, legacy `SkillRegistry` context, two-turn CodeAct with changing dynamic state, and context-budget eviction. New custom skill views are excluded because they have no legacy equivalent.

The comparator preserves order and content. Capture removes only NOOA's transport-only cache-boundary field and request option. Comparison normalizes generated IDs, CRLF line endings, trailing whitespace, and the legacy formatter's outer `<context>` envelope/inter-block separators. It does not ignore rewording, block changes, role changes, message reordering, tool changes, or schema changes.

## Metrics

- normalized structural/content differences;
- raw capture equality;
- scenario, request, message, and serialized-character counts;
- successful completion of every deterministic agent call.

## Run

```bash
uv run python experiments/context_parity/run.py --baseline main --candidate HEAD
```

Results are written to `results/<baseline>_vs_<candidate>/` with a manifest, report, exact captures, and per-arm logs.

## Results

The final implementation passed against upstream `main`; see [REPORT.md](REPORT.md).
