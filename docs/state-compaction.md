# Background state compaction

`TokenBudgetSummarizer` replaces older stored events with a summary while the
agent keeps working. Actual request usage triggers compaction; the summary
input comes directly from stored history, independently of the context view.
Views remain responsible for prompt selection and any prompt-only compression.

```python
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig

TokenBudgetSummarizer.install(
    agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10)
)
await agent.aclose()  # stop background work before closing the shared client
```

Before dispatch, the compactor snapshots the oldest eligible contiguous range
that fits the effective client's summary input budget. It preserves recent events,
diagnostics and tool-call group integrity. Oversized, unfinished and image-bearing
groups remain active. Events created during the request are outside this snapshot. Summary input
contains public event content; persistence metadata and opaque provider state stay private.

The dedicated Predict request uses the parent's effective client, without its
prompt, tools, output schema or cache-key contract. Parent-prefix cache reuse is
not guaranteed. At most one summary runs or waits for application.

A completed summary applies at the next `BeforeTurn` only if the source range,
identities and contents still match. Raw events remain accessible in storage.
Empty, incomplete or failed summaries leave history unchanged. Two consecutive
failures stop scheduling; fix the cause before reinstalling. Closing the agent
cancels and awaits background work.

Compacting history may not relieve prompt pressure caused by other context
sources. A custom view also decides whether to include the resulting summary.

The ordinary tests cover source snapshots, bounded ranges, replay groups, custom
views, provider request bodies and lifecycle behavior. The opt-in
`tests/integration/test_summarizer_live.py` exercises summary generation,
application and continuation with exact handoff facts. Run with
`NOOA_RUN_SUMMARIZER_E2E=1` and configured `release-gate-openai` and
`release-gate-anthropic` aliases. Missing aliases skip; a skip is not live evidence.
