# State compaction

`TokenBudgetSummarizer` pauses the agent while it summarizes older stored events.
Actual request usage triggers compaction after a model response; the source
range is selected from stored history, independently of the context view.
Views remain responsible for prompt selection and any prompt-only compression.

```python
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig

TokenBudgetSummarizer.install(
    agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10)
)
await agent.aclose()  # stop background work before closing the shared client
```

Before dispatch, the compactor selects every stored event except the
`preserve_recent` most recent ones; one summary replaces that range, including
earlier summaries and records the model never sees. Events created during the
request are outside this selection. Archived events remain in storage.

When the parent request carries every model-visible event in the range
(`LLMCallContext.event_ids`), a summary request reuses the completed parent
request and appends a summary instruction, preserving its cache prefix, tools,
schema and settings. This holds for the built-in views and for any view that
keeps their history. Tools are never executed by the summary request. Otherwise
the summarizer uses a dedicated Predict request built from the range's
model-visible stored events, bounded to the summary input budget, with the
parent's effective client, without inheriting its prompt, tools, schema or
cache key.
At most one summary runs or waits for application.
Cached-summary requests pass through agent middleware again; use `summary_fork_active()` to distinguish them, and preserve source history and the prefix when cache reuse is desired.

A completed summary applies at the next `BeforeTurn` only if the source range's
tags and event ids still match. Raw events remain accessible in storage.
Empty, incomplete or failed summaries leave history unchanged. Two consecutive
failures stop scheduling; fix the cause before reinstalling. Closing the agent
cancels and awaits background work.

Compacting history may not relieve prompt pressure caused by other context
sources. A custom view also decides whether to include the resulting summary.

The ordinary tests cover source selection, custom views, provider request bodies and lifecycle behavior. The opt-in
`tests/integration/test_summarizer_live.py` exercises summary generation,
application and continuation with exact handoff facts. Run with
`NOOA_RUN_SUMMARIZER_E2E=1` and configured `release-gate-openai` and
`release-gate-anthropic` aliases. Missing aliases skip; a skip is not live evidence.
