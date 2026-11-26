# Summarizing history

Long-running agents collect more history than fits in a model request.
`TokenBudgetSummarizer` keeps it in check: when a request gets too large, it
replaces older events with a short summary written by the same model.

```python
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig

TokenBudgetSummarizer.install(
    agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10)
)
await agent.aclose()  # at shutdown, before closing the shared model client
```

## How it works

1. **Trigger.** After each model response, the summarizer compares the
   request's actual input tokens with `max_tokens`. Below the limit, nothing
   happens.
2. **What gets summarized.** Every stored event except the `preserve_recent`
   newest ones. The range is fixed before the request is sent, so events from
   the current turn are never included. Earlier summaries in the range are
   folded into the new one.
3. **Writing the summary.** The agent waits while the summary is written,
   in one of two ways:
   - If the request just sent already contains all of those events (the
     default views always do), the summarizer resends it with one added
     instruction: "summarize events X to Y". The provider can reuse its cached
     copy of the request, so this is cheap. Tools in the request are never run.
   - Otherwise it sends a separate request to the same model, built from the
     stored events.
4. **Applying it.** Before the agent's next turn, the summary replaces the
   events, but only if they have not changed in the meantime. The originals
   are archived, not deleted: `self.events` can still read them.

If a summary comes back empty, incomplete or fails, history stays unchanged.
After two failures in a row, the summarizer stops; fix the cause and install it
again.

## Custom views

The summarizer works on stored history, not on what a view sends, so it works
with any view. A custom view decides whether to show the summary.

One summary can cover several method calls. Its `related_call_ids` lists all of
them, so a view that filters with `call.id in event.related_call_ids` keeps the
summary, including what it says about the other calls. If a view must only
ever show one call's information, it should leave out summaries that cover
other calls, or use `MethodSummarizer`, which summarizes one call (with the
calls nested in it) at a time.

## Middleware

The summarizer trusts `LLMCallContext.event_ids` to know which events a request
contains. Middleware that removes events from a request must update
`event_ids`; otherwise those events are archived without being summarized.
Summary requests also pass through middleware (`summary_fork_active()` is true
for them); middleware must not remove the events being summarized from them.

## Testing

The regular test suite covers this offline. A paid live test,
`tests/integration/test_summarizer_live.py`, checks a real summary end to end.
Run it with `NOOA_RUN_SUMMARIZER_E2E=1` and the `release-gate-openai` and
`release-gate-anthropic` model aliases configured. Without them, it is skipped,
and a skip does not count as a pass.
