# Cache-sharing background summaries

`TokenBudgetSummarizer` replaces older conversation messages with a summary while
the agent keeps working. It asks the same model to summarize the conversation
using the request the agent just sent, plus a summary instruction. That lets
the provider reuse its cached input. It does not copy the running agent or
execute tools.

## Use it

```python
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig

TokenBudgetSummarizer.install(
    agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10)
)
# At shutdown, before closing the shared model client:
await agent.aclose()
```

The threshold starts a summary; `preserve_recent` keeps the newest events intact.
There is no separate summarizer model, mode setting or fallback request.
`agent.aclose()` waits for registered background components to stop, including
an unfinished summary. The coding agent calls it automatically on shutdown.

## What happens to the conversation

The collapse range is chosen before the parent request. Recent events and the
response just produced remain active. The parent continues while the summary
runs. At most one summary waits or runs at a time. A completed summary is applied
before the next turn only if the original event IDs still match.

## If a summary fails

The summary can be plain text or one `return_result` call containing a nonempty
string. That call is read as data, never executed. Empty, malformed, truncated,
errored or executable-tool replies leave the history unchanged and log a warning.
After two consecutive failures, automatic summaries stop, with another warning,
to avoid repeatedly paying for unusable summaries.
Fix the reported cause before reinstalling the summarizer.

A summary failure does not itself truncate history. If a later main model call
exceeds the context window, the agent's existing overflow recovery can archive older
events and retry with less context. Those raw events remain in session storage.

## Limitations

Filtered-history requests are skipped: the fork cannot summarize events it did
not see. A known filter produces a warning at installation; later scoped filters
warn on first use, once per summarizer. The runtime's context overflow safety net
still applies. Structured-output parents remove `output_model` from the fork so
it can return text; changing that schema may reduce cache reuse. Tools stay the
same, including typed `return_result` schemas; an incompatible reply is rejected.

`reuse_parent_prefix` and a separate `llm=` are no longer supported. The separate
`MethodSummarizer` feature still summarizes completed methods through its own
rendered input; it is not used as a token-budget fallback.

## Code walkthrough: what and why

- `src/nooa/agents/summarization.py`: intercepts the completed request so the
  fork uses the actual parent prefix, without re-rendering it. Copies only dict
  and list containers, sharing tools and immutable response/boundary objects.
  Checks failure, cancellation and event identities before collapsing history.
- `src/nooa/runtime/actor.py` and `middleware.py`: expose the effective client,
  cache key and filtered-history status at the existing middleware boundary.
  The fork uses the same middleware chain, with a task-local recursion guard.
- `src/nooa/runtime/event_manager.py`: awaits registered close callbacks, so
  the agent does not need to know which background components are installed.
- `tests/agents/test_forked_summarizer*.py`: check parent-request parity, HTTP
  prefix equality, background execution, ownership, safe output handling and
  collapse timing. These are permanent offline tests, not an experiment.

Offline tests verify that the fork preserves the parent request's prefix.
They do not measure provider cache hits or summary quality. Cache lifetime,
routing, changed suffixes and structured-output settings can lower reuse.
Keep deployment-specific measurements with the configuration used to run them.

## Live release smoke test

`tests/integration/test_summarizer_live.py` installs the summarizer on a real
CodeAct agent. It requires a background summary, application on the next agent
turn, preservation of three handoff identifiers, access to the raw archived
events, and a cache read on the summary request. It also checks the outgoing
request prefix and that the fork adds no parent events or tool executions.

This paid test is opt-in (`NOOA_RUN_SUMMARIZER_E2E=1`) and resolves the
`release-gate-openai` and `release-gate-anthropic` registry aliases. Missing
aliases skip; a skip is not successful release evidence. It allows three
requests per provider, caps each at 2,048 output tokens, and disables provider
retries. Private configuration and run results belong in the release repository.
The ordinary PR suite runs the same scenario with mocked HTTP and negative
controls that disable forking, disable summary application, or lose a fact.
