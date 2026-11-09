# Type-hint resolution during formatting

This experiment measures whether reusing resolved annotations within one
formatting call reduces formatting time without changing the text passed to an agent.

The input is 50 synthetic records containing 20 events each. Each event has
visible fields and one hidden field. The formatter uses a 100,000-character cap,
depth 12, at most 200 elements per container, and a 10,000-character string limit.
There are no model calls or external datasets.

Run from the repository root:

```bash
uv run python experiments/agentdoc_type_hints/benchmark.py
```

Run the same script against the base revision and the changed revision in the
same Python environment. Compare median wall time and the output text SHA-256.
The script reports each repetition so timing variation remains visible.

On November 2, 2026, a macOS/Python 3.13 run measured a median of 1.365 seconds
before the change and 0.230 seconds with the scoped cache, about 6 times faster.
Both outputs had SHA-256
`46a6ef2023be8b9c57bff1944a719c3c852e6c7174fa9af279aeadef11715b84`.
These are formatter measurements, not end-to-end agent latency estimates.

A November 3 follow-up added explicit cache expiration for copied execution
contexts and isolated inherited contexts in other threads. In the same environment,
the unchanged base measured 1.276 seconds and the completed ContextVar version
measured 0.211 seconds (three repetitions each, about 6 times faster), with the same
output hash above. A separate cache-miss check formatting 200 distinct one-field
dataclass types measured 11.9 ms before and 9.7 ms after (seven repetitions each),
with identical output hashes. These small timings are descriptive, not a precise
claim of improvement for inputs without repeated types.

Resolved annotations are stable within the outermost formatting call, including
recursive and reentrant formatting on the same thread. Imperative instance/class `spec()`
overrides remain live. The cache holder is expired and cleared on normal or
exceptional exit, so a context captured by a formatter callback or inherited by a
later task cannot reuse stale annotations or retain cached objects. An inherited
context used in another thread resolves independently. Inspection outside a
formatting call continues to resolve annotations without caching.
