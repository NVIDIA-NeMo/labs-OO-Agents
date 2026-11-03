# Type-hint resolution during formatting

This experiment measures whether reusing resolved annotations within one render
reduces formatting time without changing the text passed to an agent.

The input is 50 synthetic records containing 20 events each. Each event has
visible fields and one hidden field. The formatter uses a 100,000-character cap,
depth 12, at most 200 elements per container, and a 10,000-character string limit.
There are no model calls or external datasets.

Run from the repository root:

```bash
uv run python experiments/agentdoc_type_hints/benchmark.py
```

Run the same script against the base revision and the changed revision in the
same Python environment. Compare median wall time and the rendered SHA-256.
The script reports each repetition so timing variation remains visible.

On November 2, 2026, a macOS/Python 3.13 run measured a median of 1.365 seconds
before the change and 0.230 seconds with the scoped cache, about 6 times faster.
Both outputs had SHA-256
`46a6ef2023be8b9c57bff1944a719c3c852e6c7174fa9af279aeadef11715b84`.
These are formatter measurements, not end-to-end agent latency estimates.

The explicit-cache variant passes a private annotation cache through formatter
recursion and built-in field extraction. Public extraction and visibility
functions retain their existing signatures and resolve annotations without any
ambient cache. A reentrant public render starts its own cache; registry and
protocol callbacks do not inherit the caller's cache. Successful resolutions
are reused for the duration of that render, while instance visibility overrides
remain live and failed resolutions can retry.

A matched macOS/Python 3.13 comparison on November 3 measured 1.284 seconds
without caching and 0.204 seconds with explicit cache passing (three repetitions
each), about 6.3 times faster. The output hash matched the value above.
