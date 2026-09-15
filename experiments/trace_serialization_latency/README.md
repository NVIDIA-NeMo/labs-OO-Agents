# Trace serialization latency reproduction

## Research question

Does tracing fully serialize large, JSON-compatible method arguments before applying
the 50,000-character trace limit, and does that serialization block the agent's
asyncio event loop?

## Experiment design

`reproduce.py` builds a JSON-compatible argument shaped like a method input:

```python
{"args": ([{"content": <shared 1 KiB string>, "index": ...}, ...],), "kwargs": {}}
```

The source object is deliberately much smaller than its JSON representation because
the content string is shared. The script compares:

- `current`: `OpenInferenceHooks._safe_json_value`, the production tracing path.
- `hook-before`: the complete `OpenInferenceHooks.before_agent_call` production hook.
- `bounded-first`: formatting the two top-level values with the existing bounded
  trace formatter before JSON encoding. This is a diagnostic comparison, not a
  proposed wire format.

It also runs the complete `before_agent_call` hook in an asyncio task while a 10 ms
heartbeat is scheduled. The largest heartbeat delay measures how long unrelated agent
work can be prevented from running.

A second case models the relevant shape from Gaia's workaround: a list of Pydantic
candidate entities, each containing nested trials with large outputs and metadata. The
same candidate and trial are deliberately shared so the input consumes modest memory;
the production serializer nevertheless formats them again at every list position.

Key metrics are elapsed serialization time, output size, peak process RSS, and maximum
asyncio heartbeat delay. Input sizes are estimates of the expanded JSON payload; exact
encoded sizes are reported.

## How to run

From the repository root:

```bash
uv run python experiments/trace_serialization_latency/reproduce.py
```

Use larger inputs to extrapolate or reproduce machine-specific long stalls:

```bash
uv run python experiments/trace_serialization_latency/reproduce.py \
  --sizes-mib 64 256 1024 --event-loop-mib 1024 --entity-counts 1000
```

On a machine similar to the one used below, this targets an approximately one-minute
stall in the complete production before hook while skipping the other serializer
variants:

```bash
uv run python experiments/trace_serialization_latency/reproduce.py \
  --sizes-mib 8 --event-loop-mib 8 --entity-counts 5000 \
  --serializers hook-before
```

The script does not enable exporters, write traces, or call an LLM. This isolates the
argument serialization performed before an agent method begins.

## Results summary

Run on 2026-09-15 with Python 3.13.3 at commit `6097d05e`:

| Expanded JSON | `_safe_json_value` | Full before hook | Bounded-first | Final trace value |
|---:|---:|---:|---:|---:|
| 8.25 MiB | 0.0200 s | 0.0189 s | 0.0008 s | 6,513 chars |
| 66.05 MiB | 0.1435 s | 0.1440 s | 0.0014 s | 6,513 chars |
| 264.39 MiB | 0.5748 s | 0.5709 s | 0.0031 s | 6,513 chars |

Peak process RSS reached 540.1 MiB in the 264.39 MiB case. The 10 ms asyncio
heartbeat was delayed by 0.5828 s while the full before hook took 0.5826 s. Thus the
production cost grows with the complete input even though every case stores the same
6,513-character trace attribute, and the synchronous hook blocks unrelated event-loop
work for essentially its entire duration.

The entity-shaped case reproduces much worse latency with modest source-object memory:

| Candidate positions | `_safe_json_value` | Full before hook | Bounded-first | Final trace value |
|---:|---:|---:|---:|---:|
| 10 | 0.5623 s | 0.5624 s | 0.4477 s | 6,503 chars |
| 100 | 2.0029 s | 2.0055 s | 0.8797 s | 6,511 chars |
| 1,000 | 12.0059 s | 12.0764 s | 0.8837 s | 6,511 chars |
| 5,000 | — | 57.2013 s | — | 6,511 chars |

The 1,000-position input stalls for 12 seconds, and the targeted 5,000-position run
reproduced a 57.2-second stall in the complete production hook. Gaia's entity compaction
is effective because it bounds the collection before the production JSON encoder
repeatedly formats every nested domain object.

After replacing the encode-then-truncate path with bounded streaming, the same targeted
5,000-position production hook completed in **0.0151 s** and stopped at the 50,000-character
limit. A 1.03 MiB native-JSON input completed in 0.0018 s, and its full hook delayed a
10 ms asyncio heartbeat by at most 0.0023 s. The fix therefore avoids traversing the
unrecorded remainder while preserving the prior JSON representation for inputs that fit.
