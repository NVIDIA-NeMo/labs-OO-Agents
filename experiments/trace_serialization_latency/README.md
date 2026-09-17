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

- `current`: the bounded structural JSON preview used by production tracing.
- `hook-before`: the complete `OpenInferenceHooks.before_agent_call` production hook.
- `legacy`: a local recreation of the removed custom-object fallback that repeatedly
  rendered every Pydantic object before the outer encoder could stop.

It also runs the complete `before_agent_call` hook in an asyncio task while a 10 ms
heartbeat is scheduled. The largest heartbeat delay measures how long unrelated agent
work can be prevented from running.

A third case uses one large string scalar. This detects a distinct property of the
standard-library encoder: `json.dump()` normally materializes a complete encoded string
token before passing it to a file-like writer, so a bounded writer alone cannot prevent
latency and a temporary allocation proportional to that scalar.

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

Run the isolated single-string benchmark (one fresh process per size so RSS growth is
comparable):

```bash
uv run python experiments/trace_serialization_latency/scalar.py
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

The final implementation builds a bounded JSON-compatible preview tree and then uses
ordinary `json.dumps`. It never runs arbitrary rendering hooks and stops inspecting the
source after fixed character, node, and depth budgets. The tables above remain the
historical motivation; rerun the experiment to compare the current hook with the local
`legacy` recreation on the current machine.

Final implementation measurements on 2026-09-17 with Python 3.13.3:

| Case | Bounded codec | Full before hook | Trace value |
|---:|---:|---:|---:|
| 1.03 MiB expanded native JSON | 0.0008 s | 0.0009 s | 50,000 chars |
| 5,000 repeated Pydantic candidates | 0.0003 s | 0.0004 s | 50,000 chars |

During the native-JSON hook measurement, the 10 ms asyncio heartbeat was delayed by at
most 0.0014 s. Isolated 1, 16, 64, and 256 MiB string inputs each took 0.0003 s after
source allocation, produced exactly 50,000 characters, and showed no measurable peak-RSS
growth from serialization.

The single-scalar review case was measured separately before and after fragmenting JSON
string encoding. RSS growth is measured after constructing the source string, so it
captures only serialization's temporary allocation:

| String scalar | Writer-only latency | Writer-only RSS growth | Fragmented latency | Fragmented RSS growth |
|---:|---:|---:|---:|---:|
| 1 MiB | 0.0037 s | 2.0 MiB | 0.0016 s | ≤0.5 MiB |
| 16 MiB | 0.0398 s | 32.2 MiB | 0.0016 s | ≤0.5 MiB |
| 64 MiB | 0.1947 s | 128.0 MiB | 0.0016 s | ≤0.5 MiB |
| 256 MiB | 0.7127 s | 512.0 MiB | 0.0017 s | ≤0.5 MiB |

The small cases confirm that ordinary JSON encoding is fast. The larger cases establish
that encoding a complete source scalar is still material when tracing retains only a
small preview. The final codec copies and encodes only a fitting string prefix, so both
costs depend on the trace limit instead of the discarded string tail.
