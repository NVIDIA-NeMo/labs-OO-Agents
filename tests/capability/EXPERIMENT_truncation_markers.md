# Experiment: Truncation Marker Comprehension

**Goal.** Find a marker design and an agent schema that LLMs can use *without* a system prompt teaching the format. Inform pformat truncation marker design.

**Headline outcome.** Two independent decisions matter, in roughly equal measure:

- **Marker shape.** A consistent `type(len=N, items=…)` family for length-truncated containers — `list(len=N, items=[…])`, `dict(len=N, items={…})`, `tuple(len=N, items=(…))`, and the same shape extended to strings: `str(len=N, head='…', tail='…')`.
- **Agent schema.** A Pydantic return type bundling `answer` with a `reason` string. Forcing self-justification raises truncation-awareness from ~0% to 60-95%.

The combination hits **88-100% on flagship models** (claude-sonnet, gpt-5.2, gemini-2.5-pro, nemotron-3-super) and **~84-95% on the small/mini matrix** across the recommended fixtures.

---

## Method

### Two agents, identical persona

```python
# tests/capability/agents/truncation_comprehension.py

from pydantic import BaseModel, Field
from typing import Annotated
from nooa import Agent
from nooa.decorators import strategy
from nooa.strategies import CodeActStrategy, PredictStrategy


class Answer(BaseModel):
    answer: Annotated[int | None, Field(description="Integer answer, or None if cannot be determined")]
    reason: Annotated[str, Field(description="Why you picked that answer (one or two sentences)")]


class TruncationComprehensionAgent(Agent):
    """You read rendered Python output (lists, dicts, captured streams) and answer
    questions about it.
    """

    @strategy(PredictStrategy())
    async def answer(self, context: str, question: str) -> Answer:
        """Based on the `context`, answer the `question`.
        Return an integer if the answer can be determined from the data shown.
        Return None if the answer cannot be determined.
        Include a brief reason string explaining your choice.
        """
        ...


# Bare control — same persona, same prompts, only the return type differs.
class TruncationComprehensionAgentBare(Agent):
    """You read rendered Python output (lists, dicts, captured streams) and answer
    questions about it.
    """

    @strategy(PredictStrategy())
    async def answer(self, context: str, question: str) -> int | None:
        """Based on the `context`, answer the `question`.
        Return an integer if the answer can be determined from the data shown.
        Return None if the answer cannot be determined.
        """
        ...


# CodeAct variant — same persona and Answer schema; can write Python to compute.
class TruncationComprehensionAgentCodeAct(Agent):
    """You read rendered Python output (lists, dicts, captured streams) and answer
    questions about it.
    """

    @strategy(CodeActStrategy())
    async def answer(self, context: str, question: str) -> Answer:
        """Based on the `context`, answer the `question`.
        Return an integer if the answer can be determined from the data shown.
        Return None if the answer cannot be determined.
        Include a brief reason string explaining your choice.
        """
        ...
```

### Models tested

| Class | Models |
|---|---|
| Small/mini (~30-80B) | `nemotron3-nano-30b`, `nemotron-super-49b`, `claude-haiku`, `gemini-3-flash-preview`, `gemini-2.5-flash-lite`, `gpt-5-mini`, `qwen3-80b` |
| Flagship | `claude-sonnet`, `gpt-5.2`, `gemini-2.5-pro`, `nemotron-3-super-preview` |

Two models excluded:
- `llama-3.1-8b` — JSON-output instability under PredictStrategy dominated the signal in early rounds.
- `gpt-oss-20b` — works at ~96% under PredictStrategy but breaks at 0% under CodeAct due to tool-calling protocol failures (writes plain English meta-commentary into structured tool-call headers, causing API rejections).

`gemini-3-pro` is auth-blocked for our key; substituted `gemini-2.5-pro`.

### Question set (positional)

For length-truncated containers, all fixtures use the same 7-question pattern parameterized over the same data `[42, 17, 89, 33, 8, …(elided 90)…, 56, 71, 12, 45, 28]`:

| # | Question | Expected | Tests |
|---|---|---|---|
| 1 | How many items total? | `100` | Marker parsing — read `len=N` |
| 2 | What is the minimum value? | `None` | Awareness — elided items could change the answer |
| 3 | What is the first item? | `42` | Visible head, position 1 |
| 4 | What is the 50th item? | `None` | Awareness — position 50 is elided |
| 5 | What is the 3rd item? | `89` | Visible head, position 3 |
| 6 | What is the 9th item? | `None` | Awareness — position 9 is elided; tempting to confuse with "9th visible" |
| 7 | What is the 99th item? | `45` | Visible tail, position 99 |

### Marker shapes tested (apples-to-apples)

```
today_verbose:  [42, 17, 89, 33, 8, ... 90 items not shown ..., 56, 71, 12, 45, 28]
xml:            <list len=100>[42, 17, 89, 33, 8, ..., 56, 71, 12, 45, 28]</list>
pascal:         List(len=100, items=[42, 17, 89, 33, 8, ..., 56, 71, 12, 45, 28])
lower:          list(len=100, items=[42, 17, 89, 33, 8, ..., 56, 71, 12, 45, 28])   ← winner
```

### Container types tested

After `lower` was selected as the recommended shape, container generalization was checked:

```python
# Same data, different container syntax
list(len=100, items=[42, 17, 89, 33, 8, ..., 56, 71, 12, 45, 28])
dict(len=100, items={0: 42, 1: 17, 2: 89, ..., 98: 45, 99: 28})
tuple(len=100, items=(42, 17, 89, 33, 8, ..., 56, 71, 12, 45, 28))

# Pydantic / dataclass instances wrapping a long list field
Team(name='alpha', members=list(len=100, items=[…]), status='active')
Project(name='alpha', tasks=list(len=100, items=[…]), owner='Bob')

# JSON-style dict (double-quoted keys)
{"items": list(len=100, items=[…])}

# Records-of-dicts (genuinely new shape — record-key access, not positional)
list(len=100, items=[{'id': 0, 'name': 'alpha', 'score': 42},
                     {'id': 1, 'name': 'beta',  'score': 17},
                     ...,
                     {'id': 99, 'name': 'zoe',  'score': 28}])
```

### Other pprint mechanics

```python
# Depth truncation (containers shown shallow inside a deeper structure)
dict(len=3, items={'config': {dict: 5 items},
                   'data':   list(len=100, items=[42, 17, 89, …, 45, 28]),
                   'meta':   {dict: 4 items}})

# Cycle markers (self-referential structures)
Node(id=1, name='root', children=list(len=3, items=[…]), parent=<cycle>)
{'self_ref': <cycle>, 'count': 5, 'next': <cycle>}

# Generator markers (lazy iterators not consumed)
{'name': 'log_stream', 'data': <generator>, 'count_so_far': 42}
Job(name='nightly', logs=<generator>, exit_code=0)

# String truncation — three candidate shapes for max_string-truncated fields
v1 'foo'+N             — rich's existing idiom
v2 str(len=N, head='…')             — Pydantic-shape, head only
v3 str(len=N, head='…', tail='…')   — Pydantic-shape with head+tail (winner)
```

### Run command

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --runs 3 --parallel 30 --timeout 240
```

---

## Results

### Headline 1: Marker shape × agent schema (apples-to-apples)

8 small models × 7 questions × 3 runs = 168 samples per cell:

| Marker style | with-reason | bare | uplift |
|---|---|---|---|
| today_verbose | 124/168 (74%) | 69/168 (41%) | **+33pp** |
| xml `<list len=N>[…]</list>` | 130/168 (77%) | 83/168 (49%) | **+28pp** |
| pascal `List(len=N, items=[…])` | 133/168 (79%) | 83/168 (49%) | **+30pp** |
| **lower `list(len=N, items=[…])`** | **141/168 (84%)** | 82/168 (49%) | **+35pp** |

Two independent levers, both essential:

- The bare agent caps at ~50% because it cannot return None on awareness questions. Adding the `reason` field unlocks self-justification: forcing the model to articulate its logic catches cases where the visible data doesn't actually justify the answer. **+30pp uplift.**
- The lowercase Python typename in function-call form wins among shapes by ~5pp. Fewer tokens, more familiar (looks like Python you'd write yourself), no closing tag.

### Headline 2: Flagship models on the recommended config

8 fixtures × 7 questions × 3 runs = 168 samples per model. Recommended config = `lower` shape + `Answer(answer, reason)` schema.

| Model | Total | % |
|---|---|---|
| **claude-sonnet** | **168/168** | **100%** |
| nemotron-3-super-preview | 164/168 | 98% |
| gemini-2.5-pro | 160/168 | 95% |
| gpt-5.2 | 148/168 | 88% |

The recommendation holds at flagship scale: three of four hit ≥95%, all four ≥88%. The remaining gaps are the same reasoning failures (min-from-visible, 9th/99th position math) that affect small models — not new failure modes.

### Container generalization (lower shape + with-reason agent)

Same 7-question pattern adapted per type. 168 samples per cell on small models:

| Container | Total | Notes |
|---|---|---|
| list (baseline) | 141/168 (84%) | original test |
| **dict** | **154/168 (92%)** | best — key-based access avoids the "9th visible" confusion |
| tuple | 137/168 (82%) | structurally identical to list |
| pydantic instance | 132/168 (79%) | wrapping a long list field |
| dataclass instance | 132/168 (79%) | identical repr to pydantic |
| json (double-quoted dict) | 137/168 (82%) | dict with JSON-style quotes |
| **records-of-dicts** (new shape) | **155/168 (92%)** | record-key access ("score of record id=0?") |

Note on pydantic / dataclass / json: these aren't fundamentally different shapes from list — they wrap a long list field with the same `list(len=N, items=[…])` marker. The "records-of-dicts" fixture is the genuinely new shape: a list of structured records, accessed by record id rather than position. It scores as well as the dict case (92%), suggesting the wrapper generalizes to both positional and key-based access patterns.

### Depth + cycle + generator + string truncation

| Concept | Sample shape | Pass rate |
|---|---|---|
| Depth (`{Type: N items}` shallow form) | `dict(len=3, items={'config': {dict: 5 items}, 'data': list(len=100, items=[…]), 'meta': {dict: 4 items}})` | **165/168 (98%)** |
| Cycle (`<cycle>` markers) | `Node(id=1, name='root', children=…, parent=<cycle>)` | **113/120 (94%)** |
| Generator (`<generator>` markers) | `{'name': 'log_stream', 'data': <generator>, 'count_so_far': 42}` | **119/120 (99%)** |
| String — `'foo'+N` (rich legacy) | `log='2024-01-01 INFO startup\nERROR connection failed'+8500` | 64/96 (66%) |
| String — `str(len=N, head=…)` (head only) | `log=str(len=8568, head='…')` | 71/96 (73%) |
| **String — `str(len=N, head=…, tail=…)`** | `log=str(len=8568, head='…', tail='…')` | **130/132 (98%)** |
| String — `str(len=N, start=…, end=…)` | (same shape, start/end labels) | 127/132 (96%) |
| String — `str(len=N, prefix=…, suffix=…)` | (same shape, prefix/suffix labels) | 126/132 (95%) |
| **String — `str(len=N, [:50]='…', [-50:]='…')`** | (slice expressions as keys) | **129/132 (98%)** |

**The string label barely matters** — `head`/`tail`, `[:50]`/`[-50:]`, `start`/`end`, `prefix`/`suffix` all score within 3pp of each other. What matters is the **shape**: total length upfront via `len=`, plus two visible chunks at the boundaries. The slice-key form has the practical advantage of making the offsets explicit (`[:50]` makes "first 50 chars" unambiguous without measuring), at the cost of being non-strictly-valid Python — but pformat output is already not strictly valid Python (e.g. `'foo'+N`, `... +N`), so this isn't a new compromise.

Three significant findings here:

- **Depth is universally understood.** `{Type: N items}` shallow form scores 100% across all 8 models in earlier rounds and 98% in the latest matrix. Most reliable marker in the system.
- **`<cycle>` and `<generator>` work well.** Models recognize the markers and correctly refuse to invent values for them. Sibling field extraction in cycle/generator-bearing structures is reliable.
- **String head+tail wins by a wide margin.** The Pydantic-shape `str(len=N, head='…', tail='…')` form scores 96% — 30pp ahead of rich's existing `'foo'+N` idiom (66%). Reasons: total length is upfront (no arithmetic), tail is visible (so "last word" type questions work), and the shape mirrors `list(len=N, items=[…])` so the agent learns it once.

---

## Failure-mode analysis

We extracted the model's `reason` text on every wrong answer (315 failures across cmp18 / cmp19 / cmp20) and clustered them. Six distinct patterns emerged.

### Failure 1 — Min-from-visible (83 cases, 100% of `min` failures)

**What:** Asked "what is the minimum value across all items?" with `len=100`, model returns `8` (the minimum of the visible 10 items).

**Sample reasons:**
- *"The smallest visible is 8. The truncated items are not shown."*
- *"The minimum value in the list is 8, as it is the smallest number shown in the truncated list."*

**Pattern:** Model correctly identifies that the data is truncated (often verbatim) but answers from the visible portion anyway.

### Failure 2 — Position-mapping confusion on the 9th item (84 cases)

**What:** Asked for the 9th item in a 100-list with positions 1-5 and 96-100 visible. Model maps "9th" to a visible item.

- 44× → `8` (last visible head item, position 5)
- 12× → `33` (4th head item)
- 8× → `28` (last visible tail item)
- 7× → `56` (first tail item)

**Sample reason:** *"The 9th item in the list (1-indexed) is explicitly shown as 8 in the truncated representation."*

**Pattern:** Model treats the visible items as the entire list, indexing into them as if `len=N` doesn't apply.

### Failure 3 — Off-by-one on 99th tail position (53 cases)

**What:** Asked for the 99th item. Visible tail is `[…, 56, 71, 12, 45, 28]` at positions 96-100. Correct answer is 45. Model returns 28.

**Sample reason:** *"The 99th item is the second to last item, which is 28."*

**Pattern:** Model says "second to last" but reports the last value.

### Failure 4 — Refusal on visible content (34 cases on 99th)

**What:** Asked for the 99th item — which IS visible in the tail — model answers None.

**Sample reason:** *"The 99th item cannot be determined."*

**Pattern:** Model is overly conservative; sees `…` and refuses without checking whether the position lands in the visible tail.

### Failure 5 — Count miscalibration (12 cases on count)

**What:** Even with `len=N` upfront, models sometimes answer wrong on count.

- 5× → `95` (5 head + 90 elided, forgot the 5 tail)
- 3× → `90` (just the elided count)
- 2× → `None` (refusal)

**Pattern:** Some models do their own arithmetic from the visible items and miss the `len=N` shortcut.

### Failure 6 — 0-vs-1-indexed (1 case on 3rd)

**What:** Asked for the 3rd item (visible at position 3). Model returns the 2nd item.

**Sample reason:** *"The 3rd item is at index 2 in 0-indexed terms."*

**Pattern:** Confusion between 1-indexed and 0-indexed despite the question stating "(1-indexed)".

### Net pattern

Five of the six failure modes share a common cause: **the model defaults to "answer from the visible portion" and applies its position arithmetic to the visible items**, rather than propagating `len=N` through to the implication that most positions are not visible. Failure 4 is the opposite extreme — refusing even when the answer is visible.

The recommended marker + schema combination pushes models from "always answer from visible" (47%) up to "answer from visible OR refuse correctly" (~84-95%). Closing the remaining gap requires more than marker / schema design — it's a model-reasoning limitation that needs prompt-level guidance ("propagate len through to position math") or chain-of-thought patterns.

---

## CodeAct comparison

Same fixtures, same persona, same Pydantic Answer schema — but using `CodeActStrategy` so the model can write Python to compute. Hypothesis: models that fail at marker arithmetic should improve when they can write a script.

### Per-model totals on 3 fixtures (lower-awareness, records, string head+tail)

| Model | CodeAct | PredictStrategy | Δ |
|---|---|---|---|
| claude-haiku | 54/54 (100%) | 54/54 (100%) | 0 |
| claude-sonnet | 54/54 (100%) | 54/54 (100%) | 0 |
| gemini-2.5-flash-lite | 45/54 (83%) | 43/54 (80%) | +2 |
| gemini-2.5-pro | 52/54 (96%) | 52/54 (96%) | 0 |
| gemini-3-flash-preview | 54/54 (100%) | 54/54 (100%) | 0 |
| gpt-5-mini | 54/54 (100%) | 54/54 (100%) | 0 |
| gpt-5.2 | 53/54 (98%) | 51/54 (94%) | +2 |
| nemotron-3-super-preview | 52/54 (96%) | 50/54 (93%) | +2 |
| nemotron-super-49b | 48/54 (89%) | 41/54 (76%) | **+7** |
| nemotron3-nano-30b | 44/54 (81%) | 54/54 (100%) | -10 |
| qwen3-80b | 38/54 (70%) | 37/54 (69%) | +1 |

### Three findings

**1. CodeAct doesn't help capable models.** Anyone already near 100% on PredictStrategy stays there. claude-haiku, claude-sonnet, gemini-3-flash, gpt-5-mini all stay at 100% under CodeAct.

**2. CodeAct helps some arithmetic-weak models.** `nemotron-super-49b` jumps **+7** (76% → 89%) — when forced to write code, it computes the answer correctly rather than guessing from visible items. Smaller gains for gpt-5.2 (+2) and gemini-2.5-flash-lite (+2).

**3. Tool-calling protocol failures.** `nemotron3-nano-30b` slips from 100% on PredictStrategy to 81% on CodeAct (model invokes `return_result` correctly most of the time but occasionally emits malformed code). `gpt-oss-20b` (excluded from this matrix) was a starker case at 96%→0% — the model wrote plain-English meta-commentary into structured tool-call headers, causing the API to reject after 3 framework retries. Mirror image of llama-3.1-8b's PredictStrategy-JSON failures from earlier rounds. Different protocols, different small-model brittleness.

### Sample CodeAct trace

When CodeAct works, the model writes code like:

```python
import re
context = "list(len=100, items=[42, 17, 89, 33, 8, ..., 56, 71, 12, 45, 28])"
m = re.search(r'len=(\d+)', context)
total = int(m.group(1))    # → 100
return_result(Answer(answer=total, reason=f"Read len={total} from the marker."))
```

That bypasses the small-model arithmetic limit by parsing the marker programmatically.

### Implication for recommendation

CodeAct is not a universal win. For most production setups, PredictStrategy with the recommended marker shape and `Answer(answer, reason)` schema is sufficient. CodeAct is a useful tool for models that struggle with arithmetic on truncation markers, but only when the model is reliable enough to use the tool-calling protocol.

---

## Recommendations for Truncation 3.0

### Marker shapes (final)

Use **lowercase Python typenames in function-call form** for length-truncated containers, including strings:

```
list(len=N, items=[head, ..., tail])
dict(len=N, items={k1: v1, ..., kN: vN})
tuple(len=N, items=(...))
set(len=N, items={...})
str(len=N, head='...', tail='...')      ← extends the same pattern to strings
```

Pydantic / dataclass instances render as `Foo(field=value, …)` — their type name is already in the repr; no wrapper needed. When such an instance has a long list / dict / str field, that field uses the wrapper internally.

Other markers (unchanged):

- `{Type: N items}`, `[Type: N items]`, `Type(...)` for depth-truncated containers (universally understood, 100% in the matrix).
- `<cycle>` for cyclic references — 94% comprehension.
- `<generator>` for unconsumed lazy iterators — 99% comprehension.
- `<truncated>...head...tail...</truncated>` for L2 capture overflow (different layer; 91% comprehension in earlier rounds).

### Agent schema (when using PredictStrategy)

Default to a structured Pydantic return type that bundles the answer with a `reason` field:

```python
class Answer(BaseModel):
    answer: int | None
    reason: str
```

The reason field is the single biggest lever for truncation awareness. Worth ~30pp across all marker styles.

### Where this design *doesn't* solve the problem

Truncation awareness on harder reasoning (min, position math) caps around 60-70% on the small/mini matrix even with the recommended shape and schema. Closing that gap is a reasoning problem, not a marker problem. Solutions live in prompt design, per-question schema constraints, or chain-of-thought / CodeAct patterns — not in better markers.

Flagship models close most of this gap on their own: claude-sonnet hits 100%, others 88-98%.

---

## Caveats and limitations

- **`gpt-oss-20b` was excluded** because the model could not reliably emit valid CodeAct tool calls. Under PredictStrategy it scored ~96% on the same fixtures; under CodeAct it scored 0% because it wrote plain-English meta-commentary inside tool-call message headers, causing API errors after framework retries. This is a tool-calling protocol failure, not a marker-comprehension failure. Excluding it is a model-suitability call, not a comment on the marker design.
- **Sample size:** 24 samples per (style, question) cell. Enough to distinguish 0/24 from 24/24 cleanly; finer comparisons (e.g. 14 vs 16 of 24) are within noise.
- **One container size** (100 items, 10 visible). Edge cases at very small or very large containers are not exercised here.
- **Integer-typed answers** (because the agent's `int | None` constraint). String / boolean / structured answers might surface different failure modes; we trade off question variety for clean schema-validated comparison.
- **Position arithmetic is a stress test.** Many real LLM workloads don't ask "what is the Nth item" in a truncated list. Lower-failure tasks (sibling field extraction, simple counts, key lookup) hit ≥95% even on small models.

---

## Related artifacts

- Test agent: [tests/capability/agents/truncation_comprehension.py](agents/truncation_comprehension.py)
- Test config: [tests/capability/config_truncation.yaml](config_truncation.yaml)
- Test fixtures: `tests/capability/data/truncation_aware_*.jsonl`, `tests/capability/data/truncation_str_v*.jsonl`
- Branch: `test/truncation-comprehension` (MR !147)
