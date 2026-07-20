# Truncation Reference

Every place where output is size-limited in the framework — what value is being
truncated, where it ends up, why, and what the LLM actually sees.

All examples below are the **real output** of running the relevant code path.

## Two mechanisms

### Structural (`pformat` with `max_length/string/depth`)

Clips at element/string/depth boundaries. Emits compact inline notices like
`... 150 items not shown ...` or `'xxx...'+700`. Never emits a prose "Output
too large" banner. Use when showing **shape and flavor** to give the LLM a feel
for the data without reproducing it in full.

### Char-cap (`safe_pformat` / `TruncatingStringIO` with `max_chars`)

Hard character ceiling with head + tail split. When the cap fires it emits:

```
<truncated-output>
Output too large (N chars). Showing first X and last Y chars.

<head>

... Z chars not shown ...

<tail>
</truncated-output>
```

Use when the LLM needs the **actual content** of a value, not just its shape.

---

## Configuring truncation

All `tc.*` parameters come from `TruncationConfig`, set per-agent:

```python
from nooa.config.truncation_config import TruncationConfig

class MyAgent(Agent, llm=llm):
    _truncation = TruncationConfig(
        max_block_chars=20_000,      # per-block LLM-visible limit; also the
                                     # safe_pformat cap for all char-cap sites
                                     # (single pipeline — default 20K)
        max_stdout_chars=50_000,     # stdout per cell (default 50K)
        max_stderr_chars=20_000,     # stderr per cell (default 20K)
        stdout_tail_chars=None,      # tail window; None = limit//2 (default None → 25K)
        max_pprint_elements=50,      # container items in structural display (default 50)
        max_pprint_string=500,       # string chars in structural display (default 500)
        max_pprint_depth=4,          # nesting depth in structural display (default 4)
    )
```

---

## Truncation sites

### 1. stdout / stderr capture — `actor.py:709-715`

**What:** Raw text printed to `sys.stdout` / `sys.stderr` by agent-generated Python
code during a REPL execution cell.

**Where:** Captured into a `TruncatingStringIO` buffer during `execute_python()`;
flushed into a `PythonOutput` event that lands in the LLM's next message.

**Why:** Agent-generated code may `print()` arbitrarily large objects. Head+tail
preserves both the start of execution (setup, intermediate values) and the final
state.

**Mechanism:** Stream char-cap.  
**Parameters:** `limit=tc.max_stdout_chars` (50 K default), `tail_chars=None → limit//2 = 25 K`.
Stderr: `limit=tc.max_stderr_chars` (20 K), same tail.

**Short output — no truncation:**
```
print result: 42
status: done
```

**Long output (50 lines, limit=500 for demo — real limit is 50 K):**
```
<truncated-output>
Output too large (994 chars). Showing first 250 and last 250 chars.

Line 0: result=0
Line 1: result=1
Line 2: result=4
Line 3: result=9
Line 4: result=16
Line 5: result=25
Line 6: result=36
Line 7: result=49
Line 8: result=64
Line 9: result=81
Line 10: result=100
Line 11: result=121
Line 12: result=144
Line 13: resul

... 494 chars not shown ...

ne 38: result=1444
Line 39: result=1521
Line 40: result=1600
Line 41: result=1681
Line 42: result=1764
Line 43: result=1849
Line 44: result=1936
Line 45: result=2025
Line 46: result=2116
Line 47: result=2209
Line 48: result=2304
Line 49: result=2401
</truncated-output>
```

Note: the head/tail split mid-line (`Line 13: resul` / `ne 38:`) is correct
behaviour — the split is at byte position, not line boundary.

---

### 2. Return value pre-format — `actor.py:2082`

**What:** The Python object passed to `return_result(value)` by agent-generated
code — the agent method's final answer.

**Where:** Serialised to a string, placed into a `ReturnValue` event. Goes through
block-level truncation (`max_block_chars`) in the rendering pipeline — same cap,
so block truncation is a no-op for values that already fired here.

**Why:** Full content needed so the reflexion LLM can evaluate quality.

**Mechanism:** Char-cap (single layer).  
**Parameters:** `max_chars=tc.max_block_chars` (20 K default).

**Normal return value:**
```
{'answer': 42, 'items': [1, 2, 3]}
```

**Large object (200-char limit to illustrate format — real limit is 20 K):**
```
<truncated-output>
Output too large (68,893 chars). Showing first 100 and last 100 chars.

{
    'key_0': 'vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv',
    'key_1': 'vvvvvvvvvvvvvvvvv

... 68,693 chars not shown ...

vvvvvvvvvvvvvvvvvvvvvvvvvvv',
    'key_999': 'vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv',
}
</truncated-output>
```

---

### 3. `Out` repr — `out_accessor.py:144`

**What:** The combined display of all recorded outputs when agent-generated code
evaluates `Out` as an expression or calls `print(Out)`.

**Where:** The `__repr__` of `OutAccessor` — shown in the REPL output that lands
in the LLM's message.

**Why:** `Out` accumulates all cell outputs. Reprinting every full value would
flood the REPL context with redundant data. Each entry is capped so the LLM
sees the shape of each result without being overwhelmed.

**Note:** `Out[n]` indexing returns the raw Python value untruncated —
truncation only applies to the combined `repr(Out)` display.

**Mechanism:** Char-cap per item.  
**Parameters:** `max_chars=500` (hardcoded — intentionally small; this is a quick
preview, not the full value).

**Int — no truncation:**
```
Out[1]: 42
```

**600-char string — fires:**
```
Out[2]: <truncated-output>
Output too large (600 chars). Showing first 250 and last 250 chars.

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

... 100 chars not shown ...

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
</truncated-output>
```

**1 000-item list — fires:**
```
Out[3]: <truncated-output>
Output too large (8,893 chars). Showing first 250 and last 250 chars.

[
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
  

... 8,393 chars not shown ...

 972,
    973,
    974,
    975,
    976,
    977,
    978,
    979,
    980,
    981,
    982,
    983,
    984,
    985,
    986,
    987,
    988,
    989,
    990,
    991,
    992,
    993,
    994,
    995,
    996,
    997,
    998,
    999,
]
</truncated-output>
```

---

### 4. Context block — `context_builder.py:297-298`

**What:** A Python object stored via `self.context["key"] = value`.

**Where:** Serialised to a string, placed into a `ResolvedBlock`, rendered into
the LLM's system prompt.

**Why:** Full content needed for the LLM to use the stored value.

**Mechanism:** Char-cap (single layer).  
**Parameters:** `max_chars=tc.max_block_chars` (20 K default). Block-level truncation
uses the same limit, so it is a no-op — one notice, never two.

**Small dict — no truncation:**
```
{'plan': 'step 1, step 2, step 3'}
```

**1 K string, max_chars=300 (real output — real limit is 20 K):**
```
<truncated-output>
Output too large (1,000 chars). Showing first 150 and last 150 chars.

word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word 

... 700 chars not shown ...

word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word 
</truncated-output>
```

---

### 5. Plain event content — `plain_formatter.py:38`, `codeact_lite.py:84,166,191`

**What:** An event's `.value` field when rendering the conversation history in
CodeActLite strategy (which uses plain-text event rendering instead of XML
context-blocks).

**Where:** Each event entry in the LLM's message history.

**Why:** Full content needed so the LLM can see prior results.

**Mechanism:** Char-cap (single layer).  
**Parameters:** `max_chars=tc.max_block_chars` (20 K default). Block-level truncation
uses the same limit, so it is a no-op — one notice, never two.

**Small event — no truncation:**
```
result = [1,2,3]
computed ok
```

**Same mechanism as site 4 — identical output format.** The head and tail are
raw cuts with no added ellipses. Example with max_chars=300:
```
<truncated-output>
Output too large (1,000 chars). Showing first 150 and last 150 chars.

word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word 

... 700 chars not shown ...

word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word 
</truncated-output>
```

---

### 6. Prefill parameter preview — `prefill.py:145-167`

**What:** Each input parameter to the agent method being called, printed at the
start of a CodeAct session.

**Where:** Printed to stdout in the first assistant turn (prefill code), which
is then captured by the stdout `TruncatingStringIO` (site 1) and lands in the
LLM's `PythonOutput` context. The LLM reads this to understand the shape of its
inputs before writing solution code.

**Why:** Structural truncation shows the LLM the *shape* of its inputs — how
many items, how long strings are, how deep nesting goes — without dumping a full
dataset into the prompt. The literal parameter values are embedded in the
generated code string so the LLM learns the `pprint()` API.

**Mechanism:** Structural (via generated `pprint()` call in prefill code).  
**Parameters:** `max_length=tc.max_pprint_elements` (50), `max_string=tc.max_pprint_string`
(500), `max_depth=tc.max_pprint_depth` (4).

**Small list — no truncation:**
```
[0, 1, 2, 3, 4]
```

**200-item list of ints, max_length=50 — `max_length` fires (inline format for simple scalars):**
```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, ... 150 items not shown ..., 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
```

**30-item list of dicts, max_length=10 — multi-line format for complex objects:**
```
[
    {'id': 0, 'name': 'item_0'},
    {'id': 1, 'name': 'item_1'},
    {'id': 2, 'name': 'item_2'},
    {'id': 3, 'name': 'item_3'},
    {'id': 4, 'name': 'item_4'},
    ... 20 items not shown ...
    {'id': 25, 'name': 'item_25'},
    {'id': 26, 'name': 'item_26'},
    {'id': 27, 'name': 'item_27'},
    {'id': 28, 'name': 'item_28'},
    {'id': 29, 'name': 'item_29'},
]
```

**1 200-char string — `max_string` fires:**
```
'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'+700
```

**5-level dict — `max_depth` fires:**
```
{'a': {'b': {'c': {'d': {dict: 1 items}}}}}
```

**Realistic — list of user dicts with long bio and many scores:**
```
[
    {
        'name': 'Alice',
        'bio': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'+300,
        'scores': [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            ... 30 items not shown ...
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        ],
    },
]
```

**Note on numpy/pandas:** The element-aware `... N items not shown ...` format only
applies to Python containers (`list`, `dict`, `tuple`, `set`). numpy arrays have no
`__dict__`, so pformat falls back to `repr()`, which uses numpy's own internal
truncation (`[0 1 ... 98 99]`). pandas DataFrames have `__dict__` and are formatted
as structured instances (`DataFrame(field=val, ...)`) — not as a table. The
`max_string` cap applies after the fact to cap the repr output for both.

---

### 7. Validation / argument error values — `codeact_errors.py`, `generated_code.py`, `predict.py:230`

**What:** The value that failed type validation — an argument passed to an agent
method, a field in the LLM's structured output, or a return value that didn't
match the declared return type.

**Where:** Embedded in a `ToolError` or `ValidationError` event. The LLM reads
it to understand what it returned vs. what was expected, and how to fix it.

**Why:** Structural truncation gives the LLM enough of the value's shape to
diagnose the error without flooding the error message with a full dump. The LLM
doesn't need to reproduce the value — it needs to understand the type mismatch.

**Mechanism:** Structural.  
**Parameters:** `max_length=tc.max_pprint_elements` (50), `max_string=tc.max_pprint_string`
(500), `max_depth=tc.max_pprint_depth` (4).

**Wrong type — int:**
```
42
```

**Wrong type — 200-item list (`max_length` fires, head=25 items, tail=25 items):**
```
[
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    ... 150 items not shown ...
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
]
```

**Nested object — users with 100-item scores list:**
```
{
    'users': [
        {
            'name': 'Alice',
            'scores': [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                ... 50 items not shown ...
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
            ],
        },
    ],
}
```

---

### 8. PredictStrategy parameter size guard — `predict.py:317-347`

**What:** Each argument passed to a method using `PredictStrategy`.

**Where:** Not rendered to the LLM — this is a **measurement-only** path. A
`TruncatingStringIO` is used as a size probe; if `was_truncated` is True, a
`ValueError` is raised before any LLM call is made.

**Why:** PredictStrategy is single-shot. A silently truncated 200 K input
produces wrong output with no indication of failure. Failing loudly forces the
caller to chunk, summarise, or explicitly raise the limit.

**Mechanism:** `TruncatingStringIO` for measurement (repr for non-strings; string
fast-path uses `len()` directly to skip the `repr()` allocation); `ValueError` on overflow.  
**Parameters:** `limit=PredictConfig.max_param_chars` (default 200 K).
Demo below uses limit=10 K to show the firing behaviour.

```
short string (33 chars)            → PASS → LLM call proceeds
list of 100 ints (390 chars)       → PASS → LLM call proceeds
500 × 50-char strings (27,000 chars) → FAIL → ValueError raised:
```
```
PredictStrategy: parameter 'data' is 27,000 chars (repr), exceeding max_param_chars=10,000.
Chunk the input, summarise it, or raise PredictConfig(max_param_chars=...) if the size is intentional.
```

---

### 9. Reflexion result — `reflexion.py:295`

**What:** The agent method's return value, formatted for the reflexion LLM to
evaluate for quality.

**Where:** Embedded in a `Feedback` event sent as the reflection prompt. Goes
through block-level truncation in the rendering pipeline — same cap, so block
truncation is a no-op for values that already fired here.

**Why:** The reflexion LLM must read the actual content to judge quality —
structural truncation would hide too much.

**Mechanism:** Char-cap (single layer).  
**Parameters:** `max_chars=tc.max_block_chars` (20 K default).

**Normal result dict — no truncation:**
```
{'summary': 'Analysis complete', 'score': 0.95, 'issues': []}
```

**Large dict result, max_chars=300 (real output — real limit is 20 K):**
```
<truncated-output>
Output too large (345 chars). Showing first 150 and last 150 chars.

{
    'summary': 'Analysis complete Analysis complete Analysis complete Analysis complete Analysis complete Analysis complete Analysis complete Analys

... 45 chars not shown ...

te Analysis complete Analysis complete Analysis complete Analysis complete Analysis complete Analysis complete Analysis com'+600,
    'score': 0.95,
}
</truncated-output>
```

---

### 10. CurrentCall argument display — `current_call.py:117`

**What:** The actual argument values passed to the current agent method call,
formatted for inclusion in a prompt template via `{call.format_parameters_as_code()}`.

**Where:** Embedded inside docstring templates (`predict.py`, `pure_python.py`)
that become system prompt blocks. Those blocks go through block-level truncation
at `max_block_chars`.

**Why:** The LLM needs to see the real argument values to work with them.
`PredictStrategy` additionally guards against large params with `_assert_param_sizes()`
which raises before the prompt is built. `PurePythonStrategy` relies on block
truncation as the backstop.

**Mechanism:** Char-cap (500 K agentdoc default). Block truncation at
`max_block_chars` applies to the containing block.  
**Parameters:** `max_chars` defaults to 500 K — only fires for extreme values
(> 500 K repr); block truncation at 20 K is the effective LLM-visible limit.

**String arg — no truncation:**
```
analyze this document carefully
```

**200-item list — neither cap fires (1,693 chars; well under both limits):**

`safe_pformat` has no element-wise `max_length`, so all items appear. At 1,693 chars
this is well under the 500 K safe_pformat cap and the 20 K block truncation limit —
no truncation fires at all. Both caps exist for extreme inputs (> 500 K single value,
or > 20 K total block).

```
[
    0,
    1,
    2,
    3,
    ...
    199,
]
```
_(abbreviated for readability — all 200 items appear)_

---

### 11. Summarization event body — `summarization.py:397`

**What:** An event from the agent's history (any type — `PythonOutput`,
`ReturnValue`, `Feedback`, etc.), formatted as input for the summarization LLM.

**Where:** The body of a `ResolvedBlock` passed to the summarization agent.

**Why:** Current safety net. Events arriving at the summarizer should ideally be
pre-truncated at the source (stdout/stderr are capped at capture time; LLM
responses are bounded by max_tokens). The `safe_pformat` here protects against
cases where a raw `value` field (e.g. a large Python return value) was never
bounded. **Design intent:** once all events are guaranteed pre-truncated, this
site should be removed.

**Mechanism:** Char-cap (single layer).  
**Parameters:** `max_chars=self._truncation.max_block_chars` (20 K default).

**Normal `PythonOutput` event:**
```
PythonOutput(tool_call_id='tc-1', execution_status=<ResultStatus.COMPLETE: 'complete'>,
stdout='result = 42\n', stderr='', error='', value=None)
```

**40 K stdout event, max_chars=300 (real output — real limit is 20 K):**
```
<truncated-output>
Output too large (437 chars). Showing first 150 and last 150 chars.

PythonOutput(tool_call_id='tc-1', execution_status=<ResultStatus.COMPLETE: 'complete'>, stdout='word word word word word word word word word word word

... 137 chars not shown ...

ord word word word word word word word word word word word word word word word word word word word word word '+39700, stderr='', error='', value=None)
</truncated-output>
```
