# One snapshot model for values, output streams and spilled files

This page describes `nooa.snapshot`: one model for describing a large value or a large output stream, with what was kept, how much was left out and where the rest is. It builds on Severin Klingler's bounded trace JSON builder (PR #342). The machinery is in place; consumers move to it in later steps (see [Plan](#plan)).

## Why

NOOA cuts large output in about seven places today, each with its own limits and its own notation:

| Where | Limit | Unit | Notation the model sees | Full output kept? |
|---|---|---|---|---|
| Atom `Shell` result (`work_shell._Capture`) | whole up to 10 kB, else 2 kB head + 8 kB tail | bytes | `[... N bytes omitted; full output: PATH]` | yes, file up to 64 MiB |
| Shell progress (`ShellProgress`, merged batches) | `Throttle.max_bytes`, ¼ head | bytes | `[... progress omitted ...]` | no (final result has it) |
| Cell stdout / stderr / error (`TruncatingStringIO`) | 50,000 / 2,000 / 10,000 chars, 50/50 | chars | `<truncated-output> … chars not shown … </truncated-output>` | only if `capture.file_backed` (off by default) |
| Job output tails (`jobs.py`) | tail only, per-line cap | chars | `[... N chars omitted]`, `[... N chars of this line omitted]` | no |
| Core bash tool (`_bash_session`) | 30,000 chars | chars | `TruncatingStringIO` notice | no |
| ACP activity cards (`activity.py`) | own limits, `… +N lines` for diffs | chars/lines | card text, "output omitted" flag | no |
| `CommandResult` display (`spec(max_string=2000)`) | 2,000 chars | chars | `pformat` string preview | n/a |
| Trace `input.value` / `output.value` (#342) | 50,000 chars / 2,000 nodes / depth 16 | chars | plain JSON; `nooa.{input,output}.preview.*` attributes | no |

All of them do the same thing: keep a bounded head and tail of something large, say how much was left out, and sometimes say where the rest is. One small model can express all of it, so every consumer renders the same facts the same way.

## The model

A snapshot is a small, detached tree. It holds no live objects. Every node says whether it is complete and, if not, why.

```python
Snapshot = Scalar | Text | Sequence | Mapping | Object | Opaque | Unavailable | Ref

@dataclass(frozen=True)
class Text:
    head: str                 # kept start, decoded
    tail: str                 # kept end, decoded ("" when complete)
    length: int               # total size of the original, in `unit`
    unit: Literal["chars", "bytes"]
    complete: bool
    reason: str | None        # "max_length", ... (None when complete)
    full: Spill | None = None # where the whole content was kept, if anywhere
    replaced: int = 0         # bytes that failed to decode (unit == "bytes")

@dataclass(frozen=True)
class Spill:
    path: str
    kept: int                 # bytes written to the file
    total: int                # bytes produced
    # kept < total: the file is itself cut ("file_limit")
```

`Sequence`, `Mapping` and `Object` carry `length` (the original count), their kept children, `complete` and `reason`. `Object` also carries an optional `kind`, a rendering key (see *Extending for third-party types*). `Opaque(module, type_name)` stands for a type NOOA will not inspect. `Unavailable(reason)` stands for a read that failed or was refused. `Ref(id)` stands for a repeated or cyclic reference. Reason names: `max_length`, `max_items`, `max_depth`, `max_nodes`, `cycle`, `opaque`, `file_limit`, `not_kept`.

The proposal's `Text` gains three fields: `unit`, `full` and `replaced`. They are the whole extension for spilling.

## Two ways to produce a snapshot

### 1. `capture(value, limits)`: finished Python values

As in the proposal and in #342:
- One shared budget (`Limits`): characters, nodes, depth, items per container, string head and tail.
- Exact built-in types are handled natively. Pydantic and dataclass declared fields are read from raw storage. `spec()` hidden fields are refused before they are read. Anything else is `Opaque`.
- `@capture.register(T)` extends it, using nearest-MRO dispatch.
- No `repr()`, `str()`, properties, descriptors or custom iteration on untrusted values.
- #342's builder (`_trace_json.py`) is the starting implementation. It already does the raw Pydantic view, visibility metadata, cycle tracking and accounting. It moves to the shared package and produces snapshot nodes instead of JSON directly.

### 2. `TextStream`: output that arrives over time

Shell output and printed cell output arrive as a stream, and the whole stream must never be held in memory. Calling `capture()` on a 64 MiB string would defeat the purpose. So streams build the same `Text` node incrementally:

```python
class TextStream:
    def __init__(self, *, unit: Literal["chars", "bytes"], head: int, tail: int,
                 spill: SpillTarget | None = None): ...
    def write(self, data: str | bytes) -> None   # O(len(data)); memory ≤ head + tail
    def snapshot(self) -> Text                    # any time, also while running
    def close(self) -> Text                       # final snapshot; closes the file

@dataclass(frozen=True)
class SpillTarget:
    path: str | None = None    # None: a new temp file
    max_bytes: int = 64 * 1024 * 1024
    always: bool = False       # False: the file is deleted on close if nothing was cut
```

- **Head, rolling tail, counter, file.** This is what `work_shell._Capture` and `TruncatingStringIO` do today, as one class.
- **Byte streams** cut on UTF-8 code-point boundaries (`work_progress._safe_end` / `_safe_start` move here) and count replacement characters in `replaced`.
- **Spilling** writes every byte up to `max_bytes` while the stream runs. The resulting `Text.full` says where the full output is and whether the file holds all of it.
- **Progress** is a delta view: `TextStream.since(mark) -> (Text, mark)` returns what arrived after a mark, bounded the same way. `merge(a: Text, b: Text, head, tail) -> Text` replaces `ShellProgress._merge`. Merged unread batches stay bounded and keep a correct `length`.
- **Per-line caps** (`LineSplitter`, used for following logs) stay a separate helper. A capped line becomes a `Text` whose `reason` is `max_length`.

## One set of renderers

Rendering is the last step, and every consumer picks a renderer and limits:

| Renderer | Used by | Cut text looks like |
|---|---|---|
| `render_output(text)` | Shell results, cell stdout/stderr, job tails, core bash tool, progress | `HEAD\n[... 1,234,567 bytes omitted; full output: /path]\nTAIL` |
| `render_python(snapshot)` | CodeAct prefill, `pformat`, `pprint`, `CommandResult` display | `str(len=10_000_000, [:4]='xxxx', [-4:]='xxxx')` (today's prefill notation) |
| `render_json(snapshot)` | trace `input.value` / `output.value` | plain JSON, as in #342; nothing embedded in user data |
| `preview_attributes(direction, snapshot)` | trace spans | `nooa.{dir}.preview.version/incomplete/paths` (#342), plus `.reasons` and `.full_output` |
| `render_card(text)` | ACP activity cards | the output notation, then the card's "output omitted" flag from `complete` |

The output notation is one line, with three variants built from `Text` alone:
- `[... N bytes omitted; full output: PATH]`
- `[... N bytes omitted; PATH holds the first K of N bytes]` (file limit)
- `[... N bytes omitted; full output not kept]`

It is today's `Shell` wording. It is the newest notation and the one models already see most.

## Extending for third-party types (numpy, pandas, …)

Two hooks, one per step. Neither makes NOOA import or depend on the library.

### Extension contract

When each part is built, and what bounds it:

| Part | Built | Budget | Limits that apply |
|---|---|---|---|
| `out.capture(v)`, `out.sequence(items)`, `out.mapping(items)` | deferred: when the part takes its slot (a field of `out.object`, an item or value of `out.sequence` / `out.mapping`, or the function's return value) | the slot's share of the capture budget, one level deeper than its container (returned: the value's own slot) | capture `Limits`: `max_chars` (canonical JSON characters), shared `max_nodes`, `max_depth`, `max_items`, `max_string` |
| `out.object(fields)` | at once, when called | the value's own slot (the `out`'s budget and depth), not the slot it is later placed in; as a field of another `out.object` it is shrunk to fit that slot | the same capture `Limits` |
| `out.opaque()` | at once | the value's own slot | capture `max_chars` |
| `render_python` / `render_output` hook | when the snapshot is rendered | **none**: these renderers have no length limit | none: the hook formats nodes that capture already bounded. Its return value is not checked for length; a hook is trusted code, and its output is as long as the string it builds |
| `render_json` hook | when the snapshot is rendered | the node's `r.max_chars` | its text must parse as JSON and its canonical encoding must fit `r.max_chars`, otherwise the node renders generically |

So capture limits bound what a snapshot holds, and only `render_json` bounds what a renderer produces. Children a text hook renders through `r` get the renderer's notation, cut markers and hooks, nothing more.

**Reservation, and where it stops.** With a `dict`, `list` or `tuple` of fields, `out.object` builds them in order and each leaves room for the minimal fallback of every later field, so a greedy field gets what is left. That holds only while the slot fits the wrapper, the current field's own minimum and those fallbacks: a field is never held below its own minimum, so in a smaller slot later fields are still cut. Fields from any other iterable are pulled one at a time, and nothing can be reserved for fields not yet seen. The one detailed explanation is under [The implementation](#the-implementation-nooasnapshot) (`out.object`).

### Capture: lazy registration by name

```python
@capture.register_lazy("numpy", "ndarray")
def capture_ndarray(array, out):
    return out.object({
        "shape": out.capture(array.shape),
        "dtype": out.capture(array.dtype.name),
        "sample": out.sequence(array.flat),   # gets what shape and dtype leave
    }, type_name="ndarray", kind="numpy.ndarray")
```

- `register_lazy(module, qualname)` records the name only. It never imports the module. `capture.register_lazy(module, qualname, function)` is the non-decorator form, and `capture.unregister_lazy(module, qualname)` removes it.
- At capture time, the dispatcher walks the value's raw MRO (`type.__dict__["__mro__"]`). For each class it reads `__module__` and `__qualname__` through `type`'s own descriptors (the class's native storage), without metaclass hooks. The nearest class that matches an eager registration (`capture.register(T)`) or a lazy one (`register_lazy`) wins. At the same class, the eager one wins. The match is cached per class but checked against the class's current raw name, so renaming a class (assigning `__module__` or `__qualname__`) takes effect at the next capture.
- Resolved lazy matches are cached by class identity (keyed by `id`, checked through a weak reference, dropped when the class dies, cleared when a registration changes), never by hashing the class, so a custom metaclass `__hash__`/`__eq__` can't run.
- A value whose class is not registered stays `Opaque`, exactly as before.
- The registered function is trusted extension code, as in the proposal. It builds children only through `out`, so the shared budget holds. An exception becomes `Unavailable`.

### Rendering: a `kind` on objects and per-kind renderers

Renderers work on detached snapshots, so a render hook can't receive the live object. The key it dispatches on must be stored in the snapshot itself:

```python
@dataclass(frozen=True)
class Object:
    type_name: str
    fields: tuple[tuple[str, Snapshot], ...]   # in order; node.field(name) looks one up
    complete: bool
    reason: str | None
    module: str = ""
    kind: str | None = None       # rendering key set by the capture function, e.g. "numpy.ndarray"
```

Each renderer has its own registry, keyed by `kind`:

```python
@render_python.register("numpy.ndarray")
def render_ndarray(node: Object, r: PythonRenderer) -> str:
    shape = r.render(node.field("shape"))
    sample = r.render(node.field("sample"))
    return f"array({sample}, shape={shape}, dtype={r.render(node.field('dtype'))})"
```

- A hook receives the node and the renderer `r`. It renders children only through `r`, so they keep the renderer's notation, cut markers and their own hooks. Nothing bounds the string a `render_python` / `render_output` hook returns (see [Extension contract](#extension-contract)): it formats nodes that capture already bounded, and it is trusted code.
- No hook for a `kind`, or a hook that raises (or returns something other than a `str`): generic rendering (`ndarray(shape=…, dtype=…, sample=…)`). Rendering never fails. A hook that renders its own node through `r` gets the generic rendering for it.
- `render_python` and `render_output` accept hooks (`render_python.register(kind)` as a decorator, `register(kind, hook)`, `unregister(kind)`). `render_output` also takes any node now: a `Text` renders in the output notation exactly as before; other nodes use the Python notation, with nested `Text` in the output notation.
- `render_json` stays generic by default, so trace files remain plain and tool-readable. It accepts hooks too (`hook(node, r: JsonRenderer) -> str`, JSON text), but its output is still checked: the hook's text must parse as JSON (no `NaN`/`Infinity`) and its canonical encoding must fit the node's budget `r.max_chars`; otherwise the node renders generically, so the result is still valid and within `max_chars`. `r.render(child, max_chars=…, key=…)` returns a child's JSON fitted to the budget; `key` names it in the paths that `preview_attributes` reports, which uses the same hooks.
- `kind` is a plain string, so render hooks need no lazy mechanism.

### First users

Open PR **#87** ("shape-aware pformat previews for numpy and pandas", pierreolivierbonin) already designs previews for these types. Once PR 1 lands, its previews can be re-expressed as `register_lazy` capture functions plus `render_python` hooks for `numpy.ndarray`, `pandas.DataFrame` and `pandas.Series`.

## The implementation (`nooa.snapshot`)

Public API, all importable from `nooa.snapshot`:

| Name | What it is |
|---|---|
| `Scalar`, `Text`, `Sequence`, `Mapping`, `Object`, `Opaque`, `Unavailable`, `Ref`, `Spill` | the frozen snapshot nodes; `Snapshot` is their union, `Reason` / `REASONS` the reason names, `Key` a mapping key |
| `capture(value, limits=DEFAULT_LIMITS)`, `Limits`, `DEFAULT_LIMITS` | a snapshot of a finished value; `capture.register(T)` / `capture.unregister(T)` and `capture.register_lazy(module, qualname)` / `capture.unregister_lazy(...)` extend it |
| `Out` | what a registered function gets as `(value, out: Out)`: `out.capture/sequence/mapping/object/opaque` |
| `capture_fields(**fields, limits=...)` | #342's `trace_fields`: a framework wrapper whose every root field survives with its type |
| `TextStream`, `SpillTarget`, `DEFAULT_SPILL_BYTES`, `merge` | output that arrives over time; `merge` joins unread deltas |
| `render_output`, `render_python`, `render_json`, `preview_attributes` | the renderers; the first three take per-kind hooks (`.register(kind)`) |
| `PythonRenderer`, `OutputRenderer`, `JsonRenderer` | the `r` a render hook gets: `r.render(child)` |

Details that the sections above leave open:

- **`Limits`** has `max_chars`, `max_nodes`, `max_depth` (#342's defaults: 50,000 / 2,000 / 16), `max_items` and `max_string` (`None`: only the shared budget) and `tail_share`, the part of a cut string or container kept from its end (0: a prefix, as in trace previews; 0.5: agentdoc's split). `max_items` caps the items of every container, the fields of an `Object` too: an object keeps its first `max_items` fields (not a head and tail) and is marked `max_items` when a later field was left out. `max_string` is an upper bound: a byte value keeps only whole code points within it, so a code point wider than what is left is omitted.
- **`max_chars` counts canonical JSON**, exactly as #342 does. So `render_json(capture(v, limits), max_chars=limits.max_chars)` gives #342's text and incomplete paths (checked against the original builder on random shapes). `render_json` also walks the snapshot with the same accounting, so it stays valid and within any smaller `max_chars`, and `preview_attributes(..., max_chars=...)` reports the cuts that rendering made.
- **Nodes beyond the design's fields:** `Text.omitted` (the units left out between head and tail: for bytes it can't be derived from the decoded strings), `Sequence.tail` / `Mapping.tail` (items kept from the end), `Sequence.type_name` / `Mapping.type_name`, `Object.module`, `Unavailable.type_name` (picks #342's typed fallback), `Ref.kind`, and `Mapping.wrapper` (marks a `capture_fields` root). `Ref.id` is the JSON Pointer of the enclosing container that the cycle points back to. Repeated references that are not cycles are captured again each time, as in #342.
- **`out.capture`, `out.sequence` and `out.mapping` return deferred parts**: opaque values whose type is private; pass one on unchanged (as below, or return it) and use nothing else of it. Nothing is built and no item is pulled until the part takes its slot, as a field value of `out.object(...)`, an item of `out.sequence(...)`, a value of `out.mapping(...)`, or as the function's return value. A part is built in its slot's context: as a field, item or value it is one level deeper than the container, under that field name, index or key (so `max_depth` holds for recursive registered types and cycle `Ref` ids are field-name pointer paths); returned, it takes the value's own depth and path. A part used twice is built once; the second use is shrunk to fit its slot.
- **`out.object(fields, *, type_name=None, kind=None)`** takes `(name, value)` pairs or a plain `dict`, and builds at once. Fields are built in order, and each field leaves room (characters and node work) for the minimal fallback of every later field, as #342's `trace_fields` does for root fields. So in the ndarray example `shape` and `dtype` stay complete and `sample` gets what is left: present, possibly empty, and incomplete with the reason that stopped it (`max_length` or `max_nodes`). A greedy first field leaves its later siblings at least their fallback when the slot fits them: a field is never held below its own minimal value (it gets `max(min(room, its minimum), room - reserved)` characters), and node work is held back only down to one node for the current field, so in a slot too small for all the minimums the later fields are cut. `out.object` itself is built at once in the value's own slot (the `out`'s budget and depth), not in the slot it later takes. The reservation needs the field list up front: for a `dict`, `list` or `tuple` (only for the first `max_items` fields); other iterables are pulled lazily, without it. A value may also be a node the same `out` already built, such as a nested `out.object(...)`: it is reused, not captured again, and shrunk to fit its slot when it is too large (trailing items or fields dropped, the tail of a sequence or mapping first, text cut, marked incomplete), never replaced by `Unavailable`. Fields that fit are kept whole. A non-`str` `kind` or `type_name` (of `out.object`, `out.sequence` or `out.mapping`), or a `length` that is not a non-negative `int` or `None`, is an error in the capture function, so the value becomes `Unavailable("error")`. A `str` subclass is copied to an exact `str`: a snapshot never holds a live object a renderer could run.
- **Reason `error`** is added to the list. A registered function that raises, or that returns a node its `out` did not build, becomes `Unavailable("error")`.
- **Dataclasses:** the declared fields come from the nearest `__dataclass_fields__` in raw class namespaces. Values come from the native `__dict__` getset descriptor or the slot member descriptors, never through properties. `_private` fields, `repr=False` fields, `ClassVar`/`InitVar` fields and hidden fields (`Annotated[..., hidden]`, `spec(hidden=True)`, `spec()` overrides on the class or instance) are left out. String annotations (PEP 563) are never evaluated: they are parsed with `ast`, and their names (`A`, `typing.Annotated`, `agentdoc.hidden`, ...) are looked up in the raw `__dict__` of the already-loaded module that declared the field (then builtins), with no attribute hooks. Resolved objects are compared by identity with `Annotated` and the `hidden`/`spec` markers, so aliases hide like the literal form. Only the outermost `Annotated` counts, as at runtime. When a field's metadata can't be read safely (an unknown name inside `Annotated`, a `spec(...)` call without a literal `hidden=`), that field fails closed: it is left out and the object is marked incomplete (reason `opaque`). Unknown names outside `Annotated` (`list[Foo]`, forward references) don't matter. A string that doesn't parse (or is over 2000 chars) falls back to the textual rule: one that mentions `hidden` or `spec` inside `Annotated` fails closed the same way.
- **Mapping keys** may be exact `str`, `int`, `float`, `bool` or `None`. `render_json` converts them the way `json.dumps` does, and stops the mapping at a key that would collide.
- **`bytes` values** become a `Scalar` when they fit, otherwise a byte `Text` cut on code points. `render_json` shows them decoded.
- **`TextStream.since(mark)`** returns what memory still holds of the delta: the part still in the head and the part in the rolling tail. A byte stream's mark stops before an incomplete UTF-8 sequence. `Text.replaced` counts the replacement characters that decoding introduced, not counting genuine U+FFFD in the input, and for a delta only those in the window: a byte stream decodes the window itself; a chars stream remembers its cumulative count at the last 16 marks `since()` returned and reports the difference. For a mark it doesn't remember it reports the count since the nearest remembered later mark (0 if none), a lower bound that never over-reports; `since(0)` is exact. So `merge` of consecutive deltas sums without double-counting. `merge` counts original units: a byte that decoded to U+FFFD counts as one byte, not the three its re-encoding takes (exactly for whole parts; cutting inside a part, a U+FFFD counts the width its part implies, or three bytes when that is ambiguous, so the limits hold and `omitted` is then a lower bound). A merged bytes `Text` counts in `replaced` only the replacements it still holds.
- **`TextStream` threads:** one writer. `snapshot()` and `since()` only read the writer's buffers (they never rebind them) and flush the spill file, but are not synchronised with `write`: call them from the writer's thread, or serialise them with writes yourself. PR 1 has no lock.
- **Spill file on a running stream:** `snapshot()` and `since()` flush the spill file first, so `Spill.kept` bytes are on disk; a flush `OSError` is handled like a write failure (reason `not_kept`, file removed, memory kept). A chars stream encodes for the file only while a spill file is active: UTF-8 with surrogate escapes as raw bytes, else (a lone surrogate) `backslashreplace`; memory always keeps the chunk as written.
- **Spill reasons:** a cut `Text` has reason `max_length`. It is `file_limit` when the spill file is itself cut, and `not_kept` when a spill was requested but the file could not be created or written. In that case the stream keeps memory only and removes the partial file.
- **`render_python`** copies agentdoc's markers: `str(len=N, [:h]='…', [-t:]='…')`, `list(len=N, [:h]=[…], [-t:]=[…])`, `dict(len=N, items={…, ..., …})`, `[list: N items]` for depth cuts, `<cycle>` and `Type(a=1, ...)`. It renders on one line: it does not wrap at 120 columns the way `pformat` does. Values NOOA does not inspect render as `<opaque: Type>`, and failed reads as `<unavailable: reason>`.
- **Not in this step:** `render_card` (ACP, with PR 2a) and moving `LineSplitter`.

## What changes for each consumer (PR 2)

| Consumer | Today | After |
|---|---|---|
| `work_shell._Capture` | own head/tail/file | `TextStream(unit="bytes", head=2_000, tail=8_000, spill=SpillTarget(path=…, max_bytes=max_output))`; `CommandResult.stdout = render_output(…)`, `stdout_path = text.full.path` |
| `ShellProgress` | own `_keep` / `_merge` | `TextStream.since` + `merge` |
| Cell stdout/stderr/error | `TruncatingStringIO` / `FileBackedTruncatingStringIO`; spill off by default | `TextStream(unit="chars", …)`; **spill on for Atom** (decided), in the session's scratch directory, same lifetime rules as `Shell` (decided); limits unchanged (50,000 / 2,000 / 10,000 chars, decided) |
| `jobs.py` tails | `TruncatingStringIO(tail only)` | `TextStream(head=0, …)` |
| Core `_bash_session` | 30,000-char `TruncatingStringIO` | `TextStream` + `render_output` |
| ACP `activity.py` | own limits, own flags | `render_card` |
| Trace hooks | #342's `trace_json` / `trace_fields` | `capture` + `render_json` + `preview_attributes` |
| Prefill / `pformat` | agentdoc head/tail | `capture` + `render_python` |

Configuration stays in `TruncationConfig`. `capture.max_stdout` etc. become `TextStream` limits, and `file_backed` becomes the spill switch, defaulting to on in Atom.

## Plan

1. **PR 1, the machinery, into `main`.** It adds `nooa.snapshot` (model, `capture` + registry, `TextStream`, `merge`, renderers, trace attributes) with tests, and changes no consumers. #342's builder is moved in, with credit to Severin. Tests cover: budget accounting, never calling user code on unknown types, hidden fields, cycles, UTF-8 cut points, spill file limits, `since`/`merge` lengths, and renderer output for every variant.
2. **PR 2a, consumers the model doesn't read, into `main`:** traces (supersedes #342's consumer change, including its explorer notice) and ACP cards. Low risk; checked with trace and ACP tests.
3. **PR 2b, consumers the model reads:** `Shell`, progress, cell output, jobs, the core bash tool and prefill. This changes what the model sees after every command, so it lands after an A/B on one SWE run and one TB run against the current notation.

## Out of scope

- `describe()`, the API-description half of the proposal. It doesn't touch output or spilling, and it overlaps Paul's #213.
- Content hashes for spill files: hashing would cost a full read. It can be added to `Spill` later.
- Media (`show()` attachments), which have their own count cap.

## Decisions (Paul, 10-31)

1. **Atom spills cell stdout by default:** yes, as `Shell` does.
2. **The per-cell stdout limit stays at 50,000 characters.** It does not shrink to `Shell`'s 10 kB, even though the full output is on disk.
3. **Spill files follow `Shell`'s lifetime rules:** kept while the job is listed, or until the `Shell` or session closes.
