# One snapshot model for values, output streams and spilled files

This page describes `nooa.snapshot`: one model for describing a large value or a large output stream, with what was kept, how much was left out and where the rest is. It builds on Severin Klingler's bounded trace JSON builder (PR #342). The machinery is in place; consumers move to it in later steps (see [Plan](#plan)).

## Principles

Three rules decide every design question in this package. When code and these rules disagree, the code is wrong.

1. **Never run code on values; read class metadata the way Python does.** Capture never calls a value's `repr`, `str`, properties, descriptors or iteration: values are captured over and over, and their code can be slow, have side effects or reveal hidden fields. Class declarations are different: annotations and field declarations are code the class's author wrote, which Python already runs when the class is defined. Capture reads them once per class with `typing.get_type_hints(cls, include_extras=True)`, cached by class identity, the same way `agentdoc` does. If they can't be read, the affected fields fail closed: they are left out and the object is marked incomplete.
2. **Store what you cut in its own units.** A cut string keeps characters; cut bytes keep bytes. Nothing is decoded until it is rendered, so counts stay exact and no unit is reconstructed later.
3. **One extension point per job, and nothing without a user.** Shapes are the only way to describe a type to capture, and `render_python` hooks the only way to change how a node prints. A field, reason or hook that no consumer reads is not added.

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
Snapshot = Scalar | Text | Sequence | Mapping | Object | Unavailable | Ref

@dataclass(frozen=True)
class Text:
    head: str | bytes         # kept start: str for chars, the original bytes for bytes
    tail: str | bytes         # kept end ("" / b"" when complete)
    length: int               # total size of the original, in `unit`
    unit: Literal["chars", "bytes"]
    complete: bool
    reason: str | None        # "max_length", ... (None when complete)
    full: Spill | None = None # where the whole content was kept, if anywhere
    omitted: int = 0          # units left out between head and tail

@dataclass(frozen=True)
class Spill:
    path: str
    kept: int                 # bytes written to the file
    total: int                # bytes produced
    # cut (kept < total): the file is itself cut; `Text.full` is None when nothing was kept
```

`Sequence`, `Mapping` and `Object` carry `length` (the original count), their kept children, `complete` and `reason`. `Object` also carries an optional `kind`, a rendering key (see *Extending for third-party types*). `Unavailable(reason, type_name="", module=None)` stands for a value with no content kept: with reason `opaque`, a value of a type NOOA will not inspect (`module` is its type's module; it renders as `<opaque: Type>`), otherwise a read that failed or was refused. `Ref(id)` stands for a repeated or cyclic reference.

Reason names (`Reason`; `REASONS = typing.get_args(Reason)`), each saying why a node is incomplete:

| Reason | Meaning |
|---|---|
| `max_length`, `max_items`, `max_depth`, `max_nodes` | a limit cut it |
| `cycle` | a reference to an enclosing container (`Ref`) |
| `opaque` | a value of a type NOOA does not inspect (`Unavailable("opaque", ...)`), and only that |
| `unreadable` | an object's fields, or a field's hidden-ness (its `spec` marker or annotation), could not be read safely, so they were left out (fail closed, principle 1) |
| `key` | a mapping stopped at a key with no JSON form (not an exact `str`/`int`/`float`/`bool`/`None`, an `int` over 128 bits, a non-finite `float`) or one that collides with an earlier key |
| `error` | a shape accessor raised, or returned something its shape can't use |

Whether the whole content was kept in a file is not a reason: it is `Text.full` (`None`: no file; `full.cut`: the file holds only the first `full.kept` bytes).

Relative to the proposal, `Text` gains `unit` and `full` for spilling, and `omitted` (see *The implementation*) for exact accounting. A bytes `Text` keeps the original bytes (principle 2): `len(head)` and `len(tail)` are its kept units, so `length == len(head) + len(tail) + omitted` holds for every `Text` that `capture`, `TextStream` and `merge` produce. Renderers decode only when they print: `render_output` and `render_json` decode as UTF-8 with replacement, `render_python` shows `bytes` literals of the kept bytes.

## Two ways to produce a snapshot

### 1. `capture(value, limits)`: finished Python values

As in the proposal and in #342:
- One shared budget (`Limits`): characters, nodes, depth, items per container, string head and tail.
- Exact built-in types are handled natively. Pydantic and dataclass declared fields are read from raw storage. `spec()` hidden fields are refused before they are read. Anything else is `Unavailable("opaque")`.
- `capture.register(T, shape)` extends it, using nearest-MRO dispatch: a declarative shape (`array_like`, `ndarray_like`, `mapping_like`, `object_like`, `text_like`) maps out the type's structure, and capture builds the nodes.
- No `repr()`, `str()`, properties, descriptors or custom iteration on untrusted values.
- #342's builder (`_trace_json.py`) is the starting implementation. It already does the raw Pydantic view, visibility metadata, cycle tracking and accounting. It moves to the shared package and produces snapshot nodes instead of JSON directly.

### 2. `TextStream`: output that arrives over time

Shell output and printed cell output arrive as a stream, and the whole stream must never be held in memory. Calling `capture()` on a 64 MiB string would defeat the purpose. So streams build the same `Text` node incrementally:

```python
class TextStream:
    def __init__(self, *, unit: Literal["chars", "bytes"], head: int, tail: int,
                 spill: SpillTarget | None = None): ...
    def write(self, data: str | bytes) -> None   # O(len(data)); memory ≤ head + tail
                                                  # chars: str only (bytes raise TypeError)
    def snapshot(self) -> Text                    # any time, also while running
    def close(self) -> Text                       # final snapshot; closes the file

@dataclass(frozen=True)
class SpillTarget:
    path: str | None = None    # None: a new temp file
    max_bytes: int = 64 * 1024 * 1024
    always: bool = False       # False: the file is deleted on close if nothing was cut
```

- **Head, rolling tail, counter, file.** This is what `work_shell._Capture` and `TruncatingStringIO` do today, as one class.
- **Byte streams** keep raw bytes, cut on UTF-8 code-point boundaries (`work_progress._safe_end` / `_safe_start` move here). A chars stream takes `str` only and raises `TypeError` for bytes: raw output goes to a bytes stream, which keeps it as written.
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
| `render_trace(snapshot, direction=…, max_chars=…)` | trace hooks | both of the above, `(text, attributes)`, from one JSON walk |
| `render_card(text)` | ACP activity cards | the output notation, then the card's "output omitted" flag from `complete` |

The output notation is one line, with three variants built from `Text` alone:
- `[... N bytes omitted; full output: PATH]`
- `[... N bytes omitted; PATH holds the first K of N bytes]` (file limit)
- `[... N bytes omitted; full output not kept]`

It is today's `Shell` wording. It is the newest notation and the one models already see most.

## Extending for third-party types (numpy, pandas, …)

Two hooks, one per step. Neither makes NOOA import or depend on the library: the shape is registered by whoever sets up the agent, where the library is already imported.

### Capture: shapes, registered by type

An extension doesn't print anything and doesn't build nodes. It maps out the structure of its type, a **shape**: "this is array-like, here is how to get its size, here is how to get its data, here are a couple of other properties". Capture then builds the snapshot itself, with the same builders and size rules as the built-in types, so the notation, the cuts and the shared budget are the ones a `list`, `dict` or `str` gets.

```python
import numpy
from nooa.snapshot import capture, ndarray_like

capture.register(numpy.ndarray, ndarray_like(
    lambda a: a.shape,
    lambda a, index: a[index].item(),          # called only for the shown elements
    fields={"dtype": lambda a: a.dtype.name},
    kind="numpy.ndarray",
))
# ndarray(dtype='float64', shape=(1000, 1000),
#         data=list(len=1000, [:2]=[list(len=1000, [:2]=[0.0, 1.0], [-2:]=[998.0, 999.0]), …], [-2:]=[…]))
```

A plain class:

```python
capture.register(Point, object_like({"x": lambda p: p.x, "y": lambda p: p.y}))
# Point(x=1, y=2)
```

A huge text-like object (a rope, a memory-mapped log), never copied whole:

```python
capture.register(Rope, text_like(lambda r: r.length, lambda r, start, stop: r.read(start, stop)))
# str(len=…, [:h]='…', [-t:]='…'): slice is called for that head and tail only
```

`capture.register(T, shape)` takes the class itself and one of five shapes, all exported from `nooa.snapshot`:

| Shape | Accessors | Node |
|---|---|---|
| `array_like(length, items, *, fields=None, kind=None)` | `length(v) -> int`, called once; `items(v) -> iterable`, pulled lazily, only as many as are kept (plus at most one) | without fields a `Sequence` (`type_name`: the class name); with fields an `Object` whose fields are followed by `items`, that `Sequence` (`type_name` `list`) |
| `ndarray_like(shape, element, *, fields=None, labels=None, kind=None)` | `shape(v) -> tuple[int, ...]`; `element(v, index_tuple)`, called only for the shown elements; `labels`: axis -> `accessor(v)` returning that axis's labels, read only at the shown indices (`labels[i]`) | an `Object`: its fields, then `shape` (the tuple), `labels` (a `Mapping` axis -> `Sequence` of the shown labels, when given) and `data`: nested `Sequence` nodes, one level per axis, each with the axis's length and its kept head and `tail`; a 0-d array's `data` is its one element |
| `mapping_like(length, items, *, fields=None, kind=None)` | `length(v)`; `items(v)` yields `(key, value)` pairs lazily | a `Mapping`, or an `Object` whose fields are followed by `items`, that `Mapping` (`type_name` `dict`) |
| `object_like(fields, *, kind=None)` | ordered `dict` name -> `accessor(v)` | an `Object` |
| `text_like(length, slice, *, unit="chars", kind=None)` | `length(v)`; `slice(v, start, stop) -> str` (`"chars"`) or `bytes` (`"bytes"`) | a `Text` with the right `length`, `omitted` and `unit`, cut exactly as a `str` or `bytes` of that content would be |

The rules that hold for every shape:

- **Accessors run only when their result is needed.** `fields` is an ordered `dict` name -> `accessor(v)`: each accessor is called once, and only when its field is kept (a field cut by the budget or by `max_items` never runs). Fields keep back room for the later fields, as for the root fields of `capture_fields`: each leaves the minimal value of every later field (and some node work), so a greedy first field gets what is left. The structure fields of a shape (`items`, `shape`, `labels`, `data`) come after the accessor fields, get what those leave, and are never cut by `max_items` (the accessor fields are: an object keeps its first `max_items` of them).
- **What accessors return is captured normally,** recursively and under the same budget: a returned list is a list, a returned registered type uses its own shape, an unknown type is `Unavailable("opaque")`, a returned `v` is a cycle (`Ref`).
- **`text_like` never reads the whole of a text it cuts.** It calls `slice` only for the head and the tail it keeps, at most twice (for bytes, the head read takes one byte more, to see whether the head ends on a code point boundary; when that byte is the last one, it is read on its own). A text short enough to be kept whole (at most `max_string` and what the budget leaves, in units) is read in one call.
- **`ndarray_like` element selection** (numpy's summarised print): along every axis, the first and the last entries, at most `K` per axis (all of a shorter axis), the start taking the odd one: `K = 4` shows indices `0, 1, n-2, n-1`. `K` is `max_items` when it is set, and never more than the budget affords: the largest `K` whose shown elements and labels (6 characters each: a short number and its separator) and nested brackets fit three quarters of the value's characters, and whose elements and levels fit three quarters of the node work left (the rest is for the fields, the shape and the keys). The split ignores `tail_share`: an array always shows both ends. Elements larger than assumed are still cut by the usual budget, from the end. `element` is called only for the elements in the snapshot.
- **Errors.** An exception in `length` or `shape` (or a result that is not a non-negative `int`, or a tuple of them) makes the whole value `Unavailable("error")`. An exception in a field accessor, `element` or one label makes just that part `Unavailable("error")`. An exception from `items` (calling it, or a pair that is not a 2-tuple) makes the `items` part `Unavailable("error")` when the shape has fields, the whole value when it has none; raised while pulling, it ends the items there, the kept ones stay and the `Sequence` / `Mapping` is incomplete with reason `error`. A `text_like` has no sensible part: an exception in `slice`, or a slice of the wrong type or size, makes the whole value `Unavailable("error")`.
- **`kind`** is stored on the `Object` (`Object.kind`) and feeds the per-kind render hooks below; `Sequence`, `Mapping` and `Text` have no `kind`, so a shape without fields renders with the generic notation. Names and kinds must be `str` (a `str` subclass is copied); constructors check their arguments and raise `TypeError`.

Registration:

- `capture.register(T, shape)` takes the class itself, so registering imports the type. Support for a library is therefore opt-in: whoever sets up the agent, and already imports numpy or pandas, registers it; NOOA itself never imports them. `capture.unregister(T)` removes it, and registering `T` again replaces it. Anything but a class, or anything but one of the five shapes (a function, the earlier `register(T)(fn)` form), raises `TypeError`.
- The registry is process-wide: one registry for the whole process, with no per-agent scope.
- At capture time, the dispatcher walks the value's raw MRO (`type.__dict__["__mro__"]`, read through `type`'s own descriptor) and compares each class with the registered classes by identity (`is`); the nearest registered class wins, also against the built-in Pydantic and dataclass views. A class is never hashed or compared with `==`, so a custom metaclass `__hash__`/`__eq__` can't run, and a lookup runs no user code. Matching is by identity, not name, so a class renamed by assigning `__module__` or `__qualname__` still dispatches, and another class with the same name does not. Only registered classes are retained.
- A value whose class is not registered stays `Unavailable("opaque")`, exactly as before. Exact built-in types (`str`, `bytes`, `int`, `float`, `bool`, `None`, `list`, `tuple`, `dict`) are always native, checked first.
- Accessors are trusted extension code, as in the proposal; they run only when capture asks for their result, never during dispatch.

**Internal shapes.** Pydantic models and dataclasses are internal `object_like` shapes too: their field list and accessors are the privileged readers (raw instance storage, the hidden-field rules, string annotations resolved once per class by `typing.get_type_hints`; see *Dataclasses* below), and they go through the same object builder as a registered `object_like`. The exact built-in types keep their fast paths, and those share their building code with the shapes: `array_like` items go through the list builder, `mapping_like` through the dict builder, `text_like` through the `str` / `bytes` cutting, and every `Object` through one field builder.

### Extension contract

When each part is built, and what bounds it:

| Part | Built | Budget | Limits that apply |
|---|---|---|---|
| shape accessors (`length`, `items`, `element`, `slice`, field accessors) | during capture, only when their result is kept (see above) | the slot of the part they fill: its share of the capture budget, depth and path | capture `Limits`: `max_chars` (canonical JSON characters), shared `max_nodes`, `max_depth`, `max_items`, `max_string` |
| `render_python` hook | when the snapshot is rendered | **none**: `render_python` has no length limit | hook nesting: at most 64 hooks run inside one another, deeper nodes render generically. Otherwise none: the hook formats nodes that capture already bounded. Its return value is not checked for length; a hook is trusted code, and its output is as long as the string it builds |

Every part of a shape is built once, top-down, in its final slot: an accessor field and an element take the capture of what they return (one node of `max_nodes` work, as any value); a structure part (`items`, `labels`, a level of `data`) is one node, and a structure part left no node work is its empty, typed fallback with reason `max_nodes`. So capture limits bound what a snapshot holds, and only `render_json` bounds what a renderer produces. Children a hook renders through `r` get the renderer's notation, cut markers and hooks, nothing more.

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
    kind: str | None = None       # rendering key set by the shape, e.g. "numpy.ndarray"
```

`render_python` has the one hook registry, keyed by `kind` (principle 3): `render_output` and `render_json` have none.

```python
@render_python.register("numpy.ndarray")
def render_ndarray(node: Object, r: PythonRenderer) -> str:
    shape = r.render(node.field("shape"))
    data = r.render(node.field("data"))
    return f"array({data}, shape={shape}, dtype={r.render(node.field('dtype'))})"
```

- A hook receives the node and the renderer `r` (a `PythonRenderer`). It renders children only through `r.render(child)`, so they keep the renderer's notation, cut markers and their own hooks; `r.generic(node)` renders a node without its hook. Nothing bounds the string a hook returns (see [Extension contract](#extension-contract)): it formats nodes that capture already bounded, and it is trusted code.
- No hook for a `kind`, or a hook that raises (or returns something other than a `str`): generic rendering (`ndarray(dtype=…, shape=…, data=…)`). Rendering never fails.
- Hooks nest at most 64 deep, the one re-entry rule: a node met while 64 hooks are running renders generically. So a hook that renders its own node through `r`, or builds a fresh node of its kind each time, ends after 64 levels, well inside Python's recursion limit; no `RecursionError` is raised or swallowed. Siblings don't count: each running hook adds one level.
- `render_python.register(kind)` as a decorator, `register(kind, hook)`, `unregister(kind)`; registering again replaces the hook.
- `render_output` renders a `Text` in the output notation; other nodes use the generic Python notation, with nested `Text` in the output notation. `render_json` encodes every node generically, so trace files remain plain and tool-readable.
- `kind` is a plain string, so render hooks don't need the type: a hook can be registered without importing the library.

### First users

Open PR **#87** ("shape-aware pformat previews for numpy and pandas", pierreolivierbonin) already designs previews for these types. Once PR 1 lands, its previews become shape registrations, made where numpy and pandas are imported: `numpy.ndarray` as `ndarray_like` (shape, element, `dtype` field), `pandas.DataFrame` as `ndarray_like` over its values with `labels={1: columns, 0: index}` (and a `dtypes` field), `pandas.Series` as `ndarray_like` with `labels={0: index}` (or `mapping_like` over its index and values), plus `render_python` hooks for their kinds where the generic notation is not enough.

## The implementation (`nooa.snapshot`)

Public API, all importable from `nooa.snapshot`:

| Name | What it is |
|---|---|
| `Scalar`, `Text`, `Sequence`, `Mapping`, `Object`, `Unavailable`, `Ref`, `Spill` | the frozen snapshot nodes; `Snapshot` is their union, `Reason` / `REASONS` the reason names, `Key` a mapping key |
| `capture(value, limits=DEFAULT_LIMITS)`, `Limits`, `DEFAULT_LIMITS` | a snapshot of a finished value; `capture.register(T, shape)` / `capture.unregister(T)` extend it (process-wide, matched by class identity) |
| `array_like`, `ndarray_like`, `mapping_like`, `object_like`, `text_like` | the five shapes `capture.register` takes (see *Capture: shapes, registered by type*) |
| `capture_fields(**fields, limits=...)` | #342's `trace_fields`: a framework wrapper whose every root field survives with its type |
| `TextStream`, `SpillTarget`, `DEFAULT_SPILL_BYTES`, `merge` | output that arrives over time; `merge` joins unread deltas |
| `render_output`, `render_python`, `render_json`, `preview_attributes` | the renderers; only `render_python` takes per-kind hooks (`.register(kind)`) |
| `render_trace(snapshot, direction=…, max_chars=…)` | the trace entry point: `(render_json text, preview_attributes)` from one JSON walk |
| `PythonRenderer` | the `r` a `render_python` hook gets: `r.render(child)`, `r.generic(node)` |

Details that the sections above leave open:

- **`Limits`** has `max_chars`, `max_nodes`, `max_depth` (#342's defaults: 50,000 / 2,000 / 16), `max_items` and `max_string` (`None`: only the shared budget) and `tail_share`, the part of a cut string or container kept from its end (0: a prefix, as in trace previews; 0.5: agentdoc's split). `max_items` caps the items of every container, the fields of an `Object` too: an object keeps its first `max_items` fields (not a head and tail) and is marked `max_items` when a later field was left out. `max_string` is an upper bound: a byte value keeps only whole code points within it, so a code point wider than what is left is omitted.
- **`max_chars` counts canonical JSON**, exactly as #342 does. So `render_json(capture(v, limits), max_chars=limits.max_chars)` gives #342's text and incomplete paths (checked against the original builder on random shapes). `render_json` also walks the snapshot with the same accounting, so it stays valid and within any smaller `max_chars`, and `preview_attributes(..., max_chars=...)` reports the cuts that rendering made. For the trace path, `render_trace(snapshot, direction=…, max_chars=…)` returns the `render_json` text and the attributes from one JSON walk. Spill files (`.full_output`) are collected in their own small walk over the snapshot: every `Text.full` path, depth first, heads and tails, whatever the JSON shows, so they don't depend on `max_chars`.
- **Nodes beyond the design's fields:** `Text.omitted` (the units left out between head and tail), `Sequence.tail` / `Mapping.tail` (items kept from the end), `Sequence.type_name` / `Mapping.type_name`, `Object.module`, `Unavailable.type_name` (picks #342's typed fallback), `Ref.kind`, and `Mapping.wrapper` (marks a `capture_fields` root). `Ref.id` is the JSON Pointer of the enclosing container that the cycle points back to. Repeated references that are not cycles are captured again each time, as in #342.
- **Shape fields and reservation.** The accessor fields of a shape are built in order, and each leaves room (characters and node work) for the minimal value of every later field, as #342's `trace_fields` does for root fields; a later accessor field, whose value is not known before its accessor runs, counts as a 2-character value and 16 nodes, a structure field (`items`, `data`, …) as an empty container and one node. So in the ndarray example `dtype` and `shape` stay complete and `data` gets what is left: present, possibly cut, and incomplete with the reason that stopped it (`max_items`, `max_length` or `max_nodes`). A field is never held below its own minimal value (it gets `max(min(room, its minimum), room - reserved)` characters), and node work is held back only down to one node for the current field, so in a slot too small for all the minimums the later fields are cut. Each part is built in its slot's context: one level deeper than the object, under its field name, index or key (so `max_depth` holds for recursive registered types, and cycle `Ref` ids are field-name pointer paths). Fields that fit are kept whole.
- **Reason `error`** is added to the list: a shape accessor that raised, or returned something its shape can't use (see *Errors* under the shapes).
- **Dataclasses:** the declared fields come from the nearest `__dataclass_fields__` in raw class namespaces. Values come from the native `__dict__` getset descriptor or the slot member descriptors, never through properties. `_private` fields, `repr=False` fields, `ClassVar`/`InitVar` fields and hidden fields (`Annotated[..., hidden]`, `spec(hidden=True)`, `spec()` overrides on the class or instance) are left out. When a field's annotation is a string (PEP 563), capture asks Python for the class's resolved annotations with `typing.get_type_hints(cls, include_extras=True)`, once per class, and applies the same rule to them as to a real annotation: only the outermost `Annotated` counts, and it hides the field when its metadata holds `hidden` or a `spec(...)` marker (exact type) with `hidden=True`. So an alias (`A = Annotated`, `Secret = Annotated[str, hidden]`), a marker made by a call (`spec(hidden=FLAG)`, a helper that returns `spec(hidden=True)`) and foreign metadata (`Field(gt=0)`, which stays visible) all behave as they do at runtime. The result is cached by class identity (keyed by `id`, re-checked with `is`, dropped with the class), so the class is never hashed or compared. If Python can't evaluate the annotations (a name that exists only under `TYPE_CHECKING`, a missing or unloaded module, any other error), every field of that class with a string annotation fails closed: it is left out and the object is marked incomplete (reason `unreadable`). The same happens to a field whose marker can't be read (a `spec` marker whose `hidden` is not a `bool`). Fields whose annotations are real objects are read directly and are not affected.
- **Mapping keys** may be exact `str`, `int`, `float`, `bool` or `None`. `render_json` converts them the way `json.dumps` does, and stops the mapping at a key that would collide.
- **`bytes` values** become a `Scalar` when they fit, otherwise a byte `Text` holding the original bytes, cut on code points (a cut never ends inside a valid sequence, so the decoded JSON preview stays as #342-style text; invalid bytes are kept as they are). `render_json` shows them decoded with replacement, `render_python` as `bytes(len=N, [:h]=b'...', [-t:]=b'...')`.
- **`TextStream.since(mark)`** returns what memory still holds of the delta: the part still in the head and the part in the rolling tail. A byte stream's mark stops before an incomplete UTF-8 sequence. `merge` cuts each input in its own units: a bytes delta holds raw bytes, so whole parts and cuts are exact (`length == len(head) + len(tail) + omitted`, both limits hold), and a bytes cut keeps whole code points as the stream does.
- **`TextStream` threads:** one writer. `snapshot()` and `since()` only read the writer's buffers (they never rebind them) and flush the spill file, but are not synchronised with `write`: call them from the writer's thread, or serialise them with writes yourself. PR 1 has no lock.
- **Spill file on a running stream:** `snapshot()` and `since()` flush the spill file first, so `Spill.kept` bytes are on disk; a flush `OSError` is handled like a write failure (`full` becomes `None`, file removed, memory kept). A chars stream encodes for the file only while a spill file is active: UTF-8 with surrogate escapes as raw bytes, else (a lone surrogate) `backslashreplace`; memory always keeps the chunk as written.
- **Spill state is in `Text.full`, not in the reason:** a cut stream `Text` always has reason `max_length`. `full.cut` says the spill file is itself cut (`render_output`: `PATH holds the first N of M bytes`). When a spill was requested but the file could not be created or written, the stream keeps memory only, removes the partial file and reports `full=None`, as when no spill was requested (`full output not kept`).
- **`render_python`** copies agentdoc's markers: `str(len=N, [:h]='…', [-t:]='…')`, `list(len=N, [:h]=[…], [-t:]=[…])`, `dict(len=N, items={…, ..., …})`, `[list: N items]` for depth cuts, `<cycle>` and `Type(a=1, ...)`. It does not column-wrap at 120 columns the way `pformat` does (a complete multiline string still keeps its literal newlines, in triple quotes). Values NOOA does not inspect render as `<opaque: Type>`, and other `Unavailable` nodes as `<unavailable: reason>`.
- **Module layout:** `_model` (the nodes), `_codec` (a private leaf: canonical JSON encoding and size accounting, #342's fallbacks and opaque label, JSON Pointers, mapping-key conversion, UTF-8 cut points and decoding with replacement), `_capture` (live values), `_stream` (`TextStream`, `merge`) and `_render`. Capture, stream and render share `_codec`; render and stream do not import the live-object walker. Each keeps its own safety policy: capture never runs user code on a live value, rendering reads detached nodes only. So mapping keys convert under one helper with an explicit policy: capture refuses a key of a live value that is an `int` over 128 bits or a non-finite `float`, the renderer converts a detached key as it is.
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
| Trace hooks | #342's `trace_json` / `trace_fields` | `capture` + `render_trace` |
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
4. **Capture extensions register by type only:** `capture.register(T, shape)`, one registry per process; registration by module and class name was removed. ("I would do 2 register(T) and make it per process. Don't overcomplicate things.")
5. **Extensions declare shapes, not handlers:** "rather than writing a handler from scratch, could we have a handler that says, 'Hey, this is array-like. Here's how to get the size. Here's how to get the data. Here are a couple of other properties.' Rather than doing the printing yourself, you're mapping out the structure of the type", "maybe also ndarray like". Shapes (`array_like`, `ndarray_like`, `mapping_like`, `object_like`, `text_like`) are the only public extension API; the function handler with its `out` / `Out` and deferred parts was removed, and its builder became internal. "Use shapes for internal types (minus hot path)": Pydantic models and dataclasses are internal `object_like` shapes over their privileged readers, and the exact built-in fast paths stay, checked first, sharing their building code with the shapes.
