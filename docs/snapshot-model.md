# One snapshot model for values, output streams and overflow files

This page is the contract of `nooa.snapshot`. The docstring of every public name is its documentation: `help(name)` or `nooa.agentdoc.doc(name)` shows it.

## Principles

Three rules decide every design question in this package. When code and these rules disagree, the code is wrong.

1. **Never run code on values; read class metadata the way Python does.** Capture's readers never call a value's `repr`, `str`, properties, descriptors or iteration: values are captured over and over, and their code can be slow, have side effects or reveal hidden fields. Two kinds of code are run on purpose, and only these: the accessors of a shape you registered (your code) and, when the caller asks for it with `Limits(fallback="repr")`, the value's `repr`. Class declarations are different: annotations and field declarations are code the class's author wrote, which Python already runs when the class is defined. Capture resolves the string annotations of a dataclass with `typing.get_type_hints(cls, include_extras=True)`, the same way `agentdoc` does; a success is cached by class identity, and a failure holds for that capture only (the next one tries again). When that fails, each annotation is resolved on its own, as agentdoc does, and only the fields whose annotations can't be read fail closed: they are left out and the object is marked incomplete. Pydantic models are read from the `FieldInfo`s Pydantic already built, and plain objects from their class annotations as `annotationlib.get_annotations(cls, format=Format.FORWARDREF)` gives them, read from the raw class namespace: the `__annotations__` dict before Python 3.14, and on 3.14 (PEP 649) the cached dict or the class's `__annotate__` function, so a lazily annotated `Annotated[T, hidden]` stays hidden. Annotations that can't be read fail closed. The markers themselves (`hidden`, `SpecAnnotation`, the attributes `spec()` stores its keywords in) come from the public `nooa.agentdoc.markers`, which agentdoc uses too.
2. **Store what you cut in its own units.** A cut string keeps characters; cut bytes keep bytes. Nothing is decoded until it is rendered, so counts stay exact and no unit is reconstructed later.
3. **One extension point per job, and nothing without a user.** Shapes are the only way to describe a type to capture, and `render_python` hooks the only way to change how a node prints. A field, reason or hook that no consumer reads is not added.

## What it is for

NOOA cuts large values and large output in many places: shell results, progress, cell stdout, job tails, prefill previews and trace previews. `nooa.snapshot` gives them one model. A producer turns a value or a stream into a small, detached snapshot that says what was kept, how much was left out and where the rest is. A renderer turns the snapshot into the notation a consumer shows. Every consumer then reports the same facts the same way.

## The model

A snapshot is a frozen tree of nodes. It holds no live objects. Every node has `complete` and `reason` (`Reason`); `complete` is `reason is None`. Both are local to the node: a complete `Sequence` kept every item, but an item may itself be cut. `render_trace`'s `incomplete` attribute answers for the whole tree. `Text`, `Sequence` and `Mapping` read the same: `head` is the kept start, `tail` the kept end, `length` the original count.

| Node | Stands for |
|---|---|
| `Scalar` | a complete `None`, `bool`, `int`, finite `float` or short `bytes` |
| `Text` | a string, bytes or a stream: kept head and tail in their own unit, `length`, `omitted`, and `full` (an `Overflow`) |
| `Sequence` | a list, tuple, set, standard container or `array_like` value: kept `head` and `tail` items, original `length` |
| `Mapping` | a dict, standard mapping, `mapping_like` value or `capture_fields` wrapper: kept `head` and `tail` entries (keys: JSON scalars, or a small node for any other key), original `length` |
| `Object` | a Pydantic model, dataclass, namedtuple, plain object, standard type, shape with fields or bounded `repr`: kept fields in order, optional `kind` |
| `Unavailable` | a value with no content kept: `opaque`, `error` or a limit |
| `Ref` | a cycle: the JSON Pointer of the enclosing container |

## Public API

Everything is importable from `nooa.snapshot`; see each docstring for arguments, defaults, guarantees and errors.

| Name | What it is |
|---|---|
| `Scalar`, `Text`, `Sequence`, `Mapping`, `Object`, `Unavailable`, `Ref` | the snapshot nodes (table above) |
| `Snapshot`, `Reason` | the union of the nodes; the reason names |
| `Overflow` | where a `Text`'s whole content was written (path, kept, total) |
| `capture`, `Limits` | a bounded snapshot of a finished value, and its shared budget |
| `capture.register`, `capture.unregister` | describe another type with a shape (process-wide, by class identity) |
| `array_like`, `ndarray_like`, `mapping_like`, `object_like`, `text_like` | the five shapes `capture.register` takes |
| `capture_fields` | a call's arguments captured together, so every named part survives, at least in short form |
| `TextStream` | output that arrives over time, with an optional overflow file in a folder, opened only once the output is cut (`TextStream(overflow=OverflowDir(), overflow_bytes=...)`) |
| `cut` | a string or bytes already whole, as a `Text` keeping its first `head` and last `tail` units (what a one-write `TextStream` gives) |
| `merge` | join two `Text`s of the same unit (progress batches) |
| `render_output` | command output: `HEAD`, `[... N bytes omitted; full output: PATH]`, `TAIL`; `note=` adds a sentence after `full output not kept]` |
| `render_python`, `PythonRenderer` | agentdoc's prefill notation, its per-kind hooks, and the `r` a hook gets |
| `render_json` | plain JSON within `max_chars` (trace previews) |
| `render_trace` | `render_json` text plus the trace span attributes, from one walk |

## Extension contract

- **Capture: shapes.** `capture.register(T, shape)` takes the class itself and one of the five shapes. A shape maps out the type's structure through accessors; capture builds the nodes with the builders and limits of the built-in types. Accessors are trusted code: they run in every capture of the type, traces included, only when their result is kept; `Limits` bounds what is kept of their result, not their own time or memory, and they may read private data. An accessor that raises becomes `Unavailable("error")`. The nearest registered class in a value's MRO wins; exact built-in types are always native.
- **Rendering: `render_python` hooks.** `render_python.register(kind)` registers `hook(node, r) -> str` for `Object` nodes whose `kind` matches. A hook renders children only through `r.render(child)`. A hook that raises or returns a non-`str`, or one nested more than 64 deep, falls back to the generic notation. `render_output` and `render_json` have no hooks.
- Nothing else is an extension point.

## Limits and guarantees

- **No user code on values, with two named exceptions.** Capture's readers never call a value's `repr`, `str`, properties, descriptors, iteration, hashing or equality. They read class metadata the way Python does (see principle 1). `src/nooa/snapshot/_raw.py` and `_std.py` (the standard types, read through their C-level descriptors after an identity check) are the modules to audit. The code that does run: (a) the accessors of registered shapes, with every `Limits`; (b) only with `Limits(fallback="repr")`, for model-facing previews, `repr()` of what capture cannot read and of a class that defines its own `__repr__`. The default (`"opaque"`, for traces) runs no `repr` and labels the value `<opaque: module.QualName>`. A `repr` slower than `SLOW_REPR_SECONDS` (0.1 s) or longer than `LARGE_REPR_CHARS` (1,000,000 characters) logs a `logging` warning naming the type, the time and the size, after it returns; nothing interrupts it.
- **Model-facing precedence.** A shape, then declared structure (Pydantic, dataclasses), then the standard types, then (with `fallback="repr"` only) the class's own `__repr__`, bounded, then public attributes, then the plain `repr` fallback. Traces skip the `__repr__` step and never run `repr`.
- **Standard types and plain objects are native.** Sets, ranges, complex numbers, dates and times, `Decimal`, `UUID`, paths, patterns, exceptions, enum members, namedtuples, `Counter`/`OrderedDict`/`defaultdict`, functions, classes and modules become nodes, and `render_python` prints them in their familiar notation. An int too long to keep shows its digit count (`int(200,001 digits)`), never converted to decimal. A plain object shows its public instance attributes; a cut one also says how many it has (`... of 1,000,000 attributes`).
- **Fail closed.** The field readers (Pydantic, dataclasses, plain objects) never read hidden fields. A field whose visibility can't be read is left out, and the object gets reason `unreadable`. One unresolvable annotation (a name imported only under `TYPE_CHECKING`) costs only its own field. This holds for capture's own readers; a shape accessor or an opted-in `repr` shows whatever its code chooses, private data included, so a caller that must not show private data uses the default `"opaque"` fallback and registers no shape that reads it.
- **Bounded size, time and memory.** `Limits` bounds a capture. `max_chars` (canonical JSON) and `max_nodes` (work) are one budget for the whole capture; `max_depth`, `max_items` and `max_string` cap each value separately (`max_items=2` keeps two items of every list, not two in all). `tail_share` keeps both ends of a cut string, list or dict in that ratio, whichever limit cuts it; at its default 0.0 a capture keeps a head only. These bound capture's own work; the time and memory of a shape accessor or an opted-in `repr` are not bounded, only what is kept of their result. A `TextStream` retains at most `head + tail` units between writes; one `write` may also hold a copy or the UTF-8 encoding of its chunk while it runs. Its overflow file holds at most `overflow_bytes` bytes (at most the folder's `file_bytes`), created exclusively in the folder, mode 0600, only when the output is first cut: an output that is never cut makes no file.
- **Shared file budget.** The overflow files live in `nooa.overflow` (see [overflow.md](overflow.md)), which `nooa.snapshot` imports and which imports nothing from it. An `OverflowDir` holds at most `max_files` files of at most `file_bytes` each and deletes the closed files finished longest ago to make room, never an open stream's file, so a `Text.full` path from an earlier stream can stop existing; opening it then fails with `FileNotFoundError`, which is how a reader can tell the full output is gone.
- **Exact accounting.** `Text.length == len(head) + len(tail) + omitted`, in the text's own unit. `render_json(capture(v, limits), max_chars=limits.max_chars)` cuts nothing further: capture already stops a mapping with reason `key` at a key that collides with an earlier one in JSON (`1` and `"1"`), keeping the first, so capture, `render_json` and `render_trace` agree. A key that is not a JSON scalar gets its JSON key from one canonical encoder (`_keys.key_text`, pinned by golden tests), which capture and the renderers share; changing the display notation does not change it.
- **UTF-8-safe cuts.** Bytes stay bytes until rendered, and every bytes cut lands on a code point boundary. A chars stream's overflow file is UTF-8, cut on a code point boundary too, and `Overflow.kept` counts the bytes written. One exception: a `surrogateescape` character (`"\udcff"`, how Python decodes an undecodable byte) is written back as its original byte, so a file of text holding such characters need not be valid UTF-8 (the bytes are preserved rather than replaced).
- **Trace previews say what they cut.** `render_trace` gives the preview text plus span attributes saying whether it is complete, which paths were cut and why.
- **Rendering never fails** because of a hook. Only `render_json` bounds its output; the text renderers format what capture bounded.
