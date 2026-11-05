# One snapshot model for values, output streams and spilled files

This page is the contract of `nooa.snapshot`. The docstring of every public name is its documentation: `help(name)` or `nooa.agentdoc.doc(name)` shows it. The package builds on Severin Klingler's bounded trace JSON builder (PR #342).

## Principles

Three rules decide every design question in this package. When code and these rules disagree, the code is wrong.

1. **Never run code on values; read class metadata the way Python does.** Capture never calls a value's `repr`, `str`, properties, descriptors or iteration: values are captured over and over, and their code can be slow, have side effects or reveal hidden fields. Class declarations are different: annotations and field declarations are code the class's author wrote, which Python already runs when the class is defined. Capture reads them once per class with `typing.get_type_hints(cls, include_extras=True)`, cached by class identity, the same way `agentdoc` does. If they can't be read, the affected fields fail closed: they are left out and the object is marked incomplete.
2. **Store what you cut in its own units.** A cut string keeps characters; cut bytes keep bytes. Nothing is decoded until it is rendered, so counts stay exact and no unit is reconstructed later.
3. **One extension point per job, and nothing without a user.** Shapes are the only way to describe a type to capture, and `render_python` hooks the only way to change how a node prints. A field, reason or hook that no consumer reads is not added.

## What it is for

NOOA cuts large values and large output in many places: shell results, progress, cell stdout, job tails, prefill previews and trace previews. `nooa.snapshot` gives them one model. A producer turns a value or a stream into a small, detached snapshot that says what was kept, how much was left out and where the rest is. A renderer turns the snapshot into the notation a consumer shows. Every consumer then reports the same facts the same way.

## The model

A snapshot is a frozen tree of nodes. It holds no live objects. Every node has `complete` and `reason` (`Reason`); `complete` is `reason is None`.

| Node | Stands for |
|---|---|
| `Scalar` | a complete `None`, `bool`, `int`, finite `float` or short `bytes` |
| `Text` | a string, bytes or a stream: kept head and tail in their own unit, `length`, `omitted`, and `full` (a `Spill`) |
| `Sequence` | a list, tuple or `array_like` value: kept head and tail items, original `length` |
| `Mapping` | a dict, `mapping_like` value or `capture_fields` wrapper: kept entries, original `length` |
| `Object` | a Pydantic model, dataclass or shape with fields: kept fields in order, optional `kind` |
| `Unavailable` | a value with no content kept: `opaque`, `error` or a limit |
| `Ref` | a cycle: the JSON Pointer of the enclosing container |

## Public API

Everything is importable from `nooa.snapshot`; see each docstring for arguments, defaults, guarantees and errors.

| Name | What it is |
|---|---|
| `Scalar`, `Text`, `Sequence`, `Mapping`, `Object`, `Unavailable`, `Ref` | the snapshot nodes (table above) |
| `Snapshot`, `Reason` | the union of the nodes; the reason names |
| `Spill` | where a `Text`'s whole content was written |
| `capture`, `Limits` | a bounded snapshot of a finished value, and its shared budget |
| `capture.register`, `capture.unregister` | describe another type with a shape (process-wide, by class identity) |
| `array_like`, `ndarray_like`, `mapping_like`, `object_like`, `text_like` | the five shapes `capture.register` takes |
| `capture_fields` | a framework wrapper whose every root field survives (#342's `trace_fields`) |
| `TextStream`, `SpillTarget` | output that arrives over time, and its optional spill file |
| `SpillDir` | a folder of spill files for many streams, within a file count and a byte total |
| `merge` | join two unread progress deltas |
| `render_output` | command output: `HEAD`, `[... N bytes omitted; full output: PATH]`, `TAIL` |
| `render_python`, `PythonRenderer` | agentdoc's prefill notation, its per-kind hooks, and the `r` a hook gets |
| `render_json` | plain JSON within `max_chars` (trace previews) |
| `render_trace` | `render_json` text plus the trace span attributes, from one walk |

## Extension contract

- **Capture: shapes.** `capture.register(T, shape)` takes the class itself and one of the five shapes. A shape maps out the type's structure through accessors; capture builds the nodes with the builders and limits of the built-in types. Accessors are trusted code, run only when their result is kept, and an accessor that raises becomes `Unavailable("error")`. The nearest registered class in a value's MRO wins; exact built-in types are always native.
- **Rendering: `render_python` hooks.** `render_python.register(kind)` registers `hook(node, r) -> str` for `Object` nodes whose `kind` matches. A hook renders children only through `r.render(child)`. A hook that raises or returns a non-`str`, or one nested more than 64 deep, falls back to the generic notation. `render_output` and `render_json` have no hooks.
- Nothing else is an extension point.

## Limits and guarantees

- **No user code on values.** Capture never calls a value's `repr`, `str`, properties, descriptors, iteration, hashing or equality. It reads class metadata once per class with `typing.get_type_hints(cls, include_extras=True)`. `src/nooa/snapshot/_raw.py` is the module to audit.
- **Fail closed.** Hidden fields are never read. A field whose visibility can't be read is left out, and the object gets reason `unreadable`.
- **Bounded size, time and memory.** `Limits` bounds a capture: `max_chars` (canonical JSON), `max_nodes` (work), `max_depth`, `max_items`, `max_string`. A `TextStream` holds at most `head + tail` units; its spill file at most `SpillTarget.max_bytes` bytes.
- **Shared spill budget.** A `SpillDir` holds at most `max_files` files and `max_bytes` bytes at once (an open stream counts its target's `max_bytes`, a closed file its size), evicting the oldest closed files for room, and deletes what it created on `close()`; a `Text.full` in it may name a file deleted since.
- **Exact accounting.** `Text.length == len(head) + len(tail) + omitted`, in the text's own unit. `render_json(capture(v, limits), max_chars=limits.max_chars)` cuts nothing further.
- **UTF-8-safe cuts.** Bytes stay bytes until rendered, and every bytes cut lands on a code point boundary.
- **#342 parity.** On #342's domain, `render_json` and `render_trace` give #342's text, incomplete paths and attribute names.
- **Rendering never fails** because of a hook. Only `render_json` bounds its output; the text renderers format what capture bounded.

## Plan

1. **PR 1 (this package), into `main`:** the model, `capture` and its registry, `TextStream`, `merge` and the renderers, with tests. No consumer changes.
2. **PR 2a, consumers the model doesn't read:** trace hooks (`capture` + `render_trace`) and ACP activity cards.
3. **PR 2b, consumers the model reads:** `Shell`, progress, cell output, jobs, the core bash tool and prefill, after an A/B run. It ends by deleting `src/nooa/agentdoc/_truncating_stream.py`.
