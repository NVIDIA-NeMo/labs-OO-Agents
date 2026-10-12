# Bounded Trace JSON Preview

## Goal

Replace the current streaming-prefix envelope with one small, maintainable
preview policy for arbitrary Python values recorded as trace inputs or outputs.
Capture work and retained memory must not scale with the unvisited remainder of
large supported values. The resulting `input.value` / `output.value` must remain
ordinary, valid JSON whose framework-owned outer fields survive truncation.

This deliberately does not promise a wall-clock deadline for every trace
operation. Known text paths such as exception formatting, system messages, and
stack traces remain separate from arbitrary Python value capture.

## Public behavior

- Trace values are bounded, non-replayable previews.
- Inputs and outputs containing arbitrary Python values use
  `application/json`.
- No truncation marker is embedded in user data.
- Direction-scoped span attributes report whether and where the preview is
  incomplete:

  ```text
  nooa.input.preview.version = 1
  nooa.input.preview.incomplete = true
  nooa.input.preview.paths = ["/args/0"]

  nooa.output.preview.version = 1
  nooa.output.preview.incomplete = true
  nooa.output.preview.paths = [""]
  ```

- The paths use RFC 6901 JSON Pointer syntax. A root path (`""`) means the
  bounded path metadata cannot identify every incomplete location.
- Consumers parse ordinary JSON and show a generic `Incomplete preview`
  notice. They do not sniff payload keys or reconstruct missing data.

## Internal API

Add `src/nooa/tracing/_trace_json.py`:

```python
@dataclass(frozen=True)
class TraceJSON:
    text: str
    incomplete_paths: tuple[str, ...]


@dataclass(frozen=True)
class Limits:
    max_chars: int = 50_000
    max_nodes: int = 2_000
    max_depth: int = 16


def trace_json(value: object, *, limits: Limits = DEFAULT_LIMITS) -> TraceJSON:
    """Return a bounded JSON preview of one arbitrary Python value."""


def trace_fields(*, limits: Limits = DEFAULT_LIMITS, **fields: object) -> TraceJSON:
    """Capture a small framework-owned object while reserving every root field."""
```

`trace_fields` is used only for fixed framework wrappers:

- method/agent input: `args`, `kwargs`
- code input: `code`
- execution output: `stdout`, `stderr`, `returned_value`

It validates that `max_chars` can hold the complete minimal wrapper before it
inspects values. Invalid internal limits raise `ValueError`; they never create a
fallback envelope.

The minimal wrapper preserves root value types, not only keys. Before walking
children, `trace_fields` classifies its small framework-owned set of values and
reserves `[]` for list/tuple values, `{}` for dictionaries/Pydantic values, and
`""` for strings. Bounded scalars may reserve their complete scalar; unknown
values reserve a fixed opaque string. If the inspection budget cannot even
classify these framework-provided roots, configuration is invalid. Tests cover
field presence and type with the smallest valid work/depth/character budgets.

Every node reserves a worst-case valid fallback before it is committed to its
parent. If a preferred opaque label cannot fit that reservation, retain the
smaller type-appropriate placeholder (`{}`, `[]`, `""`, or `null`), mark the
path incomplete, and keep the parent's accounting unchanged. A later
fail-closed decision must never replace a committed fallback with a larger
value.

`trace_json` likewise validates that its limits can hold at least one valid
JSON value before inspecting the source. `max_nodes <= 0`, `max_depth < 0`, or
character budgets too small for the root's bounded fallback are rejected.

## Supported values

The walker handles only exact built-in types so subclass hooks cannot run:

- `str`, `None`, `bool`, `int`, `float`
- `list`, `tuple`
- `dict` with exact `str` keys
- a narrow Pydantic v2 `BaseModel` stored-field view

Unknown values and subclasses become a bounded string such as
`"<opaque: SlowWidget>"` and their path is incomplete. Obtain the type name
without invoking instance formatting or custom metaclass hashing/equality.

The walker must never invoke arbitrary:

- `repr()` / `str()` / truthiness
- iteration on custom containers
- `getattr()`, properties, or descriptors
- Pydantic `model_dump()`, computed fields, or serializers
- user conversion callbacks

Use active-ancestor identity tracking for cycles. Repeated non-cyclic references
are not cycles.

## Pydantic v2 view

Pydantic support is deliberately not normal Pydantic serialization. It is a
bounded view of declared values already stored on the instance.

1. Detect supported `BaseModel` instances using raw `type()` and identity-based,
   inspection-budgeted MRO checks. Do not use `isinstance`, `getattr`,
   descriptor access, type-keyed dispatch that can invoke metaclass hashing or
   equality, or instance attribute checks.
2. Read raw class namespaces with `type.__getattribute__(cls, "__dict__")`
   and raw instance storage with `object.__getattribute__(obj, "__dict__")`.
   Require exact built-in dictionaries. Obtain the already-built Pydantic field
   map only from a verified raw class namespace (`__pydantic_fields__`, or a
   version layout verified in tests). Unsupported layouts make the model
   opaque; do not evaluate `model_fields` descriptors.
3. Before emitting any field, gather NeMo imperative visibility overrides from
   raw `_agentdoc_fields_docs` dictionaries. Accept only exact dictionaries,
   exact-string keys, exact dictionaries for field entries, and exact boolean
   `hidden` values. Instance overrides win, followed
   by the most-derived class in MRO order, followed by resolved field metadata.
   MRO and metadata inspection consume the inspection budget. If the complete
   override state cannot be established safely or within budget, make the
   entire model opaque before emitting any field.
4. Iterate declared fields in field-map order. A considered field consumes an
   inspection unit even when omitted.
5. Include only fields present in native stored values.
6. Unconditionally omit `exclude=True` and `repr=False`. `hidden=False` cannot
   override either policy.
7. Require verified ordinary Pydantic `FieldInfo` objects, not malicious
   subclasses. Apply NeMo `hidden` / `spec(hidden=...)` only from recognized
   marker identities/types. Ignore unrelated Pydantic constraint metadata
   without invoking or inspecting it.
8. Do not include extras, private values, computed/cached properties, or
   undeclared instance attributes in v1.

Intentional visibility/exclusion is complete under this view and does not mark
the preview incomplete. Work/depth/size exhaustion after safe fields retains
those fields and marks the model path incomplete. Malformed visibility storage
or field metadata fails closed to an opaque model.

Do not reuse `agentdoc.is_hidden_field()` or `_extract_instance_values()`:
those helpers can resolve annotations or eagerly discover values and therefore
do not satisfy this capture path's bounded-work and no-user-code requirements.

## Budget algorithm

Use one recursive preview-tree builder with shared mutable state:

- `max_chars`: exact final character count using one canonical encoder
  configuration everywhere: `ensure_ascii=True`, `allow_nan=False`,
  `separators=(", ", ": ")`, `sort_keys=False`.
- `max_nodes`: Python-level inspection budget. Every considered container
  item, Pydantic field, MRO entry, and visibility entry consumes work.
- `max_depth`: recursion budget.

The root starts at depth zero. A node is charged whenever the implementation
examines a source value, container item, Pydantic field, MRO entry, or visibility
entry. Syntax reservation itself is not a node. The builder returns
`(preview_value, encoded_length)` for each retained node:

1. Reserve a valid type-appropriate fallback, then charge inspection work
   before classifying a value. Work exhaustion returns the already-reserved
   fallback and marks the current path incomplete.
2. Reserve container opening/closing characters.
3. Before a child, reserve its separator and a dictionary's complete encoded
   key.
4. Visit the child with only the remaining budget.
5. Atomically commit a complete key and valid child preview. The child itself
   may be an incomplete prefix/container.
6. Stop immediately on space or work exhaustion; never scan for a later value
   that might fit.

Strings are sliced to no more source characters than the available encoded
budget before calling `json.dumps()`. If escaping expands that bounded
candidate beyond the budget, use bounded binary search with `json.dumps()` to
find a fitting prefix. The unvisited suffix is never copied or escaped.

Do not truncate dictionary keys. If an entire bounded key does not fit, omit
that entry, mark the containing path incomplete, and stop that dictionary.
Unsupported key types behave the same way; never scan later entries for one
that might fit. Preflight exact string keys with `len(key) + 2` as a lower
bound. If that already exceeds the remaining complete-key budget, stop without
copying or escaping the key. Call `json.dumps()` only when the source key length
is itself bounded by the remaining budget. Apply the same rule to Pydantic
field names.

Guard integers using `int.bit_length()` before decimal conversion. Values whose
decimal representation could exceed the remaining/bounded budget become
opaque. Convert non-finite floats to a fixed bounded opaque string and mark the
path incomplete, because `allow_nan=False` rejects them.

The final `json.dumps()` must use the exact same fixed options as size
accounting. Tests compare the accounted length with the actual output length.

Opaque labels are assembled from a bounded prefix of a type name; never
interpolate an unbounded class name and truncate afterward. Type names are read
through native `type` access only.

Incomplete paths have a fixed small internal count and character budget,
independent of payload limits. Overflow collapses the set to the root path.
Pointer construction escapes `~` and `/` per RFC 6901 and is covered at exact
metadata-size boundaries.

## Hook integration

Add one helper in `_hooks_impl.py` that writes a `TraceJSON` to a span:

- set `input.value` or `output.value`
- set the matching MIME type to `application/json`
- always set direction-scoped `version` and `incomplete`
- set `paths` only when incomplete

Migrate every existing arbitrary-value producer explicitly:

- `before_agent_call` input (`trace_fields(args=..., kwargs=...)`)
- `after_agent_call` output
- `after_generation` output
- `before_code_execution` input (`trace_fields(code=...)`)
- `after_code_execution` output, both ExecutionResult and generic branches
- `before_method_invocation` input (`trace_fields(args=..., kwargs=...)`)
- `after_method_invocation` output
- `before_tool_execution` input
- `after_tool_execution` output

Remove input pre-truncation such as `code[:10000]`; otherwise the codec cannot
report that data was omitted. Continue recording the original `code.length`.

Execution results keep their explicit three-field wrapper. Recognize only
`type(result) is nooa.events.ExecutionResult`; subclasses and unrelated
lookalikes take the generic path. Read its exact-dict native storage without
`hasattr` or attribute access,
map `_NO_RETURN` to `None` by identity, and pass raw `stdout`, `stderr`, and
`returned_value` into `trace_fields`. Generic/subclassed lookalikes use
`trace_json`; do not probe them. Do not format execution values first.

Avoid incidental user hooks while touching these paths, including result
truthiness checks used only to derive `result.type`. Derive type names through
the same bounded native helper used for opaque labels.

The same input JSON string remains mirrored into
`tool_call.function.arguments` where required by the current OpenInference
mapping.

Known text-only trace fields remain unchanged and outside the architectural
claim: system messages, error messages, and stack traces.

Completion includes an `rg` audit proving no production reference remains to
`_safe_serialize`, `_safe_json_value`, `_safe_serialize_execution_result`,
`_truncated_json_envelope`, `_TRACE_TRUNCATION_KIND`, or `_limited_writer`.
Update `tests/unifiedllm/test_response_display_privacy.py`, whose direct tracing
privacy assertion currently calls `_safe_serialize`.

## Consumer integration

### Python explorer

- Delete `$nooa` envelope handling from `_io_json_field`.
- Continue parsing standard JSON objects for `args`, `kwargs`, `code`, and
  execution results.
- Add one shared MIME-aware decoder. For `application/json`, parse JSON: a JSON
  string such as `"done"` becomes human text `done`, while objects/arrays stay
  structured. For `text/plain`, never JSON-parse text such as
  `"{'stock': 7}"`. When MIME is missing, retain the current legacy heuristic
  only for existing fallback attributes.
- Preserve `return_result` preview behavior for an incomplete string value.
- Apply the rule to agent results, generation/session results, execution
  results, tool results, and `return_result` previews. Expose the
  direction-scoped incomplete flag/paths only where needed for a warning.

### React viewer

- Delete `truncatedJson.ts` and envelope-specific branches.
- Add one shared MIME-aware value decoder and a tiny metadata reader/helper for
  direction-scoped preview state.
- Method, code, and tool plugins parse normal JSON and show `Incomplete
  preview` when indicated.
- Apply the decoder to MethodPlugin, CodeExecutionPlugin,
  ToolExecutionPlugin, and generic SpanPlugin. Render JSON-string outputs as
  their string contents where the existing view expects human text. Legacy
  text/plain and missing-MIME behavior is covered explicitly.
- Rebuild checked-in `dist/` assets.

Do not retain experimental envelope decoding unless an actual persisted-data
compatibility requirement is demonstrated.

## Regression-first implementation sequence

1. Replace envelope-focused tests with failing tests for the approved public
   behavior before editing production code:
   - preserved `args`/`kwargs` and `code` roots under truncation
   - nested Pydantic model with a huge inner string
   - opaque object whose `repr`/`str` explode or sleep
   - bounded tail traversal using sentinel values
   - direction-scoped input/output metadata
   - JSON output MIME and string output parsing
2. Confirm those focused tests fail against the current implementation.
3. Implement `_trace_json.py` and its exhaustive unit tests.
4. Integrate producers through one hook helper and remove
   `_limited_writer.py`, `_safe_json_value`, `_safe_serialize`, and the special
   execution-result serializer where no longer used.
5. Simplify Python/React consumers and replace envelope fixtures with ordinary
   preview fixtures.
6. Update the experiment and README to benchmark the new API and accurately
   scope claims.
7. Build the frontend distribution.

## Test matrix

### Codec unit tests

- Exact standard JSON for fitting native values.
- Exact cap and accounting agreement for normal and escape-heavy strings.
- A huge nested string only copies/encodes a bounded prefix.
- Wide empty containers are stopped by inspection work.
- Deep containers stop at `max_depth` without recursion failure.
- Active cycles are incomplete; shared non-cyclic references are retained.
- Huge keys, unsupported keys, giant integers, NaN/infinity, tiny/invalid
  wrapper limits, and metadata overflow.
- A multi-megabyte escape-heavy dictionary key and Pydantic field-name preflight
  do bounded work without encoding the rejected key.
- Exact fits around quotes, commas, colons, closing delimiters, canonical
  separators, and JSON Pointer escaping.
- Unknown subclasses and custom objects never run `repr`, `str`, iteration,
  truthiness, hashing/equality, getters, or descriptors.
- Pydantic fail-closed fallback and long opaque type names fit exactly inside
  root, list, tuple, dictionary, and reserved-field containers.
- Tail sentinels prove no traversal after exhaustion.

### Pydantic tests

- Small and nested models remain useful dictionaries.
- Huge nested strings are bounded.
- `exclude=True`, `repr=False`, Annotated `hidden`, `spec(hidden=True/False)`,
  inherited overrides, secrets, extras, computed properties, custom
  serializers, and custom attribute access.
- Exclusions beat visibility opt-ins.
- Unsupported/malformed storage and metaclass/subclass bombs fail opaque.
- Large field/MRO metadata is inspection-bounded without disclosure.
- Malicious FieldInfo subclasses, huge class names/MROs, inherited annotations,
  and visibility-budget exhaustion cannot leak a secret.

### Integration/consumer tests

- Every migrated arbitrary-value producer enumerated under Hook integration
  carries valid bounded JSON, JSON MIME, version, and incomplete state. Named
  context/text/error paths remain text and separately retain text semantics.
- Required framework wrapper fields survive.
- `input.value` and `tool_call.function.arguments` match.
- Explorer and viewer display method calls, code, execution results, JSON
  string outputs, and `return_result` previews.
- User data resembling the old `$nooa` envelope remains ordinary data.
- Existing legacy native-attribute fallbacks continue working where already
  supported.
- MIME-aware decoding covers agent, generation/session, code, tool,
  `return_result`, MethodPlugin, CodeExecutionPlugin, ToolExecutionPlugin, and
  SpanPlugin behavior for JSON strings, structures, text/plain, and missing
  MIME.
- OpenInference assertions require byte-identical `input.value` and
  `tool_call.function.arguments`, required root child types under truncation,
  direction-scoped bounded metadata, no marker inside payloads, JSON MIME on
  every migrated arbitrary-value path, and unchanged text MIME for text-only
  paths.

### Verification

- Focused tracing and explorer tests.
- Full relevant Python test suites and Ruff/pre-commit checks.
- Frontend type-check/build and formatting check.
- Original latency reproduction plus nested-string, object-tail, memory, and
  event-loop benchmarks. Document measured results without claiming an
  absolute real-time guarantee.

## Files expected to change

- `src/nooa/tracing/_trace_json.py` (new)
- `src/nooa/tracing/_limited_writer.py` (delete)
- `src/nooa/tracing/_hooks_impl.py`
- `src/nooa/trace_explorer/explorer.py`
- React viewer plugins/utilities and checked-in `dist/`
- tracing/explorer tests, replacing envelope-specific tests
- `experiments/trace_serialization_latency/*`

No dependency or public configuration change is required.
