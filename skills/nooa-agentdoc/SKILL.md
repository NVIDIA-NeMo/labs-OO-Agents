---
name: nooa-agentdoc
description: Make NVIDIA OO agent types render beautiful documentation for the LLM — doc(), spec(), hidden, Annotated field descriptions, and pformat/pprint tuning. Use when designing Pydantic models/dataclasses the LLM will see, controlling what appears in doc(self), hiding internals, adding field descriptions, fixing noisy or missing type docs, or tuning value truncation.
compatibility: nooa package (agentdoc ships inside it — import from nooa.agentdoc)
---

# agentdoc: Beautiful Docs for the LLM

Everything the LLM knows about your API comes from `agentdoc`: `doc(self)` is injected into every prompt, tools are discovered via `doc(self.tool)`, and the CodeAct prefill `pprint()`s your arguments. Two-step model: **`spec()` specifies how a type renders; `doc()` renders the API contract.** `pformat()`/`pprint()` are the value-level printers underneath.

```python
from nooa.agentdoc import doc, spec, hidden, pformat, pprint
# inside CodeAct-generated code, doc/pprint (and methods/variables) are auto-injected;
# spec/hidden/pformat still need the import above
```

## What good output looks like

```python
class Priority(Enum):
    LOW = 1
    HIGH = 2

class Address(BaseModel):
    """A postal address."""
    street: str = "1 Main St"
    city: str = "Anytown"

class User(BaseModel):
    """A user of the system."""
    name: str = Field(description="Full legal name")
    age: int = Field(default=0, ge=0, le=150, description="Age in years")
    role: Literal["admin", "guest"] = "guest"
    priority: Priority = Priority.LOW
    address: Address = Address()
    api_key: Annotated[str, hidden] = ""          # never rendered

    def greet(self, loud: bool = False) -> str:
        """Say hello to the user."""
```

`doc(User)` produces:

```
class User(BaseModel):
    """A user of the system."""

    name: str  # Full legal name
    age: int = 0  # Age in years [≥0, ≤150]
    role: Literal[admin, guest] = 'guest'
    priority: Priority = <Priority.LOW: 1>
    address: Address = Address(street='1 Main St', city='Anytown')  # A postal address.

    def greet(self, loud: bool = False) -> str:
        """Say hello to the user."""
## Referenced Types
class Address(BaseModel):
    """A postal address."""
    ...
class Priority(Enum):
    LOW = 1
    HIGH = 2
```

Note what the renderer did: `Field(description=...)` became the `# comment`; Pydantic constraints became `[≥0, ≤150]`; `Address`'s own class docstring became the field comment (no description needed); types referenced in fields/signatures were expanded once under `## Referenced Types`; the `hidden` field vanished. `doc(instance)` renders the identical structure with **current values** in place of defaults.

## Rules for types that render well

1. **Describe fields where the schema lives**: `Field(description=...)`, `Annotated[str, "plain string"]`, or `Annotated[str, spec(description=...)]` all render as the trailing `# comment` (priority: Annotated string > `spec` > Pydantic Field). A parameter's `Annotated` description is also auto-synthesized into the method's `Args:` section when the docstring lacks one.
2. **Give every class and public method a docstring.** Only the class's *own* docstring renders (inherited ones are dropped). A method with no docstring renders a bare `...` body — indistinguishable from a generation method.
3. **Define referenced types at module level** so they can be discovered and expanded under `## Referenced Types` (shown once, deduped, first-line docstrings). Collapse a heavy type with `@spec(expand=False)` on the class — it stays a one-liner. `__agentdoc_skip__ = True` keeps a type out of Referenced Types entirely (this is how `Skill` avoids self-documenting).
4. **Constrain with types, not prose**: `Literal["a", "b"]`, `ge=/le=`, enums — they render compactly and validate. (`Literal` renders unquoted: `Literal[admin, guest]`.)
5. **Keep defaults short** — long signature defaults render truncated (`first27chars...`; only reprs starting with `<` collapse to `...`); class-valued defaults render as `ClassName` / `ClassName()` markers.
6. **Cap noisy fields per-field**: `Annotated[list, spec(max_length=20)]`, `spec(max_string=200)`, `spec(max_depth=2)` bound how that field's *values* render everywhere (doc, pprint, prefill).
7. **Hide plumbing** so the signal survives: secrets, caches, wiring (below). Everything visible costs tokens in *every* prompt.

Dataclasses, NamedTuples, TypedDicts, attrs, enums, and plain classes (annotations + `__init__` assignments + properties) all render with the right header (`@dataclass`, `class X(TypedDict):`, ...). Fields merge across the full MRO, leaf-wins; methods appear in source order.

## `spec()` — one callable, four forms

| Form | Example | Effect |
|---|---|---|
| Field annotation | `Annotated[str, spec(description="...", max_string=200)]` | description + per-field render caps |
| Method/class decorator | `@spec(hidden=False)` on `_private_method` / `@spec(expand=False)` on a class | opt a private method back into docs / collapse a type |
| Third-party patch | `spec(pd.DataFrame, "attrs", hidden=True)` | set metadata on classes you don't own |
| Instance-level | `spec(self, "context", hidden=False)` in `__init__` | per-instance visibility — the highest-priority rule (how agents expose `context`/`events`) |

Accepted kwargs: `hidden`, `description`, `expand`, `concise`, `max_length`, `max_string`, `max_depth`. For fully custom rendering of a type you don't control, register an extractor: `spec.define_doc(SomeType)`.

## `hidden` — one object, three roles

```python
from nooa.agentdoc import hidden

@hidden                              # 1. decorator: hide a method/property
def rebuild_index(self): ...

api_key: Annotated[str, hidden] = "" # 2. Annotated marker: hide a field

with hidden:                         # 3. module-level context manager: hide imports/names
    import secrets
```

- `_private` and dunder names are hidden by default; re-include a `_private` **method** with `@spec(hidden=False)`. A `_private` **field** cannot be re-included — rename it public instead.
- Pydantic `Field(repr=False)` fields are also skipped by both `doc()` and `pformat()` (`exclude=True` is NOT honored — it only affects serialization).
- There is no `visible` marker in agentdoc — un-hiding is always `spec(..., hidden=False)` (the top-level `nooa.visible` is a no-op kept for backward compat).

## `doc()` in prompts and generated code

```python
doc(obj)                          # full contract; instances show live values
doc(obj, concise=True)            # first-line docstrings only
doc(a, b, c)                      # multiple objects, one deduped Referenced Types section
doc(obj, inline_depth=2)          # expand referenced types transitively
```

Progressive-disclosure patterns: `{doc(self)}` in a docstring for the agent's full API; `doc(item)` in CodeAct when a factory returns `Any`; pin `self.context["tool_guide"] = doc(self.tool)` for repeated use. Agents also honor `__type_info__`/`__instance_values__` — that's why `doc(agent_instance)` shows user state but not framework internals.

`methods(obj)` and `variables(obj)` (auto-injected in CodeAct) are lighter-weight listings — note they hide **all** `_`-prefixed names regardless of `@spec(hidden=False)`, unlike `doc()`.

## `pformat` / `pprint` — value rendering

```python
pformat(obj, max_length=None, max_string=None, max_depth=None, expand_all=False,
        concise=False, instance_mode="repr", unquote_strings=False)   # bare pformat is unlimited;
        # the 50/500/4-style caps are DocConfig/TruncationConfig defaults applied by callers
pprint(obj, ...)                       # same, to stdout (Rich drop-in: console/indent_guides accepted+ignored)
truncating_pformat(obj, max_chars=N)   # hard char cap via TruncatingStringIO
```

Truncation is always *visible* — a marker means truncated, a bare literal is complete:

```
list(len=100, [:3]=[0, 1, 2], [-3:]=[97, 98, 99])
str(len=540, [:20]='Lorem ipsum dolor si', [-20:]='psum dolor sit amet ')
dict(len=100, items={0: 0, 1: 1, ..., 99: 99})
{'a': {'b': {dict: 1 items}}}                    # max_depth cutoff
```

- Classes with their own `__repr__` (pandas, numpy) are trusted and rendered via their repr (truncated head/tail if huge).
- Implement `__instance_values__(self) -> dict[str, Any]` (the `SupportsInstanceValues` protocol) to control exactly which fields an instance renders — omitted keys are hidden.
- These same knobs are what `TruncationConfig`'s `event_format`/`prefill_format` splat into `pformat` framework-wide — see `nooa-codeact-advanced`.

## Gotchas

- **`__init__`-declared fields require real source files.** They're discovered by AST-parsing `inspect.getsource`, which fails silently for classes defined in a REPL/`exec` — fields vanish from docs. Keep documented classes in modules.
- `doc()` on a class defined with no docstring and no annotations renders nearly nothing — the renderer can only surface what you declare.
- `format_type` strips quotes and module paths (`Literal[admin]`, `X | Y`) — cosmetic, don't parse doc output.
- Field-count/method-count overflow renders `...` / `# ... +N more methods` under `max_length` — raise it or split the class.
- Pydantic BaseModel's framework methods (`model_dump`, ...) are blocklisted from docs automatically.

## Related skills

- `nooa-agent-authoring` — visibility rules in the agent context (`doc(self)`, exec_globals).
- `nooa-codeact-advanced` — how prefill and truncation configs drive `pprint` of your parameters.
- `nooa-tools-and-skills` — skill docstring conventions (`doc(self.skill)` is the usage guide).
