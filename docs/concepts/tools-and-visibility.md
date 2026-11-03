# Tools and visibility

In NOOA, a tool is usually a visible Python capability on the agent. Regular
methods, helper objects, skills, and MCP clients all appear through `self`
rather than through a separate tool-registration layer.

## Methods are the simplest tools

```python
class InventoryAgent(Agent, llm=llm):
    def stock(self, sku: str) -> int:
        """Return the current stock for a SKU."""
        return INVENTORY.get(sku, 0)

    async def assess(self, requested: list[str]) -> str:
        """Assess availability. Use self.stock() for exact quantities."""
        ...
```

The method name, signature, type annotations, and docstring form the capability
description. Keeping that description next to the implementation prevents a
separate schema from drifting.

## Attach stateful tools per instance

```python
from nooa.tools import ShellTools, TodoManager


class DeveloperAgent(Agent, llm=llm):
    def __init__(self, repo: str, **kwargs):
        super().__init__(**kwargs)
        self.shell = ShellTools(cwd=repo)
        self.todos = TodoManager()

    async def implement(self, request: str) -> str:
        """Implement and verify the requested change."""
        ...
```

NOOA reads ordinary `self.<field> = DirectConstructor(...)` assignments from
`__init__`, so `shell` and `todos` become part of the documented agent API
without duplicate class-level declarations. Constructing them in `__init__`
gives every agent its own shell session, working directory, locks, and task
state.

When the assigned expression does not reveal the type, annotate the assignment
in place:

```python
self.wiki: MCPTool = MCPManager.create_from_server("wiki")
```

This is needed for factories, injected parameters, and other expressions that
cannot be inferred from their syntax. A class-level annotation is also valid
and remains the most robust option when source inspection is unavailable, such
as some dynamic or interactive definitions.

Do not put `shell = ShellTools()` on the class. Python would construct it once
and share it across all instances.

## Visibility follows Python-style rules

The model-visible API is public by default:

| Surface | Default |
|---|---|
| Public methods and annotated fields | Visible |
| `_private` methods and fields | Hidden |
| Module-level imports and definitions | Available to CodeAct unless hidden |
| `context` and `events` APIs | Present but hidden until explicitly exposed |

Hide plumbing and secrets explicitly:

```python
from typing import Annotated

from nooa import hidden


class SearchAgent(Agent, llm=llm):
    api_key: Annotated[str, hidden] = ""

    @hidden
    async def run(self, query: str) -> list[str]:
        """Python entry point; hidden to prevent recursive model calls."""
        return await self.search(query)

    async def search(self, query: str) -> list[str]:
        """Search the configured source."""
        ...
```

`@hidden` affects model visibility, not whether Python can call the method or
whether tracing can record it.

## Progressive disclosure with `doc()`

NOOA renders the agent's visible API with `doc(type(self))`. CodeAct also has a
`doc(obj)` helper for discovering an unfamiliar object when it becomes
relevant:

```python
item = self.lookup(item_id)
print(doc(item))
```

This keeps the initial prompt bounded. A model can inspect a returned object or
tool on demand instead of receiving every nested API up front.

Well-typed APIs render best. Prefer named types, field descriptions, compact
defaults, and public method docstrings. Types referenced by a public signature
should be defined or imported at module level.

## Expose the smallest useful surface

Public visibility is convenient, but every visible name costs attention and can
be called by generated code. Prefer a narrow deterministic wrapper over a raw
client with dangerous or irrelevant operations.

For example, expose `lookup_customer(customer_id)` instead of an entire database
connection. The wrapper becomes a stable capability, a validation point, and a
clean error boundary.

## Common mistakes

- Constructing a stateful tool on the class and unintentionally sharing it.
- Expecting NOOA to infer a tool type from a factory or injected value without
  an inline or class-level annotation.
- Exposing credentials, storage managers, or unrestricted clients.
- Using a lambda or dynamically attached function where a documented class
  method would be discoverable.
- Hiding a method the model must call, or exposing an outer method it might call
  recursively.

## Continue

- [Prompts and context](prompts-and-context.md)
- [Safety](safety.md)
- Runnable examples: [methods as tools](../../examples/quickstart/03_codeact_tools.py),
  [progressive disclosure](../../examples/quickstart/05_progressive_disclosure.py),
  [skills](../../examples/quickstart/10_skills.py), and
  [MCP](../../examples/quickstart/11_mcp.py)
