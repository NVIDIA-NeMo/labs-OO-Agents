---
name: nooa-tools-and-skills
description: Give a NVIDIA OO agent capabilities — methods as tools, built-in tools (ShellTools, TodoManager), MCP servers, agent skills (Skill/TextSkill/SkillRegistry), and multimodal media. Use when adding tools or external integrations to an agent, wiring MCP, or packaging reusable guidance as an agent skill.
compatibility: nooa package; [mcp] extra for MCP
---

# Tools, Skills, and Integrations

## Methods are the tools

There is no tool-registration abstraction: **any visible method or attribute on `self` is callable from CodeAct-generated code**, with `doc(self)` telling the LLM what exists. Mix deterministic helpers (SW1) with generation methods (SW3):

```python
class InventoryAgent(Agent, llm=llm):
    """Checks inventory using deterministic helpers."""

    def get_stock(self, item: str) -> int:          # SW1: a "tool"
        return self.inventory.get(item, {}).get("stock", 0)

    async def can_fulfill(self, items: list[str], budget: float) -> Result:
        """Check whether the order fits the budget. Use self.get_stock()."""
        ...                                          # SW3: LLM orchestrates the helpers
```

Design rubric: exact semantics (matching rules, parsing, arithmetic) → deterministic helper; fuzzy judgment → generation method. Give helpers precise names, type hints, and docstrings — those are their tool schema.

## Built-in tools

```python
from nooa.tools import ShellTools, TodoManager

class DevAgent(Agent, llm=llm):
    def __init__(self):
        super().__init__()
        self.shell = ShellTools(cwd=".")          # persistent shell + file ops
        self.todos = TodoManager()                # t = self.todos.add("step", deps=[...]); self.todos.done(t.id)
```

`ShellTools` is the one shell/file tool (it replaced the removed `BashTool`/`FileTool`). One persistent bash session — `cd`, `export`, and cwd survive across calls — and four methods:

```python
r = await shell.run("grep -rn 'def foo' src/")   # ShellResult: str subclass; .stdout/.stderr/.returncode/.success
await shell.run("python -", stdin=script)        # pass payloads via stdin= instead of quoting heredocs
region = await shell.read("f.py", lines=(10, 25))  # view a file region -> Match
await shell.replace(r.matches[0], new_code)      # edit at a Match anchor (pure grep/rg results carry .matches)
await shell.write_file("out.py", content)        # create/overwrite
```

Tools attached as public attributes are automatically visible; the LLM discovers their APIs via `doc(self.shell)`.

For an object the LLM might misuse, prefer wrapping the dangerous surface in a small deterministic method over exposing the raw object.

## Agent skills (`Skill` / `TextSkill` / `SkillRegistry`)

Skills inject curated guidance + optional helper APIs into an agent. Three forms:

```python
from nooa import Skill, TextSkill

class MyAgent(Agent, llm=llm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1. Subclass with a docstring-as-usage-guide (methods become the API)
        self.git = GitWorkflow()                        # class GitWorkflow(Skill): """guide..."""
        # 2. Wrap a third-party library for discovery
        self.pd = Skill(pd)                             # LLM uses self.pd.read_csv(...)
        # 3. A SKILL.md directory on disk
        self.frontend = TextSkill(path=Path("skills/frontend-design"))
```

- Agents see a `# Skills` block (one-liner per skill) in their execution context and call `doc(self.<skill>)` for the full guide.
- `TextSkill` exposes `read_file(path)` and `await run_script(name, *args)` for files/scripts bundled with the SKILL.md.
- SKILL.md frontmatter: required `name`, `description`; optional `compatibility`, `metadata`, `user-invocable`, `allowed-tools` — Claude-Code-compatible format.
- Bulk discovery/activation is `SkillRegistry(agent)` + `discover_skills_dirs([...])` / `activate([...])`. **`SkillManager` does not exist** — `README`/`examples/quickstart/10_skills.py` referencing `SkillManager.install(...)` are stale.
- `@slash_command` on a `Skill` method marks it user-invocable via a host that reads the agent's `slash_commands` queue (see `InteractiveAgent`).
- Docstring conventions: see `skills/nooa-agent-authoring/SKILL.md` and `AGENTS.md`.

## MCP servers

Core library keeps MCP optional: `uv sync --extra mcp` (or `uv add 'nooa[mcp]'`).

```python
from nooa.mcp import MCPManager

class ConfluenceAgent(Agent, llm=llm):
    confluence = MCPManager.create_from_server("my-mcp-server")   # from .mcp.json
    # or inline, no .mcp.json:
    # confluence = MCPManager.create_from_server("name", url="https://.../mcp",
    #                                            transport="streamable-http", headers={...})

    async def respond(self, prompt: str) -> str:
        """Answer using the Confluence tool."""
        ...
```

MCP tools appear alongside regular methods — the LLM calls them like any other attribute. Servers are configured in `.mcp.json` (VS Code/Claude Code format). Transports: stdio, SSE, streamable-http; OAuth supported (`nooa.mcp.OAuthConfig`).

`${VAR}` placeholders in `.mcp.json` and inline `servers` configuration remain
literal. If a server needs a secret, resolve it in trusted caller code and pass
the resulting URL, header, argument, or environment value directly to
`MCPManager.create_from_server()`.

## Multimodal media

```python
from nooa import Image, Audio, Video, File

class MediaAgent(Agent, llm=llm):
    async def describe(self, image: Image) -> str:
        """Describe what you see."""
        ...

img = Image.from_file("photo.png")     # also .from_bytes(media_type=...), .from_url(...)
await MediaAgent().describe(img)
```

Only works with multimodal-capable models — text-only models error when handed media. See `examples/quickstart/12_multimodal.py`.

## Pitfalls

- A tool attribute assigned as a bare function/lambda isn't introspectable — use class methods or proper objects.
- Skills/tools on `self` are per-instance; attach them in `__init__` (after `super().__init__()`).
- Hide internal-only tools with `Annotated[T, hidden]` so they don't pollute every prompt.
- External API fragility: wrap flaky calls in a small deterministic method with clear errors, so CodeAct sees a clean failure it can react to.

## Related skills

- `nooa-agent-authoring` — visibility rules and method design that make tools discoverable.
- `nooa-capturing-traces` — every tool invocation becomes a `tool_execution.*` span you can inspect.
