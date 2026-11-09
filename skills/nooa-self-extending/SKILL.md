---
name: nooa-self-extending
description: Let NOOA agents extend themselves — persistent skills the agent writes — code packages it hot-reloads or SKILL.md text instructions (SkillWriting / self.skill_writing, LibraryManager), in-cell helper functions and standalone @strategy sub-calls (MethodWriting), and @slash_command user actions. Use when an agent should accumulate reusable code across sessions, define its own helpers or LLM-powered sub-functions at runtime, or ship user-typed /commands from a skill.
compatibility: nooa package
---

# Self-Extending Agents

Three escalating levels of agent-authored code, from ephemeral to persistent:

| Level | Mechanism | Lifetime | Give the agent |
|---|---|---|---|
| In-cell helpers | plain `def` in a CodeAct cell | REPL locals, gone after the call | `MethodWriting` skill (guidance only) |
| LLM-powered sub-calls | standalone `@strategy` functions in a cell | same cell scope | `MethodWriting` |
| Persistent skills | Python packages or SKILL.md text under `.nooa/skills/` | on disk, hot-reloaded, across sessions | `SkillWriting` (`self.skill_writing`) |

## In-cell helpers and standalone sub-calls (`MethodWriting`)

`MethodWriting` (`nooa.tools.method_writing_lib`) is a guidance `Skill` — attach it and the LLM learns to:

- define plain helpers at the top of a cell for deterministic logic, then use them;
- define **standalone generation functions** for per-item LLM sub-tasks and fan them out:

```python
@strategy(PredictStrategy())
async def detect_language(message: str) -> str:
    """Return the ISO 639-1 code for the message."""
    ...

codes = await asyncio.gather(*(detect_language(m) for m in messages))
return_result(codes)
```

Standalone functions (`nooa.standalone`) are `@strategy`-decorated async functions **without `self`** — each call runs on a fresh agent stub (no shared state, history discarded after the call; context blocks via the decorator's `ScopedContext`, `llm=` via the decorator or inherited from the calling context). They work in module code too, not just generated cells — the lightest way to get an LLM-powered function without defining an Agent class.

Helpers defined in cells persist as REPL locals for the rest of the method call but are never attached to the agent (attaching callables to `self` is validator-rejected — see `nooa-codeact-advanced`).

## Persistent skills (`SkillWriting` / `self.skill_writing`)

`SkillWriting` (`nooa.tools.library_writing_lib`) is how an agent makes the skills later sessions get, in a managed directory: `<workspace>/.nooa/skills` for Atom (an older `.nooa/libs` still loads; a name in `.nooa/skills` wins; new skills always go to `.nooa/skills`). A skill is either a code skill (a real Python package) or a text skill (a `SKILL.md` with front matter `name`/`description` and instructions). Both load in a later session as `local.<name>` (`self.<name>`, found by `self.skills.search`). Requires the shell skill (`requires = ("nemo.shell",)`) since file I/O goes through `self.shell`.

The lifecycle the agent follows (documented to it via `doc(self.skill_writing)`):

```python
await self.skill_writing.create("release", "How to cut a release.", text=steps)  # text skill: SKILL.md, active at once
await self.skill_writing.create("stats", "Statistical utilities.")      # 1. scaffold package (pyproject + __init__)
await self.shell.write_file(self.skill_writing.path("stats", "stats.py"), source)  # 2. write code
await self.skill_writing.reload("stats")                                # 3. lint + hot-reload → self.stats exists
await self.shell.replace(self.skill_writing.path("stats", "stats.py"), old, new)  # 4. edit
await self.skill_writing.reload("stats")                                #    ...and reload again
await self.skill_writing.run_tests("stats")                             # 5. pytest on the skill's tests/
await self.skill_writing.list(); await self.skill_writing.repo_tree()   # discovery
```

- **Skill contract:** each code skill's `__init__.py` exports a `Skill` subclass — the registry auto-attaches it as `self.<lib_name>` with full `doc()` discovery. Libraries without one get a `Skill(module)` fallback (docstring + attributes).
- **Linting on reload** (`LintReport`): hard errors E001 (forbidden builtins) and E003 (star imports) block the write; E002 warns when an import isn't in the agent's allowed set. Report says written/loaded status explicitly.
- Library code is plain Python — no agent `self`, no `...` bodies, no async requirement. Attachment goes through `SkillRegistry.discover_libs(path)` + `activate(["local.*"])` when the agent has a skill registry, else `LibraryManager.install(agent, libs_dir=...)`.
- **Reload ≠ relearn:** after `self.skill_writing.reload(...)`, the new API is live on `self.<lib>`; guide agents to re-read `doc(self.<lib>)` rather than assume.

`LibraryManager` (`nooa.library_manager`) is the low-level loader you can also use directly from Python: `LibraryManager.install(agent, libs_dir=Path("libs"))` scans for subdirectories with a `pyproject.toml`, imports each, attaches as `agent.<lib_name>`; `mgr.reload()` hot-reloads all; `LibraryManager.discover(path)` lists without loading. Modules are attached as attributes only — never injected into exec_globals.

## Slash commands (`@slash_command`)

Skills can ship user-typed `/commands` (dispatch: `nooa.slash_dispatch`; hosts driving `InteractiveAgent` surface them via the `slash_commands` queue):

```python
from nooa.skill import Skill, slash_command

class MySkill(Skill):
    @slash_command("check", argument_hint="<target>", completions=("deps", "lint", "tests"))
    async def check_command(self, args: str) -> str:
        """Run checks on the project."""
        ...do work...
        return "Ran lint. 3 issues found.\n\nFix with `await self.shell.run(...)` or ask the user."
```

- **The return value becomes a user message** — it's a prompt for the agent, not display output. Frame it: what happened, what to do next.
- `completions=(...)` powers tab-completion of subcommands; `argument_hint` shows in `/help`; `user_only=True` blocks LLM invocation (destructive ops).
- Commands are discovered automatically when the skill is activated or hot-reloaded — no registration step. That includes libraries the agent itself writes: an agent can author a library that ships its own slash commands.

## Pitfalls

- `self.skill_writing.create()` requires a meaningful description — it becomes the docstring (or SKILL.md description) the LLM later reads.
- Enforce "run `run_tests()` before claiming done" in the orchestrator, not just the prompt.
- Write a skill for logic or advice worth naming and reusing; inline cell code for one-offs — skills are forever, and a junk drawer of one-off skills pollutes `doc(self)` every turn.
- Hot-reload replaces the module object; stale references held in REPL locals from earlier cells keep pointing at the old code until re-fetched via `self.<lib>`.

## Related skills

- `nooa-tools-and-skills` — the Skill/TextSkill/SkillRegistry model these libraries plug into.
- `nooa-codeact-advanced` — the validator rules that shape what generated code may define.
- `nooa-agent-authoring` — standalone `@strategy` functions share generation-method semantics (docstring prompt, return-type contract).
