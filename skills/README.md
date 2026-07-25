# Coding-agent skills for authoring NVIDIA OO Agents

Portable `SKILL.md` bundles for coding agents (Claude Code, Cursor, Codex, or any Agent Skills host) that help developers **author application agents with** the NVIDIA OO Agents framework, capture traces, and debug runs with the trace viewer and trace explorer.

These are instructions *for coding agents about the framework* — not `nooa.Skill` runtime skills (though the file format is compatible with `TextSkill`, see Validate below).

## Skills

| Skill | Use for |
|---|---|
| [`nooa-agent-authoring`](nooa-agent-authoring/SKILL.md) | Core authoring: Agent subclasses, generation methods (`...`), docstring prompts, structured output, strategies (CodeAct/Predict), visibility, orchestrators, subagents, LLM config, prompt debugging |
| [`nooa-codeact-advanced`](nooa-codeact-advanced/SKILL.md) | Advanced strategy tuning: prefill (custom/disable/pre-ellipsis), loop guards, truncation tuning, code restrictions, execution internals, PredictConfig |
| [`nooa-agentdoc`](nooa-agentdoc/SKILL.md) | Making types render beautiful docs for the LLM: `doc()`, `spec()`, `hidden`, `Annotated` descriptions, `pformat`/`pprint` tuning |
| [`nooa-context-and-state`](nooa-context-and-state/SKILL.md) | Context blocks, event history and `EventQuery`, history summarization, persistence and memory |
| [`nooa-tools-and-skills`](nooa-tools-and-skills/SKILL.md) | Methods as tools, built-in tools (Bash/File/Todo), MCP integration, agent skills (`Skill`/`TextSkill`), multimodal media |
| [`nooa-channels`](nooa-channels/SKILL.md) | Reactive input: Channel/QueueManager, race() dispatch loops, spawn() background jobs, monitor/cron/tail producers |
| [`nooa-self-extending`](nooa-self-extending/SKILL.md) | Agent-authored code: persistent skill libraries (self.libs), in-cell helpers and standalone @strategy sub-calls, @slash_command |
| [`nooa-middleware-hooks`](nooa-middleware-hooks/SKILL.md) | Intercepting execution: middleware (`intercept()` guardrails/transforms/blocking), event observers (`on()`), InstrumentationHooks protocol |
| [`nooa-capturing-traces`](nooa-capturing-traces/SKILL.md) | Capturing traces: auto-tracing, `enable_tracing` + exporters (jsonl/otlp/langfuse/journal), `@no_trace`, span model, env vars |
| [`nooa-trace-viewer`](nooa-trace-viewer/SKILL.md) | Running and using the trace viewer (`nooa start-dev`): UI, import/export, REST API |
| [`nooa-trace-explorer`](nooa-trace-explorer/SKILL.md) | Programmatic trace analysis: `trace-explorer` CLI, `TraceExplorer` library, thin client, experiment-level debugging |

All content was verified against `src/nooa` at the time of writing; where the skills contradict older docs (e.g. `SkillManager`, `from agentdoc import ...`, `enable_tracing(trace_dir=...)`, "private methods aren't traced"), the skills reflect the code.

## Install

Copy or symlink skill directories into the location your coding agent reads:

```bash
# Claude Code — all skills, user-global
mkdir -p ~/.claude/skills
cp -R skills/nooa-* ~/.claude/skills/

# Or one skill into a project
mkdir -p .claude/skills
cp -R skills/nooa-agent-authoring .claude/skills/
```

Other hosts read from different directories (e.g. `.codex/skills/`, `.agents/skills/`).

## Validate

The bundles are loadable as framework `TextSkill`s, which validates their frontmatter:

```bash
uv run python - <<'PY'
from pathlib import Path
from nooa import TextSkill
for path in sorted(Path("skills").glob("*/SKILL.md")):
    s = TextSkill(path=path.parent)
    print(f"{s.id}: {s.description[:88]}")
PY
```
