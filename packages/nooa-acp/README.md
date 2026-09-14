# nooa-acp

**Run the NOOA coding agent inside your editor.** `nooa-acp` is an
[Agent Client Protocol](https://agentclientprotocol.com) server, so any
ACP-speaking client can drive the shared coding agent: CodeAct, repository
tools, a persistent shell, installed skills, workspace
slash commands and durable sessions, with file edits and terminal commands
surfaced as structured activity.

The shared factory in `nooa_cli.coding.factory` constructs an
`ExperimentalCodingAgent` by default. Its single `python_cell` tool executes
Python that can call the agent's repository and shell tools. Pass
`--legacy-agent` to use the standard multi-tool `CodingAgent`. Repository instructions (`AGENTS.md`),
summarization, titles, skills, and turn dispatch are shared Python APIs in
`nooa_cli.coding` and `nooa_cli.interactive`; they do not import a terminal UI.

This is new and we would like it exercised. If something breaks, please say so.

## Install

```bash
uv add nooa-acp                 # or: uv add "nooa[acp]"
```

There is no default model. Set `NOOA_MODEL` or pass `--model`, or the command
exits with a usage error.

## Quick start: Zed

Zed launches ACP agents as "external agents". Add NOOA to `settings.json`
(`cmd-,`):

```json
{
  "agent_servers": {
    "NOOA": {
      "type": "custom",
      "command": "uvx",
      "args": ["nooa-acp"],
      "env": {
        "NOOA_MODEL": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        "NVIDIA_API_KEY": "nvapi-..."
      }
    }
  }
}
```

Open a repository, then pick **NOOA** from the `+` menu in the agent panel. Zed
runs the command with your worktree as its working directory, so repository
instructions, project skills and sessions resolve against the open project.

Credentials go in `env` here rather than in Zed's own settings: the agent is a
separate process and inherits only what Zed passes it. Use a secret-manager
wrapper as the `command` if you would rather not put a key in `settings.json`.

From a checkout of this repository, point the client at the workspace package
instead:

```bash
uv run --project "$PWD" --package nooa-acp -- nooa-acp
```

### MCP servers do not carry over from Zed

**Remote MCP servers you authenticated inside Zed are not usable from an ACP
agent.** Zed holds those OAuth tokens itself and does not pass them down, so a
server showing a green indicator in Zed's own UI arrives at the agent either
with no tools at all or with nothing but its `authenticate` /
`__complete_authentication` stubs. Local stdio MCP servers are unaffected.

This is a known Zed limitation, tracked in
[zed-industries/zed#54410](https://github.com/zed-industries/zed/issues/54410)
(open, labelled `area:ai/mcp` + `area:ai/acp`). A maintainer has said the
plumbing largely exists and the work is queued, but as of this writing it is
unresolved.

Configure the MCP server directly for NOOA instead — through NOOA's own
`.mcp.json` — and it works normally, because the agent then owns the
connection and its credentials rather than borrowing Zed's.

## Launching the server yourself

```bash
nooa-acp --model nvidia_nim/nvidia/nemotron-3-super-120b-a12b
```

This is a JSON-RPC server, not an interactive program: it speaks ACP on
stdin/stdout and exits when its input closes, so running it in a terminal
without a client does nothing. Launch it this way to wire up an ACP client
other than Zed, or to watch the diagnostics it writes to stderr while a client
drives it. `--model` accepts any LiteLLM model name or configured NOOA alias.

## Opening a repository runs code from it

**Creating a session imports Python from the workspace, before you send a
prompt.** This is deliberate — it is how workspace skills work — but it means
opening a folder is enough to execute code it contains. Treat opening a
repository with NOOA as equivalent to running its build.

Three paths load workspace code at `session/new` and `session/load`:

- **Skill roots.** Every `.py` file under `.agents/skills`, `.cursor/skills`,
  `.claude/skills`, or `.claude/commands` is imported. Module-level code runs
  during import, before anything checks whether the file defines a skill, so the
  contents are irrelevant.
- **Workspace settings.** `<workspace>/.nooa/settings.yaml` and the legacy
  `.nooa/config.toml` may name *additional* skill roots. Those paths are not
  confined to the workspace: a relative path escaping it, an absolute path, or a
  symlink is accepted as written.
- **Libraries.** `<workspace>/.nooa/libs/<package>/` is imported and its
  directory is prepended to `sys.path` for the life of the process. One ACP
  server serves several workspaces, so a package name there can shadow the same
  import for later sessions on other workspaces.

The agent runs as you, in a process holding your model credentials. There is no
consent prompt on these paths.

**Open repositories you would run.** For anything else, use an OS-level sandbox,
or start a separate server per workspace with credentials scoped to that task.

## How it behaves

ACP uses standard input and output for JSON-RPC. Diagnostics are written to
standard error. The agent can execute generated Python and shell commands, so
use an OS-level sandbox for untrusted tasks. Generated code shares the agent's
process environment, including model credentials; launch it with only the
credentials and network access that the session may use.
Cancellation stops cooperative local work immediately. An in-flight provider
request may finish in the background when its client does not support
transport-level aborts. Slash commands run on the agent's event loop so they
have the same semantics as the native TUI and can safely start agent jobs. An
async command is cooperatively cancellable; a synchronous command that blocks
that loop cannot be preempted by the current in-process adapter. The planned
one-process-per-agent boundary is the safe kill mechanism for that case.

## Sessions and skills

Each ACP session has an independent live agent and allows one foreground prompt
at a time. Sessions are stored in `<workspace>/.nooa/sessions`, where the TUI
and ACP adapter can share list and replay metadata. These files are inside the
workspace trust boundary: a repository can supply session records that appear
in `session/list` and are replayed as conversation history by `session/load`.
Open only repositories whose code and conversation history you trust. The
adapter also advertises session close; closing a live session preserves its
durable history.

Resume listings omit empty sessions and sessions currently owned by a live
agent. SQLite ownership also uses a sibling `.active` directory to prevent
another process from opening the same session. Normal close releases it; a
crash may leave a claim behind. Recovery requires removing the stale claim
after confirming that its owning process is no longer running.

The current stdio adapter hosts those live agents in its own process. The core
`nooa.sessions` runtime owns turn serialization, cancellation-safe cleanup, and
registration until resources are released. The ACP adapter owns the agent and
event-bridge bundle and decides when it is ready for client requests: a loaded
session stays unavailable until transcript replay finishes. These ACP policies
stay outside the core runtime. The adapter can later use handles to an agent
daemon without changing stored sessions or the shared coding agent.

Python skill packages use the interpreter's normal import machinery. Multiple
sessions may use distinct skill package names, but two workspaces must not load
different checkouts under the same top-level Python package name in one ACP
server process. Launch a separate stdio server for those workspaces. A future
one-process-per-agent daemon will make that isolation an OS process boundary.

Installed `nooa.skills` entry points are loaded into the shared skill registry
but remain opt-in. The agent can activate a relevant skill with
`self.skills.activate(["name"])`. Stdio MCP servers supplied by an ACP client
are registered and activated as `mcp.<name>` skills for that session.

Workspace and user skill roots are shared with the terminal host through
layered `settings.yaml`. New configuration should use:

```yaml
coding:
  additional_skills_dirs:
    - ../nemo-oo-skills
```

The existing `tui.additional_skills_dirs` key remains supported during the
migration, as does the older project-local `.nooa/config.toml` key
`[tui].libs_dirs`. Packaged libraries declared through `nooa.skills`, `SKILL.md`
skills, and standalone Python skills are discovered from each configured root.
Loaded `@slash_command` methods are advertised through ACP and matching
`/command arguments` prompts are dispatched through the shared typed command
router. Command discovery is refreshed when loaded skills change.

### Saved workspace preferences

The shared `SessionOptions` model resolves behavioral settings. Terminal
presentation settings are outside this model. Both built-in agents use the
`nooa_cli.coding.agent:CodingAgent` key for memory and reflection preferences;
historical TUI agent keys are normalized when settings are read. Historical
`TUIAgent` memory owners are migrated to `CodingAgent`, retaining session
suffixes and archived records.

The agent's `self.workspace_settings` skill exposes named operations for
remembering and forgetting skills and MCP definitions, configuring memory and
reflection, and saving a default model. These write
`<workspace>/.nooa/settings.yaml` and affect future agents in that workspace.
Skill and memory changes also apply to the agent making the change. Other live
agents keep their current state. Ordinary `self.skills.load()` and
`self.skills.activate()` remain local to the live agent.

Memory requires the optional `nooa[memory]` package. `/memory local` selects a
session sidecar database; `/memory on` selects the workspace memory store.
`/reflection on` enables idle reflection once memory is attached. These commands
and `/skills` are advertised through ACP and use the same controls as the
workspace settings skill.

Remembering an MCP definition preserves its exact-configuration approval
requirement. The shared startup helper reconnects remembered servers; an
unapproved or unavailable server produces a warning while the session opens.
Client-forwarded servers are session inputs and are not automatically saved as
workspace defaults. ACP still requires an explicit launch model (`--model` or
`NOOA_MODEL`), which takes precedence over a saved default.

### Acceptance tests without a terminal UI

From the repository root:

```bash
uv run pytest packages/nooa-acp/tests/test_shared_sessions.py
```

These tests run real agents with scripted LLM responses through the shared
Python dispatcher and the ACP adapter. They hand sessions in both directions,
checking persistent variables, Todos, titles, active skills, provider reasoning
retention and cache boundaries. They also exercise workspace preferences and
MCP subprocesses on session creation, resume, and saved reconnection. Testing a
particular ACP client's display and session picker still requires that client.

Interactive turn results now use `DONE`, `NEED_INPUT`, and `WAIT`. Custom agents
that returned the old `GET_USER_INPUT` value must return `NEED_INPUT` instead.
Interactive agents no longer attach the web publishing skill automatically.

The current adapter accepts text and resource-link prompts plus stdio, HTTP,
and SSE MCP servers forwarded by an ACP client. ACP-transport MCP proxies,
additional workspace directories, images, and embedded resources are not
advertised yet. An unavailable, duplicate, or unsupported MCP server is skipped
with a session warning so it cannot prevent a new or restored NOOA session from
opening.
