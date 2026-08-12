# nooa-acp

ACP adapter for running the NOOA coding agent from compatible clients.

The adapter hosts the same `nooa_cli.coding.CodingAgent` used by the native
terminal path. Repository instructions (`AGENTS.md`), coding tools,
summarization, installed `nooa.skills` entry points, and semantic file and
terminal activity therefore do not have separate ACP implementations.

## Connect an ACP client

Set the model and its provider credentials, then configure the client to launch
this command with the repository as its working directory:

```bash
export NVIDIA_API_KEY=nvapi-...
uvx nooa-acp
```

From this repository, use the workspace package as the client command:

```bash
uv run --project "$PWD" --package nooa-acp -- nooa-acp
```

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

## Standalone server

```bash
nooa-acp --model nvidia_nim/nvidia/nemotron-3-super-120b-a12b
```

The process waits for an ACP client on standard input. `--model` accepts any
LiteLLM model name or configured NOOA model alias.

## Sessions and skills

Each ACP session has an independent live agent and allows one foreground prompt
at a time. Sessions are stored in `<workspace>/.nooa/sessions`, where the TUI
and ACP adapter can share list and replay metadata. The adapter advertises ACP
session list, load, and close capabilities; closing a live session preserves
its durable history.

The current stdio adapter hosts those live agents in its own process. That is
an adapter-private implementation detail rather than part of the durable
session API: the live-session registry is isolated inside `nooa-acp` so it can
later be replaced by handles to an agent daemon without changing stored
sessions, the shared coding agent, or the ACP protocol surface.

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

The current adapter accepts text and resource-link prompts plus stdio, HTTP,
and SSE MCP servers forwarded by an ACP client. ACP-transport MCP proxies,
additional workspace directories, images, and embedded resources are not
advertised yet.
