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
export NOOA_MODEL=nvidia_nim/nvidia/nemotron-3-super-120b-a12b
export NVIDIA_API_KEY=nvapi-...
uvx nooa-acp
```

There is no default model. Pass `--model` or set `NOOA_MODEL`; the command
exits with a usage error if neither is set.

From this repository, use the workspace package as the client command:

```bash
uv run --project "$PWD" --package nooa-acp -- nooa-acp
```

## Configuring Zed

Zed launches ACP agents as "external agents". Add NOOA to `settings.json`:

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

Pick NOOA from the `+` menu in the agent panel. Zed runs the command with the
worktree as its working directory, so repository instructions and project
sessions resolve against the open project.

Credentials have to go in `env` here rather than Zed's own settings: the agent
is a separate process and inherits only what Zed passes it. Use a secret
manager wrapper as the `command` if you would rather not put a key in
`settings.json`.

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

ACP uses standard input and output for JSON-RPC. Diagnostics are written to
standard error. The agent can execute generated Python and shell commands, so
use an OS-level sandbox for untrusted tasks. Generated code shares the agent's
process environment, including model credentials; launch it with only the
credentials and network access that the session may use.
Cancellation stops local work immediately. An in-flight provider request may
finish in the background when its client does not support transport-level aborts.

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

Installed `nooa.skills` entry points are loaded into the shared skill registry
but remain opt-in. The agent can activate a relevant skill with
`self.skills.activate(["name"])`. Stdio MCP servers supplied by an ACP client
are registered and activated as `mcp.<name>` skills for that session.

The current adapter accepts text and resource-link prompts and stdio MCP
servers. Additional workspace directories, remote MCP transports, images, and
embedded resources are not advertised yet.
