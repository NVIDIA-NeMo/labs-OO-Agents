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

Installed `nooa.skills` entry points are loaded into the shared skill registry
but remain opt-in. The agent can activate a relevant skill with
`self.skills.activate(["name"])`. Stdio MCP servers supplied by an ACP client
are registered and activated as `mcp.<name>` skills for that session.

The current adapter accepts text and resource-link prompts and stdio MCP
servers. Additional workspace directories, remote MCP transports, images, and
embedded resources are not advertised yet.
