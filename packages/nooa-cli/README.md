# nooa-cli

CLI for [nemo-oo-agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Ships the `nooa` command, including the native coding-agent TUI.

## Install

```bash
uv add nooa-cli

# ...with numpy/pandas/plotly/scipy/sklearn pre-loaded into the LLM REPL
uv add "nooa-cli[datascience]"
```

`nooa-cli` automatically pulls in matching `nemo-oo-agents` (the core framework). The `[datascience]` extra adds libraries the LLM can use in REPL-generated code.

## Usage

```bash
nooa --help
nooa start-dev            # launch the trace viewer
nooa eval ...             # eval pipeline runner
nooa traces ...           # inspect/manage trace files
nooa tui                  # interactive coding agent
```

Install the separate `nooa-acp` package to add the `nooa acp` plugin command and
run the NOOA coding agent from an ACP-compatible client:

```bash
uv add nooa-acp
export NOOA_MODEL=nvidia_nim/nvidia/nemotron-3-super-120b-a12b
export NVIDIA_API_KEY=nvapi-...
uv run nooa-acp
```

## TUI configuration

### Connect a model

Start the TUI and run `/connect` — it guides you through picking a model and
stores your credentials for you.

```bash
nooa tui
```

```text
/connect https://api.anthropic.com          # Anthropic (Claude)
/connect https://api.openai.com/v1           # OpenAI
/connect http://localhost:11434              # Local Ollama
/connect http://localhost:8000/v1            # Local vLLM
/connect https://inference-api.nvidia.com/v1 # NVIDIA inference API
```

Give it a URL and `/connect` figures out the rest: it fetches the available
models, prompts for an API key if the backend needs one, saves an alias to your
project, and switches to the model you pick. Rerun `/connect` on the same URL
any time to update the saved alias.

### Editing saved config

Everything `/connect` writes lives under your project's `.nooa/` folder:

- `.nooa/llm_config.yaml` — saved model aliases
- `.nooa/secrets.yaml` — API keys keyed by env-var name
- `.nooa/settings.yaml` — TUI preferences and default model

Edit any of them from inside the TUI with `/edit .nooa/<file>`, or open them in
your usual editor. Changes to `settings.yaml` and `llm_config.yaml` are picked
up on the next launch.

Agent Skills are discovered from installed `nooa.skills` entry points and from
conventional `.agents/skills`, `.claude/skills`, and `.cursor/skills`
directories. Discovered workflow skills are loaded but remain model-inactive
until `/skills activate <id>` or explicit invocation. Operations marked with
`@slash_command` still appear as TUI slash commands. This is the extension path
for project-specific workflows; they are not hard-coded into the terminal host.

### Extending the coding agent

The default `nooa_cli.coding.CodingAgent` is shared infrastructure for
interactive hosts. An internal package can subclass it and select the subclass
without forking the TUI:

```python
from nooa_cli.coding import CodingAgent


class InternalCodingAgent(CodingAgent):
    pass
```

```bash
nooa tui --agent internal_agents:InternalCodingAgent
```

Private or organization-specific model registries stay outside this package.
Place the registry at `.nooa/llm_config.yaml` for automatic project-local
discovery, or pass a downloaded file explicitly:

```bash
nooa tui --llm-config /path/to/llm_config.yaml
```

Explicit paths have highest precedence. Run `nooa config show` to inspect the
registry layers that were discovered.

The TUI passes `llm`, durable `storage`, `cwd`, and `skills_dirs` when those
parameters are declared by the custom class. Installed Python skills should be
published through the `nooa.skills` entry-point group. Toolbar extensions can
similarly publish named providers through `nooa_cli.tui.toolbar_items`; users
select their order with `/toolbar set <item> ...`.

Keep-going mode is an explicit opt-in. It audits a completed turn with a
separate judge model and sends an internal continuation only when autonomous
work remains:

```text
/keep-going model nemotron3-nano-30b
/keep-going on
```

Use `/keep-going off` to disable it. New user input supersedes and cancels an
in-flight audit.

Long-term memory and idle reflection are also explicit opt-ins:

```text
/memory on        # project-wide store shared across sessions
/memory local     # sidecar store for only this session
/memories         # browse, forget, or complete stored memories
/reflection on    # consolidate new memories while the agent is idle
/reflection now   # run a consolidation pass immediately
```

`/memory off` detaches memory from the current agent. `/reflection off` stops
idle consolidation without deleting stored memories. Project memory lives at
`.nooa/memory/memory.sqlite` by default; session memory lives beside the
session database. Both choices are persisted per agent in `settings.yaml`.

### MCP servers

Use the TUI commands for the common lifecycle:

```text
/mcp list
/mcp add docs https://docs.example.com/mcp
/mcp approve docs
/mcp approve docs <confirmation-code>
/mcp connect docs
/mcp disconnect docs
/mcp remove docs
```

`/mcp add` writes an HTTP URL or a single stdio command to the project
`.nooa/settings.yaml`. Configuration is discovery, not trust: the TUI shows a
secret-safe fingerprint review and requires the user to repeat its confirmation
code before any transport or local process starts. Any configuration change
invalidates that approval. Environment placeholders are resolved only after
approval. `/mcp remove` removes inline project servers and revokes their stored
approvals without disturbing sibling settings. Servers sourced from an external
`.mcp.json` must be removed from that file.

For a richer server definition—stdio arguments, environment mappings, headers,
or OAuth settings—use `/mcp-add <the details you have>`. That user-invocable
skill asks the coding agent to edit the same project settings without embedding
secret values, then directs you back through the user-owned review and approval
flow. Removal is always `/mcp remove <name>` (or an edit to the source
`.mcp.json` for externally defined servers).

For stdio arguments, environment variables, headers, or OAuth options, use the
full settings form:

```yaml
tui:
  mcp_servers:
    local-tools:
      command: uvx
      args: [my-mcp-server]
      env:
        LOG_LEVEL: info
    hosted-tools:
      url: https://tools.example.com/mcp
      transport: streamable-http
      oauth_client_id: my-client-id
      oauth_scope: "tools.read tools.write"
```

Keep secret values in the host environment and use literal `${VAR}` placeholders
in repository config. First-time OAuth consent remains a human browser step;
manual codes are collected in a masked in-app prompt. Cached credentials are
reused by later `/mcp connect` calls after the exact server definition remains
approved.

See the main repo [README](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/README.md) for the framework documentation.

## Interactive coding sessions

`nooa_cli.sessions` owns durable coding-agent session identity, metadata, and
conversation replay shared by CLI hosts such as the native TUI and ACP. The
process running an agent owns the writable session handle; other hosts attach
through their transport or use read-only discovery. Generic event and SQLite
storage primitives remain in the core `nooa` package.
