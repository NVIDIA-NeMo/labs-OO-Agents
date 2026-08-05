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

## TUI configuration

The TUI reads layered `settings.yaml` files from the user config directory and
the current project's `.nooa/` directory. It also honors `AGENTS.md` files from
the repository root down to the current working directory.

### Models and endpoints

For reusable aliases, connect a provider or backend from inside the TUI:

```text
/connect anthropic
/connect http://localhost:11434
/connect http://localhost:8000/v1
/connect https://inference-api.nvidia.com/v1
```

For native providers such as Anthropic, the TUI uses the provider's model-list
API and conventional secret name. For OpenAI-compatible servers, it fetches the
`/models` catalog. In both cases it can save a pasted key to
`.nooa/secrets.yaml`, write a project-local alias to `.nooa/llm_config.yaml`,
reload the registry, and switch to the selected model. If a local server such as
Ollama or vLLM exposes OpenAI-compatible routes under `/v1`, `/connect
http://host:port` will retry `/v1/models` after a root `/models` 404.
Conventional local Ollama endpoints are saved with the `ollama_chat/` LiteLLM
provider and no API key.

For an advanced one-shot run, pass the same model string and API base you would
pass to `get_llm_client`. `--api-base` requires `--model`:

```bash
nooa tui --model ollama_chat/qwen3:1.7b --api-base http://localhost:11434
nooa tui --model hosted_vllm/Qwen/Qwen3-1.7B --api-base http://localhost:8000/v1
nooa tui --model openai/nvidia/my-model --api-base https://inference-api.nvidia.com/v1 --api-key-env NVIDIA_INFERENCE_API_KEY
```

To persist a direct endpoint override manually, put it in `.nooa/settings.yaml`:

```yaml
tui:
  default_model: hosted_vllm/Qwen/Qwen3-1.7B
  api_base: http://localhost:8000/v1
```

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
