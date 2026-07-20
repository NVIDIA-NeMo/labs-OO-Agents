# NeMo OO Agents Reference

Quick reference for paths, examples, commands, and configuration.

## Key Paths

| Path | What |
|------|------|
| `src/nooa/` | Core framework source (the `nemo-labs-oo-agents` published package) |
| `src/nooa/context_blocks/` | Context block rendering subpackage |
| `src/nooa/unifiedllm/` | LLM client subpackage |
| `packages/nooa-cli/` | CLI + TUI (separately-published `nemo-labs-oo-agents-cli`) |
| `packages/nooa-memory/` | Long-term memory subsystem (separately-published `nemo-labs-oo-agents-memory`) |
| `packages/nooa-bench/` | BenchAgent + `nemo-harbor` runner (separately-published `nemo-labs-oo-agents-bench`) |
| `examples/` | Example agents (see below for details) |
| `docs/guides/` | Detailed framework guides |

## Examples

**Quickstart** (`examples/quickstart/`) — step-by-step tutorial sequence. Start here. Each example builds on the previous one and can be run standalone with `uv run python examples/quickstart/XX_name.py`.

| Example | File |
|---------|------|
| Simplest agent | `examples/quickstart/01_first_generation_method.py` |
| Structured output | `examples/quickstart/02_structured_outputs.py` |
| Deterministic + generative methods | `examples/quickstart/03_codeact_tools.py` |
| Strategies compared | `examples/quickstart/04_strategies.py` |
| Progressive disclosure | `examples/quickstart/05_progressive_disclosure.py` |
| Tracing (all method types) | `examples/quickstart/06_tracing.py` |
| Dynamic prompts | `examples/quickstart/07_dynamic_prompts.py` |
| Context blocks | `examples/quickstart/08_context_blocks.py` |
| Summarization | `examples/quickstart/09_summarization.py` |
| Skills | `examples/quickstart/10_skills.py` |
| MCP tools | `examples/quickstart/11_mcp.py` |
| Long-term memory | `examples/quickstart/12_memory.py` |
| Multimodal (images) | `examples/quickstart/13_multimodal.py` |
| ATIF trajectory export | `examples/quickstart/14_atif_trajectory.py` |
| NeMo Flow (nemo_relay) integration | `examples/quickstart/15_nemo_relay.py` |

**Advanced** (`examples/advanced/`) — deeper dives into specific features and integrations. Each example is self-contained; read the file docstring for prerequisites.

| Example | File |
|---------|------|
| Agent memory | `examples/advanced/memory.py` |
| CodeAct event flow | `examples/advanced/codeact_event_sequence.py` |
| Pre-ellipsis prefill | `examples/advanced/prefill.py` |
| OTLP tracing | `examples/advanced/tracing_otlp.py` |
| Langfuse tracing | `examples/advanced/tracing_langfuse.py` |
| Phoenix tracing | `examples/advanced/tracing_phoenix.py` |
| Swappable execution engines | `examples/advanced/swappable_execution_engines.py` |

## Detailed Guides

For deeper understanding, see these topic-specific docs:

| Topic | File | When to Read |
|-------|------|-------------|
| How prompts work | `docs/guides/prompt-mechanics.md` | Building any agent |
| Strategies in depth | `docs/guides/strategies.md` | Choosing/configuring strategies |
| Context blocks | `docs/guides/context-blocks.md` | Managing agent state and memory |
| Single vs multi-agent | `docs/guides/single-vs-multi-agent.md` | Architectural decisions |
| Structured output | `docs/guides/structured-output.md` | Return types, Pydantic validation, PythonSource pattern |
| Method design | `docs/guides/writing-generation-methods.md` | Writing effective generation methods |
| Truncation | `docs/guides/truncation.md` | How large values are previewed to the model |
| Config migration (0.4.x → 0.5.0) | `docs/guides/config-migration.md` | Moving existing config to the unified `nooa` layout (settings/secrets/llm_config YAML) — breaking changes in 0.5.0 |

## Logging

NeMo OO Agents uses Python's standard `logging` module throughout.  Every module
creates a logger with `logging.getLogger(__name__)`, producing a hierarchy
rooted at `nooa`.

### Quick start

```python
from nooa import enable_logging

enable_logging()                                     # everything at DEBUG
enable_logging(level=logging.INFO)                   # calmer overview
enable_logging(name="nooa.strategies")           # just strategies
enable_logging(name="nooa.runtime.actor")        # just the executor
```

### Logger hierarchy

| Logger name | What it covers |
|-------------|---------------|
| `nooa` | Root — catches everything below |
| `nooa.agent` | Agent lifecycle, configuration |
| `nooa.runtime.actor` | Execution engine, code execution |
| `nooa.runtime.hooks` | Hook dispatch |
| `nooa.runtime.code_validator` | Code validation / safety checks |
| `nooa.strategies.codeact` | CodeAct strategy |
| `nooa.strategies.predict` | Predict strategy |
| `nooa.strategies.pure_python` | PurePython strategy |
| `nooa.strategies.reflexion` | Reflexion strategy |
| `nooa.tools.*` | Individual tool modules |
| `nooa.storage.*` | Storage backends |
| `nooa.library_manager` | Library / skill loading |
| `nooa.skill_manager` | Skill discovery |

### For application developers

`enable_logging()` is a convenience for development.  In production, use
`logging.config.dictConfig()` as usual — nooa loggers are standard
`logging.Logger` instances.  The library adds only a `NullHandler` to the
root logger, so no output appears unless you configure handlers.

## Config files

Three config files share one layout, one precedence chain, and one loader
(`nooa.layered_config`). All are optional — absence means "use the
layer below."

| File | What | Bundled defaults | Env-var override |
|------|------|------------------|------------------|
| `llm_config.yaml` | LLM model aliases | entry-point group `nooa.bundled_configs` | `NEMO_OO_LLM_CONFIG` |
| `settings.yaml` | TUI settings (`tui:` / `agent:` sections, dataclass field names) | in-code defaults | `NEMO_OO_SETTINGS` |
| `secrets.yaml` | API keys (`env:` name→value map, pushed into `os.environ` non-clobbering) | — | `NEMO_OO_SECRETS` |

```text
~/.config/nooa/           # user-global   (XDG_CONFIG_HOME aware; override base with NEMO_OO_USER_DIR)
├── settings.yaml
├── secrets.yaml             # chmod 600
└── llm_config.yaml

.nooa/                    # project-local (override base with NEMO_OO_PROJECT_DIR)
├── settings.yaml
├── secrets.yaml             # gitignore strongly recommended
└── llm_config.yaml
```

Precedence (low → high, last wins): bundled defaults → user → project →
env-var path(s). For `secrets.yaml`, an env var already set in the process
always wins over a file value. `null` deletes a key inherited from a lower
layer. Run `nooa config show` to see which layers are loading (secret
values redacted).

`secrets.yaml`:

```yaml
env:
  NVIDIA_INFERENCE_API_KEY: sk-...
  ANTHROPIC_API_KEY: sk-ant-...
```

## Environment Variables

See `.env` for API keys (library use); the CLI/TUI reads `secrets.yaml`:
- `OPENAI_API_KEY` — OpenAI
- `NVIDIA_API_KEY` — NVIDIA NIM (integrate.api.nvidia.com)
- `NVIDIA_INFERENCE_API_KEY` — NVIDIA inference gateway (inference-api.nvidia.com); the legacy name `NVIDIA_INTERNAL_API_KEY` is still accepted

| Provider | Endpoint | Key |
|----------|----------|-----|
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `nvidia` | integrate.api.nvidia.com | `NVIDIA_API_KEY` |
| `nvidia_internal` | inference-api.nvidia.com | `NVIDIA_INFERENCE_API_KEY` |

### `NEMO_OO_*` variables

All framework-owned env vars use the `NEMO_OO_` prefix:

| Variable | What |
|----------|------|
| `NEMO_OO_USER_DIR` | Override the user config base (default `~/.config/nooa`, XDG-aware) |
| `NEMO_OO_PROJECT_DIR` | Override the project config dir (default `<root>/.nooa`) |
| `NEMO_OO_LLM_CONFIG` | Comma-separated YAML path(s) — highest-priority `llm_config.yaml` layer |
| `NEMO_OO_SETTINGS` | Comma-separated YAML path(s) — highest-priority `settings.yaml` layer |
| `NEMO_OO_SECRETS` | Comma-separated YAML path(s) — highest-priority `secrets.yaml` layer |
| `NEMO_OO_TRACE_VIEWER_PORT` | Port for the trace viewer (`nooa start-dev`; default 5001) |
| `NEMO_OO_TRACE_DB` | SQLite trace-store path for the viewer (default `~/.config/nooa/traces.db`) |
| `NEMO_OO_RICH_URL` | Rich-content POST endpoint, set by `nooa term` for the web frontend (internal) |

Any var named under a `secrets.yaml` `env:` map (e.g. `NVIDIA_INFERENCE_API_KEY`)
is pushed into the process env non-clobbering — an already-exported value always wins.
