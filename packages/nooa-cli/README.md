# nooa-cli

CLI for [nemo-oo-agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Ships the `nooa` command with subcommands for running evaluations, browsing traces, and managing config.

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
```

See the main repo [README](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/README.md) for the framework documentation.

## Interactive coding sessions

`nooa_cli.sessions` owns durable coding-agent session identity, metadata, and
conversation replay shared by CLI hosts such as the native TUI and ACP. The
process running an agent owns the writable session handle; other hosts attach
through their transport or use read-only discovery. Generic event and SQLite
storage primitives remain in the core `nooa` package.
