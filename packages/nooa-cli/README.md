# nemo-labs-oo-agents-cli

CLI and TUI for [nemo-oo-agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Ships the `nooa` command (subcommands like `nooa tui`, `nooa term`, `nooa eval`, `nooa traces`) and the agent TUI (a REPL-driven CodeAct frontend).

## Install

```bash
# CLI + TUI
uv add nemo-labs-oo-agents-cli

# ...with numpy/pandas/plotly/scipy/sklearn pre-loaded into the LLM REPL
uv add "nemo-labs-oo-agents-cli[datascience]"

# ...with web-served TUI (uvicorn + ptyprocess + fastapi, plus datascience)
uv add "nemo-labs-oo-agents-cli[web]"
```

`nemo-labs-oo-agents-cli` automatically pulls in matching `nemo-oo-agents` (the core framework). The `[datascience]` extra adds libraries the LLM can use in REPL-generated code; the `[web]` extra adds the `nooa term` web frontend.

## Usage

```bash
nooa --help
nooa tui                  # interactive agent REPL
nooa term                 # web terminal
nooa eval ...             # eval pipeline runner
```

See the main repo [README](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/README.md) for the framework documentation.
