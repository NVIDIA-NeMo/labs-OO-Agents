# nooa-cli

CLI for [nemo-oo-agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Ships the `nooa` command with subcommands for running evaluations, browsing traces, and managing config.

## Install

```bash
uv add nooa-cli
```

`nooa-cli` automatically pulls in matching `nemo-oo-agents` (the core framework).

## Usage

```bash
nooa --help
nooa start-dev        # launch the trace viewer
nooa eval ...         # eval pipeline runner
nooa traces ...       # inspect/manage trace files
```

Install the separate `nooa-coder` package to add the `nooa acp` plugin command and
run the NOOA coding agent from an ACP-compatible client (`nooa-acp` still works
as a package name; it now only depends on `nooa-coder`):

```bash
uv add nooa-coder
export NOOA_MODEL=nvidia_nim/nvidia/nemotron-3-super-120b-a12b
export NVIDIA_API_KEY=nvapi-...
uv run nooa-acp
```

See the main repo [README](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/README.md) for the framework documentation.

## Interactive coding sessions

The coding agent, its tools, and durable coding-agent sessions live in the
[`nooa-coder`](../nooa-coder/README.md) package. `nooa-cli` depends only on the
core `nooa` package.
