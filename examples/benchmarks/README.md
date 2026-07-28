# Benchmarks Example

Everything needed to run the NOOA BenchAgent under the open-source
[Harbor](https://github.com/harbor-framework/harbor) benchmark harness
(the setup used for the tech report's SWE-bench Verified and
Terminal-Bench 2.0 results).

| File | What |
|------|------|
| `harbor_adapter.py` | Harbor "installed agent" adapter: clones this repo into the trial container, installs `packages/nooa-bench`, and invokes the `nemo-harbor` CLI |
| `harbor_minimal.yaml` | Minimal Harbor config wiring the adapter to a model and dataset |
| `harbor_copilot_minimal.yaml` | Minimal Harbor config for the GitHub Copilot SDK-backed agent |
| `bench_agent.py` | Standalone 35-line minimal agent, for reading — the real BenchAgent lives in [`packages/nooa-bench`](../../packages/nooa-bench/) |

## Run

```bash
# 1. Install Harbor (see the Harbor repo for details) and have apptainer/docker available.
# 2. Point the config at your task dataset, then from the repo root:
PYTHONPATH=examples/benchmarks harbor run --config examples/benchmarks/harbor_minimal.yaml
```

Credentials: set `NVIDIA_INFERENCE_API_KEY` (inference.nvidia.com gateway) or
`NVIDIA_API_KEY` (public NIM endpoint) on your host — the adapter forwards it
into the agent process, and since the NVIDIA endpoints are OpenAI-compatible
it is also exposed as `OPENAI_API_KEY` for litellm. To use another provider
directly, add its key to `FORWARDED_ENV_VARS` in `harbor_adapter.py`. The
trial container needs network access during install (git clone + dependency
download).

Inside the container, Harbor invokes:

```
nemo-harbor --instruction '...' --model '...' --agent-type bench
```

which runs the BenchAgent and writes `result.json` (success, response, token
counts) to `/logs/agent/` — token usage is surfaced back into Harbor's
per-trial context for cost analysis.

## Copilot SDK agent

Direct local runs reuse the signed-in GitHub Copilot CLI account:

```bash
uv run --package nooa-bench nemo-harbor \
  --instruction '...' \
  --agent-type copilot \
  --model gpt-5.6-sol \
  --reasoning-effort xhigh \
  --context-tier long_context
```

`xhigh` is the highest reasoning-effort literal supported by the Python
`github-copilot-sdk` version used here; `long_context` requests the SDK's
largest context tier.

For Harbor containers, use `harbor_copilot_minimal.yaml`. The Copilot path
approves tool permissions non-interactively because each task runs in an
isolated benchmark container. Containerized runs do not automatically inherit
your host login: forward `COPILOT_GITHUB_TOKEN`, or mount supported Copilot
state into the container. Do not copy secrets into this repo or commit them.
The adapter pre-downloads the Copilot runtime during install with
`COPILOT_CLI_EXTRACT_DIR=/opt/nooa-copilot-runtime`. It keeps Copilot
runtime/session state in the private, container-local
`COPILOT_HOME=/tmp/nooa-copilot-home`; that state is not persisted with Harbor
artifacts.

Future Pareto-loop orchestration and GPU allocation should stay outside this
agent boundary; this agent only adapts Harbor tasks to the Copilot SDK runtime.
