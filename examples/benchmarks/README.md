# Benchmarks Example

Everything needed to run the NOOA BenchAgent under the open-source
[Harbor](https://github.com/harbor-framework/harbor) benchmark harness
(the setup used for the tech report's SWE-bench Verified and
Terminal-Bench 2.0 results).

| File | What |
|------|------|
| `harbor_adapter.py` | Harbor "installed agent" adapter: clones this repo into the trial container, installs `packages/nooa-bench`, and invokes the `nemo-harbor` CLI |
| `harbor_minimal.yaml` | Minimal Harbor config wiring the adapter to a model and dataset |
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
