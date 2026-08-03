# Benchmarks Example

Everything needed to run the NOOA BenchAgent under the open-source
[Harbor](https://github.com/harbor-framework/harbor) benchmark harness
(the setup used for the tech report's SWE-bench Verified and
Terminal-Bench 2.0 results).

| File | What |
|------|------|
| `harbor_adapter.py` | Harbor "installed agent" adapter: clones this repo into the trial container, installs `packages/nooa-bench`, and invokes the `nemo-harbor` CLI |
| `harbor_minimal.yaml` | Minimal Harbor config wiring the adapter to a model and dataset |
| `harbor_codeact_context_ab.yaml` | Minimal two-agent config for a CodeAct baseline vs CodeAct + context-management A/B run |
| `harbor_lhtb_codeact_context_ab_small.yaml` | Small Long-Horizon Terminal-Bench subset for the same two CodeAct arms |
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

## CodeAct Context-Management A/B

For a SWE-bench-style long-context evaluation without training, use
`harbor_codeact_context_ab.yaml`. It runs the same model and dataset with two
agent registry keys:

| `agent_type` | Arm |
|--------------|-----|
| `bench-codeact` | CodeAct baseline. The LLM does not see `self.context`, `self.events`, or event-collapse instructions. |
| `bench-codeact-acm` | CodeAct + context management. The LLM sees `self.context`, `self.events`, and the `self.events.collapse(...)` self-compression hint. |

Keep every other setting fixed: model, endpoint, dataset, attempts, container
image, and task timeout. Compare task pass rate first, then use the trace
outputs for peak prompt tokens, input/output tokens, iteration count, and any
context-collapse calls. The existing `bench` alias remains the context-enabled
BenchAgent used by the tech-report setup.

## LHTB Small Subset

Long-Horizon Terminal-Bench (LHTB) is a Harbor benchmark with 46 long terminal
tasks. For LHTB-compatible scoring, use the patched Harbor bundled in the LHTB
repo because many LHTB tasks set `continue_until_timeout = true`; stock Harbor
ignores that flag and runs those tasks single-shot.

Minimal local setup:

```bash
# From the parent directory that contains this repo:
git clone https://github.com/zli12321/LHTB.git
cd LHTB
git lfs install
git lfs pull
uv tool install --editable ./harbor --force

cd ../nemo_oo_agents_acm_swe_eval
export DOCKER_DEFAULT_PLATFORM=linux/amd64   # recommended on Apple Silicon
export NVIDIA_INFERENCE_API_KEY=...          # or NVIDIA_API_KEY
PYTHONPATH=examples/benchmarks harbor run --config examples/benchmarks/harbor_lhtb_codeact_context_ab_small.yaml
```

The small config runs three LHTB tasks:
`langchain-version-migration`, `commit0-multilib-tdd`, and
`duckdb-optimizer-closure`. It creates two Harbor agents with the same model,
endpoint, dataset, attempts, Docker runtime, and timeout; only `agent_type`
differs (`bench-codeact` vs `bench-codeact-acm`). Set each agent's `git_ref`
to this branch or a commit SHA once the branch is available to the trial
container.
