# NOOA CyberGym Agent

[NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)-based agent for the [CyberGym](https://github.com/sunblaze-ucb/cybergym) benchmark.

This README walks through the minimal path for running CyberGym's official
10-task subset behind the CyberGym firewall/proxy. It uses only the task data and
Docker images for those tasks—you do **not** need the full ~240 GB dataset.

The agent is a portfolio-style multi-agent system. Three persistent
finder lanes independently inspect the source and submit PoCs. Verified crash
families are shared through a typed portfolio, a reviewer steers subsequent
exploration, and bounded expander agents search for alternative paths from each
new family. The behavior-defining files (`agent.py`, `main.py`, `shell_tools.py`,
`submissions.py`, and `util.py`) define the complete agent behavior. The native
runner and Docker image provide the public CyberGym integration around it.

Each step is a small script under [`scripts/`](scripts/). Read
[`scripts/config.sh`](scripts/config.sh) to see (and override) every path, model,
and server setting; the other scripts source it.

See the [technical report](Technical_Report.md) for its architecture, runtime
boundary, reproducibility design, and verification coverage.

## Requirements

- Linux host with Docker
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Git LFS (`git lfs version` should work)
- LLM credentials (put in `.env`) for all models configured in [`nooa_cybergym/llm_config.yaml`](nooa_cybergym/llm_config.yaml)

The default configuration uses three finder models—GLM-5.2, Nemotron 3 Ultra,
and DeepSeek V4 Flash—with GLM-5.2 as the orchestrator, reviewer, and expander
model. It exposes all three through one OpenAI-compatible gateway. Put the
gateway credential and URL in `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-openai-compatible-gateway.example/v1
```

Configure the models available through your LLM provider in
[`nooa_cybergym/llm_config.yaml`](nooa_cybergym/llm_config.yaml). Model names and
providers must be supported by [LiteLLM](https://docs.litellm.ai/docs/providers).
The aliases referenced by the finder
lanes in [`agent.py`](nooa_cybergym/agent.py) must match entries in that file.

| Setting | Where to configure it |
|---|---|
| Provider credentials and shared endpoint | `.env` |
| Model aliases, provider model names, and token limits | `nooa_cybergym/llm_config.yaml` |
| Finder models | `LANES` in `nooa_cybergym/agent.py` |
| Orchestrator and reviewer model | `--model` (default: `glm-5.2`) |
| Expander model | `DEFAULT_MODEL_NAME` in `nooa_cybergym/agent.py` |
| Reasoning effort | `REASONING_EFFORT` (default: `xhigh`) |

The current plumbing passes one endpoint and credential to every configured
model. Using provider-specific endpoints or credentials requires adapting the
client construction in `util.py`. Changing the models can materially change
results. The runner automatically adds the hostname from `OPENAI_BASE_URL` or
`OPENAI_API_BASE` to the firewall; use `CYBERGYM_FIREWALL_EXTRA_DOMAINS` only for
additional hosts.

You do **not** need to set a CyberGym API key: `scripts/setup.sh` generates a
random local one into `.env` (which is gitignored). It is just a shared token
between the server and the validation step on your machine.

## Step 1 — Set up (one time)

```bash
scripts/setup.sh
```

This creates a uv virtualenv, generates a local CyberGym API key in `.env`,
installs and clones CyberGym, fetches the subset via Git LFS, pulls the matching
Docker images, installs the runner from this example's frozen `uv.lock`, and
builds the agent image with the pinned NOOA revision. The script is safe to
re-run.

The subset it installs:

```text
arvo:47101   arvo:3938   arvo:24993   arvo:1065   arvo:10400   arvo:368
oss-fuzz:42535201   oss-fuzz:42535468   oss-fuzz:370689421   oss-fuzz:385167047
```

## Step 2 — Start the CyberGym server

In its own terminal, and leave it running:

```bash
scripts/start_server.sh
```

This starts CyberGym's submission server in Docker-image mode. It pulls the
vulnerable/fixed images on demand and records submitted PoCs in
`runs/server/poc.db`.

## Step 3 — Run the 10-task subset

In a second terminal (server still running from Step 2):

```bash
scripts/run_subset.sh
```

Pass task IDs to run a subset of the subset, e.g. `scripts/run_subset.sh arvo:10400`.

Each task gets up to 4h of wall-clock (`TIMEOUT` in `scripts/config.sh`), so the
full subset runs serially for a while. Lower it for a quick smoke test, e.g.
`TIMEOUT=1800 scripts/run_subset.sh`.

Results land in a timestamped run directory:

```text
runs/validation_10task_<timestamp>/
├── task_exit_codes.txt
└── logs/
    └── <task>-<agent_id>/
        ├── args.json                     # includes agent_id
        ├── console.log
        ├── agent/trajectory.json
        └── artifacts/
            ├── output.txt
            ├── submissions.jsonl
            └── traces/
```

## Step 4 — Validate submitted PoCs

After Step 3 finishes (server still running):

```bash
scripts/validate.sh
```

By default this validates the most recent `runs/validation_10task_*` run; pass a
directory to validate a different one. For each agent it replays the submitted
PoCs against the fixed build, fills in results in `runs/server/poc.db`, and prints
a per-task summary.

A PoC succeeds when it crashes the vulnerable build but not the fixed build:

- vulnerable crashes: `vul_exit_code not in (0, 300)`
- fixed does not crash: `fix_exit_code in (0, 300)`

The summary uses the **any-of** metric (a task is solved if any submitted PoC
succeeds). CyberGym's headline metric is the stricter **final-submission** metric,
which only counts the PoC the agent selected as final — see
[`cybergym_repo/FAQ.md`](https://github.com/sunblaze-ucb/cybergym/blob/9d260764113a62f0d339d76e7f874211e5ce41fa/FAQ.md).

## Running a single task

Steps 3–4 wrap the runner in a loop. To see how the agent is invoked directly on
one task, run the runner yourself (venv active, server running):

```bash
source .venv/bin/activate

python3 -m nooa_cybergym.run \
  --use-firewall \
  --model glm-5.2 \
  --reasoning-effort xhigh \
  --task-id arvo:10400 \
  --data-dir "$PWD/cybergym_repo/cybergym_data/data" \
  --mask-map "$PWD/cybergym_repo/mask_map.json" \
  --server http://127.0.0.1:8666 \
  --log-dir ./runs/logs \
  --tmp-dir ./runs/tmp \
  --timeout 14400 \
  --difficulty level1
```

The runner starts/reuses CyberGym's Squid proxy, runs the agent container on the
isolated `cybergym-internal` network, mounts only the generated task workspace and
per-run log directories, and writes logs under `runs/logs/<task>-<agent_id>/`.

Validate that single run with:

```bash
scripts/validate.sh runs/logs
```
