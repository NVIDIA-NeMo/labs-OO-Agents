# NOOA CyberGym

<!-- **Contact:** TODO -->

## 1. Overview

This submission evaluates an agent built on [**NVIDIA-labs Object-Oriented Agents (NOOA)**](https://github.com/NVIDIA-NeMo/labs-OO-Agents) on the **CyberGym Level 1** benchmark ([cybergym.io](https://www.cybergym.io/cybergym/)), where the agent gets a vulnerability description plus the pre-patch codebase and must produce a proof-of-concept input that crashes the pre-patch binary but not the patched one.

The submitted agent uses a portfolio of three persistent finder agents. Each finder independently analyzes the source and submits candidate PoCs. Verified crash families are shared through a typed portfolio, a reviewer steers further exploration, and bounded expander agents search for alternative trigger paths.

The finder models are **GLM-5.2**, **Nemotron 3 Ultra**, and **DeepSeek V4 Flash**. GLM-5.2 is also used by the orchestrator, reviewer, and expanders.

**Result: 1,286 / 1,507 tasks solved = 85.3% pass@1.**

## 2. Architecture

### 2.1 NOOA SDK

NVIDIA-labs Object-Oriented Agents (NOOA) is a model-agnostic, open-source Python framework for building AI agents. Where most frameworks split prompts, tools, callbacks, and workflow graphs into separate abstractions, NOOA represents an agent as a single Python class: its fields are state, its methods are capabilities, its docstrings are prompts, and its type annotations are enforced contracts. A method whose body is an ellipsis (`...`) is completed at runtime by an LLM-driven loop, while a method with a normal body runs as ordinary deterministic Python.

The design unifies six model-facing ideas: typed input/output, pass by reference to live Python objects, code as action, programmable orchestration loops, explicit typed object state, and model-callable harness APIs.

* Code: [NVIDIA-labs Object-Oriented Agents (NOOA)](https://github.com/NVIDIA-NeMo/labs-OO-Agents).
* Paper: [NVIDIA-labs OO Agents: Native Python Object-Oriented Agents](https://arxiv.org/abs/2607.20709).

### 2.2 NOOA CyberGym Agent

The NOOA CyberGym agent runs inside each trial container as a portfolio-style multi-agent system. Three persistent finder lanes independently inspect the vulnerability description, pre-patch source tree, input harness, and build metadata. Both finders and expanders use NOOA's CodeAct strategy with a Python runtime, a persistent shell, and a typed method for submitting candidate input files.

The submission manager keeps benchmark mechanics out of model prompts. It invokes the CyberGym submission interface, classifies verifier output, fingerprints sanitizer crashes and fatal signals, and records each candidate together with the finder's trigger hypothesis. The shared portfolio exposes only distinct verified crash families and reviewer guidance to the workers.

The orchestrator reviews the portfolio when a finder finishes or a new crash family appears. The reviewer assesses whether the crashes target the described vulnerability, provides guidance, and recommends when to stop. Each new finder-sourced family can seed an expander that searches for alternate trigger paths; expander results do not recursively create more expanders. A minimum exploration interval prevents an early stop, and bounded concurrency, iteration limits, memory checks, summarization, and a soft timeout keep the run within the trial budget.

No cybersecurity domain knowledge, exploit templates, or benchmark-specific hints are supplied to the agent beyond what the configured models already bring from pretraining; the workflow above is generic vulnerability validation. Performance is therefore attributable to the agent architecture and the underlying models rather than to task-specific steering.

* Code: [NOOA CyberGym](nooa_cybergym/agent.py)

## 3. Method

### 3.1 Benchmark

[CyberGym](https://www.cybergym.io/cybergym/) is a benchmark for evaluating AI agents on realistic cybersecurity tasks. It contains 1,507 real-world vulnerabilities from 188 open-source projects, where agents must analyze vulnerable codebases and generate proof-of-concept (PoC) exploits.

In the primary *Level 1* setting, agents receive a vulnerability description and the vulnerable (pre-patch) codebase, and must generate a proof-of-concept (PoC) input that triggers the vulnerability. Solutions are evaluated using differential execution: a PoC must crash the pre-patch binary while failing to crash the post-patch version, ensuring it targets the intended vulnerability rather than an unrelated bug.

*Level 0* is a harder setting in which agents receive only the vulnerable codebase and must first discover the vulnerability. We train and evaluate our agent only on the standard *Level 1* setting.

### 3.2 Agent Configuration

* **Agent framework**: NVIDIA-labs Object-Oriented Agents (NOOA)
* **NOOA revision**: `8229922d7274628c9be83f745589b40852680d60`
* **Finder models**: GLM-5.2, Nemotron 3 Ultra, and DeepSeek V4 Flash
* **Orchestrator, reviewer, and expander model**: GLM-5.2
* **Reasoning effort**: `xhigh`
* **Tools**: Python runtime with persistent shell and typed CyberGym submission interface
* **Minimum exploration time**: 1,200 s
* **Maximum concurrent expanders**: 2
* **Soft timeout**: 13,920 s (~3.87 h), returns the best verified portfolio found so far

The submitted evaluation used NOOA commit [`8229922d7274628c9be83f745589b40852680d60`](https://github.com/NVIDIA-NeMo/labs-OO-Agents/commit/8229922d7274628c9be83f745589b40852680d60). The open-source example pins the framework to this revision and installs its runtime dependencies from the revision's own frozen `uv.lock`.

### 3.3 Access to Vulnerable vs. Patched Builds

The agent is provided only the pre-patch (vulnerable) program (`repo-vul.tar.gz`); the post-patch (`-fix`) image is never accessible to the agent during runtime. Only the submission server uses the `-fix` image, and only to verify that the submitted PoC crashes the vulnerable build but no longer crashes the patched build. The agent must therefore reason about which PoC best matches the described vulnerability without ever seeing the fix.

### 3.4 Pass@1

Tasks were run only once. Only infrastructure failures triggered a retry, specifically when the agent returned a non-zero exit code due to crashes caused by API issues, Docker failures, or out-of-memory kills. Each attempt was capped at 4 hours of agent wall-clock time.

### 3.5 Network Isolation

Each CyberGym task runs in an isolated Docker environment: the agent and task server share an internal-only network with no direct egress, while a mitmproxy sidecar connected to both the internal and external networks provides the sole external route for processes in the agent container. The proxy permits only explicitly allowlisted package repositories and configured LLM endpoints, rejects other destinations, and inspects supported gateway API requests to remove known hosted web-search, web-fetch, remote-execution, and MCP tools. These interventions are logged per trial, providing auditable restricted runtime internet access. In addition, automated and manual inspection of the logs and trajectories revealed no successful web fetch attempts.

### 3.6 Scoring

An agent can submit many PoCs while working a task, so a task's success can be counted two ways ([CyberGym FAQ](https://github.com/sunblaze-ucb/cybergym/commit/9d260764113a62f0d339d76e7f874211e5ce41fa), Q3):

* **Any-of**: the task counts as solved if *any* submitted PoC succeeds.
* **Final-submission**: the task counts as solved only if the single PoC the agent designates as its final answer succeeds.

**We report the any-of metric**: a task is solved if any PoC the agent submitted during the run satisfies the differential-execution check. We adopt *any-of* because our portfolio workflow is built around iterative submission. Its finders author, submit, and refine candidate PoCs against the sanitizer-instrumented binary, retaining verified crash families as soon as they are found, and *any-of* scores exactly that behavior without penalizing exploration.

### 3.7 Dynamic Analysis Setup

Agents did not have direct access to the vulnerable or fixed binaries. The agent had shell access to its own task container, including `/workspace/task_data/` and a `submit()` wrapper around `/workspace/submit.sh`. Submissions were sent to a task-server sidecar, which ran the PoC on the vulnerable binary and returned sanitizer feedback. The fixed binary and reference PoC were not exposed to the agent and were used only by the verifier/scoring path. The agent could write and execute helper code in its container and submit arbitrarily many PoCs, but it could not inspect or directly execute the hidden vulnerable/fixed binaries, read `/tmp/poc`, or access git history.

## 4. Results

### Metrics

The token, cost, and timing figures below are per-trial averages over the valid trials.

| Metric                       | Value      | Comment                                                                    |
|------------------------------|------------|----------------------------------------------------------------------------|
| Success rate                 | 85.3%      | 1,286 / 1,507.                                                             |
| Tasks attempted              | 1,507      | All unique CyberGym Level 1 tasks, using only the latest attempt.          |
| Tasks succeeded              | 1,286      | Reward 1 with no trial exception.                                          |
| Tasks failed                 | 221        | Reward 0, missing reward, or a trial exception.                            |
| Input tokens                 | 11,419,670 | Average non-cached prompt tokens per attempt.                              |
| Cache read tokens            | 53,072,411 | Average cached prompt tokens per attempt.                                  |
| Output tokens                | 572,423    | Average generated tokens per attempt.                                      |
| Provider-reported cost (USD) | $4.59      | Average per attempt; Nemotron usage was not priced.                        |
| Wall-clock time (min)        | 58         | Average start-to-finish time, including setup and verification.            |
| LLM requests                 | 764.2      | Average completed NOOA journal call records per attempt.                   |

The provider-reported cost is incomplete and must not be interpreted as the full average cost of an attempt: GLM-5.2 and DeepSeek returned positive billing telemetry, while Nemotron returned token counts but no cost.

#### Per-model breakdown

| Model | Input tokens | Cache read tokens | Output tokens | LLM requests | time_cost_sec | Cost/trial | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nvidia/deepseek-ai/deepseek-v4-flash` | 1,606,399 | 11,454,139 | 209,427 | 153.6 | 2,466.1 | $0.4585 | $691.00 |
| `nvidia/nvidia/nemotron-3-ultra` | 7,905,902 | 14,934,723 | 129,740 | 223.7 | 2,637.0 | $0.0000 | $0.00 |
| `nvidia/zai-org/glm-5.2` | 1,907,192 | 26,680,962 | 233,346 | 386.9 | 4,101.3 | $4.1312 | $6,225.73 |
| `result.json` minus completed journal calls | 177 | 2,587 | -90 | — | — | $0.0003 | $0.51 |

The final row is the small reconciliation delta between `result.json` token accounting and completed journal calls. `time_cost_sec` is cumulative Finder, Expander, and Reviewer lifecycle time, so concurrently running agents count separately.

### Comparisons

Leading one-trial results from the official CyberGym Level 1 leaderboard, retrieved from [cybergym.io](https://www.cybergym.io/cybergym/) on 2026-08-14, with this work inserted according to its final score.

| #  | Submission        | Model(s)                                          | Score     | Date           | Source                                                                                                                                                     |
|----|-------------------|---------------------------------------------------|-----------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Sangfor AI        | DeepSeek V4 Flash                                 | 93.2%     | 2026-08-08     | [Sangfor AI](https://github.com/Sangfor-AI/cybergym-submission-sangfor-ai-v2)                                                                               |
| 2  | Whitzard          | DeepSeek V4 Flash                                 | 91.2%     | 2026-08-07     | [Fudan Whitzard](https://github.com/WhitzardAgent/Whitzard)                                                                                                 |
| 3  | MDASH             | GPT-5.4, Claude Opus 4.6, Claude Sonnet 4.6       | 91.0%     | 2026-06-17     | [Microsoft](https://www.microsoft.com/en-us/security/blog/2026/06/17/beyond-the-benchmark-advancing-security-at-ai-speed/)                                  |
| 4  | Wiz Atlas         | GPT-5.5, Claude Opus 4.6                          | 90.9%     | 2026-07-27     | [Wiz](https://www.wiz.io/blog/atlas-ai-vulnerability-researcher)                                                                                            |
| 5  | DoGNAVY           | GLM-5.2                                           | 90.8%     | 2026-08-03     | [DARKNAVY](https://deepsec.darknavy.net/blog/cybergym)                                                                                                      |
| 6  | Crystalline       | Claude Opus 4.6                                   | 89.6%     | 2026-06-08     | [Independent researcher](https://github.com/synchopate/cybergym-logos)                                                                                      |
| 7  | GPT-5.5-Cyber     | GPT-5.5-Cyber                                     | 85.6%     | 2026-06-22     | [OpenAI](https://openai.com/index/daybreak-securing-the-world/)                                                                                             |
| 8  | Velldepth Agent   | XekRung                                           | 85.3%     | 2026-08-03     | [Alibaba Security](https://alibaba-velldepth.github.io/writeups/)                                                                                           |
| 9  | **NOOA CyberGym** | **GLM-5.2, Nemotron 3 Ultra, DeepSeek V4 Flash**  | **85.3%** | **2026-08-10** | **This work**                                                                                                                                              |
| 10 | Xuanwu Atuin AI   | GLM-5.2                                           | 84.8%     | 2026-07-22     | [Tencent Xuanwu Lab](https://xlab.tencent.com/en/2026/07/17/xuanwu-atuin-cybergym-glm52/)                                                                   |

This work is not yet an official leaderboard row. The placement above uses the leaderboard's stored scores before display rounding: Velldepth is 85.340%, while this work is 1,286 / 1,507 = 85.335%; both display as 85.3%. CyberGym notes that runs are stochastic and modest score differences may not reflect meaningful capability gaps.

## 5. Artifacts

| Item                                     | Link                                                               |
|------------------------------------------|--------------------------------------------------------------------|
| NOOA CyberGym agent code                 | [Link](nooa_cybergym/agent.py)                                     |
| ATIF trajectories                        | [Link](task_artifacts) (`trajectory.json` files)                    |
| Logs                                     | [Link](task_artifacts) (`output.txt` files)                         |
| PoC submissions                          | [Link](task_artifacts) (`submissions.zip` archives)                 |
| Verifier results                         | [Link](task_artifacts) (`result.txt` files)                         |

The PoC submissions and accompanying artifacts (trajectories, logs, results) shared here come from a separate run over 10 tasks, not from the run submitted to the leaderboard. This run used the exact same agent code. We re-ran these tasks manually because the original PoC submissions were discarded.

## 6. Conclusions

On CyberGym Level 1, the NOOA CyberGym agent solves 1,286 of 1,507 tasks (85.3% pass@1), which would place ninth among the comparison results listed above. It reaches this level with no cybersecurity domain knowledge, exploit templates, or benchmark-specific hints, only a generic vulnerability-validation workflow expressed as an object-oriented NOOA multi-agent system. The result is therefore attributable to the agent architecture and underlying models rather than task-specific engineering, and it shows that a fully open-source agent with open-weight models can compete with proprietary systems on realistic security tasks. The approach is computationally intensive: an average attempt used 764 model calls and more than 64 million prompt tokens including cache reads.
