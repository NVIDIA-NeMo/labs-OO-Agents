# NVIDIA-labs Object Oriented Agents — Examples

This page is a catalog of runnable examples. For the explanation of how NOOA's
ideas fit together—and the design practices behind them—read the
[framework tour](../docs/tour.md).

## Run an example

From a repository checkout:

```bash
uv sync
uv run python examples/quickstart/01_first_generation_method.py
```

Most quickstarts use the model selector in
[`nooa.util.quickstart`](../src/nooa/util/quickstart.py), which chooses a default
from the provider credentials in your environment. See the root README's
[Choose a model](../README.md#1-choose-a-model) section when adapting an example
to use an explicit model. For Ollama and vLLM setup, see
[Running NOOA With Local Models](../docs/local-models.md).

NOOA can execute LLM-generated Python. Run code-executing agents in an OS-level
sandbox, as described in the root README's [Quick Start safety note](../README.md#quick-start).

## Quickstart catalog

The files are ordered from the smallest agent to integrations and operational
features. Each file is standalone and includes its exact run command.

| # | Example | What to inspect | Additional setup |
|---|---|---|---|
| 1 | [`01_first_generation_method.py`](quickstart/01_first_generation_method.py) | The smallest generation method: signature, docstring, ellipsis, return type | — |
| 2 | [`02_structured_outputs.py`](quickstart/02_structured_outputs.py) | Pydantic contracts, field constraints, and `PredictStrategy` | — |
| 3 | [`03_codeact_tools.py`](quickstart/03_codeact_tools.py) | Regular Python methods used as deterministic tools | — |
| 4 | [`04_strategies.py`](quickstart/04_strategies.py) | Choosing between Predict and CodeAct; per-instance `ShellTools` | — |
| 5 | [`05_progressive_disclosure.py`](quickstart/05_progressive_disclosure.py) | Discovering unfamiliar runtime objects with `doc()` | — |
| 6 | [`06_tracing.py`](quickstart/06_tracing.py) | Nested spans for orchestrators, generation methods, helpers, and durable journal traces | Optional: `uv run nooa start-dev` |
| 7 | [`07_dynamic_prompts.py`](quickstart/07_dynamic_prompts.py) | Trusted instance configuration in method docstrings | — |
| 8 | [`08_context_blocks.py`](quickstart/08_context_blocks.py) | Fixed and expression-backed context blocks | — |
| 9 | [`09_summarization.py`](quickstart/09_summarization.py) | Bounded event history with automatic summarization | — |
| 10 | [`10_skills.py`](quickstart/10_skills.py) | Direct `TextSkill` attachment and model-facing `SkillRegistry` activation | — |
| 11 | [`11_mcp.py`](quickstart/11_mcp.py) | A local wiki MCP server exposed as an agent tool | `uv sync --extra mcp` |
| 12 | [`12_memory.py`](quickstart/12_memory.py) | Offline long-term memory, recall, reflection, and forgetting | `uv sync --extra memory` |
| 13 | [`13_multimodal.py`](quickstart/13_multimodal.py) | Image inputs with CodeAct and Predict | A vision-capable model |
| 14 | [`14_atif_trajectory.py`](quickstart/14_atif_trajectory.py) | Exporting ATIF trajectories for evals and downstream tooling | — |
| 15 | [`15_nemo_relay.py`](quickstart/15_nemo_relay.py) | NeMo Relay intercepts, guardrails, events, and nested generation | `uv sync --extra nemo-relay` |

If you are new to NOOA, run examples 1–6 in order. After that, choose by the
capability you need rather than treating the remaining files as required steps.

## Advanced mechanics

These examples isolate lower-level extension points. Read the source first;
they are references rather than a second tutorial.

| Example | Focus |
|---|---|
| [`codeact_event_sequence.py`](advanced/codeact_event_sequence.py) | Raw events emitted during a CodeAct run |
| [`durable_hitl.py`](advanced/durable_hitl.py) | Resume a durable workflow after an interruption, illustrated with a human-input pause |
| [`memory.py`](advanced/memory.py) | Conversation history across method calls |
| [`prefill.py`](advanced/prefill.py) | Customizing the input prefill shown to generated code |
| [`swappable_execution_engines.py`](advanced/swappable_execution_engines.py) | Replacing the default Python execution engine |
| [`tracing_langfuse.py`](advanced/tracing_langfuse.py) | Langfuse trace export |
| [`tracing_otlp.py`](advanced/tracing_otlp.py) | Generic OTLP trace export |
| [`tracing_phoenix.py`](advanced/tracing_phoenix.py) | Phoenix trace export |

## Complete systems and benchmarks

These directories contain their own setup, architecture, and operational
documentation:

- [`arc_agi_3/`](arc_agi_3/README.md) — an interactive ARC-AGI-3 solver with
  isolated agent/environment processes, reusable skills, memory variants, and a
  live fleet dashboard.
- [`benchmarks/`](benchmarks/README.md) — the minimal Harbor-compatible benchmark
  adapter and agent example.
- [`cybergym/`](cybergym/README.md) — a portfolio-style multi-agent CyberGym
  implementation with independent finder lanes and deterministic PoC
  verification.

For guided notebooks rather than standalone scripts, see the repository's
[`notebook_tutorials/`](../notebook_tutorials/README.md).

## Shared example assets

[`assets/`](assets/) contains data and helper services used by the quickstarts,
including the frontend-design `TextSkill` and the local wiki MCP server. It is
supporting material, not a separate example suite.
