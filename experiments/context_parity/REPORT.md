# Context parity result

Tested 2026-09-08 against `fbbbfb16d68d66dc1ff029a9c06844d3b900e29d`
(`main`). Implementation commit: `5af7a04c2867aadbfd63890c05bdc6810e20ee72`.

## Result

**PASS.** Five deterministic scenarios produced six LLM requests and 29 messages
in each arm. After documented normalization, there was no content, role, order,
tool-contract, or output-schema difference.

Raw captures differ only in generated IDs and the intended rendering change:
the old formatter wrapped trailing blocks in `<context>` and used single-newline
separators; the new formatter emits the same blocks directly with double-newline
separators.

Artifacts are generated under
`experiments/context_parity/results/<baseline>_vs_<candidate>/`:
exact captures, logs, manifest, and diff report. Reproduce with:

```bash
uv run python experiments/context_parity/run.py --baseline main --candidate HEAD
```

## Live verification

NVIDIA internal inference was tested with
`openai/openai/openai/gpt-5.6-terra` at
`https://inference-api.nvidia.com/v1/`:

- Default view: static, skill, prior user/assistant events, current task, dynamic
  state, and Predict strategy instructions were present in the expected roles.
- Custom view: received model/provider from `CurrentCall`; manager context was
  absent as requested.
- Quickstart 02: structured Predict output passed.
- Quickstart 03: two-turn CodeAct tool execution passed with valid assistant/tool
  pairing and stable system context.
- All OTel traces completed with zero error spans.

Local raw traces are grouped by session under `tmp/terra-context-e2e/{journal,otlp}`.
Final sessions are `terra-{default,custom}-context-v7` and
`terra-parity-final-v7-quickstart-{02,03}`.

## Regressions found and fixed

1. Protected dynamic blocks lost their expression metadata. Restored
   `source_dynamic` independently of block protection.
2. `LLMComplete.dynamic_context` became empty after removing the provider-visible
   wrapper. Reconstructed the legacy trace-only envelope from renderer block parts;
   provider messages are unchanged.
3. Standalone generation functions lacked the new public `active_skills()` seam.
   Their adapter now returns an empty skill tuple.

No design change was required. Verification: 509 focused tests passed, plus both
live quickstarts and the default/custom live audit.
