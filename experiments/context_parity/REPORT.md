# Context parity result

Tested 2026-09-08 against `fbbbfb16d68d66dc1ff029a9c06844d3b900e29d`
(`main`). Implementation commit: `85924da313709991819dfa395c7bc695ebab7f27`.

## Result

**PASS.** Six deterministic scenarios produced seven LLM requests and 33 messages
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
- Quickstart 10: direct and registry-managed skills passed; the registry block was
  trailing USER context after events with its expression metadata intact.
- The five final OTel sessions contain 48 spans and zero error spans.

Local raw traces are grouped by session under `tmp/terra-context-e2e/{journal,otlp}`.
Final sessions are `terra-{default,custom}-context-v8`,
`terra-parity-final-v8-quickstart-{02,03}`, and
`terra-parity-final-v8-quickstart-10-retry`. The first quickstart-10 run is also
retained: its 1,024-token output cap truncated generated HTML; the 4,096-token
retry passed.

## Regressions found and fixed

1. Protected dynamic blocks lost their expression metadata. Restored
   `source_dynamic` independently of block protection.
2. `LLMComplete.dynamic_context` became empty after removing the provider-visible
   wrapper. Reconstructed the legacy trace-only envelope from renderer block parts;
   provider messages are unchanged.
3. Standalone generation functions lacked the new public `active_skills()` seam.
   Their adapter now returns an empty skill tuple.
4. Legacy `Skill.context_block` moved before events into SYSTEM context and lost
   expression metadata. Default shorthand now joins the normal block source before
   strategy/decorator/scoped overrides; custom skill views keep their explicit seam.

No design change was required. Verification: 574 focused tests passed, six-scenario
parity passed, and three live quickstarts plus the default/custom audit passed.
