# Context parity result

Tested 2026-09-11 against `origin/main` at `f1c2587b` and the context-view
implementation at `5eff5ddb`.

## Result

**PASS.** Six deterministic scenarios produced seven LLM requests and 33 messages.
After the documented transport and legacy-envelope normalization, there was no
content, role, order, tool-contract, or output-schema difference. The baseline and
candidate contained 25,817 and 25,682 serialized message characters respectively;
the raw difference is fully covered by those normalizations.

Reproduce against the checked-out implementation with:

```bash
uv run python experiments/context_parity/run.py \
  --baseline origin/main --candidate HEAD
```

Artifacts are generated under
`experiments/context_parity/results/<baseline>_vs_<candidate>/`.

## Live verification

NVIDIA internal inference was tested with `openai/openai/openai/gpt-5.6-terra` at
`https://inference-api.nvidia.com/v1/`:

- The default view rendered the intended system, API, event, state, skill, and
  strategy blocks. The internal cache-boundary marker was absent at the provider.
- A custom research context API and view progressively exposed selected research;
  unused manager context did not leak into either request.
- Quickstart 02 structured Predict output and quickstart 03 CodeAct tool execution
  passed. Assistant/tool IDs paired correctly.
- Four sessions contain 31 spans, five LLM calls, four code-execution spans, and no
  error spans or recorded exceptions.

Local traces are under `tmp/terra-context-e2e-final-v4/{journal,otlp}`.

## Local verification

- `7220 passed, 7 skipped, 238 deselected, 3 xfailed` for the release-equivalent
  suite (`not integration and not stress`).
- Context, replay, middleware, and parity-focused suites passed (340 tests).
- Ruff, formatting, SPDX validation, and Pyright passed.
