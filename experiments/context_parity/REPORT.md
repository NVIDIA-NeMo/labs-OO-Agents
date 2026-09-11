# Context parity result

Tested 2026-09-11 against `origin/main` at `f1c2587b` and the context-view
implementation at `a814d9d2`.

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
- Four sessions contain 32 spans, five LLM calls, five code-execution spans, and no
  error spans or recorded exceptions.

Local traces are under `tmp/terra-context-e2e-final-v5/{journal,otlp}`.

## Local verification

- `7223 passed, 7 skipped, 238 deselected, 3 xfailed` for the release-equivalent
  suite (`not integration and not stress`).
- Context, replay, middleware, and parity-focused suites passed; the final focused
  formatter/view run passed 259 tests.
- Ruff, formatting, SPDX validation, and Pyright passed.
- GPT-6 Astra approved the corrected worktree after 403 independent targeted tests.
