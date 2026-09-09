# Context parity result

Tested 2026-09-09 against the rebased pre-boundary branch commit
`8c2fee9cb8bd5c297081f0b3594209221d97b973` and current `origin/main`
`ee61c9012f2e4135eabba1e5db80c8e991a88184`.

## Result

**PASS.** Each comparison covered six deterministic scenarios, seven LLM requests,
and 33 messages. After documented transport and legacy-rendering normalization,
there was no content, role, order, tool-contract, or output-schema difference.
Against the pre-boundary commit, both arms contained 25,629 serialized message
characters.

Reproduce against the checked-out implementation with:

```bash
uv run python experiments/context_parity/run.py \
  --baseline 8c2fee9cb8bd5c297081f0b3594209221d97b973 --candidate HEAD
```

Artifacts are generated under
`experiments/context_parity/results/<baseline>_vs_<candidate>/`.

## Live verification

NVIDIA internal inference was tested with
`openai/openai/openai/gpt-5.6-terra` at
`https://inference-api.nvidia.com/v1/`:

- Default view: static, skill, history, current task, dynamic state, and strategy
  context had the expected content and roles. The cache annotation landed exactly
  after the current task; the internal marker was absent from the API request.
- Custom view: manager context and implicit cache annotations were absent.
- Quickstart 02 structured Predict output and quickstart 03 CodeAct tool execution
  passed; assistant/tool IDs paired correctly.
- The four final trace sessions contain 29 spans, five LLM calls, three code-execution
  spans, and no error spans or recorded exceptions.

Local traces are under `tmp/terra-context-e2e/{journal,otlp}`. Final sessions are
`terra-context-boundary-{default-v4,custom-v4,quickstart-02-v4,quickstart-03-v4}`.

## Local verification

- 250 context-block tests, 362 UnifiedLLM tests, 93 related runtime/parity tests, and two
  loopback HTTP tests passed.
- Ruff checks, formatting, SPDX validation, and focused Pyright checks passed.
- GPT-6 Astra independently approved the implementation and post-rebase port after
  focused runs of 122 and 83 tests.
- The repository-wide run reaches an existing asyncio default-executor teardown
  hang, reproduced in unchanged LiteLLM bridge/context tests; affected focused
  suites complete normally.
