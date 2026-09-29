# Provider checks before drafting a release

Before it drafts a release, the release gate (`scripts/make_release.py`) runs
seven short tests against real providers. Each test saves a conversation to a
SQLite session, reopens it, and continues it with the same provider. The tests
confirm that the provider accepts the continued conversation. Mocked tests
cannot show this. These are smoke tests, not quality evaluations, and they do
not run on pull requests.

- **OpenAI, Anthropic and Gemini** (three calls each): the provider's native
  reasoning state is sent back unchanged after the session is reopened, and
  the provider still reports a prompt-cache read when only a trailing block of
  dynamic context changes. No particular cache hit rate is required.
- **DeepSeek, Kimi, GLM and Qwen** (two calls each): readable reasoning and a
  tool call survive the reopen, and the next request carries the same
  `reasoning_content`.

## Where the models come from

The test code is public: `tests/integration/test_cache_resume_live.py` and
`tests/integration/test_open_model_tool_reasoning_live.py`. The model routes and
credentials are not. Each case uses the registry alias `release-gate-<family>`,
which comes from the private model-alias package. In CI, the release runner
receives that package with `--internal-wheel`. A local release uses the aliases
installed in the developer's environment, as its capability comparison does.

The aliases must come from that package. If a user, project or
`NEMO_OO_LLM_CONFIG` registry file defines one of them, the gate fails instead
of testing a different route.

## When the gate fails

Before the build, the runner checks that every alias resolves, has an endpoint,
and has its credential variable set. This check makes no provider calls. After
the build, it runs the seven cases: at most 17 requests, no retries, and a
15-minute limit. A skipped, failed or missing case stops the release before a
draft is created. Reports and session databases stay in the private job
artifacts. They contain provider state, so do not attach them to public release
notes.

`--skip-capability` (local runs only) skips these checks and the capability
comparison.

## Running the checks by hand

```sh
uv run --frozen python -m tests.integration._release_gate  # setup check, no provider calls

NOOA_RUN_CACHE_RESUME_LIVE=1 NOOA_RUN_OPEN_MODEL_REPLAY=1 \
uv run --frozen pytest -q -m integration --reruns 0 \
  tests/integration/test_cache_resume_live.py::test_reasoning_and_prompt_cache_survive_sqlite_resume \
  tests/integration/test_open_model_tool_reasoning_live.py::test_open_model_tool_reasoning_after_sqlite_resume
```

The two variables enable tests that spend provider tokens. Without them, the
tests skip, so the ordinary integration suite never runs them by accident.
Budgets, alias definitions and recorded results are kept with the private
release controller documentation.
