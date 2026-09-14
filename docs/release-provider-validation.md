# Provider checks before drafting a release

The existing release gate (`scripts/make_release.py`) runs seven real-provider
tests on the candidate through NVIDIA Inference Hub, before the capability
comparison and draft creation. They do not run on PRs. GitLab remains the release
controller; GitHub's publishing workflow is unchanged.

## What and why

- OpenAI, Anthropic Sonnet and Gemini: three calls each check native reasoning,
  exact replay after closing/reopening SQLite, and a nonzero cache read while a
  trailing dynamic context block changes. We do not require a fixed hit rate.
- DeepSeek, Kimi, GLM and Qwen: two calls each check readable reasoning and a
  tool exchange after SQLite resume, including the exact `reasoning_content`
  field in the next HTTP request.
- Offline contract tests remain on every PR. Live checks catch gateway and SDK
  changes that mocks cannot detect; they are smoke tests, not capability or
  answer-quality evaluations. The existing capability gate still runs separately.

## Budget and setup

One run makes at most 17 requests, with retries disabled. Output limits total
27,648 tokens; typical output is much smaller. The closed-provider cache cases
use roughly 90,000 input tokens in total, plus the small four-model tool loops.
Do not budget as if cache hits were guaranteed. This budget is additional to
capability evaluation, including in reduced release rehearsals. There is a
15-minute timeout for the seven cases.

The gate uses `NVIDIA_INFERENCE_API_KEY`, or the controller's existing
`NVIDIA_INTERNAL_API_KEY` if the former is unset. The runner must reach
`https://inference-api.nvidia.com`. No GitHub Actions secret or new workflow is
needed. The test modules pin model routes; a retired route fails visibly and its
replacement must be reviewed. Missing credentials, failed calls, skipped cases,
or incomplete reports prevent drafting a release. Maintainers can rerun after
investigating an outage, spending another test budget.

## Manual run without preparing a release

Export the Hub credential, then run only these cases:

```sh
NOOA_RUN_CACHE_RESUME_LIVE=1 NOOA_RUN_OPEN_MODEL_REPLAY=1 \
uv run --frozen pytest -q -m integration --reruns 0 \
  tests/integration/test_cache_resume_live.py::test_reasoning_and_prompt_cache_survive_sqlite_resume \
  tests/integration/test_open_model_tool_reasoning_live.py::test_open_model_tool_reasoning_after_sqlite_resume
```

For this direct pytest invocation use `NVIDIA_INFERENCE_API_KEY`; the release
runner performs the internal-key aliasing. Do not invoke the full release gate
just to run these tests: that also runs the larger capability comparison.

`NOOA_TEST_OMITTED_REASONING=1` adds one continuation per open model with
`reasoning_content` removed after SDK serialization. This is a manual diagnostic,
not a release requirement: accepting the omitted field does not prove reasoning
was retained, and not every gateway requires it. The release runner clears it.

## Code walkthrough

`provider_checks()` in `scripts/make_release.py` selects the seven tests, sets
their opt-in flags and disables retries. A fresh private directory holds the
JUnit report (including model and per-call token/cache usage) and SQLite sessions,
avoiding stale results from previous runs.
These session databases contain provider state: keep them in private GitLab
artifacts, never attach them to public release notes.

The function requires seven passing cases because pytest exits successfully even
when everything is skipped. It records the outcome and report path in the existing
release manifest. CI and local release paths both call it before drafting; the
GitHub publishing workflow still only builds and uploads after human approval.

`tests/test_make_release.py` checks the gate without paid calls: missing keys,
process failures, skipped/partial/malformed/missing reports, and the rule that a
provider failure prevents draft creation.
