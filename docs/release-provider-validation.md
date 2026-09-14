# Provider checks before drafting a release

The release gate (`scripts/make_release.py`, `provider_checks()`) runs seven
real-provider tests on the candidate before the capability comparison and draft
creation. They do not run on pull requests.

- Three closed-provider families check native reasoning, exact replay after
  closing and reopening SQLite, and a nonzero cache read while a trailing
  dynamic context block changes. No fixed hit rate is required.
- Four open-model families check readable reasoning and a tool exchange after
  SQLite resume, including the exact `reasoning_content` field in the next
  HTTP request.
- Offline contract tests remain on every pull request. The live checks are
  smoke tests for gateway and SDK changes that mocks cannot detect, not
  capability or answer-quality evaluations.

The test logic is public and lives in `tests/integration/test_cache_resume_live.py`
and `tests/integration/test_open_model_tool_reasoning_live.py`. The model routes
are not: each case resolves a registry alias named `release-gate-<family>`
through `tests/integration/_release_gate.py`. Those aliases, with their
endpoint, credential variable and client type, come from a bundled-config
package installed on the release runner. Without it every case skips, and the
gate rejects skips, so a release cannot be drafted unless the checks ran.

One run makes at most 17 requests with retries disabled and a 15-minute
timeout. The runner requires seven passing cases, records the outcome and the
JUnit report path in the release manifest, and keeps reports and session
databases in private artifacts; they contain provider state and must not be
attached to public release notes. `tests/test_make_release.py` checks the gate
without paid calls: missing credentials, process failures, and skipped,
partial, malformed or missing reports.

Budgets, credentials, the alias definitions, manual invocation and recorded
results are maintained with the private release controller documentation.
