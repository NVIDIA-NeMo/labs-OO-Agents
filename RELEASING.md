# Releasing

`nooa`, `nooa-cli`, `nooa-acp`, `nooa-memory` and `nooa-bench` release together from the
same commit. **The version comes from the git tag** — there is no `version =`
in any `pyproject.toml` and no bump step. On tag `v0.0.9` the wheels are
`0.0.9`; between tags they are `0.0.9.devN`.

## Cutting a release

```bash
git checkout main && git pull
uv run python scripts/make_release.py v0.0.9
```

The script runs everything in order and stops at two prompts — after the
capability report, and after the draft notes:

| Step | Fails the run? |
|---|---|
| Preflight — on `main`, clean, in sync with origin, tag unused | yes |
| Lint, SPDX headers, unit tests | yes |
| Build 5 wheels, version == tag, smoke import | yes |
| Capability diff vs the previous release, 4 models × 3 runs | only below the floor |
| `gh release create --draft`, capability report appended | — |
| `gh release edit --draft=false` → triggers `publish.yml` | — |

Publishing uploads to PyPI via Trusted Publishing. Each package waits on its
`pypi-<package>` GitHub Environment, so a reviewer approves before upload.

### Capability gate

Both arms run fresh: HEAD in the working tree, the previous tag in a temporary
worktree. Comparing against a stored baseline cannot tell a real regression
from the endpoint behind a model alias changing.

The only hard threshold is a **floor** on the stable tier (60%) — clearing it
requires typing `OVERRIDE`. Everything else (collapses, new error types, drops
beyond ±5 points) is reported for a human to judge. An arm where >50% of
samples error is rejected as infrastructure failure rather than reported as a
result. Results cache under `tmp/release-check/`, so aborting at a prompt does
not mean paying for another run.

### Flags

| Flag | Use |
|---|---|
| `--checks-only` | run everything, print the report, touch nothing |
| `--dry-run` | print the `gh` commands instead of running them |
| `--skip-capability` | docs-only releases |
| `--models` / `--runs` / `--limit` | cheap rehearsal; requires `--checks-only` |

Rehearse without spending real money:

```bash
uv run python scripts/make_release.py v0.0.9 --checks-only \
  --models claude-haiku --runs 1 --limit 1
```

Report logic is covered by `tests/test_make_release.py`.

## One-time PyPI setup

Each project needs a pending publisher at
<https://pypi.org/manage/account/publishing/> — owner `NVIDIA-NeMo`, repo
`labs-OO-Agents`, workflow `publish.yml`, and a **distinct environment per
package**: `pypi-nooa`, `pypi-nooa-cli`, `pypi-nooa-acp`, `pypi-nooa-memory`, `pypi-nooa-bench`.
PyPI keys a pending publisher on (owner, repo, workflow, environment), so a
shared environment makes the second registration fail. The matching GitHub
Environments must exist too.

Repeat on <https://test.pypi.org> with `testpypi-<package>` names. Running the
**Publish** workflow manually always targets TestPyPI.

> Every `uses:` in `publish.yml` must be an `actions/*` action. This org
> enforces an allowlist, and a disallowed action fails the whole workflow at
> startup — that is what left CI dead for eight days (PR #50).
