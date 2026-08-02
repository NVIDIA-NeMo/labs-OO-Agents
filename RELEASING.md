# Releasing

Five workspace packages release together from the same git commit:

- **`nooa`** — the core framework
- **`nooa-cli`** — the `nooa` command and REPL
- **`nooa-acp`** — the ACP coding agent
- **`nooa-memory`** — the long-term memory subsystem
- **`nooa-bench`** — the benchmark agent and Harbor runner

The version is derived from the git **tag** at build time by
[`uv-dynamic-versioning`](https://github.com/ninoseki/uv-dynamic-versioning).
There is no `version = "..."` in any `pyproject.toml` and no manual bump step.
**Tagging the commit is the release ceremony.**

## Versioning

The version comes from the last `vX.Y.Z` tag reachable from the commit, plus
the distance to that tag:

| Repo state | Version |
|---|---|
| Exactly on tag `v0.0.6` | `0.0.6` |
| 5 commits past `v0.0.6` | `0.0.7.dev5` |
| No `vX.Y.Z` tag reachable yet | `0.0.1.dev<distance>` |

> `fallback-version = "0.0.6"` in `pyproject.toml` is used **only** when git
> is unavailable (e.g. building from an unpacked sdist with no `.git/`).
> Whenever git is present, the version is derived from `git describe`.

This is a `0.x` **research preview** — the public API is not yet stable and may
change between releases (per [SemVer](https://semver.org/), `0.y.z` signals
initial development).

## Cutting a release

Publishing to PyPI is automated by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). **Publishing a
GitHub Release is the release ceremony** — the release's tag is what
`uv-dynamic-versioning` turns into the version.

```bash
git checkout main && git pull
gh release create v0.0.7 --title "NOOA 0.0.7" --generate-notes --draft
# review the draft notes, then publish it:
gh release edit v0.0.7 --draft=false
```

Publishing the release triggers the workflow, which:

1. Builds all five packages from the tagged commit.
2. Fails the run if the built version does not match the tag, or is a `.devN`
   version (which means the tag was not reachable from the checked-out commit).
3. Smoke-tests the wheels in a clean venv (imports + `nooa --version`).
4. Uploads to PyPI via **Trusted Publishing** (`uv publish`) — no API tokens.
5. Attaches the wheels and sdists to the GitHub Release.

Each upload waits on its `pypi-<package>` GitHub Environment, so a required
reviewer there gives a second pair of eyes before the irreversible step.

> **Why no third-party actions.** Every `uses:` in `publish.yml` is an
> `actions/*` action. This org enforces a GitHub Actions allowlist, and a
> disallowed action fails the *entire workflow* at startup — that is what left
> CI dead for eight days (PR #50). A publish workflow that cannot start is one
> that silently never ships, so uv is installed from a pinned install script
> and does the upload itself.
>
> The tradeoff is **no PEP 740 attestations**: `uv publish` uploads them but
> [does not generate them](https://docs.astral.sh/uv/guides/package/), and the
> action that does (`pypa/gh-action-pypi-publish`) may not be allowlisted.
> Worth revisiting if it is added to the allowlist, or once uv can generate
> them. Trusted Publishing itself is unaffected.

### Dry run against TestPyPI

Run the **Publish** workflow manually (Actions → Publish → Run workflow). This
exercises the identical build, version check, and smoke test.

A manual run always targets TestPyPI — there is no index selector. Real PyPI is
reachable only by publishing a GitHub Release, so a mis-click here cannot burn
a version number on PyPI.

### Doing it by hand

`--no-sources` disables `tool.uv.sources` so the build is exercised the way a
non-uv consumer sees it — the [uv packaging
guide](https://docs.astral.sh/uv/guides/package/) recommends it for release
builds.

```bash
rm -rf dist
for p in nooa nooa-cli nooa-acp nooa-memory nooa-bench; do
  uv build --no-sources --package "$p" --out-dir dist
done
uvx twine check dist/*
uv venv /tmp/nooa-smoke --python 3.12
VIRTUAL_ENV=/tmp/nooa-smoke uv pip install dist/nooa-*.whl dist/nooa_cli-*.whl \
  dist/nooa_acp-*.whl dist/nooa_memory-*.whl dist/nooa_bench-*.whl
/tmp/nooa-smoke/bin/python -c "import nooa, nooa_cli, nooa_acp, nooa_memory, nooa_bench; print(nooa.__version__)"
/tmp/nooa-smoke/bin/nooa --help
/tmp/nooa-smoke/bin/nooa-acp --help
```

### Pre-release tags

Annotated tags like `v0.0.6-rc1` build as `0.0.6rc1` (PEP 440 normalized). Mark
the GitHub Release as a pre-release; PyPI will not serve it to plain
`pip install nooa`.

## One-time PyPI setup

Each of the five project names needs a **pending publisher** registered at
<https://pypi.org/manage/account/publishing/> before its first upload. Owner
`NVIDIA-NeMo`, repository `labs-OO-Agents`, workflow `publish.yml` for all five
— but the **environment name differs per package**:

| PyPI Project Name | Environment name |
|---|---|
| `nooa` | `pypi-nooa` |
| `nooa-cli` | `pypi-nooa-cli` |
| `nooa-acp` | `pypi-nooa-acp` |
| `nooa-memory` | `pypi-nooa-memory` |
| `nooa-bench` | `pypi-nooa-bench` |

> **Why one environment per package.** PyPI keys a *pending* publisher on
> (owner, repo, workflow filename, environment). If all five shared one
> environment, the second registration fails with *"a pending trusted publisher
> matching this configuration has already been registered for a different
> project name"* — PyPI cannot tell which project to create on first upload.
> The restriction lifts once a project exists, but the per-package environment
> is kept because it also gives per-package approval gates.

Repeat on <https://test.pypi.org> using `testpypi-<package>` environment names
for dry runs. After the first successful upload each pending publisher becomes
a normal one.

The matching **GitHub Environments** must exist too (Settings → Environments):
`pypi-nooa`, `pypi-nooa-cli`, `pypi-nooa-acp`, `pypi-nooa-memory`,
`pypi-nooa-bench`, and the five `testpypi-*` equivalents.

## Distribution

```bash
uv add nooa nooa-cli nooa-acp
```

Installing straight from a tag also works, and does not require a release:

```bash
uv add "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@v0.0.7"
```

## Cross-package dependencies

`nooa-cli`, `nooa-acp`, `nooa-memory`, and `nooa-bench` depend on the core
`nooa` package. They are always released together at the same derived version,
so their dependency on `nooa` carries **no version floor** — CI never rewrites
it.
