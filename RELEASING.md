# Releasing nemo-oo-agents

Three workspace packages — `nemo-oo-agents`, `nemo-labs-oo-agents-cli`, and
`nemo-oo-agents-benchmarks` — release together from the same git commit.
Version is derived from `git describe` at build time by
[`uv-dynamic-versioning`](https://github.com/ninoseki/uv-dynamic-versioning).

There is no `version = "..."` in any pyproject.toml and no manual bump step
between releases. **Tagging the commit is the entire release ceremony.**

## Continuous publishes (every main commit)

Every push to `main` triggers `build-package-*` and `publish-package-*` jobs.
The published version is derived from the last `vX.Y.Z` tag reachable from
the commit, plus the distance to that tag:

| Repo state | Wheel version |
|---|---|
| Exactly on tag `v0.3.0` | `0.3.0` |
| 5 commits past `v0.3.0` | `0.3.1.dev5` |
| No `vX.Y.Z` tag reachable yet | `0.0.1.dev<commit-count>` (until first tag — see "Bootstrapping") |

> Note: the `fallback-version = "0.2.0"` in pyproject.toml is the version
> used only when git itself is unavailable (e.g. building from an unpacked
> sdist with no `.git` dir). In CI git is always present, so the version
> is always derived from `git describe` — the fallback never fires.

Dev versions sort *above* the latest stable release and *below* the next
stable, so `--pre` consumers always get the freshest dev:

```bash
# Latest dev wheel from main
uv add nemo-oo-agents --pre

# Pin to a specific dev wheel
uv add nemo-oo-agents==0.3.1.dev5
```

`uv add nemo-oo-agents` (no `--pre`) keeps picking the last stable release.

## Bootstrapping (one-time, after this MR merges)

Until a `vX.Y.Z` tag exists in the repo, dev publishes ship as
`0.0.1.dev<commit-count>` — derived from "0 tags reachable" + the `bump = true`
config. To switch dev publishes to a meaningful base, cut the first tag:

```bash
git checkout main
git pull
git tag -a v0.2.0 -m "Initial release with dynamic versioning"
git push origin v0.2.0
```

Subsequent main pushes will then publish as `0.2.1.dev<distance>`. After
that, the rest of this doc is the entire release process.

## Cutting a stable release

```bash
git checkout main
git pull
git tag -a v0.3.0 -m "Release 0.3.0"
git push origin v0.3.0
```

That's it. The tag pipeline builds and publishes `0.3.0` of all three
packages to the GitLab Package Registry. No pyproject changes, no follow-up
MRs, no version bumps anywhere.

### Pre-release tags

CI accepts annotated tags matching `^v\d+\.\d+\.\d+([.-][a-zA-Z0-9]+)?$`,
which covers pre-release suffixes:

```bash
git tag -a v0.3.0-rc1 -m "Release candidate"
git push origin v0.3.0-rc1
```

The wheel ships as `0.3.0-rc1` (PEP 440 normalizes to `0.3.0rc1`).

## Cross-package dependencies

`nemo-labs-oo-agents-cli` and `nemo-oo-agents-benchmarks` declare their
dependency on core as a static lower-bound floor in their pyproject.toml:

```toml
"nemo-oo-agents>=0.2.0",
```

The floor reflects the actual minimum-compatible core version. **Bump it in
a normal MR when the package starts to require a newer core API** — same as
you would for any third-party dep. CI does not rewrite this declaration.

## Republishing a main commit

`uv publish` is not idempotent against the GitLab Package Registry — a
second attempt with the same version returns HTTP 400. CI does not retry
publish jobs.

If a transient failure leaves a dev publish broken, **push an empty commit**
to advance the dev version:

```bash
git commit --allow-empty -m "ci: re-trigger publish"
git push
```

Do NOT use the GitLab "Retry pipeline" button on the build stage — it
rebuilds at the same git HEAD and therefore the same derived version, which
then 400s on publish.
