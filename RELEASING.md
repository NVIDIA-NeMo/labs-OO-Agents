# Releasing

`nooa`, `nooa-cli`, `nooa-acp`, `nooa-memory`, and `nooa-bench` release together from one
commit. The version comes from the Git tag: on `v0.0.10` the distributions are
`0.0.10`; between tags they are development versions.

## Normal release path

Releases are gated by a manually started pipeline on protected `main` in the
private `interactive-agents/nooa-dev` GitLab project. Supply both:

- `NOOA_RELEASE_TAG`: a new canonical `vX.Y.Z` version;
- `NOOA_RELEASE_SHA`: the full 40-character SHA of a commit currently reachable
  from public GitHub `main`.

The controller freezes that SHA. A later push to `main` does not change the
candidate under test. The job builds the internal NVIDIA model-alias wheel from
the controller commit, clones this repository at the candidate, and calls
`scripts/make_release.py --ci`. The public runner remains the implementation of
the gate; GitLab YAML only provisions and invokes it.

The strict gate performs:

1. Ruff lint and formatting, SPDX checks, unit tests, and explicit OS sandbox
   containment tests.
2. Builds all five wheels and source distributions under a temporary local tag,
   verifies their versions, and smoke-tests imports and `nooa --version` in a
   clean environment.
3. Runs the full capability suite for the candidate and previous release, fresh
   and back-to-back: four gate models, three runs, full data, no response cache.
4. Writes private results, traces, distributions, checksums, a JSON manifest,
   and sanitized public notes to the GitLab job artifacts.
5. After every hard gate passes, creates or safely updates one GitHub **draft**
   targeting the exact tested SHA.

The candidate and baseline environments receive the same explicit
`nemo-oo-agents-nvidia` wheel. Strict CI never copies `.env`, discovers ambient
packages, or reuses capability results from another candidate. Each sample,
stalled run, and complete arm has a timeout; completed JSONL evidence remains
available when a later sample fails.

The stable-tier 60% floor, unusable/missing results, and an arm with more than
50% errors are hard failures. Collapses, new errors, removed tests, and drops
beyond the noise band are advisory: they are prominent in the draft but remain
a human judgment when hard gates pass.

## Human approval

The GitLab job can create only a draft. It has no command that publishes a
release or uploads to PyPI.

- **Accept:** inspect the draft and private GitLab evidence, then click GitHub's
  **Publish release** button.
- **Reject:** delete the draft with its tag, or leave it unpublished until a
  maintainer performs cleanup. A subsequent pipeline may reuse only a draft
  whose target is the identical frozen SHA; any different target fails closed.

Publishing is the single human approval. `.github/workflows/publish.yml` listens
for `release: published` and automatically rebuilds, smoke-tests, and uploads
all five packages to PyPI using Trusted Publishing. It also builds and
smoke-tests the version's Bullseye `nooa-bench` Scaled Evals sidecar, publishes
an `X.Y.Z` tag to `ghcr.io/nvidia-nemo/nooa-bench-agent`, and attaches its
descriptor, source lock, immutable image digest, and checksums to the GitHub
Release. Despite their names, the current `pypi-*` GitHub
Environments have no configured reviewer protection, so there is no second
approval after **Publish release**.

The GitHub sidecar is deliberately marked `internal-promotion-required`. It is
not automatically registered with production Scaled Evals: the production path
must mirror the manifest's exact GHCR digest into the approved registry, sign
that digest with the internal CI Toolkit, validate the copied runtime, and
create a new immutable Scaled Evals agent-bundle record. A package release and
an agent-bundle registration therefore share a version but remain separate
supply-chain events.

## Evidence and recovery

GitLab retains the release evidence for 90 days. Start with `job-summary.md` and
`release-manifest.json`; the latter indexes exact commits, lock/config hashes,
toolchain identity, check outcomes, capability scope/results, distribution
checksums, and draft identity. Raw `.noo-eval.jsonl` files and traces are private
artifacts and must not be copied into the public draft.

Hard-gate failure or capability-infrastructure failure before draft creation
creates no draft. Fix the problem and start a new pipeline with the same tag
and SHA. A failure after `gh release create` or `gh release edit` may leave a
draft requiring reconciliation; rerun to update an exact matching draft without
duplication. If a draft has the wrong target or a release is already published,
the runner stops instead of mutating it.

Use the controller's one-model/one-run/one-sample rehearsal mode to exercise
setup and reporting cheaply; rehearsals can never create a draft. Before first
production use, run the complete gate once with draft creation disabled and
review all artifacts.

Before requesting review of coordinated release-process changes, the private
controller may run an unmerged rehearsal against a canonical GitHub
`refs/pull/<number>/head` ref and its exact SHA. That mode may relax only the
candidate's reachability from GitHub `main`; it remains reduced-scope, requires
the private controller to authenticate the pull ref, runs every deterministic,
containment, build, smoke, and evidence check, and can never create a draft.

## Publication rehearsal and artifact promotion

A manual run of `publish.yml` always targets TestPyPI and is the safe way to
rehearse its build, OIDC, and upload jobs without consuming a production PyPI
version. The production trigger remains `release: published`; do not publish a
throwaway GitHub release to test it because that event intentionally reaches
real PyPI.

The current first increment rebuilds after publication from the exact published
tag. The candidate artifacts and checksums are retained for review, but they are
not yet promoted. A follow-up should attach the checked artifacts to the draft
and make `publish.yml` download and verify those exact files while preserving
the existing PyPI OIDC identities.

## Emergency local fallback

If the private controller is unavailable, a trusted maintainer may run the
public script from a clean, current `main` checkout:

```bash
uv run python scripts/make_release.py v0.0.10
```

This is deliberately narrower than the old workflow: it may create a draft but
cannot publish it. It retains local compatibility for model aliases, prompts
before drafting advisory results, and still blocks the capability floor. Use it
only for recovery, record the evidence separately, and publish only through the
GitHub draft UI.

## Trusted Publishing setup

Each project needs a publisher configured for owner `NVIDIA-NeMo`, repository
`labs-OO-Agents`, workflow `publish.yml`, and its distinct environment:
`pypi-nooa`, `pypi-nooa-cli`, `pypi-nooa-acp`, `pypi-nooa-memory`, or
`pypi-nooa-bench`. Repeat
with `testpypi-*` environments on TestPyPI.

Every `uses:` entry in `publish.yml` must remain compatible with the NVIDIA
organization's GitHub Actions allowlist.

The repository's `GITHUB_TOKEN` also needs `packages: write` permission for the
`build-agent-bundle` job. The workflow grants that job-level permission. On the
first release, an organization owner may additionally need to allow package
creation or link the new `nooa-bench-agent` container package to this
repository. Set its visibility and internal pull access according to NVIDIA's
registry policy; production automation should always consume the recorded
digest rather than relying on visibility or a floating tag.
