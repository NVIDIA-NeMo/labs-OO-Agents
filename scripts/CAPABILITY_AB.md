# Capability A/B testing

`scripts/capability_ab.py` compares any two git revisions across the capability
suite. Both revisions run fresh in isolated worktrees with the same credentials,
model endpoints, dependency extras, run count, and concurrency. The resulting
Markdown and JSON reports include aggregate, model, tier, error, case-collapse,
token, run-variance, and paired task-bootstrap statistics.

## Full comparison

```bash
set -a
source .env
set +a

uv run python scripts/capability_ab.py main HEAD \
  --models claude-haiku,gpt-5.4-mini,nemotron3-nano-30b,claude-opus-4-8 \
  --runs 3 \
  --parallel 40
```

Without `--models`, the runner uses `agent_models` from the selected config.
Use `--require-clean` in automation when review findings should produce a
non-zero exit code. Infrastructure-invalid arms always fail.

## Config semantics

A relative `--config` path is resolved separately inside each revision. This is
appropriate for release checks because changes to capability agents, data, and
scorers are part of the revision being reviewed.

An absolute `--config` path is shared verbatim between arms. This holds the
harness constant for experiments on framework, model, API, agent, or prompt
changes and also permits private model definitions without committing them.

```bash
uv run python scripts/capability_ab.py BASE_SHA HEAD \
  --config /absolute/path/to/six-model-config.yaml \
  --models model-a,model-b,model-c
```

## Reuse and arm order

Runs are fresh by default. `--reuse` reuses an arm only when its revision,
config content, models, repetitions, concurrency, filters, timeout, and cache
settings match its recorded signature. Candidate-first order matches the
release workflow; use `--base-first` when desired and record the choice.

## Release integration

`scripts/make_release.py` calls the same `run_ab()` implementation with the
release model matrix, policy thresholds, and result reuse enabled. The release
script remains responsible for version tags, builds, human approval, GitHub
release creation, and publishing; it no longer owns a separate A/B engine.
