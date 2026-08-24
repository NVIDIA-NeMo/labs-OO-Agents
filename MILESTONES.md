# Milestones

## 2026-08-20 — SkillsBench TextSkill Plumbing

Status: complete local plumbing milestone.

Implemented local NOOA SkillsBench execution for paired `no_skill` and
`text_skill` conditions:
- `BenchAgent` supports `skill_mode` and `skills_dir`.
- Task-bundled `SKILL.md` directories are discovered and exposed as NOOA
  TextSkill context only in `text_skill` mode.
- `nooa_bench.runner` forwards skill options into `_run_evaluation`.
- `nooa-skillsbench-task` runs one SkillsBench task locally through
  BenchFlow/Docker in paired conditions.
- The runner maps `.env` `API_KEY`/`API_URL` to `OPENAI_API_KEY` and
  `OPENAI_BASE_URL` without printing secrets.
- The source upload excludes local `.env` and common key material.
- The runner preserves agent logs/results, verifier outputs, rewards, and
  rollout summaries under `jobs/`.
- BenchFlow lock paths protect `/oracle`, `/solution`, `/verifier`, `/tests`,
  and `/testbed_verify`.

Verification:
- `uv run pytest packages/nooa-bench/tests/test_bench_agent.py packages/nooa-bench/tests/test_skillsbench_runner.py -q`
- `uv run ruff check packages/nooa-bench/src/nooa_bench/bench_agent.py packages/nooa-bench/src/nooa_bench/runner.py packages/nooa-bench/src/nooa_bench/skillsbench_runner.py packages/nooa-bench/tests/test_bench_agent.py packages/nooa-bench/tests/test_skillsbench_runner.py`

Paper top-5 reproduction progress:
- `mario-coin-counting`: `no_skill=0.0`, `text_skill=1.0` — reproduced lift.
- `sales-pivot-analysis`: `no_skill=0.0`, `text_skill=0.0` — no lift under
  current NOOA harness run.
- `flood-risk-analysis`: `no_skill=0.0`, `text_skill=1.0` — reproduced lift.
- `sec-financial-report`: `no_skill=0.0`, `text_skill=1.0` — reproduced lift.
- `protein-expression-analysis`: `no_skill=0.0`, `text_skill=1.0` —
  reproduced lift.

Current result: 4 of 5 completed paper examples reproduced the expected skill
lift. `sales-pivot-analysis` remains the only no-lift case under the frozen
NOOA harness prompt.

Local artifacts:
- `jobs/nooa-skillsbench/citation-check__nooa__2026-08-19__22-47-09/`
- `jobs/nooa-skillsbench-paper-top5/mario-coin-counting__nooa__2026-08-20__09-02-06/`
- `jobs/nooa-skillsbench-paper-top5/sales-pivot-analysis__nooa__2026-08-20__09-08-19/`
- `jobs/nooa-skillsbench-paper-top5/flood-risk-analysis__nooa__2026-08-20__09-13-19/`
- `jobs/nooa-skillsbench-paper-top5/sec-financial-report__nooa__2026-08-20__f4f78e76/`
- `jobs/nooa-skillsbench-paper-top5/protein-expression-analysis__nooa__2026-08-20__f4f78e76/`

Notes:
- The earlier `mario-coin-counting__nooa__2026-08-20__08-59-18` attempt was an
  infrastructure bootstrap failure caused by a task image without `curl` or
  `uv`; `_install_nooa()` now installs `curl` via `apt-get` when needed.
- Follow-up hardening confirmed `sales-pivot-analysis` was not a container,
  NOOA install, or TextSkill injection failure. Both conditions returned from
  the NOOA runner with exit code 0 and `success: true`; the verifier rejected
  the workbook because the generated sheets were static pivot-style summaries
  rather than actual Excel pivot table objects (`workbook[sheet]._pivots[0]`
  was absent). The text-skill condition also used Australian state
  abbreviations where the verifier expected full state names.
- Hardening after the sales diagnosis records activated TextSkills in
  `/logs/agent/result.json` and unit-tests the sandbox uv/curl bootstrap
  command. The agent task prompt is intentionally unchanged for cleaner
  comparison with the existing milestone runs.
- The committed milestone intentionally excludes local `jobs/` artifacts and
  the untracked `skillsbench/` checkout.

## 2026-08-20 — Script-Backed Skills Smoke Subset

Status: complete local smoke run.

Ran a 9-task CPU-only smoke subset chosen for SkillsBench tasks whose
`environment/skills` folders include bundled scripts/code. The NOOA harness
prompt remained frozen; this was an operational smoke run, not prompt tuning.

Results:
- `court-form-filling`: `no_skill=0.0`, `text_skill=0.0`
- `invoice-fraud-detection`: `no_skill=0.0`, `text_skill=0.0`
- `organize-messy-files`: `no_skill=0.0`, `text_skill=0.0`
- `pdf-excel-diff`: `no_skill=1.0`, `text_skill=0.0`
- `pptx-reference-formatting`: `no_skill=0.0`, `text_skill=0.0`
- `xlsx-recover-data`: `no_skill=0.0`, `text_skill=0.0`
- `3d-scan-calc`: `no_skill=1.0`, `text_skill=1.0`
- `weighted-gdp-calc`: `no_skill=0.0`, `text_skill=0.0`
- `powerlifting-coef-calc`: `no_skill=1.0`, `text_skill=1.0`

Current result: 0 new skill-lift cases in this smoke subset. Activated
TextSkills were recorded in each text-skill summary, so this primarily shows
that the larger-run plumbing is working and that this subset is harder/less
skill-sensitive for the current NOOA harness.

Local artifacts:
- `jobs/nooa-skillsbench-smoke-scripts/`

## 2026-08-20 — Codex Subscription Control Probe

Status: complete local Codex ACP control, with one model caveat.

Attempted an exact-model Codex ACP control first. The NOOA runner records
`openai/openai/openai/gpt-5.2`, and its LiteLLM logs show calls routed as
`openai/openai/gpt-5.2` against `https://inference-api.nvidia.com/v1`.

Exact-model attempts were not scoreable:
- With the NVIDIA endpoint key, Codex ACP normalized to bare `gpt-5.2`, and
  the provider rejected it with `key not allowed to access model`.
- With host Codex subscription auth, Codex rejected `gpt-5.2` with
  `The 'gpt-5.2' model is not supported when using Codex with a ChatGPT account`.
- Using `openai/openai/openai/gpt-5.2` or `gpt-5.2-codex` directly failed
  earlier at `session/set_model`.

Then ran a scoreable Codex subscription control using the local Codex default
model family, `gpt-5.5`, with `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `API_KEY`,
and `API_URL` explicitly unset so BenchFlow used host subscription auth
(`~/.codex/auth.json`) rather than the NVIDIA endpoint.

Control subset:
- `mario-coin-counting`
- `sec-financial-report`
- `pdf-excel-diff`
- `xlsx-recover-data`

Codex subscription results (`codex-acp`, `model=gpt-5.5`, Docker):
- `mario-coin-counting`: `no-skill=0.0`, `with-skill=1.0`
- `sec-financial-report`: `no-skill=0.0`, `with-skill=1.0`
- `pdf-excel-diff`: `no-skill=1.0`, `with-skill=1.0`
- `xlsx-recover-data`: `no-skill=0.0`, `with-skill=0.0`

Current result: 2/4 skill-lift cases on this control subset. The two paper
controls reproduced the expected lift under Codex subscription auth. The smoke
checks match the NOOA suspicion pattern: `pdf-excel-diff` is pass/pass, while
`xlsx-recover-data` is fail/fail.

Local artifacts:
- `jobs/codex-control-subscription-gpt55-noskill/2026-08-20__15-20-59/`
- `jobs/codex-control-subscription-gpt55-withskill/2026-08-20__15-38-39/`
- `jobs/codex-control-subscription-probe/2026-08-20__15-13-16/`
- `jobs/codex-control-subscription-gpt55-probe/2026-08-20__15-15-41/`
- `jobs/codex-control-gpt52-noskill/2026-08-20__15-02-55/`
- `jobs/codex-control-probe-gpt52/`
- `jobs/codex-control-probe-codexmodel/`

## 2026-08-21 — SkillsBench 10-Task gpt-5.5 Control

Status: complete scoreable local control run.

Results checkpoint commit: `b5aff107`.

Ran the corrected 10-task SkillsBench matrix with Docker and concurrency 1:
- NOOA `openai/openai/openai/gpt-5.5`, paired `no_skill` and `text_skill`.
- Codex ACP `gpt-5.5`, `no-skill`, host Codex subscription auth with
  OpenAI/API environment variables explicitly unset.
- Codex ACP `gpt-5.5`, `with-skill`, host Codex subscription auth with
  OpenAI/API environment variables explicitly unset.

Results:

| Task | NOOA no_skill | NOOA text_skill | Codex no-skill | Codex with-skill |
|---|---:|---:|---:|---:|
| `fix-visual-stability` | 0.0 | 1.0 | 0.0 | 1.0 |
| `fix-erlang-ssh-cve` | 1.0 | 1.0 | 1.0 | 1.0 |
| `video-silence-remover` | 0.0 | 0.0 | 0.0 | 0.0 |
| `dynamic-object-aware-egomotion` | 0.0 | 0.0 | 0.0 | 0.0 |
| `manufacturing-fjsp-optimization` | 0.0 | 1.0 | 0.0 | 1.0 |
| `llm-prefix-cache-replay` | 0.0 | 1.0 | 0.0 | 1.0 |
| `dapt-intrusion-detection` | 0.0 | 1.0 | 0.0 | 1.0 |
| `offer-letter-generator` | 0.0 | 1.0 | 1.0 | 1.0 |
| `parallel-tfidf-search` | 1.0 | 1.0 | 1.0 | 1.0 |
| `reserves-at-risk-calc` | 0.0 | 0.0 | 0.0 | 0.0 |

Current result:
- NOOA text_skill vs no_skill: 5 lifts, 5 ties, 0 regressions.
- Codex with-skill vs no-skill: 4 lifts, 6 ties, 0 regressions.
- Codex no-skill vs NOOA no_skill: 1 win, 9 ties, 0 losses.
- Codex with-skill vs NOOA text_skill: 0 wins, 10 ties, 0 losses.
- Codex no-skill aggregate: 3/10, 0 agent errors, 0 verifier errors.
- Codex with-skill aggregate: 7/10, 0 agent errors, 0 verifier errors.
- Failed rewards were verifier failures, not infrastructure errors.

Local artifacts:
- `jobs/nooa-skillsbench-gpt55-10/`
- `jobs/codex-control-subscription-gpt55-10-noskill/2026-08-21__15-49-49/`
- `jobs/codex-control-subscription-gpt55-10-withskill/2026-08-20__18-42-44/`

## 2026-08-23 — SkillsBench LibrarySkill 10-Task Sweep

Status: complete scoreable local LibrarySkill run.

Integrated the SkillsBench runner plumbing and the TextSkill-to-LibrarySkill
translator into this branch, then added a third NOOA skill condition:
`library_skill`. For this condition, each task-bundled TextSkill under
`environment/skills/*/SKILL.md` is translated host-side into a package-backed
NOOA LibrarySkill, validated through `SkillRegistry.discover_libs()`, mounted
into the rollout at `/skills`, and activated in `BenchAgent` as package skills.

The run used:
- SkillsBench checkout:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/skillsbench`
- Credentials file:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/.env`
- NOOA model: `openai/openai/openai/gpt-5.5`
- Sandbox: Docker
- Condition: `library_skill`
- Concurrency: 1

Validation:
- `uv run pytest packages/nooa-bench/tests/test_bench_agent.py packages/nooa-bench/tests/test_runner.py packages/nooa-bench/tests/test_skillsbench_runner.py tests/tools/test_skill_translator.py -q`
  - `63 passed`
- `uv run ruff check packages/nooa-bench/src/nooa_bench/bench_agent.py packages/nooa-bench/src/nooa_bench/runner.py packages/nooa-bench/src/nooa_bench/skillsbench_runner.py packages/nooa-bench/tests/test_bench_agent.py packages/nooa-bench/tests/test_runner.py packages/nooa-bench/tests/test_skillsbench_runner.py src/nooa/tools/skill_translator.py tests/tools/test_skill_translator.py`
  - passed
- Translation validation over the 10-task skill set translated and validated
  all 30 task skill directories.

Results:

| Task | NOOA no_skill | NOOA text_skill | NOOA library_skill |
|---|---:|---:|---:|
| `fix-visual-stability` | 0.0 | 1.0 | 0.0 |
| `fix-erlang-ssh-cve` | 1.0 | 1.0 | 1.0 |
| `video-silence-remover` | 0.0 | 0.0 | 0.0 |
| `dynamic-object-aware-egomotion` | 0.0 | 0.0 | 0.0 |
| `manufacturing-fjsp-optimization` | 0.0 | 1.0 | 0.0 |
| `llm-prefix-cache-replay` | 0.0 | 1.0 | 0.0 |
| `dapt-intrusion-detection` | 0.0 | 1.0 | 0.0 |
| `offer-letter-generator` | 0.0 | 1.0 | 1.0 |
| `parallel-tfidf-search` | 1.0 | 1.0 | 1.0 |
| `reserves-at-risk-calc` | 0.0 | 0.0 | 0.0 |

Current result:
- NOOA library_skill aggregate: 3/10.
- NOOA library_skill vs no_skill: 1 lift, 9 ties, 0 regressions.
- NOOA library_skill vs text_skill: 0 wins, 6 ties, 4 losses.
- LibrarySkill reproduced the no-skill aggregate and did not reproduce four of
  the five TextSkill lift cases from the corrected 10-task control.
- `reserves-at-risk-calc` first hit the 600s agent execution timeout with
  `reward=None`; a single rerun completed scoreably with `reward=0.0`, and the
  scoreable rerun result is the one recorded in the table.

Local artifacts:
- `jobs/nooa-skillsbench-gpt55-10-library-v2/`
- `jobs/nooa-skillsbench-gpt55-10-library-v2/reserves-at-risk-calc__nooa__library-rerun/`
- `jobs/nooa-skillsbench-gpt55-10-library-translation-validation/`

## 2026-08-24 — LibrarySkill Guidance Preservation Rerun

Committed the pre-fix experiment snapshot locally as
`64660e7d chore: snapshot library skill skillsbench run`, then patched the
TextSkill-to-LibrarySkill translator in `a84ef7a8 fix: preserve translated
skill guidance`.

Translator change:
- Preserve the original TextSkill body in the generated LibrarySkill docstring
  and README.
- Add a dynamic `context_block` that exposes preserved guidance plus a bundled
  resource index while the LibrarySkill is activated.
- Expose public `list_resources()`, `read_resource()`, and
  `read_resource_bytes()` helpers for non-script bundled resources.
- Continue hiding raw script runners and only expose generated native APIs for
  safely translated scripts.

Validation:
- `uv run pytest tests/tools/test_skill_translator.py packages/nooa-bench/tests/test_bench_agent.py packages/nooa-bench/tests/test_runner.py packages/nooa-bench/tests/test_skillsbench_runner.py -q`
  - `63 passed`
- `uv run ruff check src/nooa/tools/skill_translator.py tests/tools/test_skill_translator.py packages/nooa-bench/src/nooa_bench/bench_agent.py packages/nooa-bench/src/nooa_bench/runner.py packages/nooa-bench/src/nooa_bench/skillsbench_runner.py packages/nooa-bench/tests/test_bench_agent.py packages/nooa-bench/tests/test_runner.py packages/nooa-bench/tests/test_skillsbench_runner.py`
  - passed

Patched LibrarySkill rerun:
- SkillsBench checkout:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/skillsbench`
- Credentials file:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/.env`
- NOOA model: `openai/openai/openai/gpt-5.5`
- Sandbox: Docker
- Condition: `library_skill`
- Artifact root:
  `jobs/nooa-skillsbench-gpt55-10-library-guidance-fix-regressions/`

Results:

| Task | Previous LibrarySkill | Patched LibrarySkill |
|---|---:|---:|
| `fix-visual-stability` | 0.0 | 1.0 |
| `fix-erlang-ssh-cve` | 1.0 | 1.0 |
| `video-silence-remover` | 0.0 | 0.0 |
| `dynamic-object-aware-egomotion` | 0.0 | 0.0 |
| `manufacturing-fjsp-optimization` | 0.0 | 0.0 |
| `llm-prefix-cache-replay` | 0.0 | 1.0 |
| `dapt-intrusion-detection` | 0.0 | 1.0 |
| `offer-letter-generator` | 1.0 | 1.0 |
| `parallel-tfidf-search` | 1.0 | 1.0 |
| `reserves-at-risk-calc` | 0.0 | 0.0 |

Current result:
- Patched NOOA LibrarySkill aggregate: 6/10.
- Recovered three of the four prior LibrarySkill regressions versus TextSkill:
  `fix-visual-stability`, `llm-prefix-cache-replay`, and
  `dapt-intrusion-detection`.
- `manufacturing-fjsp-optimization` still fails scoreably with
  `agent_return_code=0`, `error=null`, and `verifier_error=null`. The translated
  LibrarySkill now contains the exact right-shift/local-minimality guidance, and
  the agent trajectory shows it used that guidance, but the final optimized
  schedule violates verifier local minimality for `(1, 1)`:
  `start=25`, `anchor=9`, and `start-1` is feasible. This remaining failure is
  no longer an obvious missing-guidance translator surface failure.

## 2026-08-24 — Native LibrarySkill Guidance Rerun

After removing translator output references to prior TextSkill packaging in
`ce36d58e fix: hide translation provenance in library skills`, reran the same
10-task NOOA `library_skill` SkillsBench sample.

The run used:
- SkillsBench checkout:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/skillsbench`
- Credentials file:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/.env`
- NOOA model: `openai/openai/openai/gpt-5.5`
- Sandbox: Docker
- Condition: `library_skill`
- Artifact root:
  `jobs/nooa-skillsbench-gpt55-10-library-native-guidance/`

Results:

| Task | LibrarySkill native guidance |
|---|---:|
| `fix-visual-stability` | 1.0 |
| `fix-erlang-ssh-cve` | 1.0 |
| `video-silence-remover` | 0.0 |
| `dynamic-object-aware-egomotion` | 0.0 |
| `manufacturing-fjsp-optimization` | 0.0 |
| `llm-prefix-cache-replay` | 1.0 |
| `dapt-intrusion-detection` | 1.0 |
| `offer-letter-generator` | 1.0 |
| `parallel-tfidf-search` | 1.0 |
| `reserves-at-risk-calc` | 0.0 |

Current result:
- Native-guidance NOOA LibrarySkill aggregate: 6/10.
- This matches the previous patched LibrarySkill aggregate.
- The pass/fail set is unchanged from the previous guidance-preservation rerun.
