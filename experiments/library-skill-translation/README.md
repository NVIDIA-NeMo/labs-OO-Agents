# LibrarySkill Translation Evaluation

## Research Question

Can package-backed NOOA `LibrarySkill`s preserve the task performance of
SkillsBench `TextSkill`s while providing a more maintainable and executable
skill surface?

The target claim is deliberately narrow:

> LibrarySkill translation should preserve TextSkill performance on average,
> and should improve maintainability or executability for code-backed skills.

This experiment is not evidence that LibrarySkill is intrinsically better than
TextSkill. A translated LibrarySkill must add something real: activation-time
guidance, native Python APIs for safe helper code, structured resource access,
or smaller and safer model-facing context.

## Experiment Design

Use a frozen split:

- `dev_tasks.txt`: the 10 SkillsBench tasks already used during translator
  development. These can be used for mechanical regression checks and debugging.
- `test_tasks.txt`: every remaining task under `skillsbench/tasks`. Do not tune
  translator behavior against this set.

Conditions:

- `no_skill`: NOOA with no task skills.
- `text_skill`: NOOA with the original task-bundled TextSkills.
- `library_skill`: NOOA with task-bundled TextSkills translated to package
  LibrarySkills by `SlimTextSkillTranslator`.

Candidate translation layer:

- `SlimTextSkillTranslator` is the held-out candidate once frozen.
- It owns a standalone package writer, validator, context-block generator,
  guidance renderer, function-script planner, and resource method renderer.
- It narrows script translation to import-safe Python functions only.
- It deliberately omits argparse and CLI-shaped script synthesis; omitted
  scripts are recorded in the host-side translation summary, outside the
  generated package.

Default run configuration:

- SkillsBench checkout:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/skillsbench`
- Credentials file:
  `/Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/.env`
- Model: `openai/openai/openai/gpt-5.5`
- Sandbox: Docker
- Concurrency: 1

Translator invariants before running held-out test:

- Activated LibrarySkills register a `context_block`.
- The context expression renders `format_guidance()` on the attached skill.
- Visible guidance is LibrarySkill-native and contains no stale TextSkill path
  or script-running instructions.
- Safe Python helper code is exposed as typed public methods or omitted from the
  generated library package.
- Root `SKILL.md` and raw `scripts/` trees are not bundled as resources.
- Generic resource plumbing is hidden from `doc(skill)`.
- Named public resource methods expose copied non-script resources.
- Resource docstrings include only small previews.
- Generated packages import, discover through `SkillRegistry`, activate, and
  pass repo-level smoke tests.
- Argparse or CLI-shaped scripts are not converted unless they also contain
  import-safe public Python functions that can be exposed directly.

## Key Metrics

- Aggregate pass rate per condition.
- LibrarySkill delta versus TextSkill: wins, ties, losses.
- Skill lift preservation: tasks where TextSkill beats `no_skill` and
  LibrarySkill also passes.
- Regressions versus `no_skill`.
- Scoreable failure count versus infrastructure failure count.
- Breakdown by skill type after the held-out run:
  prose-only, resource-backed, script/code-backed, and multi-skill tasks.

## How To Run

Run the dev split only for mechanical regression checks:

```bash
while read -r task; do
  uv run nooa-skillsbench-task \
    --skillsbench-dir /Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/skillsbench \
    --env-file /Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/.env \
    --model openai/openai/openai/gpt-5.5 \
    --sandbox docker \
    --condition library_skill \
    --jobs-dir jobs/nooa-skillsbench-library-dev \
    --job-name "${task}__nooa__library-dev" \
    --task "$task"
done < experiments/library-skill-translation/dev_tasks.txt
```

Run the held-out test only after the translator is frozen:

```bash
while read -r task; do
  uv run nooa-skillsbench-task \
    --skillsbench-dir /Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/skillsbench \
    --env-file /Users/adevoto/.herdr/worktrees/nemo_oo_agents/worktree-silver-river-5d47/.env \
    --model openai/openai/openai/gpt-5.5 \
    --sandbox docker \
    --condition all \
    --jobs-dir jobs/nooa-skillsbench-library-heldout \
    --job-name "${task}__nooa__heldout" \
    --task "$task"
done < experiments/library-skill-translation/test_tasks.txt
```

## Results Summary

Development set status:

- Original NOOA `text_skill` control: 7/10.
- Initial generated `library_skill`: 3/10.
- Guidance-preserving `library_skill`: 6/10.
- Resource-preview and activation-tested `library_skill`: 7/10.
- Naive skill-guided package generation: 5/10.
- Frozen protocol dev rerun at `b5c04755`: 6/10 scoreable, using a single
  scoreable rerun for `manufacturing-fjsp-optimization` after the first attempt
  timed out before verifier scoring.
- Slim translator dev preflight: 30/30 generated dev-set packages validated;
  3 scripts were omitted by policy.
- Slim translator dev rerun at `a84fb256`: 7/10 scoreable. All tasks recorded
  activated LibrarySkills, `agent_return_code=0`, no agent errors, and no
  verifier infrastructure errors.
- Post-review hardened slim rerun at `934b0f93`: 6/10 scoreable. All tasks
  recorded activated LibrarySkills, `agent_return_code=0`, no agent errors, and
  no verifier infrastructure errors.
- Standalone resource-tail slim rerun: 6/10 scoreable. Resource APIs are now
  listed after the adapted guidance, old `<path-to-this-skill>/<resource>`
  command examples are rewritten into LibrarySkill-native file-writing
  guidance, and the translator is 997 LOC.

Frozen protocol dev rerun:

| Task | LibrarySkill `b5c04755` |
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

Run artifacts:

- Main dev rerun:
  `jobs/nooa-skillsbench-library-dev-b5c04755/`
- Manufacturing scoreable rerun:
  `jobs/nooa-skillsbench-library-dev-b5c04755-scoreable-reruns/`

Slim translator dev rerun:

| Task | Slim LibrarySkill `a84fb256` |
|---|---:|
| `fix-visual-stability` | 1.0 |
| `fix-erlang-ssh-cve` | 1.0 |
| `video-silence-remover` | 0.0 |
| `dynamic-object-aware-egomotion` | 1.0 |
| `manufacturing-fjsp-optimization` | 0.0 |
| `llm-prefix-cache-replay` | 1.0 |
| `dapt-intrusion-detection` | 1.0 |
| `offer-letter-generator` | 1.0 |
| `parallel-tfidf-search` | 1.0 |
| `reserves-at-risk-calc` | 0.0 |

Slim run artifact root:

- `jobs/nooa-skillsbench-library-dev-slim-a84fb256/`

Post-review hardened slim rerun:

| Task | Hardened Slim LibrarySkill `934b0f93` |
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

Hardened slim artifact root:

- `jobs/nooa-skillsbench-library-dev-934b0f93/`

Standalone resource-tail slim rerun:

| Task | Resource-tail Slim LibrarySkill |
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

Resource-tail slim artifact root:

- `jobs/nooa-skillsbench-library-dev-slim-resource-tail/`

Trimmed slim translator six-pass check:

| Task | Trim candidate | Final candidate |
|---|---:|---:|
| `fix-visual-stability` | 1.0 | inconclusive |
| `fix-erlang-ssh-cve` | 1.0 | 1.0 |
| `llm-prefix-cache-replay` | 1.0 | 1.0 |
| `dapt-intrusion-detection` | 1.0 | 1.0 |
| `offer-letter-generator` | 1.0 | 1.0 |
| `parallel-tfidf-search` | 1.0 | 1.0 |

Trimmed slim artifact roots:

- `jobs/nooa-skillsbench-library-dev-slim-trim/`
  - Six of six known-passing dev tasks passed.
- `jobs/nooa-skillsbench-library-dev-slim-trim-final-five/`
  - Five of five non-visual known-passing dev tasks passed after the final
    helper-name trim.
- `jobs/nooa-skillsbench-library-dev-slim-trim-final/`
  - `fix-visual-stability` final-candidate attempt was interrupted after
    repeated task-shell deaths during browser/build commands; the generated
    LibrarySkill source matched the previous passing trim run.

Heldout translation-only validation:

- Translator commit: `4e1d5e68 refactor: trim slim skill translator`
- Artifact root: `jobs/nooa-skillsbench-library-heldout-translation-only/`
- Result: 77/77 heldout tasks translated and validated.
- Package result: 202/202 translated LibrarySkill packages validated.

Heldout stratified LibrarySkill smoke:

- Sample file: `heldout_sample_tasks.txt`
- Selection seed: `4e1d5e68-heldout-stratified-v1`
- Condition: `library_skill`
- Model: `openai/openai/openai/gpt-5.5`
- Artifact roots:
  - `jobs/nooa-skillsbench-library-heldout-sample-18/`
  - `jobs/nooa-skillsbench-library-heldout-sample-18-remainder/`
- Combined summary:
  `jobs/nooa-skillsbench-library-heldout-sample-18/combined_summary.json`
- Result: 11/17 scoreable tasks passed; one task was inconclusive.

| Task | Bucket | LibrarySkill |
|---|---|---:|
| `azure-bgp-oscillation-route-leak` | `prose_only_single` | 0.0 |
| `manufacturing-codebook-normalization` | `prose_only_single` | 1.0 |
| `energy-unit-commitment` | `multi_prose_only` | 0.0 |
| `hvac-control` | `multi_prose_only` | 1.0 |
| `manufacturing-equipment-maintenance` | `multi_prose_only` | 1.0 |
| `setup-fuzzing-py` | `multi_prose_only` | 0.0 |
| `quantum-numerical-simulation` | `resource_only` | inconclusive |
| `radar-vital-signs` | `resource_only` | 1.0 |
| `paper-anonymizer` | `resource_only` | 1.0 |
| `xlsx-recover-data` | `resource_only` | 0.0 |
| `civ6-adjacency-optimizer` | `script_only` | 0.0 |
| `energy-ac-optimal-power-flow` | `script_only` | 1.0 |
| `sec-financial-report` | `script_only` | 0.0 |
| `pptx-reference-formatting` | `script_and_resource` | 1.0 |
| `grid-dispatch-operator` | `script_and_resource` | 1.0 |
| `citation-check` | `script_and_resource` | 1.0 |
| `court-form-filling` | `script_and_resource` | 1.0 |
| `threejs-structure-parser` | `script_and_resource` | 1.0 |

Heldout stratified TextSkill comparison:

- TextSkill artifact roots:
  - `jobs/nooa-skillsbench-text-heldout-sample-18/`
  - `jobs/nooa-skillsbench-text-heldout-sample-18-remainder/`
  - `jobs/nooa-skillsbench-text-heldout-sample-18-final-seven/`
- Paired comparison artifact:
  `jobs/nooa-skillsbench-library-heldout-sample-18/text_vs_library_comparison.json`
- TextSkill result: 8/16 scoreable tasks passed; two tasks were inconclusive.
- LibrarySkill result on the same sample: 11/17 scoreable tasks passed; one
  task was inconclusive.
- On the 15 tasks scoreable in both conditions: 4 LibrarySkill wins, 1
  TextSkill win, 10 ties.

| Task | Bucket | TextSkill | LibrarySkill | Delta |
|---|---|---:|---:|---|
| `azure-bgp-oscillation-route-leak` | `prose_only_single` | 0.0 | 0.0 | tie |
| `manufacturing-codebook-normalization` | `prose_only_single` | 0.0 | 1.0 | LibrarySkill win |
| `energy-unit-commitment` | `multi_prose_only` | inconclusive | 0.0 | n/a |
| `hvac-control` | `multi_prose_only` | 1.0 | 1.0 | tie |
| `manufacturing-equipment-maintenance` | `multi_prose_only` | 1.0 | 1.0 | tie |
| `setup-fuzzing-py` | `multi_prose_only` | 0.0 | 0.0 | tie |
| `quantum-numerical-simulation` | `resource_only` | 0.0 | inconclusive | n/a |
| `radar-vital-signs` | `resource_only` | 0.0 | 1.0 | LibrarySkill win |
| `paper-anonymizer` | `resource_only` | 1.0 | 1.0 | tie |
| `xlsx-recover-data` | `resource_only` | 0.0 | 0.0 | tie |
| `civ6-adjacency-optimizer` | `script_only` | inconclusive | 0.0 | n/a |
| `energy-ac-optimal-power-flow` | `script_only` | 1.0 | 1.0 | tie |
| `sec-financial-report` | `script_only` | 1.0 | 0.0 | TextSkill win |
| `pptx-reference-formatting` | `script_and_resource` | 1.0 | 1.0 | tie |
| `grid-dispatch-operator` | `script_and_resource` | 0.0 | 1.0 | LibrarySkill win |
| `citation-check` | `script_and_resource` | 1.0 | 1.0 | tie |
| `court-form-filling` | `script_and_resource` | 0.0 | 1.0 | LibrarySkill win |
| `threejs-structure-parser` | `script_and_resource` | 1.0 | 1.0 | tie |

Current interpretation:

- LibrarySkill parity is possible on the 10-task dev set when activation-time
  guidance is preserved and executable skill assets become ergonomic APIs.
- Naive packaging regresses because valid Python packages are not necessarily
  good LibrarySkills: activation context and API discoverability matter.
- The frozen dev rerun did not reproduce the one-off
  `dynamic-object-aware-egomotion` pass from the resource-preview rerun. Treat
  that task as dev-set flaky or marginal rather than evidence of a stable
  translator gain.
- The slim translator preserved the 7/10 dev-set aggregate while removing the
  argparse/CLI synthesis path from the evaluated translation layer.
- The post-review hardening rerun was 6/10 because
  `dynamic-object-aware-egomotion` flipped back to failure. The run had no
  infrastructure failures, so this reinforces treating that task as marginal
  rather than using it as a stable translator signal.
- The standalone resource-tail rerun preserved the 6/10 hardened aggregate
  after removing the legacy translator dependency. The first standalone rerun
  exposed timeout-prone visual/manufacturing rollouts; the accepted rerun
  produced scoreable summaries for all 10 dev tasks.
- The trimmed standalone translator is 967 LOC and preserved the stable
  six-pass signal. The final visual rerun remained infrastructure-sensitive, so
  treat its interrupted final attempt as inconclusive rather than a translator
  regression.
- The trimmed translator mechanically generalizes on heldout translation:
  all heldout task skills package and validate.
- The 18-task heldout comparison is still a single-run smoke, but on paired
  scoreable tasks it favors LibrarySkill over TextSkill.
- Remaining inconclusive tasks should be rerun before treating the heldout
  sample as closed.
