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
  LibrarySkills by the frozen translator.

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
  pass generated smoke tests.

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

Current interpretation:

- LibrarySkill parity is possible on the 10-task dev set when activation-time
  guidance is preserved and executable skill assets become ergonomic APIs.
- Naive packaging regresses because valid Python packages are not necessarily
  good LibrarySkills: activation context and API discoverability matter.
- The held-out test set has not been run under the frozen protocol.
