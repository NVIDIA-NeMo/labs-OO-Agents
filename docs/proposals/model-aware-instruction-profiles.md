# Proposal: Composable Instruction Overlays for Agent Harnesses

- **Status:** Experimental proposal with an initial NOOA implementation
- **Author:** Ryan Angilly
- **Audience:** Coding-agent harness authors and repository tooling maintainers

This document proposes new harness capabilities. It does not describe an
existing cross-harness standard. NOOA is implementing the model-aware and
private-developer portions so the idea can be exercised in a real coding agent,
refined with evidence, and considered by other harnesses.

The name, configuration location, and portable exchange format are open to
change. The important part is to compose instructions from sources with
different owners and selection rules, without forcing all guidance into a
single repository file.

## The mismatch

Coding-agent harnesses increasingly let a user change models without changing
repositories or sessions. Repository instructions such as `AGENTS.md`, however,
remain static.

Different models do not respond identically to the same guidance. Smaller
models may benefit from explicit workflow steps. A more capable model may
already perform those steps and become slower, more tentative, or less
effective when given the same scaffolding. Repository prompts can then grow
into a lowest-common-denominator policy or a collection of workarounds for
models that are no longer in use.

OpenAI's article
[Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)
describes this problem directly: shared skills may be used by different models,
and guidance that helps Sol or Luna may overconstrain Astra. The article
recommends revisiting accumulated skill, `AGENTS.md`, and task instructions as
model capabilities change.

That advice improves instruction quality, but it does not solve dynamic model
selection. The remaining mismatch is:

> Model selection is dynamic, while repository instruction selection is
> static.

## A second mismatch: private developer workflows

A repository's checked-in instructions are shared with every contributor. That
is the right place for project-wide build commands, architecture rules, and
contribution expectations. It is the wrong place for every instruction a
particular developer needs while working in that repository.

For example, a developer may want an agent to create or update a ticket in a
particular Linear project whenever it performs substantive work in one local
repository. The repository may be public while that Linear project, its tools,
and its workflow are private to the developer's employer. Adding the rule to
the checked-in `AGENTS.md` would expose internal context and give other
contributors an instruction they cannot follow.

This is not merely a global preference. The instruction applies to one
developer in one repository context. Nor is it a replacement for the
repository's instructions. The developer needs both the shared project rules
and the private workflow rule.

Codex's documented [`AGENTS.override.md`
behavior](https://learn.chatgpt.com/docs/agent-configuration/agents-md) solves a
different problem. At a given directory level, Codex loads the override file
instead of the sibling `AGENTS.md`; the sibling file is ignored. That is useful
when a developer needs to replace instructions temporarily, but it does not
provide an additive, private, repository-scoped layer.

The second mismatch is therefore:

> Repository instructions are shared, while some repository-specific workflows
> belong to an individual developer or environment.

This proposal calls the new layer a **developer instruction overlay**. The term
is meant to distinguish it from a task prompt and from a model-selected
instruction profile. Its name and storage format remain open for discussion.

## Proposed capability

A harness should be able to compose an effective instruction stack from
separate layers:

```text
shared repository instructions
        +
private developer overlay for this repository
        +
selected instruction-profile overlay
        +
skill and task instructions
        =
effective prompt
```

Shared repository instructions should contain facts and expectations that
apply regardless of the selected model: build commands, architectural rules,
generated-file policy, API compatibility requirements, and safety boundaries.

An instruction profile should contain guidance chosen for a model, model
family, capability group, or explicit selection policy. It is an overlay, not a
replacement for the repository's shared instructions.

A developer overlay should contain additive, repository-scoped guidance owned
by the person or environment running the harness: internal issue-tracking
workflow, local review expectations, organization-specific tools, or other
context that should not be committed to the repository. It must not cause a
shared repository instruction file to be skipped.

This separation lets repositories avoid embedding conditional prose such as
"if you are model X" in `AGENTS.md`. It also gives harnesses one place to
resolve and explain what the model actually received. It lets developers add
private workflow guidance without changing or replacing the instructions that
the repository publishes for everyone.

## Two independent selection axes

The two overlays answer different questions:

| Layer | Selection question | Example |
|---|---|---|
| Developer overlay | Whose repository-specific workflow applies here? | Ryan's Linear workflow for this repository |
| Instruction profile | How should the selected model be guided? | A concise profile selected for Astra |

They should remain independent. Switching models should not remove a
developer's repository workflow. Moving to a different checkout should not
silently carry a repository-specific overlay with it unless both checkouts
resolve to the same repository identity.

This produces a broader conceptual resolver:

```python
resolve_instruction_stack(
    repository=repository,
    model=model,
    explicit_profile=None,
    developer_context=developer_context,
)
```

The exact API is illustrative. A harness could resolve the developer layer and
the model profile through separate functions as long as the resulting stack is
ordered, additive, and inspectable.

## Worked example: one repository, one developer, two models

Consider a public repository named `example-agent`. The repository contains
instructions that every contributor should receive:

```text
example-agent/
├── AGENTS.md
├── docs/
├── packages/
├── tests/
└── .nooa/
    ├── settings.yaml
    └── models/
        ├── gpt-5.6.md
        └── gpt-6-astra.md
```

Its checked-in `AGENTS.md` acts as a map of the project and states its shared
contribution rules:

```markdown
# Repository guide

- Architecture and design documents live in `docs/architecture/`.
- Package code lives in `packages/`; matching tests live in `tests/`.
- Run `uv run pytest` before submitting a change.
- Preserve the public API unless the change includes a migration plan.
- Pull requests must explain the user-visible effect and include test evidence.
```

Nothing in that file assumes who the contributor works for, which private tools
they can access, or which model their harness selected.

### The developer's private repository workflow

One developer also wants every substantive change in this repository connected
to an internal Linear project. That instruction is useful to that developer but
does not belong in the public repository. Their local user configuration looks
like this:

```text
~/.config/nooa/
├── developer-instructions.yaml
└── instructions/
    └── example-agent-workflow.md
```

The private configuration associates the repository with the instruction file:

```yaml
# ~/.config/nooa/developer-instructions.yaml
repositories:
  "/Users/alex/code/example-agent":
    - instructions/example-agent-workflow.md
  "/Users/alex/.codex/worktrees/*/example-agent":
    - instructions/example-agent-workflow.md
```

The referenced file can contain organization-specific workflow and links:

```markdown
# My workflow for example-agent

- Before substantive implementation, find or create an issue in the private
  Platform Agents Linear project at
  `https://linear.app/<company>/project/<project-id>`.
- Keep the issue status and a short progress note current while working.
- Include the issue identifier when preparing the pull request.
- If the Linear integration is unavailable, report that clearly instead of
  claiming the issue was updated.
```

This file is not checked into `example-agent`. It can live outside the checkout,
as shown here, or a host can explicitly point `CodingAgent` at another private
configuration file. The overlay adds to `AGENTS.md`; it does not hide or replace
it. The link is workflow context, not a credential.

The instruction content could instead live in a deliberately gitignored file
inside the checkout and be referenced by absolute path from the private
configuration. Keeping it outside the repository is the safer default because
it removes the risk of accidentally committing organization-specific context.

### Model guidance checked into the repository

The maintainers may choose to publish model profiles because they are useful to
everyone using those models:

```yaml
# example-agent/.nooa/settings.yaml
instructions:
  profiles:
    gpt-5.6:
      - .nooa/models/gpt-5.6.md
    astra:
      - .nooa/models/gpt-6-astra.md

  models:
    "openai/gpt-5.6-*": gpt-5.6
    "openai/gpt-6-astra": astra
```

The GPT-5.6 profile might provide a more explicit working loop:

```markdown
# GPT-5.6 working guidance

- Keep a short task list for multi-step changes.
- Inspect the relevant implementation and focused tests before editing.
- After each meaningful edit, run the narrowest useful test before continuing.
- Before finishing, review the diff against every requested behavior.
```

The Astra profile can be intentionally small:

```markdown
# Astra working guidance

No additional workflow is required. Apply the repository, developer, skill,
and task instructions directly, using your judgment about execution details.
```

It is also valid to omit the Astra mapping entirely. In that case Astra gets no
model-profile overlay, while the repository and developer layers remain active.
Profiles do not need matching size or structure across models.

If maintainers want to reuse optional guidance, a profile can list several
files in order:

```yaml
instructions:
  profiles:
    astra:
      - .nooa/models/shared-suggestions.md
      - .nooa/models/gpt-6-astra.md
```

The Astra-specific file can explain that the shared suggestions are heuristics
to adapt to the task rather than a required procedure. This is ordinary ordered
file composition; it does not introduce a separate inheritance mechanism.

### Model guidance kept private

Model tuning does not have to be checked into the repository. A developer can
define a named profile in their user-level `~/.config/nooa/settings.yaml` and
reference a private file with an absolute path:

```yaml
instructions:
  profiles:
    alex-gpt-5.6:
      - /Users/alex/.config/nooa/model-profiles/gpt-5.6.md
```

A host can then select it explicitly with
`instruction_profile="alex-gpt-5.6"`. This private model profile is still a
different layer from `example-agent-workflow.md`: one tunes a model's working
style, while the other applies the developer's repository workflow regardless
of model.

### What the models receive

For `openai/gpt-5.6-sol`, the effective order is:

```text
1. example-agent/AGENTS.md
   Shared project map, tests, API rules, and pull-request requirements

2. ~/.config/nooa/instructions/example-agent-workflow.md
   This developer's private Linear workflow for this repository

3. example-agent/.nooa/models/gpt-5.6.md
   Detailed execution guidance for the selected model family

4. Active skill instructions and the current task
```

For `openai/gpt-6-astra`, only the third layer changes:

```text
1. example-agent/AGENTS.md
2. ~/.config/nooa/instructions/example-agent-workflow.md
3. example-agent/.nooa/models/gpt-6-astra.md  # intentionally minimal
4. Active skill instructions and the current task
```

If Astra has no mapping, item 3 is absent. Switching models never removes the
repository map or the developer's Linear workflow. Changing developers can
change item 2 without modifying the repository. Changing tasks or skills can
change item 4 without changing either overlay-selection mechanism.

The resolved debug output makes that composition visible:

```text
model: openai/gpt-5.6-sol
instruction profile: gpt-5.6 (model)
matched model pattern: openai/gpt-5.6-*
repository instructions:
  - /Users/alex/code/example-agent/AGENTS.md
developer overlay:
  matched repository: /Users/alex/code/example-agent
  config: /Users/alex/.config/nooa/developer-instructions.yaml
  - /Users/alex/.config/nooa/instructions/example-agent-workflow.md
profile overlay:
  - /Users/alex/code/example-agent/.nooa/models/gpt-5.6.md
```

This is the practical goal: each instruction has a clear owner, applicability
rule, and reason for appearing in the prompt.

## Harness responsibility

The harness owns the mechanism because it knows:

- which model will receive the prompt;
- when the selected model changes;
- which repository and user instructions are active;
- which skills are active; and
- how the final prompt is assembled.

For direct model access, the harness resolves the active model, selects the
applicable profile, and builds the prompt. The developer overlay is selected
from the local user, host, and repository context independently of the model.

## Developer overlay semantics

The developer overlay is additive by definition:

- The normal `AGENTS.md` discovery chain still runs.
- A matching developer overlay is appended after the shared repository layer.
- No matching overlay preserves the harness's prior behavior.
- Selecting a developer overlay does not select a model profile, and selecting
  a model profile does not select a developer overlay.
- Model changes refresh the profile layer without removing the developer layer.

Composition order must be deterministic, but source order should not turn one
kind of instruction into authority over another. Shared repository instructions
describe project policy. Developer overlays add personal or organization-local
workflow. Model profiles tune how a model performs the work. A profile should
not negate project or developer policy, and a developer overlay should not
claim that it can relax repository requirements. Existing system and harness
safety rules remain outside and above this proposed stack.

### Repository identity and storage

The difficult part is not reading another Markdown file. It is selecting that
file without leaking private context or applying it to the wrong checkout.

A harness could support one or more of these storage models:

1. User-owned configuration outside the repository, keyed by repository
   identity. This is the safest default for private organization context.
2. An explicitly untracked file inside a local checkout. This is convenient but
   creates an accidental-commit risk and behaves less cleanly across worktrees.
3. A client-managed setting stored in the harness and associated with a saved
   project or workspace.

The portable behavior should not depend on choosing one location. It should
define how the harness identifies the repository, resolves the ordered sources,
and reports the result. Candidate repository identifiers include an explicit
local path, a normalized Git remote plus repository-relative root, or a
client-assigned project ID. Paths alone are brittle across clones; remotes alone
need normalization and may be absent or shared by several local environments.

A repository-controlled file should not be able to opt the harness into loading
an arbitrary private file from the developer's machine. The association must be
created in user-owned or client-owned configuration, or require an explicit
trust decision. Instruction files must not contain credentials; the overlay can
name a Linear project or request use of a tool, while authentication remains in
the tool or client credential store.

The overlay also does not grant capabilities. If it asks the agent to update
Linear but the current harness has no Linear integration, the debug view should
make the active instruction and unavailable capability understandable. The
agent must not pretend the external action succeeded.

## Proposed semantic contract

The portable concept is profile-oriented rather than tied directly to model
names. A conceptual resolver looks like:

```python
resolve_instruction_profile(
    model=model,
    explicit_profile=None,
)
```

It returns a resolved profile containing at least:

- the model identifier used for resolution;
- the selected profile name, if any;
- why it was selected;
- the model pattern that matched, if any; and
- the ordered instruction sources in the overlay.

Model mappings are one way to select a profile. They are not the profile
abstraction itself. This distinction leaves room for a user, host, or future
capability matcher to select a profile explicitly.

### Resolution precedence

The initial resolution order is:

1. An explicit profile supplied by the host.
2. An exact model mapping.
3. The most specific matching model-family pattern.
4. No overlay.

The repository's shared instruction files come first. Files within the selected
developer overlay retain their configured order and come next. Files within the
selected profile retain their configured order and follow the developer layer.
Skill and task instructions continue through their existing harness paths.

When no profile or model mapping applies, the harness must preserve its previous
behavior. Repositories should not need model-specific configuration to keep
working.

### Dynamic selection

The harness should resolve the profile whenever the effective model changes.
This includes model changes during a session, not just process startup.

A host should also be able to select or clear a profile independently of model
identity. This supports named policies such as `explicit-coding-v2` that may
apply to several models.

### Inspectability

The resolved stack should be inspectable before or during a run. At minimum, a
debug view should answer:

- Which model identifier was matched?
- Was the profile selected explicitly or through a model mapping?
- Which exact key or pattern won?
- Which files were loaded, in what order?
- Which configured files were missing or unreadable?

A future portable debug record might look like:

```json
{
  "model": "openai/gpt-5.6-luna",
  "profile": "explicit-coding-v2",
  "selection": "explicit",
  "matched_model_pattern": null,
  "sources": [
    {"layer": "repository", "path": "AGENTS.md"},
    {
      "layer": "developer",
      "scope": "repository",
      "path": "~/.config/harness/repositories/nooa.md"
    },
    {"layer": "profile", "path": ".agents/profiles/explicit-coding-v2.md"}
  ]
}
```

That JSON shape is illustrative and is not implemented as a public NOOA format.

## Proposed configuration

Configuration syntax is a harness choice. NOOA's initial syntax lives in
layered `settings.yaml` and supports a direct model-to-files form:

```yaml
instructions:
  models:
    "openai/gpt-6-astra":
      - .nooa/models/gpt-6-astra.md
    "openai/gpt-5.6-*":
      - .nooa/models/gpt-5.6.md
```

It also supports named profiles so model matching and profile definition remain
separate:

```yaml
instructions:
  profiles:
    concise:
      - .nooa/profiles/concise.md
    explicit-coding-v2:
      - .nooa/profiles/explicit-coding-v2.md

  models:
    "openai/gpt-6-astra": concise
    "openai/gpt-5.6-*": explicit-coding-v2
```

NOOA stores private developer configuration separately from repository
configuration. For example:

```yaml
# ~/.config/nooa/developer-instructions.yaml
repositories:
  "/Users/ryan/code/project":
    - instructions/project-workflow.md
  "/Users/ryan/.codex/worktrees/*/labs-OO-Agents":
    - instructions/nooa-workflow.md
```

Repository selectors are absolute paths or path globs matched against the
resolved Git worktree root. Exact paths win; otherwise the glob with the most
literal characters wins, with lexical order as the tie-breaker. Instruction
paths may be absolute or relative to `developer-instructions.yaml`.

NOOA reads this layer only from the user configuration directory. It does not
read a repository's `.nooa/developer-instructions.yaml`. A host may pass a
different private configuration path through
`CodingAgent(..., developer_instruction_config=...)`, and a process may set
`NEMO_OO_DEVELOPER_INSTRUCTIONS` to select one explicitly.

## Current NOOA reference implementation

NOOA now provides an initial implementation in its
[shared coding-agent package](../../packages/nooa-cli/src/nooa_cli/coding/instructions.py):

- `resolve_instruction_profile(...)` resolves model mappings and explicit
  profile overrides.
- `resolve_developer_overlay(...)` selects private instructions for the
  resolved repository root from user-owned configuration.
- `resolve_instruction_stack(...)` combines the applicable `AGENTS.md` chain,
  developer overlay, and selected model profile.
- `CodingAgent` selects a profile from its resolved `llm.model` value during
  construction.
- `CodingAgent.set_llm(...)` recalculates the model-selected profile.
- `CodingAgent.set_instruction_profile(...)` lets a host set or clear an
  explicit profile.
- `CodingAgent(..., developer_instruction_config=...)` lets a host select a
  private configuration file without changing process-global state.
- `agent.instruction_stack.format_debug()` explains the result.

NOOA's model and repository-pattern rules are deterministic: exact keys win;
otherwise the matching glob with the greatest number of literal characters
wins; lexical order breaks a tie. A matched configuration that references an
unknown profile or missing file fails visibly. Instruction reads retain the
existing size limits.

This is a reference implementation and a place to learn. It should not prevent
another harness from choosing TOML, a conventional `.agents/` directory, a
client-managed configuration surface, or a different pattern language.

The initial developer implementation deliberately uses local repository paths
as identity. This makes matching explicit and testable, supports worktrees
through globs, and avoids reading Git credentials or invoking Git during prompt
construction. Remote-based or client-assigned identities remain possible
future extensions.

## Minimum model-profile adoption contract for another harness

A harness can adopt the proposal without copying NOOA's file layout or YAML
schema. A useful first implementation needs to:

1. Preserve existing repository instruction behavior when no overlay is
   configured.
2. Add one optional, ordered profile overlay after shared repository
   instructions.
3. Resolve it from the effective model and refresh it when the model changes.
4. Accept an explicit profile choice that takes priority over model matching.
5. Expose the resolved source order and selection reason for debugging.
6. Apply clear limits and error behavior to configured instruction files.

That common behavior is more important than standardizing the first config
file location.

### Developer-overlay extension

A harness that also adopts private developer overlays should:

1. Keep the overlay additive; never use it as a reason to skip shared
   repository instructions.
2. Scope it to an explicit repository identity and developer or client
   context.
3. Store the association in a user-controlled location or obtain a clear trust
   decision before loading it.
4. Preserve existing behavior when no developer overlay matches.
5. Keep developer selection independent from model-profile selection.
6. Show the overlay's scope, source, and position in the resolved stack.
7. Keep secrets out of instruction content and treat tool availability as a
   separate capability.

### Suggested conformance scenarios

An implementation should demonstrate at least these cases:

1. With no profile configuration, the prompt is unchanged from the harness's
   previous behavior.
2. An exact model mapping wins over a matching family pattern.
3. Overlapping family patterns resolve the same way every time.
4. An explicit profile wins over model-derived selection.
5. Changing the model refreshes the overlay before the next prompt is built.
6. Clearing an explicit profile returns selection to model matching.
7. The debug view reports repository sources before profile sources.
8. An invalid configured profile or source produces a visible error.
9. A developer overlay and the normal repository chain both appear in the
   effective stack.
10. Two developers can apply different private overlays without modifying the
    repository.
11. A model change refreshes the profile while preserving the developer
    overlay.
12. A repository cannot cause arbitrary user-owned files to enter the prompt.

NOOA's
[focused tests](../../packages/nooa-cli/tests/test_coding_agent.py) exercise
these model-profile and developer-overlay behaviors and can serve as examples
for other harnesses.

## Safety and trust

Profile files enter the model's trusted instruction channel. A harness should
treat them with the same care as repository and developer instruction files:

- bound file reads and rendered prompt size;
- report missing, unreadable, and malformed configuration;
- define whether paths may leave the workspace;
- show the final source order; and
- keep task input and tool output out of trusted instruction text.

NOOA currently allows absolute profile paths and resolves relative paths from
the Git worktree root. Other harnesses may choose a stricter workspace-only
policy, but the behavior should be explicit and inspectable.

Private does not mean secret once an overlay is active: its contents are sent
to the selected model provider as part of the prompt. A client should state
that clearly and expose which private sources were included. Developers should
put workflow guidance in this layer, not tokens, passwords, customer data, or
other secrets.

## Non-goals for the first version

This proposal does not attempt to:

- define the content of a good profile;
- maintain a universal taxonomy of model behavior;
- generate profiles automatically;
- replace `AGENTS.md`, skills, or task prompts;
- synchronize private developer overlays between machines;
- choose a model for a task.

The first version supplies composition and selection. Evaluation and profile
authoring can evolve separately.

## Open questions

The implementation is intentionally early. Feedback from other harness authors
should shape:

- whether the shared term should be *instruction profile*, *prompt profile*, or
  something else;
- whether profile definitions belong in repository files, user configuration,
  client configuration, or a shared conventional directory;
- what the cross-harness name for a private additive layer should be;
- how a developer overlay should identify a repository across clones,
  worktrees, remote URL changes, and repositories without remotes;
- whether a local untracked overlay is safe enough to support alongside
  user-owned external configuration;
- how clients should surface an active overlay that references unavailable
  tools;
- whether resolution should use provider model IDs, harness aliases, capability
  tags, or several identifiers;
- whether a run may compose several profiles or must select exactly one;
- whether profile selection is scoped to a process, session, turn, or model
  call;
- which parts of the resolved stack should be recorded in traces; and
- what a portable conformance test should verify.

NOOA's current API and config are a concrete starting point, not a claim that
these questions are settled.

## Direction

The immediate NOOA feature solves manual model switching. The developer-overlay
extension solves a separate ownership problem. Together, they make the
effective prompt a visible composition instead of a single static repository
document.

We invite other harness authors to adopt, challenge, or reshape this proposal.
Moving the document or changing the syntax is acceptable. The core ideas to
test are additive composition, separation of ownership from model policy,
deterministic selection, and a complete explanation of what entered the prompt.

## A note on routers

Thinking through this problem raises a related question about routers, but this
proposal and the current NOOA implementation do not address router integration.
One possible future interface could allow a harness to send the instruction
sets for the models available behind an endpoint so the router can apply the
appropriate set after choosing a model. Another could send the resolved
components with enough metadata for an inference service to place them in the
appropriate system or developer prompt.

Organizing instructions into named, inspectable components makes that future
work possible without defining it here. This proposal does not specify a
transport, assign responsibility between a harness and router, or implement any
router-specific behavior.
