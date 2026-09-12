# Shared interactive behavior audit

Audited 2026-09-12 on `refactor/shared-interactive-session`, targeting `dev/tui`,
through `15a412e2`. Scope: native bootstrap, configuration, command dispatch,
session lifecycle and input handling compared with the NOOA ACP adapter.
This is a local source audit with isolated behavioral probes, not a new Pool
client acceptance run. No LLM backend or response-IR changes are proposed here.

## Finding

The agent factory and persistent runner are shared, but the session composition
and control layer still has two implementations. The title omission was one
instance of host behavior being attached outside that shared boundary.

Three additional divergences were reproduced without calling an LLM. The
configuration and skill-command follow-up below now addresses those three.
Other source-confirmed omissions still need their own parity acceptance cases.

## Original reproduced divergences (resolved in the follow-up)

### 1. Configuration merging can change agent behavior

Native recursively applies nested settings in
[`tui/settings.py`](../../packages/nooa-cli/src/nooa_cli/tui/settings.py).
ACP's [`SessionOptions.load`](../../packages/nooa-cli/src/nooa_cli/interactive/options.py)
overwrites the entire `summarization` dictionary when the `coding` section
provides a partial override.

With the same isolated settings file:

```yaml
agent:
  summarization:
    policy: none
    preserve_recent: 3
coding:
  summarization:
    max_tokens: 1234
```

| Resolved option | Native | ACP |
| --- | --- | --- |
| policy | `none` | `token_budget` |
| max_tokens | 1234 | 1234 |
| preserve_recent | 3 | 10 |

ACP therefore enables automatic summarization where native disables it. Both
use the same summarizer implementation; configuration resolution is the gap.

Move layered behavior loading, nested merging and validation into one shared
loader. Native should add presentation settings and invocation overrides to
that result. Cover partial nested overrides and explicit workspace selection.

### 2. Settings writes still target the legacy namespace

`Command._persist_tui_settings` in
[`tui/commands.py`](../../packages/nooa-cli/src/nooa_cli/tui/commands.py)
writes `tui.*`. Both loaders give `coding.*` precedence. `settings_to_dict`
also serializes behavior under `tui` and `agent`.

Probe: with `coding.keep_going: true`, write `tui.keep_going: false` through
the same writer used by native commands. Reloading shared options still gives
`keep_going == True`. A live command may change the in-memory state or snapshot
variables, so the symptom depends on whether the user resumes that session or
starts a fresh one. The saved setting does not reliably represent their choice.

Provide a shared behavior settings writer with one canonical namespace and an
explicit compatibility policy for old keys. Acceptance must include changing a
setting, starting a fresh session in the other host, and checking the result.

### 3. Markdown skill commands were not extracted

Native `CommandRegistry._discover_user_skills` scans `SKILL.md` frontmatter,
honors `user-invocable`, and expands `$ARGUMENTS`. ACP uses
[`CodingSlashCommandRegistry`](../../packages/nooa-cli/src/nooa_cli/coding/slash_commands.py),
which discovers only Python `@slash_command` methods.

Probe: construct one coding agent with a skill root containing:

```markdown
---
name: audit-skill
description: Host parity probe
---
Inspect $ARGUMENTS
```

Native `get_user_skill("audit-skill")` returned a command; the ACP registry's
`get("audit-skill")` returned `None`. Model-side skill discovery is shared, but
user invocation is incomplete. If an unrecognized slash command reaches ACP
`prompt`, it falls through as ordinary user text rather than invoking the skill.

Unify discovery, collision rules, argument expansion and invocation for both
Markdown skills and Python skill methods. Test the advertised commands and the
resulting agent input, not just that the same skill roots were discovered.

## Other source-confirmed gaps

| Area | Current difference | Shared extraction and acceptance |
| --- | --- | --- |
| Behavior controls | The shared layer now owns `/skills`, `/memory`, and `/reflection`. `/model`, `/reasoning`, `/compact`, and MCP interaction still need ACP mappings. ACP also exposes Markdown and Python skill commands. | Extract the operations and structured results. Map them to native commands and appropriate ACP controls. Verify state changes, saved preferences, and cancellation. The Pool client's own commands do not establish that NOOA performed these operations. |
| MCP interaction | The registry and approvals are shared. Native `CommandRegistry._bind_mcp_oauth_prompt` binds user interaction and `/mcp approve` records approval. ACP does not supply these interaction paths. | Shared interaction requests with host adapters. Test a previously unapproved server and a fresh OAuth flow. Preapproved MCP success covers only part of parity. |
| Input normalization | Native `tui/completer.py:expand_mentions` resolves typed `@path` mentions into absolute Markdown links; the composer preserves pasted text as opaque. Both hosts now expand mentions in skill results through the shared helper. ACP `_prompt_text` accepts text and resource links, with links rendered as `Resource name: URI`. | Define common semantic input/attachment handling with provenance. Test literal pasted `@text`, real file references, and references returned by skills. Preserve opaque payloads. Do not expand them indiscriminately. |
| Startup and restore policy | Native `bootstrap` handles invalid custom agents with a fallback and snapshot restoration errors with warnings. ACP `_create_runtime` propagates failures. Native configures memory before skill setup; ACP configures skills before memory. Their MCP connection and `SessionResumed` notification ordering also differ. | One create/load lifecycle with explicit fallback/restoration results and a readiness barrier. Test a failing custom agent, missing/corrupt snapshot, and a skill whose resume hook inspects all configured resources. The ordering differences are confirmed; their effects on arbitrary custom skills were not dynamically tested. |
| Session eligibility and metadata | The store is canonical, but native picker, native `--continue`, and ACP still apply separate selection rules. The title and empty-session fixes exposed this duplication. Native `--continue` examines only its initial limited list before filtering; ACP now filters all candidates before pagination. | Shared resumable-session queries and title/display metadata. Test empty, untitled-but-nonempty, active, and paginated histories; keep direct ID loading a separate operation. |
| Model readiness and diagnostics | Native owns startup health probes, deferred-input handling, model-switch validation, and actionable diagnostics in `tui/health_check.py`, `session.py`, and `commands.py`. ACP has no equivalent readiness layer; its prompt handler maps only specific generation limits and otherwise propagates generation errors. | Shared readiness/failure results with host presentation. Test an unresolved alias, endpoint failure and recovery, and inspect the actual ACP error message. Exact Pool rendering of these failures remains unverified. |
| Tracing and observations | Native bootstrap initializes configured exporters and trace/session correlation. ACP does not use that bootstrap. Native exposes Todo/job/memory views; ACP's event bridge currently forwards messages, tool activity, usage and session titles, with no equivalent Todo/plan projection. | Extract trace setup and stable observation data where needed. Keep terminal explorers and layouts with native. Verify the same session identity in diagnostics and the same Todo state through each host's supported view. |

Model-selection entry points remain host-specific:
native reads its default model from settings, while ACP requires `--model` or
`NOOA_MODEL`. The current acceptance commands deliberately equalize these.
Native also retains its explicit `NEMO_OO_PROJECT_DIR` override, whereas ACP
selects the session workspace's project settings. Aligning that scope contract
and explicit `--llm-config` paths remains follow-up work; acceptance leaves the
project override unset or points it at the tested workspace.

## Already shared; preserve these boundaries

- Default/custom agent construction, repository instructions, agent tools and
  model-side skill discovery.
- Persistent dispatch, WAIT/background notifications, foreground admission,
  cancellation and queue cleanup.
- Keep-going and reflection algorithms, including invalidation through the
  shared notification hook. Their user-facing configuration commands are the
  missing portion, not a second implementation of those algorithms.
- Core session storage/runtime ownership and snapshot handoff. The recent title
  request is now triggered by the shared runner; ACP publishes title changes.
- Memory setup and MCP registry/approval storage implementations.

Historical `TUIAgent` module aliases, `_tui_*` snapshot keys and compatibility
names are not, by themselves, missing behavior. Preserve old sessions while
moving responsibility. Terminal themes, keybindings, rendering and explorer
layouts can remain native.

## Proposed migration order

1. **Common configuration read/write contract.** Fix nested merges and canonical
   persistence first so parity tests really configure both agents identically.
2. **Common commands and input preparation.** Bring Markdown skills and Python
   commands under one registry, then extract the behavior operations from native
   command classes. Prioritize skills, compaction and memory/reflection.
3. **Common session lifecycle.** Own create/load readiness, configured resources,
   resume events, metadata eligibility, checkpoint and shutdown policy together.
   Reuse the existing agent factory, `LocalAgentRunner` and core session store.
4. **Host interaction and observations.** Map shared MCP approval/OAuth requests,
   readiness/errors and state observations into ACP and native presentation.
   Align model controls with the landing LLM stack's public interface.

The intended boundary is an interactive session service owning configuration,
commands, lifecycle and semantic input. Native and ACP translate user actions
into that service and render its results/events. An acceptance test should assert
the resolved options, available commands, prepared input, lifecycle events and
durable effects in addition to checking that both frontends produce an answer.

## Work completed during this audit

`15a412e2` filters zero-turn sessions out of ACP listing before pagination,
matching the native picker. It preserves untitled conversations, including a
user prompt with no assistant reply, and leaves database files intact. Updated
protocol fixtures exercise actual populated histories and exclusion of a fresh
ACP startup session. Validation: all ACP tests, **94 passed, 3 existing xfailed**;
Ruff and diff checks passed.

The three configuration/command probes above used temporary directories and an
isolated Python environment, with a fake LLM for agent construction. No user
settings or session databases were changed by the audit. The remaining lifecycle, controls and interaction gaps are documented for
follow-up implementation.


## Configuration and skill-command follow-up

The shared `interactive/settings.py` now resolves behavior for both hosts,
including partial summarization overrides. It writes `coding.*`, reads legacy
`tui.*` and `agent.summarization` aliases, and removes same-file aliases when
explicitly deleting a value. Native export and the first-run scaffold use the
canonical namespace. Skill roots resolve against the session workspace, and an
explicit `coding.additional_skills_dirs` list replaces its legacy YAML alias,
including when the new list is empty.

Both hosts now use `CodingSlashCommandRegistry` to discover Markdown and Python
commands. Markdown frontmatter, user visibility, root precedence, reserved host
names, Python-over-Markdown collisions, refresh, typed argument parsing and
skill-result file mentions have one implementation. Native preserves quoted
Python arguments. ACP dispatches expanded Markdown bodies through the shared
user-turn path, so recording and first-turn title instructions also agree.
Direct prompt/paste/attachment normalization remains a separate gap.

Regression coverage checks settings round trips through a real native command,
shared discovery and quoted arguments, actual native/ACP LLM input and durable
turns, and Markdown advertisement/invocation over the ACP subprocess protocol.
The LLM backend and response IR are untouched.

Validation after the follow-up: full CLI and ACP suites, **1,911 passed,
2 skipped, 3 existing xfailed** in 166.27 seconds; Ruff, formatting and diff
checks passed. ACP default session construction still imports no native TUI
modules. The full run also exposed a native picker shutdown race: completed
preview tasks could starve their own cleanup callbacks. A separate native-only
commit fixes it with a deterministic regression test; keep that commit out of
the eventual non-TUI upstream slice.


## Behavior-control follow-up

`interactive/controls.py` now owns `/skills`, `/memory`, and `/reflection`. Native renders its structured messages/tables; ACP advertises the
same operations and returns their output without generating an agent turn.
Skills directory and activation choices persist for new sessions, including
when a skill was already active from a one-off model action. Other reserved
NOOA commands return a clear unsupported-control message in ACP.

Automatic WebPublisher attachment was removed from `InteractiveAgent`, covering
both default and legacy coding agents. A resume hook removes historical
WebPublisher context instructions; the standalone publishing utility remains.

Validation for the controls and WebPublisher removal: the full CLI/ACP plus
focused core regression run had **1,962 passed, 2 skipped, 3 existing xfailed**
and one native picker rendering test failure. That test was waiting for any
rendered frame instead of the picker; after correcting its wait, the complete
picker suite passed **69 tests**. All ACP tests passed in the full run, including
wire-level controls and fresh-agent skill persistence. Ruff, formatting,
`git diff --check`, and `uv lock --check` pass. A fresh-process check confirmed
that ACP controls execute without importing native TUI modules.

## Keep-going removal

Keep-going was subsequently removed at user request from both native and ACP:
the judge, automatic continuations, control, completion, and saved-setting fields
are gone. Legacy settings and snapshot flags cannot reactivate it. The shared
turn policy retains reflection scheduling and normal completion notices. The
keep-going settings probe above records the original settings-drift finding;
use skill activation for the current persistence acceptance test.

Validation: the full CLI/ACP run had **1,909 passed, 2 skipped, 3 existing
xfailed**, with one timeout in the native input-buffer submission test. The
complete native app-behavior suite then passed **155 tests**, including that
test, without further code changes. All ACP tests passed in the full run.
Focused removal/configuration/parity checks passed **92 tests**. Ruff, formatting,
and `git diff --check` pass.

## Agent-facing skill persistence

The shared interactive setup now attaches `nooa.persisting_skills` as
`self.persisting_skills`. Its `remember(skill_id, directory=None)` and
`forget(skill_id)` methods reuse the operations behind `/skills` and write only
this workspace's NOOA settings. Remember activates now and saves the default;
forget deactivates now and records an inactive preference for future sessions.
Source directories, package installations, and other live sessions are retained.
No core registry API or general agent persistence contract was added.

The skill raises on a save failure, including the shared operation's account of
any live changes. Skills controls read current saved preferences before updating
lists, so sequential writes from another live session are retained.

Validation: **159 passed, 3 existing xfailed** across the complete ACP suite and
native settings, skill controls, resume events, and coding-agent tests. Generated
agent code remembers a repository skill from either host; fresh agents in both
hosts restore it. Coverage also checks forgetting, interleaved agent/command
writes, workspace isolation, and save failures. Ruff, formatting, and
`git diff --check` pass.
