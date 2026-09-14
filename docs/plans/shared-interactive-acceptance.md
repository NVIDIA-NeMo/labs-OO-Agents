# Native and Poolside acceptance test

Target: `dev/tui`. Both frontends must run NOOA's configured agent through the
shared interactive runtime, with the same skills, workspace and durable state.
Poolside is the ACP **client** here: use `pool --agent-server nooa-acp`.
`pool acp` starts Poolside's own agent server instead.
See [Poolside's client instructions](https://github.com/poolsideai/pool#run-as-an-acp-client-pool---agent-server).

## Prepare

Use one checkout containing this change for both executables. In that checkout:

```bash
uv sync --package nooa-acp
export NOOA_CHECKOUT="$PWD"
export PATH="$PWD/.venv/bin:$PATH"
export NOOA_MODEL='<your model or configured NOOA alias>'
export NOOA_WORKSPACE='/absolute/path/to/acceptance-workspace'
export NOOA_PARITY='/absolute/path/to/a-new-parity-directory'
```

Both terminals need those environment variables and the same credentials and
model-registry environment. Launch from the acceptance workspace so registry
discovery agrees. Do not use a native-only `--llm-config` override in this test.
Record `pool --version`, the NOOA commit, and the model alias/configuration.
Leave `NEMO_OO_PROJECT_DIR` unset, or point it at this workspace's `.nooa`
directory: native honors that explicit override, while ACP selects its session
workspace.

Configure behavior once in `$NOOA_WORKSPACE/.nooa/settings.yaml`. Existing `tui`
behavior fields remain readable; shared `coding` fields take precedence:

```yaml
coding:
  additional_skills_dirs: [skills]
  active_skills: [your.installed.skill]
  inactive_skills: []
  memory: session
  reflection: false
```

Use an actual installed skill name, ideally one exposing a simple slash command.
Both clients discover conventional project/user skill roots too. For a custom
agent, set `coding.agent_spec: './agent.py:MyAgent'` once. For legacy-agent
comparison, pass `--legacy-agent` to both NOOA entry points.

Create a seed in native NOOA:

```bash
cd "$NOOA_WORKSPACE"
nooa tui --model "$NOOA_MODEL"
```

Ask it to set `self.v.parity_marker = 'seed'`, create a Todo with a persistent
checkpoint, and report active skills and its working directory. Exercise the
fixture skill's slash command. Verify that the agent generates a descriptive
session title during its first turn without an explicit user request to rename.
Exit cleanly so the snapshot is saved, then record the full session ID as
`NOOA_SESSION`.

```bash
export NOOA_SESSION='<full session id>'
uv run --project "$NOOA_CHECKOUT" --no-sync python -m nooa_cli.interactive.parity \
  "$NOOA_WORKSPACE/.nooa/sessions/$NOOA_SESSION.db" --output "$NOOA_PARITY"
```

The helper holds the source session's exclusive lock, uses SQLite backup, and
creates byte-identical `native/` and `pool/` stores plus a SHA-256 manifest. It
includes the default session memory database, refuses a live source, and refuses
to overwrite an existing output directory. It does not clone workspace files,
project memory, a custom memory path, or user configuration. Use read-only
workspace tasks during simultaneous comparison; run file-edit tests one client
at a time, resetting the fixture between them.

## Run side by side

Terminal A:

```bash
cd "$NOOA_WORKSPACE"
NOOA_SESSIONS_DIR="$NOOA_PARITY/native" \
  nooa tui --model "$NOOA_MODEL" --continue "$NOOA_SESSION"
```

Terminal B, with `pool` installed:

```bash
cd "$NOOA_WORKSPACE"
NOOA_SESSIONS_DIR="$NOOA_PARITY/pool" \
  pool --agent-server nooa-acp --resume "$NOOA_SESSION"
```

If that Poolside version requires a named server entry, configure an
`agent_servers.nooa-acp.command: nooa-acp` entry using its settings format.
`NOOA_MODEL` is inherited by the NOOA server. Use Poolside's `/resume` picker if
its CLI cannot resume an ID directly; the server advertises session listing and
loading. Failure to list or load this seed is an acceptance failure to capture,
not a reason to substitute a new session.

| Check | Pass condition in both clients |
| --- | --- |
| Restore | Same session ID/title/transcript; marker is `seed`; Todo checkpoint survives. |
| Automatic title | A fresh session in each client receives the same title housekeeping instruction on its first prompt; ACP reports the chosen title to the client. User-selected titles survive later prompts. |
| Configuration | Same agent class, model, workspace, discovered/active skills and instructions from AGENTS.md. |
| Skill command | Same advertised fixture command, arguments and observable result. |
| Agent execution | Same prompted operation reaches the same durable state; compare tool effects, not exact LLM wording. |
| Background work | A delayed queue job reports after WAIT and after an earlier DONE without needing another prompt. |
| Cancellation | Stop a long-running task, then submit another prompt; no stale output or stuck prompt admission. |
| Memory | A session-scoped fact survives exit/resume in each copy. |
| Keep-going/reflection | Repeat with each option enabled identically; audit/idle behavior agrees. |
| MCP | Same preapproved server definition, connected tools and read-only tool result. |

For MCP comparison, configure servers in NOOA settings for both clients; avoid
also injecting the same server through Poolside. This slice shares the existing
approval store and enforces its fingerprints. Native `/mcp approve` and OAuth
dialogs have not yet been mapped into ACP interaction: preapprove the fixture
using native NOOA, and record fresh approval/OAuth parity as a follow-up gap.
Native settings menus and some host controls remain native. `/skills`, `/memory`,
`/reflection`, and skill-provided slash commands are now shared.

## Shared settings and Markdown command checks

In the acceptance workspace, add `skills/parity-review/SKILL.md`:

```markdown
---
name: parity-review
description: Check shared skill command behavior
argument-hint: [target]
---
Read $ARGUMENTS and report its first line. Do not modify it.
```

Create a `notes.md` with an easily recognized first line. Restart both hosts so
they discover the new command. Native completion and Pool's agent commands
should both offer `/parity-review`. Run `/parity-review "@notes.md"` in each:
both should read the same file and report the same first line. The saved user
turn should contain the expanded skill instruction once. Repeat on fresh
sessions to check automatic titling from a skill command as the first prompt.
A skill marked `user-invocable: false` should not appear in either command list.

For settings, add and activate a skill using the commands below, then exit.
Verify `.nooa/settings.yaml` contains its `coding.active_skills` entry. Start a
fresh session in each host and confirm the skill is active. For an exact
configuration check from that workspace:

```bash
uv run --project "$NOOA_CHECKOUT" --no-sync python - <<'PYTHON'
from pathlib import Path
from nooa_cli.interactive.options import SessionOptions
from nooa_cli.tui.config import Config
root = Path.cwd()
native = SessionOptions.from_native_config(Config.load(working_dir=str(root)))
acp = SessionOptions.load(root)
assert native == acp
print("Both hosts load the same behavior and skill preferences.")
PYTHON
```

New behavior writes use `coding.*`; legacy `tui.*` and
`agent.summarization` settings remain readable. Partial summarization overrides
preserve unspecified legacy fields. Presentation preferences remain in `tui.*`.
Both clients now support `/skills`, `/memory`, and `/reflection`
through shared operations. Model/reasoning selection, compaction, and MCP
approval/connection commands still need ACP mappings. Unsupported reserved NOOA
commands now report that limitation without invoking the model.

## Persistent skills and memory controls

Ordinary `self.skills.load/activate` changes the current session. The interactive
hosts also attach `self.workspace_settings` for saving workspace defaults. In Pool,
ask the agent to remember an exact Python skill ID and its source directory:

```python
await self.workspace_settings.remember_skill(
    "nvzurich.session_search",
    directory="/localhome/local-pfurgale/dev/nemo-oo-skills",
)
```

Restart and start fresh sessions in both hosts; the skill should be active.
Repeat from native NOOA. Then ask the agent to call
`await self.workspace_settings.forget_skill("nvzurich.session_search")`: it should
deactivate here and no longer auto-activate in fresh sessions in either host.
Other live sessions keep their state. Source directories and session data remain.
This skill belongs to the interactive hosts; the core `SkillRegistry` is unchanged.

These slash commands use the same operations:

```text
/skills add /absolute/path/to/nemo-oo-skills
/skills list
/skills activate <skill-id-from-the-list>
/skills commands
```

The directory and activation preferences are saved in `.nooa/settings.yaml`.
Activation is idempotent: it saves the preference even if the agent already
activated that skill. Restart both hosts and start **new** sessions; check that
the chosen skill is active and its Python slash commands are offered. Existing
live sessions keep their own skill instances and are not reconfigured by another
session's settings write.

In both clients, try `/memory local`, `/memory`, `/reflection on`,
`/reflection off`, and `/memory off`. Status should describe the
actual NOOA agent. Commands display results without a model turn, automatic
titling, or adding conversation turns to an otherwise empty resume entry.
For a memory handoff, enable local memory, ask the agent to remember a test fact,
exit, resume the same session in the other host, and recall it.

MCP connections belong to each live session. Saved server definitions are
workspace configuration; use `coding.mcp_auto_connect` for restart behavior.
Approvals are user-level fingerprints of the exact server definition.
ACP `/mcp` interaction is still pending.

The web publisher implementation and browser POST/replay support have been deleted.
Check fresh and resumed sessions in both hosts: the agent should have no
`self.web`, including when launched with `NEMO_OO_RICH_URL` set. Historical
WebPublisher context instructions are removed on the resume event.

## Handoff the same database

Exit **both** clients. Point Poolside at `$NOOA_PARITY/native` and resume the
native copy's ID. Confirm native changes, then set the marker to `pool` and
change the Todo checkpoint. Exit Poolside. Resume that same directory/ID in
native NOOA and verify both changes. Repeat in the opposite direction using
`$NOOA_PARITY/pool`.

Only one host owns a live database. Two simultaneous windows above operate on
copies; attaching both to one live agent would require a separate daemon/client
architecture. With a session open in native NOOA, refresh Poolside's `/resume`
picker against that same store: the active session must be absent. Close it in
native NOOA and refresh again: it must reappear and resume successfully. If a
session becomes active after the picker was populated, loading it must explain
that it is already open and must be closed in the other client or tab. Native's
existing recovery may offer/start a fresh session after a lock conflict; verify
the ID so that this cannot be mistaken for successful resume.

Create a fresh session through Poolside, send a prompt, and exit Poolside without
an explicit session-close command. Restart Poolside and verify that `/resume`
lists and loads that session with its generated title. Repeat while a turn is
running. Sessions with no conversation turns must be absent from `/resume`.
Older untitled conversations get an `Untitled session [id]` picker label
without rewriting their saved metadata. The subprocess tests cover both stdin
closure and SIGTERM. SIGKILL cannot run cleanup; old `.active` claims still
require confirming the original
owner has exited before removing them on that host.

## Automated evidence and boundaries

`packages/nooa-acp/tests/test_native_parity.py` uses the real native bootstrap,
ACP adapter, SQLite snapshots and an explicitly activated fixture skill, with a
deterministic fake LLM. It checks both handoff directions, variables, Todos,
title protection, skill resume hooks, workspace bindings, transcript parity,
Markdown command input and writer exclusion. `test_protocol.py` verifies
Markdown command advertisement and invocation over ACP JSON-RPC. `test_foreground_runtime.py` exercises both admission styles,
WAIT, notifications after DONE, cancellation and runner ownership.
`test_parity_sessions.py` checks identical independent copies and live-source
rejection. These tests do not substitute for running the actual Poolside client.

The change extracts behavior from native TUI modules into `coding`/`interactive`,
consolidates duplicate session registries and stores, and makes ACP use the
native persistent engine. Compatibility imports preserve old module paths.
The LLM backend and response IR are unchanged; repeat these gates after the
ordered #312 → #310 → #311 → #313 stack is integrated before upstreaming to main.

Previous configuration/skill-command regression gate: **1,911 passed, 2 skipped,
3 existing xfailed** across the CLI and ACP suites. Ruff and formatting pass.
ACP default session creation was also checked in a fresh process for absence of
native TUI imports. The actual Poolside checks above remain manual acceptance.

Validation for the controls and WebPublisher removal: the full CLI/ACP plus
focused core regression run had **1,962 passed, 2 skipped, 3 existing xfailed**
and one native picker rendering test failure. That test was waiting for any
rendered frame instead of the picker; after correcting its wait, the complete
picker suite passed **69 tests**. All ACP tests passed in the full run, including
wire-level controls and fresh-agent skill persistence. Ruff, formatting,
`git diff --check`, and `uv lock --check` pass. A fresh-process check confirmed
that ACP controls execute without importing native TUI modules.

## Removed keep-going behavior

Keep-going is removed from both hosts. Neither command list should offer
`/keep-going`, and a completed turn should not start a judge or enqueue a
continuation. Legacy `keep_going` / `keep_going_model` settings and snapshot
preferences are ignored; no manual settings or session cleanup is required.

Validation: the full CLI/ACP run had **1,909 passed, 2 skipped, 3 existing
xfailed**, with one timeout in the native input-buffer submission test. The
complete native app-behavior suite then passed **155 tests**, including that
test, without further code changes. All ACP tests passed in the full run.
Focused removal/configuration/parity checks passed **92 tests**. Ruff, formatting,
and `git diff --check` pass.

## Workspace settings skill

`self.workspace_settings` replaces `self.persisting_skills`. It is attached by
NOOA's shared interactive setup, with no core SkillRegistry API changes.

```python
self.workspace_settings.status()
await self.workspace_settings.remember_skill("your.skill", directory="/path/to/skills")
await self.workspace_settings.forget_skill("your.skill")
await self.workspace_settings.configure_memory(scope="session")
await self.workspace_settings.configure_reflection(enabled=True)
self.workspace_settings.set_default_model("your-model-alias")
```

Status distinguishes effective saved defaults from current runtime state and
shows MCP names without connection credentials. Memory/reflection preferences
apply to this agent type in the workspace, and use the same operations as the
native/ACP commands. Verify those commands see changes made by the agent.

For NOOA-owned MCP definitions:

```python
self.mcp.register("docs", url="https://example.test/mcp", transport="streamable-http")
self.workspace_settings.remember_mcp("docs", auto_connect=True)
self.workspace_settings.forget_mcp("docs")
```

Remember saves an existing NOOA registry definition without connecting or
approving it. First-time approval/authentication follows the existing MCP flow.
Forget disables startup connection and masks the workspace definition; it does
not revoke approvals or credentials. Client-supplied MCP servers are not
implicitly copied into NOOA settings. Test Pool's contribution separately with
[the Pool MCP probe](pool-mcp-acceptance.md).

The saved default model does not switch the running agent. Explicit model
launch arguments and NOOA_MODEL take precedence; the current ACP CLI requires
one of them, so Pool sessions continue using that override.
