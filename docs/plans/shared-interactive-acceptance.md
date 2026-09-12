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

Configure behavior once in `$NOOA_WORKSPACE/.nooa/settings.yaml`. Existing `tui`
behavior fields remain readable; shared `coding` fields take precedence:

```yaml
coding:
  additional_skills_dirs: [skills]
  active_skills: [your.installed.skill]
  inactive_skills: []
  memory: session
  keep_going: false
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
Likewise, native settings menus and the full set of host slash commands are not
part of this slice; skill-provided slash commands are shared.

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
running. Older untitled sessions get an `Untitled session [id]` picker label
without rewriting their saved metadata. The subprocess tests cover both stdin
closure and SIGTERM. SIGKILL cannot run cleanup; old `.active` claims still
require confirming the original
owner has exited before removing them on that host.

## Automated evidence and boundaries

`packages/nooa-acp/tests/test_native_parity.py` uses the real native bootstrap,
ACP adapter, SQLite snapshots and an explicitly activated fixture skill, with a
deterministic fake LLM. It checks both handoff directions, variables, Todos,
title protection, skill resume hooks, workspace bindings, transcript parity and
writer exclusion. `test_foreground_runtime.py` exercises both admission styles,
WAIT, notifications after DONE, cancellation and runner ownership.
`test_parity_sessions.py` checks identical independent copies and live-source
rejection. These tests do not substitute for running the actual Poolside client.

The change extracts behavior from native TUI modules into `coding`/`interactive`,
consolidates duplicate session registries and stores, and makes ACP use the
native persistent engine. Compatibility imports preserve old module paths.
The LLM backend and response IR are unchanged; repeat these gates after the
ordered #312 → #310 → #311 → #313 stack is integrated before upstreaming to main.
