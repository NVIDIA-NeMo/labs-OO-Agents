# Acceptance checks for the stacked native TUI and ACP agents

The stack is `main` → benchmark prerequisites (#331) → direct provider SDKs (#337)
→ shared agent/ACP (#330)
→ AionUi experiment (#346) → optional execution trees (#347) → native TUI
(`#350`, `dev/tui`). Run both clients from the final TUI checkout so they
use the same shared implementation. Set `NOOA_MODEL` to the same configured model
alias in both terminals and use the same workspace/configuration roots.

```bash
export NOOA_MODEL='<configured model alias>'
cd /absolute/path/to/acceptance-workspace
nooa tui --model "$NOOA_MODEL"
```

For a Poolside version that supports the existing agent-server launcher:

```bash
export NOOA_MODEL='<same configured model alias>'
cd /absolute/path/to/acceptance-workspace
pool --agent-server nooa-acp
```

The `nooa` and `nooa-acp` executables must resolve to the final checkout's
environment. The ACP CLI reads `NOOA_MODEL`. Use the client's resume picker for
session selection. A session can have only one live owner: close it in one client
before resuming it in the other.

1. Create a native session. Ask the agent to set `self.v.parity_marker = 'seed'`,
   create a Todo, and run an observed shell command. Check that an automatic title
   is generated and the working directory is correct.
2. Close native cleanly and resume that session through ACP. Verify the title,
   transcript, marker, and Todo. Continue the conversation, then close ACP and
   resume in native. A live session must be hidden from the other resume picker;
   a stale selection must explain that it is already open.
3. Set an explicit title in native and repeat the handoff. Automatic housekeeping
   must preserve the user-selected title.
4. Load a fixture skill with `/skills` or ask the agent to activate it. Exercise
   its slash command through both clients. Ask the agent to remember it with
   `self.workspace_settings.remember_skill(...)`, then create a new session and
   verify automatic activation. Forgetting it must stop future activation.
5. Register a harmless MCP fixture and remember it. Restart a session: remembering
   must not grant approval. Use `/mcp approve NAME` to inspect the exact config,
   then `/mcp approve NAME CODE` to approve it. Verify reconnect and revocation.
   Client-supplied MCP availability depends on what the installed Pool version
   actually sends; workspace registration is a separate path.
6. In native, check the cumulative token toolbar after multiple responses and
   resume. Totals must restore from session history and reset for a new session;
   worker histories are separate.
7. Exercise native interruption, clear, resume, themes, and restart (when enabled).
   Check that a cancelled prompt releases foreground ownership and that shutdown
   finishes before another client resumes the session.

For simultaneous comparison, create a stopped seed and make separate copies:

```bash
python -m nooa_cli.interactive.parity /path/to/SESSION.db --output /new/parity-dir
```

The helper holds the source session's exclusive lock, copies committed SQLite
contents, and writes a hash manifest. Set `NOOA_SESSIONS_DIR` to its `native/` and
`pool/` directories in the respective terminals. The workspace files and user
configuration remain shared, so perform concurrent read-only checks or explicitly
coordinate edits. No memory sidecar is copied.

Long-term memory, idle reflection, and keep-going are deferred. Old preferences
are ignored; the native/ACP agent does not advertise their controls. Existing
memory databases are retained. The standalone memory package is separate and its
existing dev-branch fixes remain in the final layer.

For a custom agent, pass `--agent module:Class` or `--agent file.py:Class` explicitly
to the relevant NOOA executable. Repository `agent_spec` settings are ignored.
