# nooa-coder

`nooa-coder` holds the Session layer for NOOA interactive agents: a `Session`
that owns one agent and its turn loop, a `SessionRegistry` that keeps the tree
of sessions (a root and the children it delegates to), the agent-side port
(`self.session`) through which an agent creates and talks to children, and a
headless in-process host (`open_tree`). It also holds the coding agent that
runs on that layer: `nooa_coder.coding.agent:CodingAgent` is the agent spec a
host passes in `SessionOptions.agent_spec`.

Status: pre-release. The package is part of the workspace but is not published
yet, and its API may change until the hosts switch over to it. The ACP
adapter and the benchmark host move onto it in later changes.

Sessions are stored per workspace, in `<workspace>/.nooa/sessions`
(`sessions_root(workspace)`), where the `nooa-acp` server and the TUI keep
theirs; sessions they wrote are listed and load, with their saved agent
state when it can be restored. `NOOA_SESSIONS_DIR`, or an explicit
`SessionOptions.sessions_dir` / `--sessions-dir`, names one shared directory
for all workspaces instead. A `SessionRegistry` serves one such directory;
a subagent's session goes in its root's.

A host never holds a session's agent (`Session._agent` is private). It
submits items (`submit`, `prompt`, `steer`, `withdraw`, `cancel`) and reads
and changes the session through data: `info`, `transcript()`, `channels()`,
`model_info()`, `set_model()`, `set_reasoning()`, `set_mode()`,
`commands()` and `invoke_command()`, `plan()` (the agent's plan entries,
such as the coding agent's todos), and, before the first turn,
`prepare_tools()` and `register_tools()`. `subscribe()` delivers every
change as a pydantic update, in order: turns, items, title, mode, model,
reasoning level, commands, usage, children, close, and each of the agent's
own events (`AgentEventUpdate`). The ACP adapter uses only these; a test
checks that the `nooa_coder.acp` package reads no `agent` attribute.

Over ACP, a `session/prompt` sent during a turn is queued, not steered.
`_nooa/session/inject` queues or steers a message without a prompt request,
and `_nooa/session/revoke_inject` takes one back; see `docs/acp-router.md`
("Messages during a turn") and the ACP RFD for message injection
(agent-client-protocol PR #1261). Pool sends what the person types during a
turn with its own request, `_poolside/session_steer`, which the server
advertises. The message is queued for the next turn, not steered, until the
Pool team says which of the two the request means. The open prompt stays open
until those messages are handled; Stop withdraws the ones no turn took and
lists them.

A question the agent asks (`NeedInput`) goes to the client as a form when
the client advertises `elicitation.form`, as a permission request when it
is a yes/no choice, and otherwise as text answered by the next message.
When the client is Pool (`clientInfo.name` is `pool`), free-text, choice and
typed questions use Pool's `_poolside/elicitation` form instead. Pool shows
string fields only, so every field is sent as a string with the expected type
in its description ("a whole number", "yes or no", "a comma-separated list")
and the answer is converted back. A choice question, and a string `Literal`
field, is a picker (a `oneOf` of `{const, title}` entries); a
`Literal[...] | str` field is the picker plus free text (`anyOf`). A form with
more than one field sends `_meta["poolside/field_order"]` with the fields in
model order. A choice answer typed as text matches a choice ignoring case. An
answer that does not convert or
match is asked once more, then left as text. Yes/no questions keep the
permission request. If Pool fails the request, questions fall back to text for
the rest of the connection.

## Recovering a session marked in use

A session in use is marked by its lock file, so that no two processes write
it, including a sandbox and its host sharing the directory. A process that
crashes leaves the mark behind, and the session is then left out of the
session list and cannot be opened. In an ACP session, `/recover` lists the
sessions of the workspace that are marked in use, with their owner and the
time of their last write. `/recover <id, id prefix or title>` copies one
into a new session titled "<title> (recovered)", which `/resume` then
opens. The original file is never modified; if it is still in use
elsewhere, the copy is a branch from that point. Queued messages the
original never read and its subagent sessions are not carried over, and
the agent is told so. A damaged file is copied event by event, and the
reply says how many events could not be read. The store API is
`SessionStore.in_use()` and `SessionStore.fork()`.

The design is tracked in issue #388.
