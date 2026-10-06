# nooa-atom

`nooa-atom` is NOOA Atom: an interactive agent, served over ACP with
`nooa atom`, and the Session layer it runs on.

The Session layer serves any NOOA interactive agent: a `Session`
that owns one agent and its turn loop, a `SessionRegistry` that keeps the tree
of sessions (a root and the children it delegates to), the agent-side port
(`self.session`) through which an agent creates and talks to children, and a
headless in-process host (`open_tree`). The Atom agent runs on that layer:
`nooa_atom.agent:AtomAgent` is the agent spec a host passes in
`SessionOptions.agent_spec`.

Atom reads its settings from the `atom` section of `settings.yaml` (user and
workspace `.nooa/`). The TUI and `nooa-acp` read `coding`; the first time
Atom reads a settings file that has `coding` and no `atom`, it adds an `atom`
copy, and the two sections change independently from then on.

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

`run_task(options, task)` runs one task unattended in a new tree: the root
runs `handle_batch` turns until the agent returns `Done`, follows a turn
that ends `Waiting` to the next one, and stops at `max_turns` or `timeout`.
It returns a `TaskRun` with the final `Done` and its `TaskResult`, the
reason it stopped otherwise, the turn count, the usage and the root agent's
events. `nooa-bench run` and the benchmark runner's `atom` agent type use
it.

A host never holds a session's agent (`Session._agent` is private). It
submits items (`submit`, `prompt`, `steer`, `withdraw`, `cancel`) and reads
and changes the session through data: `info`, `transcript()`, `channels()`,
`model_info()`, `set_model()`, `set_reasoning()`, `set_mode()`,
`commands()` and `invoke_command()`, `plan()` (the agent's plan entries,
such as the Atom agent's todos), and, before the first turn,
`prepare_tools()` and `register_tools()`. `subscribe()` delivers every
change as a pydantic update, in order: turns, items, title, mode, model,
reasoning level, commands, usage, children, close, and each of the agent's
own events (`AgentEventUpdate`). The ACP adapter uses only these; a test
checks that the `nooa_atom.acp` package reads no `agent` attribute.

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
and the answer is converted back. `NeedInput(options=[...])` offers suggestions
in a picker plus free text (`anyOf` of a `oneOf` of `{const, title}` entries
and a string). The person can type an alternative in the form without Escape.
A listed choice matches ignoring case and surrounding spaces; any other
nonblank string is passed through exactly as entered. Missing, blank or
non-string option answers are asked once more, then left as text.

Typed `answer_type` fields still validate against the Pydantic model. A string
`Literal` is a strict picker; `Literal[...] | str` deliberately adds free text.
A typed answer that does not validate is asked once more, then left as text.
A form with more than one field sends `_meta["poolside/field_order"]` with the
fields in model order. Yes/no questions keep the permission request. Other ACP
clients still receive an enum for `options` when they support standard forms.
If Pool fails the request, questions fall back to text for the rest of the
connection.

These picker/text shapes were measured in Pool 1.0.16. Its built-in agent's
wire capture accepts unlisted text for both single-field and multi-field
`anyOf[oneOf, string]` forms. This is a Pool extension, not general JSON Schema
support: native number, boolean and array widgets are not supported. The
capture establishes accepted values, not the exact on-screen label or key
sequence; rendering in other versions must be checked separately.

## Skills and MCP servers

The Atom agent reaches all its capabilities through one `SkillManager`,
`self.skills`: code skills (`Skill` classes from `nooa.skills` entry points,
workspace libraries and `.py` files in skill directories), text skills
(`SKILL.md` directories) and MCP servers (`.mcp.json`, `atom.mcp_servers`
in the settings files, and the servers an ACP client sends). The model uses:

- `search(query, limit=10)`: one line per match with name, kind, state and
  description. Searching loads nothing; an installed skill that is not
  loaded is described by its package's summary.
- `await activate(names)` / `await deactivate(names)`: a code skill is loaded
  and becomes `self.<name>`; an MCP server is connected and its tools are
  methods of `self.<server>` (deactivating keeps it connected); a text
  skill's instructions arrive on the `system_messages` channel at the start
  of the next turn.
- `read(name)`: a text skill's instructions, as text.
- `doc(name)`: the full description of one skill, with an MCP server's tools.

The prompt holds one small block, which changes only when something is
activated:

```text
<skills>
Active: libwriting, methodwriting, repo, shell, todo, workspace_settings, github (mcp)
38 more (code, text, MCP): self.skills.search('query'); await activate(['name']); read('name') for text skills
</skills>
```

`/skills list` shows every skill with its kind and state; `/skills activate`
and `/skills deactivate` save the choice for code skills in the workspace
settings. `/mcp` shows the MCP servers and their state. A server runs only
after the person approves its exact configuration with
`/mcp approve NAME` (review) and `/mcp approve NAME CODE`.

MCP sign-in uses the MCP SDK's OAuth support (discovery, client
registration, PKCE, token refresh). Because a sandbox cannot receive a
browser callback, sign-in is manual: the link is shown (as an agent message
when the agent activated the server, in the command output for
`/mcp approve`), the person opens it and signs in, then copies the address
the browser ends on (the page may fail to load) and runs
`/mcp auth NAME ADDRESS`. The agent is told when the server connects.
Tokens are kept in `~/.config/nooa/mcp_oauth.json`, readable only by the
user, so later connections need no sign-in. The `oauth_open_browser` and
`oauth_manual` server settings are accepted but no longer used.

A session saved before the skill manager loads normally; the old `<mcp>`
block is dropped and the agent is told what was not restored.

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


## Dispatch and durable input

Session supplies two core turn hooks: cancellable preparation for pending model
activation, and synchronous commit for durable input ownership. Initial
`ItemConsumed` rows and `TurnStarted` are appended in one SQLite transaction before
any dequeue or agent invocation. Core does not know Session IDs or storage.
`admit()` is the synchronous record-before-queue surface for registry event delivery;
`submit()` remains its async facade. `requeue()` owns `ItemRequeued` and the queue
transfer, without changing identity or replay policy. Lifecycle/client-sharing hooks
are constructor parameters (`before_close`, `llm_in_use`); `closing` is public.

A preparation/commit failure leaves inputs queued and pauses dispatch. Existing
prompt/outcome waiters fail with `TurnFailedError`, including a later outcome query;
new external admission is rejected while `dispatch_error` is set. Repair the fault,
then call `session.resume_dispatch()` explicitly, or close/reopen to replay eligible
input. Resume first requeues retained buffered steers. Already reported terminal
outcomes are not rewritten by retry: observe updates or submit a new prompt. A
preparation cancellation likewise pauses without consumption and resolves open
prompts as cancelled; explicit resume can retry the still-queued input. A normal
committed cancellation retains the existing cancelled-output/TurnCancelled contract.

Blocked dispatch rejects external submit/prompt/steer work, but internal registry
admission still records and queues child deliveries while paused. Finished children
are auto-closed only after successful delivery; if ledger admission itself fails,
the child remains live with its own durable result available for recovery.

Pause is live dispatch control, not durable suspension. Session close pauses before
children cleanup, then stops/settles the loop and closes owned resources. Recursive
close from cleanup (including callback child tasks) is safe; self-close from an
executing turn is rejected. Closed sessions resolve late outcome queries for queued
admissions too. Core loop details are in `src/nooa/runtime/README.md`.

Steers during preparation are ordinary queued messages. During execution, buffered
steers are consumed one at a time before the next model call; close/cancel/failure
leftovers are admitted as messages with the same IDs. Recording failures retain
remaining buffer ownership, fail relevant outcomes and block dispatch. Mid-turn
get/drain consumption keeps its identity until the ledger write succeeds; failure
moves consumed-but-unrecorded ownership into a separate orphan recovery map (not
queued identity tracking), retaining the object, channel and error. It fails the
relevant prompt and the turn even if the method returns Done. Submitting the same
object after resume creates a new, independently settled receipt. The raw
item has already reached the consumer at that point, so resume cannot magically
replay it in memory. Close/reopen can replay its never-recorded consumption; external
side effects may already have occurred. This is not exactly-once execution.

Model selections are persisted immediately; effective model info/options change
only on successful activation. Pending activation is attempted once, not implicitly
retried every turn. A failed activation restores the old client and retains the new
client for cleanup. Old/pending/failed clients remain owned across cancelled disposal
or metadata failure; clients shared with children are retired until children close.
A hung old-client close is cancellable preparation with input still queued and old
cleanup ownership retained. Final resource close requires cooperative clients; there
is no timeout/forced-disposal policy in this change.

### Compatibility and scope

The prototype `before_turn(notification)` was new, unpublished and had no external
callers in this checkout. It combined async preparation with recording after
consumption, so preserving that contract would defeat the approved boundary. It is
replaced by `prepare()` and synchronous `commit(batch)`, rather than a compatibility
shim that silently changes notification timing. Supplied hook tests migrate to these
APIs and explicit blocked-dispatch recovery; unrelated assertions stay intact.
`stop_starting()` remains an alias for pause. Existing race callers are unchanged.

Crash load still requeues only admitted items without recorded consumption,
withdrawal or discard. It does not resume Python coroutines or consumed Waiting
obligations. No new delivery IDs or replay policy are introduced. Commands/mode
remain host-neutral; ask mode still does not enforce permissions. Later reasoning,
MCP/auth, command serialization and adapter initialization are outside this change.
Shared reader guides and GitHub are not modified.
