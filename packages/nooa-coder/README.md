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
(agent-client-protocol PR #1261). Pool steers with its own request,
`_poolside/session_steer`, which the server advertises and handles the same
way as a steer inject.

A question the agent asks (`NeedInput`) goes to the client as a form when
the client advertises `elicitation.form`, as a permission request when it
is a yes/no choice, and otherwise as text answered by the next message.
When the client is Pool (`clientInfo.name` is `pool`), free-text and typed
questions use Pool's `_poolside/elicitation` form instead. Pool shows string
fields only, so every field is sent as a string with the expected type in its
description ("a whole number", "yes or no", "a comma-separated list") and the
answer is converted back; an answer that does not convert is asked once more,
then left as text. Choices keep the permission request or text. If Pool fails
the request, questions fall back to text for the rest of the connection.

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
