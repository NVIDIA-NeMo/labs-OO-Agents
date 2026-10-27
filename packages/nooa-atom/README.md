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

## The Atom agent, its role and resuming sessions

`nooa atom` with no `--agent` serves `nooa_atom.agent:AtomAgent` (also
spelled `nooa_atom.agent.agent:AtomAgent`). Commands and child agents are
its `self.jobs` work (`Shell`, `Subagent`), and files are `self.files`. It
builds on a private base, `nooa_atom.agent.base.AtomAgentBase`, which holds
the session plumbing and is not offered as an agent. `--agent` picks the
class for *new* sessions only; a loaded session uses the agent spec it
recorded.

A session an earlier Atom saved resumes as this agent in place: the same
session id, history, todos, `self.v` and skill-private vars (such as the
mesh identity in `_mesh_vars`). Earlier Atoms recorded the spec
`nooa_atom.agent:AtomAgent` (or a legacy spelling); the spike's
`nooa_atom.agent.unified:UnifiedAtomAgent` loads it too. Custom and file
specs load as they are. On restore the agent sets its own context blocks
again (`python_cell_tools`, `status`, `role`...; the old `queues` block goes)
and keeps any other block. A job that was running when the session stopped
is reported once as a `failed` `JobEvent` ("lost: ...") and is not
relaunched. Child sessions from before the restart are not adopted as jobs
and are not reported; their saved sessions stay on disk.

The agent has one turn method, `handle`. The host picks the agent's role
with `SessionOptions.turn_method`, which the session records, so a resumed
session keeps its role (`nooa_atom.agent.roles`):

| `turn_method` | Role | Enforced at the end of a turn |
| --- | --- | --- |
| `handle` (default) | interactive | `Waiting` only on work that will deliver |
| `handle_batch` | unattended | no `NeedInput`; `Done` has a result; no running work at `Done` |
| `handle_orchestrate` | orchestrator (unattended; subagents do the work) | as unattended |
| `handle_work` (every child session) | child | `Done` has a `TaskResult`; a one-shot child may not ask; no running work at `Done` |

The agent factory turns that selection into the agent's `policy` and its
`instructions`. The instructions are one static `role` block in the prompt
prefix, and empty instructions add no block. Code that builds the agent
itself can pass `AtomAgent(policy=..., instructions=...)`. Whatever the
`turn_method`, the session calls `handle` (`AtomAgent.turn_entry`); an
agent without `turn_entry` still runs the method its `turn_method` names.

Hosts and benchmarks select the role with that same field, so no other
flag is needed:

- `nooa atom` and Pool: new sessions are interactive (`handle`); children
  are `handle_work`.
- `run_task` / `nooa-bench run` / the Harbor `atom` agent type:
  unattended (`handle_batch`, the default), or the orchestrator with
  `--turn-method handle_orchestrate` (`options.turn_method`).

Benchmark runs pinned to the frozen spike commits (b612e2d46 and the 7f9…
pin) measure the code at those commits: the earlier `AtomAgent` with its
four turn methods, and the spike agent selected by its spike spec. They are
not repinned. Runs from this branch measure today's `AtomAgent` with one
`handle` and the role table above.

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
runs unattended turns (`turn_method="handle_batch"`) until the agent returns
`Done`, follows a turn that ends `Waiting` to the next one, and stops at
`max_turns` or `timeout`. With `options.turn_method="handle_orchestrate"`
the root is an orchestrator instead: it plans the task, hands the work to
child sessions, verifies their results and does not edit files itself.
Its children are `handle_work` sessions (`Subagent` work). For the Atom
agent these values select a role, and every turn runs `handle` (see above).
`nooa-bench run --turn-method handle_orchestrate` picks it from the command
line.
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

### ACP activity verbosity

Use `/verbosity silent`, `/verbosity concise`, or `/verbosity verbose` to control
activity sent to the ACP GUI. Enter `/verbosity` with no argument to open a
three-choice **Verbosity** form in Pool (or another ACP client supporting forms).
The form and its answer are handled entirely in ACP, without an agent turn.
Dismissing it leaves the setting unchanged; clients without forms receive the
current level and text-command usage instead.

The adapter also advertises a **Verbosity** select option through
`session/set_config_option` (`configId: "verbosity"`), the protocol used by Pool's
`/mode` picker. Whether arbitrary config options appear in the client's UI is
client-dependent; `/verbosity` explicitly requests a form rather than relying
on that UI. All input paths share the same setting. Changes are saved as
`atom.verbosity` in the workspace's `.nooa/settings.yaml`, preserving other
settings. New sessions and newly loaded attachments use that saved preference;
already-open sibling sessions keep their current level.

```yaml
atom:
  verbosity: concise
```

You can also set a user default in `~/.config/nooa/settings.yaml`. Normal settings
precedence applies: user, workspace, then `NEMO_OO_SETTINGS` overrides. Saves go
to workspace scope, not to user or environment-override files. An environment
override can therefore take precedence again on the next attachment. Without a
configured value (or with an invalid value), verbosity defaults to `verbose`.
This is ACP-only presentation state: it does not change agent/model behavior,
reasoning effort, or stored agent events.

| Level | Python cells | Bash and diffs | Summarization and returned turn results |
| --- | --- | --- | --- |
| `verbose` (default) | Full source and output cards | Full activity cards | Full summary and result details |
| `concise` (experiment) | One rolling full Python tool card per agent/helper turn | Hidden | Same thought output, shortened to one status line each |
| `silent` | Hidden | Hidden | Hidden |

Concise uses a rolling **full Python tool card** for the parent and a separate
card for each helper, including grandchildren. Each agent's first cell opens
its card; subsequent cells replace only that card's title, source, output and
status. Helper cards show the name and a short session ID, for example
`python [reviewer · abcdef12]: …`, so helpers with the same name remain
recognizable. Delayed results from superseded cells cannot replace newer
activity from that agent, and concurrent helpers never overwrite each other.

Each display slot resets on its own agent's turn boundaries. Parent completion
or cancellation does not end a still-running helper's card; helper completion,
cancellation or closure affects only that helper. A verbosity change or bridge
shutdown finalizes all relevant cards. Completed turns leave their last card in
the GUI; a new turn gets a fresh card. Bash and diffs are still hidden. Summarization and
turn-result notices keep their thought output type with one status line. This
experiment is isolated in a separate commit for easy reversion to title-only
per-cell concise cards. ACP presentations are not added to the agent's history.

Model reasoning is only forwarded in `verbose`. Normal agent messages, final
answers, questions/forms, command confirmations, cancellation notices, plans,
and protocol metadata remain visible at every level. Setup/synthetic Python
cells and truncation without a summary do not produce activity lines. Changing
levels finalizes any already-visible open cards (including mirrored child
cards); subsequent activity uses the new level.

A question the agent asks (`NeedInput`) is never a dialog: the bridge sends
its question (and reason) as the agent's message, the prompt ends with
`end_turn`, and the person's next prompt answers it as plain text.

`/verbosity` with no argument is the adapter's own picker and never reaches the
agent. When the client is Pool (`clientInfo.name` is `pool`) it is Pool's
`_poolside/elicitation` form with one required string `verbosity` field whose
`oneOf` lists the three levels (`{const, title}` entries; Pool declines
non-string inputs). Other clients that advertise `elicitation.form` get a
standard `elicitation/create` form with an `enum` of the levels. Declining or
dismissing keeps the level; without a form, or with an answer that is not a
level, the adapter shows `Usage: /verbosity silent|concise|verbose`. If Pool
fails the request, the picker falls back to that text for the rest of the
connection. Stop cancels an open picker.

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
- `doc(name)`: the full description of one skill, with an MCP server's tools. An unconnected MCP skill returns a connection-status message instead of its tools.

The prompt holds one small block, which changes only when something is
activated:

```text
<skills>
Active: libwriting, methodwriting, repo, shell, todo, workspace_settings, github (mcp)
38 more (code, text, MCP): self.skills.search('query') (sync); self.skills.doc('name') (sync) for one skill's full docs (an unconnected MCP skill returns its connection status); await self.skills.activate(['name']) (async); self.skills.read('name') (sync) for text skills
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

Issuer changes remain fail-closed. To intentionally start over, run
`/mcp forget-auth NAME` (one exact configured name, not a glob), or call
`await self.skills.mcp.forget_auth("NAME")`. This cancels and drains live
connections and pending sign-ins for every alias sharing that URL, then
atomically forgets only that URL's local tokens, expiry, registration and
issuer bindings. Other servers' caches, permissions, settings and approvals
are preserved. This is **local forgetting, not remote token revocation**;
reconnect explicitly and complete a fresh manual sign-in. A configured
`oauth_client_id` is retained: intentionally reconfigure it if the new
issuer requires a different client. Unknown names and stdio servers are
rejected; HTTP servers without cached OAuth simply disconnect, retaining
configured headers. URLs with environment placeholders require existing
approval to resolve them.

For legacy same-origin servers, genuinely absent protected-resource metadata
(404/410 candidates) plus valid authorization-server metadata can recover
from an unusable refresh token through normal manual sign-in. Only tokens
and expiry are invalidated, persistently, before waiting for the person;
registration and issuer bindings remain. Malformed metadata, discovery
outages and different issuer bindings do not trigger this recovery. An
alternate issuer with absent metadata still requires explicit forgetting.

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
