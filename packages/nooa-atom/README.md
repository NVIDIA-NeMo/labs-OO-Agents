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

### Questions and explicit forms

Module-visible constructors from `nooa.interactive` work in generated CodeAct
cells and tagged JSON/Predict outputs:

```python
from nooa.interactive import (
    NeedInput, NeedInputForm, FormResponse, TextQuestion,
    PickOneQuestion, PickOneOrTextQuestion, FormChoice,
)

# Suggestions only: next ordinary message answers; never a dialog/permission.
return_result(NeedInput(question="Which branch?", options=["main", "dev"]))

# Explicit ordered questions; ids are stable, unique answer keys.
return_result(NeedInputForm(
    heading="Release details", reason="Choose the destination before deploying.",
    questions=[
        TextQuestion(id="name", label="Release name?", help="A human-readable name"),
        PickOneQuestion(id="branch", label="Branch?", choices=[
            FormChoice(value="main", title="Main branch"),
            FormChoice(value="dev", title="Development branch"),
        ]),
        PickOneOrTextQuestion(id="target", label="Destination?", choices=[
            FormChoice(value="staging", title="Staging cluster"),
        ]),
        TextQuestion(id="notes", label="Notes?", required=False),
    ],
))

# Outcome on notification["user_messages"], or external host/parent submission:
await session.submit(FormResponse(action="accept", content={
    "name": "Aurora", "branch": "dev", "target": "another cluster", "notes": "",
}))
await child_question.answer(FormResponse(action="decline"))
FormResponse(action="cancel")
```

`NeedInputForm` has `heading`, optional `reason`, and a nonempty `questions` list.
`FormQuestion` is a discriminated union (`kind`: `text`, `pick_one`,
`pick_one_or_text`); constructors supply tags, JSON descriptors must include them.
Each has `id`, `label`, optional `help`, and `required` (default true). Choice
values and question ids must be unique and nonblank. Every accepted answer is a
**dictionary of strings keyed by ids**, live and after JSON roundtrip. Strict
pickers accept exact **values**, not titles or case/whitespace-normalized matches.
Optional missing/whitespace-only answers explicitly become `""`; required missing
or blank answers fail protocol validation. Nonblank text is preserved verbatim.
The required list is sent to clients, but server-side checks do not assume clients
enforce it. No native multi-select, integer, boolean, implicit parsing, defaults,
or domain constraints are promised by this API. Agents validate domain meaning
and ask targeted follow-ups instead of automatically repeating the entire wizard.

`InputRequest` discriminates `kind="question"`/`"form"`; the turn result annotation
is `Done | InputRequest | Waiting`. Both input results are forbidden in unattended
`handle_batch`. The host renders requests, so do not also send them via `message()`.

### Capabilities and failure guarantees

Pool (`clientInfo.name="pool"`) uses `_poolside/elicitation`, with experimentally
tested text (`type: string`), strict picker (`oneOf` const/title), and picker or
free text (`anyOf` picker/string) schemas. These mappings derive from Pool 1.0.16;
**they are not an inventory of all Pool UI types**. Distinct choice labels are
preserved. **Pickers must be required.** Pool 1.0.16 automatically declines the
entire form if any non-text picker is optional; constructors reject
`PickOneQuestion(required=False)` and `PickOneOrTextQuestion(required=False)`
with actionable guidance before a UI request. Optional text remains supported.
If text is appropriate, explicitly author a `TextQuestion(required=False)` with
choice values/titles in its help. This is a text input, not a strict picker:
validate any domain restriction yourself. Neither required flags nor picker
meaning are silently changed. The original seven-field shape with two optional
pickers cannot render unchanged in this version.

Multi-question requests send `_meta["poolside/field_order"]` in list order. No
additional widget flags or speculative constraints are sent. The nested
picker branch of `anyOf` does not need `type: string` in this binary; missing
that type was not the cause of the seven-field failure.

Standard ACP uses advertised `elicitation.form`: text and single string `enum`
map directly. Distinct labels appear in help because a distinct enum-label field
has not been verified. Choice-or-text explicitly falls back to a text property
with suggestions in help, **not** a strict enum or an unverified `anyOf`.

Accept, decline, and cancel retain their separate `FormResponse` actions. Only
accept carries content. A Pool automatic unsupported-schema decline and a user's
Escape decline return the same `{"action":"decline"}` without a reason. The
adapter cannot infer intent from this wire action and does not label it a user
decision or reinterpret it as a pending request. The original heading/reason and
complete descriptors are rendered as visible text before asking, and preserved
on replay; decline is still admitted once as the actual client-reported outcome.
No automatic reask of the whole form is performed. Stop is transport/prompt cancellation and submits no
answer, even if a client swallows cancellation. Ownership is checked without an
await before admission; superseded dialogs cannot admit stale responses. Successful
user-message admission claims/invalidates the form once, even before dispatch starts.
Withdrawal does not resurrect the dialog. ChildQuestion carries an internal durable
request token; its answer method rejects superseded same-id forms. Manual session
submit targets the current request; hosts retaining a request can supply request_id. Responses
are recorded before queueing, echoed once, and replayed without reopening dialogs.
Malformed client payloads fail closed **once** with actionable fallback guidance;
there is no whole-form retry or manufactured answer. Failed Pool extension calls
disable the extension for that connection. Unsupported/text-only hosts preserve
an explicit request with descriptors and response guidance; raw text stays raw,
not an accepted form outcome. No URL/nested-object widget support is claimed.

`FormResponse` validates its envelope and string-dictionary shape. The owning
request's `validate_response()` additionally checks ids, required answers, and
strict selections. External hosts should use `session.submit()`; `admit()` is
record-before-queue plumbing, not an external form-validation API. Child requests
carry descriptors as data; the owning child validates at submission.

### Migration and durable records

Replace draft `NeedInputForm(question=..., answer_type=Model/options=...)` with
`NeedInputForm(heading=..., questions=[...])`. Old typed authoring and presentation
switches are rejected. No dynamic Pydantic answer class is needed or restored.

New durable forms store `outcome_kind="need_input_form"`, `kind="form"`, heading,
reason, ordered descriptors, and an internal ownership token directly; JSON roundtrips preserve them.
Questions retain `need_input`. Recovery preserves request intent without reopening
historical dialogs, and does not restore forms superseded by later admission or
an unfinished turn. Legacy child forms without ownership tokens require a fresh
request before they can be answered through ChildQuestion.answer(). Old explicitly untyped text/choice forms migrate to one
`answer` descriptor with string-dictionary content. Old records containing an
answer class or JSON answer schema remain unavailable original data: acceptance
is refused and the agent must request a new descriptor form, even if the old class
is importable. This does not claim recovery of old custom validators. Decline,
cancel, or explanatory raw text remain possible. Old options-only/auto question
records remain conversational. Old ChildQuestion schema data is retained only as
legacy display data. Historical decline markers cannot recover lost action intent.

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
