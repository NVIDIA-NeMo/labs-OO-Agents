# Changelog — nooa-atom

Changes to the `nooa-atom` package (NOOA Atom: the interactive coding agent
served over ACP with `nooa atom`, and the Session layer it runs on). Core NOOA
changes are in the repository's root `CHANGELOG.md`. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

- `nooa-atom`: reflection turns. When a root session's turn ends with `Done`
  or `Waiting` and a reflection is due (20 turns or 60 minutes since the last,
  or the context half full; at least 5 turns apart), the session queues
  housekeeping instructions on the `slash_commands` channel: rename the
  session, compact history, delete stale names, tidy todos and `self.v`. It
  runs only when nothing else waits, at most one is pending, and over ACP the
  turn is one "Reflected: ..." status card with no reply. On by default for
  ACP hosts, off headless (benchmarks, Gym, `nooa-bench`); configure
  it in `atom.reflection` or override with `SessionOptions(reflection=...)`.
  Child sessions never reflect; a `children:` key left in `atom.reflection`
  is ignored with a warning. Each one emits an `atom.reflection` span.
- `nooa-atom`: a job lost in a restart now reports the launching cell's code
  (at most 40 lines / about 4,000 characters) and the names that cell used
  which are gone after the restart.
- `nooa-atom`: AtomAgent drops the core `<events_api>` guidance block (added on
  main for agents that expose `self.events`). `self.events` stays exposed, and
  Atom's `context_usage` block already names `self.events.collapse(...)` and
  `print(doc(self.events))`. Generic agents keep the block.
- `nooa-atom`: the ACP server speaks Pool 2's `pool/*` extensions as well as
  Pool 1's `poolside/*` ones, picked per connection from `clientInfo`
  (`name: "pool"`; version major 1 or 2; both for an unversioned build until
  it steers). Pool 2 gets `pool/session_steer` advertised, `pool/inputEventId`
  and `pool/clientInputId` input events, and `_pool/elicitation` forms with an
  `elicitationId`; `_poolside/session_steer` and `_pool/session_steer` are both
  accepted. Pool 1 and other clients get exactly what they got before.
- `nooa-atom`: `/compact` is a built-in slash command. It sends the agent a
  short directive on its `slash_commands` channel ("Please compact your history
  now using self.events.collapse(...)", plus what to keep), and the agent
  compacts with its own tools in that turn. Over ACP (Pool) it used to answer
  "not available through ACP yet". Any extra words are passed on as
  instructions; a host control named `compact` still replaces it.
- `nooa-atom`: `AtomAgent.handle_orchestrate()` (from #446), an unattended
  root that plans a task, hands the work to child sessions (they run
  `handle_work`) and verifies it without editing files itself. It is opt-in:
  `SessionOptions(turn_method="handle_orchestrate")`, which `run_task` keeps
  (any other turn method still means `handle_batch`), or
  `nooa-bench run --turn-method handle_orchestrate`. A Waiting-target check
  now rejects a `Waiting` whose `on` names nothing that can deliver (a child
  that is running, `working` or has queued input, a buffered `delegates`
  item, or a running job); the person is never a valid target. `handle`
  points the model at `Done`/`NeedInput`, the unattended turns at sending or
  spawning a child or finishing with `Done`.
- `nooa-atom`: the child-session interface is reworked (breaking). The two
  verbs carry the difference: `delegate(description, prompt, *, brief=None,
  model=None) -> Done` runs a one-shot worker that may not ask questions and
  is closed after its result; `spawn(description, prompt, *, brief=None,
  model=None) -> ChildRef` starts a kept child that may ask its parent
  questions (`ChildQuestion`, a data item; reply with `child.send()`),
  takes follow-ups with `send()` and stays open until `close()` or the end
  of its tree. Both run the new `handle_work` turn, written for a parent,
  whose `Done` must carry a `TaskResult`; a one-shot worker that returns
  `NeedInput` gets turn feedback to decide and state its assumptions.
  `InteractiveAgent` defines a generic `handle_work` (with the
  `require_task_result` postcondition and `TaskResult`, now in
  `nooa.interactive` and re-exported from `nooa_atom.session.items`), so every
  child runs it: the fallback to `handle_batch`/`handle` for a class without
  one is removed, and such a class raises a `TypeError` at delegate time.
  `retain=`, `mode=` and `keep=` raise a `TypeError`; `context=` is removed.
  `ChildRef` has `wait()`, `send()` (steers a running child, or queues and
  starts a turn) and `close()` only: `steer()`, `info` and
  `pending_question` are removed (a pending question shows in the ref's
  repr). `ChildRef.wait()` returns a `ChildResult` or `ChildQuestion` and
  raises `ChildFailedError` instead of hanging on an idle child or one
  waiting for an answer. A Todo passed to `spawn()`/`delegate()` is a one-way
  brief: it is rendered into the child's first message, and the child's own
  list gets a local copy, activated, that is its working copy. Nothing flows
  back: the parent gets the child's `TaskResult` (and `ChildMessage`/
  `ChildQuestion`) and updates its own Todo after verifying. Any Todo works
  as a brief, managed or hand-built, since only its fields are read;
  `ChildRef.todo_id` records which Todo the child was briefed on. A shared
  per-tree todo manager is on the backlog.
  `brief=` reaches the child with the Todo, after a line naming its parent
  and saying whether it may ask. A closed child stays closed (`send()`
  raises), `ChildRef.status` is live, a child whose turn ended `Waiting`
  reads `working`, and pending questions and queued outcomes are rebuilt
  from the record after a reload. `SessionOptions` refuses unknown fields
  and carries one child flag, `kept`. `AgentMessage.is_result_text` marks a
  turn result's message.
- `nooa-atom`: the Todo copy/merge-back contract of #451 is replaced by the
  one-way brief (breaking for code written against the earlier #451 heads).
  Removed: `AtomAgent.merge_delegated_todo` and `update_delegated_todo`, the
  registry's `_merge_todo`/`record_todo_base`, the `ChildTodoBase` event,
  `TodoManager.adopt()`, `receive_todo()`, `copy_todo()`, `merge_todo()` and
  `with_todo()` (`insert_copy()` gives a child its brief and raises on a
  same-id clash), the `send(todo)` special case,
  and the parent Todo's `child` var. A saved session holding `ChildTodoBase`
  rows still loads; the rows are ignored. `delegate()` no longer adds a
  "Delegated to …: report" comment to the parent's Todo (breaking): the
  parent records what it verified itself.
- `nooa-atom`: Removed ExperimentalAtomAgent; it differed from AtomAgent only in
  its prompt wording and the missing context-usage block. Saved sessions that used
  it now load as AtomAgent (the spec `nooa_atom.agent:ExperimentalAtomAgent` and
  the older experimental-agent spellings, such as `ExperimentalTUIAgent` and
  `ExperimentalCodingAgent`, are session-restore aliases for
  `nooa_atom.agent:AtomAgent`).
- `nooa-atom`: `_nooa/session/inject` inputs now echo once on admission with a
  distinct `acp:inject` source, including a buffered steer's same-ID queued
  fallback. Ordinary prompts/Pool inputs remain un-echoed; replay is unchanged.
  Pool input acknowledgements now fail on failed/skipped transport sends or a
  stopped bridge, and a consumed item's outcome cannot falsely report it shown.
  Failed acknowledgement associations remain bridge-local for explicit same-ID
  notification-only retry after a flush resets the send error. This is not a wire
  retry API or an exactly-once delivery/rendering guarantee across reconnection.

- `nooa-atom`: accepted Pool and standard ACP `NeedInput` form answers now echo
  once as user messages before the agent continues, through the existing admission
  bridge. Typed answers display their serialized JSON. Ordinary prompts,
  yes/no permissions and live decline/cancel behavior are unchanged. Invalid standard
  form answers stay as text questions rather than being admitted as raw data.
- `nooa-coder` is now `nooa-atom` (NOOA Atom), with no aliases for the old names:
  the package `nooa_atom` (extra `nooa[atom]`), the command `nooa atom`, the
  environment variables `NOOA_ATOM_TOKEN` and `NOOA_ATOM_LOG_LEVEL`, the ACP agent
  name `nooa-atom` and the benchmark agent type `atom` (`AtomBenchAgent`). The agent
  is `nooa_atom.agent:AtomAgent` (was `CodingAgent` in `nooa_coder.coding`), its
  options `AtomOptions` and its slash commands `SlashCommand`; its prompt no longer
  calls it a software-development agent, and its state block is `workspace_state`.
  Sessions nooa-coder saved still load: their agent specs, snapshot classes and
  recorded item types map to the new names. Atom keeps its settings in a new `atom`
  section of `settings.yaml`, copied once from `coding`, which the TUI and `nooa-acp`
  keep using.
- `nooa-atom`: `AtomAgent.plan()` and `AtomAgent.get_summarization_status()`, which
  hosts read (the ACP bridge reads the plan after each Python cell), no longer record a
  trace span per call.
- `nooa-atom`: a message steered into a running turn reaches the model in the form of
  a turn's input: a `Notification` whose `value` is `{"user_messages": [text]}`, whose
  `source` names the channel and the sender, and whose `description` says it came during
  the turn and how to reach it (`self.events["N"].value`). A steer from an ACP client
  counts as the user.
- `nooa-atom`: a message typed in Pool during a turn now appears in Pool's conversation.
  When a turn takes it, the server sends the input event Pool's own agent sends
  (`session_info_update` with `_meta` `poolside/clientInputId` and `poolside/inputEventId`),
  and answers `_poolside/session_steer` after it. Prompts and replayed user messages also
  carry `poolside/inputEventId` for Pool. The Session reports a taken item as
  `ItemConsumedUpdate`.
- `nooa-atom`: the ACP server takes the messages typed in Pool during a running turn.
  It advertises `poolside/session_steer`, and each such message is queued for the next
  turn instead of waiting in Pool's own queue, where Esc drops it. It is not steered
  into the running turn until the Pool team says which of the two the request means.
- `nooa-atom`: the open `session/prompt` stays open until the messages Pool handed over
  with `_poolside/session_steer` during it are handled, in order, questions included, as
  Pool expects; before, the prompt ended with its own turn and the next turn ran with no
  prompt open, and Pool did not show its reply. Stop withdraws the ones no turn took and
  lists them in one agent message. A question the agent asks while such a message is
  still waiting is not opened as a form: the queued message's turn comes first and can
  answer it.
- `nooa-atom`: Pool `NeedInput(options=[...])` forms offer the picker plus a
  text alternative in the same field, without Escape. Listed answers normalize
  case and surrounding spaces; other nonblank strings are preserved. Malformed
  or blank answers retry once. Typed `Literal` fields remain strict, and yes/no
  permissions and standard ACP enum forms are unchanged.
- `nooa-atom`: Pool forms show choices as a picker: a choice question and a string
  `Literal` field are a `oneOf` of `{const, title}` entries, and a `Literal[...] | str`
  field is the picker plus free text. A form with more than one field sends
  `_meta["poolside/field_order"]`. Before, a choice was a text box listing the choices.
- `nooa-atom`: with Pool as the ACP client, free-text, choice and typed questions
  (`NeedInput`) open Pool's form (`_poolside/elicitation`) instead of ending the turn as
  text. Typed fields are asked as strings and converted; a choice question is one text
  field that lists the choices, and the answer matches a choice ignoring case. A bad
  value is asked once more. Yes/no questions still use the permission dialog.
- `nooa-atom`: the ACP server accepts Pool's steering. It advertises
  `poolside/session_steer`, and a message typed in Pool during a running turn reaches
  the turn's next model call (queued when no turn is running) instead of waiting in
  Pool's queue.
- `nooa-atom`: with Pool as the ACP client, free-text and typed questions (`NeedInput`)
  open Pool's form (`_poolside/elicitation`) instead of ending the turn as text. Typed
  fields are asked as strings and converted; a bad value is asked once more.
- `nooa-atom`: one `SkillManager` (`self.skills`) for code skills, text skills and MCP
  servers, with `search`, `activate`, `deactivate`, `read` and `doc`, and one small
  `<skills>` block in place of the `<skills>` and `<mcp>` blocks. MCP sign-in uses the
  MCP SDK's OAuth provider with a pasted redirect address (`/mcp auth NAME ADDRESS`);
  tokens are in `~/.config/nooa/mcp_oauth.json`, so servers signed in before sign in once more.
- `nooa-atom`: `/recover` in ACP sessions continues a session that is marked in use,
  for example after a crash on another machine sharing the directory, in a new copy.
  Without an argument it lists those sessions. The original file is never modified;
  queued messages it never read and its subagent sessions are not carried over, and the
  agent is told. `SessionStore.in_use()` and `SessionStore.fork()` are the store API.
- `nooa-atom`: the ACP model picker re-reads the model registry when a registry file changes, so an alias added with `nooa connect` in a terminal appears without restarting the server.
- `nooa-atom`: a session in workspace W also reads `W/.nooa/llm_config.yaml`, the
  file `nooa connect` writes when run in W: for the model picker, for the model client
  of the session and of `set_model`, and for `default_llm_factory()`. It ranks above
  the user and package-project files and below `NEMO_OO_LLM_CONFIG`. Each workspace's
  aliases are kept apart, also when one process serves several workspaces. The server
  logs a workspace's configuration files when its first session starts.
- `nooa atom` (package `nooa-atom`): the ACP server now runs as a
  router by default, with one worker process for each root session. Each
  worker runs in its own process group and talks to the router over a Unix
  socket pair. The router answers `initialize` and `session/list` itself and
  forwards all other messages unchanged. A subagent's session is loaded in
  the worker of its root. Workers exit when their root session closes, when
  the client disconnects, or when the router dies. `--single-process` keeps
  every session in one process. See `docs/acp-router.md`.
- `nooa atom --http` serves ACP over WebSocket at
  `ws://HOST:PORT/acp` (default `127.0.0.1:8765`), the WebSocket profile of
  the ACP remote transport proposal. Each connection gets its own router and
  workers. Clients authenticate with the token in `NOOA_ATOM_TOKEN`
  (`Authorization: Bearer` or `?token=`); `--no-auth` is allowed on loopback
  only. Browser origins other than loopback need `--allowed-origin`. No TLS:
  use an SSH tunnel or a proxy. See `docs/acp-router.md`.

- New workspace package `nooa-atom` (pre-release, not published yet; the
  `atom` extra installs it). It holds the Session layer of the session
  tree design (#388):
  - `Session` owns one `InteractiveAgent` and its turn loop. Items are
    recorded before they are queued (`submit`), each with a receipt that
    can be withdrawn; `prompt` waits for the turn that consumed the item;
    `steer` hands text to the running turn's next model call (or queues
    it); `cancel` returns once the interrupted cell's output and a
    model-visible `TurnCancelled` event are recorded. The agent is
    checkpointed after each turn when its state changed. Listeners get
    data-only session updates.
  - `SessionRegistry` creates root and child sessions in two phases,
    caps depth, loads sessions from disk (re-queueing unhandled items),
    refuses a parent whose child is live elsewhere, and closes children
    first.
  - An agent's `self.session` port delegates to child sessions
    (`delegate`, `ChildRef.wait/send/steer/close`); results arrive as
    `ChildResult`, `ChildQuestion` or `ChildFailed` on the `delegates`
    channel. Children with `retain=False` run unattended (`handle_batch`).
  - `open_tree()` runs a tree in-process for headless use.
  - For hosts: `Session.outcome(item_id)`, `commands()` and
    `invoke_command()` (through an agent's `slash_commands` registry),
    `set_model()` (applied before the next turn, with the registry's
    `llm_factory`), a `prepare` hook on `create()`/`load()`, and `load()`
    options taken from the session's record.
  - Sessions live in their workspace, in `<workspace>/.nooa/sessions`
    (where `nooa-acp` and the TUI keep them), unless `NOOA_SESSIONS_DIR`
    or `--sessions-dir` names one shared directory for all workspaces;
    `sessions_root()` gives the directory. Sessions written by `nooa-acp`
    are listed and load (its `AtomAgent` record maps to the coding
    agent); a saved agent state that cannot be restored is logged, noted
    in the transcript, and the session goes on from an empty state. The
    store is adapted from `nooa_cli.sessions`.
  - Session-layer fixes from its review: the turn loop survives a failed
    turn and `close()` never raises; a cancelled `wait()` or a cancelled
    child turn reaches the parent as a result or `ChildFailed`;
    half-built sessions are closed when `create()`/`load()` fails; the
    session owns the `llm_factory`'s client (built for every session
    without one, including the default model) and records its alias;
    `load()` takes keyword overrides only; `prepare` runs before any item
    is admitted; a steer buffered at shutdown is re-queued; a closing
    parent gets no new children; `TurnFailedError` chains the original
    exception; updates carry full item text and usage changes;
    `install_port` can hide the port from the model docs.
  - The coding agent runs on it: `nooa_atom.agent:AtomAgent`
    (the agent spec hosts name) ends turns with `Done`/`NeedInput`/
    `Waiting`; its `handle_batch` must return a `TaskResult`. The model
    delegates with `delegate()` (a throwaway child, awaited, returning the
    child's `Done`), `spawn()` and `children()`; `rename_session()` is
    async, and a host asks for a title by submitting
    `session_title_request(text)` on `system_messages`.
    The workspace settings, MCP registry and skill controls moved in as
    `nooa_atom.workspace` (`AtomOptions`), and the repository tools as
    `nooa_atom.tools`.
  - The registry builds agents with `create_session_agent` by default
    (workspace settings and skills for coding agents); hosts pass
    `default_llm_factory()` as the registry's `llm_factory`. Agent specs
    may name a file (`./agent.py:Class`, relative to the workspace), and
    the older `nooa_cli.*` coding-agent specs still load.
  - `nooa atom` serves the coding agent over ACP on
    stdio, one process for every session, on the Session layer; the older
    `nooa-acp` is unchanged. Sessions: new, load (attaching to a live
    session and replaying its transcript), list (root sessions of the `cwd`'s
    workspace, or without `cwd` of every workspace the client has named,
    with `_meta["dev.nooa/status"]`), close, and delete through
    the `_nooa/session/delete` extension method. Turns: a prompt sent
    during a turn is queued and returns with the turn that takes it;
    `_nooa/session/inject` queues or steers a message and
    `_nooa/session/revoke_inject` takes it back (after the ACP RFD in
    agent-client-protocol PR #1261); `Waiting` keeps the
    prompt open; cancel closes open tool cards as "Cancelled" before the
    prompt answers `cancelled`; generation limits map to `max_tokens` and
    `max_turn_requests`. A `NeedInput` question is the turn's final
    message and, when the client supports forms, an `elicitation/create`
    form (a yes/no permission request otherwise); the answer continues
    the same prompt. The first prompt asks the agent for a session title.
    Also: slash commands through `Session.invoke_command()`, the `auto`
    mode, a `model` select option over the model registry's aliases,
    thinking chunks, todos as plan updates, usage with children's cost and
    context status, messages from other senders echoed as user chunks, and
    child sessions announced in `_meta["dev.nooa/children"]` with their
    tool cards mirrored into the parent. `/connect` points to the CLI.
  - `nooa atom --tee PATH` and `python -m nooa_atom.acp.tee --log PATH -- COMMAND...`
    record every ACP frame in both directions to a JSON Lines file (mode
    0600); `docs/acp-tee.md` has the Pool settings entry and what the log
    shows about Pool's behaviour.
  - The `nooa-atom` package depends on `nooa-cli`, which provides the `nooa`
    command; there is no separate `nooa-atom` script.
  - `nooa-atom` and `nooa-acp` need `agent-client-protocol` 0.12.1 or
    later (below 0.13). That release adds the library's HTTP and WebSocket
    transports and renames two schema fields: the multi-select item type is
    `StringMultiSelectItems`, and an ACP-transport MCP server's `id` is
    `server_id`.
