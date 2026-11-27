# ACP v2 in NOOA Atom

Status: in progress (branch `feat/acp-v2`). ACP v2 is in
[Preview](https://agentclientprotocol.com/rfds/v2/overview); this follows the
stable v2 baseline schema (`schema/v2/schema.json`) and the
[migration guide](https://agentclientprotocol.com/protocol/v2/migration).

Atom's ACP surface is `nooa_atom.acp` (router, worker, `AtomACPAgent`).
The deprecated `nooa-acp` package is not changed.

## What v2 changes for an agent

| Area | v1 (what Atom does today) | v2 |
|---|---|---|
| Negotiation | `initialize` answers `min(requested, 1)` | Pick the highest supported version ≤ requested: v1 for 1, v2 for ≥ 2. `-32022` when none. Atom serves v2 only to a v2-shaped request (`info`, no `clientCapabilities`/`clientInfo`): T3 Code asks for 2 in v1 shape and reads v1 |
| Initialize shape | `clientCapabilities`/`clientInfo`, `agentCapabilities`/`agentInfo`, boolean markers, `loadSession` | `capabilities`/`info` both ways (`info` required), object markers, session groups under `capabilities.session` (`prompt`, `mcp.stdio`/`mcp.http`, `delete`, ...). Advertising `session` commits to new/list/resume/close/prompt/cancel |
| Prompt lifecycle | `session/prompt` stays open for the turn; its response carries `stopReason` | The response acknowledges insertion with `{messageId}`. A `user_message` update echoes the prompt with that id; `state_update` reports `running` / `requires_action` / `idle` (+ `stopReason`, `error`). Failures after insertion are `idle` + `stopReason: error`, never an error response. Optional `idempotencyKey` |
| Messages | chunks without ids | every chunk carries a `messageId`; whole-message upserts exist |
| Tool calls | `tool_call` + `tool_call_update` | only `tool_call_update` (upsert); statuses gain `paused`/`cancelled`; optional `capabilities.cancel` + `tool_call/cancel` |
| Diffs | `{path, oldText, newText}` | `{changes: [{operation, path, fileType}], patch: {format: git_patch, text}}` with absolute paths |
| Locations | `line` (Atom sends 0-based) | `line` ≥ 1 |
| Plans | `plan` | `plan_update` with `planId` and `type: items` |
| Modes | `modes`, `session/set_mode`, `current_mode_update` | removed; config options only |
| Config options | `id`, `group` | `configId`, `groupId`; `session/set_config_option` has `type: id \| boolean` |
| Commands | `{name, description, input: {hint}}` | `{type: command, ..., input: {type: text, hint}}`; also in `session/new` / `session/resume` responses |
| Load / resume | `session/load` replays history | `session/load` removed; `session/resume` with optional `replayFrom: {type: start}`; `replay_complete` ends a replay; a current `state_update` precedes every resume response |
| Close | close | also ends foreground work: idle `cancelled` before the close response |
| Delete | `_nooa/session/delete` (extension) | `session/delete` behind `session.delete` |
| Permission requests | `{toolCall, options}` | `{title, description?, subject: {type: tool_call \| command}, options}` |
| Elicitation | form/url, string `pattern` | no `pattern`; `requires_action` while the foreground waits on it |
| Subagents | Atom announces children only in `session_info_update._meta["dev.nooa/children"]` and mirrors their tool cards into the parent | `subagent_update` on the parent announces a child; the child's updates carry its own `sessionId` (`state_update`, messages, tool calls); `session_message` for the parent's message to the child; `capabilities.cancel` allows `session/cancel` for it |
| Background work | a `Waiting` turn keeps the prompt open | foreground goes `idle`; `state_update.background` lists the tool calls / terminals / subagents still running |
| MCP servers | stdio has no `type`; SSE allowed | every server has `type`; SSE removed; stdio `command` must be absolute; duplicate names are `-32602` |
| Client fs / terminal | not used by Atom | removed |
| Errors | | `-32601` unknown, `-32002` (+ `data.sessionId`) unknown session, `-32600` before initialize |

Roy (the client this was checked against) asks every agent for v2, falls
back to a v1 connection when the agent answers 1, and on v1 loses
background work, single tool-call/terminal stop, agent terminals,
subagents and MCP-over-ACP.

## Design

v2 is Atom's native ACP. v1 is a thin shim over it, kept for v1 clients
(T3 Code, Pool, Zed by default) and meant to be deleted.

- `nooa_atom.acp.v2`: the protocol module: negotiation, the v2
  `initialize` answer and the installer probe, builders for v2 updates
  (messages with ids, `tool_call_update`, git-patch diffs, plans, state,
  config options, commands), request checks, `V2Peer` and `V2Dispatcher` /
  `serve_v2` (the JSON-RPC dispatch of a v2 connection).
- `AtomACPAgent` (`server.py`) is the v2 core. Its methods take and return
  v2 JSON objects (`new_session`, `resume_session`, `prompt` → `{messageId}`,
  `set_config_option`, `delete_session`, `cancel`...). It talks to the
  client through a `Peer` (`update(session_id, update)`, `request(method,
  params)`), never through the `acp` library's v1 types.
- `ACPEventBridge` emits v2 updates only: `state_update` for the
  foreground state, `subagent_update` and child-scoped updates for
  children, background jobs as `job:<name>` tool calls, whole messages on
  replay followed by `replay_complete`.
- `nooa_atom.acp.v1_shim`: everything v1. `V1Agent` is the `acp` library
  `Agent` for a v1 connection over the core (holds the v1 prompt open
  until its turn ends, maps modes, `session/load` → resume with replay,
  config options and MCP servers); `V1Peer` converts each v2 update to its
  v1 form (first `tool_call_update` → `tool_call`, git patch →
  `{path, oldText, newText}`, `plan_update` → `plan`, child cards mirrored
  into the parent under `<child>:<id>`) and drops what v1 lacks
  (`state_update`, `subagent_update`, `replay_complete`, background jobs);
  `initialize_response` is the v1 `initialize` answer.
- `serve_connection` reads the first frame and negotiates: v2 →
  `v2.serve_v2`; otherwise it imports the shim and serves v1. The router
  answers `initialize` itself the same way (`v2.initialize_result` or
  `v1_shim.initialize_result`).
- Negotiation: v2 only for a v2-shaped request (`protocolVersion` ≥ 2,
  `info`, no `clientCapabilities`/`clientInfo`); T3 Code sends 2 in v1
  shape and gets v1. A v1 `initialize` with
  `_meta["dev.nooa/v2"] = {"probe": true}` (the installer's `acp --sbx` /
  `acp` routers) also gets `_meta["dev.nooa/v2"] = {"protocolVersions":
  [1, 2], "initialize": <v2 answer>}`.

### Removing v1

When no client needs v1 (Zed's default is still v1 as of Zed main
b47d9288, Nov 2026):

1. Delete `acp/v1_shim.py`.
2. In `server.py` `serve_connection`, answer a non-v2 `initialize` with
   `-32022` instead of importing the shim; in `router.py` `_initialize`,
   do the same instead of `v1_shim.initialize_result`.
3. Drop `V1_ONLY` (`current_mode_update`) from `v2.py` and the
   `ModeChangedUpdate` branch in the bridge, and `set_mode`/`MODES` in the
   core.
4. Move the v1 suites (`tests/acp_adapter` through `make_adapter`, which
   wraps the core in `V1Agent`, and the v1 router tests) to v2 or delete
   them.

Mapping of turn outcomes to `idle` stop reasons: `done` / `need_input` →
`end_turn`; `cancelled` → `cancelled`; generation limits →
`max_tokens` / `max_turn_requests`; an error → `error` with the JSON-RPC
error; `waiting` → `end_turn` with the waited-on jobs listed as background
tool calls until a later turn ends.

## Plan (priority as Roy uses it)

1. Negotiation and initialize shapes; v1 unchanged byte for byte.
2. Prompt lifecycle: acknowledgement, `user_message`, `state_update`,
   stop reasons, errors, idempotency keys, cancel and close ordering.
3. Update and request shapes: message ids, tool calls, diffs, plans,
   commands, config options (and `set_config_option`), permission and
   elicitation requests (with `requires_action`).
4. `session/resume` (with and without replay, `replay_complete`, state
   snapshot) and `session/delete`.
5. Subagent sessions: `subagent_update`, child-scoped updates, child state
   mirroring, `session_message`, child cancel.
6. Background work: waiting turns end foreground work; jobs listed in
   `background`.
7. Router: v2 initialize, routing of the renamed methods, child routing.
8. Tests: the v1 suites run through the shim; `tests/acp_adapter/test_v2.py`
   and `tests/acp_router/test_router_v2.py` are native v2; an end-to-end
   check with Roy.

Not in this change: MCP-over-ACP (`mcp/message`, draft), `tool_call/cancel`
and `terminal/cancel` (not advertised: Atom cannot yet stop one job or
command on its own), agent-owned display terminals (`terminal_update`;
commands keep streaming as tool-call content), `auth/*` (Atom advertises no
auth methods), prompt images/audio/embedded resources.
