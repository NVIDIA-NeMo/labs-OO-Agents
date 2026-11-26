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
| Negotiation | `initialize` answers `min(requested, 1)` | Pick the highest supported version ≤ requested: v1 for 1, v2 for ≥ 2. `-32022` when none |
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

v2 is a protocol adapter at the edge of each ACP connection, over the
existing v1 implementation, so the session layer, the event bridge and
every v1 client stay as they are.

- `nooa_atom.acp.v2` holds the v1↔v2 wire mapping (pure functions over JSON
  objects) and `V2Translator`, the per-connection state (negotiated
  version, prompt acknowledgements, message ids, foreground state,
  children, background items).
- `serve_connection` (plain `nooa-atom acp` and each router worker) puts
  the translator between the stream and the `acp` library connection. On a
  v1 `initialize` it is a pass-through: v1 frames are forwarded byte for byte.
- The router answers `initialize` with the v2 shape when v2 is
  negotiated, routes `session/resume` like `session/load` and
  `session/delete` like `_nooa/session/delete`, and maps announced child
  sessions to their root's worker so `session/cancel` reaches them.
- `AtomACPAgent` learns the negotiated version from `initialize`'s
  `_meta` and, on v2 connections only, (a) acknowledges a prompt as soon as
  `Session.submit` inserted it, with the receipt's item id as the
  `messageId`, (b) returns the session's commands with the setup response,
  (c) has its bridges send turn and child hints in `_meta` that the
  translator turns into `state_update`, `subagent_update`, child-scoped
  updates and background items, and (d) cancels a child session on
  `session/cancel`.

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
8. Tests in `packages/nooa-atom/tests/acp_v2/`; an end-to-end check with Roy.

Not in this change: MCP-over-ACP (`mcp/message`, draft), `tool_call/cancel`
and `terminal/cancel` (not advertised: Atom cannot yet stop one job or
command on its own), agent-owned display terminals (`terminal_update`;
commands keep streaming as tool-call content), `auth/*` (Atom advertises no
auth methods), prompt images/audio/embedded resources.
