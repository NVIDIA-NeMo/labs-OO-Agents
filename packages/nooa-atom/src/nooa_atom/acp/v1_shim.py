# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP v1, as a shim over the v2 adapter. The only module that knows v1.

Atom speaks ACP v2 (``nooa_atom.acp.v2``, ``AtomACPAgent``). A client that
negotiates v1 is served by this shim, which translates both ways:

- ``V1Agent`` is the ``acp`` library's v1 ``Agent``: each v1 request is
  turned into the adapter's v2 call and the v2 result back into the v1
  response. It also keeps what only v1 has: session modes, and a
  ``session/prompt`` that stays open until the turn that consumes it ends
  (with Pool's steered messages, which hold it open too).
- ``V1Peer`` is the adapter's ``Peer`` for a v1 client: v2 updates become
  v1 ones (``tool_call`` for a card's first update, ``plan``, diffs with
  old and new text, 0-based lines) and v2 requests the v1 client calls.
  v1 has no subagent sessions: a child's cards are shown in its root
  session with ids ``"{child_id}:{tool_call_id}"``; the rest of a child's
  activity, foreground states, replay markers and background tool calls
  have no v1 form and are dropped.
- ``serve_v1`` serves a v1 connection; ``initialize_response`` and
  ``initialize_result`` are the v1 ``initialize`` answer (with the v2
  answer for the installer's probe, ``v2.is_probe``).

Removing v1 is deleting this module and its tests, and the two places that
call it: ``server.serve_connection`` and ``router.Router._initialize``
(see ``docs/acp-v2.md``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
)
from acp.agent.connection import AgentSideConnection
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    CloseSessionResponse,
    ElicitationFormSessionMode,
    ElicitationSchema,
    Implementation,
    InitializeRequest,
    ListSessionsResponse,
    McpCapabilities,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SessionNotification,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
)
from pydantic import TypeAdapter, ValidationError

from nooa_atom.acp import v2
from nooa_atom.acp.protocol import (
    INJECT_CAPABILITY,
    POOL_DIALECT,
    POOLSIDE_DIALECT,
    pool_dialects,
    steer_capability,
)

if TYPE_CHECKING:
    from nooa_atom.acp.server import AtomACPAgent
    from nooa_atom.session.items import Receipt

_MODES = [
    SessionMode(
        id="auto",
        name="Auto",
        description="The agent runs its tools without asking first.",
    )
]

_POOL_STEER_METHODS = {
    dialect.steer_method: dialect for dialect in (POOLSIDE_DIALECT, POOL_DIALECT)
}

ROLLING_PREFIX = "concise-python-"
"""Ids of concise (rolling) Python cards: v1 shows a child's under the same id."""

BACKGROUND_PREFIX = "job:"
"""Ids of the background tool calls a waiting turn lists (v1 keeps its prompt open instead)."""

_CONFIG_OPTION = TypeAdapter(SessionConfigOptionSelect | SessionConfigOptionBoolean)


# ---- initialize ---------------------------------------------------------------------


def initialize_response(
    protocol_version: int, client_info: Implementation | None = None
) -> InitializeResponse:
    """The static v1 answer to ``initialize``: what this agent supports.

    ``session/delete`` and ``logout`` are not advertised: the 0.12 library
    does not route them, so a client calling them would get "method not
    found". Deleting is the ``_nooa/session/delete`` extension method.
    """
    info = v2.agent_info()
    return InitializeResponse(
        protocol_version=min(protocol_version, PROTOCOL_VERSION),
        agent_capabilities=AgentCapabilities(
            load_session=True,
            # McpCapabilities defaults to all-false, and a client that honours
            # the handshake then filters its HTTP/SSE servers out of
            # session/new. The adapter connects both transports, so say so.
            # `acp` stays off: it is unstable in the spec and not
            # implemented here.
            mcp_capabilities=McpCapabilities(http=True, sse=True),
            session_capabilities=SessionCapabilities(
                list=SessionListCapabilities(),
                close=SessionCloseCapabilities(),
            ),
            field_meta={**INJECT_CAPABILITY, **steer_capability(pool_dialects(client_info))},
        ),
        auth_methods=[],
        agent_info=Implementation(**info),
    )


def _dump(model: Any) -> Any:
    if model is None or isinstance(model, dict):
        return model
    return model.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


def _with_probe(result: dict[str, Any], client_info: Implementation | None) -> dict[str, Any]:
    meta = dict(result.get("_meta") or {})
    meta[v2.META_KEY] = v2.probe_answer(_dump(client_info))
    return {**result, "_meta": meta}


def initialize_result(params: Any) -> dict[str, Any]:
    """The v1 ``initialize`` answer for raw params (the router answers it itself).

    ``RequestError`` (``-32602``) for params v1 cannot read.
    """
    try:
        request = InitializeRequest.model_validate(params)
    except ValidationError as exc:
        raise RequestError.invalid_params({"errors": exc.errors()}) from exc
    result = _dump(initialize_response(request.protocol_version, request.client_info))
    if v2.is_probe(params):
        result = _with_probe(result, request.client_info)
    return result


def _capabilities_to_v2(capabilities: ClientCapabilities | None) -> dict[str, Any]:
    """v1 client capabilities as the adapter reads them (v2's)."""
    if capabilities is None:
        return {}
    out: dict[str, Any] = {}
    elicitation = capabilities.elicitation
    if elicitation is not None:
        out["elicitation"] = {
            mode: {} for mode in ("form", "url") if getattr(elicitation, mode, None) is not None
        }
    return out


# ---- requests -------------------------------------------------------------------------


def _mcp_server_to_v2(server: Any) -> dict[str, Any]:
    """A v1 MCP server config (stdio has no ``type``) as a v2 one; SSE is kept as ``sse``."""
    data = _dump(server) if not isinstance(server, dict) else dict(server)
    kind = data.get("type")
    if kind is None and "command" in data:
        kind = "stdio"
    if kind not in ("stdio", "http", "sse"):
        return {"name": data.get("name", "<unnamed>"), "_typeName": type(server).__name__}
    return {**data, "type": kind}


def _blocks(prompt: list[Any]) -> list[Any]:
    return [_dump(block) for block in prompt]


def config_option_to_v1(option: dict[str, Any]) -> Any:
    out = {key: value for key, value in option.items() if key != "configId"}
    out["id"] = option.get("configId")
    options = out.get("options")
    if isinstance(options, list):
        out["options"] = [
            {**{k: v for k, v in group.items() if k != "groupId"}, "group": group["groupId"]}
            if isinstance(group, dict) and "groupId" in group
            else group
            for group in options
        ]
    return _CONFIG_OPTION.validate_python(out)


def config_options_to_v1(options: Any) -> list[Any]:
    return [config_option_to_v1(option) for option in options or []]


def command_to_v1(command: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": command.get("name"), "description": command.get("description")}
    hint = (command.get("input") or {}).get("hint")
    if isinstance(hint, str):
        out["input"] = {"hint": hint}
    return out


# ---- updates -------------------------------------------------------------------------

_CHUNKS = ("user_message_chunk", "agent_message_chunk", "agent_thought_chunk")
_WHOLE = {
    "user_message": "user_message_chunk",
    "agent_message": "agent_message_chunk",
    "agent_thought": "agent_thought_chunk",
}
_DROPPED = frozenset(
    {"state_update", "subagent_update", "session_message", "replay_complete", "plan_removed"}
)


def tool_content_to_v1(item: Any) -> Any:
    if not isinstance(item, dict) or item.get("type") != "diff":
        return item
    changes = item.get("changes") or [{}]
    patch = item.get("patch") or {}
    old, new = v2.patch_texts(patch.get("text") or "")
    out: dict[str, Any] = {"type": "diff", "path": changes[0].get("path"), "newText": new}
    if old is not None:
        out["oldText"] = old
    return out


class V1Peer:
    """The adapter's ``Peer`` for a v1 client (``acp`` ``Client``)."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self._seen: set[tuple[str, str]] = set()  # cards a v1 client was told of
        self._parents: dict[str, str] = {}  # child -> the session that announced it
        self._names: dict[str, str] = {}

    def _root(self, session_id: str) -> str:
        for _ in range(64):
            parent = self._parents.get(session_id)
            if parent is None:
                return session_id
            session_id = parent
        return session_id

    async def update(
        self, session_id: str, update: dict[str, Any], *, mirrored: bool = False
    ) -> None:
        converted = self._convert(session_id, update, mirrored)
        if converted is None:
            return
        target, v1 = converted
        notification = SessionNotification.model_validate({"sessionId": target, "update": v1})
        await self.client.session_update(target, notification.update)

    def _convert(
        self, session_id: str, update: dict[str, Any], mirrored: bool = False
    ) -> tuple[str, dict[str, Any]] | None:
        kind = update.get("sessionUpdate")
        if kind == "subagent_update" and isinstance(update.get("sessionId"), str):
            child = update["sessionId"]
            if child not in self._parents and child != session_id:
                self._parents[child] = session_id
            if isinstance(update.get("title"), str):
                self._names[child] = update["title"]
            return None
        if kind in _DROPPED:
            return None
        if mirrored and session_id in self._parents:
            # A child mirrored by its parent: v1 shows only its cards, in the root
            # session. (A child the client loaded has its own bridge: not mirrored.)
            if kind != "tool_call_update":
                return None
            return self._root(session_id), self._tool_call(session_id, update)
        if kind in _CHUNKS:
            return session_id, {"sessionUpdate": kind, "content": update.get("content")}
        if kind in _WHOLE:
            blocks = update.get("content") or []
            if not blocks:
                return None
            # A whole message in v1: its blocks as chunks (the replay has one each).
            return session_id, {"sessionUpdate": _WHOLE[kind], "content": blocks[0]}
        if kind == "tool_call_update":
            if str(update.get("toolCallId", "")).startswith(BACKGROUND_PREFIX):
                return None
            return session_id, self._tool_call(None, update)
        if kind == "plan_update":
            plan = update.get("plan") or {}
            return session_id, {"sessionUpdate": "plan", "entries": plan.get("entries") or []}
        if kind == "config_option_update":
            options = [
                _dump(option) for option in config_options_to_v1(update.get("configOptions"))
            ]
            return session_id, {"sessionUpdate": kind, "configOptions": options}
        if kind == "available_commands_update":
            commands = [command_to_v1(c) for c in update.get("availableCommands") or []]
            return session_id, {"sessionUpdate": kind, "availableCommands": commands}
        return session_id, update

    def _tool_call(self, child: str | None, update: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = str(update.get("toolCallId"))
        out = {key: value for key, value in update.items() if key != "sessionUpdate"}
        if child is not None:
            if tool_call_id.startswith(ROLLING_PREFIX):
                if isinstance(out.get("title"), str):
                    name = " ".join(self._names.get(child, "helper").split())[:40] or "helper"
                    out["title"] = out["title"].replace(
                        "python", f"python [{name} · {child[:8]}]", 1
                    )
            else:
                tool_call_id = f"{child}:{tool_call_id}"
            out["toolCallId"] = tool_call_id
        if isinstance(out.get("content"), list):
            out["content"] = [tool_content_to_v1(item) for item in out["content"]]
        if isinstance(out.get("locations"), list):
            out["locations"] = [
                {**location, "line": location["line"] - 1}
                if isinstance(location.get("line"), int)
                else location
                for location in out["locations"]
            ]
        key = (self._root(child) if child else "", tool_call_id)
        first = key not in self._seen
        self._seen.add(key)
        if first:
            out.setdefault("title", "")
            return {"sessionUpdate": "tool_call", **out}
        return {"sessionUpdate": "tool_call_update", **out}

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        if method == "elicitation/create":
            mode = ElicitationFormSessionMode(
                session_id=params["sessionId"],
                requested_schema=ElicitationSchema.model_validate(params["requestedSchema"]),
            )
            return await self.client.create_elicitation(message=params["message"], mode=mode)
        if method.startswith("_"):
            return await self.client.ext_method(method[1:], params)
        raise RequestError.method_not_found(method)


# ---- the agent -----------------------------------------------------------------------


class V1Agent:
    """The adapter as a v1 ``Agent``; anything else is the adapter's (``core``)."""

    def __init__(self, core: AtomACPAgent) -> None:
        self.core = core
        self.peer: V1Peer | None = None
        self.client_capabilities: ClientCapabilities | None = None
        # How many session/prompt requests of each session are open, and the
        # messages Pool handed over with ``_poolside/session_steer``
        # meanwhile (receipt, text), in order: the prompt stays open until
        # they are handled (see ``prompt``).
        self._open: dict[str, int] = {}
        self._followers: dict[str, list[tuple[Receipt, str]]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.core, name)

    def on_connect(self, conn: Client) -> None:
        self.peer = V1Peer(conn)
        self.core.on_connect(self.peer)

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self.client_capabilities = client_capabilities
        await self.core.initialize(
            {
                "protocolVersion": v2.V2,
                "info": _dump(client_info),
                "capabilities": _capabilities_to_v2(client_capabilities),
            }
        )
        response = initialize_response(protocol_version, client_info)
        if v2.is_probe({"_meta": {v2.META_KEY: kwargs.get(v2.META_KEY)}}):
            response = InitializeResponse.model_validate(_with_probe(_dump(response), client_info))
        return response

    def _modes(self, session_id: str) -> SessionModeState:
        session = self.core.session(session_id)
        mode = session.info.mode if session is not None else None
        return SessionModeState(current_mode_id=mode or "auto", available_modes=_MODES)

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        result = await self.core.new_session(
            {
                "cwd": cwd,
                "additionalDirectories": additional_directories,
                "mcpServers": [_mcp_server_to_v2(server) for server in mcp_servers or []],
            }
        )
        return NewSessionResponse(
            session_id=result["sessionId"],
            modes=self._modes(result["sessionId"]),
            config_options=config_options_to_v1(result.get("configOptions")),
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        result = await self.core.resume_session(
            {
                "sessionId": session_id,
                "cwd": cwd,
                "additionalDirectories": additional_directories,
                "mcpServers": [_mcp_server_to_v2(server) for server in mcp_servers or []],
                "replayFrom": {"type": "start"},
            }
        )
        return LoadSessionResponse(
            modes=self._modes(session_id),
            config_options=config_options_to_v1(result.get("configOptions")),
        )

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        result = await self.core.list_sessions({"cwd": cwd, "cursor": cursor})
        return ListSessionsResponse.model_validate(result)

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        await self.core.close_session({"sessionId": session_id})
        return CloseSessionResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        """Answered when the turn that consumes the prompt ends, then its followers.

        Followers are the messages Pool handed over with
        ``_poolside/session_steer`` while this prompt was open, in the order
        they were admitted, including ones admitted while it waits. Each is
        waited for as the prompt is; the prompt answers ``end_turn`` when all
        are done, and stops at the first other answer.
        """
        _message_id, completion = await self.core.start_prompt(
            session_id, _blocks(prompt), echo=False
        )
        self._open[session_id] = self._open.get(session_id, 0) + 1
        # Taken now: Stop pops the list but leaves in it the followers a turn
        # took, whose outcome is then the cancelled turn's.
        followers = self._followers.setdefault(session_id, [])
        try:
            stop_reason = await completion
            while stop_reason == "end_turn" and followers:
                receipt, _text = followers[0]
                try:
                    stop_reason = await self.core.finish_item(session_id, receipt.item_id)
                finally:
                    if followers and followers[0][0] is receipt:
                        followers.pop(0)
            return PromptResponse(stop_reason=cast(Any, stop_reason))
        finally:
            self._open[session_id] -= 1
            if not self._open[session_id]:
                # Left over only when the prompt failed: they stay queued.
                del self._open[session_id]
                self._followers.pop(session_id, None)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Stop the running turn; the messages Pool handed over during the prompt go with it.

        The ones no turn took are withdrawn, so they do not run with no
        prompt open, and listed, so none is lost silently. v1 has no
        subagent sessions to cancel.
        """
        session = self.core.session(session_id)
        if session is None or session_id not in self.core._bridges:
            return
        stopped = [
            text
            for receipt, text in self._followers.pop(session_id, [])
            if session.withdraw(receipt)
        ]
        if stopped:
            listed = "\n".join(f"- {text}" for text in stopped)
            self.core.bridge(session_id).say(
                f"Stopped before these messages were handled:\n\n{listed}"
            )
        await self.core.cancel({"sessionId": session_id})

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse:
        """Set the permission mode. Only ``auto`` exists until tools can ask first."""
        await self.core.set_mode(session_id, mode_id)
        await self.core.bridge(session_id).flush()  # its current_mode_update
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        result = await self.core.set_config_option(
            {"sessionId": session_id, "configId": config_id, "value": value}
        )
        return SetSessionConfigOptionResponse(
            config_options=config_options_to_v1(result.get("configOptions"))
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        dialect = _POOL_STEER_METHODS.get(method)
        if dialect is not None:
            return await self.core._pool_steer(params, dialect, on_submitted=self._steered)
        return await self.core.ext_method(method, params)

    def _steered(self, session_id: str, receipt: Receipt, text: str) -> None:
        if self._open.get(session_id):
            self._followers.setdefault(session_id, []).append((receipt, text))

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        await self.core.ext_notification(method, params)

    async def close(self) -> None:
        await self.core.close()


async def serve_v1(
    core: AtomACPAgent,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    id_base: int | None = None,
    observers: list[Callable[[Any], None]] | None = None,
) -> None:
    """Serve ACP v1 for ``core`` on one stream pair until the peer closes it."""
    # session/close is registered by the library as unstable. initialize()
    # advertises the close capability, so without this flag the agent
    # promises a method that answers "method not found".
    conn = AgentSideConnection(
        cast(Agent, V1Agent(core)),
        writer,
        reader,
        listening=False,
        use_unstable_protocol=True,
        observers=list(observers or []),
    )
    if id_base is not None:
        # acp 0.12 has no option for the first request id: every
        # agent-to-client request takes Connection._next_request_id and
        # increments it. Seed it before listen() so no request can go out
        # with the library's default of 0.
        # TODO(upstream): ask agent-client-protocol for a request-id option.
        conn._conn._next_request_id = id_base
    try:
        await conn.listen()
    finally:
        await asyncio.shield(conn.close())


__all__ = [
    "V1Agent",
    "V1Peer",
    "initialize_response",
    "initialize_result",
    "serve_v1",
]
