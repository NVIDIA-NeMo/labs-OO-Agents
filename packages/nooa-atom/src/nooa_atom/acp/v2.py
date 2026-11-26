# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP v2 at the edge of a connection, over the v1 implementation.

A client that asks for ``protocolVersion`` 2 or later gets v2; any other
client gets v1, byte for byte as before (``serve_v2_edge`` is then a pipe).
On a v2 connection, ``V2Translator`` maps each client frame to the v1
frame the ``acp`` library serves, and each v1 frame the agent writes to
its v2 form. See ``docs/acp-v2.md`` for the mapping.

The adapter (``AtomACPAgent``) cooperates through ``_meta[META_KEY]``:

- ``initialize`` carries ``{"protocolVersion": 2}``, so the adapter knows
  the connection speaks v2;
- a forwarded ``session/prompt`` carries ``{"ack": token}``; the adapter
  answers with the ``_nooa/v2/accepted`` notification
  (``{"ack", "messageId"}``) once the prompt is inserted, and the
  translator then answers the client's prompt request;
- setup responses carry ``{"availableCommands": [...]}``;
- the bridges send ``session_info_update`` hints (``turn``, ``child``,
  ``childTurn``, ``childMessage``) and tag a child's updates with
  ``{"session": child_id}``; hints never reach the client.

Only this module and those hooks know about v2.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import logging
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nooa_atom.acp.framing import FRAME_LIMIT, Frame, encode, read_frame

logger = logging.getLogger(__name__)

V1 = 1
V2 = 2
SUPPORTED = (V1, V2)
META_KEY = "dev.nooa/v2"
"""``_meta`` key of the translator's exchanges with the adapter."""
ACCEPTED_METHOD = "nooa/v2/accepted"
"""The adapter's prompt acknowledgement, as ``ext_notification`` names it (no underscore)."""

INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002
REQUEST_CANCELLED = -32800
UNSUPPORTED_VERSION = -32022

_MESSAGE_CHUNKS = {
    "user_message_chunk": "user_message",
    "agent_message_chunk": "agent_message",
    "agent_thought_chunk": "agent_thought",
}
# Updates that do not end a run of message chunks (v1 has no message ids:
# a run of chunks of one kind is one message, as Roy reads v1).
_TRANSPARENT = frozenset(
    {"usage_update", "session_info_update", "available_commands_update", "config_option_update"}
)
_PROMPT_ID_PREFIX = "nooa-v2-prompt-"
"""Ids of the v1 prompts the translator sends; the client's own ids are answered at insertion."""
_STOP_REASONS = {
    "done": "end_turn",
    "need_input": "end_turn",
    "waiting": "end_turn",
    "cancelled": "cancelled",
    "error": "error",
}


def error_object(code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return error


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


class _Reject(Exception):
    """A client request answered with an error instead of forwarded."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.error = error_object(code, message, data)


# ---- negotiation and initialize ---------------------------------------------


def requested_version(params: Any) -> int | None:
    """The client's ``protocolVersion``, or ``None`` when it is missing or not an integer."""
    if not isinstance(params, dict):
        return None
    value = params.get("protocolVersion")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535:
        return None
    return value


def negotiate(params: Any) -> int | None:
    """The version to speak: v2 for a v2 request of 2 or more, else v1.

    A v2 request carries ``info`` (required in v2) and none of v1's
    ``clientCapabilities`` / ``clientInfo``. T3 Code asks for version 2 in
    v1 shape and reads v1: it keeps v1, as before.

    ``None`` for a missing or malformed ``protocolVersion`` (``-32602``,
    left to the v1 library to report). A request below 1 also gets v1,
    the lowest supported version, as before.
    """
    requested = requested_version(params)
    if requested is None:
        return None
    if requested < V2 or not isinstance(params.get("info"), dict):
        return V1
    if "clientCapabilities" in params or "clientInfo" in params:
        return V1
    return V2


def initialize_params_to_v1(params: dict[str, Any]) -> dict[str, Any]:
    """A v2 ``initialize`` as the v1 library reads it, marked as v2 for the adapter."""
    capabilities = params.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    client: dict[str, Any] = {}
    elicitation = capabilities.get("elicitation")
    if isinstance(elicitation, dict):
        client["elicitation"] = {
            mode: {} for mode in ("form", "url") if isinstance(elicitation.get(mode), dict)
        }
    auth = capabilities.get("auth")
    if isinstance(auth, dict) and isinstance(auth.get("terminal"), dict):
        client["auth"] = {"terminal": True}
    v1: dict[str, Any] = {"protocolVersion": V1, "clientCapabilities": client}
    info = params.get("info")
    if isinstance(info, dict) and isinstance(info.get("name"), str):
        v1["clientInfo"] = {
            "name": info["name"],
            "version": str(info.get("version") or ""),
            **({"title": info["title"]} if isinstance(info.get("title"), str) else {}),
        }
    meta = dict(params.get("_meta") or {}) if isinstance(params.get("_meta"), dict) else {}
    meta[META_KEY] = {"protocolVersion": V2, "capabilities": capabilities}
    v1["_meta"] = meta
    return v1


def _marker(value: Any) -> dict[str, Any] | None:
    """A v1 boolean or object support marker as a v2 object marker."""
    if value is True:
        return {}
    if isinstance(value, dict):
        return value
    return None


def initialize_result_to_v2(result: dict[str, Any]) -> dict[str, Any]:
    """The adapter's v1 ``initialize`` answer in v2 shape.

    Everything the adapter advertises maps: prompt content markers, the
    MCP transports (stdio is implicit in v1; SSE has no v2 form), and the
    session baseline, which needs list, resume and close: Atom serves
    resume as v1 load. Deleting is ``_nooa/session/delete``, offered as
    ``session/delete``. Root ``_meta`` moves to ``capabilities._meta``.
    """
    agent = result.get("agentCapabilities") or {}
    prompt = agent.get("promptCapabilities") or {}
    mcp = agent.get("mcpCapabilities") or {}
    session: dict[str, Any] = {
        "mcp": {"stdio": {}, **({"http": {}} if mcp.get("http") else {})},
        "delete": {},
    }
    prompt_v2 = {
        name: {}
        for name in ("image", "audio", "embeddedContext")
        if _marker(prompt.get(name)) is not None
    }
    if prompt_v2:
        session["prompt"] = prompt_v2
    sessions = agent.get("sessionCapabilities") or {}
    if _marker(sessions.get("additionalDirectories")) is not None:
        session["additionalDirectories"] = {}
    capabilities: dict[str, Any] = {"session": session}
    if isinstance(agent.get("_meta"), dict):
        capabilities["_meta"] = agent["_meta"]
    info = dict(result.get("agentInfo") or {"name": "nooa-atom", "version": "0.0.0"})
    info.setdefault("version", "0.0.0")
    v2: dict[str, Any] = {"protocolVersion": V2, "info": info, "capabilities": capabilities}
    methods = [_auth_method_to_v2(method) for method in result.get("authMethods") or []]
    if methods:
        v2["authMethods"] = methods
    if isinstance(result.get("_meta"), dict):
        v2["_meta"] = result["_meta"]
    return v2


def is_probe(params: Any) -> bool:
    """A v1 ``initialize`` that also asks what Atom would answer a v2 client.

    The installer's routers (``acp --sbx``, plain ``acp``) answer
    ``initialize`` themselves from a cached answer and need to know whether
    the Atom they run speaks v2: they ask with
    ``_meta["dev.nooa/v2"] = {"probe": true}``.
    """
    meta = params.get("_meta") if isinstance(params, dict) else None
    own = meta.get(META_KEY) if isinstance(meta, dict) else None
    return isinstance(own, dict) and own.get("probe") is True


def with_probe_answer(result: dict[str, Any]) -> dict[str, Any]:
    """A v1 ``initialize`` answer carrying its v2 form in ``_meta`` (see ``is_probe``)."""
    meta = dict(result.get("_meta") or {}) if isinstance(result.get("_meta"), dict) else {}
    meta[META_KEY] = {
        "protocolVersions": list(SUPPORTED),
        "initialize": initialize_result_to_v2(result),
    }
    return {**result, "_meta": meta}


def _auth_method_to_v2(method: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in method.items() if key != "id"}
    out["methodId"] = method.get("id")
    out.setdefault("type", "agent")
    env = out.get("env")
    if isinstance(env, dict):
        out["env"] = [{"name": name, "value": value} for name, value in env.items()]
    return out


# ---- session setup ----------------------------------------------------------------


def mcp_servers_to_v1(servers: Any) -> list[dict[str, Any]]:
    """v2 MCP server configs as v1 ones; ``_Reject`` (-32602) for what v2 forbids."""
    if servers is None:
        return []
    if not isinstance(servers, list):
        raise _Reject(INVALID_PARAMS, "mcpServers must be an array")
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for server in servers:
        if not isinstance(server, dict):
            raise _Reject(INVALID_PARAMS, "Each MCP server must be an object")
        name = server.get("name")
        kind = server.get("type")
        if not isinstance(name, str):
            raise _Reject(INVALID_PARAMS, "MCP server name must be a string")
        if name in names:
            raise _Reject(INVALID_PARAMS, f"Duplicate MCP server name {name!r}", {"name": name})
        names.add(name)
        if kind == "stdio":
            command = server.get("command")
            if not isinstance(command, str) or not os.path.isabs(command):
                raise _Reject(
                    INVALID_PARAMS,
                    f"MCP server {name!r}: command must be an absolute path",
                    {"name": name, "command": command},
                )
            converted = {key: value for key, value in server.items() if key != "type"}
            converted.setdefault("args", [])
            converted.setdefault("env", [])
            out.append(converted)
        elif kind == "http":
            converted = dict(server)
            converted.setdefault("headers", [])
            out.append(converted)
        else:
            raise _Reject(
                INVALID_PARAMS,
                f"MCP server {name!r}: unsupported transport {kind!r}",
                {"name": name, "type": kind},
            )
    return out


def config_option_to_v2(option: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in option.items() if key != "id" and value is not None}
    out["configId"] = option.get("id", option.get("configId"))
    options = out.get("options")
    if isinstance(options, list):
        out["options"] = [
            {
                **{key: value for key, value in group.items() if key != "group"},
                "groupId": group["group"],
            }
            if isinstance(group, dict) and "group" in group
            else group
            for group in options
        ]
    return out


def config_options_to_v2(options: Any) -> list[dict[str, Any]]:
    return [config_option_to_v2(option) for option in options or [] if isinstance(option, dict)]


def command_to_v2(command: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "command",
        "name": command.get("name"),
        "description": command.get("description") or "",
    }
    hint = (
        (command.get("input") or {}).get("hint") if isinstance(command.get("input"), dict) else None
    )
    if isinstance(hint, str):
        out["input"] = {"type": "text", "hint": hint}
    if isinstance(command.get("_meta"), dict):
        out["_meta"] = command["_meta"]
    return out


def commands_to_v2(commands: Any) -> list[dict[str, Any]]:
    return [command_to_v2(command) for command in commands or [] if isinstance(command, dict)]


def _setup_result_to_v2(result: Any) -> dict[str, Any]:
    """A ``session/new`` or ``session/load`` answer in v2 shape (no ``modes``)."""
    result = dict(result) if isinstance(result, dict) else {}
    result.pop("modes", None)
    meta = result.pop("_meta", None)
    commands = None
    if isinstance(meta, dict):
        own = meta.pop(META_KEY, None)
        if isinstance(own, dict):
            commands = own.get("availableCommands")
        if meta:
            result["_meta"] = meta
    if result.get("configOptions") is not None:
        result["configOptions"] = config_options_to_v2(result["configOptions"])
    else:
        result.pop("configOptions", None)
    if commands:
        result["availableCommands"] = commands_to_v2(commands)
    return result


# ---- updates ------------------------------------------------------------------------


def _absolute(path: Any, cwd: str | None) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path)) if cwd else None


def git_patch(path: str, old: str | None, new: str) -> str:
    """A ``git_patch`` for one text file (absolute paths, no ``a/``/``b/`` prefixes)."""
    old_lines = (old or "").splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    head = [f"diff --git {path} {path}\n"]
    if old is None:
        head += ["new file mode 100644\n", "--- /dev/null\n", f"+++ {path}\n"]
    else:
        head += [f"--- {path}\n", f"+++ {path}\n"]
    body: list[str] = []
    for line in list(difflib.unified_diff(old_lines, new_lines, n=3))[2:]:
        if line.endswith("\n"):
            body.append(line)
        else:
            body.append(line + "\n\\ No newline at end of file\n")
    return "".join(head + body)


def tool_content_to_v2(item: Any, cwd: str | None) -> dict[str, Any] | None:
    """One v1 tool-call content item in v2 shape (``None`` to drop it)."""
    if not isinstance(item, dict):
        return None
    if item.get("type") != "diff":
        return item
    path = _absolute(item.get("path"), cwd)
    if path is None:
        return {
            "type": "content",
            "content": {"type": "text", "text": f"Edited {item.get('path')}"},
        }
    old, new = item.get("oldText"), item.get("newText")
    new = new if isinstance(new, str) else ""
    out: dict[str, Any] = {
        "type": "diff",
        "changes": [
            {"operation": "add" if old is None else "modify", "path": path, "fileType": "text"}
        ],
        "patch": {"format": "git_patch", "text": git_patch(path, old, new)},
    }
    if isinstance(item.get("_meta"), dict):
        out["_meta"] = item["_meta"]
    return out


def tool_call_to_v2(update: dict[str, Any], cwd: str | None) -> dict[str, Any]:
    """A v1 ``tool_call`` or ``tool_call_update`` as a v2 ``tool_call_update`` upsert.

    v1 treats ``null`` like an omitted field; in v2 ``null`` clears, so
    ``null`` fields are dropped.
    """
    out = {key: value for key, value in update.items() if value is not None}
    out["sessionUpdate"] = "tool_call_update"
    if isinstance(out.get("content"), list):
        out["content"] = [
            converted
            for converted in (tool_content_to_v2(item, cwd) for item in out["content"])
            if converted is not None
        ]
    if isinstance(out.get("locations"), list):
        locations = []
        for location in out["locations"]:
            if not isinstance(location, dict):
                continue
            path = _absolute(location.get("path"), cwd)
            if path is None:
                continue
            converted = {"path": path}
            line = location.get("line")
            if isinstance(line, int) and not isinstance(line, bool) and line >= 0:
                # Atom reports v1 lines 0-based (as Zed reads them); v2 lines start at 1.
                converted["line"] = line + 1
            locations.append(converted)
        out["locations"] = locations
    return out


def plan_to_v2(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "sessionUpdate": "plan_update",
        "plan": {"type": "items", "planId": "plan", "entries": list(update.get("entries") or [])},
    }


def permission_request_to_v2(params: dict[str, Any], cwd: str | None) -> dict[str, Any]:
    tool_call = params.get("toolCall")
    title = "Allow this?"
    out: dict[str, Any] = {
        "sessionId": params.get("sessionId"),
        "options": params.get("options") or [],
    }
    if isinstance(tool_call, dict):
        converted = tool_call_to_v2(tool_call, cwd)
        converted.pop("sessionUpdate", None)
        title = tool_call.get("title") or title
        out["subject"] = {"type": "tool_call", "toolCall": converted}
    out["title"] = title
    if isinstance(params.get("_meta"), dict):
        out["_meta"] = params["_meta"]
    return out


def elicitation_request_to_v2(params: dict[str, Any]) -> dict[str, Any]:
    """v2 forms have no string ``pattern``: the agent checks the value itself."""
    out = dict(params)
    schema = out.get("requestedSchema")
    if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        properties = {}
        for name, prop in schema["properties"].items():
            if isinstance(prop, dict):
                prop = {key: value for key, value in prop.items() if key != "pattern"}
            properties[name] = prop
        out["requestedSchema"] = {**schema, "properties": properties}
    return out


# ---- the translator --------------------------------------------------------------


@dataclass
class _Prompt:
    client_id: Any
    session_id: str
    blocks: list[Any]
    key: str | None
    message_id: str | None = None  # set once acknowledged
    turn_seen: bool = False


@dataclass
class _Session:
    cwd: str | None = None
    parent: str | None = None
    state: str = "idle"  # what the client was last told: idle, running, requires_action
    asks: set[str] = field(default_factory=set)  # pending agent requests that block it
    turn_ended: bool = False  # a turn ended while a prompt request was still open
    chunk_kind: str | None = None
    chunk_id: str | None = None
    seen_ids: set[tuple[str, str]] = field(default_factory=set)
    replay: bool = False  # replaying history for a session/resume
    mute: bool = False  # a resume without replay: drop the replayed history
    background: dict[str, str] = field(default_factory=dict)  # job tool call id -> name
    reports_background: bool = False
    described: bool = False
    closed: bool = False


@dataclass
class _Request:
    method: str
    session_id: str | None = None
    replay: bool = False
    params: dict[str, Any] = field(default_factory=dict)


Send = Callable[[dict[str, Any]], None]


def _key(request_id: Any) -> str:
    return json.dumps(request_id)


class V2Translator:
    """One v2 connection's mapping between the client and the v1 agent.

    ``to_client`` and ``to_agent`` take JSON-RPC objects; both are called
    synchronously and in order, so the frames each side sees keep the
    order in which they were produced.
    """

    def __init__(self, to_client: Send, to_agent: Send) -> None:
        self._to_client = to_client
        self._to_agent = to_agent
        self._sessions: dict[str, _Session] = {}
        self._requests: dict[str, _Request] = {}
        self._prompts: dict[str, _Prompt] = {}  # by internal request id
        self._prompt_ids: dict[str, str] = {}  # client request id key -> internal id
        self._keys: dict[tuple[str, str], str] = {}  # (session, idempotency key) -> message id
        self._asks: dict[str, str] = {}  # agent request id key -> session id
        self._children: dict[str, str] = {}  # child id -> parent id
        self._next = 0

    # ---- helpers ---------------------------------------------------------

    def _session(self, session_id: str) -> _Session:
        state = self._sessions.get(session_id)
        if state is None:
            state = self._sessions[session_id] = _Session(parent=self._children.get(session_id))
            if state.parent is not None:
                state.cwd = self._session(state.parent).cwd
        return state

    def _reply(self, request_id: Any, result: Any) -> None:
        self._to_client({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _fail(self, request_id: Any, error: dict[str, Any]) -> None:
        self._to_client({"jsonrpc": "2.0", "id": request_id, "error": error})

    def _update(self, session_id: str, update: dict[str, Any], meta: Any = None) -> None:
        params: dict[str, Any] = {"sessionId": session_id, "update": update}
        if meta:
            params["_meta"] = meta
        self._to_client({"jsonrpc": "2.0", "method": "session/update", "params": params})

    def _state(
        self,
        session_id: str,
        state: str,
        *,
        stop_reason: str | None = None,
        error: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        """Tell the client the session's foreground state (repeats are dropped)."""
        session = self._session(session_id)
        if session.state == state and not force and state != "idle":
            return
        if session.state == state == "idle" and not force:
            return
        session.state = state
        update: dict[str, Any] = {"sessionUpdate": "state_update", "state": state}
        if state == "idle":
            session.turn_ended = False
            session.asks.clear()
            if stop_reason is not None:
                update["stopReason"] = stop_reason
            if error is not None:
                update["error"] = error
        if session.reports_background:
            update["background"] = [
                {"type": "tool_call", "toolCallId": tool_call_id}
                for tool_call_id in session.background
            ]
        self._update(session_id, update)
        parent = self._children.get(session_id)
        if parent is not None:
            mirrored = {key: value for key, value in update.items() if key != "sessionUpdate"}
            self._update(
                parent,
                {"sessionUpdate": "subagent_update", "sessionId": session_id, "state": mirrored},
            )

    def _open_prompts(self, session_id: str) -> list[_Prompt]:
        return [prompt for prompt in self._prompts.values() if prompt.session_id == session_id]

    # ---- client to agent ---------------------------------------------------

    def from_client(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if not isinstance(method, str):
            self._client_reply(message)
            return
        if "id" in message:
            try:
                self._client_request(message, method)
            except _Reject as exc:
                self._fail(message["id"], exc.error)
        else:
            self._client_notification(message, method)

    def _client_request(self, message: dict[str, Any], method: str) -> None:
        request_id = message["id"]
        if request_id is None:
            raise _Reject(INVALID_REQUEST, "Request ids must not be null")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        session_id = params.get("sessionId") if isinstance(params.get("sessionId"), str) else None
        forward = dict(message)
        if method == "initialize":
            # A repeated initialize: answered as the first was, in v2.
            forward["params"] = initialize_params_to_v1(params)
        elif method == "session/new":
            forward["params"] = {
                **params,
                "mcpServers": mcp_servers_to_v1(params.get("mcpServers")),
            }
            self._requests[_key(request_id)] = _Request(method, params=params)
            self._to_agent(forward)
            return
        elif method == "session/resume":
            self._resume(message, params, session_id)
            return
        elif method == "session/prompt":
            self._prompt(message, params, session_id)
            return
        elif method == "session/set_config_option":
            kind = params.get("type")
            if kind not in ("id", "boolean"):
                raise _Reject(INVALID_PARAMS, f"Unsupported config value type {kind!r}")
            v1 = {key: value for key, value in params.items() if key != "type"}
            if kind == "boolean":
                v1["type"] = "boolean"
            forward["params"] = v1
        elif method == "session/delete":
            forward["method"] = "_nooa/session/delete"
        elif method == "session/close":
            pass
        elif method in ("session/load", "session/set_mode", "authenticate", "logout"):
            raise _Reject(METHOD_NOT_FOUND, f"Method not found: {method}", {"method": method})
        elif method in ("tool_call/cancel", "terminal/cancel"):
            # Not advertised: no tool call or terminal carries capabilities.cancel.
            raise _Reject(METHOD_NOT_FOUND, f"Method not found: {method}", {"method": method})
        if session_id is not None and session_id in self._children and method != "session/cancel":
            raise _Reject(
                INVALID_PARAMS,
                f"{method} is not permitted for a subagent session",
                {"sessionId": session_id},
            )
        self._requests[_key(request_id)] = _Request(method, session_id, params=params)
        self._to_agent(forward)

    def _resume(
        self, message: dict[str, Any], params: dict[str, Any], session_id: str | None
    ) -> None:
        if session_id is None:
            raise _Reject(INVALID_PARAMS, "sessionId is required")
        if session_id in self._children:
            raise _Reject(
                INVALID_PARAMS,
                "session/resume is not permitted for a subagent session",
                {"sessionId": session_id},
            )
        replay_from = params.get("replayFrom")
        if replay_from is not None and (
            not isinstance(replay_from, dict) or replay_from.get("type") != "start"
        ):
            raise _Reject(
                INVALID_PARAMS, "Unsupported replayFrom cursor", {"replayFrom": replay_from}
            )
        v1 = {
            "sessionId": session_id,
            "cwd": params.get("cwd"),
            "mcpServers": mcp_servers_to_v1(params.get("mcpServers")),
        }
        if params.get("additionalDirectories"):
            v1["additionalDirectories"] = params["additionalDirectories"]
        if isinstance(params.get("_meta"), dict):
            v1["_meta"] = params["_meta"]
        session = self._session(session_id)
        session.cwd = params.get("cwd") if isinstance(params.get("cwd"), str) else session.cwd
        session.closed = False
        # Children are discarded on resume; they are announced again.
        for child, parent in list(self._children.items()):
            if parent == session_id:
                del self._children[child]
        if replay_from is None:
            session.mute = True
        else:
            session.replay = True
            session.seen_ids.clear()
            session.chunk_kind = session.chunk_id = None
        self._requests[_key(message["id"])] = _Request(
            "session/resume", session_id, replay=replay_from is not None, params=params
        )
        self._to_agent(
            {"jsonrpc": "2.0", "id": message["id"], "method": "session/load", "params": v1}
        )

    def _prompt(
        self, message: dict[str, Any], params: dict[str, Any], session_id: str | None
    ) -> None:
        if session_id is None:
            raise _Reject(INVALID_PARAMS, "sessionId is required")
        if session_id in self._children:
            raise _Reject(
                INVALID_PARAMS,
                "session/prompt is not permitted for a subagent session",
                {"sessionId": session_id},
            )
        blocks = params.get("prompt")
        if not isinstance(blocks, list) or not blocks:
            raise _Reject(INVALID_PARAMS, "prompt must be a non-empty array")
        key = params.get("idempotencyKey")
        if key is not None and (not isinstance(key, str) or not key or len(key) > 256):
            raise _Reject(INVALID_PARAMS, "idempotencyKey must be a non-empty string")
        if key is not None and (session_id, key) in self._keys:
            self._reply(message["id"], {"messageId": self._keys[(session_id, key)]})
            return
        self._next += 1
        internal = f"{_PROMPT_ID_PREFIX}{self._next}"
        self._prompts[internal] = _Prompt(message["id"], session_id, blocks, key)
        self._prompt_ids[_key(message["id"])] = internal
        meta = dict(params.get("_meta") or {}) if isinstance(params.get("_meta"), dict) else {}
        meta[META_KEY] = {"ack": internal}
        v1 = {"sessionId": session_id, "prompt": blocks, "_meta": meta}
        self._to_agent({"jsonrpc": "2.0", "id": internal, "method": "session/prompt", "params": v1})

    def _client_notification(self, message: dict[str, Any], method: str) -> None:
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "$/cancel_request":
            request_key = _key(params.get("requestId"))
            internal = self._prompt_ids.get(request_key)
            if internal is not None:
                prompt = self._prompts.get(internal)
                # An inserted prompt is answered with its messageId; session/cancel stops its work.
                if prompt is None or prompt.message_id is not None:
                    return
                message = {**message, "params": {**params, "requestId": internal}}
        self._to_agent(message)

    def _client_reply(self, message: dict[str, Any]) -> None:
        session_id = self._asks.pop(_key(message.get("id")), None)
        if session_id is not None:
            session = self._session(session_id)
            session.asks.discard(_key(message.get("id")))
            if session.state == "requires_action" and not session.asks:
                self._state(session_id, "running")
        self._to_agent(message)

    # ---- agent to client ----------------------------------------------------

    def from_agent(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if method is None:
            self._agent_reply(message)
        elif method == "session/update":
            self._agent_update(message)
        elif method == "_" + ACCEPTED_METHOD:
            self._accepted(message.get("params") or {})
        elif method == "session/request_permission" and "id" in message:
            params = message.get("params") or {}
            session_id = params.get("sessionId")
            cwd = self._session(session_id).cwd if isinstance(session_id, str) else None
            self._ask(message, session_id)
            self._to_client({**message, "params": permission_request_to_v2(params, cwd)})
        elif method == "elicitation/create" and "id" in message:
            params = message.get("params") or {}
            self._ask(message, params.get("sessionId"))
            self._to_client({**message, "params": elicitation_request_to_v2(params)})
        else:
            self._to_client(message)

    def _ask(self, message: dict[str, Any], session_id: Any) -> None:
        """A request that blocks the session's foreground work: requires_action."""
        if not isinstance(session_id, str):
            return
        session = self._session(session_id)
        request_key = _key(message.get("id"))
        self._asks[request_key] = session_id
        session.asks.add(request_key)
        if session.state == "running":
            self._state(session_id, "requires_action")

    def _accepted(self, params: dict[str, Any]) -> None:
        prompt = self._prompts.get(params.get("ack"))
        message_id = params.get("messageId")
        if prompt is not None and isinstance(message_id, str):
            self._acknowledge(prompt, message_id)

    def _acknowledge(self, prompt: _Prompt, message_id: str) -> None:
        if prompt.message_id is not None:
            return
        prompt.message_id = message_id
        if prompt.key is not None:
            self._keys[(prompt.session_id, prompt.key)] = message_id
        self._reply(prompt.client_id, {"messageId": message_id})
        self._update(
            prompt.session_id,
            {"sessionUpdate": "user_message", "messageId": message_id, "content": prompt.blocks},
        )
        session = self._session(prompt.session_id)
        session.chunk_kind = session.chunk_id = None

    def _agent_reply(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        prompt = self._prompts.pop(request_id, None) if isinstance(request_id, str) else None
        if prompt is not None:
            self._prompt_ids.pop(_key(prompt.client_id), None)
            self._prompt_done(prompt, message)
            return
        if isinstance(request_id, str) and request_id.startswith(_PROMPT_ID_PREFIX):
            return  # a prompt the client no longer waits for (its session closed)
        request = self._requests.pop(_key(request_id), None)
        if request is None:
            self._to_client(message)
            return
        ok = "error" not in message
        result = message.get("result")
        if request.method == "initialize":
            if ok:
                message = {**message, "result": initialize_result_to_v2(result or {})}
        elif request.method == "session/new" and ok:
            session_id = (result or {}).get("sessionId")
            if isinstance(session_id, str):
                self._session(session_id).cwd = request.params.get("cwd")
            message = {**message, "result": _setup_result_to_v2(result)}
        elif request.method == "session/resume":
            session = self._session(request.session_id or "")
            session.mute = False
            if ok:
                if session.replay:
                    self._update(request.session_id or "", {"sessionUpdate": "replay_complete"})
                session.replay = False
                self._state(request.session_id or "", session.state, force=True)
                message = {**message, "result": _setup_result_to_v2(result)}
            session.replay = False
        elif request.method == "session/set_config_option" and ok:
            options = (result or {}).get("configOptions")
            message = {**message, "result": {"configOptions": config_options_to_v2(options)}}
        elif request.method == "session/close" and ok and request.session_id is not None:
            self._closed(request.session_id)
        self._to_client(message)

    def _closed(self, session_id: str) -> None:
        """Before the close answer: end the foreground work the close cancelled."""
        session = self._session(session_id)
        for internal, prompt in list(self._prompts.items()):
            if prompt.session_id != session_id:
                continue
            del self._prompts[internal]
            self._prompt_ids.pop(_key(prompt.client_id), None)
            if prompt.message_id is None:
                self._fail(
                    prompt.client_id, error_object(REQUEST_CANCELLED, "The session was closed")
                )
        if session.state != "idle":
            self._state(session_id, "idle", stop_reason="cancelled")
        session.closed = True
        for child, parent in list(self._children.items()):
            if parent == session_id:
                del self._children[child]

    def _prompt_done(self, prompt: _Prompt, message: dict[str, Any]) -> None:
        """The v1 prompt answered: the foreground work it started has ended."""
        session = self._session(prompt.session_id)
        if session.closed:
            return
        error = message.get("error")
        if prompt.message_id is None:
            if error is not None:
                # Never inserted: the v2 prompt fails as the v1 one did.
                self._fail(prompt.client_id, error)
                return
            # Handled without an insertion the adapter reported (a local
            # command): the prompt is still the user's message.
            self._acknowledge(prompt, f"msg-{uuid.uuid4().hex}")
        stop_reason = (
            "error"
            if error is not None
            else (message.get("result") or {}).get("stopReason", "end_turn")
        )
        if any(other.message_id is not None for other in self._open_prompts(prompt.session_id)):
            # Another open prompt shares this foreground work; its answer ends it.
            if session.state == "idle":
                return
        if session.state != "idle":
            self._state(prompt.session_id, "idle", stop_reason=stop_reason, error=error)
        elif stop_reason != "cancelled" and not prompt.turn_seen:
            self._state(prompt.session_id, "running")
            self._state(prompt.session_id, "idle", stop_reason=stop_reason, error=error)

    # ---- session updates -------------------------------------------------

    def _agent_update(self, message: dict[str, Any]) -> None:
        params = message.get("params") or {}
        session_id = params.get("sessionId")
        update = params.get("update")
        if not isinstance(session_id, str) or not isinstance(update, dict):
            self._to_client(message)
            return
        update = dict(update)
        meta = update.get("_meta")
        own = meta.get(META_KEY) if isinstance(meta, dict) else None
        if isinstance(own, dict):
            meta = {key: value for key, value in meta.items() if key != META_KEY}
            if meta:
                update["_meta"] = meta
            else:
                update.pop("_meta", None)
            if self._hint(session_id, own):
                rest = {
                    key: value
                    for key, value in update.items()
                    if key not in ("sessionUpdate", "_meta") and value is not None
                }
                if update.get("sessionUpdate") == "session_info_update" and not rest and not meta:
                    return
            if isinstance(own.get("session"), str):
                session_id = own["session"]
        kind = update.get("sessionUpdate")
        tool_call_id = update.get("toolCallId")
        if isinstance(tool_call_id, str) and ":" in tool_call_id:
            prefix, rest_id = tool_call_id.split(":", 1)
            if prefix in self._children:
                session_id = prefix
                update["toolCallId"] = rest_id
        session = self._session(session_id)
        if session.mute and kind in (*_MESSAGE_CHUNKS, "tool_call", "tool_call_update", "plan"):
            return
        for converted in self._convert(session_id, session, update):
            self._update(session_id, converted, params.get("_meta"))

    def _convert(
        self, session_id: str, session: _Session, update: dict[str, Any]
    ) -> list[dict[str, Any]]:
        kind = update.get("sessionUpdate")
        if kind in _MESSAGE_CHUNKS:
            out = []
            message_id = update.get("messageId")
            if not isinstance(message_id, str):
                if session.chunk_kind == kind and session.chunk_id is not None:
                    message_id = session.chunk_id
                else:
                    message_id = f"msg-{uuid.uuid4().hex}"
            if session.replay and (kind, message_id) not in session.seen_ids:
                # A replayed message starts empty, so its chunks do not append to
                # content the client may already hold under this id.
                out.append(
                    {"sessionUpdate": _MESSAGE_CHUNKS[kind], "messageId": message_id, "content": []}
                )
            session.seen_ids.add((kind, message_id))
            session.chunk_kind, session.chunk_id = kind, message_id
            out.append(
                {**{k: v for k, v in update.items() if v is not None}, "messageId": message_id}
            )
            return out
        if kind not in _TRANSPARENT:
            session.chunk_kind = session.chunk_id = None
        if kind in ("tool_call", "tool_call_update"):
            return [tool_call_to_v2(update, session.cwd)]
        if kind == "plan":
            return [plan_to_v2(update)]
        if kind == "current_mode_update":
            return []
        if kind == "config_option_update":
            return [{**update, "configOptions": config_options_to_v2(update.get("configOptions"))}]
        if kind == "available_commands_update":
            return [
                {**update, "availableCommands": commands_to_v2(update.get("availableCommands"))}
            ]
        if kind == "session_info_update":
            return [_without_none(update)]
        if kind == "usage_update":
            return [_without_none(update)]
        return [update]

    def _hint(self, session_id: str, hint: dict[str, Any]) -> bool:
        """Apply an adapter hint; ``True`` if the update carried one."""
        handled = False
        turn = hint.get("turn")
        if turn == "started":
            handled = True
            for prompt in self._open_prompts(session_id):
                prompt.turn_seen = True
            self._state(session_id, "running")
        elif turn == "ended":
            handled = True
            self._turn_ended(session_id, hint)
        child = hint.get("child")
        if isinstance(child, dict) and isinstance(child.get("sessionId"), str):
            handled = True
            self._announce(session_id, child)
        child_turn = hint.get("childTurn")
        if isinstance(child_turn, dict) and child_turn.get("sessionId") in self._children:
            handled = True
            child_id = child_turn["sessionId"]
            if child_turn.get("state") == "running":
                self._state(child_id, "running")
            else:
                reason = _STOP_REASONS.get(str(child_turn.get("outcome")), "end_turn")
                if self._session(child_id).state != "idle":
                    self._state(child_id, "idle", stop_reason=reason)
        child_message = hint.get("childMessage")
        if isinstance(child_message, dict) and child_message.get("sessionId") in self._children:
            handled = True
            self._child_message(child_message)
        return handled

    def _turn_ended(self, session_id: str, hint: dict[str, Any]) -> None:
        session = self._session(session_id)
        outcome = str(hint.get("outcome") or "done")
        waiting_on = hint.get("waitingOn") if outcome == "waiting" else None
        names = [name for name in waiting_on or [] if isinstance(name, str)]
        if names or session.background:
            self._background(session_id, names)
        if outcome == "waiting":
            # The prompt stays open in v1; in v2 the foreground work is over and
            # what it waits on runs in the background.
            self._state(session_id, "idle", stop_reason="end_turn")
            return
        if any(prompt.message_id is not None for prompt in self._open_prompts(session_id)):
            session.turn_ended = True  # the prompt's answer carries the stop reason
            return
        error = None
        if outcome == "error":
            error = error_object(INTERNAL_ERROR, str(hint.get("error") or "The turn failed"))
        if session.state != "idle":
            self._state(
                session_id, "idle", stop_reason=_STOP_REASONS.get(outcome, "end_turn"), error=error
            )

    def _background(self, session_id: str, names: list[str]) -> None:
        """List what a waiting turn waits on as background tool calls; finish the rest."""
        session = self._session(session_id)
        session.reports_background = True
        wanted = {f"job:{name}": name for name in names}
        for tool_call_id in list(session.background):
            if tool_call_id not in wanted:
                del session.background[tool_call_id]
                self._update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tool_call_id,
                        "status": "completed",
                    },
                )
        for tool_call_id, name in wanted.items():
            if tool_call_id not in session.background:
                session.background[tool_call_id] = name
                self._update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tool_call_id,
                        "title": f"Waiting on {name}",
                        "kind": "other",
                        "status": "in_progress",
                    },
                )

    def _announce(self, session_id: str, child: dict[str, Any]) -> None:
        child_id = child["sessionId"]
        parent = child.get("parentId") if isinstance(child.get("parentId"), str) else session_id
        if child_id in self._children or child_id == parent:
            return
        self._children[child_id] = parent
        state = self._session(child_id)
        state.parent = parent
        state.cwd = (
            child.get("cwd") if isinstance(child.get("cwd"), str) else self._session(parent).cwd
        )
        update: dict[str, Any] = {
            "sessionUpdate": "subagent_update",
            "sessionId": child_id,
            "capabilities": {"cancel": {}},
        }
        if isinstance(child.get("title"), str) and child["title"]:
            update["title"] = child["title"]
        if state.cwd and os.path.isabs(state.cwd):
            update["cwd"] = state.cwd
        self._update(parent, update)

    def _child_message(self, hint: dict[str, Any]) -> None:
        child_id = hint["sessionId"]
        parent = self._children[child_id]
        text = str(hint.get("text") or "")
        message_id = str(hint.get("messageId") or f"msg-{uuid.uuid4().hex}")
        self._update(
            child_id,
            {
                "sessionUpdate": "session_message",
                "messageId": message_id,
                "senderSessionId": parent,
                "recipientSessionId": child_id,
                "content": [{"type": "text", "text": text}],
            },
        )
        child = self._session(child_id)
        if not child.described and text.strip():
            child.described = True
            self._update(
                parent,
                {
                    "sessionUpdate": "subagent_update",
                    "sessionId": child_id,
                    "description": text[:500],
                },
            )

    def children_of(self) -> dict[str, str]:
        """Announced children and their parents (for tests and the router)."""
        return dict(self._children)


# ---- the edge ------------------------------------------------------------------------


async def serve_v2_edge(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    serve: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Any],
) -> None:
    """Serve ``serve(inner_reader, inner_writer)`` (a v1 ACP agent) to a v1 or v2 client.

    The agent's connection runs on one end of a socket pair; this pumps
    frames between the other end and the client. Until the client's
    ``initialize`` names v2, and for good when it does not, frames pass
    through unchanged.
    """
    ours, theirs = socket.socketpair()
    inner_reader, inner_writer = await asyncio.open_connection(sock=theirs, limit=FRAME_LIMIT)
    edge_reader, edge_writer = await asyncio.open_connection(sock=ours, limit=FRAME_LIMIT)
    to_client: asyncio.Queue[bytes | None] = asyncio.Queue()
    to_agent: asyncio.Queue[bytes | None] = asyncio.Queue()
    translator: V2Translator | None = None
    decided = False

    def send_client(message: dict[str, Any]) -> None:
        to_client.put_nowait(_encode(message))

    def send_agent(message: dict[str, Any]) -> None:
        to_agent.put_nowait(_encode(message))

    def from_client(frame: Frame) -> None:
        nonlocal translator, decided
        if not decided and frame.method == "initialize" and frame.is_request:
            decided = True
            if negotiate(frame.params) == V2:
                translator = V2Translator(send_client, send_agent)
                logger.info("ACP: speaking protocol v2")
        if translator is None:
            to_agent.put_nowait(frame.raw)
        else:
            translator.from_client(frame.message)

    def from_agent(frame: Frame) -> None:
        if translator is None:
            to_client.put_nowait(frame.raw)
        else:
            translator.from_agent(frame.message)

    async def pump_in() -> None:
        try:
            while (frame := await read_frame(reader)) is not None:
                try:
                    from_client(frame)
                except Exception:
                    logger.exception("ACP v2: could not translate a client frame")
        except (ConnectionError, OSError) as exc:
            logger.info("ACP v2 edge: the client connection failed: %s", exc)
        finally:
            to_agent.put_nowait(None)

    async def pump_out() -> None:
        try:
            while (frame := await read_frame(edge_reader)) is not None:
                try:
                    from_agent(frame)
                except Exception:
                    logger.exception("ACP v2: could not translate an agent frame")
        except (ConnectionError, OSError) as exc:
            logger.info("ACP v2 edge: the agent connection failed: %s", exc)
        finally:
            to_client.put_nowait(None)

    async def drain(
        queue: asyncio.Queue[bytes | None], out: asyncio.StreamWriter, *, eof: bool
    ) -> None:
        try:
            while (data := await queue.get()) is not None:
                out.write(data)
                await out.drain()
            if eof and out.can_write_eof():
                out.write_eof()
        except (ConnectionError, OSError) as exc:
            logger.info("ACP v2 edge: a write failed: %s", exc)
            while await queue.get() is not None:
                pass

    tasks = [
        asyncio.create_task(pump_in(), name="nooa-acp-v2-in"),
        asyncio.create_task(pump_out(), name="nooa-acp-v2-out"),
        asyncio.create_task(drain(to_agent, edge_writer, eof=True), name="nooa-acp-v2-to-agent"),
        asyncio.create_task(drain(to_client, writer, eof=False), name="nooa-acp-v2-to-client"),
    ]
    try:
        await serve(inner_reader, inner_writer)
    finally:
        with contextlib.suppress(Exception):
            inner_writer.close()
        # The agent is gone: let its last frames reach the client, then stop reading.
        try:
            await asyncio.wait_for(asyncio.shield(tasks[3]), 5.0)
        except Exception:
            pass
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in (edge_writer,):
            with contextlib.suppress(Exception):
                stream.close()


def _encode(message: dict[str, Any]) -> bytes:
    try:
        return encode(message)
    except (TypeError, ValueError):
        return (json.dumps(message, separators=(",", ":"), default=str) + "\n").encode()


__all__ = [
    "ACCEPTED_METHOD",
    "META_KEY",
    "SUPPORTED",
    "V1",
    "V2",
    "V2Translator",
    "config_options_to_v2",
    "initialize_params_to_v1",
    "initialize_result_to_v2",
    "negotiate",
    "serve_v2_edge",
]
