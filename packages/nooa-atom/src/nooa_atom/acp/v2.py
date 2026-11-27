# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP v2, the protocol Atom speaks: negotiation, wire values and the connection.

The adapter (``AtomACPAgent``) and its bridges produce and consume v2
values: requests arrive as v2 params, answers and ``session/update``
notifications leave in v2 shape, through a ``Peer``. This module holds
what belongs to the wire: which version a client gets (``negotiate``),
the v2 ``initialize`` answer, the shapes the bridges build (messages,
tool calls, diffs, plans, states), the checks v2 makes on requests, and
``serve_v2``, which serves the adapter on a JSON-RPC connection.

A v1 client is served by ``nooa_atom.acp.v1_shim``, which translates
between v1 and this API; nothing here knows about v1. See
``docs/acp-v2.md``.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import uuid
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from acp import RequestError
from acp.connection import Connection

from nooa_atom.acp.protocol import INJECT_CAPABILITY, pool_dialects, steer_capability

logger = logging.getLogger(__name__)

V1 = 1
V2 = 2
SUPPORTED = (V1, V2)
META_KEY = "dev.nooa/v2"
"""``_meta`` key of the installer's probe (``is_probe``) and its answer."""

INVALID_PARAMS = -32602
REQUEST_CANCELLED = -32800

TEXT_LIMIT = 64_000
"""Longest text one message or thought chunk carries.

A longer text (a large paste, a long answer, a transcript entry replayed by
``session/resume``) goes out as several chunks of the same message, which
a client joins as it joins any stream of chunks. This keeps every message
well under the 1 MiB that WebSocket clients accept by default
(``websockets``): 64,000 characters are at most about 384 KiB of JSON,
even if every character needs a six-byte escape.
"""


class Peer(Protocol):
    """The client, as the adapter and its bridges talk to it: in v2 terms."""

    async def update(
        self, session_id: str, update: dict[str, Any], *, mirrored: bool = False
    ) -> None:
        """Send one ``session/update`` (``update`` is its v2 ``update`` object).

        ``mirrored``: a followed child's update sent by an ancestor's bridge
        (``session_id`` is the child's), not by the child's own bridge.
        """
        ...

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a request (``elicitation/create``, an extension) and return its result."""
        ...


# ---- negotiation ----------------------------------------------------------------


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
    v1 shape and reads v1: it gets v1.

    ``None`` for a missing or malformed ``protocolVersion`` (``-32602``,
    reported by the v1 shim, as before). A request below 1 also gets v1,
    the lowest supported version.
    """
    requested = requested_version(params)
    if requested is None:
        return None
    if requested < V2 or not isinstance(params.get("info"), dict):
        return V1
    if "clientCapabilities" in params or "clientInfo" in params:
        return V1
    return V2


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


def probe_answer(client_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a probe gets in ``_meta["dev.nooa/v2"]``: the versions and the v2 answer."""
    return {"protocolVersions": list(SUPPORTED), "initialize": initialize_result(client_info)}


def agent_info() -> dict[str, Any]:
    try:
        package_version = version("nooa-atom")
    except PackageNotFoundError:
        package_version = "0.0.0"
    return {
        "name": "nooa-atom",
        "title": "NVIDIA Labs Object Oriented Agents (NOOA)",
        "version": package_version,
    }


def initialize_result(client_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """The v2 answer to ``initialize``: what this agent supports.

    The session baseline (list, resume, close) needs no capability.
    ``session/delete`` is offered; MCP servers over stdio and streamable
    HTTP; ``_meta`` carries the NOOA inject extension and Pool's steer
    capability (``client_info`` is the client's v2 ``info``).
    """
    return {
        "protocolVersion": V2,
        "info": agent_info(),
        "capabilities": {
            "session": {"mcp": {"stdio": {}, "http": {}}, "delete": {}},
            "_meta": {**INJECT_CAPABILITY, **steer_capability(pool_dialects(client_info))},
        },
    }


# ---- values the adapter builds -------------------------------------------------------


def new_message_id() -> str:
    return f"msg-{uuid.uuid4().hex}"


def text(value: str) -> dict[str, Any]:
    """A text content block."""
    return {"type": "text", "text": value}


def message(kind: str, value: str, message_id: str | None = None) -> list[dict[str, Any]]:
    """A whole message as ``*_chunk`` updates of one ``messageId``, split at ``TEXT_LIMIT``.

    ``kind`` is ``agent_message``, ``agent_thought`` or ``user_message``.
    """
    message_id = message_id or new_message_id()
    parts = [value[start : start + TEXT_LIMIT] for start in range(0, len(value), TEXT_LIMIT)]
    return [
        {"sessionUpdate": f"{kind}_chunk", "messageId": message_id, "content": text(part)}
        for part in parts or [""]
    ]


def message_update(kind: str, message_id: str, content: list[Any]) -> dict[str, Any]:
    """A whole message (``user_message``, ``agent_message``...): its content replaces the id's."""
    return {"sessionUpdate": kind, "messageId": message_id, "content": content}


def tool_text(value: str) -> dict[str, Any]:
    """A tool call content item holding text."""
    return {"type": "content", "content": text(value)}


def tool_call(tool_call_id: str, **fields: Any) -> dict[str, Any]:
    """A ``tool_call_update`` (an upsert); fields given as ``None`` are left out.

    In v2 ``null`` clears a field, so only what is known is sent.
    """
    update: dict[str, Any] = {"sessionUpdate": "tool_call_update", "toolCallId": tool_call_id}
    for name, value in fields.items():
        if value is not None:
            update[_camel(name)] = value
    return update


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def git_patch(path: str, old: str | None, new: str, *, start_line: int | None = None) -> str:
    """A ``git_patch`` for one text file (absolute paths, no ``a/``/``b/`` prefixes).

    The hunk keeps every line of ``old`` and ``new`` as context or change,
    so both texts can be read back from the patch (``patch_texts``).
    ``start_line`` (1-based) is where ``old`` starts in the file.
    """
    old_lines = (old or "").splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    head = [f"diff --git {path} {path}\n"]
    if old is None:
        head += ["new file mode 100644\n", "--- /dev/null\n", f"+++ {path}\n"]
    else:
        head += [f"--- {path}\n", f"+++ {path}\n"]
    context = max(len(old_lines), len(new_lines), 1)
    body: list[str] = []
    offset = (start_line - 1) if start_line and start_line > 1 else 0
    for line in list(difflib.unified_diff(old_lines, new_lines, n=context))[2:]:
        if line.startswith("@@") and offset:
            line = _shift_hunk(line, offset)
        if line.endswith("\n"):
            body.append(line)
        else:
            body.append(line + "\n\\ No newline at end of file\n")
    return "".join(head + body)


def _shift_hunk(header: str, offset: int) -> str:
    """``@@ -a,b +c,d @@`` with both start lines moved down by ``offset``."""
    try:
        _at, old, new, *rest = header.split(" ")
        old_start, _, old_len = old[1:].partition(",")
        new_start, _, new_len = new[1:].partition(",")
        old_part = f"-{int(old_start) + offset}" + (f",{old_len}" if old_len else "")
        new_part = f"+{int(new_start) + offset}" + (f",{new_len}" if new_len else "")
        return " ".join(["@@", old_part, new_part, *rest])
    except ValueError:
        return header


def patch_texts(patch: str) -> tuple[str | None, str]:
    """The old and new texts of a ``git_patch`` made by ``git_patch`` (old ``None``: a new file)."""
    old: list[str] = []
    new: list[str] = []
    created = False
    in_hunk = False
    last: list[list[str]] = []
    for line in patch.splitlines(keepends=True):
        if not in_hunk:
            if line.startswith("--- /dev/null"):
                created = True
            if line.startswith("@@"):
                in_hunk = True
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("\\"):
            for side in last:
                if side and side[-1].endswith("\n"):
                    side[-1] = side[-1][:-1]
            continue
        tag, body = line[:1], line[1:]
        if tag == " ":
            old.append(body)
            new.append(body)
            last = [old, new]
        elif tag == "-":
            old.append(body)
            last = [old]
        elif tag == "+":
            new.append(body)
            last = [new]
    return (None if created else "".join(old)), "".join(new)


def diff(path: str, old: str | None, new: str, *, start_line: int | None = None) -> dict[str, Any]:
    """A diff content item for one text file (``old`` ``None``: the file was created)."""
    return {
        "type": "diff",
        "changes": [
            {"operation": "add" if old is None else "modify", "path": path, "fileType": "text"}
        ],
        "patch": {
            "format": "git_patch",
            "text": git_patch(path, old, new, start_line=start_line),
        },
    }


def absolute(path: str, cwd: str | os.PathLike[str] | None) -> str | None:
    """``path`` as an absolute path (v2 requires them), ``None`` when it cannot be."""
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path)) if cwd else None


def plan_update(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sessionUpdate": "plan_update",
        "plan": {"type": "items", "planId": "plan", "entries": entries},
    }


def state_update(state: str, **fields: Any) -> dict[str, Any]:
    update: dict[str, Any] = {"sessionUpdate": "state_update", "state": state}
    for name, value in fields.items():
        if value is not None:
            update[_camel(name)] = value
    return update


def error_object(code: int, message_text: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message_text}
    if data is not None:
        error["data"] = data
    return error


def command(name: str, description: str, hint: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "command", "name": name, "description": description}
    if hint:
        out["input"] = {"type": "text", "hint": hint}
    return out


def select_option(
    config_id: str,
    name: str,
    current: str,
    options: list[tuple[str, str]],
    *,
    description: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """A ``select`` config option; ``options`` are ``(value, name)`` pairs."""
    option: dict[str, Any] = {
        "configId": config_id,
        "name": name,
        "type": "select",
        "currentValue": current,
        "options": [{"value": value, "name": label} for value, label in options],
    }
    if description is not None:
        option["description"] = description
    if category is not None:
        option["category"] = category
    return option


# ---- request checks ---------------------------------------------------------------


def invalid(reason: str, **data: Any) -> RequestError:
    return RequestError(INVALID_PARAMS, reason, {"reason": reason, **data})


def check_mcp_servers(servers: Any) -> list[dict[str, Any]]:
    """v2 MCP server configs, checked as v2 requires (``-32602`` otherwise)."""
    if servers is None:
        return []
    if not isinstance(servers, list):
        raise invalid("mcpServers must be an array")
    names: set[str] = set()
    out: list[dict[str, Any]] = []
    for server in servers:
        if not isinstance(server, dict):
            raise invalid("Each MCP server must be an object")
        name, kind = server.get("name"), server.get("type")
        if not isinstance(name, str):
            raise invalid("MCP server name must be a string")
        if name in names:
            raise invalid(f"Duplicate MCP server name {name!r}", name=name)
        names.add(name)
        if kind == "stdio":
            command_path = server.get("command")
            if not isinstance(command_path, str) or not os.path.isabs(command_path):
                raise invalid(
                    f"MCP server {name!r}: command must be an absolute path",
                    name=name,
                    command=command_path,
                )
        elif kind != "http":
            raise invalid(f"MCP server {name!r}: unsupported transport {kind!r}", type=kind)
        out.append(server)
    return out


def _session_id(params: dict[str, Any]) -> str:
    session_id = params.get("sessionId")
    if not isinstance(session_id, str):
        raise invalid("sessionId is required")
    return session_id


def check_prompt(params: dict[str, Any]) -> None:
    blocks = params.get("prompt")
    if not isinstance(blocks, list) or not blocks:
        raise invalid("prompt must be a non-empty array")
    key = params.get("idempotencyKey")
    if key is not None and (not isinstance(key, str) or not key or len(key) > 255):
        raise invalid("idempotencyKey must be a string of 1 to 255 characters")


def check_resume(params: dict[str, Any]) -> None:
    replay_from = params.get("replayFrom")
    if replay_from is not None and (
        not isinstance(replay_from, dict) or replay_from.get("type") != "start"
    ):
        raise invalid("Unsupported replayFrom cursor", replayFrom=replay_from)


# ---- the connection -------------------------------------------------------------


# Updates the core emits only for the v1 shim (v2 has no session modes).
V1_ONLY = frozenset({"current_mode_update"})


class V2Peer:
    """``Peer`` over a JSON-RPC connection."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    async def update(
        self, session_id: str, update: dict[str, Any], *, mirrored: bool = False
    ) -> None:
        if update.get("sessionUpdate") in V1_ONLY:
            return
        await self._conn.send_notification(
            "session/update", {"sessionId": session_id, "update": update}
        )

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        return await self._conn.send_request(method, params)


_SESSION_METHODS = {
    "session/new": "new_session",
    "session/resume": "resume_session",
    "session/list": "list_sessions",
    "session/close": "close_session",
    "session/delete": "delete_session",
    "session/prompt": "prompt",
    "session/set_config_option": "set_config_option",
}


class V2Dispatcher:
    """Routes v2 requests and notifications to the adapter's methods."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def __call__(self, method: str, params: Any, is_notification: bool) -> Any:
        params = params if isinstance(params, dict) else {}
        if is_notification:
            await self._notification(method, params)
            return None
        if method == "initialize":
            return await self._agent.initialize(params)
        if method.startswith("_"):
            return await self._agent.ext_method(method[1:], params)
        name = _SESSION_METHODS.get(method)
        if name is None:
            raise RequestError.method_not_found(method)
        if method in ("session/new", "session/resume"):
            params = {**params, "mcpServers": check_mcp_servers(params.get("mcpServers"))}
        if method not in ("session/new", "session/list"):
            session_id = _session_id(params)
            if self._agent.is_subagent(session_id):
                raise invalid(
                    f"{method} is not permitted for a subagent session", sessionId=session_id
                )
        if method == "session/prompt":
            check_prompt(params)
        elif method == "session/resume":
            check_resume(params)
        return await getattr(self._agent, name)(params)

    async def _notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "session/cancel":
            await self._agent.cancel(params)
        elif method.startswith("_"):
            await self._agent.ext_notification(method[1:], params)


async def serve_v2(
    agent: Any,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    id_base: int | None = None,
    observers: list[Callable[[Any], None]] | None = None,
) -> None:
    """Serve ``agent`` (an ``AtomACPAgent``) to a v2 client until the client leaves."""
    conn = Connection(
        V2Dispatcher(agent), writer, reader, listening=False, observers=list(observers or [])
    )
    if id_base is not None:
        # The library has no option for the first request id (see serve_v1).
        conn._next_request_id = id_base
    agent.on_connect(V2Peer(conn))
    try:
        await conn.main_loop()
    finally:
        await asyncio.shield(conn.close())


__all__ = [
    "META_KEY",
    "SUPPORTED",
    "V1",
    "V2",
    "Peer",
    "V2Peer",
    "check_mcp_servers",
    "diff",
    "git_patch",
    "initialize_result",
    "is_probe",
    "message",
    "negotiate",
    "patch_texts",
    "probe_answer",
    "serve_v2",
    "tool_call",
]
