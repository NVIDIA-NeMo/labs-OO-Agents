# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send each ACP session's turns through NeMo Relay when the host sets
``NEMO_RELAY_PLUGIN_CONFIG_PATH`` to a Relay ``plugins.toml``, the variable the
``nemo-relay`` CLI reads; see "Capturing ACP sessions with NeMo Relay" in the
nooa-atom README. Hosts import this module only then
(``session.tracing.relay_activated``, ``trace_in_relay``).
"""

from __future__ import annotations

import contextvars
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nooa_atom.session.items import ClosedUpdate
from nooa_atom.session.tracing import RELAY_CONFIG_ENV as CONFIG_ENV

if TYPE_CHECKING:
    from nooa_atom.session.session import Session

METADATA_ENV = "NEMO_RELAY_SESSION_METADATA"
_INJECTOR = "nooa-atom-session"
_KEY = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")

_host_metadata: dict[str, Any] | None = None
_session_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "nooa_atom_relay_session", default=None
)
# Each traced session's running turn span, so a child's turn can nest under it.
_open_turns: dict[str, Any] = {}


@asynccontextmanager
async def activated() -> AsyncIterator[None]:
    """Hold Relay's plugin host, configured by ``CONFIG_ENV``, while the server runs."""
    global _host_metadata
    path = os.environ[CONFIG_ENV]
    if not Path(path).is_file():
        raise FileNotFoundError(f"{CONFIG_ENV} names no file: {path}")
    metadata = json.loads(os.environ.get(METADATA_ENV) or "{}")
    # Relay drops all of an injector's metadata for one key off its syntax or one nested or
    # null value.
    if not isinstance(metadata, dict) or not all(
        _KEY.fullmatch(key) and isinstance(value, (str, int, float, bool))
        for key, value in metadata.items()
    ):
        raise ValueError(
            f"{METADATA_ENV} must be a JSON object of strings, numbers or booleans whose keys"
            " are dot-separated segments of ASCII letters, digits, '_' or '-'"
        )
    try:
        import nemo_relay
    except ImportError as error:
        raise ImportError(
            f"{CONFIG_ENV} is set, but nemo_relay is not: install nooa[nemo-relay]"
        ) from error

    async with nemo_relay.plugin.activate({}, path):
        nemo_relay.event_metadata.register_injector(
            _INJECTOR, 0, lambda _event: _session_metadata.get() or {}
        )
        _host_metadata = metadata
        try:
            yield
        finally:
            _host_metadata = None
            nemo_relay.event_metadata.deregister_injector(_INJECTOR)


def _in_turns(middleware: Any) -> Any:
    async def gated(ctx: Any, nxt: Any) -> Any:
        if _session_metadata.get() is None:
            return await nxt(ctx)
        return await middleware(ctx, nxt)

    return gated


def trace_session(session: Session) -> None:
    """Send ``session``'s turns through Relay while the plugin host is held, one
    ``invoke_agent atom`` span per turn (as Relay's own integrations do per run).

    A child that a traced turn creates (call it there) joins that trace: its
    events carry the caller's metadata plus its own ``subagent_session_id``, and
    each of its turns nests under the caller's turn that is running then. Any
    other session, such as a child that ACP loads directly, is a trace of its own.
    """
    if _host_metadata is None:
        return
    import nemo_relay

    from nooa import nemo_relay_middleware as relay_middleware
    from nooa.runtime import middleware

    caller = _session_metadata.get() if session.parent_id is not None else None
    if caller is None:
        metadata = {**_host_metadata, "session_id": session.id, "agent_kind": "atom"}
        parent_id = None
    else:
        metadata = {**caller, "subagent_session_id": session.id}
        parent_id = session.parent_id
    events = session._agent.event_manager
    uninstalls = [
        events.intercept(kind, _in_turns(handler))
        for kind, handler in (
            (middleware.MIDDLEWARE_LLM_CALL, relay_middleware.nemo_relay_llm_middleware),
            (middleware.MIDDLEWARE_EXECUTE_PYTHON, relay_middleware.nemo_relay_tool_middleware),
        )
    ]
    opened: list[tuple[Any, contextvars.Context]] = []

    def begin(_event: object) -> None:
        # In the turn's own context: post-settlement maintenance (Done
        # reflection) runs in a copy of the loop's and stays out of Relay.
        _session_metadata.set(metadata)
        caller_turn = _open_turns.get(parent_id) if parent_id is not None else None
        handle = nemo_relay.scope.push(
            "atom", nemo_relay.ScopeType.Agent, handle=caller_turn, metadata=metadata
        )
        _open_turns[session.id] = handle
        # Relay's scope stack lives in a ContextVar and the turn settles in the
        # loop's context: pop in a copy of the turn's.
        opened.append((handle, contextvars.copy_context()))

    def end() -> None:
        _open_turns.pop(session.id, None)
        for handle, context in reversed(opened):
            context.run(nemo_relay.scope.pop, handle)
        opened.clear()

    def on_update(update: object) -> None:
        if isinstance(update, ClosedUpdate):
            for uninstall in uninstalls:
                uninstall()
            end()

    uninstalls.append(events.on("TurnBegan", begin))
    uninstalls.append(events.on("TurnSettled", lambda _event: end()))
    session.subscribe(on_update)
