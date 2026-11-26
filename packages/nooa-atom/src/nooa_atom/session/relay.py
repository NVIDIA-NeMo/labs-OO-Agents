# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send each ACP session's turns through NeMo Relay when the host sets
``NEMO_RELAY_PLUGIN_CONFIG_PATH`` to a Relay ``plugins.toml``, the variable the
``nemo-relay`` CLI reads; see "Capturing ACP sessions with NeMo Relay" in the
nooa-atom README. Unset, ``nemo_relay`` is not imported and nothing changes.
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

if TYPE_CHECKING:
    from nooa_atom.session.session import Session

CONFIG_ENV = "NEMO_RELAY_PLUGIN_CONFIG_PATH"
METADATA_ENV = "NEMO_RELAY_SESSION_METADATA"
_INJECTOR = "nooa-atom-session"
_KEY = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")

_host_metadata: dict[str, Any] | None = None
_session_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "nooa_atom_relay_session", default=None
)


@asynccontextmanager
async def activated() -> AsyncIterator[None]:
    """Hold Relay's plugin host while the server runs, when ``CONFIG_ENV`` is set."""
    global _host_metadata
    path = os.environ.get(CONFIG_ENV)
    if not path:
        yield
        return
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
    """Send ``session``'s turns through Relay while the plugin host is held: a child's, when a
    traced turn creates it (call it there), in that turn's trace; any other session's, such as
    a child that ACP loads directly, as a trace of its own."""
    if _host_metadata is None:
        return
    import nemo_relay

    from nooa import nemo_relay_middleware as relay_middleware
    from nooa.runtime import middleware

    turn = _session_metadata.get() if session.parent_id is not None else None
    if turn is None:
        metadata = {**_host_metadata, "session_id": session.id, "agent_kind": "atom"}
        parent = None
    else:
        # The caller's trace, plus the child's own id to tell its events apart.
        metadata = {**turn, "subagent_session_id": session.id}
        parent = nemo_relay.scope.get_handle()
    events = session._agent.event_manager
    uninstalls = [
        events.intercept(kind, _in_turns(handler))
        for kind, handler in (
            (middleware.MIDDLEWARE_LLM_CALL, relay_middleware.nemo_relay_llm_middleware),
            (middleware.MIDDLEWARE_EXECUTE_PYTHON, relay_middleware.nemo_relay_tool_middleware),
        )
    ]
    opened: list[tuple[Any, contextvars.Context]] = []

    def enter() -> None:
        handle = nemo_relay.scope.push(
            "atom", nemo_relay.ScopeType.Agent, handle=parent, metadata=metadata
        )
        # Relay's scope stack lives in a ContextVar: pop in a copy of this context.
        opened.append((handle, contextvars.copy_context()))

    def on_update(update: object) -> None:
        if isinstance(update, ClosedUpdate):
            for uninstall in uninstalls:
                uninstall()
            for handle, context in opened:
                context.run(nemo_relay.scope.pop, handle)
            opened.clear()

    # Set in each turn's context, not the loop's: post-settlement maintenance
    # (Done reflection) runs in a copy of the loop's and stays out of Relay.
    uninstalls.append(events.on("TurnBegan", lambda _event: _session_metadata.set(metadata)))
    session.add_loop_context_hook(enter)
    session.subscribe(on_update)
