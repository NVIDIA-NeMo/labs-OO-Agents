# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-owned execution observations; no client or ACP dependencies."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal
from uuid import uuid4

from nooa.runtime.context_vars import _get_agent_call_stack
from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks
from nooa_atom.session.items import AgentEventUpdate, ExecutionUpdate

logger = logging.getLogger(__name__)

_owner: ContextVar[ExecutionObserver | None] = ContextVar("atom_execution_owner", default=None)


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


@dataclass
class _Node:
    metadata: dict[str, Any]
    parent_session: str | None
    turn: str
    manager: Any = None


class ExecutionObserver:
    """Capture in the emitter context, deliver immutable snapshots on the session loop.

    Causal run ids group linked sessions; separate turn tokens scope cleanup.
    Only active calls retain a nested agent's subscription. Completed methods
    and tools are discarded, so this is a live feed rather than a history store.
    """

    def __init__(self, session: Any) -> None:
        self.session = session
        self._loop = asyncio.get_running_loop()
        self._thread = threading.get_ident()
        self._lock = threading.RLock()
        self._closed = False
        self._pending: deque[tuple[Any, Callable, tuple[Any, ...]]] = deque()
        self._delivering = False
        self._run: ContextVar[tuple[str, str] | None] = ContextVar(
            "atom_execution_run", default=None
        )
        self._python: ContextVar[tuple[str, str | None] | None] = ContextVar(
            "atom_execution_python", default=None
        )
        self._nodes: dict[str, _Node] = {}
        self._python_ids: dict[tuple[str | None, str], str] = {}
        self._terminals: dict[str, str] = {}
        self._managers: dict[Any, tuple[int, Callable[[], None]]] = {}
        self._turn: str | None = None
        caller = _owner.get()
        self._origin = (
            caller.position() if caller and caller.session.id == session.parent_id else None
        )

    def install(self) -> None:
        """Install in the session's fresh loop context, alongside tracing."""
        _owner.set(self)
        set_hooks(compose_hooks(get_hooks(), self))

    def begin_turn(self) -> None:
        """Choose a fresh cleanup token, retaining the creator's causal run for children."""
        self._turn = str(uuid4())
        self._run.set((self._origin[0] if self._origin else str(uuid4()), self._turn))

    def dispatch(self, callback: Callable, *args: Any) -> None:
        """Deliver on the owning loop, draining older thread events before inline ones.

        A main-loop failure must not overtake a queued worker-thread start.
        Reentrant emissions stay synchronous within the currently delivered
        event (e.g. its usage update), preserving the session's normal ordering.
        """
        own_thread = threading.get_ident() == self._thread
        if own_thread and self._delivering:
            callback(*args)
            return
        with self._lock:
            schedule = not self._pending
            self._pending.append((copy_context(), callback, args))
        if own_thread:
            self._drain()
        elif schedule:
            self._loop.call_soon_threadsafe(self._drain)

    def _drain(self) -> None:
        if self._delivering:
            return
        self._delivering = True
        try:
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    context, callback, args = self._pending.popleft()
                try:
                    context.run(callback, *args)
                except (Exception, asyncio.CancelledError):
                    logger.warning("Execution observation delivery failed", exc_info=True)
        finally:
            self._delivering = False

    def _parent(self, call_id: str | None) -> tuple[str | None, str | None]:
        python = self._python.get()
        if python is not None and python[1] == call_id:
            return self.session.id, python[0]
        if call_id:
            return self.session.id, f"nooa-method-{call_id}"
        if self._origin:
            return self._origin[1], self._origin[2]
        return None, None

    @_locked
    def position(self) -> tuple[str, str | None, str | None] | None:
        run = self._run.get()
        if self._closed or run is None:
            return None
        stack = _get_agent_call_stack()
        session, span = self._parent(stack[-1] if stack else None)
        return run[0], session, span

    def _new(self, span: str, kind: str, name: str, parent_call: str | None) -> _Node:
        run = self._run.get()
        assert run is not None
        parent_session, parent = self._parent(parent_call)
        node = _Node(
            {
                "version": 1,
                "runId": run[0],
                "spanId": span,
                "parentSpanId": parent,
                "nodeType": kind,
                "name": name,
                "startedAtMs": time.time_ns() / 1_000_000,
            },
            parent_session,
            run[1],
        )
        self._nodes[span] = node
        return node

    @staticmethod
    def _packet(node: _Node) -> dict[str, Any]:
        return {**node.metadata, "_parentSessionId": node.parent_session}

    def _method_update(
        self,
        node: _Node,
        status: Literal["in_progress", "completed", "failed"],
        error: str | None = None,
    ) -> None:
        self.dispatch(
            self.session._emit,
            ExecutionUpdate(
                session_id=self.session.id, node=self._packet(node), status=status, error=error
            ),
        )

    def _subscribe(self, agent: Any) -> Any:
        manager = agent.event_manager
        if manager is self.session._agent.event_manager:
            return None
        known = self._managers.get(manager)
        if known:
            self._managers[manager] = (known[0] + 1, known[1])
        else:

            def forward(event):
                if _owner.get() is self and not self._closed:
                    packet = self.capture(event)
                    self.dispatch(
                        self.session._emit,
                        AgentEventUpdate(session_id=self.session.id, event=event, execution=packet),
                    )

            self._managers[manager] = (1, manager.on("*", forward))
        return manager

    def _release(self, manager: Any) -> None:
        if manager is None or manager not in self._managers:
            return
        count, unsubscribe = self._managers[manager]
        if count == 1:
            del self._managers[manager]
            unsubscribe()
        else:
            self._managers[manager] = (count - 1, unsubscribe)

    @_locked
    def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id, **extra):
        if self._closed or self._run.get() is None or _owner.get() is not self:
            return None
        node = self._new(
            f"nooa-method-{call_id}",
            "method",
            f"{type(agent).__name__}.{method_name}",
            parent_call_id,
        )
        node.metadata["agent"] = type(agent).__name__
        node.manager = self._subscribe(agent)
        self._method_update(node, "in_progress")
        return node.metadata["spanId"]

    @_locked
    def after_agent_call(self, agent, method_name, result, exception, context, **kwargs):
        node = self._nodes.pop(context, None)
        if node is None:
            return
        self._release(node.manager)
        node.metadata["endedAtMs"] = time.time_ns() / 1_000_000
        error = None
        if exception is not None:
            if isinstance(exception, asyncio.CancelledError):
                error = "Cancelled."
            else:
                error = type(exception).__name__
                try:
                    error += f": {str(exception)[:2000]}"
                except (Exception, asyncio.CancelledError):
                    pass  # Error rendering cannot leave the method card open.
        self._method_update(node, "failed" if exception is not None else "completed", error)

    @contextmanager
    def activate_agent_call(self, context: str | None) -> Iterator[None]:
        with self._lock:
            node = self._nodes.get(context) if context is not None else None
            run = (node.metadata["runId"], node.turn) if node else None
        token = self._run.set(run) if run else None
        owner = _owner.set(self)
        try:
            yield
        finally:
            _owner.reset(owner)
            if token is not None:
                self._run.reset(token)

    def _python_key(self, provider_id: str) -> tuple[str | None, str]:
        stack = _get_agent_call_stack()
        return (stack[-1] if stack else None, provider_id)

    @_locked
    def capture(self, event: Any) -> dict[str, Any] | None:
        """Correlate before queuing: code hooks on worker threads need the ids immediately."""
        if self._closed or self._run.get() is None or _owner.get() is not self:
            return None
        kind = event.event_type
        stack = _get_agent_call_stack()
        caller = stack[-1] if stack else None
        node = None
        finish = False
        if kind == "ToolCallEvent" and event.name in {"python_cell", "execute_python"}:
            if event.metadata.get("prefill") or event.metadata.get("synthetic"):
                return None
            span = f"nooa-python-{event.id}"
            self._python_ids[self._python_key(event.tool_call_id)] = span
            node = self._new(span, "python", "Running Python", caller)
        elif kind == "PythonOutput":
            span = self._python_ids.pop(self._python_key(event.tool_call_id), None)
            node = self._nodes.pop(span, None) if span is not None else None
            finish = True
        elif kind == "FileEdit":
            node = self._new(f"nooa-file-{event.id}", "file", "File edit", caller)
            self._nodes.pop(node.metadata["spanId"])
            finish = True
        elif kind == "TerminalCommandStarted":
            span = f"nooa-terminal-{event.id}"
            self._terminals[event.command_id] = span
            node = self._new(span, "terminal", "Terminal command", caller)
        elif kind in {"TerminalCommandOutput", "TerminalCommandFinished"}:
            span = self._terminals.get(event.command_id)
            node = self._nodes.get(span) if span is not None else None
            if kind == "TerminalCommandFinished":
                self._terminals.pop(event.command_id, None)
                if span is not None:
                    self._nodes.pop(span, None)
                finish = True
        if node is None:
            return None
        if finish:
            node.metadata["endedAtMs"] = time.time_ns() / 1_000_000
        return self._packet(node)

    def before_code_execution(self, agent, code, execution_id, generation_id=None, **kwargs):
        with self._lock:
            span = self._python_ids.get(self._python_key(kwargs.get("tool_call_id", "")))
        if span is None:
            return None
        stack = _get_agent_call_stack()
        return self._python.set((span, stack[-1] if stack else None))

    def after_code_execution(self, agent, code, result, exception, context, execution_id, **kwargs):
        if context is not None:
            self._python.reset(context)

    @_locked
    def end_turn(self, failed: bool) -> None:
        if failed:
            self._finish(self._turn, "Turn interrupted.")

    def _finish(self, turn: str | None, reason: str) -> None:
        for span, node in tuple(self._nodes.items()):
            if turn is not None and node.turn != turn:
                continue
            self._nodes.pop(span, None)
            if node.metadata["nodeType"] == "method":
                self._release(node.manager)
                node.metadata["endedAtMs"] = time.time_ns() / 1_000_000
                self._method_update(node, "failed", reason)
        live = self._nodes.keys()
        self._python_ids = {k: v for k, v in self._python_ids.items() if v in live}
        self._terminals = {k: v for k, v in self._terminals.items() if v in live}

    @_locked
    def close(self) -> None:
        self._closed = True
        self._finish(None, "Session closed.")
        for _, unsubscribe in self._managers.values():
            unsubscribe()
        self._managers.clear()

    # Other instrumentation stages do not add visible nodes.
    def before_generation(self, *args: Any, **kwargs: Any):
        pass

    def after_generation(self, *args: Any, **kwargs: Any):
        pass

    def before_method_invocation(self, *args: Any, **kwargs: Any):
        pass

    def after_method_invocation(self, *args: Any, **kwargs: Any):
        pass

    def before_tool_execution(self, *args: Any, **kwargs: Any):
        pass

    def after_tool_execution(self, *args: Any, **kwargs: Any):
        pass

    def on_messages_built(self, *args: Any, **kwargs: Any):
        pass
