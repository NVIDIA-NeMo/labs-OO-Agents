# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional ACP execution metadata from NOOA's native instrumentation hooks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any
from uuid import uuid4

from acp import start_tool_call, text_block, tool_content, update_tool_call
from acp.schema import ToolCallProgress, ToolCallStart

from nooa.runtime.context_vars import _get_agent_call_stack
from nooa.runtime.hooks import hooks_scope

EXECUTION_META_KEY = "nooa.dev/execution"


class ACPExecutionTree:
    """Observe real method lifecycles and correlate existing ACP activity cards.

    Hooks are installed only in a foreground dispatch's context. Child tasks
    inherit them, while another session gets its own run and observer. Method
    arguments and return values are deliberately absent from the extension.
    """

    def __init__(
        self,
        publish: Callable[[Any], None],
        observe_agent: Callable[[Any], None],
    ) -> None:
        self._publish = publish
        self._observe_agent = observe_agent
        self._run: ContextVar[str | None] = ContextVar("nooa_acp_execution_run", default=None)
        self._python: ContextVar[tuple[str, str | None] | None] = ContextVar(
            "nooa_acp_execution_python", default=None
        )
        self._nodes: dict[str, dict[str, Any]] = {}
        self._python_ids: dict[tuple[str | None, str], str] = {}
        self._closed = False

    @property
    def active(self) -> bool:
        return not self._closed and self._run.get() is not None

    @contextmanager
    def turn(self) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("ACP execution tree is closed")
        token = self._run.set(str(uuid4()))
        try:
            with hooks_scope(self):
                yield
        finally:
            self._run.reset(token)

    @contextmanager
    def activate_agent_call(self, context: str | None) -> Iterator[None]:
        # Generator bodies resume in slices and can cross task/turn boundaries.
        # Their descendants still belong to the run that owns the generator.
        node = self._nodes.get(context) if context is not None else None
        token = self._run.set(node["runId"]) if node is not None else None
        try:
            yield
        finally:
            if token is not None:
                self._run.reset(token)

    def _parent(self, call_id: str | None) -> str | None:
        python = self._python.get()
        if python is not None and python[1] == call_id:
            return python[0]
        return f"nooa-method-{call_id}" if call_id else None

    @staticmethod
    def _python_key(tool_call_id: str) -> tuple[str | None, str]:
        stack = _get_agent_call_stack()
        return (stack[-1] if stack else None, tool_call_id)

    def start_python(self, tool_call_id: str, event_id: str) -> str:
        # Provider tool ids are not globally unique. Two child agents may both
        # receive "call_0", including while their executions overlap.
        span_id = f"nooa-python-{event_id}"
        self._python_ids[self._python_key(tool_call_id)] = span_id
        return span_id

    def python_id(self, tool_call_id: str) -> str | None:
        return self._python_ids.get(self._python_key(tool_call_id))

    def decorate(self, update: ToolCallStart | ToolCallProgress) -> None:
        """Attach a complete metadata value to every update, not a partial patch."""
        node = self._nodes.get(update.tool_call_id)
        if node is None and isinstance(update, ToolCallStart) and self.active:
            stack = _get_agent_call_stack()
            node = {
                "version": 1,
                "runId": self._run.get(),
                "spanId": update.tool_call_id,
                "parentSpanId": self._parent(stack[-1] if stack else None),
                "nodeType": {"edit": "file", "execute": "terminal"}.get(
                    update.kind or "other", "python"
                ),
                "name": update.title,
                "startedAtMs": time.time_ns() / 1_000_000,
            }
            self._nodes[update.tool_call_id] = node
        if node is None:
            return
        if update.status in {"completed", "failed"}:
            node["endedAtMs"] = time.time_ns() / 1_000_000
            self._nodes.pop(update.tool_call_id, None)
            if node["nodeType"] == "python":
                for key, span_id in tuple(self._python_ids.items()):
                    if span_id == update.tool_call_id:
                        del self._python_ids[key]
        update.field_meta = {**(update.field_meta or {}), EXECUTION_META_KEY: dict(node)}

    def before_agent_call(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call_id: str,
        parent_call_id: str | None,
        **extra_kwargs: Any,
    ) -> str | None:
        if not self.active:
            return None
        self._observe_agent(agent)
        span_id = f"nooa-method-{call_id}"
        name = f"{type(agent).__name__}.{method_name}"
        self._nodes[span_id] = {
            "version": 1,
            "runId": self._run.get(),
            "spanId": span_id,
            "parentSpanId": self._parent(parent_call_id),
            "nodeType": "method",
            "name": name,
            "agent": type(agent).__name__,
            "startedAtMs": time.time_ns() / 1_000_000,
        }
        self._publish(start_tool_call(span_id, name, kind="other", status="in_progress"))
        return span_id

    def after_agent_call(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: BaseException | None,
        context: str | None,
        **kwargs: Any,
    ) -> None:
        if context is None or context not in self._nodes:
            return
        content = None
        if exception is not None:
            message = (
                "Cancelled by user."
                if isinstance(exception, asyncio.CancelledError)
                else f"{type(exception).__name__}: {str(exception)[:2_000]}"
            )
            content = [tool_content(text_block(message))]
        self._publish(
            update_tool_call(
                context,
                status="failed" if exception is not None else "completed",
                content=content,
            )
        )

    def before_code_execution(
        self,
        agent: Any,
        code: str,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> Token[tuple[str, str | None] | None] | None:
        tool_call_id = self.python_id(kwargs.get("tool_call_id", ""))
        # Only actual visible Python cards own children; prefill executions
        # have no card and must never become invisible parents.
        if tool_call_id is None or tool_call_id not in self._nodes:
            return None
        stack = _get_agent_call_stack()
        return self._python.set((tool_call_id, stack[-1] if stack else None))

    def after_code_execution(
        self,
        agent: Any,
        code: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        if context is not None:
            self._python.reset(context)

    def fail_open_methods(self, reason: str) -> None:
        for span_id, node in tuple(self._nodes.items()):
            if node["nodeType"] == "method":
                self._publish(
                    update_tool_call(
                        span_id, status="failed", content=[tool_content(text_block(reason))]
                    )
                )

    def close(self) -> None:
        self._closed = True
        self._nodes.clear()
        self._python_ids.clear()

    # The instrumentation protocol also observes generation internals. Those
    # do not add separate ACP cards: the method and Python nodes represent them.
    def before_generation(self, *args: Any, **kwargs: Any) -> None:
        return None

    def after_generation(self, *args: Any, **kwargs: Any) -> None:
        return None

    def before_method_invocation(self, *args: Any, **kwargs: Any) -> None:
        return None

    def after_method_invocation(self, *args: Any, **kwargs: Any) -> None:
        return None

    def before_tool_execution(self, *args: Any, **kwargs: Any) -> None:
        return None

    def after_tool_execution(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_messages_built(self, *args: Any, **kwargs: Any) -> None:
        return None
