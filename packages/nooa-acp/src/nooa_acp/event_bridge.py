# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate observational NOOA events into ACP session updates."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from acp import start_tool_call, text_block, tool_content, update_agent_message, update_tool_call
from acp.interfaces import Client
from acp.schema import Cost, UsageUpdate

from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from nooa.events import LLMComplete, PythonOutput
from nooa.interactive import AgentMessage
from nooa_acp.coding_agent import CodingInteractiveAgent

_STOP = object()


class ACPEventBridge:
    def __init__(self, agent: CodingInteractiveAgent, client: Client, session_id: str) -> None:
        self.agent = agent
        self.client = client
        self.session_id = session_id
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._error: Exception | None = None
        self._closed = False
        self._open_tools: set[str] = set()
        self._cost_usd = 0.0
        self._unsubscribers: list[Callable[[], None]] = [
            agent.event_manager.on("AgentMessage", self._on_agent_message),
            agent.event_manager.on("ToolCallEvent", self._on_tool_call),
            agent.event_manager.on("PythonOutput", self._on_python_output),
            agent.event_manager.on("LLMComplete", self._on_llm_complete),
        ]
        self._pump_task = asyncio.create_task(self._pump(), name="nooa-acp-events")

    def _enqueue(self, update: Any) -> None:
        if not self._closed:
            self._queue.put_nowait(update)

    def _on_agent_message(self, event: EventBase) -> None:
        if not isinstance(event, AgentMessage):
            return
        self._enqueue(update_agent_message(text_block(event.content)))

    def _on_tool_call(self, event: EventBase) -> None:
        if (
            not isinstance(event, ToolCallEvent)
            or event.name != "execute_python"
            or event.metadata.get("prefill") is True
        ):
            return
        self._open_tools.add(event.tool_call_id)
        self._enqueue(
            start_tool_call(
                event.tool_call_id,
                "Running Python",
                kind="execute",
                status="in_progress",
                raw_input=event.arguments,
            )
        )

    def _on_python_output(self, event: EventBase) -> None:
        if not isinstance(event, PythonOutput) or event.tool_call_id not in self._open_tools:
            return
        self._open_tools.discard(event.tool_call_id)
        parts = [
            part.rstrip() for part in (event.stdout, event.stderr, event.error) if part.strip()
        ]
        output = "\n".join(parts) or "Completed."
        status: Literal["failed", "completed"] = (
            "failed" if event.execution_status is ResultStatus.ERROR else "completed"
        )
        self._enqueue(
            update_tool_call(
                event.tool_call_id,
                status=status,
                content=[tool_content(text_block(output))],
            )
        )

    def _on_llm_complete(self, event: EventBase) -> None:
        if not isinstance(event, LLMComplete):
            return
        self._cost_usd += event.cost_usd
        context_window = getattr(self.agent.llm, "context_window", None)
        if context_window is None:
            return
        self._enqueue(
            UsageUpdate(
                session_update="usage_update",
                used=event.prompt_tokens,
                size=max(context_window, event.prompt_tokens),
                cost=Cost(amount=self._cost_usd, currency="USD"),
            )
        )

    async def _pump(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, asyncio.Future):
                    if not item.done():
                        if self._error is None:
                            item.set_result(None)
                        else:
                            item.set_exception(self._error)
                    continue
                if self._error is None:
                    try:
                        await self.client.session_update(self.session_id, item)
                    except Exception as exc:
                        self._error = exc
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        if self._closed:
            return
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(future)
        await asyncio.shield(future)

    async def fail_open_tools(self, reason: str) -> None:
        for tool_call_id in tuple(self._open_tools):
            self._enqueue(
                update_tool_call(
                    tool_call_id,
                    status="failed",
                    content=[tool_content(text_block(reason))],
                )
            )
        self._open_tools.clear()

    async def close(self) -> None:
        if self._closed:
            return
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        with suppress(Exception):
            await self.flush()
        self._closed = True
        self._queue.put_nowait(_STOP)
        await self._pump_task
