# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate observational NOOA events into ACP session updates."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from acp import (
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import Cost, ToolCallLocation, UsageUpdate
from nooa_cli.coding import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)

from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from nooa.events import LLMComplete, PythonOutput
from nooa.interactive import AgentMessage
from nooa_acp.coding_agent import CodingInteractiveAgent

# ACP owns stdout for JSON-RPC; diagnostics belong on stderr, which is where
# the logging default sends them.
logger = logging.getLogger(__name__)

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
        self._terminal_output: dict[str, str] = {}
        self._cost_usd = 0.0
        self._unsubscribers: list[Callable[[], None]] = [
            agent.event_manager.on("AgentMessage", self._on_agent_message),
            agent.event_manager.on("ToolCallEvent", self._on_tool_call),
            agent.event_manager.on("PythonOutput", self._on_python_output),
            agent.event_manager.on("LLMComplete", self._on_llm_complete),
            agent.event_manager.on("FileEdit", self._on_file_edit),
            agent.event_manager.on("TerminalCommandStarted", self._on_terminal_started),
            agent.event_manager.on("TerminalCommandOutput", self._on_terminal_output),
            agent.event_manager.on("TerminalCommandFinished", self._on_terminal_finished),
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

    def _on_file_edit(self, event: EventBase) -> None:
        if not isinstance(event, FileEdit):
            return
        tool_call_id = f"file-edit-{uuid4()}"
        path = event.path
        title = f"{'Created' if event.operation == 'create' else 'Edited'} {Path(path).name}"
        if event.content_complete:
            content = [tool_diff_content(path, event.new_text, event.old_text)]
        else:
            content = [tool_content(text_block(event.diff or "File content was truncated."))]
        line = max(0, event.start_line - 1) if event.start_line is not None else None
        self._enqueue(
            start_tool_call(
                tool_call_id,
                title,
                kind="edit",
                status="completed",
                content=content,
                locations=[ToolCallLocation(path=path, line=line)],
                raw_input={"path": path, "operation": event.operation},
            )
        )

    def _on_terminal_started(self, event: EventBase) -> None:
        if not isinstance(event, TerminalCommandStarted):
            return
        self._open_tools.add(event.command_id)
        self._terminal_output[event.command_id] = ""
        self._enqueue(
            start_tool_call(
                event.command_id,
                f"$ {event.command}",
                kind="execute",
                status="in_progress",
                raw_input={
                    "command": event.command,
                    "working_directory": event.working_directory,
                },
            )
        )

    def _on_terminal_output(self, event: EventBase) -> None:
        if not isinstance(event, TerminalCommandOutput) or event.command_id not in self._open_tools:
            return
        chunk = event.stdout
        if event.stderr:
            chunk += ("\n" if chunk and not chunk.endswith("\n") else "") + event.stderr
        output = self._terminal_output.get(event.command_id, "") + chunk
        self._terminal_output[event.command_id] = output
        self._enqueue(
            update_tool_call(
                event.command_id,
                status="in_progress",
                content=[tool_content(text_block(output))],
            )
        )

    def _on_terminal_finished(self, event: EventBase) -> None:
        if not isinstance(event, TerminalCommandFinished):
            return
        self._open_tools.discard(event.command_id)
        output = self._terminal_output.pop(event.command_id, "")
        # ACP has no cancelled status, so a stopped command is still "failed" —
        # but it must read as the user's own action, not as a crash.
        reason = "Cancelled by user." if event.cancelled else event.error
        if reason:
            output += ("\n" if output and not output.endswith("\n") else "") + reason
        failed = (
            event.timed_out
            or event.cancelled
            or bool(event.error)
            or (event.exit_code is not None and event.exit_code != 0)
        )
        self._enqueue(
            update_tool_call(
                event.command_id,
                status="failed" if failed else "completed",
                content=[tool_content(text_block(output or "Completed."))],
                raw_output={
                    "exit_code": event.exit_code,
                    "timed_out": event.timed_out,
                    "output_truncated": event.output_truncated,
                },
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
                    # Hand the failure to the turn that is flushing, then clear
                    # it. Latching it would mean every later turn ran the model
                    # and edited files while the client saw nothing and got a
                    # stale exception it could not act on.
                    error, self._error = self._error, None
                    if not item.done():
                        if error is None:
                            item.set_result(None)
                        else:
                            item.set_exception(error)
                    continue
                if self._error is None:
                    try:
                        await self.client.session_update(self.session_id, item)
                    except Exception as exc:
                        # Skip the rest of this turn's updates rather than
                        # hammering a transport that just failed; the next
                        # flush reports the error and resets.
                        self._error = exc
                        logger.warning(
                            "ACP session %s dropped an update after a failed send",
                            self.session_id,
                            exc_info=True,
                        )
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
        self._terminal_output.clear()

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
