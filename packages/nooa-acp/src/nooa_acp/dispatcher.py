# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side dispatcher for a NOOA interactive agent."""

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from nooa_cli.coding import CodingAgent, CodingSlashCommandRegistry
from pydantic import BaseModel

from nooa.interactive import Done, NeedInput, Waiting
from nooa.slash_dispatch import SlashCommandResult

TurnResult = Done | NeedInput | Waiting


class InteractiveSessionDispatcher:
    def __init__(self, agent: CodingAgent) -> None:
        self.agent = agent
        self._active_task: asyncio.Task[Any] | None = None
        self._cancel_requested = False
        self._cancelling = False

    @property
    def active(self) -> bool:
        return self._cancelling or (self._active_task is not None and not self._active_task.done())

    async def submit(self, text: str) -> TurnResult | None:
        self._ensure_idle()
        self.agent.queue_manager.get_channel("user_messages").put(text)
        return await self._run_active(self._dispatch())

    async def invoke_slash(
        self,
        commands: CodingSlashCommandRegistry,
        name: str,
        raw_args: str,
    ) -> tuple[SlashCommandResult, TurnResult | None] | None:
        """Invoke and, when requested, dispatch a slash command as one cancellable turn."""

        async def _invoke() -> tuple[SlashCommandResult, TurnResult | None]:
            result = await commands.invoke(name, raw_args)
            if not result.output_to_agent:
                return result, None
            self.agent.queue_manager.get_channel("slash_commands").put(result)
            return result, await self._dispatch()

        return await self._run_active(_invoke())

    def _ensure_idle(self) -> None:
        if self.active:
            raise RuntimeError("A prompt is already running")

    async def _run_active(self, operation: Coroutine[Any, Any, Any]) -> Any:
        if self.active:
            operation.close()
            raise RuntimeError("A prompt is already running")

        self._cancel_requested = False
        task = asyncio.create_task(operation, name="nooa-acp-dispatch")
        self._active_task = task
        try:
            return await task
        except asyncio.CancelledError:
            if self._cancel_requested:
                return None
            raise
        finally:
            if self._active_task is task:
                self._active_task = None

    async def _dispatch(self) -> TurnResult:
        while True:
            wins = await self.agent.queue_manager.race()
            notification: dict[str, list[Any]] = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            for name, channel in self.agent.queue_manager.channels().items():
                if drained := channel.drain():
                    notification.setdefault(name, []).extend(drained)

            result = await self.agent.handle(notification)
            self._show(result)
            # Keep the prompt open while the turn waits on a job or queue.
            if isinstance(result, Waiting):
                continue
            return result

    def _show(self, result: TurnResult) -> None:
        """Send the person the text a typed result carries, as an agent message.

        ``Done.message`` is the reply, ``Waiting.message`` the line shown while
        waiting, and a ``NeedInput`` shows its question, its reason, and its
        choices or the fields of its ``answer_type`` with their types and
        descriptions. This server has no forms, so a typed answer still
        arrives as the person's text reply. The agent is told not to send
        these itself, so the host must.
        """
        if isinstance(result, NeedInput):
            parts = [result.question]
            if result.reason:
                parts.append(result.reason)
            if result.options:
                parts.append("\n".join(f"- {option}" for option in result.options))
            if result.answer_type is not None:
                parts.append("Reply with these fields:\n" + _describe_fields(result.answer_type))
            text: str | None = "\n\n".join(parts)
        else:
            text = result.message
        if text:
            self.agent.message(text)

    async def cancel(self) -> bool:
        """Cancel the foreground turn and background jobs without closing the session."""
        task = self._active_task
        if task is None or task.done():
            return False

        self._cancelling = True
        self._cancel_requested = True
        try:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            for channel_name in ("user_messages", "slash_commands"):
                self.agent.queue_manager.get_channel(channel_name).flush()
            await self.agent.queue_manager.shutdown()
            return True
        finally:
            self._cancelling = False

    async def close(self) -> None:
        await self.cancel()
        await self.agent.close()


def _describe_fields(model: type[BaseModel]) -> str:
    """One ``- name (type): description`` line per field of ``model``."""
    lines = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        type_name = annotation.__name__ if isinstance(annotation, type) else str(annotation)
        line = f"- {name} ({type_name})"
        if field.description:
            line += f": {field.description}"
        lines.append(line)
    return "\n".join(lines)
