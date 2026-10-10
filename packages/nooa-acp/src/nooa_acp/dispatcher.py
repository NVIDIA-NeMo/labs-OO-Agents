# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side dispatcher for a NOOA interactive agent."""

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from nooa_cli.coding import CodingAgent, CodingSlashCommandRegistry

from nooa.interactive import Done, InputRequest, NeedInput, NeedInputForm, Waiting
from nooa.slash_dispatch import SlashCommandResult

TurnResult = Done | InputRequest | Waiting


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

        Done/Waiting carry optional messages. Questions show suggestions; explicit
        forms show response/validation guidance and field descriptions. This legacy
        text-only host does not validate text as accepted form content.
        """
        if isinstance(result, (NeedInput, NeedInputForm)):
            parts = [result.heading if isinstance(result, NeedInputForm) else result.question]
            if result.reason:
                parts.append(result.reason)
            if isinstance(result, NeedInput) and result.options:
                parts.append("\n".join(f"- {option}" for option in result.options))
            if isinstance(result, NeedInputForm):
                parts.append(
                    "Explicit form requested; this host has no dialog. Text is unvalidated.\n"
                    "Text replies require agent interpretation and targeted follow-up. Structured "
                    "FormResponse submission requires a capable session host."
                )
                parts.append(
                    "\n".join(
                        f"- {q.id}: {q.label}"
                        + (f" — {q.help}" if q.help else "")
                        + (
                            " Choices: " + ", ".join(f"{c.title} ({c.value})" for c in q.choices)
                            if q.kind != "text"
                            else ""
                        )
                        for q in result.questions
                    )
                )
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
