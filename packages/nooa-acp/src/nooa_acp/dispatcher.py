# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side dispatcher for a NOOA interactive agent."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from nooa_cli.coding import CodingAgent, CodingSlashCommandRegistry

from nooa.interactive import Done, NeedInput, Waiting
from nooa.slash_dispatch import SlashCommandResult
from nooa_acp._runtime import _acp_capture, _acp_join, _acp_task, _ACPOutcome

TurnResult = Done | NeedInput | Waiting


class InteractiveSessionDispatcher:
    def __init__(self, agent: CodingAgent) -> None:
        self.agent = agent
        self._active_task: asyncio.Task[_ACPOutcome] | None = None
        self._close_task: asyncio.Task[_ACPOutcome] | None = None
        self._cancel_requested = False
        self._cancelling = False
        self._close_completed: set[str] = set()
        self._close_failure: _ACPOutcome | None = None

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
        task = _acp_task(operation, name="nooa-acp-dispatch")
        self._active_task = task
        try:
            outcome = await _acp_join(task)
            if isinstance(outcome.failure, asyncio.CancelledError) and self._cancel_requested:
                return None
            return outcome.unwrap()
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
        waiting, and a ``NeedInput`` shows its question and its reason; the
        person's next message is the answer, as plain text. The agent is told
        not to send these itself, so the host must.
        """
        if isinstance(result, NeedInput):
            text: str | None = "\n\n".join(p for p in (result.question, result.reason) if p)
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
        first = _ACPOutcome()
        try:
            try:
                task.cancel()
            except BaseException as error:
                first = _ACPOutcome(failure=error, traceback=error.__traceback__)
            outcome = await _acp_join(task)
            if not isinstance(outcome.failure, asyncio.CancelledError) and first.failure is None:
                first = outcome
            for channel_name in ("user_messages", "slash_commands"):

                async def flush(name=channel_name):
                    self.agent.queue_manager.get_channel(name).flush()

                result = await _acp_capture(flush())
                if first.failure is None:
                    first = result

            async def shutdown():
                await self.agent.queue_manager.shutdown()

            result = await _acp_capture(shutdown())
            if first.failure is None:
                first = result
            first.unwrap()
            return True
        finally:
            self._cancelling = False

    async def close(self) -> None:
        if self._close_task is None or (
            self._close_task.done() and self._close_completed != {"cancel", "agent"}
        ):
            self._close_task = _acp_task(self._close_resources(), name="nooa-acp-dispatch-close")
        (await _acp_join(self._close_task)).unwrap()

    async def _close_resources(self) -> None:
        async def close_agent():
            await self.agent.close()

        for label, operation in (("cancel", self.cancel), ("agent", close_agent)):
            if label in self._close_completed:
                continue

            async def invoke(callback=operation):
                await callback()

            result = await _acp_capture(invoke())
            if result.failure is None:
                self._close_completed.add(label)
            elif self._close_failure is None:
                self._close_failure = result
        if self._close_failure is not None:
            self._close_failure.unwrap()
