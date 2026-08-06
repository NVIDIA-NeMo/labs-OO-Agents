# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side dispatcher for a NOOA interactive agent."""

import asyncio
from contextlib import suppress
from typing import Any, cast

from nooa.interactive import RespondReason, RespondResult
from nooa_acp.coding_agent import CodingInteractiveAgent


class InteractiveSessionDispatcher:
    def __init__(self, agent: CodingInteractiveAgent) -> None:
        self.agent = agent
        self._active_task: asyncio.Task[RespondResult] | None = None
        self._cancel_requested = False
        self._cancelling = False

    @property
    def active(self) -> bool:
        return self._cancelling or (self._active_task is not None and not self._active_task.done())

    async def submit(self, text: str) -> RespondResult | None:
        if self.active:
            raise RuntimeError("A prompt is already running")

        self._cancel_requested = False
        self.agent.queue_manager.get_channel("user_messages").put(text)
        task = asyncio.create_task(self._dispatch(), name="nooa-acp-dispatch")
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

    async def _dispatch(self) -> RespondResult:
        while True:
            wins = await self.agent.queue_manager.race()
            notification: dict[str, list[Any]] = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            for name, channel in self.agent.queue_manager.channels().items():
                if drained := channel.drain():
                    notification.setdefault(name, []).extend(drained)

            result = cast(RespondResult, await self.agent.handle(notification))
            if result.kind is not RespondReason.WAIT:
                return result

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
            self.agent.queue_manager.get_channel("user_messages").flush()
            await self.agent.queue_manager.shutdown()
            return True
        finally:
            self._cancelling = False

    async def close(self) -> None:
        await self.cancel()
        await self.agent.close()
