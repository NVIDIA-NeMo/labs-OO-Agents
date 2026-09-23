# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral foreground-turn dispatch for interactive NOOA agents."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from nooa_coder.interactive.local_agent import LocalAgentRunner
from nooa_coder.interactive_agent import RespondResult


class InteractiveSessionDispatcher:
    """Awaitable host facade over the native host's persistent agent runner.

    This class owns foreground request admission, not another dispatch loop.
    Both ACP and headless execution use exactly the native runtime's queue,
    observation, background wake and lifecycle implementation.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.runtime = LocalAgentRunner(agent, emit_text=lambda text: None, agent_id=str(id(agent)))
        self._active_task: asyncio.Task[Any] | None = None
        self._cancel_requested = False
        self._cancelling = False
        self._closed = False

    @property
    def active(self) -> bool:
        return self._cancelling or (self._active_task is not None and not self._active_task.done())

    async def submit(self, text: str) -> RespondResult | None:
        """Submit one user prompt and wait through any background-job notifications."""
        self._ensure_idle()
        return await self._run_active(self.runtime.submit_and_wait(text))

    async def invoke_slash(
        self,
        commands: Any,
        name: str,
        raw_args: str,
    ) -> tuple[Any, RespondResult | None] | None:
        """Invoke and, when requested, dispatch a slash command as one turn."""

        async def _invoke() -> tuple[Any, RespondResult | None]:
            result = await commands.invoke(name, raw_args)
            if not result.output_to_agent:
                return result, None
            command = commands.get(name)
            if command is not None and command._method is None:
                # Markdown skills prepare the next user turn in both hosts.
                # The runner records that input once and requests its title.
                return result, await self.runtime.submit_and_wait(result.text or "")
            return result, await self.runtime.submit_slash_and_wait(result)

        return await self._run_active(_invoke())

    def _ensure_idle(self) -> None:
        if self._closed:
            raise RuntimeError("Interactive session is closed")
        if self.active:
            raise RuntimeError("A prompt is already running")

    async def _run_active(self, operation: Coroutine[Any, Any, Any]) -> Any:
        if self.active or self._closed:
            operation.close()
            self._ensure_idle()

        self._cancel_requested = False
        task = asyncio.create_task(operation, name="nooa-interactive-dispatch")
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

    async def cancel(self) -> bool:
        """Cancel the foreground turn and background jobs without closing the agent."""
        task = self._active_task
        foreground = task is not None and not task.done()
        had_work = foreground or not self.runtime.is_quiescent
        self._cancelling = True
        self._cancel_requested = foreground
        try:
            if foreground:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            await self.runtime.cancel_work()
            return had_work
        finally:
            self._cancelling = False

    async def close(self) -> None:
        """Cancel active dispatch and close the owned agent."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.cancel()
        finally:
            try:
                await self.runtime.shutdown()
            finally:
                await self.agent.close()
