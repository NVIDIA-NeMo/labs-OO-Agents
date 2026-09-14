# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection and turn-status policy shared by native and protocol hosts."""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from nooa_cli.interactive.runtime import AgentRuntime

logger = logging.getLogger(__name__)


class LocalTurnPolicy:
    """Apply agent behavior independently of terminal or protocol presentation."""

    def __init__(
        self,
        agent: Any,
        runtime: AgentRuntime,
        *,
        emit_output: Callable[[Any], Awaitable[None]],
        invalidate: Callable[[], None] | None = None,
    ) -> None:
        self._agent = agent
        self._runtime = runtime
        self._emit_output = emit_output
        self._invalidate = invalidate
        self._closed = False
        self._state_lock = threading.Lock()

    async def before_handle(self, agent: Any) -> None:
        if not self._is_active():
            return
        reflection = getattr(agent, "_reflection_runner", None)
        if reflection is not None:
            await reflection.interrupt()

    async def after_handle(self, agent: Any, result: Any) -> None:
        """Schedule reflection and report the completed turn."""
        if not self._is_active():
            return
        explanation = getattr(result, "explanation", "")
        logger.debug("[DISPATCHER] handle() returned kind=%r", result.kind)
        self._schedule_reflection(agent)
        if explanation and self._is_active():
            from .policy_events import TurnStatus as StopReasonOutput

            await self._emit_output(StopReasonOutput(result.kind, explanation))

    async def shutdown(self) -> None:
        """Atomically close and quiesce policy producers before host teardown."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        await self.interrupt_reflection(teardown=True)

    async def interrupt_reflection(self, *, teardown: bool = False) -> None:
        runner = self._reflection_runner()
        if runner is None:
            return

        async def _stop() -> None:
            await runner.interrupt()
            if teardown:
                runner.teardown()

        await self._runtime.run_async(_stop)

    def _is_active(self) -> bool:
        with self._state_lock:
            return not self._closed

    def _reflection_runner(self) -> Any | None:
        runner = getattr(self._agent, "_reflection_runner", None)
        if runner is not None:
            runner.invalidate = self._invalidate
        return runner

    def _schedule_reflection(self, agent: Any) -> None:
        if not self._is_active():
            return
        runner = getattr(agent, "_reflection_runner", None)
        if runner is not None:
            runner.invalidate = self._invalidate
            runner.on_response_done()
