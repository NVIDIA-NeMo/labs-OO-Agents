# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Callable, Any

from nooa.acs.schema import ACSDecision, ACSHook

logger = logging.getLogger(__name__)

class ACSHandler(ABC):
    """
    Base interface for an ACS Guardian.
    """
    
    @abstractmethod
    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        """
        Evaluate a hook and return a governance decision.
        """
        pass


class DefaultLocalHandler(ACSHandler):
    """
    NOOA-as-Guardian: Always allows execution. 
    This represents the default unrestricted local development mode.
    """
    
    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        return ACSDecision(status="allow")


class AuditOnlyHandler(ACSHandler):
    """
    Audit-only Guardian: Records hooks to a provided sink function,
    but always allows execution without blocking.
    """

    MAX_PENDING_DELIVERIES = 100
    
    def __init__(self, sink_func: Callable[[ACSHook], Any]):
        self.sink_func = sink_func
        self._delivery_tasks: set[asyncio.Task[None]] = set()
        self._accepting_deliveries = True

    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        if not self._accepting_deliveries:
            logger.warning("Skipping audit delivery after handler shutdown")
            return ACSDecision(status="allow")
        if len(self._delivery_tasks) >= self.MAX_PENDING_DELIVERIES:
            logger.warning("Skipping audit delivery: pending delivery limit reached")
            return ACSDecision(status="allow")

        async def deliver() -> None:
            try:
                if inspect.iscoroutinefunction(self.sink_func):
                    result = self.sink_func(hook)
                else:
                    result = await asyncio.to_thread(self.sink_func, hook)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Audit sink delivery failed")

        task = asyncio.create_task(deliver())
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)
        return ACSDecision(status="allow")

    async def drain(self) -> None:
        """Wait for all audit deliveries currently pending to finish."""
        while self._delivery_tasks:
            await asyncio.gather(
                *tuple(self._delivery_tasks), return_exceptions=True
            )

    async def shutdown(self) -> None:
        """Stop accepting audit deliveries and wait for pending deliveries."""
        self._accepting_deliveries = False
        await self.drain()
