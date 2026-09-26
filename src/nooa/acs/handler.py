# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Any

from nooa.acs.schema import ACSDecision, ACSHook

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
    
    def __init__(self, sink_func: Callable[[ACSHook], Any]):
        self.sink_func = sink_func

    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        # Call the sink function (can be sync or async)
        if asyncio.iscoroutinefunction(self.sink_func):
            await self.sink_func(hook)
        else:
            self.sink_func(hook)
        
        return ACSDecision(status="allow")
