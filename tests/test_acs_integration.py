# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from nooa.acs.schema import ACSDecision, ACSHook
from nooa.acs.handler import ACSHandler
from nooa.acs.integration import enforce_acs_decision, ACSBlockedError

class MockHandler(ACSHandler):
    def __init__(self, decisions):
        self.decisions = decisions
        self.call_count = 0
        
    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        decision = self.decisions[self.call_count]
        self.call_count += 1
        return decision

@pytest.mark.asyncio
async def test_enforce_allow():
    handler = MockHandler([ACSDecision(status="allow")])
    
    async def mock_execute():
        return "success"
        
    result = await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, mock_execute
    )
    assert result == "success"

@pytest.mark.asyncio
async def test_enforce_deny():
    handler = MockHandler([ACSDecision(status="deny", reason="Unauthorized")])
    
    async def mock_execute():
        return "success"
        
    with pytest.raises(ACSBlockedError, match="Unauthorized"):
        await enforce_acs_decision(
            handler, "toolCallRequest", {}, {}, mock_execute
        )

@pytest.mark.asyncio
async def test_enforce_defer_then_allow():
    # First it defers for 10ms, then it allows
    handler = MockHandler([
        ACSDecision(status="defer", retry_after_ms=10),
        ACSDecision(status="allow")
    ])
    
    async def mock_execute():
        return "success after retry"
        
    result = await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, mock_execute
    )
    assert result == "success after retry"
    assert handler.call_count == 2
