# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionCleared is emitted after /clear resets the agent's working state."""

import pytest
from nooa_cli.sessions import SessionCleared


def test_event_is_runtime_role_and_fields():
    from nooa.context_blocks.models import Role

    e = SessionCleared(session_id="abc")
    assert e.session_id == "abc"
    assert type(e)._role is Role.RUNTIME_EVENT
    # session_id is optional (may be unknown at clear time).
    assert SessionCleared().session_id is None


def test_event_keeps_legacy_handler_alias_during_migration():
    assert SessionCleared.handler_aliases == ("TuiSessionCleared",)


@pytest.mark.asyncio
async def test_reset_emits_cleared_event_to_subscribers():
    """_reset_agent_working_state emits SessionCleared; a subscriber receives it."""
    from nooa_cli.tui.commands import _reset_agent_working_state

    from nooa import Agent
    from nooa.unifiedllm import FakeLLMClient

    class _A(Agent, llm=FakeLLMClient()):
        pass

    agent = _A()
    agent.event_manager.register_event_type(SessionCleared)
    seen = []
    agent.event_manager.on("SessionCleared", lambda e: seen.append(e))

    await _reset_agent_working_state(agent)

    assert len(seen) == 1
    assert isinstance(seen[0], SessionCleared)
