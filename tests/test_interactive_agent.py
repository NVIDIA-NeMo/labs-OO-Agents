# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.interactive — the dispatcher-driven agent base.

Deeper behavioral coverage (dispatcher loop, snapshot restore, echo hooks)
lives with the TUI package, whose BaseTUIAgent subclasses this. These tests
pin the core contract the ARC-AGI-3 example and other hosts rely on.
"""

import pytest
from pydantic import ValidationError

from nooa.interactive import (
    AgentMessage,
    AgentVars,
    InteractiveAgent,
    RespondReason,
    RespondResult,
    SummarizationConfig,
    install_summarizer,
)
from nooa.unifiedllm import FakeLLMClient


class _Host(InteractiveAgent, llm=FakeLLMClient()):
    """Minimal InteractiveAgent subclass standing in for a host-driven agent."""


@pytest.fixture
def agent():
    return _Host(llm=FakeLLMClient())


def test_declares_standard_queues(agent):
    for name in ("user_messages", "slash_commands", "system_messages"):
        assert name in agent.queue_manager.channels()
    # Reader facade exposed under the public name; producer side hidden.
    assert agent.user_messages is agent._user_messages_in.reader


async def test_queue_roundtrip(agent):
    agent._user_messages_in.put("hello")
    assert await agent.user_messages.get() == "hello"


def test_persistent_vars_proxy(agent):
    agent.v.cursor = 3
    assert agent.v.cursor == 3
    assert "cursor" in agent.v
    assert agent.vars["cursor"] == 3
    del agent.v.cursor
    with pytest.raises(AttributeError):
        _ = agent.v.cursor
    assert isinstance(agent.v, AgentVars)


def test_message_records_event_and_renders(agent):
    rendered: list[str] = []
    agent._render_message = lambda text, **kw: rendered.append(text)
    agent.message("**hi**")
    assert rendered == ["**hi**"]
    events = [e for e in agent.event_manager.values() if isinstance(e, AgentMessage)]
    assert len(events) == 1
    assert events[0].content == "**hi**"


def test_respond_result_requires_explanation():
    result = RespondResult(kind=RespondReason.DONE, explanation="did the thing")
    assert result.kind is RespondReason.DONE
    with pytest.raises(ValidationError):
        RespondResult(kind=RespondReason.DONE, explanation="   ")


def test_install_summarizer_none_policy_is_noop(agent):
    install_summarizer(SummarizationConfig(policy="none"), agent=agent)
    assert not getattr(agent, "_summarizers", [])


def test_install_summarizer_attaches(agent):
    install_summarizer(SummarizationConfig(max_tokens=50_000), agent=agent)
    summarizers = getattr(agent, "_summarizers", [])
    assert len(summarizers) == 1
    assert summarizers[0].config.max_tokens == 50_000
