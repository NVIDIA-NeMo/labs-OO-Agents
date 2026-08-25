# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.interactive — the dispatcher-driven agent base.

Deeper behavioral coverage (dispatcher loop, snapshot restore, echo hooks)
lives with the TUI package, whose BaseTUIAgent subclasses this. These tests
pin the core contract the ARC-AGI-3 example and other hosts rely on.
"""

import pytest
from pydantic import ValidationError

from nooa.agentdoc import doc
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


def test_declares_only_the_user_channel(agent):
    """Being dispatcher-driven implies a human feeding it, and nothing more.

    Hosts declare whatever else they need. slash_commands and system_messages
    are coding-host concepts and live on CodingAgent — see
    packages/nooa-cli/tests/test_coding_agent.py.
    """
    assert agent.queue_manager.channels().keys() == {"user_messages"}
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


def test_persistent_vars_inspection_and_cleanup_api(agent):
    agent.v.cursor = 3
    agent.v.plan = "draft"

    assert agent.v.keys() == ["cursor", "plan"]
    assert agent.v.items() == [("cursor", 3), ("plan", "draft")]
    assert agent.v.get("cursor") == 3
    assert agent.v.get("missing", "fallback") == "fallback"

    rendered = doc(type(agent.v))
    assert "def keys(self) -> list[str]" in rendered
    assert "def items(self) -> list[tuple[str, Any]]" in rendered
    assert "def get(self, key: str, default: Any = None) -> Any" in rendered
    assert "def clear(self) -> None" in rendered

    agent.v.clear()
    assert agent.v.keys() == []


def test_message_records_event_and_renders(agent):
    rendered: list[str] = []
    agent._render_message = lambda text, **kw: rendered.append(text)
    agent.message("**hi**")
    assert rendered == ["**hi**"]
    events = [e for e in agent.event_manager.values() if isinstance(e, AgentMessage)]
    assert len(events) == 1
    assert events[0].content == "**hi**"


def test_rename_session_uses_host_session_manager(agent):
    class SessionManager:
        user_named = False
        name = None

        def __init__(self):
            self.renames = []

        def rename(self, title, *, user_named):
            self.name = title
            self.renames.append((title, user_named))

    manager = SessionManager()
    agent._session_manager = manager

    assert agent.rename_session('  "Debug   TUI input"  ') == "Debug TUI input"
    assert manager.renames == [("Debug TUI input", False)]


def test_rename_session_preserves_user_selected_title(agent):
    class SessionManager:
        user_named = True
        name = "My chosen title"

        def rename(self, title, *, user_named):
            raise AssertionError("automatic title overwrote a user-selected title")

    agent._session_manager = SessionManager()
    assert agent.rename_session("Automatic title") == "My chosen title"


def test_rename_session_is_model_visible_but_request_helper_is_hidden(agent):
    rendered = str(doc(agent))
    assert "rename_session" in rendered
    assert "request_session_title" not in rendered


def test_session_title_request_is_not_a_core_agent_concern(agent):
    assert not hasattr(agent, "request_session_title")


def test_respond_result_requires_explanation():
    result = RespondResult(kind=RespondReason.DONE, explanation="did the thing")
    assert result.kind is RespondReason.DONE
    with pytest.raises(ValidationError):
        RespondResult(kind=RespondReason.DONE, explanation="   ")


def test_respond_result_rejects_removed_get_user_input_reason():
    with pytest.raises(ValidationError):
        RespondResult(kind="GET_USER_INPUT", explanation="legacy reason")


def test_install_summarizer_none_policy_is_noop(agent):
    install_summarizer(SummarizationConfig(policy="none"), agent=agent)
    assert not getattr(agent, "_summarizers", [])


def test_install_summarizer_attaches(agent):
    install_summarizer(SummarizationConfig(max_tokens=50_000), agent=agent)
    summarizers = getattr(agent, "_summarizers", [])
    assert len(summarizers) == 1
    assert summarizers[0].config.max_tokens == 50_000
