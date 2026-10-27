# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt visibility for the context and events APIs.

The execution context no longer renders the old Skills table. Events usage
guidance lives in a dedicated framework block, enabled by
instance-level visibility and shared across strategies. Hidden events must
not inject that block, including after visibility changes on an existing agent.
"""

from __future__ import annotations

import pytest

from nooa import Agent
from nooa.agentdoc import spec
from nooa.decorators import strategy
from nooa.prompts import build_prompt_data
from nooa.skill import Skill
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.predict import PredictStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


# ---------------------------------------------------------------------------
# Test agents — each declares a different visibility configuration
# ---------------------------------------------------------------------------


class _BaseAgent(Agent, llm=_LLM):
    """No spec() calls — context and events remain hidden."""

    async def go(self) -> str:
        """Run."""
        ...


class _ContextOnlyAgent(Agent, llm=_LLM):
    """Unhides self.context only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "context", hidden=False)

    async def go(self) -> str:
        """Run."""
        ...


class _EventsOnlyAgent(Agent, llm=_LLM):
    """Unhides self.events only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "events", hidden=False)

    async def go(self) -> str:
        """Run."""
        ...

    @strategy(CodeActV2())
    async def go_v2(self) -> str:
        """Run."""
        ...

    @strategy(PredictStrategy())
    async def predict(self) -> str:
        """Run."""
        ...

    @strategy(PurePythonStrategy())
    async def pure_python(self) -> str:
        """Run."""
        ...


class _BothUnhiddenAgent(Agent, llm=_LLM):
    """Unhides both self.context and self.events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)

    async def go(self) -> str:
        """Run."""
        ...


class _UserSkill(Skill):
    """A made-up tool exposed as a Skill."""


class _UserSkillPlusContextAgent(Agent, llm=_LLM):
    """A user-defined Skill plus an un-hidden Context API."""

    tool: _UserSkill = _UserSkill()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "context", hidden=False)

    async def go(self) -> str:
        """Run."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_execution_context(system_prompt: str) -> str:
    """Pull just the <execution_context> block from the system prompt."""
    start = system_prompt.find("<execution_context")
    assert start >= 0, "No <execution_context> block in system prompt"
    end = system_prompt.find("</execution_context>", start)
    assert end >= 0, "No </execution_context> closing tag in system prompt"
    return system_prompt[start : end + len("</execution_context>")]


async def _render_execution_context(agent: Agent) -> str:
    data = await build_prompt_data(agent.go)
    return _extract_execution_context(data.system_prompt or "")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["go", "go_v2", "predict", "pure_python"])
async def test_events_guidance_tracks_instance_visibility(method_name):
    """Visibility toggles the entire guidance block across all strategies."""
    agent = _EventsOnlyAgent()
    method = getattr(agent, method_name)
    prompt = (await build_prompt_data(method)).system_prompt or ""
    assert "<events_api" in prompt
    assert "You MUST call doc(self.events) before using it" in prompt
    assert "print(doc(self.events))" in prompt
    assert "Query past events by type, tag, text, or call ID" in prompt
    assert "self.events.collapse" not in prompt

    spec(agent, "events", hidden=True)
    prompt = (await build_prompt_data(method)).system_prompt or ""
    assert "<events_api" not in prompt
    assert "You MUST call doc(self.events) before using it" not in prompt
    assert "print(doc(self.events))" not in prompt

    spec(agent, "events", hidden=False)
    prompt = (await build_prompt_data(method)).system_prompt or ""
    assert "<events_api" in prompt
    assert "You MUST call doc(self.events) before using it" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_cls", [_BaseAgent, _ContextOnlyAgent])
async def test_hidden_events_have_no_guidance(agent_cls):
    """Hidden events inject neither instructions nor an empty XML block."""
    prompt = (await build_prompt_data(agent_cls().go)).system_prompt or ""
    assert "<events_api" not in prompt
    assert "You MUST call doc(self.events) before using it" not in prompt
    assert "print(doc(self.events))" not in prompt


class TestSkillsSectionVisibility:
    """Execution context no longer renders a Skills section."""

    @pytest.mark.asyncio
    async def test_base_agent_has_no_skills_section(self):
        ec = await _render_execution_context(_BaseAgent())
        assert "## Skills" not in ec
        assert "| Skill | Description |" not in ec

    @pytest.mark.asyncio
    async def test_unhiding_context_still_has_no_skills_section(self):
        ec = await _render_execution_context(_ContextOnlyAgent())
        assert "## Skills" not in ec
        assert "`self.context`" not in ec
        assert "Pin to context" not in ec

    @pytest.mark.asyncio
    async def test_unhiding_events_still_has_no_skills_section(self):
        ec = await _render_execution_context(_EventsOnlyAgent())
        assert "## Skills" not in ec
        assert "`self.events`" not in ec

    @pytest.mark.asyncio
    async def test_unhiding_both_still_has_no_skills_section(self):
        ec = await _render_execution_context(_BothUnhiddenAgent())
        assert "## Skills" not in ec
        assert "`self.context`" not in ec
        assert "`self.events`" not in ec

    @pytest.mark.asyncio
    async def test_user_skill_and_context_still_have_no_skills_section(self):
        ec = await _render_execution_context(_UserSkillPlusContextAgent())
        assert "## Skills" not in ec
        assert "`self.context`" not in ec
        assert "`self.tool`" not in ec
