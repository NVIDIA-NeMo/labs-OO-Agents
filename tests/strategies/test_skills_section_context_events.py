# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Skills section in ``<execution_context>``.

The base ``Agent`` declares ``self.context: Annotated["ContextApi", hidden, ...]``
and ``self.events: Annotated["EventsApi", hidden, ...]`` — both are Skill
subclasses, but they're hidden from the LLM by default. An agent opts in
per-instance by calling ``spec(self, "context", hidden=False)`` (or
``"events"``) in its ``__init__``.

These tests pin the contract:

- Base Agent with no ``spec()`` calls: Skills section is absent; neither
  ``self.context`` nor ``self.events`` appears in the rendered prompt's
  Skills table.
- ``spec(self, "context", hidden=False)``: Skills section renders, contains
  a ``self.context`` row, and emits the "Pin to context" / "Unpin"
  instruction lines.
- ``spec(self, "events", hidden=False)``: Skills section renders, contains
  a ``self.events`` row; pin/unpin lines do NOT appear (those are
  Context-API-specific).
- Both ``spec()`` calls: Both rows appear and pin/unpin is present.
- A real user Skill alongside an un-hidden Context API: both rows appear.

This fixture surfaces a bug fixed alongside these tests:
``is_hidden_field`` was being called with ``type(runtime.agent)`` (the class)
instead of the instance, which made the ``spec(self, ...)`` instance-level
override a no-op. These tests will fail with the old call.
"""

from __future__ import annotations

import pytest

from nooa import Agent
from nooa.agentdoc import spec
from nooa.prompts import build_prompt_data
from nooa.skill import Skill
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
