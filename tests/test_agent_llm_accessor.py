# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public ``Agent.llm`` / ``Agent.set_llm`` accessors (issue #318).

These replace host reads/writes of the private ``agent._llm`` (e.g. the TUI
``/switch`` command and summarizer-budget syncing).
"""

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

_LLM_A = FakeLLMClient()
_LLM_B = FakeLLMClient()


class _LLMAgent(Agent, llm=_LLM_A):
    pass


def test_llm_property_returns_resolved_client():
    agent = _LLMAgent()
    assert agent.llm is _LLM_A


def test_set_llm_replaces_client():
    agent = _LLMAgent()
    agent.set_llm(_LLM_B)
    assert agent.llm is _LLM_B


def test_instance_llm_reflected_by_property():
    agent = _LLMAgent(llm=_LLM_B)
    assert agent.llm is _LLM_B


def test_llm_accessors_hidden_from_llm_docs():
    # The accessors are public to host code but must stay out of doc(self)
    # so the refactor does not change what the LLM sees (behaviour parity).
    from nooa.agentdoc._visibility import is_hidden_method

    assert is_hidden_method(Agent.set_llm)
    assert is_hidden_method(Agent.__dict__["llm"])
