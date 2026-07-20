# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test module imports passthrough to LLM-generated code.
"""

import json  # noqa: F401 — for LLM exec_globals (visible by default)

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class ImportTestAgent(Agent, llm=_TEST_LLM):
    """Agent to test module imports."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result = None

    @strategy(PurePythonStrategy())
    async def use_json(self) -> dict:
        """Use the json module to create a dict."""
        ...


@pytest.mark.asyncio
async def test_module_imports_available():
    """Test that imports from agent's module are available in generated code."""
    # Use FakeLLMClient to generate code that uses json module
    llm_client = FakeLLMClient(
        scripted_responses=[
            _resp('self.result = json.loads(\'{"key": "value"}\')\nreturn self.result')
        ]
    )

    agent_inst = ImportTestAgent(llm=llm_client)
    result = await agent_inst.use_json()

    # Verify json module was available (should not raise NameError)
    assert result == {"key": "value"}
    assert agent_inst.result == {"key": "value"}


@pytest.mark.asyncio
async def test_imports_shown_in_prompt():
    """Test that imports from agent's module appear in LLM prompt."""
    agent_inst = ImportTestAgent()

    # Get the method
    method = agent_inst.use_json

    # Build messages using runtime's _build_messages
    messages = await agent_inst.runtime._build_messages(method=method)

    # Get system message content
    system_message = next((m for m in messages if m["role"] == "system"), None)
    prompt_text = system_message["content"] if system_message else ""

    # Check that json module is listed in available imports
    assert "json" in prompt_text
    # Should show it's a module
    assert "module" in prompt_text.lower()
