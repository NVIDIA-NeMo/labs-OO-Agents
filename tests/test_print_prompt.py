# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for print_prompt parameter expansion in CodeActStrategy."""

import pytest

from nooa import Agent, build_prompt_data
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


class _TestAgent(Agent, llm=_LLM):
    async def analyze(self, data: str) -> str:
        """Analyze the given {data} and return insights."""
        ...


@pytest.mark.asyncio
async def test_build_prompt_data_expands_docstring_params():
    """Method parameter {data} in docstring should be expanded to actual value."""
    agent = _TestAgent()
    data = await build_prompt_data(agent.analyze, "test input")
    # The task prompt should contain the actual value, not the placeholder
    assert "test input" in data.task_prompt
    assert "{data}" not in data.task_prompt


@pytest.mark.asyncio
async def test_build_prompt_data_preserves_non_param_braces():
    """Curly braces that are NOT parameter references should survive."""

    class BracesAgent(Agent, llm=_LLM):
        async def process(self, items: str) -> str:
            """Process {items}. Example output: {{"key": "value"}}."""
            ...

    agent = BracesAgent()
    data = await build_prompt_data(agent.process, "my items")
    assert "my items" in data.task_prompt
    # Escaped braces {{...}} should become literal {key: value} in output
    assert '{"key": "value"}' in data.task_prompt
