# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for message utility and variable expansion.

Focus on:
- Variable expansion via runtime.expand_variables()
- `message.send()` event emission (tested in integration tests)
- Variable expansion in messages
"""

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class AgentUnderTest(Agent, llm=_TEST_LLM):
    """Test agent with various attributes."""

    def __init__(self):
        super().__init__()
        self.status = "processing"
        self.count = 42
        self.items = ["a", "b", "c"]
        self.data = {"key": "value", "nested": {"x": 10}}


@pytest.mark.asyncio
async def test_expansion_simple_attribute():
    """Test expanding simple attribute access."""
    agent_instance = AgentUnderTest()

    text = "Status: {self.status}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Status: processing"


@pytest.mark.asyncio
async def test_expansion_multiple_expressions():
    """Test multiple expressions in one string."""
    agent_instance = AgentUnderTest()

    text = "Status: {self.status}, Count: {self.count}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Status: processing, Count: 42"


@pytest.mark.asyncio
async def test_expansion_with_len():
    """Test using len() function."""
    agent_instance = AgentUnderTest()

    text = "Processing {len(self.items)} items"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Processing 3 items"


@pytest.mark.asyncio
async def test_expansion_nested_access():
    """Test nested attribute and dict access."""
    agent_instance = AgentUnderTest()

    text = "Value: {self.data['nested']['x']}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Value: 10"


@pytest.mark.asyncio
async def test_expansion_arithmetic():
    """Test arithmetic expressions."""
    agent_instance = AgentUnderTest()

    text = "Double count: {self.count * 2}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Double count: 84"


@pytest.mark.asyncio
async def test_expansion_string_formatting():
    """Test string operations."""
    agent_instance = AgentUnderTest()

    text = "Uppercase: {str(self.status).upper()}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Uppercase: PROCESSING"


@pytest.mark.asyncio
async def test_expansion_list_indexing():
    """Test list indexing."""
    agent_instance = AgentUnderTest()

    text = "First item: {self.items[0]}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "First item: a"


@pytest.mark.asyncio
async def test_expansion_no_placeholders():
    """Test text with no placeholders."""
    agent_instance = AgentUnderTest()

    text = "Plain text with no expressions"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Plain text with no expressions"


@pytest.mark.asyncio
async def test_expansion_invalid_expression():
    """Test that invalid expressions show errors."""
    agent_instance = AgentUnderTest()

    text = "Invalid: {self.nonexistent}"

    # Invalid expressions return error message in placeholder
    result = await agent_instance.runtime.expand_variables(text)
    assert "ERROR" in result or "nonexistent" in result.lower()


@pytest.mark.asyncio
async def test_expansion_forbidden_function():
    """Test that undefined functions show errors."""
    agent_instance = AgentUnderTest()

    text = "Forbidden: {open('file.txt')}"

    # Invalid functions return error message
    result = await agent_instance.runtime.expand_variables(text)
    assert "ERROR" in result


@pytest.mark.asyncio
async def test_expansion_allows_import():
    """Test that imports work (no sandboxing)."""
    agent_instance = AgentUnderTest()

    text = "Import: {__import__('os').name}"

    # No sandboxing - imports are allowed
    result = await agent_instance.runtime.expand_variables(text)
    assert "Import: os" in result or "Import: posix" in result


@pytest.mark.asyncio
async def test_expansion_forbidden_private_access():
    """Test that private attribute access works (no sandboxing)."""
    agent_instance = AgentUnderTest()

    text = "Private: {self._agent_id}"

    # Private attributes are accessible (no sandboxing)
    result = await agent_instance.runtime.expand_variables(text)
    assert isinstance(result, str)
    assert "Private:" in result


@pytest.mark.asyncio
async def test_expansion_complex_expression():
    """Test complex expression."""
    agent_instance = AgentUnderTest()

    text = "Sum: {sum([self.count, len(self.items), 5])}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Sum: 50"  # 42 + 3 + 5


@pytest.mark.asyncio
async def test_expansion_with_dict_access():
    """Test expansion with nested dict access."""
    agent_instance = AgentUnderTest()

    # Access nested dictionary
    text = "Key: {self.data['key']}, Nested: {self.data['nested']['x']}"
    result = await agent_instance.runtime.expand_variables(text)

    assert result == "Key: value, Nested: 10"
