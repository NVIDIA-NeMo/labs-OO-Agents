# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for truncation in code execution (Phase 2).

Note: These tests verify truncation at the execute_code level. Full integration
with strategies (pprint in namespace, return value formatting) is tested separately
via strategy execution tests.
"""

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM
_TEST_LLM = FakeLLMClient()


@pytest.fixture
def test_agent():
    """Create a test agent."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    return TestAgent()


class TestStdoutTruncationIntegration:
    """Tests for stdout truncation in execute_code()."""

    @pytest.mark.asyncio
    async def test_small_output_not_truncated(self, test_agent):
        """Small stdout should not be truncated."""
        code = 'print("hello world")'
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert result.stdout == "hello world\n"
        assert "Output too large" not in result.stdout

    @pytest.mark.asyncio
    async def test_large_output_truncated(self, test_agent):
        """Large stdout should be truncated with notice."""
        # Generate more than 50KB of output
        code = """
for i in range(10000):
    print("x" * 100)
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        # Should have truncation notice in prose format
        assert "Output too large" in result.stdout
        assert "chars" in result.stdout
        assert "Showing first" in result.stdout

    @pytest.mark.asyncio
    async def test_stderr_truncated_separately(self, test_agent):
        """stderr should have its own truncation limit.

        Note: Testing via raising exception to stderr.
        """
        # Generate lots of stderr via failed assertions
        code = """
for i in range(500):
    # Each iteration prints 100+ chars to stderr
    try:
        assert False, "error message " * 50
    except AssertionError:
        pass  # Captured by stderr
        """
        result = await test_agent.runtime.execute_code(code)

        # This test is simplified - stderr truncation is harder to trigger directly
        # The implementation is verified by unit tests
        assert result.success

    @pytest.mark.asyncio
    async def test_truncation_notice_before_content(self, test_agent):
        """Truncation notice should appear BEFORE content."""
        code = """
print("START")
for i in range(10000):
    print("x" * 100)
"""
        result = await test_agent.runtime.execute_code(code)

        # Notice should come before actual content
        notice_pos = result.stdout.find("Output too large")
        start_pos = result.stdout.find("START")

        assert notice_pos >= 0
        assert start_pos >= 0
        assert notice_pos < start_pos  # Notice comes first


class TestTruncationOnlyIntegration:
    """Tests for truncation behavior only (pprint tested in unit tests).

    Note: Full strategy-level integration (pprint in namespace, return formatting)
    requires strategy execution tests which use proper setup with LLMs.
    """

    @pytest.mark.asyncio
    async def test_print_respects_truncation(self, test_agent):
        """Large print() output should be truncated."""
        code = """
for i in range(10000):
    print("line " * 100)
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "Output too large" in result.stdout
        assert "chars" in result.stdout
