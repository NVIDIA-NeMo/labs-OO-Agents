# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval_pipeline execute."""

import asyncio

import pytest

from eval_pipeline.execute import execute_task
from eval_pipeline.models import Task


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, response: str = "result", should_fail: bool = False, delay: float = 0):
        self.response = response
        self.should_fail = should_fail
        self.delay = delay
        self.run_called = False
        self.last_input = None

    async def run(self, input: str):
        self.run_called = True
        self.last_input = input
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise ValueError("Agent failed")
        return self.response


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_successful_execution(self, tmp_path):
        agent = MockAgent(response="positive")
        task = Task(id="t1", input="classify this", expected="positive")
        trace_file = tmp_path / "traces" / "t1.jsonl"

        result = await execute_task(agent, task, trace_file)

        assert result.task_id == "t1"
        assert result.input == "classify this"
        assert result.expected == "positive"
        assert result.actual == "positive"
        assert result.error is None
        assert result.latency_ms > 0
        assert agent.run_called
        assert agent.last_input == "classify this"

    @pytest.mark.asyncio
    async def test_creates_trace_directory(self, tmp_path):
        agent = MockAgent()
        task = Task(id="t1", input="x", expected="y")
        trace_file = tmp_path / "deep" / "nested" / "trace.jsonl"

        await execute_task(agent, task, trace_file)

        assert trace_file.parent.exists()

    @pytest.mark.asyncio
    async def test_agent_failure(self, tmp_path):
        agent = MockAgent(should_fail=True)
        task = Task(id="t1", input="x", expected="y")
        trace_file = tmp_path / "trace.jsonl"

        result = await execute_task(agent, task, trace_file)

        assert result.error == "Agent failed"
        assert result.actual is None

    @pytest.mark.asyncio
    async def test_timeout_triggers_error(self, tmp_path):
        """Test that a slow agent is terminated by timeout."""
        agent = MockAgent(response="slow result", delay=1.0)  # 1 second delay
        task = Task(id="t1", input="x", expected="y")
        trace_file = tmp_path / "trace.jsonl"

        result = await execute_task(agent, task, trace_file, timeout_seconds=0.1)

        assert result.actual is None
        assert result.error is not None
        assert "Timeout" in result.error
        assert "0.1s" in result.error  # Should mention the limit
        assert result.latency_ms >= 100  # Should have run for at least 100ms

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self, tmp_path):
        """Test that no timeout is applied when timeout_seconds is None."""
        agent = MockAgent(response="result", delay=0.1)
        task = Task(id="t1", input="x", expected="y")
        trace_file = tmp_path / "trace.jsonl"

        # Should complete successfully even with delay when no timeout set
        result = await execute_task(agent, task, trace_file, timeout_seconds=None)

        assert result.actual == "result"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fast_agent_completes_before_timeout(self, tmp_path):
        """Test that a fast agent completes normally with timeout set."""
        agent = MockAgent(response="fast result", delay=0.01)  # 10ms delay
        task = Task(id="t1", input="x", expected="y")
        trace_file = tmp_path / "trace.jsonl"

        result = await execute_task(agent, task, trace_file, timeout_seconds=1.0)

        assert result.actual == "fast result"
        assert result.error is None
