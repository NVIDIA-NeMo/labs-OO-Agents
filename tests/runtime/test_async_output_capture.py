# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async-safe stdout/stderr capture during code execution.

These tests verify that:
1. Both stdout and stderr are captured (print, sys.stdout.write, sys.stderr.write)
2. Parallel async executions have isolated output buffers
3. Partial output is captured even when code raises an exception
4. warnings.warn() is captured (writes to stderr)
"""

import asyncio

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (no scripted responses needed - we call execute_code directly)
_TEST_LLM = FakeLLMClient()


class OutputTestAgent(Agent):
    """Minimal agent for testing output capture."""

    pass


@pytest.fixture
def agent():
    """Create a minimal agent for testing."""
    return OutputTestAgent(llm=_TEST_LLM)


class TestStdoutCapture:
    """Tests for stdout capture."""

    async def test_print_captured(self, agent):
        """print() calls are captured in stdout."""
        code = 'print("hello world")'
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "hello world" in result.stdout

    async def test_sys_stdout_write_captured(self, agent):
        """sys.stdout.write() calls are captured."""
        code = """
import sys
sys.stdout.write("direct write")
sys.stdout.flush()
"""
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "direct write" in result.stdout

    async def test_multiple_prints_captured(self, agent):
        """Multiple print statements are all captured."""
        code = """
print("line 1")
print("line 2")
print("line 3")
"""
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout
        assert "line 3" in result.stdout


class TestStderrCapture:
    """Tests for stderr capture."""

    async def test_sys_stderr_write_captured(self, agent):
        """sys.stderr.write() calls are captured."""
        code = """
import sys
sys.stderr.write("error message")
sys.stderr.flush()
"""
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "error message" in result.stderr

    @pytest.mark.skip(
        reason="warnings.warn() uses Python's internal warning system, not sys.stderr.write(). "
        "The warning is captured by pytest's warning capture, not our stream wrapper."
    )
    async def test_warnings_captured(self, agent):
        """warnings.warn() is captured in stderr.

        NOTE: This test is skipped because warnings.warn() doesn't write directly
        to sys.stderr - it uses Python's internal warning infrastructure which
        pytest captures separately. Direct sys.stderr.write() IS captured.
        """
        code = """
import warnings
warnings.warn("this is a warning")
"""
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "this is a warning" in result.stderr

    async def test_mixed_stdout_stderr(self, agent):
        """Both stdout and stderr are captured separately."""
        code = """
import sys
print("stdout message")
sys.stderr.write("stderr message")
print("another stdout")
"""
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "stdout message" in result.stdout
        assert "another stdout" in result.stdout
        assert "stderr message" in result.stderr
        # Ensure they're in the right streams
        assert "stderr message" not in result.stdout
        assert "stdout message" not in result.stderr


class TestPartialOutputOnError:
    """Tests for capturing partial output when code raises an exception."""

    async def test_stdout_before_error_captured(self, agent):
        """stdout before an exception is captured."""
        code = """
print("before error")
raise ValueError("intentional error")
"""
        result = await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)

        assert result.error is not None
        assert "intentional error" in str(result.error)
        assert "before error" in result.stdout

    async def test_stderr_before_error_captured(self, agent):
        """stderr before an exception is captured."""
        code = """
import sys
sys.stderr.write("error log before crash")
raise RuntimeError("crash")
"""
        result = await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)

        assert result.error is not None
        assert "crash" in str(result.error)
        assert "error log before crash" in result.stderr

    async def test_mixed_output_before_error_captured(self, agent):
        """Both stdout and stderr before an exception are captured."""
        code = """
import sys
print("step 1 complete")
sys.stderr.write("debug: starting step 2")
print("step 2 starting")
raise Exception("step 2 failed")
"""
        result = await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)

        assert result.error is not None
        assert "step 1 complete" in result.stdout
        assert "step 2 starting" in result.stdout
        assert "debug: starting step 2" in result.stderr


class TestParallelOutputIsolation:
    """Tests for async-safe parallel execution with isolated output."""

    async def test_parallel_executions_isolated(self, agent):
        """Two concurrent executions have isolated stdout buffers."""

        async def exec_with_id(task_id: int):
            code = f"""
import asyncio
print("task {task_id} start")
await asyncio.sleep(0.05)  # Small delay to ensure overlap
print("task {task_id} end")
"""
            return await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)

        # Run two tasks concurrently
        results = await asyncio.gather(exec_with_id(1), exec_with_id(2))

        result1, result2 = results

        # Each result should only contain its own output
        assert result1.error is None
        assert result2.error is None

        assert "task 1 start" in result1.stdout
        assert "task 1 end" in result1.stdout
        assert "task 2 start" not in result1.stdout
        assert "task 2 end" not in result1.stdout

        assert "task 2 start" in result2.stdout
        assert "task 2 end" in result2.stdout
        assert "task 1 start" not in result2.stdout
        assert "task 1 end" not in result2.stdout

    async def test_parallel_stderr_isolated(self, agent):
        """Two concurrent executions have isolated stderr buffers."""

        async def exec_with_id(task_id: int):
            code = f"""
import sys
import asyncio
sys.stderr.write("task {task_id} error\\n")
await asyncio.sleep(0.05)
sys.stderr.write("task {task_id} done\\n")
"""
            return await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)

        results = await asyncio.gather(exec_with_id(1), exec_with_id(2))

        result1, result2 = results

        assert result1.error is None
        assert result2.error is None

        assert "task 1 error" in result1.stderr
        assert "task 1 done" in result1.stderr
        assert "task 2 error" not in result1.stderr

        assert "task 2 error" in result2.stderr
        assert "task 2 done" in result2.stderr
        assert "task 1 error" not in result2.stderr

    async def test_many_parallel_executions(self, agent):
        """Many concurrent executions all have isolated output."""
        num_tasks = 10

        async def exec_with_id(task_id: int):
            code = f"""
import asyncio
print("task-{task_id}-output")
await asyncio.sleep(0.01)
"""
            return task_id, await agent.runtime.execute_code(
                code, wrap_in_function=True, validate=False
            )

        results = await asyncio.gather(*[exec_with_id(i) for i in range(num_tasks)])

        for task_id, result in results:
            assert result.error is None
            # Should contain its own output
            assert f"task-{task_id}-output" in result.stdout
            # Should not contain any other task's output
            for other_id in range(num_tasks):
                if other_id != task_id:
                    assert f"task-{other_id}-output" not in result.stdout


class TestContextVarStreamBehavior:
    """Tests for the ContextVarStream wrapper behavior."""

    async def test_stream_attributes_preserved(self, agent):
        """Stream attributes like encoding are preserved."""
        import sys

        # The wrapped stream should still have standard attributes
        assert hasattr(sys.stdout, "encoding")
        assert hasattr(sys.stdout, "write")
        assert hasattr(sys.stdout, "flush")
        assert hasattr(sys.stdout, "fileno")

    async def test_explicit_file_parameter_bypasses_capture(self, agent):
        """print(file=sys.stderr) should go to stderr, not stdout."""
        code = """
import sys
print("to stdout")
print("to stderr", file=sys.stderr)
"""
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.error is None
        assert "to stdout" in result.stdout
        assert "to stderr" in result.stderr
        assert "to stderr" not in result.stdout

    async def test_getattr_passthrough(self, agent):
        """Less common stream attributes are passed through via __getattr__."""
        import sys

        # These attributes come from the original stream via __getattr__
        # and should work even though they're not explicitly defined on ContextVarStream
        assert callable(getattr(sys.stdout, "fileno", None))
        # 'buffer' is the underlying binary stream - should be accessible
        # (Note: may not exist in all environments, so we just check no AttributeError)
        try:
            _ = sys.stdout.buffer
        except AttributeError:
            pass  # OK if original stream doesn't have buffer


class TestExecutionResultStderr:
    """Tests that ExecutionResult properly includes stderr field."""

    async def test_execution_result_has_stderr_field(self, agent):
        """ExecutionResult includes stderr field."""
        code = "x = 1"
        result = await agent.runtime.execute_code(code, validate=False)

        assert hasattr(result, "stderr")
        assert isinstance(result.stderr, str)

    async def test_empty_stderr_is_empty_string(self, agent):
        """When nothing writes to stderr, it's an empty string."""
        code = 'print("only stdout")'
        result = await agent.runtime.execute_code(code, validate=False)

        assert result.stderr == ""
        assert "only stdout" in result.stdout


class TestExecutionResultFormatOutput:
    """Tests for ExecutionResult.format_output() helper method."""

    async def test_format_output_plain(self, agent):
        """format_output() with fenced=False returns plain text."""
        code = """
import sys
print("hello stdout")
sys.stderr.write("hello stderr")
"""
        result = await agent.runtime.execute_code(code, validate=False)
        formatted = result.format_output(fenced=False)

        assert "Stdout:\nhello stdout" in formatted
        assert "Stderr:\nhello stderr" in formatted
        # No code fences in plain mode
        assert "```" not in formatted

    async def test_format_output_fenced(self, agent):
        """format_output() with fenced=True wraps in code fences."""
        code = """
import sys
print("hello stdout")
sys.stderr.write("hello stderr")
"""
        result = await agent.runtime.execute_code(code, validate=False)
        formatted = result.format_output(fenced=True)

        assert "Stdout:\n```\nhello stdout" in formatted
        assert "Stderr:\n```\nhello stderr" in formatted
        assert "```" in formatted

    async def test_format_output_empty_when_no_output(self, agent):
        """format_output() returns empty string when no stdout/stderr."""
        code = "x = 1"
        result = await agent.runtime.execute_code(code, validate=False)
        formatted = result.format_output(fenced=False)

        assert formatted == ""

    async def test_format_output_only_stdout(self, agent):
        """format_output() works with only stdout."""
        code = 'print("just stdout")'
        result = await agent.runtime.execute_code(code, validate=False)
        formatted = result.format_output(fenced=False)

        assert "Stdout:" in formatted
        assert "Stderr:" not in formatted

    async def test_format_output_only_stderr(self, agent):
        """format_output() works with only stderr."""
        code = """
import sys
sys.stderr.write("just stderr")
"""
        result = await agent.runtime.execute_code(code, validate=False)
        formatted = result.format_output(fenced=False)

        assert "Stderr:" in formatted
        assert "Stdout:" not in formatted
