# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for async deadlock prevention.

These tests verify that:
1. Patterns that would deadlock are blocked with clear errors
2. Nested event loop patterns (asyncio.run, run_until_complete) are blocked
3. The validation integrates correctly with the execution path

NOTE: These tests are designed to FAIL before the fix is applied,
demonstrating that the dangerous patterns are not yet blocked.
After applying the fix, all tests should PASS.
"""

import pytest

from nooa import Agent
from nooa.errors import RestrictedCodeError
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


@pytest.fixture
def test_agent():
    """Create a test agent for execution tests."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    return TestAgent()


class TestDeadlockPatternsBlocked:
    """Tests that deadlock-causing patterns are blocked at runtime.

    These patterns would cause the event loop to deadlock if executed.
    The sandbox catches them at call time and returns a clear error message.

    BEFORE FIX: These tests FAIL (patterns not blocked, may deadlock)
    AFTER FIX: These tests PASS (patterns blocked with clear error)
    """

    @pytest.mark.asyncio
    async def test_blocks_run_coroutine_threadsafe(self, test_agent):
        """run_coroutine_threadsafe() should be blocked - it's never correct in async context.

        This is the exact pattern that caused deadlocks with qwen3-80b:
        asyncio.run_coroutine_threadsafe(coro(), loop).result()
        """
        # The sandbox catches run_coroutine_threadsafe at call time
        code = """
loop = asyncio.get_event_loop()
asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success, "run_coroutine_threadsafe should be blocked"
        assert result.error is not None
        assert isinstance(result.error, RestrictedCodeError)
        assert "run_coroutine_threadsafe" in str(result.error)

    @pytest.mark.asyncio
    async def test_error_message_guides_fix(self, test_agent):
        """Error message should tell the user to use await instead."""
        # Use the same valid code as above to test the error message
        code = """
loop = asyncio.get_event_loop()
asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success
        error_msg = str(result.error).lower()
        assert "await" in error_msg, "Error should suggest using await"


class TestNestedLoopPatternsBlocked:
    """Tests that nested event loop patterns are blocked with clear errors.

    These patterns would raise "cannot run nested event loop" at runtime.
    We block them at validation time with helpful error messages.

    BEFORE FIX: These tests FAIL (patterns not blocked, RuntimeError at runtime)
    AFTER FIX: These tests PASS (patterns blocked with clear error message)
    """

    @pytest.mark.asyncio
    async def test_asyncio_run_blocked(self, test_agent):
        """asyncio.run() inside async context should be blocked.

        This pattern fails at runtime with:
        RuntimeError: asyncio.run() cannot be called from a running event loop

        We block it at validation time with a helpful message.
        """
        code = """
async def inner():
    return 42

result = asyncio.run(inner())
print(f"result={result}")
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success, "asyncio.run() should be blocked in async context"
        assert result.error is not None
        assert isinstance(result.error, RestrictedCodeError)
        assert "asyncio.run()" in str(result.error)
        assert "await" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_run_until_complete_blocked(self, test_agent):
        """loop.run_until_complete() should be blocked.

        This pattern fails at runtime with:
        RuntimeError: This event loop is already running

        We block it at validation time with a helpful message.
        """
        code = """
async def inner():
    return 123

loop = asyncio.get_event_loop()
result = loop.run_until_complete(inner())
print(f"result={result}")
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success, "run_until_complete() should be blocked in async context"
        assert result.error is not None
        assert isinstance(result.error, RestrictedCodeError)
        assert "run_until_complete()" in str(result.error)
        assert "await" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_run_forever_blocked(self, test_agent):
        """loop.run_forever() should be blocked.

        This pattern would block the event loop indefinitely.
        """
        code = """
loop = asyncio.get_event_loop()
loop.run_forever()
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success, "run_forever() should be blocked in async context"
        assert result.error is not None
        assert isinstance(result.error, RestrictedCodeError)
        assert "run_forever()" in str(result.error)


class TestCorrectPatternsAllowed:
    """Tests that correct async patterns still work.

    These should pass both before and after the fix.
    """

    @pytest.mark.asyncio
    async def test_direct_await_works(self, test_agent):
        """Direct await is the correct pattern and should work."""
        code = """
async def my_func():
    await asyncio.sleep(0)
    return "done"

result = await my_func()
print(f"result={result}")
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert result.success
        assert "result=done" in result.stdout

    @pytest.mark.asyncio
    async def test_result_on_non_future_allowed(self, test_agent):
        """Calling .result() on non-Future objects should be allowed."""
        code = """
class MyResult:
    def result(self):
        return "not a future"

obj = MyResult()
print(obj.result())
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert result.success
        assert "not a future" in result.stdout


class TestRuntimePatchesCatchCrossBlock:
    """Tests that runtime patches catch cross-block Future.result() calls.

    These patterns can't be caught by static analysis because the Future is
    created in one code block and .result() is called in a later block.
    The scoped runtime patches catch these at execution time.
    """

    @pytest.mark.asyncio
    async def test_cross_block_future_result_blocked(self, test_agent):
        """Future.result() from a previous block should be blocked at runtime.

        Turn 1: Create a Future and store it on self
        Turn 2: Try to call .result() on it - should be blocked

        Note: We disable validation for this test since we're testing runtime
        patches, not the code validator. The validator blocks concurrent.futures
        imports by default.
        """
        # Turn 1: Create a future and store it (validation disabled)
        code1 = """
import concurrent.futures
# Create a pending future (not submitted to executor, stays pending)
self.stored_future = concurrent.futures.Future()
print("future created")
"""
        result1 = await test_agent.runtime.execute_code(code1, validate=False)
        assert result1.success, f"Turn 1 should succeed: {result1.error}"
        assert "future created" in result1.stdout

        # Turn 2: Try to access .result() - should be blocked by runtime patch
        code2 = """
result = self.stored_future.result()
print(f"got: {result}")
"""
        result2 = await test_agent.runtime.execute_code(code2, validate=False)

        assert not result2.success, "Cross-block Future.result() should be blocked"
        assert result2.error is not None
        assert "deadlock" in str(result2.error).lower() or "Future.result()" in str(result2.error)

    @pytest.mark.asyncio
    async def test_runtime_patches_dont_affect_tests(self):
        """Runtime patches should NOT affect code outside agent context.

        This verifies that test infrastructure can still use concurrent.futures
        without triggering the safety checks.
        """
        import concurrent.futures

        # This should work fine - we're not inside agent code execution
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(lambda: 42)
        result = future.result(timeout=5)
        executor.shutdown(wait=True)

        assert result == 42, "Test infrastructure should not be affected by patches"


class TestAsyncSafetyAliasing:
    """Tests that aliased imports are still caught.

    In-code imports (e.g., `import asyncio as aio`) are not in exec_globals
    at validation time, so the BlockingCallValidator cannot catch them.
    However, Python itself raises RuntimeError when asyncio.run() is called
    from a running event loop, so these patterns still fail at runtime.
    """

    @pytest.mark.asyncio
    async def test_aliased_asyncio_run_blocked(self, test_agent):
        """asyncio aliased as aio, then aio.run() should fail at runtime."""
        code = """
import asyncio as aio

async def inner():
    return 42

result = aio.run(inner())
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success, "Aliased asyncio.run() should fail"
        assert result.error is not None
        # May be caught by validator (RestrictedCodeError) or by Python runtime (RuntimeError)
        assert isinstance(result.error, (RestrictedCodeError, RuntimeError))
        assert "run" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_from_asyncio_import_run_blocked(self, test_agent):
        """from asyncio import run; run() should fail at runtime."""
        code = """
from asyncio import run

async def inner():
    return 42

result = run(inner())
"""
        result = await test_agent.runtime.execute_code(code, validate=True)

        assert not result.success, "Directly imported run() should fail"
        assert result.error is not None
        # May be caught by validator (RestrictedCodeError) or by Python runtime (RuntimeError)
        assert isinstance(result.error, (RestrictedCodeError, RuntimeError))


class TestAsyncSafetyExecutionCount:
    """Tests that execution_count parameter controls Cell In[N] name."""

    @pytest.mark.asyncio
    async def test_error_shows_correct_cell_number(self, test_agent):
        """Error messages should show the correct Cell In[N] based on execution_count param.

        Note: The runtime doesn't auto-increment execution_count - the caller (codeact)
        is responsible for passing the correct value. This test verifies the wiring
        from execute_code() -> validate_async_safety() -> error message works correctly.
        """
        code = """
asyncio.run(asyncio.sleep(0))
"""
        # Default execution_count is 1
        result1 = await test_agent.runtime.execute_code(code, validate=True)
        assert not result1.success
        assert "Cell In[1]" in str(result1.error)

        # Explicit execution_count
        result42 = await test_agent.runtime.execute_code(code, validate=True, execution_count=42)
        assert not result42.success
        assert "Cell In[42]" in str(result42.error)
