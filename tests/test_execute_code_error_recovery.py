# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for IPython-style error recovery in code execution."""

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


async def _execute(runtime, code, **kwargs):
    """Run ``code`` in a fresh namespace; return the result and the names it bound."""
    namespace = ExecutionNamespace(reserved=RUNTIME_NAMES)
    with namespace.enter({}):
        result = await runtime.execute_code(code, execution_namespace=namespace, **kwargs)
    bound = namespace.user_names
    return result, {name: namespace.globals[name] for name in bound if name in namespace.globals}


@pytest.mark.asyncio
async def test_captured_locals_preserved_on_error():
    """Variables defined before an error should be preserved (IPython behavior)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code that defines variables, then raises an error
    code = """
x = 1
y = 2
raise Exception('test error')
z = 3  # This won't be reached
"""

    result, bound = await _execute(runtime, code, wrap_in_function=True)

    # Should have an error
    assert result.error is not None
    assert "test error" in str(result.error)

    # But x and y should be captured (IPython behavior)
    assert "x" in bound, "x should be captured before error"
    assert "y" in bound, "y should be captured before error"
    assert bound["x"] == 1
    assert bound["y"] == 2

    # z should NOT be captured (never executed)
    assert "z" not in bound


@pytest.mark.asyncio
async def test_objects_with_internal_state_preserved_on_error():
    """Objects with internal state should be preserved with their state intact."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code that creates objects with internal state, modifies them, then errors
    code = """
# List with accumulated state
items = []
items.append(1)
items.append(2)

# Dict with state
config = {"count": 0}
config["count"] += 10
config["name"] = "test"

# Class instance with state
class Counter:
    def __init__(self):
        self.value = 0
    def inc(self):
        self.value += 1
        return self.value

counter = Counter()
counter.inc()
counter.inc()

raise Exception('error after state mutations')
"""

    result, bound = await _execute(runtime, code, wrap_in_function=True)

    # Should have an error
    assert result.error is not None

    # List should have accumulated state
    assert "items" in bound
    assert bound["items"] == [1, 2]

    # Dict should have modified state
    assert "config" in bound
    assert bound["config"]["count"] == 10
    assert bound["config"]["name"] == "test"

    # Class instance should have its state
    assert "counter" in bound
    assert bound["counter"].value == 2


@pytest.mark.asyncio
async def test_captured_locals_preserved_on_execution_signal():
    """Variables should be preserved when ExecutionSignal is raised."""
    from nooa.events import ExecutionSignal

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code that defines variables, then raises an ExecutionSignal
    # (This simulates what return_result() does internally)
    code = """
x = 42
y = "hello"
raise ExecutionSignal("result", x + 1)
z = 99  # This won't be reached
"""

    # Inject ExecutionSignal into builtins so it's available in the executed code
    result, bound = await _execute(
        runtime, code, wrap_in_function=True, builtins={"ExecutionSignal": ExecutionSignal}
    )

    # Should have a signal, not an error
    assert result.error is None
    assert result.signal is not None

    # x and y should be captured
    assert "x" in bound, "x should be captured before signal"
    assert "y" in bound, "y should be captured before signal"
    assert bound["x"] == 42
    assert bound["y"] == "hello"

    # z should NOT be captured (never executed)
    assert "z" not in bound


@pytest.mark.asyncio
async def test_captured_locals_preserved_with_return_result():
    """Variables should be preserved when return_result() is called (full integration)."""
    from nooa.events import ExecutionSignal

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Create return_result function that mirrors the real implementation
    def return_result(value):
        """Simulates the real return_result() from CodeActStrategy."""
        raise ExecutionSignal("return_result", {"result": value})

    code = """
x = 42
y = "computed"
result = x * 2
return_result(result)
z = 99  # This won't be reached
"""

    result, bound = await _execute(
        runtime,
        code,
        wrap_in_function=True,
        builtins={"return_result": return_result, "ExecutionSignal": ExecutionSignal},
    )

    # Should have a signal, not an error
    assert result.error is None
    assert result.signal is not None

    # All variables before return_result should be captured
    assert "x" in bound
    assert "y" in bound
    assert "result" in bound
    assert bound["x"] == 42
    assert bound["y"] == "computed"
    assert bound["result"] == 84

    # z should NOT be captured (never executed)
    assert "z" not in bound


@pytest.mark.asyncio
async def test_captured_locals_preserved_on_timeout():
    """Variables should be preserved when code execution times out."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code that defines variables, then enters an infinite loop
    code = """
import asyncio
x = 1
y = 2
await asyncio.sleep(10)  # Will timeout
z = 3
"""

    result, bound = await _execute(runtime, code, wrap_in_function=True, timeout=0.1)

    # Should have a timeout error
    assert result.error is not None
    assert isinstance(result.error, TimeoutError)

    # x and y should be captured before timeout
    # Note: This depends on when exactly the timeout fires - variables assigned
    # before the blocking call should be captured
    assert "x" in bound, "x should be captured before timeout"
    assert "y" in bound, "y should be captured before timeout"
    assert bound["x"] == 1
    assert bound["y"] == 2


@pytest.mark.asyncio
async def test_captured_locals_preserved_on_nested_exception():
    """Variables should be preserved when exception is caught and re-raised."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code that catches and re-raises an exception
    code = """
x = 1
y = 2
try:
    z = 3
    raise ValueError("inner error")
except ValueError:
    w = 4  # This executes in the except block
    raise  # Re-raise the exception
final = 5  # Never reached
"""

    result, bound = await _execute(runtime, code, wrap_in_function=True)

    # Should have an error
    assert result.error is not None
    assert "inner error" in str(result.error)

    # All variables defined before re-raise should be captured
    assert "x" in bound
    assert "y" in bound
    assert "z" in bound
    assert "w" in bound
    assert bound["x"] == 1
    assert bound["y"] == 2
    assert bound["z"] == 3
    assert bound["w"] == 4

    # final should NOT be captured (never executed)
    assert "final" not in bound


@pytest.mark.asyncio
async def test_captured_locals_preserved_on_exception_in_finally():
    """Variables should be preserved when finally block raises an exception."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code with exception in finally block
    code = """
x = 1
y = 2
try:
    z = 3
finally:
    w = 4  # This executes
    raise RuntimeError("finally error")
after = 5  # Never reached
"""

    result, bound = await _execute(runtime, code, wrap_in_function=True)

    # Should have the finally error (it overrides any previous exception)
    assert result.error is not None
    assert "finally error" in str(result.error)

    # All variables including those in finally should be captured
    assert "x" in bound
    assert "y" in bound
    assert "z" in bound
    assert "w" in bound
    assert bound["x"] == 1
    assert bound["y"] == 2
    assert bound["z"] == 3
    assert bound["w"] == 4

    # after should NOT be captured
    assert "after" not in bound


@pytest.mark.asyncio
async def test_cancelled_error_propagates():
    """CancelledError propagates without being caught (expected Python behavior).

    asyncio.CancelledError inherits from BaseException, not Exception, so it
    intentionally propagates for proper task cancellation semantics. This means
    captured_locals won't be extracted for CancelledError - this is expected.
    """
    import asyncio

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent = TestAgent()
    runtime = ActorRuntime(agent)

    # Code that raises CancelledError (simulating task cancellation)
    code = """
import asyncio
x = 1
y = 2
raise asyncio.CancelledError("task cancelled")
z = 3
"""

    # CancelledError should propagate (not be caught and returned as result.error)
    with pytest.raises(asyncio.CancelledError):
        await runtime.execute_code(code, wrap_in_function=True)


@pytest.mark.asyncio
async def test_in_process_results_leave_captured_locals_empty():
    """The namespace holds what a cell bound, never a name it did not assign."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    runtime = ActorRuntime(TestAgent())
    result, bound = await _execute(runtime, "x = 1\nif False:\n    y = 2", wrap_in_function=True)

    assert result.error is None
    assert result.captured_locals == {}  # the field stays for compatibility, unfilled
    assert bound == {"x": 1}
