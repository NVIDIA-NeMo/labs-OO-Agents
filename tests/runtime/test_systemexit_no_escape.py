# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: a generated cell must not be able to terminate the parent flow.

Defense in depth against a CodeAct cell using ``raise SystemExit`` / ``sys.exit()``
(observed cancelling sibling Harbor trials through a surrounding
``asyncio.TaskGroup``):

1. AST validation rejects the *literal* forms up front (see
   ``test_code_validator.py::TestSecurityValidatorProcessTermination``).
2. The runtime backstop in ``execute_code`` catches any ``SystemExit`` /
   ``KeyboardInterrupt`` that still reaches it (indirect forms that AST can't
   see) and converts it to an ordinary execution error.

``asyncio.CancelledError`` must still propagate so real cancellation keeps working.
"""

import asyncio

import pytest

from nooa import Agent
from nooa.errors import RestrictedCodeError
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


@pytest.fixture
def test_agent():
    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    return TestAgent()


class TestLiteralFormsRejectedAtValidation:
    """The literal process-termination forms are rejected before execution."""

    @pytest.mark.asyncio
    async def test_raise_systemexit_rejected(self, test_agent):
        """Bare ``raise SystemExit`` is rejected by static validation."""
        result = await test_agent.runtime.execute_code("raise SystemExit")
        assert not result.success
        assert isinstance(result.error, RestrictedCodeError)

    @pytest.mark.asyncio
    async def test_raise_systemexit_with_code_rejected(self, test_agent):
        """``raise SystemExit(code)`` is rejected before runtime execution."""
        result = await test_agent.runtime.execute_code("raise SystemExit(3)")
        assert not result.success
        assert isinstance(result.error, RestrictedCodeError)

    @pytest.mark.asyncio
    async def test_raise_keyboardinterrupt_rejected(self, test_agent):
        """Bare ``raise KeyboardInterrupt`` is rejected by static validation."""
        result = await test_agent.runtime.execute_code("raise KeyboardInterrupt")
        assert not result.success
        assert isinstance(result.error, RestrictedCodeError)


class TestRuntimeBackstop:
    """Indirect forms bypass AST validation; the runtime catch is the backstop.

    ExecutionResult.error is typed Exception | None, so the BaseException is
    wrapped in a RuntimeError with the original kept as __cause__.
    """

    @pytest.mark.asyncio
    async def test_aliased_systemexit_becomes_error(self, test_agent):
        """An indirect ``SystemExit`` is converted to an execution error."""
        # `e = SystemExit; raise e` is not the literal `raise SystemExit`, so it
        # bypasses AST validation and must be caught at runtime.
        result = await test_agent.runtime.execute_code("e = SystemExit\nraise e")
        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert isinstance(result.error.__cause__, SystemExit)

    @pytest.mark.asyncio
    async def test_systemexit_via_validate_false(self, test_agent):
        """The runtime backstop catches ``SystemExit`` when validation is skipped."""
        # Explicitly skip validation to exercise the runtime backstop directly.
        result = await test_agent.runtime.execute_code("raise SystemExit", validate=False)
        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert isinstance(result.error.__cause__, SystemExit)

    @pytest.mark.asyncio
    async def test_keyboardinterrupt_via_validate_false(self, test_agent):
        """The runtime backstop catches ``KeyboardInterrupt`` when validation is skipped."""
        result = await test_agent.runtime.execute_code("raise KeyboardInterrupt", validate=False)
        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert isinstance(result.error.__cause__, KeyboardInterrupt)

    @pytest.mark.asyncio
    async def test_backstop_preserves_prior_stdout(self, test_agent):
        """Output emitted before ``SystemExit`` remains visible in the result."""
        result = await test_agent.runtime.execute_code(
            "print('scanned')\nraise SystemExit", validate=False
        )
        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert "scanned" in result.stdout

    @pytest.mark.asyncio
    async def test_sibling_task_survives_systemexit_cell(self, test_agent):
        """The motivating bug: a SystemExit cell must not cancel a sibling task."""
        sibling_done = False

        async def sibling():
            nonlocal sibling_done
            await asyncio.sleep(0.05)
            sibling_done = True

        async with asyncio.TaskGroup() as tg:
            tg.create_task(sibling())
            result = await test_agent.runtime.execute_code("raise SystemExit", validate=False)

        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert sibling_done is True


class TestReturnResultStillPropagatesAsSignal:
    """Control-flow signals remain separate from process-termination errors."""

    @pytest.mark.asyncio
    async def test_return_result_execution_signal_is_not_wrapped(self, test_agent):
        """``return_result()`` uses ExecutionSignal and must not become RuntimeError."""
        from nooa.events import ExecutionSignal

        def return_result(value):
            raise ExecutionSignal("return_result", {"result": value})

        result = await test_agent.runtime.execute_code(
            "answer = 41 + 1\nreturn_result(answer)",
            wrap_in_function=True,
            validate=False,
            builtins={"return_result": return_result},
        )

        assert result.error is None
        assert result.signal is not None
        assert isinstance(result.signal, ExecutionSignal)
        assert result.captured_locals["answer"] == 42


class TestCancellationStillPropagates:
    """asyncio.CancelledError must NOT be swallowed by the backstop."""

    @pytest.mark.asyncio
    async def test_cancellederror_is_not_caught(self, test_agent):
        """Real task cancellation still propagates through ``execute_code``."""
        code = "import asyncio\nraise asyncio.CancelledError"
        with pytest.raises(asyncio.CancelledError):
            await test_agent.runtime.execute_code(code, validate=False)
