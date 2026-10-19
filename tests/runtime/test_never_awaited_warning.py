# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cell's "coroutine ... was never awaited" warning becomes a one-line hint (#447)."""

from nooa.runtime.actor import _rewrite_never_awaited_warnings


class ShellTools:
    async def read(self, path):
        return path


class MyAgent:
    def __init__(self):
        self.shell = ShellTools()

    async def delegate(self, prompt):
        return prompt


STOCK = (
    "Cell In[3]:3: RuntimeWarning: coroutine 'ShellTools.read' was never awaited\n"
    "RuntimeWarning: Enable tracemalloc to get the object allocation traceback\n"
)


def test_names_the_agent_call_and_drops_the_tracemalloc_line():
    assert _rewrite_never_awaited_warnings(STOCK, MyAgent()) == (
        "`ShellTools.read(...)` was called without await and never ran; "
        "use `await self.shell.read(...)`\n"
    )


def test_a_wrapper_that_copies_the_method_name_is_found():
    import functools

    class ActivityShell:
        @functools.wraps(ShellTools.read)
        async def read(self, path):
            return path

    class WrappedAgent:
        def __init__(self):
            self.shell = ActivityShell()

    assert _rewrite_never_awaited_warnings(STOCK, WrappedAgent()) == (
        "`ShellTools.read(...)` was called without await and never ran; "
        "use `await self.shell.read(...)`\n"
    )


def test_source_line_and_agent_methods():
    text = (
        "Cell In[4]:1: RuntimeWarning: coroutine 'MyAgent.delegate' was never awaited\n"
        '  self.delegate("x")\n'
        "RuntimeWarning: Enable tracemalloc to get the object allocation traceback\n"
    )
    assert _rewrite_never_awaited_warnings(text, MyAgent()) == (
        "`MyAgent.delegate(...)` was called without await and never ran; "
        "use `await self.delegate(...)`\n"
    )


def test_unknown_owner_still_gets_the_fix():
    text = "x.py:1: RuntimeWarning: coroutine 'fetch' was never awaited\n"
    assert _rewrite_never_awaited_warnings(text, MyAgent()) == (
        "`fetch(...)` was called without await and never ran; put `await` before the call\n"
    )


def test_other_warnings_and_text_are_unchanged():
    other = (
        "before\n"
        "Cell In[2]:1: SyntaxWarning: invalid escape sequence '\\s'\n"
        "Cell In[2]:2: RuntimeWarning: divide by zero encountered\n"
        "  x = 1 / 0\n"
    )
    assert _rewrite_never_awaited_warnings(other, MyAgent()) == other
    assert _rewrite_never_awaited_warnings(other + STOCK, MyAgent()) == other + (
        "`ShellTools.read(...)` was called without await and never ran; "
        "use `await self.shell.read(...)`\n"
    )


# --- #447 review finding 4: one rewrite in execute_code covers every backend ---


def _print_warnings_to_stderr():
    """pytest records warnings instead of printing them; print them to stderr as an
    uncaptured process does, so they reach the cell's captured stderr."""
    import contextlib
    import sys
    import warnings

    @contextlib.contextmanager
    def manager():
        with warnings.catch_warnings():
            warnings.simplefilter("always", RuntimeWarning)
            warnings.showwarning = warnings._showwarning_orig  # type: ignore[attr-defined]
            original = warnings._showwarnmsg_impl  # type: ignore[attr-defined]
            warnings._showwarnmsg_impl = lambda msg: sys.stderr.write(  # type: ignore[attr-defined]
                warnings._formatwarnmsg(msg)  # type: ignore[attr-defined]
            )
            try:
                yield
            finally:
                warnings._showwarnmsg_impl = original  # type: ignore[attr-defined]

    return manager()


RAW_SANDBOX_STDERR = (
    "Cell In[1]:6: RuntimeWarning: coroutine '__repl_wrapper__.<locals>.fetch' was never awaited\n"
    "  fetch()\n"
    "RuntimeWarning: Enable tracemalloc to get the object allocation traceback\n"
)


async def test_sandbox_backend_result_stderr_is_rewritten():
    """execute_code(sandbox_executor=...) returns early with the worker's result;
    its stderr gets the same one-line hint as the in-process backend."""
    import asyncio

    import pytest

    from nooa import Agent
    from nooa.events import ExecutionResult
    from nooa.unifiedllm import FakeLLMClient

    class SandboxAgent(Agent, llm=FakeLLMClient()):
        pass

    class FakeSandbox:
        async def run_cell(self, code: str, *, execution_count: int = 1):
            return ExecutionResult(stdout="done\n", stderr=RAW_SANDBOX_STDERR)

    class CancelledSandbox:
        async def run_cell(self, code: str, *, execution_count: int = 1):
            raise asyncio.CancelledError

    agent = SandboxAgent()
    result = await agent.runtime.execute_code("fetch()", sandbox_executor=FakeSandbox())
    assert result.stdout == "done\n"
    assert result.stderr == (
        "`__repl_wrapper__.<locals>.fetch(...)` was called without await and never ran; "
        "put `await` before the call\n"
    )
    with pytest.raises(asyncio.CancelledError):
        await agent.runtime.execute_code("pass", sandbox_executor=CancelledSandbox())


async def test_in_process_backend_result_stderr_is_rewritten():
    from nooa import Agent
    from nooa.unifiedllm import FakeLLMClient

    class LocalAgent(Agent, llm=FakeLLMClient()):
        pass

    agent = LocalAgent()
    with _print_warnings_to_stderr():
        result = await agent.runtime.execute_code(
            "async def fetch():\n    return 1\n\nfetch()\nprint('done')", wrap_in_function=True
        )
    assert result.stdout == "done\n"
    assert "fetch(...)` was called without await and never ran" in result.stderr, result.stderr
    assert "tracemalloc" not in result.stderr


async def test_cancelled_cell_partial_stderr_is_rewritten():
    import asyncio

    import pytest

    from nooa import Agent
    from nooa.unifiedllm import FakeLLMClient

    class LocalAgent(Agent, llm=FakeLLMClient()):
        pass

    agent = LocalAgent()
    code = "async def fetch():\n    return 1\n\nfetch()\nraise asyncio.CancelledError()"
    with _print_warnings_to_stderr(), pytest.raises(asyncio.CancelledError) as info:
        await agent.runtime.execute_code(code, wrap_in_function=True)
    partial = info.value.execution_results[-1]
    assert "fetch(...)` was called without await and never ran" in partial.stderr, partial.stderr
    assert "tracemalloc" not in partial.stderr
