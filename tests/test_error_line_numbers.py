# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that error line numbers are correctly adjusted for wrapper offset.

This reproduces a bug where errors show wrong line numbers.
Example: error on user line 30 was showing as line 17.
"""

import re

import pytest

from nooa import Agent
from nooa.config import CodeActConfig
from nooa.errors.formatting import format_error_for_llm
from nooa.runtime.execution_namespace import RUNTIME_NAMES, ExecutionNamespace
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM
_TEST_LLM = FakeLLMClient()


def _namespace():
    """One namespace shared by a test's cells: the names a cell binds persist."""
    return ExecutionNamespace(reserved=RUNTIME_NAMES)


async def _cell(agent, namespace, code, **kwargs):
    """Run one direct cell in ``namespace`` (builtins seed only its first cell)."""
    with namespace.enter({}):
        return await agent.runtime.execute_code(
            code, wrap_in_function=True, execution_namespace=namespace, **kwargs
        )


@pytest.fixture
def test_agent():
    """Create a test agent."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    return TestAgent()


class TestErrorLineNumbers:
    """Test error line number accuracy through the full execution stack."""

    @pytest.mark.asyncio
    async def test_error_line_matches_user_code(self, test_agent):
        """Error line number should match the user's original code line."""
        # Code with error on a known line
        code = """\
line_1 = "first"
line_2 = "second"
line_3 = "third"
line_4 = "fourth"
line_5 = "fifth"
error_line = None.strip()
line_7 = "seventh"
"""
        # Line 6 is: error_line = None.strip()
        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None, "Expected an error"
        assert isinstance(result.error, AttributeError)

        # Format the error with the wrapper offset
        formatted = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

        print(f"\n=== Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result.wrapper_line_offset} ===")

        # The error should report line 6 (where error_line = None.strip() is)
        assert "line 6" in formatted, f"Expected 'line 6' in error, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_error_line_with_many_preceding_lines(self, test_agent):
        """Error on line 30 should show line 30, not a lower number."""
        # Build code with error on line 30
        lines = []
        for i in range(1, 30):
            lines.append(f'line_{i} = "value_{i}"')
        lines.append('error_line = "string".get("key")')  # Line 30 - AttributeError
        lines.append('line_31 = "after"')

        code = "\n".join(lines)

        result = await test_agent.runtime.execute_code(
            code, execution_count=3, wrap_in_function=True
        )

        assert result.error is not None, "Expected an error"

        formatted = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

        print(f"\n=== Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result.wrapper_line_offset} ===")
        print("\n=== Code lines 28-31 ===")
        for i, line in enumerate(code.split("\n")[27:31], 28):
            print(f"  {i}: {line}")

        # The error should report line 30
        assert "line 30" in formatted, f"Expected 'line 30' in error, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_error_line_with_globals(self, test_agent):
        """Error line numbers should be correct even with global declarations."""
        # First execution to set up some session state
        setup_code = """\
x = 10
y = 20
z = {"key": "value"}
"""
        namespace = _namespace()
        result1 = await _cell(test_agent, namespace, setup_code, execution_count=1)
        assert result1.error is None

        # Second execution with globals, error on line 5
        code = """\
a = x + y
b = z["key"]
c = "hello"
d = "world"
e = c.get("foo")
"""
        # Line 5 is: e = c.get("foo")
        result2 = await _cell(test_agent, namespace, code, execution_count=2)

        assert result2.error is not None, "Expected an error"

        formatted = format_error_for_llm(
            result2.error, code, line_offset=result2.wrapper_line_offset
        )

        print(f"\n=== Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result2.wrapper_line_offset} ===")

        # The error should report line 5
        assert "line 5" in formatted, f"Expected 'line 5' in error, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_wrapper_line_offset_is_zero(self, test_agent):
        """Cells compile from their own AST, so no wrapper line offset applies."""
        code = "x = 1 / 0"  # Line 1
        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None
        # The owned cell compiler keeps user line numbers: no wrapper header.
        assert result.wrapper_line_offset == 0

    @pytest.mark.asyncio
    async def test_unvalidated_syntax_error_uses_cell_filename(self, test_agent):
        """Parser failures identify the execution cell, not ``<unknown>``."""
        code = "value = (1 + )"

        result = await test_agent.runtime.execute_code(
            code,
            execution_count=17,
            validate=False,
            wrap_in_function=True,
        )

        assert isinstance(result.error, SyntaxError)
        assert result.error.filename.startswith("Cell In[17] [ns:")
        formatted = format_error_for_llm(result.error, code)
        # The model sees the IPython name; the namespace id stays internal.
        assert "Cell In[17], line 1" in formatted and "[ns:" not in formatted
        assert "<unknown>" not in formatted

    @pytest.mark.asyncio
    async def test_validated_syntax_error_identifies_cell(self, test_agent):
        """Validation errors returned by the runtime identify the cell and line."""
        from nooa.errors import RestrictedCodeError

        code = "first = 1\nsecond = 2\nvalue = (1 + 2"

        result = await test_agent.runtime.execute_code(
            code, execution_count=42, wrap_in_function=True
        )

        assert isinstance(result.error, RestrictedCodeError)
        message = str(result.error)
        assert "Cell In[42], line 3" in message, f"Expected cell and line, got:\n{message}"
        assert "<unknown>" not in message, f"Cell location missing:\n{message}"

    @pytest.mark.asyncio
    async def test_syntax_error_line_number(self, test_agent):
        """Syntax errors should also have correct line numbers."""
        code = """\
x = 1
y = 2
z = 3
if True
    pass
"""
        # Line 4 is: if True (missing colon)
        # Note: syntax errors are caught by validation and become RestrictedCodeError
        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None
        # Syntax errors become RestrictedCodeError after validation
        from nooa.errors import RestrictedCodeError

        assert isinstance(result.error, (SyntaxError, RestrictedCodeError))

        formatted = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

        print(f"\n=== Formatted error ===\n{formatted}")

        # Syntax error should mention line 4
        assert "line 4" in formatted.lower(), f"Expected 'line 4' in error, got:\n{formatted}"


class TestMultipleExecutions:
    """Test error line numbers across multiple executions (like evaluation)."""

    @pytest.mark.asyncio
    async def test_error_after_multiple_executions(self, test_agent):
        """Simulate τ-bench evaluation: multiple execs before error."""
        # Execution 1: set up user_id
        code1 = "user_id = 'yusuf_rossi_123'"
        namespace = _namespace()
        result1 = await _cell(test_agent, namespace, code1, execution_count=1)
        assert result1.error is None

        # Execution 2: set up product_types
        code2 = "product_types = ['tshirt', 'headphones']"
        result2 = await _cell(test_agent, namespace, code2, execution_count=2)
        assert result2.error is None

        # Execution 3: long code with error on line 30
        lines = []
        for i in range(1, 30):
            lines.append(f'var_{i} = "value_{i}"')
        lines.append('error_line = "string".get("key")')  # Line 30
        lines.append('var_31 = "after"')
        code3 = "\n".join(lines)

        result3 = await _cell(test_agent, namespace, code3, execution_count=3)

        assert result3.error is not None

        formatted = format_error_for_llm(
            result3.error, code3, line_offset=result3.wrapper_line_offset
        )

        print(f"\n=== Execution 3: Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result3.wrapper_line_offset} ===")
        print(f"\n=== Names bound so far: {sorted(namespace.user_values())} ===")

        # The error should still report line 30
        assert "line 30" in formatted, f"Expected 'line 30' in error, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_error_line_number_consistency(self, test_agent):
        """Test that raw traceback line matches formatted line after adjustment."""
        import traceback as tb_module

        code = """\
a = 1
b = 2
c = 3
d = 4
e = 5
error = None.strip()
g = 7
"""
        # Line 6 has the error
        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None
        error = result.error

        # Get raw traceback line number
        raw_lines = tb_module.format_exception(type(error), error, error.__traceback__)
        raw_text = "".join(raw_lines)

        print(f"\n=== Raw traceback ===\n{raw_text}")

        # Get formatted output
        formatted = format_error_for_llm(error, code, line_offset=result.wrapper_line_offset)

        print(f"\n=== Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result.wrapper_line_offset} ===")

        # Both should show line 6
        assert "line 6" in formatted.lower(), f"Formatted should show line 6:\n{formatted}"


class TestOffsetClearing:
    """Test that wrapper_line_offset is correctly calculated per-execution."""

    @pytest.mark.asyncio
    async def test_offset_does_not_accumulate(self, test_agent):
        """wrapper_line_offset should be calculated fresh each execution, not accumulated."""
        offsets = []

        # Run 5 executions with varying globals
        namespace = _namespace()
        for i in range(1, 6):
            code = f"var_{i} = 'value_{i}'"
            result = await _cell(test_agent, namespace, code, execution_count=i)
            assert result.error is None
            offsets.append(result.wrapper_line_offset)

        # Offset should be 2 (no globals) or 3 (with globals), never growing unbounded
        for i, offset in enumerate(offsets, 1):
            assert offset <= 3, f"Execution {i} has offset {offset} - should be <= 3"

    @pytest.mark.asyncio
    async def test_error_line_correct_across_many_executions(self, test_agent):
        """Error line numbers should be correct even after many prior executions."""
        namespace = _namespace()

        # Run 10 successful executions to build up state
        for i in range(1, 11):
            code = f"var_{i} = 'value_{i}'"
            result = await _cell(test_agent, namespace, code, execution_count=i)
            assert result.error is None

        print(f"After 10 executions, {len(namespace.user_values())} globals")

        # Now execution 11 with error on a specific line
        error_code = """\
line_1 = "first"
line_2 = "second"
line_3 = "third"
line_4 = "fourth"
error_line = "string".get("missing")
line_6 = "sixth"
"""
        result = await _cell(test_agent, namespace, error_code, execution_count=11)

        assert result.error is not None
        print(f"Offset after 10 prior execs: {result.wrapper_line_offset}")

        formatted = format_error_for_llm(
            result.error, error_code, line_offset=result.wrapper_line_offset
        )
        print(f"Formatted error:\n{formatted}")

        # Error should be on line 5, not some accumulated number
        assert "line 5" in formatted, f"Expected 'line 5' in error, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_linecache_isolation_between_cells(self, test_agent):
        """Each Cell In[N] should have its own linecache entry."""
        import linecache

        # Clear any existing entries
        for key in list(linecache.cache.keys()):
            if key.startswith("Cell In["):
                del linecache.cache[key]

        # Execute code in Cell In[1]
        code1 = "x = 1\ny = 2"
        await test_agent.runtime.execute_code(code1, execution_count=1, wrap_in_function=True)

        # Execute code in Cell In[2]
        code2 = "a = 10\nb = 20\nc = 30"
        await test_agent.runtime.execute_code(code2, execution_count=2, wrap_in_function=True)

        # Verify each has its own entry (keyed by cell and owning namespace)
        def entry(prefix):
            keys = [key for key in linecache.cache if key.startswith(prefix)]
            assert len(keys) == 1, f"{prefix} should be in linecache once: {keys}"
            return linecache.cache[keys[0]][2]

        assert entry("Cell In[1] [ns:") == ["x = 1\n", "y = 2"]
        assert entry("Cell In[2] [ns:") == ["a = 10\n", "b = 20\n", "c = 30"]


class TestExactTraceScenario:
    """Reproduce the exact scenario from the problematic trace."""

    @pytest.mark.asyncio
    async def test_exact_trace_scenario(self, test_agent):
        """Reproduce: exec 3 shows line 2 but error is on line 3."""
        import linecache

        # Execution 1: like the trace
        code1 = """user_id = await self.find_user('test')
order_details = {'id': 123}
description = 'test order'"""

        # Mock self.find_user
        async def mock_find_user(name):
            return "user_123"

        builtins1 = {"self": type("MockSelf", (), {"find_user": mock_find_user})()}
        namespace = _namespace()
        result1 = await _cell(test_agent, namespace, code1, execution_count=1, builtins=builtins1)
        print(f"Exec 1: offset={result1.wrapper_line_offset}")

        # Execution 2
        code2 = """keyboard_product_id = "1656367028"
thermostat_product_id = "4896585277"

keyboard_details = "some_string"
thermostat_details = "some_string" """

        result2 = await _cell(test_agent, namespace, code2, execution_count=2)
        print(f"Exec 2: offset={result2.wrapper_line_offset}")

        # Execution 3: error on line 3
        code3 = """# Find available keyboard
preferred_keyboard_item_id = None
for variant_id, variant in get_product_details("123")["variants"].items():
    if variant["available"]:
        preferred_keyboard_item_id = variant["item_id"]
        break"""

        result3 = await _cell(test_agent, namespace, code3, execution_count=3)

        print(f"\nExec 3: offset={result3.wrapper_line_offset}")
        print(f"Error: {result3.error}")

        # Check raw traceback
        if result3.error and result3.error.__traceback__:
            tb = result3.error.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            raw_line = tb.tb_lineno
            print(f"Raw traceback line: {raw_line}")
            print(f"Adjusted line: {raw_line - result3.wrapper_line_offset}")

        # Check linecache content
        cell_key = "Cell In[3]"
        if cell_key in linecache.cache:
            lines = linecache.cache[cell_key][2]
            print(f"\nLinecache {cell_key} lines:")
            for i, line in enumerate(lines[:10], 1):
                print(f"  {i}: {line.rstrip()}")

        # Format error
        formatted = format_error_for_llm(
            result3.error, code3, line_offset=result3.wrapper_line_offset
        )
        print(f"\nFormatted error:\n{formatted}")

        # The error is on line 3 (the 'for' statement)
        assert "line 3" in formatted, f"Expected 'line 3', got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_error_traceback_line_is_user_line(self, test_agent):
        """The raw traceback line is the user's line: the cell keeps its own AST."""
        code = """line_1 = "first"
line_2 = "second"
line_3 = "third"
error = None.strip()
line_5 = "fifth" """

        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        tb = result.error.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        assert tb.tb_lineno == 4, "Error should be on user line 4"


class TestCodeActStrategyErrorFormatting:
    """Test error formatting through the CodeActStrategy."""

    @pytest.mark.asyncio
    async def test_codeact_error_line_numbers(self, test_agent):
        """Test that CodeActStrategy formats errors with correct line numbers."""
        from nooa.strategies.codeact import CodeActStrategy

        # Create strategy instance directly
        strat = CodeActStrategy(config=CodeActConfig())

        # Execute some code to set up state
        namespace = _namespace()
        await _cell(test_agent, namespace, "x = 1\ny = 2", execution_count=1)

        # Execute code with error on line 4
        code_with_error = """a = x + y
b = "string"
c = "another"
error = b.get("key")
e = "never reached" """

        result2 = await _cell(test_agent, namespace, code_with_error, execution_count=2)

        assert result2.error is not None

        # Format error using strategy's _format_error method
        formatted = strat._format_error(
            result2.error, code_with_error, line_offset=result2.wrapper_line_offset
        )

        print("\n=== CodeActStrategy formatted result ===")
        print(formatted)

        # Error should be on line 4
        assert "line 4" in formatted, f"Expected 'line 4' in:\n{formatted}"


class TestCommentPreservation:
    """Test that comments don't affect line number accuracy."""

    @pytest.mark.asyncio
    async def test_error_line_with_comments(self, test_agent):
        """Comments should not affect line number calculation."""
        # Code with comments - error is on line 4
        code = """\
# This is a comment on line 1
x = 1  # Comment on line 2
# Another comment on line 3
error = None.strip()  # Error on line 4
y = 2  # Line 5
"""
        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None

        formatted = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

        print("\n=== Code with comments ===")
        for i, line in enumerate(code.split("\n"), 1):
            print(f"  {i}: {line}")

        print("\n=== Formatted error ===")
        print(formatted)

        # Error should be on line 4, not affected by comments
        assert "line 4" in formatted, f"Expected 'line 4' with comments, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_implicit_return_preserves_line_numbers(self, test_agent):
        """Implicit return transformation should not change line numbers."""
        # Code with implicit return on last line - error on line 3
        code = """\
# Comment
x = "string"
y = x.get("missing")  # Error on line 3
doc(self)  # This gets implicit return added"""

        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None

        formatted = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

        print("\n=== Code with implicit return ===")
        for i, line in enumerate(code.split("\n"), 1):
            print(f"  {i}: {line}")

        print("\n=== Formatted error ===")
        print(formatted)

        # Error should be on line 3
        assert "line 3" in formatted, f"Expected 'line 3' with implicit return, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_multiline_expression_line_numbers(self, test_agent):
        """Multi-line expressions should report correct starting line."""
        code = """\
x = 1
y = (
    "string"
    .get("key")
)
z = 3"""

        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None

        formatted = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

        print("\n=== Formatted error ===")
        print(formatted)

        # Error should be on line 2 (start of the multi-line expression) or line 4 (the .get call)
        # Python reports the line where the error actually occurs
        assert "line 4" in formatted or "line 2" in formatted, (
            f"Expected 'line 2' or 'line 4' for multi-line, got:\n{formatted}"
        )


class TestWrappedBaseExceptionTraceback:
    """Wrapped process-control exceptions retain their generated-cell context."""

    @pytest.mark.asyncio
    async def test_actor_system_exit_keeps_adjusted_source_context(self, test_agent):
        source = "marker = 1\nexit_type = SystemExit\nraise exit_type('bye')"
        result = await test_agent.runtime.execute_code(
            source,
            execution_count=70,
            wrap_in_function=True,
        )

        assert result.error is not None
        formatted = format_error_for_llm(
            result.error,
            source,
            line_offset=result.wrapper_line_offset,
        )
        assert "Cell In[70], line 3, in <module>" in formatted
        assert "[ns:" not in formatted and "_nooa_ns_" not in formatted
        assert "raise exit_type('bye')" in formatted
        assert "SystemExit: bye" in formatted
        assert "direct cause" in formatted
        assert formatted.endswith(
            "RuntimeError: SystemExit raised inside generated code. Do not use raise "
            "SystemExit / sys.exit() / exit() / quit() to stop a cell — use break, "
            "a flag, a helper return, or return_result()."
        )


class TestPersistedHelperTraceback:
    """Persisted helper frames retain their own source-relative locations."""

    @pytest.mark.asyncio
    async def test_earlier_helper_uses_original_source_and_line_number(self, test_agent):
        helper_source = """def boom():
    marker = "helper"
    raise ValueError("x")
"""
        first = _namespace()
        defined = await _cell(test_agent, first, helper_source, execution_count=71)
        assert defined.error is None

        # The helper keeps its own cell's source and line numbers in a later
        # cell of another namespace.
        failed = await _cell(
            test_agent,
            _namespace(),
            "boom()",
            execution_count=73,
            builtins={"boom": first.globals["boom"]},
        )

        assert failed.error is not None
        formatted = format_error_for_llm(
            failed.error,
            "boom()",
            line_offset=failed.wrapper_line_offset,
        )
        assert "Cell In[73], line 1, in <module>" in formatted
        assert "Cell In[71], line 3, in boom" in formatted
        assert "[ns:" not in formatted and "_nooa_ns_" not in formatted
        assert 'raise ValueError("x")' in formatted


class TestFormatterDirectly:
    """Test the formatter logic directly without execution."""

    def test_adjust_line_numbers_subtracts_offset(self):
        """_adjust_line_numbers should subtract the offset."""
        from nooa.errors.formatting import _adjust_line_numbers

        text = "Cell In[1], line 33, in <module>"
        result = _adjust_line_numbers(text, offset=3)
        assert result == "Cell In[1], line 30, in <module>"

    def test_adjust_line_numbers_never_below_1(self):
        """Line numbers should never go below 1."""
        from nooa.errors.formatting import _adjust_line_numbers

        text = "Cell In[1], line 2, in <module>"
        result = _adjust_line_numbers(text, offset=5)
        assert result == "Cell In[1], line 1, in <module>"

    def test_wrapper_names_replaced(self):
        """__repl_wrapper__ should be replaced with <module>."""
        from nooa.errors.formatting import _replace_wrapper_names

        text = "Cell In[1], line 5, in __repl_wrapper__"
        result = _replace_wrapper_names(text)
        assert result == "Cell In[1], line 5, in <module>"


class TestModelVisibleCellNames:
    """Tracebacks the model sees name cells as IPython does, for every strategy."""

    @pytest.mark.asyncio
    async def test_runtime_error_names_the_cell_and_module(self, test_agent):
        code = "a = 1\nb = 2\nc = a.missing"
        result = await _cell(test_agent, _namespace(), code, execution_count=5)
        formatted = format_error_for_llm(result.error, code)
        assert "Cell In[5], line 3, in <module>" in formatted
        assert "c = a.missing" in formatted
        assert "[ns:" not in formatted and "_nooa_ns_" not in formatted

    @pytest.mark.asyncio
    async def test_syntax_error_names_the_cell(self, test_agent):
        code = "a = 1\nb = (2 +"
        result = await _cell(test_agent, _namespace(), code, execution_count=6, validate=False)
        formatted = format_error_for_llm(result.error, code)
        assert "Cell In[6], line 2" in formatted
        assert "[ns:" not in formatted

    @pytest.mark.asyncio
    async def test_pure_python_error_feedback_has_no_internal_names(self):
        from nooa import strategy
        from nooa.events import PythonOutput, ResultStatus
        from nooa.strategies.pure_python import PurePythonStrategy

        class PPAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_retries=2))
            async def go(self) -> int:
                """Return a number."""
                ...

        llm = FakeLLMClient.with_code_responses(["x = 1\ny = x.missing", "return 3"])
        agent = PPAgent(llm=llm)
        assert await agent.go() == 3
        error = next(
            event.error
            for event in agent.event_manager.values()
            if isinstance(event, PythonOutput) and event.execution_status is ResultStatus.ERROR
        )
        assert re.search(r"Cell In\[\d+\], line 2, in <module>", error), error
        assert "[ns:" not in error and "_nooa_ns_" not in error
