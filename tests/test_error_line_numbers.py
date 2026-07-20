# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that error line numbers are correctly adjusted for wrapper offset.

This reproduces a bug where errors show wrong line numbers.
Example: error on user line 30 was showing as line 17.
"""

import pytest

from nooa import Agent
from nooa.config import CodeActConfig
from nooa.errors.formatting import format_error_for_llm
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM
_TEST_LLM = FakeLLMClient()


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
        result1 = await test_agent.runtime.execute_code(
            setup_code, execution_count=1, wrap_in_function=True
        )
        assert result1.error is None

        # Get the captured locals to pass as builtins for next execution
        builtins = result1.captured_locals.copy()

        # Second execution with globals, error on line 5
        code = """\
a = x + y
b = z["key"]
c = "hello"
d = "world"
e = c.get("foo")
"""
        # Line 5 is: e = c.get("foo")
        result2 = await test_agent.runtime.execute_code(
            code, execution_count=2, builtins=builtins, wrap_in_function=True
        )

        assert result2.error is not None, "Expected an error"

        formatted = format_error_for_llm(
            result2.error, code, line_offset=result2.wrapper_line_offset
        )

        print(f"\n=== Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result2.wrapper_line_offset} ===")

        # The error should report line 5
        assert "line 5" in formatted, f"Expected 'line 5' in error, got:\n{formatted}"

    @pytest.mark.asyncio
    async def test_wrapper_line_offset_is_calculated(self, test_agent):
        """Verify wrapper_line_offset is set in ExecutionResult."""
        code = "x = 1 / 0"  # Line 1
        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        assert result.error is not None
        assert result.wrapper_line_offset > 0, "wrapper_line_offset should be > 0"
        print(f"wrapper_line_offset = {result.wrapper_line_offset}")

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
        result1 = await test_agent.runtime.execute_code(
            code1, execution_count=1, wrap_in_function=True
        )
        assert result1.error is None
        builtins1 = result1.captured_locals.copy()

        # Execution 2: set up product_types
        code2 = "product_types = ['tshirt', 'headphones']"
        result2 = await test_agent.runtime.execute_code(
            code2, execution_count=2, wrap_in_function=True, builtins=builtins1
        )
        assert result2.error is None
        builtins2 = {**builtins1, **result2.captured_locals}

        # Execution 3: long code with error on line 30
        lines = []
        for i in range(1, 30):
            lines.append(f'var_{i} = "value_{i}"')
        lines.append('error_line = "string".get("key")')  # Line 30
        lines.append('var_31 = "after"')
        code3 = "\n".join(lines)

        result3 = await test_agent.runtime.execute_code(
            code3, execution_count=3, wrap_in_function=True, builtins=builtins2
        )

        assert result3.error is not None

        formatted = format_error_for_llm(
            result3.error, code3, line_offset=result3.wrapper_line_offset
        )

        print(f"\n=== Execution 3: Formatted error ===\n{formatted}")
        print(f"\n=== Wrapper line offset: {result3.wrapper_line_offset} ===")
        print(f"\n=== Builtins passed in: {list(builtins2.keys())} ===")

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
        builtins = {}
        for i in range(1, 6):
            code = f"var_{i} = 'value_{i}'"
            result = await test_agent.runtime.execute_code(
                code, execution_count=i, wrap_in_function=True, builtins=builtins
            )
            assert result.error is None
            offsets.append(result.wrapper_line_offset)
            builtins = {**builtins, **result.captured_locals}
            print(f"Exec {i}: offset={result.wrapper_line_offset}, globals={list(builtins.keys())}")

        # Offset should be 2 (no globals) or 3 (with globals), never growing unbounded
        for i, offset in enumerate(offsets, 1):
            assert offset <= 3, f"Execution {i} has offset {offset} - should be <= 3"

    @pytest.mark.asyncio
    async def test_offset_matches_actual_wrapper_structure(self, test_agent):
        """wrapper_line_offset should match the actual number of header lines."""
        # First execution - no globals, offset should be 2
        code1 = "x = 1"
        result1 = await test_agent.runtime.execute_code(
            code1, execution_count=1, wrap_in_function=True
        )
        assert result1.wrapper_line_offset == 2, (
            f"Expected offset 2 with no globals, got {result1.wrapper_line_offset}"
        )

        # Second execution - 1 global, offset should be 3
        code2 = "y = x + 1"
        result2 = await test_agent.runtime.execute_code(
            code2, execution_count=2, wrap_in_function=True, builtins=result1.captured_locals
        )
        assert result2.wrapper_line_offset == 3, (
            f"Expected offset 3 with globals, got {result2.wrapper_line_offset}"
        )

    @pytest.mark.asyncio
    async def test_error_line_correct_across_many_executions(self, test_agent):
        """Error line numbers should be correct even after many prior executions."""
        builtins = {}

        # Run 10 successful executions to build up state
        for i in range(1, 11):
            code = f"var_{i} = 'value_{i}'"
            result = await test_agent.runtime.execute_code(
                code, execution_count=i, wrap_in_function=True, builtins=builtins
            )
            assert result.error is None
            builtins = {**builtins, **result.captured_locals}

        print(f"After 10 executions, {len(builtins)} globals")

        # Now execution 11 with error on a specific line
        error_code = """\
line_1 = "first"
line_2 = "second"
line_3 = "third"
line_4 = "fourth"
error_line = "string".get("missing")
line_6 = "sixth"
"""
        result = await test_agent.runtime.execute_code(
            error_code, execution_count=11, wrap_in_function=True, builtins=builtins
        )

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

        # Verify each has its own entry
        assert "Cell In[1]" in linecache.cache, "Cell In[1] should be in linecache"
        assert "Cell In[2]" in linecache.cache, "Cell In[2] should be in linecache"

        # Verify contents are different
        lines1 = linecache.cache["Cell In[1]"][2]
        lines2 = linecache.cache["Cell In[2]"][2]

        print(f"Cell In[1] has {len(lines1)} lines")
        print(f"Cell In[2] has {len(lines2)} lines")

        assert lines1 != lines2, "Each cell should have different content"


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
        result1 = await test_agent.runtime.execute_code(
            code1, execution_count=1, wrap_in_function=True, builtins=builtins1
        )
        print(f"Exec 1: offset={result1.wrapper_line_offset}")
        captured1 = result1.captured_locals

        # Execution 2
        code2 = """keyboard_product_id = "1656367028"
thermostat_product_id = "4896585277"

keyboard_details = "some_string"
thermostat_details = "some_string" """

        builtins2 = {**builtins1, **captured1}
        result2 = await test_agent.runtime.execute_code(
            code2, execution_count=2, wrap_in_function=True, builtins=builtins2
        )
        print(f"Exec 2: offset={result2.wrapper_line_offset}")
        captured2 = {**captured1, **result2.captured_locals}

        # Execution 3: error on line 3
        code3 = """# Find available keyboard
preferred_keyboard_item_id = None
for variant_id, variant in get_product_details("123")["variants"].items():
    if variant["available"]:
        preferred_keyboard_item_id = variant["item_id"]
        break"""

        builtins3 = {**builtins1, **captured2}
        result3 = await test_agent.runtime.execute_code(
            code3, execution_count=3, wrap_in_function=True, builtins=builtins3
        )

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
    async def test_wrapper_line_vs_user_line_mapping(self, test_agent):
        """Verify the exact mapping between wrapper lines and user lines."""
        import linecache

        code = """line_1 = "first"
line_2 = "second"
line_3 = "third"
error = None.strip()
line_5 = "fifth" """

        result = await test_agent.runtime.execute_code(
            code, execution_count=1, wrap_in_function=True
        )

        # Get wrapper from linecache
        wrapper_lines = linecache.cache["Cell In[1]"][2]

        print("=== Wrapper structure ===")
        for i, line in enumerate(wrapper_lines, 1):
            content = line.rstrip()
            if "line_" in content or "error" in content:
                user_line_match = None
                for j, user_line in enumerate(code.split("\n"), 1):
                    if user_line.strip() in content:
                        user_line_match = j
                        break
                print(f"  wrapper {i} -> user {user_line_match}: {content}")
            else:
                print(f"  wrapper {i}: {content[:60]}")

        print(f"\nOffset: {result.wrapper_line_offset}")

        # The error (None.strip()) is on user line 4
        # With offset 2, it should be wrapper line 6
        # Let's verify
        if result.error:
            tb = result.error.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            print(f"\nRaw error line in wrapper: {tb.tb_lineno}")
            print(f"Expected user line: {tb.tb_lineno - result.wrapper_line_offset}")
            assert tb.tb_lineno - result.wrapper_line_offset == 4, "Error should be on user line 4"


class TestCodeActStrategyErrorFormatting:
    """Test error formatting through the CodeActStrategy."""

    @pytest.mark.asyncio
    async def test_codeact_error_line_numbers(self, test_agent):
        """Test that CodeActStrategy formats errors with correct line numbers."""
        from nooa.strategies.codeact import CodeActStrategy

        # Create strategy instance directly
        strat = CodeActStrategy(config=CodeActConfig())

        # Execute some code to set up state
        result1 = await test_agent.runtime.execute_code(
            "x = 1\ny = 2",
            execution_count=1,
            wrap_in_function=True,
        )

        # Execute code with error on line 4
        code_with_error = """a = x + y
b = "string"
c = "another"
error = b.get("key")
e = "never reached" """

        result2 = await test_agent.runtime.execute_code(
            code_with_error,
            execution_count=2,
            wrap_in_function=True,
            builtins=result1.captured_locals,
        )

        assert result2.error is not None

        # Format error using strategy's _format_error method
        formatted = strat._format_error(
            result2.error, code_with_error, line_offset=result2.wrapper_line_offset
        )

        print("\n=== CodeActStrategy formatted result ===")
        print(formatted)

        # Error should be on line 4
        assert "line 4" in formatted, f"Expected 'line 4' in:\n{formatted}"
        assert "<module>" in formatted, f"Expected '<module>' in:\n{formatted}"


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
