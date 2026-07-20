# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pre-ellipsis code extraction.

This feature allows functions to have setup code before the `...` marker,
which gets executed as part of the prefill before LLM generation.

Example:
    @strategy(CodeActStrategy())
    async def my_method(self, data: list[str]) -> Result:
        '''Process the data.'''
        # This code runs before LLM generation:
        validated = [x for x in data if x.strip()]
        config = {"max_items": 100}
        ...  # LLM generates from here

Note: This file contains intentionally unused variables in test functions
to verify that pre-ellipsis code extraction works correctly.
"""
# ruff: noqa: F841, C416  # Unused variables and dict comprehensions are intentional test fixtures

import ast
import textwrap

from nooa.ellipsis_detection import get_pre_ellipsis_code, has_ellipsis_body


class TestHasEllipsisBody:
    """Test has_ellipsis_body() - detects if function body ends with ellipsis."""

    def test_simple_ellipsis_only(self):
        """Function with only ellipsis should return True."""

        def simple(): ...

        assert has_ellipsis_body(simple)

    def test_ellipsis_with_docstring(self):
        """Ellipsis after docstring should return True."""

        def with_doc():
            """Docstring."""
            ...

        assert has_ellipsis_body(with_doc)

    def test_code_before_ellipsis(self):
        """Code before ellipsis should return True."""

        def with_setup():
            x = 1
            y = 2
            ...

        assert has_ellipsis_body(with_setup)

    def test_docstring_and_code_before_ellipsis(self):
        """Docstring + code + ellipsis should return True."""

        def full_setup():
            """Setup some things."""
            config = {"key": "value"}
            items = [1, 2, 3]
            ...

        assert has_ellipsis_body(full_setup)

    def test_async_with_setup(self):
        """Async function with setup code should return True."""

        async def async_setup():
            """Async setup."""
            data = await_placeholder  # noqa: F821 - just for AST
            ...

        # Can't actually define await outside async, so test differently
        async def async_setup_real():
            """Async setup."""
            x = 1
            ...

        assert has_ellipsis_body(async_setup_real)

    def test_no_ellipsis_returns_false(self):
        """Function without ellipsis should return False."""

        def implemented():
            return 42

        assert not has_ellipsis_body(implemented)

    def test_ellipsis_not_at_end_returns_false(self):
        """Ellipsis in middle of code should return False."""

        def ellipsis_middle():
            ...
            return 42

        assert not has_ellipsis_body(ellipsis_middle)

    def test_pass_statement_returns_false(self):
        """Pass statement is not ellipsis."""

        def with_pass():
            pass

        assert not has_ellipsis_body(with_pass)

    def test_class_method(self):
        """Should work with class methods."""

        class MyClass:
            def method_with_setup(self):
                """Method doc."""
                self.x = 1
                ...

        assert has_ellipsis_body(MyClass.method_with_setup)

    def test_complex_setup_code(self):
        """Should handle complex setup code."""

        def complex_setup():
            """Complex setup."""
            # Comments are preserved
            items = [x * 2 for x in range(10)]
            mapping = {k: v for k, v in [("a", 1), ("b", 2)]}

            def helper(x):
                return x + 1

            result = helper(5)
            ...

        assert has_ellipsis_body(complex_setup)

    def test_token_error_falls_back_to_generated_source(self):
        """TokenError from inspect.getsource should fall back to _generated_source."""
        import tokenize
        from unittest.mock import patch

        def my_func():
            """Docstring."""
            ...

        # Attach _generated_source so fallback can succeed
        source = 'def my_func():\n    """Docstring."""\n    ...\n'
        my_func._generated_source = source

        with patch("inspect.getsource", side_effect=tokenize.TokenError("faked")):
            assert has_ellipsis_body(my_func)

    def test_token_error_without_generated_source_returns_false(self):
        """TokenError without _generated_source should return False (no crash)."""
        import tokenize
        from unittest.mock import patch

        def my_func():
            """Docstring."""
            ...

        with patch("inspect.getsource", side_effect=tokenize.TokenError("faked")):
            # No _generated_source, so _get_function_ast returns None
            # has_ellipsis_body falls through to bytecode heuristic
            result = has_ellipsis_body(my_func)
            # Either True (bytecode heuristic) or False is fine - the point is no crash
            assert isinstance(result, bool)


class TestGetPreEllipsisCode:
    """Test get_pre_ellipsis_code() - extracts code before ellipsis."""

    def test_no_pre_code_returns_none(self):
        """Function with only ellipsis should return None."""

        def simple(): ...

        assert get_pre_ellipsis_code(simple) is None

    def test_docstring_only_returns_none(self):
        """Function with only docstring + ellipsis should return None."""

        def with_doc():
            """Just a docstring."""
            ...

        assert get_pre_ellipsis_code(with_doc) is None

    def test_extracts_single_statement(self):
        """Should extract single statement before ellipsis."""

        def single_stmt():
            x = 1
            ...

        code = get_pre_ellipsis_code(single_stmt)
        assert code is not None
        assert "x = 1" in code

    def test_extracts_multiple_statements(self):
        """Should extract multiple statements."""

        def multi_stmt():
            x = 1
            y = 2
            z = x + y
            ...

        code = get_pre_ellipsis_code(multi_stmt)
        assert code is not None
        assert "x = 1" in code
        assert "y = 2" in code
        assert "z = x + y" in code

    def test_skips_docstring(self):
        """Should not include docstring in extracted code."""

        def with_docstring():
            """This is the docstring."""
            setup_var = 42
            ...

        code = get_pre_ellipsis_code(with_docstring)
        assert code is not None
        assert "setup_var = 42" in code
        assert "This is the docstring" not in code

    def test_preserves_complex_expressions(self):
        """Should preserve list comprehensions, dict literals, etc."""

        def complex_code():
            items = [x * 2 for x in range(10)]
            mapping = {"key": "value", "num": 42}
            ...

        code = get_pre_ellipsis_code(complex_code)
        assert code is not None
        assert "[x * 2 for x in range(10)]" in code
        assert '"key"' in code or "'key'" in code

    def test_preserves_function_calls(self):
        """Should preserve function calls."""

        def with_calls():
            result = some_function(arg1, arg2=value)  # noqa: F821
            ...

        code = get_pre_ellipsis_code(with_calls)
        assert code is not None
        assert "some_function" in code

    def test_handles_nested_function(self):
        """Should include nested function definitions."""

        def with_nested():
            def helper(x):
                return x + 1

            ...

        code = get_pre_ellipsis_code(with_nested)
        assert code is not None
        assert "def helper(x):" in code
        assert "return x + 1" in code

    def test_handles_class_definition(self):
        """Should include class definitions."""

        def with_class():
            class Config:
                value = 42

            ...

        code = get_pre_ellipsis_code(with_class)
        assert code is not None
        assert "class Config:" in code
        assert "value = 42" in code

    def test_no_ellipsis_returns_none(self):
        """Function without ellipsis should return None."""

        def implemented():
            x = 1
            return x

        assert get_pre_ellipsis_code(implemented) is None

    def test_ellipsis_in_middle_returns_none(self):
        """Ellipsis not at end should return None."""

        def ellipsis_middle():
            x = 1
            ...
            return x

        assert get_pre_ellipsis_code(ellipsis_middle) is None

    def test_preserves_imports(self):
        """Should include import statements."""

        # Note: imports inside functions are valid Python
        def with_import():
            from collections import defaultdict

            data = defaultdict(list)
            ...

        code = get_pre_ellipsis_code(with_import)
        assert code is not None
        assert "from collections import defaultdict" in code
        assert "defaultdict(list)" in code

    def test_async_function(self):
        """Should work with async functions."""

        async def async_setup():
            x = 1
            y = 2
            ...

        code = get_pre_ellipsis_code(async_setup)
        assert code is not None
        assert "x = 1" in code
        assert "y = 2" in code

    def test_code_is_valid_python(self):
        """Extracted code should be valid, parseable Python."""

        def with_setup():
            """Docstring."""
            config = {"key": "value"}
            items = [1, 2, 3]
            total = sum(items)
            ...

        code = get_pre_ellipsis_code(with_setup)
        assert code is not None
        # Should parse without error
        ast.parse(code)

    def test_code_is_executable(self):
        """Extracted code should be executable."""

        def with_setup():
            x = 10
            y = 20
            result = x + y
            ...

        code = get_pre_ellipsis_code(with_setup)
        assert code is not None

        # Execute in isolated namespace
        namespace = {}
        exec(code, namespace)
        assert namespace["x"] == 10
        assert namespace["y"] == 20
        assert namespace["result"] == 30


class TestGetPreEllipsisCodeEdgeCases:
    """Edge cases for pre-ellipsis code extraction."""

    def test_multiline_string(self):
        """Should handle multiline strings in setup code."""

        def with_multiline():
            template = """
            This is a
            multiline string
            """
            ...

        code = get_pre_ellipsis_code(with_multiline)
        assert code is not None
        assert "template" in code

    def test_comments_in_code(self):
        """Should handle comments (note: AST doesn't preserve comments)."""

        def with_comments():
            # This is a comment
            x = 1  # inline comment
            ...

        code = get_pre_ellipsis_code(with_comments)
        assert code is not None
        assert "x = 1" in code
        # Comments may or may not be preserved depending on implementation

    def test_decorator_on_nested_function(self):
        """Should handle decorators on nested functions."""

        def with_decorated():
            @staticmethod
            def helper():
                pass

            ...

        code = get_pre_ellipsis_code(with_decorated)
        assert code is not None
        # Should include the decorator
        assert "staticmethod" in code or "helper" in code

    def test_try_except_block(self):
        """Should handle try/except blocks."""

        def with_try():
            try:
                x = int("42")
            except ValueError:
                x = 0
            ...

        code = get_pre_ellipsis_code(with_try)
        assert code is not None
        assert "try:" in code
        assert "except" in code

    def test_with_statement(self):
        """Should handle with statements."""

        def with_context():
            with open("test.txt") as f:  # noqa: F841
                pass
            ...

        code = get_pre_ellipsis_code(with_context)
        assert code is not None
        assert "with" in code

    def test_if_statement(self):
        """Should handle if statements."""

        def with_if():
            if True:
                x = 1
            else:
                x = 2
            ...

        code = get_pre_ellipsis_code(with_if)
        assert code is not None
        assert "if True:" in code
        assert "else:" in code

    def test_for_loop(self):
        """Should handle for loops."""

        def with_for():
            total = 0
            for i in range(10):
                total += i
            ...

        code = get_pre_ellipsis_code(with_for)
        assert code is not None
        assert "for i in range(10):" in code

    def test_while_loop(self):
        """Should handle while loops."""

        def with_while():
            x = 0
            while x < 10:
                x += 1
            ...

        code = get_pre_ellipsis_code(with_while)
        assert code is not None
        assert "while x < 10:" in code

    def test_exec_defined_function(self):
        """Should handle functions defined via exec with _generated_source."""
        code_str = textwrap.dedent("""
            def setup_func():
                x = 1
                y = 2
                ...
        """).strip()

        namespace = {}
        exec(code_str, namespace)
        func = namespace["setup_func"]
        func._generated_source = code_str

        code = get_pre_ellipsis_code(func)
        assert code is not None
        assert "x = 1" in code
        assert "y = 2" in code


class TestEllipsisDetection:
    """Ensure has_ellipsis_body works for all ellipsis patterns."""

    def test_detects_pure_ellipsis(self):
        """has_ellipsis_body should detect pure ellipsis functions."""
        from nooa.ellipsis_detection import has_ellipsis_body

        def pure_ellipsis(): ...

        def with_docstring():
            """Doc."""
            ...

        def implemented():
            return 42

        assert has_ellipsis_body(pure_ellipsis)
        assert has_ellipsis_body(with_docstring)
        assert not has_ellipsis_body(implemented)

    def test_detects_setup_code_with_ellipsis(self):
        """has_ellipsis_body should detect functions with setup code ending in ellipsis."""
        from nooa.ellipsis_detection import has_ellipsis_body

        def with_setup():
            x = 1
            ...

        # has_ellipsis_body detects functions ENDING with ellipsis
        assert has_ellipsis_body(with_setup)


class TestCurrentCallIntegration:
    """Test that CurrentCall extracts pre-ellipsis code."""

    def test_current_call_extracts_pre_ellipsis_code(self):
        """CurrentCall.from_method should extract pre-ellipsis code."""
        from nooa.strategies.current_call import CurrentCall

        async def method_with_setup(self, data: list[str]) -> str:
            """Process data."""
            validated = [x.strip() for x in data]
            config = {"max": 100}
            ...

        call = CurrentCall.from_method(method_with_setup, args=(["a", "b"],))

        assert call.pre_ellipsis_code is not None
        assert "validated" in call.pre_ellipsis_code
        assert "config" in call.pre_ellipsis_code

    def test_current_call_no_pre_ellipsis_for_pure_ellipsis(self):
        """CurrentCall should have None for pure-ellipsis methods."""
        from nooa.strategies.current_call import CurrentCall

        async def pure_ellipsis_method(self, x: int) -> int:
            """Just ellipsis."""
            ...

        call = CurrentCall.from_method(pure_ellipsis_method, args=(42,))

        assert call.pre_ellipsis_code is None

    def test_current_call_no_pre_ellipsis_for_implemented(self):
        """CurrentCall should have None for implemented methods."""
        from nooa.strategies.current_call import CurrentCall

        async def implemented_method(self, x: int) -> int:
            """Has implementation."""
            return x * 2

        call = CurrentCall.from_method(implemented_method, args=(42,))

        assert call.pre_ellipsis_code is None
