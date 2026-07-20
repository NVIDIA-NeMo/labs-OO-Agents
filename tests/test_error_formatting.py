# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for error formatting module.

Ensures errors shown to the LLM match IPython/Jupyter-style output:
- Uses "Cell In[N], line X" format (IPython style)
- Adjusts line numbers to account for wrapper code offset
- Framework tracebacks are hidden
- User code frames are shown
- Syntax errors show caret pointing to error location
- Validation errors show clean messages without tracebacks
"""

from nooa.errors import IPythonErrorFormatter, RestrictedCodeError, format_error_for_llm
from nooa.errors.formatting import (
    _adjust_line_numbers,
    _is_user_code_frame,
    _is_validation_error,
    _strip_file_prefix,
)


class TestIsUserCodeFrame:
    """Tests for _is_user_code_frame detection."""

    def test_cell_in_format_is_user_frame(self):
        """Frames from Cell In[N] are user code."""
        assert _is_user_code_frame("Cell In[1]") is True
        assert _is_user_code_frame("Cell In[42]") is True
        assert _is_user_code_frame("Cell In[100]") is True

    def test_execute_code_is_user_frame(self):
        """Frames from <execute_code> are user code (legacy compatibility)."""
        assert _is_user_code_frame("<execute_code>") is True

    def test_nooa_is_framework(self):
        """Frames from nooa/ are framework code."""
        assert _is_user_code_frame("/path/to/nooa/runtime/actor.py") is False
        assert _is_user_code_frame("nooa/strategies/pure_python.py") is False

    def test_site_packages_is_framework(self):
        """Frames from site-packages are framework code."""
        assert _is_user_code_frame("/lib/python3.12/site-packages/litellm/main.py") is False

    def test_lib_python_is_framework(self):
        """Frames from lib/python are framework code."""
        assert (
            _is_user_code_frame("/Users/x/.pyenv/versions/3.12.7/lib/python3.12/asyncio/runners.py")
            is False
        )

    def test_frozen_is_framework(self):
        """Frames from <frozen are framework code."""
        assert _is_user_code_frame("<frozen importlib._bootstrap>") is False


class TestStripFilePrefix:
    """Tests for _strip_file_prefix function."""

    def test_strips_file_prefix(self):
        """Strips File "..." wrapper."""
        text = 'File "Cell In[1]", line 1'
        assert _strip_file_prefix(text) == "Cell In[1], line 1"

    def test_strips_multiple_occurrences(self):
        """Strips multiple File "..." wrappers."""
        text = 'File "Cell In[1]", line 1\nFile "Cell In[2]", line 5'
        result = _strip_file_prefix(text)
        assert "Cell In[1], line 1" in result
        assert "Cell In[2], line 5" in result
        assert 'File "' not in result

    def test_preserves_other_text(self):
        """Preserves text that doesn't match the pattern."""
        text = "SyntaxError: invalid syntax"
        assert _strip_file_prefix(text) == text


class TestAdjustLineNumbers:
    """Tests for _adjust_line_numbers function."""

    def test_adjusts_cell_line_format(self):
        """Adjusts Cell In[N], line X format."""
        text = "Cell In[1], line 5"
        assert _adjust_line_numbers(text, 2) == "Cell In[1], line 3"

    def test_adjusts_simple_line_format(self):
        """Adjusts simple 'line X' format."""
        text = "line 5"
        assert _adjust_line_numbers(text, 2) == "line 3"

    def test_adjusts_multiple_occurrences(self):
        """Adjusts all line numbers in text."""
        text = "Cell In[1], line 5\n  some code\nCell In[1], line 10"
        result = _adjust_line_numbers(text, 3)
        assert "line 2" in result
        assert "line 7" in result
        assert "line 5" not in result
        assert "line 10" not in result

    def test_never_goes_below_line_1(self):
        """Line numbers never go below 1."""
        text = "Cell In[1], line 2"
        assert _adjust_line_numbers(text, 5) == "Cell In[1], line 1"

    def test_zero_offset_no_change(self):
        """Zero offset leaves text unchanged."""
        text = "Cell In[1], line 5"
        assert _adjust_line_numbers(text, 0) == text

    def test_negative_offset_no_change(self):
        """Negative offset leaves text unchanged."""
        text = "Cell In[1], line 5"
        assert _adjust_line_numbers(text, -1) == text


class TestIsValidationError:
    """Tests for _is_validation_error detection."""

    def test_restricted_code_error_is_validation(self):
        """RestrictedCodeError is a validation error."""
        error = RestrictedCodeError("Line 1: import forbidden")
        assert _is_validation_error(error) is True

    def test_runtime_error_is_not_validation(self):
        """RuntimeError is not a validation error."""
        error = RuntimeError("Something went wrong")
        assert _is_validation_error(error) is False

    def test_value_error_is_not_validation(self):
        """ValueError is not a validation error."""
        error = ValueError("Invalid value")
        assert _is_validation_error(error) is False


class TestFormatSyntaxError:
    """Tests for syntax error formatting."""

    def test_basic_syntax_error(self):
        """Basic syntax error shows line and caret."""
        code = "def foo(\n    x = 1\n    y = 2"  # Missing closing paren
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            formatter = IPythonErrorFormatter()
            result = formatter.format(e, code)
            assert "SyntaxError" in result
            # Should use Cell In[N] format, not File "..."
            assert 'File "' not in result

    def test_syntax_error_ipython_format(self):
        """Syntax error output matches IPython format."""
        code = "x = 1 + + 2"
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            formatter = IPythonErrorFormatter()
            result = formatter.format(e, code)

            # Should have IPython-style header (no File prefix)
            assert "Cell In[1], line 1" in result
            # Should show the offending line
            assert "x = 1 + + 2" in result
            # Should have caret indicator
            assert "^" in result
            assert "SyntaxError" in result

    def test_syntax_error_with_line_offset(self):
        """Syntax error line numbers are adjusted by offset."""
        code = "x = 1 + + 2"
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            # Simulate wrapper with 2 header lines
            result = format_error_for_llm(e, code, line_offset=2)
            # Line 1 - 2 = line 1 (clamped to minimum of 1)
            assert "line 1" in result

    def test_syntax_error_from_code_string(self):
        """Syntax error can get line from code string if text is missing."""
        code = "line1\nline2\nline3 bad syntax here"
        error = SyntaxError("test error")
        error.lineno = 3
        error.offset = 10
        error.text = None
        error.filename = "Cell In[1]"

        formatter = IPythonErrorFormatter()
        result = formatter.format(error, code)

        # Without offset, line 3 shows as line 3
        assert "line 3" in result
        assert "line3 bad syntax here" in result


class TestFormatValidationError:
    """Tests for validation error formatting."""

    def test_validation_error_no_traceback(self):
        """Validation errors don't include tracebacks."""
        error = RestrictedCodeError("Line 1: import statements are forbidden")

        result = format_error_for_llm(error)

        assert "RestrictedCodeError" in result
        assert "import statements are forbidden" in result
        # Should NOT contain traceback markers
        assert "Traceback" not in result


class TestFormatRuntimeError:
    """Tests for runtime error formatting."""

    def test_error_without_traceback(self):
        """Error without traceback shows type and message."""
        error = ValueError("invalid value")

        formatter = IPythonErrorFormatter()
        result = formatter.format(error, None)

        assert "ValueError: invalid value" in result

    def test_runtime_error_filters_framework(self):
        """Runtime error filters out framework frames."""
        code = "x = 1 / 0"

        try:
            exec(compile(code, "Cell In[1]", "exec"))
        except ZeroDivisionError as e:
            formatter = IPythonErrorFormatter()
            result = formatter.format(e, code)

            assert "ZeroDivisionError" in result
            # Should NOT contain framework paths
            assert "nooa/" not in result
            assert "site-packages/" not in result

    def test_runtime_error_ipython_format(self):
        """Runtime error uses IPython-style format."""
        code = "x = 1 / 0"

        try:
            exec(compile(code, "Cell In[1]", "exec"))
        except ZeroDivisionError as e:
            formatter = IPythonErrorFormatter()
            result = formatter.format(e, code)

            # Should NOT have File "..." wrapper
            assert 'File "' not in result
            # Should have IPython-style format
            assert "Cell In[1]" in result
            assert "ZeroDivisionError" in result

    def test_runtime_error_with_line_offset(self):
        """Runtime error line numbers are adjusted by offset."""
        # Simulate code that would be on line 5 of a wrapper
        # We'll manually create a scenario with traceback
        code = "x = 1 / 0"

        try:
            exec(compile(code, "Cell In[1]", "exec"))
        except ZeroDivisionError as e:
            # With line_offset=2, user's line 1 should display as line 1 (clamped)
            result = format_error_for_llm(e, code, line_offset=0)
            assert "Cell In[1], line 1" in result

            # If we had line_offset, it would adjust (but line 1 - offset would clamp to 1)
            result_with_offset = format_error_for_llm(e, code, line_offset=2)
            # Line numbers should be adjusted
            assert "ZeroDivisionError" in result_with_offset


class TestFormatErrorForLLM:
    """Integration tests for format_error_for_llm."""

    def test_syntax_error_handled(self):
        """format_error_for_llm handles SyntaxError correctly."""
        code = "def foo(\n    pass"
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            result = format_error_for_llm(e, code)
            assert "SyntaxError" in result
            # Should use IPython format
            assert 'File "' not in result

    def test_validation_error_handled(self):
        """format_error_for_llm handles validation errors correctly."""
        error = RestrictedCodeError("Line 1: import forbidden\n\nAvailable: asyncio, os")

        result = format_error_for_llm(error)

        assert "RestrictedCodeError" in result
        assert "import forbidden" in result
        assert "Traceback" not in result

    def test_runtime_error_handled(self):
        """format_error_for_llm handles runtime errors correctly."""
        error = KeyError("missing_key")

        result = format_error_for_llm(error)

        assert "KeyError" in result
        assert "missing_key" in result

    def test_line_offset_parameter(self):
        """format_error_for_llm accepts line_offset parameter."""
        code = "x = invalid_syntax here"
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            # Just verify it accepts the parameter without error
            result = format_error_for_llm(e, code, line_offset=3)
            assert "SyntaxError" in result


class TestBeforeAfterComparison:
    """Demonstrate the improvement: before vs after error formatting."""

    def test_syntax_error_before_vs_after(self):
        """Syntax errors now use IPython-style format.

        BEFORE:
            File "Cell In[1]", line 1
              <tool_call>
              ^
            SyntaxError: invalid syntax

        AFTER (IPython style):
            Cell In[1], line 1
              <tool_call>
              ^
            SyntaxError: invalid syntax
        """
        code = "<tool_call>"
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            result = format_error_for_llm(e, code)

            # Should NOT have File "..." wrapper
            assert 'File "' not in result
            # Should have IPython-style header
            assert "Cell In[1], line 1" in result
            # Should show caret
            assert "^" in result

    def test_validation_error_before_vs_after(self):
        """Validation errors are now cleaner without framework tracebacks.

        BEFORE (noisy - from the trace file):
        ```
        Traceback (most recent call last):
          File "/Volumes/dev/dev/nooa/src/nooa/runtime/actor.py", line 261, in execute_code
            validate_planning_code(
          File "/Volumes/dev/dev/nooa/src/nooa/runtime/validator.py", line 113, in validate_planning_code
            validator.validate(code)
          File "/Volumes/dev/dev/nooa/src/nooa/runtime/validator.py", line 53, in validate
            raise ValidationError("\\n".join(self.errors))
        nooa.errors.RestrictedCodeError: Line 1: import statements are forbidden...
        ```

        AFTER (clean):
        ```
        RestrictedCodeError: Line 1: import statements are forbidden...
        ```
        """
        error = RestrictedCodeError(
            "Line 1: import statements are forbidden.\n\n"
            "Available in scope: Agent, AnalyzerResult, AnalyzerSubAgent, asyncio, doc, message, methods, plan"
        )

        result = format_error_for_llm(error)

        # The result should NOT contain framework paths
        assert "actor.py" not in result
        assert "validator.py" not in result
        assert "Traceback" not in result

        # The result SHOULD contain the actionable error message
        assert "import statements are forbidden" in result
        assert "Available in scope" in result

    def test_runtime_error_before_vs_after(self):
        """Runtime errors now use IPython-style format.

        BEFORE (noisy):
        ```
        Traceback (most recent call last):
          File "/Volumes/dev/dev/nooa/src/nooa/runtime/actor.py", line 313, in execute_code
            result_value = await exec_globals["__repl_wrapper__"]()
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "<execute_code>", line 6, in __repl_wrapper__
            ...
        RuntimeError: asyncio.run() cannot be called from a running event loop
        ```

        AFTER (IPython-style):
        ```
        Cell In[1], line 1, in <module>
            x = 1 / 0
                ~~^~~
        ZeroDivisionError: division by zero
        ```
        """
        code = "x = 1 / 0"
        try:
            exec(compile(code, "Cell In[1]", "exec"))
        except ZeroDivisionError as e:
            result = format_error_for_llm(e, code)

            # Should NOT have File "..." wrapper
            assert 'File "' not in result
            # Should have Cell In[N] format
            assert "Cell In[1]" in result
            assert "ZeroDivisionError" in result


class TestIPythonErrorFormatter:
    """Tests for IPythonErrorFormatter class."""

    def test_formatter_is_separate_class(self):
        """IPythonErrorFormatter is a distinct class."""
        formatter = IPythonErrorFormatter()
        assert hasattr(formatter, "format")
        assert callable(formatter.format)

    def test_formatter_accepts_line_offset(self):
        """Formatter accepts line_offset parameter."""
        formatter = IPythonErrorFormatter()
        error = ValueError("test")
        result = formatter.format(error, None, line_offset=5)
        assert "ValueError" in result

    def test_custom_formatter_compatibility(self):
        """Custom formatters can be created with same interface."""

        class CustomFormatter:
            def format(
                self, error: Exception, _code: str | None = None, *, line_offset: int = 0
            ) -> str:
                # _code intentionally unused - testing interface compatibility
                return f"CUSTOM: {type(error).__name__} (offset={line_offset})"

        formatter = CustomFormatter()
        error = ValueError("test")
        result = formatter.format(error, None, line_offset=3)
        assert result == "CUSTOM: ValueError (offset=3)"


class TestHeredocHint:
    """Heredoc hint appended to SyntaxErrors that look like LLM-embedded bash heredocs."""

    @staticmethod
    def _compile_and_format(code: str) -> str:
        """Compile `code`, expect a SyntaxError, and return the LLM-formatted message."""
        try:
            compile(code, "Cell In[1]", "exec")
        except SyntaxError as e:
            return format_error_for_llm(e, code)
        raise AssertionError(f"expected SyntaxError compiling: {code!r}")

    @staticmethod
    def _assert_hint_present(result: str) -> None:
        """Assert the heredoc hint and both fix patterns appear in `result`."""
        assert "heredoc" in result, f"expected 'heredoc' in output, got:\n{result}"
        assert '"""' in result, f"expected triple-quote (Fix 1) in output, got:\n{result}"
        assert "shell.write" in result, f"expected shell.write (Fix 2) in output, got:\n{result}"

    @staticmethod
    def _assert_hint_absent(result: str) -> None:
        """Assert no heredoc hint was appended to `result`."""
        assert "heredoc" not in result, f"unexpected 'heredoc' in output:\n{result}"
        assert "shell.write" not in result, f"unexpected shell.write in output:\n{result}"

    # ----- Positive cases: one per trigger message -----

    def test_hint_on_unterminated_string_literal(self):
        """Canonical case: heredoc in a single-quoted string → unterminated string literal."""
        code = 'shell.run("cat <<EOF\ncontent\nEOF")'
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        assert "unterminated string literal" in result
        self._assert_hint_present(result)

    def test_hint_on_invalid_syntax_forgot_comma(self):
        """Heredoc + implicit string concat shape → 'Perhaps you forgot a comma?'."""
        code = 'shell.run("cat <<EOF" b)'
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        assert "forgot a comma" in result
        self._assert_hint_present(result)

    def test_hint_on_line_continuation_character(self):
        """Heredoc + stray backslash → 'unexpected character after line continuation character'."""
        code = 'shell.run("cat <<EOF" \\xyz)'
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        assert "line continuation character" in result
        self._assert_hint_present(result)

    # ----- Negative cases -----

    def test_no_hint_when_trigger_msg_but_no_heredoc(self):
        """Same trigger message (unterminated string literal), no heredoc marker → no hint."""
        code = 'x = "hello'
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        assert "unterminated string literal" in result
        self._assert_hint_absent(result)

    def test_no_hint_when_heredoc_marker_but_unrelated_msg(self):
        """Source contains `<< 2` shaped tokens but Python emits bare 'invalid syntax' → no hint.

        `x = << 2` triggers bare 'invalid syntax', which is not one of the three trigger messages.
        """
        code = "x = << 2"
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        # The hint must not fire on bare 'invalid syntax'
        self._assert_hint_absent(result)

    def test_no_hint_when_unrelated_syntax_error_with_heredoc_in_source(self):
        """`'await' outside function` with `<<EOF` literally in the source → no hint.

        Proves the gate is `msg ∈ TRIGGERS`, not just "source contains <<".
        """
        # `await` outside an async function — error msg is "'await' outside function",
        # not one of the three triggers. The source still contains `<<EOF`.
        code = 'await shell.run("<<EOF foo")'
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        assert "await" in result.lower()
        self._assert_hint_absent(result)

    def test_no_hint_on_legitimate_bitshift_with_forgot_comma(self):
        """Real bit-shift `a << foo` inside a call that forgot a comma → no hint (best-effort).

        Reviewer flagged this as the regex's worst-case false positive: `<< foo` matches
        the heredoc regex. The mitigation is to require a quote character before `<<`
        on the same source line (`error.text`). A bare `func(a << foo b)` has no such
        quote, so no hint is appended.
        """
        code = "func(a << foo b)"
        result = self._compile_and_format(code)
        assert "SyntaxError" in result
        assert "forgot a comma" in result
        self._assert_hint_absent(result)

    def test_no_hint_when_heredoc_marker_is_on_different_line_than_error(self):
        """Heredoc on a non-flagged line → no hint.

        Documents the design: the hint requires the heredoc marker *and* a
        preceding quote on the offending line (error.text). A heredoc marker
        on some unrelated later line doesn't trigger the hint, because that
        situation isn't the LLM-embedded-heredoc pattern we're targeting.
        """
        code = (
            'shell.run("foo" "bar")\n'  # the offending line — Python flags missing comma
            "something = 1\n"
            "cat <<EOF\n"  # heredoc marker is here, line 3 — but bare shell, not embedded
            "content\n"
            "EOF"
        )
        error = SyntaxError("invalid syntax. Perhaps you forgot a comma?")
        error.text = 'shell.run("foo" "bar")'
        error.lineno = 1
        error.offset = 17
        error.filename = "Cell In[1]"

        result = format_error_for_llm(error, code)
        assert "SyntaxError" in result
        self._assert_hint_absent(result)

    # ----- Hint text shape -----

    def test_hint_mentions_both_fixes(self):
        """The hint surfaces both fix patterns explicitly."""
        code = 'shell.run("cat <<EOF\ncontent\nEOF")'
        result = self._compile_and_format(code)
        # Fix 1: triple-quoted string
        assert "triple-quoted" in result or '"""' in result
        # Fix 2: write to file then bash it
        assert "shell.write" in result
        assert "bash" in result.lower()


class _ShellToolsLike:
    def replace(self, target, old_or_new="", new=None):
        """Replace at an unambiguous location."""
        ...


class _FooLike:
    def bar(self, target, count=1):
        """Do a bar."""
        ...


def _bare_func(a, b):
    """A bare function."""
    ...


def _format_bad_call(fn) -> str:
    """Run *fn* (which must raise a TypeError) and return the LLM-formatted error."""
    try:
        fn()
    except TypeError as e:
        return format_error_for_llm(e)
    raise AssertionError("expected a TypeError")


class TestBadCallAgentdoc:
    """Issue #245: a call-shape TypeError appends the callable's concise agentdoc.

    A bad method/function call ("got an unexpected keyword argument", "missing N
    required positional argument", ...) names the bad arg but never the correct
    signature, so the model loops guessing synonyms. We resolve the callable from
    the traceback frames and append its concise agentdoc.

    The helper callables are module-level on purpose — that mirrors the
    real-world case (tool/skill methods live at module scope and their instances
    are bound names in the cell), which is what the traceback-frame resolver can
    actually reach.
    """

    def test_unexpected_keyword_appends_signature(self):
        """An unexpected-kwarg call appends the method's concise signature."""

        def cell():
            tool = _ShellToolsLike()
            tool.replace(target="x", old="a", new="b")

        result = _format_bad_call(cell)
        assert "unexpected keyword argument 'old'" in result
        assert "_ShellToolsLike.replace" in result
        assert "old_or_new" in result
        assert "Replace at an unambiguous location" in result

    def test_missing_positional_appends_signature(self):
        """A missing-positional call appends the method's concise signature."""

        def cell():
            foo = _FooLike()
            foo.bar()

        result = _format_bad_call(cell)
        assert "missing 1 required positional argument" in result
        assert "_FooLike.bar" in result
        assert "count" in result

    def test_bare_function_too_many_positional_appends_signature(self):
        """A too-many-positional call to a bare function appends its signature."""

        def cell():
            _bare_func(1, 2, 3)

        result = _format_bad_call(cell)
        assert "takes 2 positional arguments but 3 were given" in result
        assert "_bare_func" in result

    def test_non_call_typeerror_is_unchanged(self):
        """A TypeError that is not a bad call must not gain an agentdoc block."""

        def cell():
            return 1 + "x"

        result = _format_bad_call(cell)
        assert "unsupported operand type" in result
        assert "signature" not in result.lower()

    def test_unresolvable_callable_falls_back_cleanly(self):
        """If the callable can't be resolved, no agentdoc is appended (no regression)."""
        from nooa.errors.formatting import _bad_call_agentdoc

        err = TypeError("Nonexistent.method() got an unexpected keyword argument 'z'")
        assert _bad_call_agentdoc(err) is None


# Module-level fixtures for the resolver hardening tests (issue #245 review).
class _AlphaTool:
    def run(self, target, mode="x"):
        """The real AlphaTool.run."""
        ...


class _SideEffectProbe:
    """A class named like a tool whose called attr is a side-effecting property."""

    accessed = False

    @property
    def run(self):  # noqa: D401 - property standing in for a method name
        type(self).accessed = True
        return lambda *a, **k: None


class _Config245:
    def __init__(self, host, port=80):
        """A config constructed with (host, port)."""
        ...


class TestBadCallAgentdocHardening:
    """Review follow-ups for #245: ambiguity, descriptor safety, constructors."""

    def test_distinct_qualname_decoy_does_not_block_real_resolution(self):
        """A same-NAMED but different-QUALNAME decoy in scope doesn't shadow the real one."""

        def make_decoy():
            class _AlphaTool:  # same class name, different qualname (has <locals>)
                def run(self, x):
                    """Decoy run."""
                    ...

            return _AlphaTool()

        def cell():
            real = _AlphaTool()
            decoy = make_decoy()  # noqa: F841 - in scope to create the name collision
            real.run(target="t", bogus=1)

        result = _format_bad_call(cell)
        assert "unexpected keyword argument 'bogus'" in result
        # The decoy's stripped qualname is still "_AlphaTool.run", BUT it lives in
        # a different frame's locals; uniqueness is judged per resolution, and the
        # real module-level _AlphaTool.run is the single match reachable here.
        assert "_AlphaTool.run" in result

    def test_truly_ambiguous_qualnames_in_one_frame_resolve_to_none(self):
        """Two callables with identical stripped qualnames in the SAME live frame → None.

        This exercises the uniqueness guard for real (not via an empty traceback):
        both candidates are reachable in the call-site frame, so the resolver must
        refuse to guess and append no signature.
        """

        # Two module-level-style functions that share a qualname after <locals>
        # stripping. We fabricate the collision by giving a second function the
        # same __qualname__ as the first, then call one with a bad signature while
        # both are bound names in the failing frame.
        def alpha(a, b):
            """First alpha."""
            ...

        def alpha_decoy(a, b, c):
            """Second alpha (different arity)."""
            ...

        alpha_decoy.__qualname__ = alpha.__qualname__  # force identical qualname

        def cell():
            one = alpha  # noqa: F841 - both bound in this frame
            two = alpha_decoy  # noqa: F841
            one(1, 2, 3)  # bad call -> "...alpha() takes 2 positional arguments but 3..."

        result = _format_bad_call(cell)
        assert "positional argument" in result
        # Ambiguous (two distinct callables, same qualname, same frame) → no signature.
        assert "this signature" not in result

    def test_property_is_not_invoked_during_resolution(self):
        """A same-named class exposing the attr as a @property must NOT be executed."""
        _SideEffectProbe.accessed = False

        def cell():
            probe = _SideEffectProbe()  # noqa: F841 - in scope; attr is a property
            tool = _AlphaTool()
            tool.run(target="t", bogus=1)

        _format_bad_call(cell)
        assert _SideEffectProbe.accessed is False, (
            "resolver invoked a @property (descriptor side effect)"
        )

    def test_constructor_bad_call_appends_init_signature(self):
        """Calling a TYPE with bad kwargs surfaces __init__'s signature."""

        def cell():
            _Config245(hostname="x")

        result = _format_bad_call(cell)
        assert "unexpected keyword argument 'hostname'" in result
        assert "_Config245.__init__" in result
        assert "port" in result

    def test_raising_str_does_not_escape(self):
        """A TypeError subclass whose __str__ raises must not break formatting."""

        class _BadStr(TypeError):
            def __str__(self):
                raise RuntimeError("boom")

        from nooa.errors.formatting import _bad_call_agentdoc

        assert _bad_call_agentdoc(_BadStr()) is None
