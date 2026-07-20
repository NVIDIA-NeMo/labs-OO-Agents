# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared response cleanup functions."""

from nooa.runtime.response_cleanup import strip_code_fences, strip_xml_wrapper


class TestStripCodeFences:
    """Test strip_code_fences shared function."""

    def test_strips_python_fence(self):
        code = "```python\nx = 1\n```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == "x = 1"
        assert token == "```python"

    def test_strips_py_fence(self):
        code = "```py\nx = 1\n```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == "x = 1"
        assert token == "```py"

    def test_strips_bare_fence(self):
        code = "```\nx = 1\n```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == "x = 1"
        assert token == "```"

    def test_strips_any_language_fence(self):
        code = "```bash\necho hello\n```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == "echo hello"
        assert token == "```bash"

    def test_no_fences_returns_original(self):
        code = "x = 1\ny = 2"
        cleaned, token = strip_code_fences(code)
        assert cleaned == code
        assert token is None

    def test_unbalanced_fences_not_stripped(self):
        code = "```python\nx = 1"
        cleaned, token = strip_code_fences(code)
        assert cleaned == code
        assert token is None

    def test_multiline_code(self):
        code = "```python\nimport os\nx = 1\nprint(x)\n```"
        cleaned, token = strip_code_fences(code)
        assert "import os" in cleaned
        assert "print(x)" in cleaned
        assert "```" not in cleaned

    def test_idempotent(self):
        code = "x = 1"
        cleaned1, _ = strip_code_fences(code)
        cleaned2, _ = strip_code_fences(cleaned1)
        assert cleaned1 == cleaned2

    def test_strips_surrounding_whitespace(self):
        code = "  ```python\nx = 1\n```  "
        cleaned, token = strip_code_fences(code)
        assert cleaned == "x = 1"
        assert token == "```python"

    def test_preserves_code_on_opening_line(self):
        """Regression: LLMs sometimes put code on the same line as the opening fence.
        That code must not be silently deleted."""
        code = "```python print('hello')\n```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == "print('hello')"
        assert token == "```python"

    def test_preserves_code_on_opening_line_multiline(self):
        code = "```python x = 1\ny = 2\n```"
        cleaned, token = strip_code_fences(code)
        assert "x = 1" in cleaned
        assert "y = 2" in cleaned
        assert token == "```python"

    def test_inline_form_with_code(self):
        """Inline form: ```python x = 1``` (no newline between lang and closing)."""
        code = "```python x = 1```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == "x = 1"
        assert token == "```python"

    def test_strips_crlf_line_endings(self):
        """Windows line endings should work."""
        code = "```python\r\nx = 1\r\n```"
        cleaned, token = strip_code_fences(code)
        assert "x = 1" in cleaned
        assert token == "```python"

    def test_closing_only_fence_not_stripped(self):
        code = "x = 1\n```"
        cleaned, token = strip_code_fences(code)
        assert cleaned == code
        assert token is None

    def test_language_tag_with_space_junk_falls_through_to_inline(self):
        """'```python 3\\n...' is a known edge case: the multiline pattern
        rejects it (non-whitespace after lang tag), the inline pattern then
        matches with the 'python' lang and leaks the ' 3' into the body.

        This leak is accepted — the priority is to NOT delete real code.
        This test pins the current behavior."""
        code = "```python 3\nprint('hello')\n```"
        cleaned, token = strip_code_fences(code)
        assert token == "```python"
        # Key assertion: real code is preserved (no deletion)
        assert "print('hello')" in cleaned


class TestStripXmlWrapper:
    """Test strip_xml_wrapper shared function."""

    def test_strips_tag_with_attributes(self):
        content = '<assistant_message expr="test">{"key": "value"}</assistant_message>'
        inner, tag = strip_xml_wrapper(content)
        assert inner == '{"key": "value"}'
        assert tag == "assistant_message"

    def test_strips_tag_without_attributes(self):
        content = "<tool_code>x = 1</tool_code>"
        inner, tag = strip_xml_wrapper(content)
        assert inner == "x = 1"
        assert tag == "tool_code"

    def test_strips_tag_with_hyphen(self):
        content = "<my-tag>content</my-tag>"
        inner, tag = strip_xml_wrapper(content)
        assert inner == "content"
        assert tag == "my-tag"

    def test_no_xml_returns_original(self):
        content = "just plain text"
        inner, tag = strip_xml_wrapper(content)
        assert inner == content
        assert tag is None

    def test_not_starting_with_angle_bracket(self):
        content = "text <tag>inner</tag>"
        inner, tag = strip_xml_wrapper(content)
        assert inner == content
        assert tag is None

    def test_mismatched_tags_not_stripped(self):
        content = "<open>content</close>"
        inner, tag = strip_xml_wrapper(content)
        assert inner == content
        assert tag is None

    def test_multiline_content(self):
        content = "<code>line1\nline2\nline3</code>"
        inner, tag = strip_xml_wrapper(content)
        assert inner == "line1\nline2\nline3"
        assert tag == "code"

    def test_idempotent(self):
        content = "just text"
        inner1, _ = strip_xml_wrapper(content)
        inner2, _ = strip_xml_wrapper(inner1)
        assert inner1 == inner2

    def test_strips_surrounding_whitespace(self):
        content = "  <tag>content</tag>  "
        inner, tag = strip_xml_wrapper(content)
        assert inner == "content"
        assert tag == "tag"

    def test_empty_string(self):
        inner, tag = strip_xml_wrapper("")
        assert inner == ""
        assert tag is None

    def test_empty_content_still_strips(self):
        """Empty content <tag></tag> should still match and return ''."""
        inner, tag = strip_xml_wrapper("<tag></tag>")
        assert inner == ""
        assert tag == "tag"

    def test_self_closing_tag_not_stripped(self):
        """<tag/> self-closing syntax should not match the open/close pattern."""
        inner, tag = strip_xml_wrapper("<tag/>")
        assert inner == "<tag/>"
        assert tag is None

    def test_nested_strips_only_outermost(self):
        """strip_xml_wrapper returns the raw inner content — callers handle nesting."""
        content = "<outer><inner>x</inner></outer>"
        inner, tag = strip_xml_wrapper(content)
        assert inner == "<inner>x</inner>"
        assert tag == "outer"
