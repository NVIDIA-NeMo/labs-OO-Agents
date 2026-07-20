# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pprint() and _pformat()."""

import io
import sys

from nooa.runtime.pprint import _pformat, pprint


class TestPformat:
    """Tests for _pformat() internal function."""

    def test_simple_string(self):
        """Simple strings should be repr'd."""
        result = _pformat("hello")
        assert result == "'hello'"

    def test_simple_int(self):
        """Simple integers should use repr."""
        result = _pformat(42)
        assert result == "42"

    def test_simple_list(self):
        """Simple list should be formatted."""
        result = _pformat([1, 2, 3])
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_simple_dict(self):
        """Simple dict should be formatted."""
        result = _pformat({"a": 1, "b": 2})
        assert "'a': 1" in result
        assert "'b': 2" in result

    def test_string_truncation(self):
        """Long strings use the truncation 3.0 marker: str(len=N, [:H]=..., [-T:]=...)."""
        long_string = "x" * 1000
        result = _pformat(long_string, max_string=50)

        assert result.startswith("str(len=1000,")  # marker carries exact length
        assert "[:25]=" in result  # head slice
        assert "[-25:]=" in result  # tail slice
        assert len(result) < 200  # Bounded by max_string + marker overhead

    def test_list_truncation(self):
        """Long lists are wrapped in the truncation 3.0 slice-keys marker."""
        long_list = list(range(100))
        result = _pformat(long_list, max_length=10)

        assert "list(len=100," in result  # marker carries the exact count
        assert "0" in result  # First item visible (head)
        assert "99" in result  # Last item visible (tail)

    def test_dict_truncation(self):
        """Long dicts are wrapped in the truncation 3.0 items marker."""
        long_dict = {f"key_{i}": i for i in range(100)}
        result = _pformat(long_dict, max_length=10)

        assert "dict(len=100, items=" in result

    def test_depth_truncation(self):
        """Nested structures should be truncated with max_depth."""
        nested = {"a": {"b": {"c": {"d": "deep"}}}}
        result = _pformat(nested, max_depth=2)

        # Should show first 2 levels but not deeper
        assert "'a'" in result
        assert "'b'" in result
        # At max depth, should show shallow representation
        assert "dict" in result or "items" in result

    def test_no_truncation_by_default(self):
        """Without limits, no truncation should occur."""
        data = {"items": list(range(100)), "text": "x" * 1000}
        result = _pformat(data)

        # Should include all data
        assert "99" in result  # Last list item
        assert "x" * 100 in result  # Long string present

    def test_nested_list_formatting(self):
        """Nested lists should be properly indented."""
        data = [[1, 2], [3, 4]]
        result = _pformat(data)

        # Should have proper structure
        assert "[" in result
        assert "]" in result

    def test_mixed_types(self):
        """Mixed types in containers should all format."""
        data = [1, "hello", 3.14, None, True, {"key": "value"}]
        result = _pformat(data)

        assert "1" in result
        assert "'hello'" in result
        assert "3.14" in result
        assert "None" in result
        assert "True" in result
        assert "'key': 'value'" in result

    def test_empty_containers(self):
        """Empty containers should format correctly."""
        assert _pformat([]) == "[]"
        assert _pformat({}) == "{}"
        assert _pformat(()) == "()"
        assert _pformat(set()) == "{}"

    def test_tuple_formatting(self):
        """Tuples should use parentheses."""
        result = _pformat((1, 2, 3))
        assert "(" in result
        assert ")" in result

    def test_set_formatting(self):
        """Sets should use curly braces."""
        result = _pformat({1, 2, 3})
        assert "{" in result
        assert "}" in result

    def test_compact_small_containers(self):
        """Small containers should format compactly when nested."""
        data = [{"a": 1, "b": 2}]
        result = _pformat(data, expand_all=False)

        # Small nested dict might be on one line
        assert "a" in result and "b" in result

    def test_expand_all(self):
        """expand_all should force vertical layout."""
        data = [1, 2, 3]
        result = _pformat(data, expand_all=True)

        # Should have multiple lines for small containers
        assert result.count("\n") > 0


class TestPprint:
    """Tests for pprint() public function."""

    def test_pprint_outputs_to_stdout(self):
        """pprint() should output to stdout."""
        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            pprint([1, 2, 3])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "1" in output
        assert "2" in output
        assert "3" in output

    def test_pprint_with_max_length(self):
        """pprint() should respect max_length."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            pprint(list(range(100)), max_length=5)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "list(len=100," in output  # truncation 3.0 marker

    def test_pprint_with_max_string(self):
        """pprint() should respect max_string."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            pprint("x" * 1000, max_string=50)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "str(len=1000," in output  # truncation 3.0 string marker

    def test_pprint_with_max_depth(self):
        """pprint() should respect max_depth."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            nested = {"a": {"b": {"c": "deep"}}}
            pprint(nested, max_depth=2)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # Should truncate at depth 2
        assert "'a'" in output

    def test_pprint_none_limits(self):
        """pprint() with None limits should not truncate (Rich default)."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            pprint(list(range(200)), max_length=None)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # Should include all items and no truncation notice
        assert "199" in output
        assert "not shown" not in output


class TestFormatBrackets:
    """Tests for _get_brackets() helper."""

    def test_list_brackets(self):
        """Lists should have square brackets."""
        result = _pformat([1, 2, 3])
        assert result.startswith("[")
        assert result.endswith("]")

    def test_tuple_brackets(self):
        """Tuples should have parentheses."""
        result = _pformat((1, 2, 3))
        assert "(" in result
        assert ")" in result

    def test_set_brackets(self):
        """Sets should have curly braces."""
        result = _pformat({1, 2, 3})
        assert "{" in result
        assert "}" in result

    def test_dict_brackets(self):
        """Dicts should have curly braces."""
        result = _pformat({"a": 1})
        assert "{" in result
        assert "}" in result


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_dataframe_like_structure(self):
        """Large dataframe-like structure should truncate well."""
        data = [
            {"id": i, "name": f"User {i}", "email": f"user{i}@example.com"} for i in range(1000)
        ]
        result = _pformat(data, max_length=5, max_string=50)

        assert "list(len=1000," in result  # truncation 3.0 marker
        assert "'id': 0" in result  # First item visible (head)
        assert "'id': 999" in result  # Last item visible (tail)

    def test_nested_api_response(self):
        """Nested API response should format clearly."""
        response = {
            "data": {"users": [{"id": 1}, {"id": 2}], "total": 100},
            "meta": {"page": 1, "limit": 50},
        }
        result = _pformat(response, max_depth=3)

        assert "'data'" in result
        assert "'meta'" in result
        assert "'users'" in result

    def test_long_log_message(self):
        """Long log messages should truncate."""
        log = {"message": "ERROR: " + "x" * 10000, "timestamp": "2024-01-01"}
        result = _pformat(log, max_string=100)

        # Message should be truncated but timestamp visible
        assert "'timestamp': '2024-01-01'" in result
        assert len(result) < 1000  # Much shorter than original
