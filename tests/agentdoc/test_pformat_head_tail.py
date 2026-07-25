# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the truncation 3.0 marker family in pformat.

When max_length fires, pformat emits a `type(len=N, ...)` marker:

* Ordered (list, tuple) — slice-keys notation:
  ``list(len=100, [:5]=[0, 1, 2, 3, 4], [-5:]=[95, 96, 97, 98, 99])``
* Unordered (dict, set, frozenset) — items wrapper:
  ``dict(len=100, items={0: 0, 1: 10, ..., 98: 980, 99: 990})``

Untruncated values render as plain Python literals (`[1, 2, 3]`, `{1: 2}`,
etc.) — the marker's *presence* is the truncation signal.
"""

import re

from nooa.agentdoc import pformat, truncating_pformat


class TestListMarker:
    """Lists: slice-keys marker when truncated, plain repr when complete."""

    def test_truncated_list_uses_slice_keys_marker(self):
        items = list(range(100))
        result = pformat(items, max_length=10)
        # Marker prefix
        assert result.startswith("list(len=100,")
        # Slice keys with explicit head/tail counts
        assert "[:5]=" in result
        assert "[-5:]=" in result
        # Visible head/tail items
        for x in [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]:
            assert re.search(rf"\b{x}\b", result), result

    def test_truncated_list_excludes_elided_items(self):
        items = list(range(200))
        result = pformat(items, max_length=10)
        # Items 5..194 are in the elided middle; pick a non-prefix sample
        assert not re.search(r"\b194\b", result)

    def test_complete_list_is_plain_python_literal(self):
        result = pformat([1, 2, 3], max_length=10)
        assert result == "[1, 2, 3]"

    def test_list_exactly_max_length_is_plain(self):
        items = list(range(10))
        result = pformat(items, max_length=10)
        assert "len=" not in result
        assert result.startswith("[")
        assert result.endswith("]")

    def test_just_over_max_length_uses_marker(self):
        items = list(range(11))
        result = pformat(items, max_length=10)
        assert "list(len=11," in result


class TestTupleMarker:
    """Tuples: slice-keys marker with tuple parens for the chunks."""

    def test_truncated_tuple_uses_tuple_marker(self):
        result = pformat(tuple(range(100)), max_length=10)
        assert result.startswith("tuple(len=100,")
        assert "[:5]=(" in result
        assert "[-5:]=(" in result
        assert "99" in result

    def test_complete_tuple_plain(self):
        assert pformat((1, 2, 3)) == "(1, 2, 3)"


class TestDictMarker:
    """Dicts: items wrapper when truncated, plain repr when complete."""

    def test_truncated_dict_uses_items_marker(self):
        d = {str(i): i for i in range(100)}
        result = pformat(d, max_length=10)
        assert result.startswith("dict(len=100, items={")
        assert result.endswith("})")
        # Head and tail keys appear
        assert "'0'" in result
        assert "'99'" in result
        # An explicit elision separator between head and tail
        assert "..." in result

    def test_complete_dict_plain(self):
        result = pformat({"a": 1, "b": 2}, max_length=10)
        assert result == "{'a': 1, 'b': 2}"


class TestSetMarker:
    """Sets: items wrapper when truncated."""

    def test_truncated_set_uses_items_marker(self):
        result = pformat(set(range(100)), max_length=10)
        assert result.startswith("set(len=100, items={")
        assert result.endswith("})")

    def test_truncated_frozenset_uses_items_marker(self):
        result = pformat(frozenset(range(100)), max_length=10)
        assert result.startswith("frozenset(len=100, items=")

    def test_complete_set_plain(self):
        result = pformat({1, 2, 3}, max_length=10)
        # Sets aren't insertion-ordered for ints<sys.maxsize but are deterministic
        # under CPython; check structure rather than exact ordering.
        assert result.startswith("{")
        assert result.endswith("}")
        for x in (1, 2, 3):
            assert str(x) in result


class TestEmptyAndEdge:
    def test_empty_list(self):
        assert pformat([]) == "[]"

    def test_empty_dict(self):
        assert pformat({}) == "{}"

    def test_max_length_one(self):
        # head=1, tail=0 — only the head slice is rendered
        result = pformat(list(range(100)), max_length=1)
        assert "list(len=100, [:1]=[0])" == result

    def test_max_length_two(self):
        # head=1, tail=1
        result = pformat(list(range(100)), max_length=2)
        assert "list(len=100, [:1]=[0], [-1:]=[99])" == result


class TestMaxCharsTruncation:
    """max_chars cap belongs to truncating_pformat, not pformat — still bounded."""

    def test_max_chars_bounds_output(self):
        items = list(range(1_000_000))
        result = truncating_pformat(items, max_chars=1000)
        assert len(result) < 10_000


class TestExpandedFormat:
    """Expanded (multi-line) form mirrors the compact marker shape."""

    def test_expanded_list_marker(self):
        result = pformat(list(range(20)), max_length=6, expand_all=True)
        assert result.startswith("list(len=20,")
        assert "[:3]=[" in result
        assert "[-3:]=[" in result
        assert result.endswith(")")
        assert ",]" not in result
        assert "0" in result
        assert "19" in result

    def test_expanded_dict_marker(self):
        d = {str(i): i for i in range(20)}
        result = pformat(d, max_length=6, expand_all=True)
        assert result.startswith("dict(len=20, items={")
        assert result.endswith("})")
        assert "'0'" in result
        assert "'19'" in result


class TestNestedContainers:
    """Inner containers use their own max_length independently."""

    def test_outer_truncated_inner_complete(self):
        outer = [[i, i + 1] for i in range(50)]
        result = pformat(outer, max_length=10)
        # Outer marker
        assert "list(len=50," in result
        # Inner pairs are short — plain literal
        assert "[0, 1]" in result
        # Last visible inner pair (index 49 → [49, 50])
        assert "[49, 50]" in result
