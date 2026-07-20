# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for truncating_pformat — string fast-path + pformat with structural bounds.

Block-level head/tail string truncation has been removed. Strings now pass
through verbatim regardless of length; per-value bounds for non-strings come
from kwargs (``max_string`` / ``max_length`` / ``max_depth``). The optional
``max_chars`` parameter only fires as a non-string OOM-safety net.
"""

import pytest

from nooa.context_blocks.utils import truncating_pformat


class TestStringPassthrough:
    """Strings are returned verbatim — no truncation, no wrapper."""

    def test_small_string_unchanged(self):
        result = truncating_pformat("hello world")
        assert result == "hello world"

    def test_empty_string(self):
        assert truncating_pformat("") == ""

    def test_large_string_passes_through_verbatim(self):
        """Even a 2 MB string is returned as-is — no head/tail squash."""
        huge = "x" * 2_000_000
        result = truncating_pformat(huge)
        assert result == huge

    def test_max_chars_does_not_truncate_strings(self):
        """max_chars is an OOM net for non-strings; it does NOT cap strings."""
        s = "y" * 10_000
        result = truncating_pformat(s, max_chars=100)
        assert result == s


class TestNonStringPformat:
    """Non-strings go through pformat with the supplied structural kwargs."""

    def test_small_list_unchanged(self):
        result = truncating_pformat([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_none_value(self):
        assert truncating_pformat(None) == "None"

    def test_max_string_bounds_inner_strings(self):
        """A list of long strings can be bounded via max_string per-element."""
        big = ["x" * 10_000]
        result = truncating_pformat(big, max_string=50)
        # Long string renders as a marker, not the full content
        assert "str(len=" in result

    def test_max_length_bounds_container(self):
        """max_length head/tail-truncates a long list with the marker family."""
        result = truncating_pformat(list(range(100)), max_length=10)
        assert "list(len=100" in result
        assert ", ".join(str(i) for i in range(50)) not in result  # middle hidden

    def test_max_depth_bounds_nesting(self):
        nested = {"a": {"b": {"c": {"d": "leaf"}}}}
        result = truncating_pformat(nested, max_depth=2)
        assert "leaf" not in result


class TestOOMSafetyNet:
    """When max_chars is set on a non-string render, TruncatingStringIO
    stops the in-memory buffer from growing without bound."""

    def test_max_chars_caps_nonstring_render(self):
        big = list(range(1_000_000))
        result = truncating_pformat(big, max_chars=5_000)
        # Output is bounded well below the full repr (~7 MB).
        assert len(result) < 10_000

    def test_no_max_chars_means_unlimited(self):
        """With max_chars=None (default), the renderer doesn't cap output."""
        # 100K-element int list — full repr is ~600KB.
        big = list(range(100_000))
        result = truncating_pformat(big)
        # Render unbounded; content present.
        assert "0" in result
        assert "99999" in result

    def test_rejects_max_chars_zero(self):
        with pytest.raises(ValueError, match="max_chars must be > 0 or None"):
            truncating_pformat([1, 2, 3], max_chars=0)

    def test_rejects_max_chars_negative(self):
        with pytest.raises(ValueError, match="max_chars must be > 0 or None"):
            truncating_pformat([1, 2, 3], max_chars=-1)
