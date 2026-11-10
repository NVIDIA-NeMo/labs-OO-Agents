# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""truncating_pformat's hard cap: a bounded nooa.snapshot TextStream, not post-cap slicing.

A rendering past ``max_chars`` keeps its first and last halves around
render_output's ``[... N chars omitted; full output not kept]`` line.
"""

from nooa.agentdoc import pformat, truncating_pformat


class TestTruncatingPformatCap:
    """truncating_pformat keeps head and tail around render_output's line when the cap fires."""

    def test_small_value_passes_through_unchanged(self):
        result = truncating_pformat([1, 2, 3], max_chars=1000)
        assert result == "[1, 2, 3]"

    def test_large_non_string_uses_the_snapshot_cut_line(self):
        big = list(range(100_000))
        whole = pformat(big)
        result = truncating_pformat(big, max_chars=200)
        assert result == (
            whole[:100]
            + f"\n[... {len(whole) - 200:,} chars omitted; full output not kept]\n"
            + whole[-100:]
        )
        assert "Output too large" not in result and "<truncated-output>" not in result

    def test_odd_cap_gives_the_extra_char_to_the_head(self):
        big = list(range(1_000))
        whole = pformat(big)
        result = truncating_pformat(big, max_chars=101)
        head, cut, tail = result.partition(
            f"\n[... {len(whole) - 101:,} chars omitted; full output not kept]\n"
        )
        assert cut and head == whole[:51] and tail == whole[-50:]

    def test_large_non_string_preserves_head_and_tail(self):
        head_marker = "START_UNIQUE_MARKER"
        tail_marker = "END_UNIQUE_MARKER"
        big = [head_marker] + ["x"] * 10_000 + [tail_marker]
        result = truncating_pformat(big, max_chars=300)
        assert head_marker in result
        assert tail_marker in result

    def test_strings_pass_through_verbatim(self):
        """Block-level string truncation has been removed. Strings now pass
        through truncating_pformat verbatim regardless of max_chars (which is
        only an OOM-safety net for non-string rendering)."""
        big = "a" * 10_000
        result = truncating_pformat(big, max_chars=100)
        assert result == big

    def test_string_fast_path_unchanged(self):
        """String fast-path: returns input verbatim regardless of length / cap."""
        huge_str = "START" + "x" * 10_000 + "END"
        result = truncating_pformat(huge_str, max_chars=100)
        assert result == huge_str

    def test_cut_keeps_a_clean_head_and_tail(self):
        big = list(range(10_000))
        result = truncating_pformat(big, max_chars=500)
        assert result.startswith("[\n    0,")
        assert result.endswith("    9999,\n]")
