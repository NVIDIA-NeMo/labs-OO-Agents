# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""truncating_pformat's hard cap: a bounded nooa.snapshot TextStream, not post-cap slicing.

A rendering past ``max_chars`` keeps its first and last halves around
render_output's ``[... N chars omitted; full output not kept]`` line.
"""

from nooa.agentdoc import pformat, truncating_pformat


class TestTruncatingPformatCap:
    """truncating_pformat: a value captured within max_chars, then head and tail around
    render_output's line if its rendering is still longer."""

    def test_small_value_passes_through_unchanged(self):
        result = truncating_pformat([1, 2, 3], max_chars=1000)
        assert result == "[1, 2, 3]"

    def test_large_non_string_is_a_bounded_snapshot_preview(self):
        """The value is captured within max_chars: the preview says what it left out."""
        big = list(range(100_000))
        result = truncating_pformat(big, max_chars=200)
        # The snapshot keeps both ends of the list and says how long it is; its
        # rendering, still over 200 chars, keeps 100 + 100 around the cut line.
        head, cut, tail = result.partition(" chars omitted; full output not kept]\n")
        assert cut and head.startswith("list(len=100000,\n    [:")
        assert tail.endswith("        99999,\n    ],\n)")
        assert len(head.rsplit("\n[... ", 1)[0]) == 100 and len(tail) == 100
        assert "Output too large" not in result and "<truncated-output>" not in result

    def test_odd_cap_gives_the_extra_char_to_the_head(self):
        result = truncating_pformat(list(range(1_000)), max_chars=101)
        head, cut, tail = result.partition(" chars omitted; full output not kept]\n")
        assert cut and len(head.rsplit("\n[... ", 1)[0]) == 51 and len(tail) == 50

    def test_memory_and_time_follow_the_output_not_the_input(self):
        import time
        import tracemalloc

        truncating_pformat([1], max_chars=100)  # warm imports
        peaks = []
        for count in (1_000, 20_000):
            value = ["x" * 1_000] * count
            tracemalloc.start()
            started = time.perf_counter()
            result = truncating_pformat(value, max_chars=100, expand_all=True)
            elapsed = time.perf_counter() - started
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            assert len(result) < 200 and elapsed < 0.5
        assert max(peaks) < 200_000  # was 4.4 MB / 86 MB: the whole rendering
        assert peaks[1] < 2 * peaks[0]  # 20x the input, about the same memory

    def test_unbounded_is_plain_pformat(self):
        big = list(range(1_000))
        assert truncating_pformat(big) == pformat(big)

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
        assert result.startswith("list(len=10000,\n    [:")
        assert "\n        0,\n        1,\n" in result
        assert result.endswith("        9999,\n    ],\n)")
