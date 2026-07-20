# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TruncatingStringIO."""

from nooa.agentdoc import TruncatingStringIO


class TestTruncatingStringIO:
    """Tests for TruncatingStringIO class."""

    def test_small_output_not_truncated(self):
        """Small output should not be truncated."""
        buffer = TruncatingStringIO(limit=100)
        buffer.write("hello world")
        result = buffer.getvalue()

        assert result == "hello world"
        assert not buffer.was_truncated

    def test_exact_limit_not_truncated(self):
        """Output at exact limit should not be truncated."""
        buffer = TruncatingStringIO(limit=10)
        buffer.write("1234567890")
        result = buffer.getvalue()

        assert result == "1234567890"
        assert not buffer.was_truncated

    def test_exceeds_limit_truncates(self):
        """Output exceeding limit should be truncated."""
        buffer = TruncatingStringIO(limit=10)
        buffer.write("12345678901234567890")
        result = buffer.getvalue()

        # Should have prose truncation notice
        assert "Output too large" in result
        assert buffer.was_truncated

    def test_truncation_notice_prepended(self):
        """Truncation notice should appear at the start of output."""
        buffer = TruncatingStringIO(limit=10)
        buffer.write("x" * 100)
        result = buffer.getvalue()

        # Notice is wrapped in <truncated-output> tag
        assert result.startswith("<truncated-output>")
        assert "Output too large" in result
        assert "100 chars" in result
        # Head and tail portions shown
        assert "Showing first" in result
        assert "and last" in result
        assert "chars not shown" in result

    def test_truncation_notice_format(self):
        """Truncation notice must use the new prose format."""
        buffer = TruncatingStringIO(limit=5)
        buffer.write("x" * 100)
        result = buffer.getvalue()

        assert "Output too large" in result
        assert "100 chars" in result
        assert "chars not shown" in result

    def test_multiple_writes_accumulate(self):
        """Multiple writes should accumulate towards the limit."""
        buffer = TruncatingStringIO(limit=20)
        buffer.write("hello ")
        buffer.write("world ")
        buffer.write("this is a test")
        result = buffer.getvalue()

        # Total: 6 + 6 + 14 = 26 chars, limit is 20
        assert buffer.was_truncated
        assert "26 chars" in result

    def test_writes_after_truncation_still_counted(self):
        """Writes after head fills should still be counted for total."""
        buffer = TruncatingStringIO(limit=10)
        buffer.write("12345678901234567890")  # 20 chars
        buffer.write("extra")  # 5 more chars

        result = buffer.getvalue()
        # Should report 25 total chars
        assert "25 chars" in result

    def test_default_limit_50kb(self):
        """Default limit should be 50,000 chars."""
        buffer = TruncatingStringIO()
        # Write exactly 50KB
        buffer.write("x" * 50_000)
        assert not buffer.was_truncated

        # Write one more char to exceed
        buffer.write("y")
        assert buffer.was_truncated

    def test_custom_limit(self):
        """Custom limits should work."""
        buffer = TruncatingStringIO(limit=5)
        buffer.write("123456789")
        result = buffer.getvalue()

        assert buffer.was_truncated
        assert "9 chars" in result
        assert "Output too large" in result

    def test_empty_buffer(self):
        """Empty buffer should return empty string."""
        buffer = TruncatingStringIO()
        result = buffer.getvalue()

        assert result == ""
        assert not buffer.was_truncated

    def test_unicode_handling(self):
        """Unicode characters should be handled correctly."""
        buffer = TruncatingStringIO(limit=20)
        buffer.write("Hello 世界! 🚀")
        result = buffer.getvalue()

        # Should not be truncated if under limit
        assert not buffer.was_truncated
        assert "世界" in result
        assert "🚀" in result

    def test_newlines_preserved(self):
        """Newlines should be preserved in output."""
        buffer = TruncatingStringIO(limit=100)
        buffer.write("line 1\nline 2\nline 3\n")
        result = buffer.getvalue()

        assert result == "line 1\nline 2\nline 3\n"
        assert result.count("\n") == 3

    def test_truncation_in_middle_of_write(self):
        """Truncation should happen mid-write if necessary."""
        buffer = TruncatingStringIO(limit=10)
        # First write fits
        buffer.write("12345")
        assert not buffer.was_truncated

        # Second write exceeds limit
        buffer.write("67890ABCDE")
        assert buffer.was_truncated

        result = buffer.getvalue()
        # Head portion should contain the first 8 chars (80% of 10)
        assert "12345" in result

    def test_head_and_tail_preserved(self):
        """Both head and tail content should appear in truncated output."""
        # Use a limit large enough so both 'HEAD' and 'TAIL' words are preserved.
        # With limit=50 (default 50/50 split): head_limit=25, tail_limit=25.
        # Total write: "HEADDATA" (8) + "x" * 200 + "TAILCONTENT" (11) = 219 chars.
        buffer = TruncatingStringIO(limit=50)
        buffer.write("HEADDATA" + "x" * 200 + "TAILCONTENT")
        result = buffer.getvalue()

        assert buffer.was_truncated
        assert "Output too large" in result
        # Head portion (first 25 chars) must contain "HEADDATA" (first 8 chars)
        assert "HEADDATA" in result
        # Tail portion (last 25 chars) must contain "TAILCONTENT" (last 11 chars)
        assert "TAILCONTENT" in result


class TestTruncatingStringIOTailChars:
    """Tests for the configurable tail_chars parameter."""

    def test_default_is_50_50_split(self):
        buf = TruncatingStringIO(limit=100)
        assert buf._tail_limit == 50
        assert buf._head_limit == 50

    def test_custom_tail_chars(self):
        buf = TruncatingStringIO(limit=100, tail_chars=30)
        assert buf._tail_limit == 30
        assert buf._head_limit == 70

    def test_custom_tail_chars_content(self):
        """With tail_chars=30, limit=100: head=70, tail=30 — verify exact split."""
        buf = TruncatingStringIO(limit=100, tail_chars=30)
        # Write 160 chars: 70 H's fill the head exactly, 60 M's overflow and
        # are evicted from the tail, 30 T's remain in the tail window.
        buf.write("H" * 70 + "M" * 60 + "T" * 30)
        assert buf.was_truncated
        val = buf.getvalue()
        # Head must contain exactly the first 70 chars (all H's)
        assert "H" * 70 in val
        # Tail must contain exactly the last 30 chars (all T's)
        assert "T" * 30 in val
        # Middle section (M's) must be dropped
        assert "M" not in val

    def test_tail_chars_none_defaults_to_half(self):
        buf = TruncatingStringIO(limit=100, tail_chars=None)
        assert buf._tail_limit == 50
        assert buf._head_limit == 50
