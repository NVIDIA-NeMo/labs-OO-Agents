# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for edit tool improvements.

Tests replay real failure patterns observed across 72 sessions (171 edit failures).
Each test simulates a file + old_str that would have failed before the improvements.
"""

import pytest

from nooa.tools.shell_tools_legacy import (
    ShellToolsLegacy as ShellTools,
)
from nooa.tools.shell_tools_legacy import (
    _find_closest_match,
    _strip_line_number_prefixes,
)

# ============================================================
# Pattern 1: Trailing whitespace mismatch (most common failure)
# Agent copies code from view() output but trailing spaces differ
# ============================================================


class TestFuzzyWhitespaceMismatch:
    """Trailing whitespace in old_str that doesn't match the file."""

    @pytest.fixture
    async def shell(self, tmp_path):
        s = ShellTools(cwd=str(tmp_path))
        yield s
        await s.close()

    @pytest.fixture
    def sample_file(self, tmp_path):
        # File has no trailing whitespace
        content = """class StudentGrades:
    def __init__(self):
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def average(self):
        return sum(self.scores) / len(self.scores)
"""
        f = tmp_path / "grades.py"
        f.write_text(content)
        return f

    @pytest.mark.asyncio
    async def test_trailing_whitespace_fuzzy_match(self, shell, sample_file):
        """Agent's old_str has trailing spaces on lines — should fuzzy match."""
        # Simulate: agent copied code but added trailing whitespace
        old_with_trailing = (
            "    def average(self):  \n        return sum(self.scores) / len(self.scores)  "
        )
        result = await shell.edit(
            sample_file.name,
            old_str=old_with_trailing,
            new_str="    def average(self):\n        if not self.scores:\n            return 0\n        return sum(self.scores) / len(self.scores)",
        )
        assert result.success, f"Expected fuzzy match success, got: {result.error}"

    @pytest.mark.asyncio
    async def test_smart_quotes_fuzzy_match(self, shell, tmp_path):
        """Agent uses smart quotes instead of ASCII quotes."""
        content = 'print("hello world")\n'
        f = tmp_path / "quotes.py"
        f.write_text(content)
        # Simulate: model outputs smart quotes
        result = await shell.edit(
            "quotes.py",
            old_str="print(\u201chello world\u201d)",
            new_str='print("goodbye world")',
        )
        assert result.success, f"Expected fuzzy match for smart quotes, got: {result.error}"


# ============================================================
# Pattern 2: Line-number prefixes from view() output
# Agent copies `  42|` prefixes directly into old_str
# ============================================================


class TestLineNumberPrefixStripping:
    """Agent copies line-number prefixes from view() output into old_str."""

    @pytest.fixture
    async def shell(self, tmp_path):
        s = ShellTools(cwd=str(tmp_path))
        yield s
        await s.close()

    def test_strip_line_prefixes_basic(self):
        text = " 1|def hello():\n 2|    pass\n 3|"
        result = _strip_line_number_prefixes(text)
        assert "1|" not in result
        assert "def hello():" in result
        assert "    pass" in result

    def test_strip_line_prefixes_wide_numbers(self):
        text = "100|def foo():\n101|    return 1\n102|"
        result = _strip_line_number_prefixes(text)
        assert "100|" not in result
        assert "def foo():" in result

    def test_no_strip_when_not_prefixed(self):
        text = "x = 1|2  # bitwise or"
        result = _strip_line_number_prefixes(text)
        assert result == text  # should not strip — not a prefix pattern

    @pytest.mark.asyncio
    async def test_edit_with_line_prefixes_succeeds(self, shell, tmp_path):
        """Agent pastes view() output with line numbers as old_str."""
        content = 'def hello():\n    print("hi")\n    return True\n'
        f = tmp_path / "hello.py"
        f.write_text(content)

        # Agent copied from view() output
        old_with_prefixes = '1|def hello():\n2|    print("hi")\n3|    return True'
        result = await shell.edit(
            "hello.py",
            old_str=old_with_prefixes,
            new_str='def hello():\n    print("hello")\n    return True',
        )
        assert result.success, f"Expected line-prefix stripping to work, got: {result.error}"


# ============================================================
# Pattern 3: Multiple matches — generic old_str not unique
# ============================================================


class TestMultipleMatchesError:
    """Agent targets a short generic snippet that matches multiple times."""

    @pytest.fixture
    async def shell(self, tmp_path):
        s = ShellTools(cwd=str(tmp_path))
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_multiple_matches_reports_count(self, shell, tmp_path):
        content = "pass\npass\npass\n"
        f = tmp_path / "multi.py"
        f.write_text(content)
        result = await shell.edit("multi.py", old_str="pass", new_str="return None")
        assert not result.success
        assert "3 times" in result.error


# ============================================================
# Pattern 4: Closest match hint on failure
# ============================================================


class TestClosestMatchHint:
    """When old_str not found, error should include closest match."""

    def test_find_closest_match_basic(self):
        content = 'def hello():\n    print("hi")\n'
        target = 'def hello():\n    print("hello")\n'
        match = _find_closest_match(content, target)
        assert match is not None
        assert "hello" in match

    def test_find_closest_match_no_match(self):
        content = "x = 1\ny = 2\n"
        target = "completely unrelated long string that has nothing in common with the file"
        match = _find_closest_match(content, target, threshold=0.6)
        assert match is None


# ============================================================
# Pattern 5: Write truncation guard
# ============================================================


class TestWriteTruncationGuard:
    """shell.write should warn when file shrinks significantly."""

    @pytest.fixture
    async def shell(self, tmp_path):
        s = ShellTools(cwd=str(tmp_path))
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_write_truncation_warning(self, shell, tmp_path):
        """Writing a much shorter file should include a warning."""
        original = "x = 1\n" * 100  # 600 chars
        f = tmp_path / "big.py"
        f.write_text(original)
        result = await shell.write("big.py", "x = 1\n")  # shrink to 6 chars
        assert "WARNING" in result.diff
        assert "shrunk" in result.diff.lower() or "shrunk" in result.diff

    @pytest.mark.asyncio
    async def test_write_no_warning_small_change(self, shell, tmp_path):
        """Small size changes should NOT trigger warning."""
        original = "x = 1\ny = 2\nz = 3\n"
        f = tmp_path / "small.py"
        f.write_text(original)
        result = await shell.write("small.py", "x = 1\ny = 2\n")
        assert "WARNING" not in (result.diff or "")
