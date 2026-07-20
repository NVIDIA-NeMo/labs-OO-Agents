# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for result types."""

from nooa.tools._results import (
    EditResult,
    LsResult,
    RunResult,
    SearchResult,
    ViewResult,
    WriteResult,
)


class TestRunResult:
    def test_success(self):
        r = RunResult(stdout="hello", stderr="", returncode=0)
        assert r.success is True
        assert r.text == "hello"

    def test_failure(self):
        r = RunResult(stdout="", stderr="error", returncode=1)
        assert r.success is False
        assert "[stderr]" in r.text
        assert "[exit code: 1]" in r.text

    def test_timeout(self):
        r = RunResult(stdout="partial", stderr="", returncode=1, timed_out=True)
        assert "[timed out]" in r.text

    def test_empty_output(self):
        r = RunResult(stdout="", stderr="", returncode=0)
        assert r.text == "(no output)"

    def test_str(self):
        r = RunResult(stdout="hello", stderr="", returncode=0)
        assert str(r) == "hello"


class TestViewResult:
    def test_basic(self):
        r = ViewResult(
            path="foo.py",
            content="1|hello\n2|world",
            start_line=1,
            end_line=2,
            total_lines=2,
        )
        assert "[foo.py]" in r.text
        assert "lines 1-2 of 2" in r.text
        assert "hello" in r.text

    def test_truncated(self):
        r = ViewResult(
            path="big.py",
            content="...",
            start_line=1,
            end_line=100,
            total_lines=5000,
            truncated=True,
        )
        assert "(truncated)" in r.text


class TestEditResult:
    def test_success(self):
        r = EditResult(path="foo.py", diff="- old\n+ new", success=True)
        assert "Edited foo.py" in r.text
        assert "- old" in r.text

    def test_failure(self):
        r = EditResult(path="foo.py", diff="", success=False, error="not found")
        assert "Edit failed" in r.text
        assert "not found" in r.text

    def test_lint_errors(self):
        r = EditResult(path="foo.py", diff="- old\n+ new", lint_errors=["E001: syntax"])
        assert "lint errors" in r.text
        assert "E001" in r.text


class TestWriteResult:
    def test_created(self):
        r = WriteResult(path="new.py", created=True, lines=10)
        assert "Created new.py" in r.text

    def test_overwritten(self):
        r = WriteResult(path="old.py", created=False, lines=5)
        assert "Wrote old.py" in r.text


class TestSearchResult:
    def test_no_matches(self):
        r = SearchResult(matches=[], total_matches=0)
        assert "No matches" in r.text

    def test_matches(self):
        r = SearchResult(matches=["foo.py:1:hello", "bar.py:2:world"], total_matches=2)
        assert "foo.py:1:hello" in r.text
        assert "(2 matches)" in r.text

    def test_truncated(self):
        r = SearchResult(matches=["a", "b"], total_matches=100, truncated=True)
        assert "100 total matches" in r.text
        assert "showing first 2" in r.text


class TestLsResult:
    def test_basic(self):
        r = LsResult(path="src/", tree="src/\n├── main.py", num_files=1)
        assert "[src/]" in r.text
        assert "main.py" in r.text

    def test_truncated(self):
        r = LsResult(path=".", tree="...", num_files=500, truncated=True)
        assert "(truncated)" in r.text
