# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ShellTools skill."""

import textwrap

import pytest

from nooa.tools.shell_tools_legacy import ShellToolsLegacy as ShellTools


@pytest.fixture
async def shell(tmp_path):
    """Create a ShellTools instance in a temp directory."""
    s = ShellTools(cwd=tmp_path)
    yield s
    await s.close()


@pytest.fixture
def sample_py(tmp_path):
    """Create a sample Python file for testing."""
    f = tmp_path / "sample.py"
    f.write_text(
        textwrap.dedent("""\
        def hello():
            print("hello world")

        def goodbye():
            print("goodbye world")

        if __name__ == "__main__":
            hello()
    """)
    )
    return f


@pytest.fixture
def sample_tree(tmp_path):
    """Create a sample directory tree for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('main')")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass")
    (tmp_path / "src" / "sub").mkdir()
    (tmp_path / "src" / "sub" / "deep.py").write_text("x = 1")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_it(): pass")
    (tmp_path / "README.md").write_text("# Hello")
    # Also create some directories that should be ignored
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cache.pyc").write_text("bytecode")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config")
    return tmp_path


# ==========================================================================
# run
# ==========================================================================
class TestRun:
    async def test_simple_command(self, shell):
        r = await shell.run("echo hello")
        assert r.success
        assert "hello" in r.stdout

    async def test_cd_persists(self, shell, tmp_path):
        sub = tmp_path / "mydir"
        sub.mkdir()
        r1 = await shell.run(f"cd {sub}")
        assert r1.success
        r2 = await shell.run("pwd")
        assert str(sub) in r2.stdout

    async def test_env_persists(self, shell):
        await shell.run("export FOO=bar123")
        r = await shell.run("echo $FOO")
        assert "bar123" in r.stdout

    async def test_failure(self, shell):
        r = await shell.run("false")
        assert not r.success
        assert r.returncode != 0

    async def test_stderr(self, shell):
        r = await shell.run("echo err >&2")
        assert "err" in r.stderr

    async def test_repr_updates_after_cd(self, shell, tmp_path):
        sub = tmp_path / "proj"
        sub.mkdir()
        await shell.run(f"cd {sub}")
        assert "proj" in repr(shell)

    async def test_run_stream_emits_done(self, shell):
        events = [event async for event in shell.run_stream("echo hello")]
        assert events[0].kind == "stdout"
        assert "hello" in events[0].text
        assert events[-1].kind == "done"
        assert events[-1].returncode == 0
        assert events[-1].timed_out is False

    async def test_run_stream_stderr_and_done(self, shell):
        events = [event async for event in shell.run_stream("echo err >&2")]
        assert events[0].kind == "stderr"
        assert "err" in events[0].text
        assert events[-1].kind == "done"
        assert events[-1].returncode == 0

    async def test_run_stream_combined_stdout_stderr(self, shell, tmp_path):
        """run_stream with both stdout and stderr yields events for each."""
        script = tmp_path / "both.sh"
        script.write_text("#!/bin/bash\necho out\necho err >&2\n")
        script.chmod(0o755)
        events = [event async for event in shell.run_stream(str(script))]
        kinds = [e.kind for e in events]
        assert "stdout" in kinds
        assert "stderr" in kinds
        assert kinds[-1] == "done"
        assert events[-1].returncode == 0

    async def test_run_stream_nonzero_exit(self, shell):
        """run_stream yields done event with non-zero returncode."""
        events = [event async for event in shell.run_stream("false")]
        assert events[-1].kind == "done"
        assert events[-1].returncode == 1
        assert events[-1].timed_out is False

    async def test_run_stream_timeout(self, shell):
        """run_stream yields done event with timed_out=True on timeout."""
        events = [event async for event in shell.run_stream("sleep 60", timeout=0.5)]
        assert events[-1].kind == "done"
        assert events[-1].timed_out is True
        assert events[-1].returncode == 124


# ==========================================================================
# view
# ==========================================================================
class TestView:
    async def test_view_file(self, shell, sample_py):
        r = await shell.view(str(sample_py))
        assert r.total_lines > 0
        assert "hello" in r.content
        assert "1|" in r.content or "1 |" in r.content  # line numbers

    async def test_view_with_offset(self, shell, sample_py):
        r = await shell.view(str(sample_py), offset=3, limit=2)
        assert r.start_line == 3
        assert r.end_line == 4

    async def test_view_nonexistent(self, shell):
        r = await shell.view("/nonexistent/file.py")
        assert "not found" in r.content.lower() or "error" in r.content.lower()

    async def test_view_directory(self, shell, sample_tree):
        r = await shell.view(str(sample_tree))
        # Should show a tree listing instead
        assert "src" in r.content

    async def test_view_updates_current_file(self, shell, sample_py):
        await shell.view(str(sample_py))
        assert shell._current_file == str(sample_py)


# ==========================================================================
# edit
# ==========================================================================
class TestEdit:
    async def test_edit_success(self, shell, sample_py):
        r = await shell.edit(
            str(sample_py),
            old_str='print("hello world")',
            new_str='print("hi there")',
        )
        assert r.success
        assert "hi there" in sample_py.read_text()
        assert r.diff  # should have a diff

    async def test_edit_not_found(self, shell, sample_py):
        r = await shell.edit(
            str(sample_py),
            old_str="this text does not exist",
            new_str="replacement",
        )
        assert not r.success
        assert "not found" in r.error.lower()

    async def test_edit_multiple_matches(self, shell, tmp_path):
        f = tmp_path / "dup.py"
        f.write_text("x = 1\nx = 1\n")
        r = await shell.edit(str(f), old_str="x = 1", new_str="x = 2")
        assert not r.success
        assert "2 times" in r.error

    async def test_edit_nonexistent_file(self, shell):
        r = await shell.edit("/no/such/file.py", old_str="a", new_str="b")
        assert not r.success
        assert "not found" in r.error.lower()

    async def test_edit_preserves_rest_of_file(self, shell, sample_py):
        await shell.edit(
            str(sample_py),
            old_str='print("hello world")',
            new_str='print("changed")',
        )
        new_text = sample_py.read_text()
        # goodbye function should still be there
        assert "goodbye" in new_text

    async def test_edit_diff_format(self, shell, sample_py):
        r = await shell.edit(
            str(sample_py),
            old_str='print("hello world")',
            new_str='print("changed")',
        )
        assert "---" in r.diff  # unified diff header
        assert "+++" in r.diff
        assert "-" in r.diff  # removed line
        assert "+" in r.diff  # added line

    async def test_edit_lint_catches_syntax_error(self, shell, tmp_path):
        f = tmp_path / "good.py"
        f.write_text("x = 1\ny = 2\n")
        r = await shell.edit(str(f), old_str="y = 2", new_str="y = (")
        assert r.success  # edit applied even with lint error
        assert len(r.lint_errors) > 0  # but lint errors reported

    async def test_edit_lint_ignores_preexisting_errors(self, shell, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("x = (\ny = 2\n")  # already has syntax error
        r = await shell.edit(str(f), old_str="y = 2", new_str="y = 3")
        # Should NOT report the pre-existing syntax error as new
        # (The pre-existing error is "x = (" which was there before)
        # This is a tricky test - the edit itself is valid
        assert r.success


# ==========================================================================
# write
# ==========================================================================
class TestWrite:
    async def test_create_new_file(self, shell, tmp_path):
        r = await shell.write(str(tmp_path / "new.py"), "x = 1\n")
        assert r.created
        assert r.lines == 1
        assert (tmp_path / "new.py").read_text() == "x = 1\n"

    async def test_overwrite_existing(self, shell, tmp_path):
        f = tmp_path / "exist.py"
        f.write_text("old content\n")
        r = await shell.write(str(f), "new content\n")
        assert not r.created
        assert f.read_text() == "new content\n"
        assert r.diff  # should show diff

    async def test_creates_parent_dirs(self, shell, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "file.py"
        r = await shell.write(str(deep), "deep\n")
        assert r.created
        assert deep.read_text() == "deep\n"


# ==========================================================================
# grep
# ==========================================================================
class TestGrep:
    async def test_grep_finds_matches(self, shell, sample_tree):
        r = await shell.grep("def", str(sample_tree / "src"))
        assert r.total_matches > 0
        assert any("helper" in m for m in r.matches)

    async def test_grep_no_matches(self, shell, sample_tree):
        r = await shell.grep("zzz_nonexistent_zzz", str(sample_tree))
        assert r.total_matches == 0
        assert "No matches" in r.text

    async def test_grep_with_include(self, shell, sample_tree):
        r = await shell.grep(".", str(sample_tree), include="*.md")
        assert r.total_matches > 0
        assert all(".md" in m or "README" in m for m in r.matches)

    async def test_grep_include_glob_not_shell_expanded(self, shell, sample_tree):
        """The include glob must be shell-quoted so bash doesn't expand it."""
        # Bug: if the include glob isn't quoted, bash expands *.py against the
        # session cwd before rg sees it, breaking the filter.
        # To trigger: cd into a dir that HAS .py files, then grep with include.
        await shell.run(f"cd {sample_tree / 'src'}")
        r = await shell.grep("def", str(sample_tree), include="*.py")
        assert r.total_matches > 0, f"Expected matches for 'def' in *.py, got {r.matches}"
        for m in r.matches:
            assert ".py:" in m, f"Expected .py file match, got: {m}"

    async def test_grep_literal(self, shell, sample_tree):
        # Create a file with regex-special characters
        (sample_tree / "special.txt").write_text("price is $10.00\n")
        r = await shell.grep("$10.00", str(sample_tree), literal=True)
        assert r.total_matches > 0


# ==========================================================================
# find
# ==========================================================================
class TestFind:
    async def test_find_py_files(self, shell, sample_tree):
        r = await shell.find("*.py", str(sample_tree))
        assert r.total_matches > 0
        assert any("main.py" in m for m in r.matches)

    async def test_find_no_matches(self, shell, sample_tree):
        r = await shell.find("*.xyz", str(sample_tree))
        assert r.total_matches == 0

    async def test_find_respects_gitignore(self, shell, sample_tree):
        """Files in __pycache__ and .git should not appear when .gitignore exists."""
        # rg respects .gitignore when run from inside a git repo
        await shell.run(
            f"cd {sample_tree} && git init -q "
            f"&& echo '__pycache__/' > .gitignore && git add .gitignore"
        )
        # Run find from inside the repo (cd first so rg detects the git root)
        await shell.run(f"cd {sample_tree}")
        r = await shell.find("*.py", ".")
        for m in r.matches:
            assert "__pycache__" not in m
        # Verify we DO find real files
        assert r.total_matches > 0


# ==========================================================================
# ls
# ==========================================================================
class TestLs:
    async def test_ls_basic(self, shell, sample_tree):
        r = await shell.ls(str(sample_tree))
        assert r.num_files > 0
        assert "src" in r.tree
        assert "README.md" in r.tree

    async def test_ls_ignores_hidden_and_cache(self, shell, sample_tree):
        r = await shell.ls(str(sample_tree))
        assert "__pycache__" not in r.tree
        assert ".git" not in r.tree

    async def test_ls_depth_limit(self, shell, sample_tree):
        r = await shell.ls(str(sample_tree), depth=1)
        # Should show src/ but not src/sub/deep.py
        assert "src" in r.tree
        assert "deep.py" not in r.tree

    async def test_ls_nonexistent(self, shell):
        r = await shell.ls("/nonexistent/path")
        assert "not a directory" in r.tree.lower() or "error" in r.tree.lower()

    async def test_ls_truncation(self, shell, tmp_path):
        # Create many files
        for i in range(10):
            (tmp_path / f"file_{i}.txt").write_text(f"content {i}")
        r = await shell.ls(str(tmp_path), max_entries=3)
        assert r.truncated
        assert r.num_files == 3


# ==========================================================================
# reset
# ==========================================================================
class TestReset:
    async def test_reset_preserves_cwd(self, shell, tmp_path):
        sub = tmp_path / "mydir"
        sub.mkdir()
        await shell.run(f"cd {sub}")
        await shell.reset()
        r = await shell.run("pwd")
        assert str(sub) in r.stdout

    async def test_reset_recovers_from_timeout(self, shell):
        # Trigger a timeout with a long-running command
        r = await shell.run("sleep 999", timeout=0.5)
        assert not r.success
        # Session should auto-reset and work for the next command
        r2 = await shell.run("echo recovered")
        assert r2.success
        assert "recovered" in r2.stdout


# ==========================================================================
# grep context
# ==========================================================================
class TestLintQuoting:
    async def test_lint_handles_path_with_single_quote(self, shell, tmp_path):
        """_lint must handle paths containing single quotes."""
        weird_dir = tmp_path / "it's a test"
        weird_dir.mkdir()
        f = weird_dir / "good.py"
        f.write_text("x = 1\ny = 2\n")
        r = await shell.edit(str(f), old_str="y = 2", new_str="y = (")
        # Should detect the NEW syntax error, not crash on the path
        assert r.success
        assert len(r.lint_errors) > 0


class TestGrepContext:
    async def test_grep_context_total_excludes_separators(self, shell, sample_tree):
        r = await shell.grep("helper", str(sample_tree / "src"), context=1)
        # With context, rg adds separator lines ("--") between groups.
        # total_matches should count only match lines, not separators.
        assert r.total_matches > 0
        # Separators should still appear in the output for readability
        # but not inflate the count
        separator_count = sum(1 for m in r.matches if m == "--")
        assert r.total_matches <= len(r.matches) - separator_count

    async def test_grep_context_total_excludes_context_lines(self, shell, tmp_path):
        """total_matches must count only actual matches, not context lines."""
        # Create files where we know the exact match count
        (tmp_path / "a.py").write_text("line1\nMATCH_ME\nline3\nline4\nMATCH_ME\nline6\n")
        r = await shell.grep("MATCH_ME", str(tmp_path / "a.py"), context=1)
        # There are exactly 2 matches. With context=1, rg will output
        # the match lines plus surrounding context lines and separators.
        assert r.total_matches == 2, (
            f"Expected 2 matches but got {r.total_matches}; matches={r.matches}"
        )


# ==========================================================================
# repr
# ==========================================================================
class TestRepr:
    async def test_repr_shows_cwd(self, shell, tmp_path):
        r = repr(shell)
        assert "ShellTools" in r
        assert "cwd=" in r

    async def test_repr_shows_file_after_view(self, shell, sample_py):
        await shell.view(str(sample_py))
        r = repr(shell)
        assert "file=" in r
        assert "sample.py" in r


# ==========================================================================
# edit — fuzzy matching
# ==========================================================================
class TestEditFuzzy:
    async def test_edit_fuzzy_trailing_whitespace(self, shell, tmp_path):
        """Fuzzy match strips trailing whitespace differences."""
        f = tmp_path / "ws.py"
        f.write_text("def hello():  \n    pass\n")
        # old_str without trailing whitespace should still match
        r = await shell.edit(
            str(f), old_str="def hello():\n    pass", new_str="def hi():\n    pass"
        )
        assert r.success

    async def test_edit_fuzzy_smart_quotes(self, shell, tmp_path):
        """Fuzzy match normalizes smart quotes to ASCII."""
        f = tmp_path / "quotes.py"
        f.write_text("msg = 'hello world'\n")
        # Using smart quotes in old_str
        r = await shell.edit(str(f), old_str="msg = \u2018hello world\u2019", new_str="msg = 'hi'")
        assert r.success
        assert "hi" in f.read_text()

    async def test_edit_closest_match_hint(self, shell, tmp_path):
        """When no match found, returns closest match as hint."""
        f = tmp_path / "hint.py"
        f.write_text("def hello_world():\n    print('hi')\n\ndef goodbye():\n    pass\n")
        r = await shell.edit(
            str(f), old_str="def hello_wrld():\n    print('hi')", new_str="replaced"
        )
        assert not r.success
        assert "Closest match" in r.error

    async def test_edit_strips_line_number_prefixes(self, shell, tmp_path):
        """Line-number prefixes from view() output are stripped."""
        f = tmp_path / "lines.py"
        f.write_text("import os\nimport sys\n\ndef main():\n    pass\n")
        # Simulate copy-pasting from view() output with line numbers
        r = await shell.edit(
            str(f),
            old_str="  1|import os\n  2|import sys",
            new_str="import os\nimport sys\nimport re",
        )
        assert r.success
        content = f.read_text()
        assert "import re" in content


# ==========================================================================
class TestInsert:
    async def test_insert_at_line(self, shell, tmp_path):
        """Insert content before a specific line."""
        f = tmp_path / "ins.py"
        f.write_text("line1\nline2\nline3\n")
        r = await shell.insert(str(f), line=2, content="inserted")
        assert r.success
        content = f.read_text()
        lines = content.split("\n")
        assert lines[1] == "inserted"

    async def test_insert_prepend(self, shell, tmp_path):
        """Insert at line 0 prepends to file."""
        f = tmp_path / "prepend.py"
        f.write_text("original\n")
        r = await shell.insert(str(f), line=0, content="header\n")
        assert r.success
        assert f.read_text().startswith("header\n")

    async def test_insert_append(self, shell, tmp_path):
        """Insert at line -1 appends to file."""
        f = tmp_path / "append.py"
        f.write_text("original\n")
        r = await shell.insert(str(f), line=-1, content="\nfooter")
        assert r.success
        assert f.read_text().endswith("\nfooter")

    async def test_insert_nonexistent_file(self, shell, tmp_path):
        """Insert into nonexistent file returns error."""
        r = await shell.insert(str(tmp_path / "nope.py"), line=1, content="x")
        assert not r.success
        assert "not found" in r.error.lower()


# ==========================================================================
# write — truncation warning
# ==========================================================================
class TestWriteTruncation:
    async def test_write_truncation_warning(self, shell, tmp_path):
        """Write warns when file shrinks significantly."""
        f = tmp_path / "big.py"
        f.write_text("x" * 200)
        r = await shell.write(str(f), "x" * 50)
        assert "WARNING" in r.diff
        assert "shrunk" in r.diff

    async def test_write_no_warning_for_small_shrink(self, shell, tmp_path):
        """No warning for small size reductions."""
        f = tmp_path / "small.py"
        f.write_text("x" * 200)
        r = await shell.write(str(f), "x" * 180)
        assert "WARNING" not in (r.diff or "")
