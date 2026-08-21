# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core behavior of the DEFAULT ShellTools (run / read / write_file / replace).

This keeps the *default* ShellTools — the one agents actually use — covered for
its primary file/run surface. Search-anchor behavior is covered separately in
test_shell_tools_modern.py.
"""

import pytest

from nooa.tools.shell_tools import Match, ShellResult, ShellTools


@pytest.fixture
def sh(tmp_path):
    return ShellTools(cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_run_persists_state(sh, tmp_path):
    r = await sh.run("echo hello")
    assert r.success
    assert "hello" in r.stdout
    # cd persists across calls in the same session.
    (tmp_path / "sub").mkdir()
    await sh.run("cd sub")
    r2 = await sh.run("pwd")
    assert r2.stdout.strip().endswith("sub")


@pytest.mark.asyncio
async def test_run_reports_failure(sh):
    r = await sh.run("false")
    assert not r.success
    assert r.returncode != 0
    assert r.timed_out is False


def test_match_requires_resolved_path():
    with pytest.raises(TypeError, match="resolved_path"):
        Match("example.py", 1, 1, "value\n")  # type: ignore[call-arg]


def test_shell_result_timeout_flag_preserves_positional_matches_argument():
    match = Match("example.py", 1, 1, "value\n", resolved_path="/tmp/example.py")
    result = ShellResult("value", "", 0, [match], timed_out=True)

    assert result.matches == [match]
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_write_file_then_read(sh, tmp_path):
    await sh.write_file("f.txt", "line1\nline2\nline3\n")
    assert (tmp_path / "f.txt").read_text() == "line1\nline2\nline3\n"
    # read with a numbered gutter (default) -> Match; inspect via .numbered/.text.
    view = await sh.read("f.txt")
    assert "line2" in view.numbered
    # read a line window -> Match for just that line.
    window = await sh.read("f.txt", (2, 2))
    assert "line2" in window.text
    assert "line1" not in window.text


@pytest.mark.asyncio
async def test_replace_path_unique(sh, tmp_path):
    await sh.write_file("f.py", "x = 1\ny = 2\nz = 3\n")
    await sh.replace("f.py", "y = 2", "y = 22")
    assert (tmp_path / "f.py").read_text() == "x = 1\ny = 22\nz = 3\n"


@pytest.mark.asyncio
async def test_replace_path_ambiguous_errors(sh, tmp_path):
    await sh.write_file("f.py", "a = 1\na = 1\n")
    with pytest.raises(ValueError, match="matched 2 times"):
        # Two matches -> must error rather than guess.
        await sh.replace("f.py", "a = 1", "a = 2")


@pytest.mark.asyncio
async def test_write_file_is_overwrite(sh, tmp_path):
    await sh.write_file("f.txt", "old")
    await sh.write_file("f.txt", "new")
    assert (tmp_path / "f.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_file_operations_allow_paths_outside_cwd(sh, tmp_path):
    sibling = tmp_path.parent / f"{tmp_path.name}-sibling"
    sibling.mkdir()
    relative = f"../{sibling.name}/relative.txt"
    absolute = sibling / "absolute.txt"

    await sh.write_file(relative, "one\ntwo\n")
    assert (await sh.read(relative)).text == "one\ntwo\n"

    await sh.replace(relative, "one", "changed")
    await sh.replace(
        Match(relative, 2, 2, "two\n", resolved_path=sibling / "relative.txt"), "replaced"
    )
    assert (sibling / "relative.txt").read_text() == "changed\nreplaced"

    await sh.write_file(str(absolute), "absolute")
    assert (await sh.read(str(absolute))).text == "absolute"


@pytest.mark.asyncio
async def test_match_from_read_stays_bound_after_cwd_change(sh, tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("before\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "original.txt").write_text("wrong file\n")

    match = await sh.read("original.txt")
    sliced = match[1:1]
    await sh.run("cd other")
    await sh.replace(sliced, "after")

    assert original.read_text() == "after"
    assert (other / "original.txt").read_text() == "wrong file\n"


@pytest.mark.asyncio
async def test_close_terminates_underlying_bash_session(sh):
    """Verify close() terminates BashSession and the shell lazily restarts."""
    r = await sh.run("echo started")
    assert r.success
    assert sh._session._process is not None

    await sh.close()

    assert sh._session._process is None
    assert not sh._session._started

    # The shell remains reusable after close(); a fresh session starts lazily.
    r2 = await sh.run("echo restarted")
    assert r2.success
    assert "restarted" in r2.stdout
    await sh.close()


def test_match_rejects_a_relative_resolved_path(tmp_path, monkeypatch):
    """The anchor must be absolute, or it silently binds to the process cwd.

    Path.resolve() on a relative path resolves against os.getcwd(), which is
    not the shell cwd — so a caller passing a relative path would produce a
    Match pointing at a different file, with no error. Every caller passes an
    absolute path today; this keeps that a rule rather than a convention.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        Match("f.txt", 1, 1, "hello\n", resolved_path="f.txt")


def test_match_keeps_an_absolute_resolved_path(tmp_path):
    """The supported form is unaffected."""
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    match = Match("f.txt", 1, 1, "hello\n", resolved_path=target)
    assert match.resolved_path == str(target.resolve())
