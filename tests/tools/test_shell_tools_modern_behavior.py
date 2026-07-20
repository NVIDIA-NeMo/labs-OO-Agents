# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core behavior of the DEFAULT ShellTools (run / read / write_file / replace).

The broad legacy behavior suite (test_shell_tools.py) targets ShellToolsLegacy.
This keeps the *default* ShellTools — the one agents actually use — covered for
its primary file/run surface, addressing the review concern that promoting v5
to ShellTools left the default implementation under-tested.
"""

import pytest

from nooa.tools.shell_tools import ShellTools


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
