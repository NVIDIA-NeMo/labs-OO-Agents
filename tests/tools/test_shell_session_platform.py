# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public ShellTools interface selects the native shell on each platform."""

import sys
from contextlib import aclosing

import pytest

from nooa.tools import BashSession, PowerShellSession, ShellTools, create_shell_session


def test_native_default_and_explicit_selection(tmp_path):
    expected = PowerShellSession if sys.platform == "win32" else BashSession
    assert isinstance(ShellTools(cwd=str(tmp_path)).session, expected)
    assert isinstance(create_shell_session(shell="bash"), BashSession)
    assert isinstance(create_shell_session(shell="powershell"), PowerShellSession)
    with pytest.raises(ValueError, match="Unknown shell"):
        create_shell_session(shell="unknown")


async def test_tools_share_state_for_commands_and_file_edits(tmp_path):
    directory = tmp_path / "中文 space's [dir]"
    directory.mkdir()
    shell = ShellTools(cwd=str(tmp_path))
    quote = "'" + str(directory).replace("'", "''") + "'"
    command = (
        f"Set-Location -LiteralPath {quote}"
        if sys.platform == "win32"
        else 'cd -- "' + str(directory) + '"'
    )
    try:
        assert (await shell.run(command)).success
        assert shell.cwd == directory
        await shell.write_file("value.txt", "old\n")
        match = await shell.read("value.txt")
        await shell.replace(match, "new\n")
        assert (directory / "value.txt").read_text() == "new\n"
        command = "[Console]::WriteLine('done')" if sys.platform == "win32" else "echo done"
        async with aclosing(shell.run_stream(command, cwd=tmp_path)) as stream:
            events = [event async for event in stream]
        assert events[-1].returncode == 0
        assert shell.cwd == directory
        assert "done" in "".join(e.text for e in events if e.kind == "stdout")
    finally:
        await shell.close()


async def test_utf8_file_tools_preserve_text_in_both_replace_forms(tmp_path):
    shell = ShellTools(cwd=str(tmp_path))
    filename = "中文 café.txt"
    path = tmp_path / filename
    original = "# café\nmessage = '初始 🐍'\n"
    try:
        await shell.write_file(filename, original)
        assert path.read_bytes().decode("utf-8").replace("\r\n", "\n") == original
        match = await shell.read(filename, lines=(2, 2))
        assert match.text == "message = '初始 🐍'\n"
        await shell.replace(match, "message = '修改 ✅'\n")
        await shell.replace(filename, "修改 ✅", "完成 café")
        assert (await shell.read(filename)).text == "# café\nmessage = '完成 café'\n"
        assert path.read_bytes().decode("utf-8").replace("\r\n", "\n") == (
            "# café\nmessage = '完成 café'\n"
        )
    finally:
        await shell.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows command language")
async def test_powershell_result_and_language_are_visible(tmp_path):
    from nooa.agentdoc import doc

    shell = ShellTools(cwd=str(tmp_path))
    try:
        result = await shell.run("Write-Error 'problem'")
        assert result.shell == "powershell" and not result.success
        assert "PowerShellOutput" in repr(result) and "problem" in result.stderr
        assert "powershell" in doc(shell).lower()
        # PowerShell variables and quoting must never go through the Bash
        # search parser, even if the output resembles an editable anchor.
        result = await shell.run("Write-Output 'file:1:example'")
        assert result.matches is None
    finally:
        await shell.close()
