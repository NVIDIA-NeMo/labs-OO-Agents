# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic activity emitted by the shared interactive coding tools."""

from nooa_cli.coding.activity import (
    ActivityShellTools,
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)

from nooa.runtime.event_manager import EventManager
from nooa.tools import ShellTools


def _observed_shell(tmp_path):
    events = []
    manager = EventManager()
    manager.on("*", events.append)
    shell = ShellTools(cwd=str(tmp_path))
    return ActivityShellTools(shell=shell, event_manager=manager), events


def test_activity_is_an_explicit_shell_tools_substitute(tmp_path):
    manager = EventManager()
    base_shell = ShellTools(cwd=str(tmp_path))
    shell = ActivityShellTools(shell=base_shell, event_manager=manager)

    assert not isinstance(shell, ShellTools)
    assert shell.session is base_shell.session
    assert shell.cwd == base_shell.cwd
    assert "event_manager" not in ShellTools.__init__.__annotations__


async def test_write_and_replace_emit_complete_file_edits(tmp_path):
    shell, events = _observed_shell(tmp_path)
    try:
        await shell.write_file("example.txt", "one\n")
        await shell.replace("example.txt", "one", "two")
    finally:
        await shell.close()

    edits = [event for event in events if isinstance(event, FileEdit)]
    assert [(event.operation, event.old_text, event.new_text) for event in edits] == [
        ("create", None, "one\n"),
        ("update", "one\n", "two\n"),
    ]
    assert edits[0].path == str(tmp_path / "example.txt")


async def test_match_replace_emits_actual_before_and_after_text(tmp_path):
    shell, events = _observed_shell(tmp_path)
    (tmp_path / "example.txt").write_text("one\ntwo\nthree\n")
    try:
        match = await shell.read("example.txt", lines=(2, 2))
        await shell.replace(match, "changed")
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    assert edit.old_text == "one\ntwo\nthree\n"
    assert edit.new_text == "one\nchanged\nthree\n"


async def test_observing_an_overwrite_does_not_break_binary_file_replacement(tmp_path):
    shell, events = _observed_shell(tmp_path)
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe")
    try:
        await shell.write_file("binary.dat", "now text")
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    assert edit.operation == "update"
    assert edit.old_text is None
    assert edit.new_text == "now text"
    assert (tmp_path / "binary.dat").read_text() == "now text"


async def test_run_emits_correlated_start_and_finish(tmp_path):
    shell, events = _observed_shell(tmp_path)
    try:
        result = await shell.run("printf hello")
    finally:
        await shell.close()

    started = next(event for event in events if isinstance(event, TerminalCommandStarted))
    finished = next(event for event in events if isinstance(event, TerminalCommandFinished))
    assert started.command == "printf hello"
    assert started.working_directory == str(tmp_path)
    assert finished.command_id == started.command_id
    assert finished.exit_code == 0
    assert finished.stdout == "hello"
    assert result.stdout == "hello"


async def test_run_stream_emits_output_chunks_and_finish(tmp_path):
    shell, events = _observed_shell(tmp_path)
    try:
        streamed = [event async for event in shell.run_stream("printf 'hello\\n'")]
    finally:
        await shell.close()

    started = next(event for event in events if isinstance(event, TerminalCommandStarted))
    output = next(event for event in events if isinstance(event, TerminalCommandOutput))
    finished = next(event for event in events if isinstance(event, TerminalCommandFinished))
    assert output.command_id == started.command_id
    assert output.stream == "stdout"
    assert output.content == "hello\n"
    assert finished.command_id == started.command_id
    assert finished.exit_code == 0
    assert streamed[-1].kind == "done"


async def test_closing_stream_after_done_does_not_emit_a_second_finish(tmp_path):
    shell, events = _observed_shell(tmp_path)
    stream = shell.run_stream("printf 'hello\\n'")
    try:
        async for item in stream:
            if item.kind == "done":
                break
    finally:
        await stream.aclose()
        await shell.close()

    finished = [event for event in events if isinstance(event, TerminalCommandFinished)]
    assert len(finished) == 1
    assert finished[0].error == ""
