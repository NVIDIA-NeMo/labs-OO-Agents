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


async def test_write_and_replace_emit_bounded_structured_file_edits(tmp_path):
    shell, events = _observed_shell(tmp_path)
    try:
        await shell.write_file("example.txt", "one\n")
        await shell.replace("example.txt", "one", "two")
    finally:
        await shell.close()

    edits = [event for event in events if isinstance(event, FileEdit)]
    assert [(event.operation, event.old_text, event.new_text) for event in edits] == [
        ("create", None, "one\n"),
        ("update", "one", "two"),
    ]
    assert edits[0].path == str(tmp_path / "example.txt")
    assert edits[0].diff.startswith("--- a/example.txt\n+++ b/example.txt")
    assert (edits[1].start_line, edits[1].end_line) == (1, 1)
    assert "-one" in edits[1].diff
    assert "+two" in edits[1].diff


async def test_match_replace_emits_actual_before_and_after_text(tmp_path):
    shell, events = _observed_shell(tmp_path)
    (tmp_path / "example.txt").write_text("one\ntwo\nthree\n")
    try:
        match = await shell.read("example.txt", lines=(2, 2))
        await shell.replace(match, "changed")
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    assert edit.old_text == "two\n"
    # replace() re-terminates the region so the following line survives; the
    # event has to report the text that actually landed in the file.
    assert edit.new_text == "changed\n"
    assert (tmp_path / "example.txt").read_text() == "one\nchanged\nthree\n"
    assert (edit.start_line, edit.end_line) == (2, 2)
    assert "@@ -2,1 +2,1 @@" in edit.diff


async def test_match_replace_at_end_of_file_reports_the_unterminated_text(tmp_path):
    shell, events = _observed_shell(tmp_path)
    (tmp_path / "example.txt").write_text("one\ntwo\nthree\n")
    try:
        match = await shell.read("example.txt", lines=(3, 3))
        await shell.replace(match, "changed")
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    # Nothing follows the region, so no newline is added and none is reported.
    assert edit.new_text == "changed"
    assert (tmp_path / "example.txt").read_text() == "one\ntwo\nchanged"


async def test_match_replace_after_cwd_change_emits_original_path(tmp_path):
    shell, events = _observed_shell(tmp_path)
    original = tmp_path / "example.txt"
    original.write_text("before\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "example.txt").write_text("wrong file\n")
    try:
        match = await shell.read("example.txt")
        await shell.run("cd other")
        await shell.replace(match, "after")
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    assert edit.path == str(original)
    assert original.read_text() == "after"
    assert (other / "example.txt").read_text() == "wrong file\n"


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
    assert edit.content_complete is False
    assert (tmp_path / "binary.dat").read_text() == "now text"


async def test_run_emits_correlated_start_and_finish(tmp_path):
    shell, events = _observed_shell(tmp_path)
    try:
        result = await shell.run("cat", stdin="hello")
    finally:
        await shell.close()

    started = next(event for event in events if isinstance(event, TerminalCommandStarted))
    finished = next(event for event in events if isinstance(event, TerminalCommandFinished))
    output = next(event for event in events if isinstance(event, TerminalCommandOutput))
    assert started.command == "cat"
    assert started.stdin == "hello"
    assert started.working_directory == str(tmp_path)
    assert finished.command_id == started.command_id
    assert finished.exit_code == 0
    assert output.stdout == "hello"
    assert output.stderr == ""
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
    assert output.stdout == "hello\n"
    assert output.stderr == ""
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


async def test_activity_payloads_are_bounded(tmp_path):
    shell, events = _observed_shell(tmp_path)
    large = "x" * 50_000
    try:
        await shell.write_file("large.txt", large)
        await shell.run("cat", stdin=large)
        streamed = [item async for item in shell.run_stream("python3 -c \"print('y' * 50000)\"")]
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    starts = [event for event in events if isinstance(event, TerminalCommandStarted)]
    stream_outputs = [
        event
        for event in events
        if isinstance(event, TerminalCommandOutput) and event.command_id == starts[1].command_id
    ]
    stream_finished = next(
        event
        for event in events
        if isinstance(event, TerminalCommandFinished) and event.command_id == starts[1].command_id
    )

    assert edit.new_text.startswith("str(len=50000,")
    assert edit.diff.startswith("str(len=")
    assert edit.content_complete is False
    assert (starts[0].stdin or "").startswith("str(len=50000,")
    assert starts[0].stdin_truncated is True
    assert sum(len(event.stdout) + len(event.stderr) for event in stream_outputs) <= 31_000
    assert stream_outputs[0].stdout.startswith("<truncated-output>")
    assert stream_finished.output_truncated is True
    assert streamed[-1].kind == "done"
