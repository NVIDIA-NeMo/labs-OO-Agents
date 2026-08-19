# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic activity emitted by the shared interactive coding tools."""

import asyncio

import nooa_cli.coding.activity as activity
import pytest
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
    assert edits[0].diff_complete is True
    assert (edits[1].start_line, edits[1].end_line) == (1, 1)
    assert "-one" in edits[1].diff
    assert "+two" in edits[1].diff
    assert edits[1].diff_complete is True


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
    # difflib omits the count for a single-line hunk; the offset is what matters.
    assert "@@ -2 +2 @@" in edit.diff


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
    assert edit.diff_complete is False
    assert "previous file content could not be read" in edit.diff
    assert (tmp_path / "binary.dat").read_text() == "now text"


async def test_large_text_file_keeps_a_real_line_oriented_diff(tmp_path):
    shell, events = _observed_shell(tmp_path)
    old_lines = [f"section {index}: {'old content ' * 8}\n" for index in range(250)]
    new_lines = list(old_lines)
    new_lines[125] = "section 125: corrected agenda text\n"
    path = tmp_path / "agenda.md"
    path.write_text("".join(old_lines))
    try:
        await shell.write_file(str(path), "".join(new_lines))
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    assert edit.content_complete is False
    assert edit.diff_complete is True
    assert edit.diff.startswith("--- a/agenda.md\n+++ b/agenda.md\n")
    assert "-section 125: old content" in edit.diff
    assert "+section 125: corrected agenda text" in edit.diff
    assert "str(len=" not in edit.diff
    assert "a//" not in edit.diff


async def test_diff_generation_has_a_separate_input_safety_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(activity, "_MAX_DIFF_INPUT_CHARS", 100)
    shell, events = _observed_shell(tmp_path)
    try:
        await shell.write_file("large.txt", "x" * 101)
    finally:
        await shell.close()

    edit = next(event for event in events if isinstance(event, FileEdit))
    assert edit.content_complete is True
    assert edit.diff_complete is False
    assert edit.diff.startswith("--- a/large.txt\n+++ b/large.txt\n")
    assert "exceeds the safe diff preview limit" in edit.diff


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
    assert edit.diff.startswith("<truncated-output>")
    assert "--- a/large.txt" in edit.diff
    assert "str(len=" not in edit.diff
    assert edit.content_complete is False
    assert edit.diff_complete is False
    assert (starts[0].stdin or "").startswith("str(len=50000,")
    assert starts[0].stdin_truncated is True
    assert sum(len(event.stdout) + len(event.stderr) for event in stream_outputs) <= 31_000
    assert stream_outputs[0].stdout.startswith("<truncated-output>")
    assert stream_finished.output_truncated is True
    assert streamed[-1].kind == "done"


async def test_cancelling_a_command_is_not_reported_as_an_error(tmp_path):
    """Cancellation is a user action, not a command failure.

    The handler stringified the exception, and CancelledError has an empty
    str(), so the event carried the literal text "CancelledError" and hosts
    rendered a deliberate stop as a crash.
    """
    shell, events = _observed_shell(tmp_path)
    try:
        running = asyncio.create_task(shell.run("sleep 30"))
        started = None
        while started is None:
            await asyncio.sleep(0.05)
            started = next(
                (event for event in events if isinstance(event, TerminalCommandStarted)),
                None,
            )
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        await shell.close()

    finished = next(event for event in events if isinstance(event, TerminalCommandFinished))
    assert finished.cancelled is True
    assert "CancelledError" not in finished.error


def test_every_hunk_is_offset_not_just_the_first():
    """Multi-hunk diffs must stay in one coordinate system.

    Only the first header was rewritten, and its counts were replaced with the
    size of the whole region — so later hunks kept region-relative numbers and
    could point *before* the first hunk.
    """
    old = "".join(f"line{i}\n" for i in range(1, 42))
    new = old.replace("line2\n", "CHANGED2\n").replace("line20\n", "CHANGED20\n")

    diff, complete = activity._edit_diff("f.py", old, new, start_line=10)

    def hunk_starts(text: str) -> list[int]:
        return [
            int(line.split()[1].lstrip("-").split(",")[0])
            for line in text.splitlines()
            if line.startswith("@@ ")
        ]

    raw, _ = activity._edit_diff("f.py", old, new, start_line=None)
    assert len(hunk_starts(raw)) >= 2, raw
    # Every hunk shifts by the same amount; none keeps region-relative numbers.
    assert hunk_starts(diff) == [start + 9 for start in hunk_starts(raw)]
    assert complete is True


def test_a_missing_final_newline_is_marked():
    """Unterminated content needs the marker, or the diff is not applicable."""
    diff, _ = activity._edit_diff("f.py", "a", "b", start_line=None)

    assert "-a" in diff and "+b" in diff
    assert "\\ No newline at end of file" in diff, diff
