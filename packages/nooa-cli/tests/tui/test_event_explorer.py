# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the terminal event explorer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui.commands import EventsCommand
from nooa_cli.tui.event_explorer import (
    EventExplorerModel,
    build_event_rows,
    detail_match_occurrences,
    highlighted_detail_lines,
    wrapped_detail_lines,
)
from nooa_cli.tui.explorer_base import (
    highlight_style_code,
)
from nooa_cli.tui.output import TextOutput


class _FakeEvent:
    def __init__(self, event_type: str, **fields):
        self.event_type = event_type
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self):
        return {"event_type": self.event_type, **self.__dict__}


def test_event_explorer_builds_rows_and_full_text_searches() -> None:
    events = [
        ("1", _FakeEvent("TUIUserInput", text="please inspect frobnicator")),
        (
            "2",
            _FakeEvent(
                "ToolCallEvent",
                name="execute_python",
                arguments={"code": "# comment\nprint('needle')"},
            ),
        ),
        ("3", _FakeEvent("PythonOutput", execution_status="complete", stdout="needle output")),
    ]
    em = SimpleNamespace(items=lambda: events)

    rows = build_event_rows(em)
    model = EventExplorerModel(rows)

    assert [row.tag for row in rows] == ["1", "2", "3"]
    assert "execute_python" in rows[1].summary

    model.set_query("needle output")
    assert [rows[i].tag for i in model.matches] == ["3"]

    model.set_query("needle")
    assert [rows[i].tag for i in model.matches] == ["2", "3"]


def test_event_explorer_list_focus_moves_rows_while_searching() -> None:
    """List focus moves between matching rows even while search is active.

    The old occurrence-jumping machinery (move_search_occurrence and friends)
    was removed with the unified navigation contract; this pins the live
    behavior it was replaced by.
    """
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="alpha alpha")),
                ("2", _FakeEvent("TUIUserInput", text="alpha alpha")),
                ("3", _FakeEvent("TUIUserInput", text="alpha alpha")),
            ]
        )
    )
    model = EventExplorerModel(rows)
    model.set_query("alpha")
    model.search_active = True

    model.move_or_scroll(1)
    assert model.cursor == 1
    model.move_or_scroll(1)
    assert model.cursor == 2
    model.move_or_scroll(-1)
    assert model.cursor == 1


def test_event_explorer_query_change_clears_cached_occurrences() -> None:
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="alpha beta")),
                ("2", _FakeEvent("TUIUserInput", text="alpha beta")),
            ]
        )
    )
    model = EventExplorerModel(rows)

    model.edit_query("beta")

    assert model._last_detail_match_lines == []
    assert model._last_detail_match_occurrences == []
    assert model.search_line_cursor == 0


def test_event_explorer_renders_shared_session_events() -> None:
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("SessionUserMessage", content="inspect the failure")),
                (
                    "2",
                    _FakeEvent(
                        "SessionStarted",
                        host="tui",
                        model="test-model",
                        agent="CodingAgent",
                        working_directory="/work/repo",
                    ),
                ),
            ]
        )
    )

    assert rows[0].summary == "inspect the failure"
    assert "## User input" in (rows[0].markdown or "")
    assert "inspect the failure" in (rows[0].markdown or "")
    assert "test-model" in (rows[1].markdown or "")


def test_event_explorer_collapses_multiline_user_message_summary() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "2",
                    _FakeEvent(
                        "SessionUserMessage",
                        content="first line\n\nexport UV_PYTHON=value",
                    ),
                )
            ]
        )
    )[0]

    assert row.summary == "first line export UV_PYTHON=value"
    assert "\n" not in row.summary


def test_event_explorer_renders_generic_events_as_markdown_sections() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "12",
                    _FakeEvent(
                        "ToolCallEvent",
                        name="execute_python",
                        arguments={"code": "print(42)"},
                        result=None,
                    ),
                )
            ]
        )
    )[0]

    markdown = row.markdown or ""
    assert "**[12]** *ToolCallEvent*" in markdown
    assert "_metadata:" not in markdown
    assert "**Tool:** `execute_python`" in markdown
    assert "## Tool" not in markdown
    assert "## Python" in markdown
    assert "```python\nprint(42)\n```" in markdown
    assert "## Result" not in markdown
    assert "## arguments" not in markdown

    plain = "\n".join(wrapped_detail_lines(row, width=80))
    assert "**[12]** *ToolCallEvent*" in plain
    assert "## Python" in plain
    assert "ToolCallEvent(" not in plain


def test_event_explorer_renders_python_output_as_markdown_sections() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "7",
                    _FakeEvent(
                        "PythonOutput",
                        tool_call_id="tc_1",
                        execution_status="complete",
                        stdout="hello\nworld\n",
                        stderr="",
                        error="",
                        value=None,
                    ),
                )
            ]
        )
    )[0]

    assert row.markdown is not None
    assert "**[7]** *PythonOutput* · tool=tc_1" in row.markdown
    assert "_metadata:" in row.markdown
    assert "tool_call_id=tc_1" in row.markdown
    assert "**Status:** `complete`" in row.markdown
    assert "## Status" not in row.markdown
    assert "## Stdout" in row.markdown
    assert "```text\nhello\nworld\n```" in row.markdown
    assert "## Stderr" not in row.markdown
    assert "## Error" not in row.markdown
    assert "## Value" not in row.markdown

    plain = "\n".join(wrapped_detail_lines(row, width=80))
    assert "**[7]** *PythonOutput* · tool=tc_1" in plain
    assert "## Stdout" in plain
    assert "PythonOutput(" not in plain


def test_event_explorer_python_output_markdown_keeps_stderr_and_error_when_present() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "8",
                    _FakeEvent(
                        "PythonOutput",
                        execution_status="error",
                        stdout="partial",
                        stderr="warning",
                        error="Traceback: boom",
                        value={"answer": 42},
                    ),
                )
            ]
        )
    )[0]

    markdown = row.markdown or ""
    assert "## Stdout" in markdown
    assert "```text\npartial\n```" in markdown
    assert "## Stderr" in markdown
    assert "```text\nwarning\n```" in markdown
    assert "## Error" in markdown
    assert "```pytb\nTraceback: boom\n```" in markdown
    assert "## Value" in markdown
    assert '"answer": 42' in markdown


def test_event_explorer_python_output_markdown_code_blocks_render_formatted() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "9",
                    _FakeEvent(
                        "PythonOutput",
                        execution_status="complete",
                        stdout="for i in range(3):\n    print(i)\n",
                    ),
                )
            ]
        )
    )[0]

    rendered = "\n".join(highlighted_detail_lines(row, width=80))
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", rendered)

    assert "```" not in stripped
    assert "for i in range(3):" in stripped
    assert "print(i)" in stripped
    assert "\x1b[" in rendered


def test_event_explorer_uses_compact_header_and_metadata_footer() -> None:
    """Compact header shows tag/type/date/short-ids; noise fields go to a metadata footer."""
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "42",
                    _FakeEvent(
                        "Task",
                        id="abc",
                        timestamp="2026-01-02T03:04:05Z",
                        metadata={"call_id": "c1", "model": "m"},
                        images=[{"url": "file://large.png"}],
                        prompt="Do the work",
                    ),
                )
            ]
        )
    )[0]

    markdown = row.markdown or ""
    header = markdown.splitlines()[0]
    footer = markdown.rsplit("\n", 1)[-1]

    assert header.startswith("**[42]** *Task* · 2026-01-02 03:04:05 · id=abc")
    assert "call=c1" in header
    assert "## Prompt" in markdown
    assert "Do the work" in markdown
    assert "---" in markdown
    assert footer.startswith("_metadata: id=abc · timestamp=")
    assert "metadata=" in footer
    assert "images=" in footer
    assert "## metadata" not in markdown.lower()
    assert "## images" not in markdown.lower()


def test_event_explorer_python_output_markdown_escapes_terminal_controls() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "11",
                    _FakeEvent(
                        "PythonOutput",
                        execution_status="complete",
                        stdout="safe\x1b]52;c;YWJj\x07after",
                    ),
                )
            ]
        )
    )[0]

    markdown = row.markdown or ""
    rendered = "\n".join(highlighted_detail_lines(row, width=100))
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", rendered)

    assert "\x1b]" not in markdown
    assert "\x07" not in markdown
    assert "\\x1b]52;c;YWJj\\x07" in markdown
    assert "\x1b]" not in rendered
    assert "\x07" not in rendered
    assert "\\x1b]52;c;YWJj\\x07" in stripped


def test_event_explorer_existing_markdown_fenced_code_still_renders_formatted() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "10",
                    _FakeEvent(
                        "TUIAgentMessage",
                        content="Before\n```python\nfor i in range(3):\n    print(i)\n```\nAfter",
                    ),
                )
            ]
        )
    )[0]

    rendered = "\n".join(highlighted_detail_lines(row, width=80))
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", rendered)

    assert "```" not in stripped
    assert "for i in range(3):" in stripped
    assert "print(i)" in stripped
    assert "\x1b[" in rendered


def test_event_explorer_renders_llm_output_as_syntax_highlighted_code() -> None:
    """LLMOutput events render their content as syntax-highlighted code blocks."""
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [("6", _FakeEvent("LLMOutput", content="def f():\n    return 1"))]
        )
    )[0]

    markdown = row.markdown or ""
    rendered = "\n".join(highlighted_detail_lines(row, width=70))
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", rendered)

    assert "## LLM output" in markdown
    assert "```python" in markdown
    assert "def f():" in stripped
    assert "\x1b[" in rendered


def test_event_explorer_renders_summary_as_readable_markdown() -> None:
    """Summary events show summary text inline and compact range/children metadata."""
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "1..5",
                    _FakeEvent(
                        "Summary",
                        summary_tag="1..5",
                        children_tags=["1", "2", "3", "4", "5"],
                        summary_text="User asked for a renderer and the agent implemented it.",
                    ),
                )
            ]
        )
    )[0]

    markdown = row.markdown or ""
    plain = "\n".join(wrapped_detail_lines(row, width=90))

    assert "**[1..5]** *Summary*" in markdown
    assert "children_tags=" in markdown.rsplit("\n", 1)[-1]
    assert "User asked for a renderer" in plain
    assert "Summary Tag" in plain
    assert "Summary(" not in plain


def test_event_explorer_fts_current_match_uses_distinct_highlight() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha beta alpha"))])
    )[0]
    line_no, occurrence_no = detail_match_occurrences(row, width=80, query="alpha")[0]
    lines = highlighted_detail_lines(
        row,
        width=80,
        query="alpha",
        current_match_line=line_no,
        current_match_occurrence=occurrence_no,
    )
    joined = "\n".join(lines)

    assert f"{highlight_style_code(current=True)}alpha\x1b[0m" in joined
    assert f"{highlight_style_code()}alpha\x1b[0m" in joined


def test_event_explorer_current_match_highlights_second_match_on_same_line() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha beta alpha gamma"))]
        )
    )[0]
    occurrences = detail_match_occurrences(row, width=120, query="alpha")
    same_line = [(line, occurrence) for line, occurrence in occurrences if occurrence == 1]
    assert same_line
    line_no, occurrence_no = same_line[0]

    lines = highlighted_detail_lines(
        row,
        width=120,
        query="alpha",
        current_match_line=line_no,
        current_match_occurrence=occurrence_no,
    )
    selected_line = lines[line_no]

    assert selected_line.count(f"{highlight_style_code(current=True)}alpha\x1b[0m") == 1
    assert selected_line.count(f"{highlight_style_code()}alpha\x1b[0m") >= 1
    assert selected_line.index(f"{highlight_style_code()}alpha\x1b[0m") < selected_line.index(
        f"{highlight_style_code(current=True)}alpha\x1b[0m"
    )


def test_event_explorer_view_highlights_selected_occurrence_on_same_line() -> None:
    from nooa_cli.tui.event_explorer import EventExplorerView

    manager = SimpleNamespace(
        items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha beta alpha gamma"))]
    )
    view = EventExplorerView(manager)
    view.model.set_query("alpha")
    view.model.search_line_cursor = 1

    lines = view.detail_lines(view.model.current, width=120)

    assert "".join(lines).count(f"{highlight_style_code(current=True)}alpha\x1b[0m") == 1


def test_event_search_text_excludes_markdown_chrome() -> None:
    """Timestamps and ids in the markdown header/footer must not match."""
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="hello"))])
    )[0]

    # The markdown header carries a timestamp; searching it must not match.
    assert "timestamp" not in row.search_text.lower()
    # The rendered markdown content itself remains searchable.
    assert "hello" in row.search_text


def test_event_explorer_search_ignores_empty_field_names() -> None:
    """Searching must not match the *names* of empty fields.

    A PythonOutput event always has an ``error`` key, but usually an empty
    one. Searching "error" matched every PythonOutput event via the raw repr
    in search_text — even when no error occurred.
    """
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "1",
                    _FakeEvent(
                        "PythonOutput",
                        stdout="all good",
                        stderr="",
                        error="",
                        execution_status="complete",
                    ),
                ),
                (
                    "2",
                    _FakeEvent("PythonOutput", error="NameError: boom"),
                ),
            ]
        )
    )

    model = EventExplorerModel(rows)
    model.set_query("error")

    assert [row.tag for row in (rows[i] for i in model.matches)] == ["2"]


def test_event_explorer_search_highlights_matches_inside_detail_text() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="find alpha here"))])
    )[0]
    joined = "\n".join(highlighted_detail_lines(row, width=80, query="alpha"))

    assert f"{highlight_style_code()}alpha\x1b[0m" in joined


def test_event_explorer_search_keeps_styled_detail_colors() -> None:
    """Searching must not strip the detail pane's syntax colors.

    Regression for the searched path rendering wrapped *plain* text, which
    drained all color from the detail view while a query was active.
    """
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [("1", _FakeEvent("ToolCallEvent", name="run", arguments={"cmd": "ls"}))]
        )
    )[0]

    searched = highlighted_detail_lines(row, width=80, query="run")

    with_query = [line for line in "\n".join(searched).splitlines() if "\x1b[" in line]
    # Non-match styling (syntax colors) must survive the search: strip the
    # search-highlight prefix itself and require another SGR to remain.
    assert with_query, "search dropped all styling"
    non_match_styles = "".join(with_query).replace(highlight_style_code(), "")
    assert "\x1b[38;" in non_match_styles or "\x1b[48;" in non_match_styles


def test_event_explorer_renders_execute_python_event_as_formatted_markdown() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "2",
                    _FakeEvent(
                        "ToolCallEvent",
                        name="execute_python",
                        arguments={"code": "for i in range(3):\n    print(i)"},
                    ),
                )
            ]
        )
    )[0]

    lines = highlighted_detail_lines(row, width=50)
    joined = "\n".join(lines)
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", joined)
    assert "ToolCallEvent" in stripped
    assert "[2]" in stripped
    assert "Tool: execute_python" in stripped
    assert "Tool\n" not in stripped
    assert "Python" in stripped
    assert "print" in stripped
    assert "event:" not in stripped
    assert "ToolCallEvent(" not in stripped
    assert "\x1b[" in joined


def test_event_explorer_renders_python_cell_event_as_formatted_markdown() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "2",
                    _FakeEvent(
                        "ToolCallEvent",
                        name="python_cell",
                        arguments={"code": "print('experimental')"},
                        result={
                            "content": "status: complete",
                            "result_status": "complete",
                            "tool_call_id": "call_2",
                        },
                    ),
                )
            ]
        )
    )[0]

    assert row.summary == "python_cell — print('experimental')"
    assert row.code == "print('experimental')"
    markdown = row.markdown or ""
    assert "## Python" in markdown
    assert "```python\nprint('experimental')\n```" in markdown
    assert "## Arguments" not in markdown


def test_event_explorer_renders_fenced_code_fields_as_formatted_markdown() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "4",
                    _FakeEvent(
                        "AgentMessage",
                        content="before\n```python\nvalue = 42\nprint(value)\n```\nafter",
                    ),
                )
            ]
        )
    )[0]

    assert row.code == "value = 42\nprint(value)"
    lines = highlighted_detail_lines(row, width=50)
    joined = "\n".join(lines)
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", joined)
    assert "AgentMessage" in stripped
    assert "before" in stripped
    assert "```" not in stripped
    assert "value = 42" in stripped
    assert "print(value)" in stripped
    assert "event:" not in stripped
    assert "\x1b[" in joined


def test_event_explorer_highlights_markdown_event_detail() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="hello"))])
    )[0]
    joined = "\n".join(highlighted_detail_lines(row, width=70))
    plain = __import__("re").sub(r"\x1b\[[0-9;]*m", "", joined)

    assert "event:" not in joined
    assert "\x1b[1;38;5;230;48;5;238m" in joined
    assert "\x1b[" in joined
    assert "[1] TUIUserInput" in plain
    assert "TUIUserInput" in plain
    assert "User input" in plain
    assert "text=" not in plain
    assert "TUIUserInput(" not in plain
    assert "hello" in plain


def test_event_explorer_renders_tui_agent_message_as_event_markdown() -> None:
    row = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                (
                    "5",
                    _FakeEvent(
                        "TUIAgentMessage",
                        content="# Answer\n\nThis is **markdown**, not repr.\n\n```python\nprint(1)\n```",
                    ),
                )
            ]
        )
    )[0]

    assert row.markdown is not None
    assert row.code is not None

    plain = "\n".join(wrapped_detail_lines(row, width=50))
    assert "**[5]** *TUIAgentMessage*" in plain
    assert "_metadata:" not in plain
    assert "# Answer" in plain
    assert "## content" not in plain
    assert "TUIAgentMessage(" not in plain
    assert "event:" not in plain

    highlighted = "\n".join(highlighted_detail_lines(row, width=50))
    stripped = __import__("re").sub(r"\x1b\[[0-9;]*m", "", highlighted)
    assert "TUIAgentMessage" in stripped
    assert "[5]" in stripped
    assert "Answer" in stripped
    assert "content=" not in stripped
    assert "TUIAgentMessage(" not in stripped
    assert "event:" not in stripped


@pytest.mark.asyncio
async def test_events_command_opens_in_app_explorer() -> None:
    agent = MagicMock()
    agent.event_manager = MagicMock()
    frontend = MagicMock()
    frontend.open_event_explorer = AsyncMock()
    config = MagicMock()

    cmd = EventsCommand(frontend, config, agent)
    result = await cmd.execute([])

    assert result.success is True
    assert isinstance(result.outputs[0], TextOutput)
    assert "closed" in result.outputs[0].content
    frontend.open_event_explorer.assert_awaited_once_with(agent.event_manager)


def test_events_command_rejects_args() -> None:
    cmd = EventsCommand(MagicMock(), MagicMock(), MagicMock())

    ok, error = cmd.validate_args(["1"])

    assert ok is False
    assert error == "Usage: /events"


@pytest.mark.asyncio
async def test_events_command_reports_failures() -> None:
    agent = MagicMock()
    agent.event_manager = MagicMock()
    frontend = MagicMock()
    frontend.open_event_explorer = AsyncMock(side_effect=RuntimeError("boom"))
    config = MagicMock()

    cmd = EventsCommand(frontend, config, agent)
    result = await cmd.execute([])

    assert result.success is False
    assert "boom" in result.outputs[0].content


@pytest.mark.asyncio
async def test_tui_app_opens_and_closes_event_explorer_in_app() -> None:
    from .tui_app_harness import FakeAgent, TUIHarness

    agent = FakeAgent()
    agent.event_manager = SimpleNamespace(
        items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha event"))]
    )
    async with TUIHarness(agent=agent) as h:
        task = asyncio.create_task(h.app.open_event_explorer(agent.event_manager))
        await h.wait_for(lambda: h.app._event_explorer_model is not None)

        assert h.app._event_explorer_model.current.tag == "1"
        await h.press("escape")
        await asyncio.wait_for(task, timeout=1)
        assert h.app._event_explorer_model is None


@pytest.mark.asyncio
async def test_tui_app_event_explorer_keys_do_not_edit_prompt() -> None:
    from .tui_app_harness import FakeAgent, TUIHarness

    agent = FakeAgent()
    agent.event_manager = SimpleNamespace(
        items=lambda: [
            ("1", _FakeEvent("TUIUserInput", text="alpha one")),
            ("2", _FakeEvent("TUIUserInput", text="alpha two")),
        ]
    )
    async with TUIHarness(agent=agent) as h:
        task = asyncio.create_task(h.app.open_event_explorer(agent.event_manager))
        await h.wait_for(lambda: h.app._event_explorer_model is not None)

        await h.type_keys("alpha")
        await h.press("down")

        assert h.capture_input() == ""
        assert h.app._event_explorer_model.query == "alpha"
        await h.press("escape")
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_tui_app_event_explorer_fts_can_search_printable_navigation_and_quit_keys() -> None:
    from .tui_app_harness import FakeAgent, TUIHarness

    agent = FakeAgent()
    agent.event_manager = SimpleNamespace(
        items=lambda: [("1", _FakeEvent("TUIUserInput", text="jqkr event"))]
    )
    async with TUIHarness(agent=agent) as h:
        task = asyncio.create_task(h.app.open_event_explorer(agent.event_manager))
        await h.wait_for(lambda: h.app._event_explorer_model is not None)

        await h.type_keys("jqkr")

        await h.wait_for(lambda: h.app._event_explorer_model.query == "jqkr")
        assert h.capture_input() == ""
        assert h.app.active_subview is not None

        await h.press("escape")
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_tui_app_event_explorer_routes_real_mouse_wheel_to_list() -> None:
    from .tui_app_harness import FakeAgent, TUIHarness

    agent = FakeAgent()
    agent.event_manager = SimpleNamespace(
        items=lambda: [
            (
                str(index),
                _FakeEvent(
                    "TUIUserInput",
                    text=(
                        "\n".join(f"detail line {line}" for line in range(100))
                        if index == 29
                        else f"event {index}"
                    ),
                ),
            )
            for index in range(30)
        ]
    )
    async with TUIHarness(agent=agent) as h:
        task = asyncio.create_task(h.app.open_event_explorer(agent.event_manager))
        await h.wait_for(lambda: h.app._event_explorer_model is not None)
        browser = h.app.active_subview
        assert browser is not None
        await h.wait_for(lambda: browser.list_control.viewport[1] > 1)
        await h.wait_for(lambda: browser.list_offset > 0)
        await h.wait_for(lambda: browser.model._last_detail_line_count > 20)
        initial_offset = browser.list_offset
        initial_detail_offset = browser.model.detail_offset

        # Xterm SGR wheel-up at column 10, row 8 (inside the list pane).
        await h.type_keys("\x1b[<64;10;8M")
        await h.wait_for(lambda: browser.list_offset < initial_offset)

        assert browser.model.detail_offset == initial_detail_offset
        assert browser.model.cursor == 29
        await h.press("escape")
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_tui_app_event_explorer_routes_real_mouse_wheel_to_detail() -> None:
    from .tui_app_harness import FakeAgent, TUIHarness

    agent = FakeAgent()
    agent.event_manager = SimpleNamespace(
        items=lambda: [
            (
                "1",
                _FakeEvent(
                    "TUIUserInput",
                    text="\n".join(f"detail line {index}" for index in range(100)),
                ),
            )
        ]
    )
    async with TUIHarness(agent=agent) as h:
        task = asyncio.create_task(h.app.open_event_explorer(agent.event_manager))
        await h.wait_for(lambda: h.app._event_explorer_model is not None)
        browser = h.app.active_subview
        assert browser is not None
        await h.wait_for(lambda: browser.preview_control.viewport[1] > 1)
        await h.wait_for(lambda: browser.model._last_detail_line_count > 20)

        # Xterm SGR wheel-down at column 10, row 30 (inside the detail pane).
        await h.type_keys("\x1b[<65;10;30M")
        await h.wait_for(lambda: browser.model.detail_offset > 0)

        assert browser.list_offset == 0
        await h.press("escape")
        await asyncio.wait_for(task, timeout=1)


def test_event_explorer_has_in_app_mouse_scroll_bindings() -> None:
    # Resolve from this test file so the check works from any cwd
    # (repo root or packages/nooa-cli).
    source = (Path(__file__).parents[2] / "src/nooa_cli/tui/tui_application.py").read_text()

    assert "open_event_explorer" in source
    assert "Keys.ScrollDown" in source
    assert "Keys.ScrollUp" in source
    assert "mouse_support=Condition(_subview_mouse_enabled)" in source
    assert 'getattr(view, "mouse_support", True)' in source
    assert "_SuspendedPromptToolkitResize" not in source


# ============================================================================
# Session explorer tests
# ============================================================================


def test_event_type_option_is_multi_select_checkbox_dropdown() -> None:
    from nooa_cli.tui.event_explorer import EventExplorerView
    from nooa_cli.tui.explorer_base import ExplorerChecklistOption

    view = EventExplorerView(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="one")),
                ("2", _FakeEvent("PythonOutput", execution_status="complete", stdout="two")),
                ("3", _FakeEvent("Task", prompt="three")),
            ]
        )
    )

    option = view.options[0]
    assert isinstance(option, ExplorerChecklistOption)
    assert option.label == "Event types"
    all_types = {"PythonOutput", "TUIUserInput", "Task"}
    assert option.checked == all_types
    assert option.display_value == "All"
    assert option.is_checked("__all__")

    option.activate()
    assert option.checked == set()
    assert view.model.enabled_types == set()
    assert not option.is_checked("__all__")

    option.activate()
    assert option.checked == all_types
    assert view.model.enabled_types == all_types


@pytest.mark.asyncio
async def test_tui_app_routes_grouped_options_without_editing_prompt() -> None:
    from nooa_cli.tui.event_explorer import EventExplorerView

    from .tui_app_harness import TUIHarness

    view = EventExplorerView(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="one")),
                ("2", _FakeEvent("PythonOutput", execution_status="complete", stdout="two")),
                ("3", _FakeEvent("Task", prompt="three")),
            ]
        )
    )
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_subview(view))
        await harness.wait_for(lambda: getattr(harness.app.active_subview, "view", None) is view)
        browser = harness.app.active_subview
        await harness.press("c-o")
        await harness.wait_for(lambda: browser.option_cursor == 0)
        assert len(browser.dropdown_floats) == 1
        event_type_menu = browser.dropdown_floats[0]
        assert event_type_menu.z_index == 10
        assert event_type_menu.attach_to_window is browser.option_windows[0]
        assert event_type_menu.xcursor is True
        assert event_type_menu.ycursor is True
        assert event_type_menu.top is None
        assert event_type_menu.right is None
        option_fragments = browser.option_controls[0]._text()
        assert option_fragments[0] == ("[SetMenuPosition]", "")
        dropdown_text = "".join(text for _style, text in browser.dropdown_controls[0]._text())
        assert "☑ All" in dropdown_text
        assert "☑ PythonOutput" in dropdown_text
        assert "☑ TUIUserInput" in dropdown_text
        await harness.type_keys(" ")
        await harness.wait_for(lambda: not view.model.enabled_types)
        assert browser.option_cursor == 0
        assert not view.model.matches
        await harness.type_keys(" ")
        await harness.wait_for(lambda: len(view.model.enabled_types) == 3)
        assert len(view.model.matches) == 3
        await harness.press("down")
        await harness.type_keys(" ")
        assert len(view.model.enabled_types) == 2
        assert len(view.model.matches) == 2
        assert harness.capture_input() == ""
        await harness.press("escape")
        await harness.wait_for(lambda: browser.option_cursor is None)
        await harness.press("c-c")
        await asyncio.wait_for(opened, timeout=1)
