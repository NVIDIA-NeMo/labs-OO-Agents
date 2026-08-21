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
    EventExplorerRow,
    build_event_rows,
    detail_match_occurrences,
    highlighted_detail_lines,
    render_event_explorer,
    wrapped_detail_lines,
)
from nooa_cli.tui.explorer_base import FTS_ACTIVE_STYLE
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
    assert "## Tool" in markdown
    assert "execute_python" in markdown
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
    assert "## Status" in row.markdown
    assert "`complete`" in row.markdown
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


def test_event_explorer_navigation_and_rendering() -> None:
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="first")),
                ("2", _FakeEvent("TUIUserInput", text="second")),
                ("3", _FakeEvent("TUIUserInput", text="third")),
            ]
        )
    )
    model = EventExplorerModel(rows)

    model.move(-1)
    assert model.current.tag == "2"
    model.move(+1)
    assert model.current.tag == "3"
    model.jump_home()
    assert model.current.tag == "1"

    rendered = render_event_explorer(model, width=70, height=14)
    assert "Event Explorer" in rendered
    assert "1" in rendered
    assert "↑/↓ matches/scroll" in rendered


def test_event_explorer_search_mode_up_down_moves_between_match_occurrences() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha " * 80))])
    )[0]
    model = EventExplorerModel([row])
    model.edit_query("alpha")
    model.focus = "list"
    render_event_explorer(model, width=44, height=8)
    first_offset = model.detail_offset

    model.move_or_scroll(+1)
    render_event_explorer(model, width=44, height=8)

    assert model.current.tag == "1"
    assert model.search_line_cursor == 1
    assert model.detail_offset >= first_offset

    model.move_or_scroll(-1)
    render_event_explorer(model, width=44, height=8)
    assert model.current.tag == "1"
    assert model.search_line_cursor == 0


def test_event_explorer_fts_list_navigation_centers_current_match() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha " * 120))])
    )[0]
    model = EventExplorerModel([row])
    model.edit_query("alpha")
    model.focus = "list"
    render_event_explorer(model, width=44, height=8)

    for _ in range(18):
        model.move_or_scroll(+1)
        render_event_explorer(model, width=44, height=8)

    target = model.current_search_line()
    assert target is not None
    assert model._last_detail_visible_lines > 0
    middle = model.detail_offset + model._last_detail_visible_lines // 2
    assert abs(target - middle) <= 1


def test_event_explorer_fts_detail_focus_up_down_scrolls_text() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha " * 120))])
    )[0]
    model = EventExplorerModel([row])
    model.edit_query("alpha")
    model.focus = "detail"
    render_event_explorer(model, width=44, height=8)

    model.move_or_scroll(+1)
    rendered = render_event_explorer(model, width=90, height=8)

    assert model.search_active is True
    assert model.search_line_cursor == 0
    assert model.detail_offset == 1
    assert "↑/↓ scroll text" in rendered


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

    assert "\x1b[30;106malpha\x1b[0m" in joined
    assert "\x1b[30;43malpha\x1b[0m" in joined


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

    assert selected_line.count("\x1b[30;106malpha\x1b[0m") == 1
    assert selected_line.count("\x1b[30;43malpha\x1b[0m") >= 1
    assert selected_line.index("\x1b[30;43malpha\x1b[0m") < selected_line.index(
        "\x1b[30;106malpha\x1b[0m"
    )


def test_event_explorer_ansi_styles_header_footer_and_mode_label() -> None:
    model = EventExplorerModel(
        build_event_rows(
            SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha"))])
        )
    )
    model.edit_query("alpha")

    rendered = render_event_explorer(model, width=90, height=10, ansi=True)
    lines = rendered.splitlines()

    assert lines[0].startswith("\x1b[48;5;236;38;5;252m")
    assert lines[-1].startswith("\x1b[48;5;236;38;5;252m")
    active_mode = f"{FTS_ACTIVE_STYLE}FTS MODE\x1b[0m"
    assert active_mode in lines[-1]
    assert "\x1b[48;5;236;38;5;252m" in lines[-1].split(active_mode, 1)[1]

    model.search_active = False
    rendered = render_event_explorer(model, width=90, height=10, ansi=True)
    assert "\x1b[1;30;46mBROWSE MODE\x1b[0m" in rendered.splitlines()[-1]


def test_event_explorer_fts_divider_prompt_matches_session_explorer_style() -> None:
    model = EventExplorerModel(
        build_event_rows(
            SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="alpha"))])
        )
    )
    model.search_active = True
    model.set_query("alpha")

    rendered = render_event_explorer(model, width=90, height=12, ansi=True)

    assert f"{FTS_ACTIVE_STYLE}[FTS: alpha] \x1b[0m" in rendered


def test_event_explorer_fts_mode_survives_tab_focus_changes() -> None:
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="alpha one")),
                ("2", _FakeEvent("TUIUserInput", text="alpha two")),
            ]
        )
    )
    model = EventExplorerModel(rows)
    model.edit_query("alpha")
    model.toggle_focus()
    assert model.search_active is True
    assert model.focus == "detail"

    rendered = render_event_explorer(model, width=90, height=12)
    assert "FTS MODE" in rendered
    assert "pane=event text" in rendered
    assert "↑/↓ scroll text" in rendered

    model.move_or_scroll(+1)
    assert model.search_active is True
    assert model.focus == "detail"
    assert model.current.tag == "1"

    model.toggle_focus()
    assert model.search_active is True
    assert model.focus == "list"
    rendered = render_event_explorer(model, width=90, height=12)
    assert "↑/↓ next match" in rendered
    model.move_or_scroll(+1)
    assert model.current.tag == "2"


def test_event_explorer_footer_does_not_advertise_noop_enter_in_browse_mode() -> None:
    model = EventExplorerModel([EventExplorerRow("1", "T", "summary", "summary", "detail")])

    browse = render_event_explorer(model, width=120, height=10)
    assert "enter browse" not in browse
    assert "enter exit FTS" not in browse

    model.search_active = True
    fts = render_event_explorer(model, width=120, height=10)
    assert "enter exit FTS" in fts


def test_event_explorer_advertises_q_close() -> None:
    model = EventExplorerModel(
        build_event_rows(
            SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="hello"))])
        )
    )

    rendered = render_event_explorer(model, width=120, height=10)
    source = Path("packages/nooa-cli/src/nooa_cli/tui/event_explorer.py").read_text()

    assert "q quit" not in rendered
    assert "esc clear" in rendered
    assert "q close" in rendered
    assert '@kb.add("q")' not in source


def test_event_explorer_browse_mode_label_is_explicit() -> None:
    model = EventExplorerModel(
        build_event_rows(
            SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="hello"))])
        )
    )

    rendered = render_event_explorer(model, width=90, height=10)

    assert "BROWSE MODE" in rendered
    assert "pane=events" in rendered


def test_event_explorer_search_mode_moves_to_next_event_after_last_occurrence() -> None:
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="alpha one")),
                ("2", _FakeEvent("TUIUserInput", text="alpha two")),
            ]
        )
    )
    model = EventExplorerModel(rows)
    model.edit_query("alpha")
    render_event_explorer(model, width=60, height=10)

    model.move_or_scroll(+1)
    render_event_explorer(model, width=60, height=10)

    assert model.current.tag == "2"
    assert model.search_line_cursor == 0


def test_event_explorer_search_no_matches() -> None:
    model = EventExplorerModel(
        build_event_rows(
            SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="hello"))])
        )
    )
    model.set_query("missing")
    rendered = render_event_explorer(model, width=60, height=10)
    assert "No matches" in rendered


def test_event_explorer_search_shows_match_position_and_highlights_matches() -> None:
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="alpha one")),
                ("2", _FakeEvent("TUIUserInput", text="beta two")),
                ("3", _FakeEvent("TUIUserInput", text="alpha three")),
            ]
        )
    )
    model = EventExplorerModel(rows)
    model.set_query("alpha")

    rendered = render_event_explorer(model, width=80, height=14, ansi=True)
    assert "match 1/2" in rendered
    assert "\x1b[30;43malpha\x1b[0m" in rendered

    model.move(+1)
    rendered = render_event_explorer(model, width=80, height=14, ansi=True)
    assert "match 2/2" in rendered
    assert model.current.tag == "3"


def test_event_explorer_search_highlights_matches_inside_detail_text() -> None:
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="find alpha here"))])
    )[0]
    joined = "\n".join(highlighted_detail_lines(row, width=80, query="alpha"))

    assert "\x1b[30;43malpha\x1b[0m" in joined


def test_event_explorer_wraps_long_event_details_and_scrolls_within_event() -> None:
    long_text = "alpha " * 40
    row = build_event_rows(
        SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text=long_text))])
    )[0]
    lines = wrapped_detail_lines(row, width=32)

    assert len(lines) > len(row.detail.splitlines())
    assert all(len(line) <= 32 for line in lines)

    model = EventExplorerModel([row])
    render_event_explorer(model, width=44, height=10)
    before = model.detail_offset
    model.page_detail(+4)
    assert model.detail_offset > before
    rendered = render_event_explorer(model, width=44, height=10)
    assert "event lines" in rendered


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
    assert "Tool" in stripped
    assert "execute_python" in stripped
    assert "Python" in stripped
    assert "print" in stripped
    assert "event:" not in stripped
    assert "ToolCallEvent(" not in stripped
    assert "\x1b[" in joined


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


def test_event_explorer_tab_focus_changes_up_down_semantics() -> None:
    long_text = "alpha " * 80
    rows = build_event_rows(
        SimpleNamespace(
            items=lambda: [
                ("1", _FakeEvent("TUIUserInput", text="first")),
                ("2", _FakeEvent("TUIUserInput", text=long_text)),
            ]
        )
    )
    model = EventExplorerModel(rows)
    assert model.focus == "list"
    assert model.current.tag == "2"

    model.toggle_focus()
    assert model.focus == "detail"
    render_event_explorer(model, width=44, height=10)
    model.scroll_detail(+3)
    assert model.current.tag == "2"
    assert model.detail_offset > 0
    rendered = render_event_explorer(model, width=44, height=10)
    assert "BROWSE MODE" in rendered
    assert "pane=event text" in rendered
    assert "❯ event lines" in rendered

    model.toggle_focus()
    model.move(-1)
    assert model.current.tag == "1"


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


def test_event_explorer_handles_tiny_resize_heights_without_overflowing_body() -> None:
    model = EventExplorerModel(
        build_event_rows(
            SimpleNamespace(items=lambda: [("1", _FakeEvent("TUIUserInput", text="hello"))])
        )
    )

    rendered = render_event_explorer(model, width=40, height=3)

    assert len(rendered.splitlines()) == 3
    assert "Event Explorer" in rendered


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
        await h.press("q")
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

        await h.type_keys("/")
        await h.type_keys("alpha")
        await h.press("down")

        assert h.capture_input() == ""
        assert h.app._event_explorer_model.query == "alpha"
        assert h.app._event_explorer_model.search_active is True
        await h.press("escape")
        await h.wait_for(lambda: h.app._event_explorer_model.search_active is False)
        await h.press("escape")
        await h.wait_for(lambda: h.app._event_explorer_model.query == "")
        await h.press("q")
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

        await h.type_keys("/")
        await h.type_keys("jqkr")

        await h.wait_for(lambda: h.app._event_explorer_model.query == "jqkr")
        assert h.capture_input() == ""
        assert h.app.active_subview is not None

        await h.press("escape")
        await h.wait_for(lambda: h.app._event_explorer_model.search_active is False)
        await h.press("q")
        await asyncio.wait_for(task, timeout=1)


def test_event_explorer_has_in_app_mouse_scroll_bindings() -> None:
    source = Path("packages/nooa-cli/src/nooa_cli/tui/tui_application.py").read_text()

    assert "open_event_explorer" in source
    assert "Keys.ScrollDown" in source
    assert "Keys.ScrollUp" in source
    assert "mouse_support=Condition(_subview_mouse_enabled)" in source
    assert 'getattr(view, "mouse_support", True)' in source
    assert "_SuspendedPromptToolkitResize" not in source


# ============================================================================
# Session explorer tests
# ============================================================================


def _session_row(id: str, name: str, turns: list[tuple[str, str]]):
    from nooa_cli.tui.session_explorer import SessionExplorerRow
    from nooa_cli.tui.session_manager import Turn

    turn_objs = [
        Turn(role=role, content=content, ts=1000.0 + i) for i, (role, content) in enumerate(turns)
    ]
    search_text = "\n".join([id, name, "test-model", *[t.content for t in turn_objs]])
    return SessionExplorerRow(
        id=id,
        name=name,
        model="test/model",
        agent="TUIAgent",
        working_dir="/tmp/project",
        started_at=1000.0,
        last_active=2000.0,
        turn_count=len(turn_objs),
        turns=turn_objs,
        search_text=search_text,
    )


def test_session_explorer_model_searches_across_sessions_and_dialog() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel

    rows = [
        _session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("user", "hello world")]),
        _session_row(
            "bbbb0000-0000-0000-0000-000000000002", "beta", [("agent", "contains frobnicator")]
        ),
    ]
    model = SessionExplorerModel(rows)

    model.set_query("frobnicator")

    assert model.matches == [1]
    assert model.current is rows[1]


def test_session_explorer_navigation_tab_and_rendering() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    rows = [
        _session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("user", "first")]),
        _session_row("bbbb0000-0000-0000-0000-000000000002", "beta", [("agent", "second")]),
    ]
    model = SessionExplorerModel(rows)

    model.move(+1)
    model.toggle_focus()
    rendered = render_session_explorer(model, width=90, height=20)

    assert model.current is rows[1]
    assert model.focus == "dialog"
    assert "Session Explorer" in rendered
    assert "session dialog" in rendered
    assert "beta" in rendered
    assert "OO:" in rendered
    assert "second" in rendered


def test_session_explorer_opens_detail_at_end_of_long_session() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    turns = [("user", f"early line {i}") for i in range(20)] + [("agent", "final answer")]
    model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "long", turns)]
    )
    model.toggle_focus()

    rendered = render_session_explorer(model, width=90, height=12)

    assert "final answer" in rendered
    assert "early line 0" not in rendered


def test_session_explorer_visible_tail_renders_markdown() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    model = SessionExplorerModel(
        [
            _session_row(
                "aaaa0000-0000-0000-0000-000000000001",
                "markdown",
                [("agent", "**bold final**\n\n- one\n- two")],
            )
        ]
    )
    model.focus = "dialog"

    rendered = render_session_explorer(model, width=80, height=16, ansi=True)

    assert "\x1b[1mbold final\x1b[0m" in rendered
    assert "\x1b[1m • \x1b[0mone" in rendered


def test_session_explorer_highlight_does_not_bleed_into_blank_lines() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    row = _session_row(
        "aaaa0000-0000-0000-0000-000000000001",
        "hi",
        [("agent", "a line ending with hi")],
    )
    model = SessionExplorerModel([row])
    model.set_query("hi")

    rendered = render_session_explorer(model, width=60, height=18, ansi=True)
    lines = rendered.splitlines()
    highlighted = [line for line in lines if "\x1b[30;43m" in line]

    assert highlighted
    assert all(line.strip(" \x1b[0m") for line in highlighted)
    assert not any(line.endswith("\x1b[30;43m") for line in lines)


def test_session_explorer_has_separate_session_and_dialog_fts_modes() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    rows = [
        _session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("agent", "shared needle")]),
        _session_row("bbbb0000-0000-0000-0000-000000000002", "beta", [("agent", "other text")]),
    ]
    model = SessionExplorerModel(rows)

    model.set_query("alpha", scope="sessions")
    assert model.matches == [0]

    model.focus = "dialog"
    model.search_active = True
    model.search_scope = "dialog"
    model.edit_query("needle")
    rendered = render_session_explorer(model, width=90, height=16, ansi=True)

    assert model.matches == [0]
    assert model.session_query == "alpha"
    assert model.detail_query == "needle"
    assert "[FTS dialog: needle]" in rendered
    assert "\x1b[30;106mneedle\x1b[0m" in rendered


def test_session_explorer_tab_switches_fts_scope_with_pane() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel, SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("agent", "needle")])]
    )

    assert view.handle_key("slash") == "handled"
    assert view.model.search_scope == "sessions"
    assert view.handle_key("tab") == "handled"
    assert view.model.focus == "dialog"
    assert view.model.search_scope == "dialog"


def test_session_explorer_dialog_fts_up_down_moves_between_matches() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    row = _session_row(
        "aaaa0000-0000-0000-0000-000000000001",
        "alpha",
        [("agent", "needle one\n" + "filler\n" * 8 + "needle two")],
    )
    model = SessionExplorerModel([row])
    model.focus = "dialog"
    model.search_active = True
    model.set_query("needle", scope="dialog")

    render_session_explorer(model, width=80, height=10)
    first_offset = model.detail_offset
    model.move_or_scroll(+1)
    render_session_explorer(model, width=80, height=10)

    assert model.detail_search_cursor == 1
    assert model.detail_offset > first_offset


def test_session_explorer_tab_to_dialog_fts_then_down_moves_to_next_match() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        SessionExplorerView,
        render_session_explorer,
    )

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [
            _session_row(
                "aaaa0000-0000-0000-0000-000000000001",
                "alpha",
                [("agent", "needle one\n" + "filler\n" * 10 + "needle two")],
            )
        ]
    )
    view.pending_input = None

    render_session_explorer(view.model, width=80, height=12)
    assert view.handle_key("slash") == "handled"
    for ch in "needle":
        assert view.handle_key("text", ch) == "handled"
    assert view.handle_key("tab") == "handled"

    # The user can press Down immediately after Tab, before the redraw that
    # discovers dialog match line numbers.
    assert view.handle_key("down") == "handled"
    rendered = render_session_explorer(view.model, width=80, height=12)

    assert view.model.search_scope == "dialog"
    assert view.model.focus == "dialog"
    assert view.model.detail_search_cursor == 1
    assert "needle two" in rendered
    assert "needle one" not in rendered


def test_session_explorer_list_fts_navigation_scrolls_detail_to_match() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    first = _session_row(
        "aaaa0000-0000-0000-0000-000000000001",
        "alpha one",
        [("agent", "alpha near top")],
    )
    second = _session_row(
        "bbbb0000-0000-0000-0000-000000000002",
        "alpha two",
        [("agent", "filler\n" * 12 + "alpha near bottom")],
    )
    model = SessionExplorerModel([first, second])
    model.search_active = True
    model.set_query("alpha", scope="sessions")

    rendered = render_session_explorer(model, width=80, height=10)
    assert "alpha near top" in rendered

    model.move_or_scroll(+1)
    rendered = render_session_explorer(model, width=80, height=10)

    assert model.cursor == 1
    assert "alpha near bottom" in rendered


def test_session_explorer_fts_divider_prompt_is_highlighted_when_active() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("agent", "needle")])]
    )
    model.search_active = True
    model.set_query("alpha", scope="sessions")

    rendered = render_session_explorer(model, width=90, height=12, ansi=True)

    assert f"{FTS_ACTIVE_STYLE}[FTS sessions: alpha] \x1b[0m" in rendered


def test_session_explorer_selected_dialog_match_uses_distinct_highlight() -> None:
    from nooa_cli.tui.session_explorer import (
        SessionExplorerModel,
        render_session_explorer,
    )

    row = _session_row(
        "aaaa0000-0000-0000-0000-000000000001",
        "alpha",
        [("agent", "needle one\n" + "filler\n" * 8 + "needle two")],
    )
    model = SessionExplorerModel([row])
    model.focus = "dialog"
    model.search_active = True
    model.set_query("needle", scope="dialog")
    render_session_explorer(model, width=80, height=10, ansi=True)
    model.move_or_scroll(+1)

    rendered = render_session_explorer(model, width=80, height=10, ansi=True)

    assert "\x1b[30;106mneedle\x1b[0m two" in rendered
    assert "\x1b[30;43mneedle\x1b[0m one" not in rendered


def test_session_explorer_mouse_scroll_actions_target_dialog() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel, SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [
            _session_row(
                "aaaa0000-0000-0000-0000-000000000001",
                "long",
                [("user", f"line {i}") for i in range(20)],
            )
        ]
    )

    view.handle_key("scroll_up")

    assert view.model.focus == "dialog"


def test_session_explorer_view_fts_accepts_navigation_and_quit_chars() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    from nooa_cli.tui.session_explorer import SessionExplorerModel

    view.model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "jqk", [("user", "jqk query")])]
    )

    assert view.handle_key("slash") == "handled"
    assert view.handle_key("j") == "handled"
    assert view.handle_key("quit") == "handled"
    assert view.handle_key("k") == "handled"

    assert view.model.query == "jqk"
    assert view.handle_key("escape") == "handled"
    assert view.model.search_active is False
    assert view.handle_key("quit") == "close"


def test_session_explorer_resume_key_closes_with_resume_prefill() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel, SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("user", "hello")])]
    )
    view.pending_input = None

    assert view.handle_key("resume") == "close"
    assert view.pending_input == "/session resume aaaa0000"


def test_session_explorer_resume_key_types_r_in_fts_mode() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel, SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("user", "hello")])]
    )
    view.pending_input = None
    view.model.search_active = True

    assert view.handle_key("resume") == "handled"
    assert view.model.query == "r"
    assert view.pending_input is None


def test_session_explorer_slash_key_types_slash_in_fts_mode() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel, SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [_session_row("aaaa0000-0000-0000-0000-000000000001", "alpha", [("user", "path /tmp")])]
    )
    view.pending_input = None

    assert view.handle_key("slash") == "handled"
    assert view.handle_key("text", "t") == "handled"
    assert view.handle_key("slash") == "handled"
    assert view.handle_key("text", "m") == "handled"

    assert view.model.search_active is True
    assert view.model.query == "t/m"


@pytest.mark.asyncio
async def test_session_list_opens_in_app_explorer_when_available() -> None:
    from nooa_cli.tui.commands import SessionCommand

    frontend = MagicMock()
    frontend.open_session_explorer = AsyncMock()
    cmd = SessionCommand(frontend, MagicMock(), MagicMock())

    result = await cmd.execute(["list"])

    assert result.success is True
    assert "closed" in result.outputs[0].content
    frontend.open_session_explorer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tui_app_opens_and_closes_session_explorer_in_app(monkeypatch) -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerRow
    from nooa_cli.tui.session_manager import Turn

    from .tui_app_harness import TUIHarness

    rows = [
        SessionExplorerRow(
            id="aaaa0000-0000-0000-0000-000000000001",
            name="alpha session",
            model="test/model",
            agent="TUIAgent",
            working_dir="/tmp/project",
            started_at=1000.0,
            last_active=2000.0,
            turn_count=1,
            turns=[Turn(role="user", content="find alpha", ts=1000.0)],
            search_text="alpha session find alpha",
        )
    ]
    monkeypatch.setattr(
        "nooa_cli.tui.session_explorer.build_session_rows",
        lambda *, limit=100: rows,
    )

    async with TUIHarness() as h:
        task = asyncio.create_task(h.app.open_session_explorer())
        await h.wait_for(lambda: h.app.active_subview is not None)

        view = h.app.active_subview
        assert view is not None
        model = view.model
        assert model.current.id.startswith("aaaa0000")
        await h.type_keys("/")
        await h.type_keys("alpha")
        await h.wait_for(lambda: model.query == "alpha")
        await h.press("tab")
        await h.wait_for(lambda: model.focus == "dialog")

        assert h.capture_input() == ""
        await h.press("escape")
        await h.press("q")
        await asyncio.wait_for(task, timeout=1)
        assert h.app.active_subview is None


@pytest.mark.asyncio
async def test_tui_app_session_explorer_resume_prefills_input(monkeypatch) -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerRow
    from nooa_cli.tui.session_manager import Turn

    from .tui_app_harness import TUIHarness

    rows = [
        SessionExplorerRow(
            id="aaaa0000-0000-0000-0000-000000000001",
            name="alpha session",
            model="test/model",
            agent="TUIAgent",
            working_dir="/tmp/project",
            started_at=1000.0,
            last_active=2000.0,
            turn_count=1,
            turns=[Turn(role="user", content="find alpha", ts=1000.0)],
            search_text="alpha session find alpha",
        )
    ]
    monkeypatch.setattr(
        "nooa_cli.tui.session_explorer.build_session_rows",
        lambda *, limit=100: rows,
    )

    async with TUIHarness() as h:
        task = asyncio.create_task(h.app.open_session_explorer())
        await h.wait_for(lambda: h.app.active_subview is not None)

        await h.type_keys("r")
        await asyncio.wait_for(task, timeout=1)

        assert h.app.active_subview is None
        assert h.capture_input() == "/session resume aaaa0000"


def test_event_explorer_copy_preserves_semantic_markdown() -> None:
    from nooa_cli.tui.event_explorer import EventExplorerView

    markdown = "## User input\n\n```python\nprint('exact')\n```"
    event = _FakeEvent("TUIUserInput", text="print exact")
    view = EventExplorerView(SimpleNamespace(items=lambda: [("1", event)]))
    view.model.rows[0].markdown = markdown
    copied: list[str] = []
    view.set_copy_handler(lambda text: copied.append(text) or True)

    assert view.handle_key("copy") == "handled"
    assert copied == [markdown]


def test_session_explorer_copy_preserves_unwrapped_dialog() -> None:
    from nooa_cli.tui.session_explorer import SessionExplorerModel, SessionExplorerView

    view = SessionExplorerView.__new__(SessionExplorerView)
    view.model = SessionExplorerModel(
        [
            _session_row(
                "aaaa0000-0000-0000-0000-000000000001",
                "copy session",
                [("user", "a very long source line"), ("agent", "exact answer")],
            )
        ]
    )
    view.native_selection = False
    view._copy_handler = None
    view._copy_status = ""

    copied = view.copy_text()

    assert copied is not None
    assert "You:\n  a very long source line" in copied
    assert "OO:\n  exact answer" in copied
