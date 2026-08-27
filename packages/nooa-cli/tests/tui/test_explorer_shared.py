# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for job and todo explorer views."""

from __future__ import annotations

from unittest.mock import MagicMock

from nooa_cli.interactive import AgentJobState, AgentJobSummary
from nooa_cli.interactive.runtime import JobSnapshot
from nooa_cli.tui.event_explorer import (
    highlight_terms_with_current,
)
from nooa_cli.tui.explorer_base import (
    ExplorerChecklistOption,
    ExplorerConfig,
    ExplorerModel,
    ExplorerOption,
    ExplorerView,
    highlight_style_code,
    highlight_terms,
    search_terms,
    style_bar,
    wrap_plain_line,
)
from nooa_cli.tui.job_explorer import (
    JobExplorerView,
    build_job_rows,
)
from nooa_cli.tui.todo_explorer import (
    TodoExplorerView,
    build_todo_rows,
)
from nooa_cli.tui.tui_application import _SubviewControl
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

# ─── explorer_base tests ─────────────────────────────────────────────────────


class TestExplorerBaseUtils:
    """Test shared utility functions."""

    def test_wrap_plain_line_empty(self):
        assert wrap_plain_line("", 80) == [""]

    def test_wrap_plain_line_short(self):
        assert wrap_plain_line("hello", 80) == ["hello"]

    def test_wrap_plain_line_long(self):
        result = wrap_plain_line("x" * 100, 50)
        assert len(result) >= 2
        assert all(len(line) <= 50 for line in result)

    def test_search_terms_empty(self):
        assert search_terms("") == []
        assert search_terms("   ") == []

    def test_search_terms_split(self):
        assert search_terms("foo bar") == ["foo", "bar"]

    def test_highlight_terms_no_terms(self):
        assert highlight_terms("hello world", []) == "hello world"

    def test_highlight_terms_with_match(self):
        result = highlight_terms("hello world", ["world"])
        assert "world" in result
        assert "\x1b[" in result

    def test_highlight_terms_with_current_occurrence(self):
        result = highlight_terms_with_current("foo bar foo", ["foo"], 1)
        assert highlight_style_code(current=True) in result
        assert highlight_style_code(current=False) in result

    def test_style_bar_no_ansi(self):
        assert style_bar("test", ansi=False) == "test"

    def test_style_bar_ansi(self):
        result = style_bar("test", ansi=True)
        assert "\x1b[" in result


class TestExplorerModel:
    """Test ExplorerModel navigation and search."""

    @staticmethod
    def _make_model(n=5):
        rows = [MagicMock(search_text=f"item {i}") for i in range(n)]
        return ExplorerModel(rows)

    def test_init(self):
        model = self._make_model()
        assert len(model.rows) == 5
        assert model.cursor == 0
        assert len(model.matches) == 5

    def test_move(self):
        model = self._make_model()
        model.move(2)
        assert model.cursor == 2
        model.move(-1)
        assert model.cursor == 1

    def test_move_clamps(self):
        model = self._make_model()
        model.move(-10)
        assert model.cursor == 0
        model.move(100)
        assert model.cursor == 4

    def test_set_query_filters(self):
        model = self._make_model()
        model.set_query("item 3")
        assert len(model.matches) == 1
        assert model.matches[0] == 3

    def test_set_query_empty_resets(self):
        model = self._make_model()
        model.set_query("item 3")
        model.set_query("")
        assert len(model.matches) == 5

    def test_toggle_focus(self):
        model = self._make_model()
        assert model.focus == "list"
        model.toggle_focus()
        assert model.focus == "detail"
        model.toggle_focus()
        assert model.focus == "list"

    def test_jump_home_end(self):
        model = self._make_model()
        model.move(3)
        model.jump_home()
        assert model.cursor == 0
        model.jump_end()
        assert model.cursor == 4


class TestExplorerView:
    """Test ExplorerView rendering and key handling."""

    @staticmethod
    def _make_view(n=5):
        rows = [MagicMock(search_text=f"item {i}") for i in range(n)]
        model = ExplorerModel(rows)
        config = ExplorerConfig(title="Test Explorer")
        return ExplorerView(model, config)

    def test_handle_key_quit_closes(self):
        view = self._make_view()
        assert view.handle_key("quit") == "close"

    def test_handle_key_slash_activates_search(self):
        view = self._make_view()
        view.handle_key("slash")
        assert view.model.search_active is True

    def test_handle_key_escape_deactivates_search(self):
        view = self._make_view()
        view.handle_key("slash")
        view.handle_key("escape")
        assert view.model.search_active is False

    def test_checklist_option_navigates_then_toggles_current_choice(self):
        view = self._make_view()
        changes = []
        option = ExplorerChecklistOption(
            "types",
            "Types",
            (("a", "A"), ("b", "B")),
            {"a", "b"},
            changes.append,
        )
        view.configure_options(option)
        view.handle_key("options")
        view.handle_key("down")
        assert option.choice_cursor == 1
        view.handle_key("space")
        assert option.checked == {"a"}
        assert changes == [{"a"}]

    def test_quit_still_closes_while_options_are_active(self):
        view = self._make_view()
        view.configure_options(
            ExplorerOption("filter", "Filter", (("all", "All"),), "all", lambda _value: None)
        )

        view.handle_key("options")

        assert view.handle_key("quit") == "close"

    def test_space_remains_search_input_outside_options_mode(self):
        view = self._make_view()
        view.handle_key("slash")
        view.model.edit_query("two")
        assert view.handle_key("space") == "handled"
        assert view.model.query == "two "

    def test_handle_key_down_moves(self):
        view = self._make_view()
        view.handle_key("down")
        assert view.model.cursor == 1

    def test_handle_key_tab_toggles_focus(self):
        view = self._make_view()
        view.handle_key("tab")
        assert view.model.focus == "detail"


def test_subview_control_preserves_vt_mouse_position_for_wheel_dispatch():
    calls = []
    control = _SubviewControl(
        "explorer",
        mouse_callback=lambda action, x, y: calls.append((action, x, y)) or True,
    )
    event = MouseEvent(
        position=Point(x=12, y=7),
        event_type=MouseEventType.SCROLL_DOWN,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )

    assert control.mouse_handler(event) is None
    assert calls == [("scroll_down", 12, 7)]


# ─── Job explorer tests ──────────────────────────────────────────────────────


class TestJobExplorer:
    """Test job explorer view."""

    def test_build_job_rows_none(self):
        assert build_job_rows(None) == []

    def test_build_job_rows_with_handles(self):
        qm = MagicMock()
        handle1 = MagicMock()
        handle1.name = "ci"
        handle1.label = "ci-pipeline"
        handle1.state = "running"
        handle1.values = ["line1", "line2"]
        qm.jobs.return_value = {"ci": "running"}
        qm.job.return_value = handle1
        ch = MagicMock()
        ch.qsize.return_value = 3
        qm.channels.return_value = {"ci": ch}

        rows = build_job_rows([JobSnapshot("ci", "ci-pipeline", "running", 3, ("line1", "line2"))])
        assert len(rows) == 1
        assert rows[0].channel == "ci"
        assert rows[0].state == "running"
        assert rows[0].delivered == 2
        assert rows[0].queued == 3

    def test_build_job_rows_accepts_observation_projection(self):
        rows = build_job_rows(
            [
                AgentJobSummary(
                    "logs",
                    "tail logs",
                    AgentJobState.RUNNING,
                    4,
                    ("first", "second"),
                )
            ]
        )

        assert len(rows) == 1
        assert rows[0].channel == "logs"
        assert rows[0].state == "running"
        assert rows[0].queued == 4
        assert rows[0].values == ["first", "second"]

    def test_job_explorer_ignores_unknown_actions(self):
        qm = MagicMock()
        handle = MagicMock()
        handle.name = "test-job"
        handle.label = "test"
        handle.state = "running"
        handle.values = []
        qm.jobs.return_value = {"test-job": "running"}
        qm.job.return_value = handle
        qm.channels.return_value = {}

        view = JobExplorerView([JobSnapshot("test-job", "test", "running", 0, ())])
        result = view.handle_key("text", "x")
        assert result == "ignored"


# ─── Todo explorer tests ─────────────────────────────────────────────────────


class TestTodoExplorer:
    """Test todo explorer view."""

    def test_build_todo_rows_none(self):
        assert build_todo_rows(None) == []

    def test_build_todo_rows_with_items(self):
        todo_mgr = MagicMock()
        todo1 = MagicMock()
        todo1.id = "abc12345"
        todo1.title = "Fix the bug"
        todo1.status = "open"
        todo1.deps = []
        todo1.created_at = "2025-01-01 12:00"
        todo1.notes = "Important fix"
        todo1.comments = []
        todo_mgr.list_todos.return_value = [todo1]

        rows = build_todo_rows(todo_mgr)
        assert len(rows) == 1
        assert rows[0].id == "abc12345"
        assert rows[0].title == "Fix the bug"
        assert rows[0].status == "open"

    def test_todo_explorer_ignores_unknown_actions(self):
        todo_mgr = MagicMock()
        todo1 = MagicMock()
        todo1.id = "xyz99999"
        todo1.title = "Some task"
        todo1.status = "open"
        todo1.deps = []
        todo1.created_at = "2025-01-01 12:00"
        todo1.notes = ""
        todo1.comments = []
        todo_mgr.list_todos.return_value = [todo1]

        view = TodoExplorerView(build_todo_rows(todo_mgr))
        result = view.handle_key("text", "x")
        assert result == "ignored"


def test_todo_explorer_grouped_options_filter_and_sort() -> None:
    rows = [
        MagicMock(status="done", created_at="2025-01-01", search_text="done", title="done"),
        MagicMock(status="open", created_at="2025-01-02", search_text="open", title="open"),
    ]
    view = TodoExplorerView(rows)
    view.handle_key("options")
    view.handle_key("down")
    assert [view.model.rows[i].status for i in view.model.matches] == ["open"]
    view.handle_key("right")
    view.handle_key("space")
    assert [view.model.rows[i].created_at for i in view.model.matches] == ["2025-01-02"]
