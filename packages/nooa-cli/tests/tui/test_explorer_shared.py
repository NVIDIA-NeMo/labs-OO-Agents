# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for job and todo explorer views."""

from __future__ import annotations

from unittest.mock import MagicMock

from nooa_cli.interactive import AgentJobState, AgentJobSummary
from nooa_cli.interactive.runtime import JobSnapshot
from nooa_cli.tui.activity_overlay import ActivityOverlayView
from nooa_cli.tui.event_explorer import (
    highlight_terms_with_current,
)
from nooa_cli.tui.explorer_base import (
    FTS_ACTIVE_STYLE,
    ExplorerConfig,
    ExplorerModel,
    ExplorerView,
    display_line,
    highlight_terms,
    search_terms,
    style_bar,
    wrap_plain_line,
)
from nooa_cli.tui.job_explorer import (
    JobExplorerView,
    build_job_rows,
)
from nooa_cli.tui.output import CodeExecution, TableOutput, TextOutput
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
        assert "\x1b[30;106m" in result  # second occurrence highlighted differently

    def test_display_line_pads(self):
        result = display_line("hi", 10, [], ansi=False)
        assert len(result) == 10

    def test_style_bar_no_ansi(self):
        assert style_bar("test", ansi=False) == "test"

    def test_style_bar_ansi(self):
        result = style_bar("test", ansi=True)
        assert "\x1b[" in result

    def test_fts_style_uses_explicit_high_contrast_truecolor(self):
        assert FTS_ACTIVE_STYLE == "\x1b[1;38;2;255;255;255;48;2;95;28;75m"


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

    def test_render_produces_string(self):
        view = self._make_view()
        output = view.render(80, 24)
        assert isinstance(output, str)
        assert "Test Explorer" in output

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

    def test_handle_key_down_moves(self):
        view = self._make_view()
        view.handle_key("down")
        assert view.model.cursor == 1

    def test_handle_key_tab_toggles_focus(self):
        view = self._make_view()
        view.handle_key("tab")
        assert view.model.focus == "detail"

    def test_f2_toggles_native_terminal_selection(self):
        view = self._make_view()

        assert view.mouse_support is True
        assert view.handle_key("native_selection") == "handled"
        assert view.mouse_support is False
        assert "F2 mouse/wheel" in view.render(80, 24)

        view.handle_key("native_selection")
        assert view.mouse_support is True

    def test_mouse_wheel_routes_to_pane_under_pointer(self):
        class LongDetailView(ExplorerView):
            def detail_lines(self, row, width):
                return [f"detail line {i}" for i in range(50)]

        rows = [MagicMock(search_text=f"item {i}") for i in range(10)]
        view = LongDetailView(ExplorerModel(rows), ExplorerConfig(title="Mouse Explorer"))
        view.render(80, 16)

        divider_y = view.model._last_divider_y
        view.handle_mouse("scroll_down", 5, divider_y - 1)
        assert view.model.focus == "list"
        assert view.model.cursor == 3

        view.handle_mouse("scroll_down", 5, divider_y + 1)
        assert view.model.focus == "detail"
        assert view.model.detail_offset == 3

    def test_mouse_click_selects_visible_row_and_focuses_list(self):
        view = self._make_view(8)
        view.render(80, 18)

        assert view.handle_mouse("click", 12, 3) == "handled"
        assert view.model.cursor == 2
        assert view.model.focus == "list"

    def test_ctrl_y_copies_current_detail_through_host_callback(self):
        class CopyView(ExplorerView):
            def detail_lines(self, row, width):
                return ["rendered", "detail"]

            def copy_text(self):
                return "exact\ndetail"

        view = CopyView(
            ExplorerModel([MagicMock(search_text="copy me")]),
            ExplorerConfig(title="Copy Explorer"),
        )
        copied = []
        view.set_copy_handler(lambda text: copied.append(text) or True)

        assert view.handle_key("copy") == "handled"
        assert copied == ["exact\ndetail"]
        assert "Copied item" in view.render(80, 12).splitlines()[0]

    def test_header_copy_action_is_clickable(self):
        class CopyView(ExplorerView):
            def detail_lines(self, row, width):
                return ["rendered detail"]

            def copy_text(self):
                return "clicked detail"

        view = CopyView(
            ExplorerModel([MagicMock(search_text="copy me")]),
            ExplorerConfig(title="Copy Explorer"),
        )
        copied = []
        view.set_copy_handler(lambda text: copied.append(text) or True)
        view.render(80, 12)

        assert view.handle_mouse("click", 79, 0) == "handled"
        assert copied == ["clicked detail"]

    def test_copy_failure_is_visible_without_closing_explorer(self):
        class CopyView(ExplorerView):
            def copy_text(self):
                return "copy me"

        view = CopyView(self._make_view().model, ExplorerConfig(title="Test Explorer"))
        view.set_copy_handler(lambda _text: False)

        assert view.handle_key("copy") == "handled"
        assert "Copy unavailable" in view.render(80, 12).splitlines()[0]

    def test_render_empty(self):
        rows = []
        model = ExplorerModel(rows)
        config = ExplorerConfig(title="Empty", empty_message="Nothing here.")
        view = ExplorerView(model, config)
        output = view.render(80, 24)
        assert "Nothing here." in output


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


def test_subview_control_dispatches_left_click_with_coordinates():
    calls = []
    control = _SubviewControl(
        "explorer",
        mouse_callback=lambda action, x, y: calls.append((action, x, y)) or True,
    )
    down = MouseEvent(
        position=Point(x=9, y=4),
        event_type=MouseEventType.MOUSE_DOWN,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )
    up = MouseEvent(
        position=Point(x=9, y=4),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )

    assert control.mouse_handler(down) is None
    assert control.mouse_handler(up) is None
    assert calls == [("click", 9, 4)]


def test_subview_control_ignores_release_without_matching_press() -> None:
    calls = []
    control = _SubviewControl(
        "explorer",
        mouse_callback=lambda action, x, y: calls.append((action, x, y)) or True,
    )
    release = MouseEvent(
        position=Point(x=9, y=4),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )

    assert control.mouse_handler(release) is NotImplemented
    assert calls == []


def test_subview_control_does_not_turn_drag_into_click() -> None:
    calls = []
    control = _SubviewControl(
        "explorer",
        mouse_callback=lambda action, x, y: calls.append((action, x, y)) or True,
    )
    events = [
        MouseEvent(Point(x=9, y=4), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset()),
        MouseEvent(Point(x=10, y=4), MouseEventType.MOUSE_MOVE, MouseButton.LEFT, frozenset()),
        MouseEvent(Point(x=9, y=4), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()),
    ]

    assert control.mouse_handler(events[0]) is None
    assert control.mouse_handler(events[1]) is None
    assert control.mouse_handler(events[2]) is NotImplemented
    assert calls == []


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

    def test_job_explorer_view_renders(self):
        qm = MagicMock()
        handle = MagicMock()
        handle.name = "monitor"
        handle.label = "health-check"
        handle.state = "done"
        handle.values = ["ok"]
        qm.jobs.return_value = {"monitor": "done"}
        qm.job.return_value = handle
        qm.channels.return_value = {}

        view = JobExplorerView([JobSnapshot("monitor", "health-check", "done", 0, ("ok",))])
        output = view.render(80, 24)
        assert "Job Explorer" in output

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

    def test_todo_explorer_view_renders(self):
        todo_mgr = MagicMock()
        todo1 = MagicMock()
        todo1.id = "def67890"
        todo1.title = "Write tests"
        todo1.status = "done"
        todo1.deps = []
        todo1.created_at = "2025-01-01 12:00"
        todo1.notes = ""
        todo1.comments = []
        todo_mgr.list_todos.return_value = [todo1]

        view = TodoExplorerView(build_todo_rows(todo_mgr))
        output = view.render(80, 24)
        assert "Todo Explorer" in output

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

    def test_todo_explorer_empty(self):
        todo_mgr = MagicMock()
        todo_mgr.list_todos.return_value = []

        view = TodoExplorerView(build_todo_rows(todo_mgr))
        output = view.render(80, 24)
        assert "No todos." in output


def test_activity_overlay_copies_full_semantic_snapshot() -> None:
    view = ActivityOverlayView([TextOutput("first line\nsecond line", "info")])
    copied: list[str] = []
    view.set_copy_handler(lambda text: copied.append(text) or True)

    assert view.handle_key("copy") == "handled"
    assert copied == ["first line\nsecond line"]
    assert "Copied item" in view.render(80, 10).splitlines()[0]


def test_activity_overlay_copy_is_width_independent_and_preserves_code() -> None:
    code = "value = '" + "x" * 140 + "'\nprint(value)"
    view = ActivityOverlayView(
        [
            TableOutput(
                columns=["Field", "Value"],
                rows=[["phase", "executing_python"]],
                footer="Still running.",
            ),
            CodeExecution(tool_call_id="call-1", code=code, stdout="done\n"),
        ]
    )

    before = view.copy_text()
    view.render(40, 8)
    narrow = view.copy_text()
    view.render(160, 20)

    assert (
        narrow
        == before
        == (
            "Field\tValue\nphase\texecuting_python\n\nStill running."
            f"\n\n```python\n{code}\n```\n\nstdout:\ndone\n"
        )
    )
