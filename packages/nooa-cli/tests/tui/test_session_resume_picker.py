import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui.resume_picker import (
    ResumePicker,
    ResumePickerModel,
    ResumePickerRow,
    ResumePickerTurn,
    _clip,
    fuzzy_match,
    render_resume_picker,
)
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType


def row(id: str, title: str, **kw) -> ResumePickerRow:
    values = {
        "model": "provider/model",
        "agent": "Agent",
        "working_directory": "/work",
        "last_active": 1,
        "turn_count": 3,
        "turns": (
            ResumePickerTurn("user", f"question for {id}"),
            ResumePickerTurn("agent", f"answer for {id}"),
        ),
    }
    values.update(kw)
    return ResumePickerRow(id=id, title=title, **values)


def test_model_searches_title_and_full_recent_conversation() -> None:
    model = ResumePickerModel(
        [
            row("1", "alpha"),
            row(
                "2",
                "resume feature",
                turns=(
                    ResumePickerTurn("user", "ordinary question"),
                    ResumePickerTurn("agent", "answer contains deep needle"),
                ),
            ),
            row("3", "ransom"),
        ]
    )
    model.set_query("rsm")
    assert model.matches[0].row.title == "resume feature"
    assert fuzzy_match("rsm", "resume feature") is not None
    model.set_query("deep needle")
    assert [match.row.id for match in model.matches] == ["2"]


def test_default_scope_is_all_and_filter_label_is_explicit() -> None:
    local = row("local", "Local", working_directory=str(Path.cwd()))
    other = row("other", "Other", working_directory="/elsewhere", last_active=30, created_at=5)
    model = ResumePickerModel([local, other], cwd=str(Path.cwd()))
    assert {match.row.id for match in model.matches} == {"local", "other"}
    assert "Filter: All sessions" in render_resume_picker(model, 80, 20)
    model.toggle_filter()
    assert [match.row.id for match in model.matches] == ["local"]
    assert "Filter: This directory" in render_resume_picker(model, 80, 20)


def test_sort_labels_explain_updated_versus_created() -> None:
    older_created = row("updated", "Recently active", last_active=30, created_at=1)
    newer_created = row("created", "Recently created", last_active=20, created_at=10)
    model = ResumePickerModel([older_created, newer_created])
    assert [match.row.id for match in model.matches] == ["updated", "created"]
    assert "Sort: Recent activity" in render_resume_picker(model, 80, 20)
    model.toggle_sort()
    assert [match.row.id for match in model.matches] == ["created", "updated"]
    assert "Sort: Creation date" in render_resume_picker(model, 80, 20)


def test_rows_are_two_lines_and_keep_state_and_title_on_first_line() -> None:
    model = ResumePickerModel(
        [
            row(
                "attached",
                "Important title",
                attached=True,
                turns=(ResumePickerTurn("agent", "a very long reply " * 20),),
            )
        ]
    )
    frame = render_resume_picker(model, 60, 16).splitlines()
    title_line = next(line for line in frame if "Important title" in line)
    assert "attached" in title_line
    assert "a very long reply" not in title_line
    assert any("Agent: a very long reply" in line for line in frame)


def test_selection_can_inspect_attached_but_cannot_resume_it() -> None:
    model = ResumePickerModel([row("1", "active", attached=True), row("2", "other")])
    assert model.current.id == "1"
    assert model.can_select is False
    assert "attached  active" in render_resume_picker(model, 80, 20)
    model.move(1)
    assert model.current.id == "2"
    assert model.can_select is True
    assert "detached  other" in render_resume_picker(model, 80, 20)


def test_tab_cycles_controls_and_space_changes_only_selected_control() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one")], app)
    assert picker.active_control == "search"
    picker.focus_next()
    assert picker.active_control == "filter"
    picker.change_active_control()
    assert picker.model.filter_cwd is True
    picker.focus_next()
    assert picker.active_control == "sort"
    picker.change_active_control()
    assert picker.model.sort_updated is False
    picker.focus_next()
    assert picker.active_control == "search"


def test_preview_scroll_is_independent_and_selection_resets_to_tail() -> None:
    turns = tuple(ResumePickerTurn("user", f"message {index}") for index in range(12))
    model = ResumePickerModel([row("1", "one", turns=turns), row("2", "two", turns=turns)])
    model.scroll_preview(-3, line_count=30, height=5)
    assert model.preview_offset == 22
    selected = model.selected
    model.list_offset = 0
    model.move(1)
    assert model.selected != selected
    assert model.preview_offset == 10**9
    assert model.list_offset == 0


def test_mouse_wheel_routes_to_list_and_preview_separately() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row(str(index), f"title {index}") for index in range(5)], app)
    down = MouseEvent(Point(0, 0), MouseEventType.SCROLL_DOWN, MouseButton.NONE, frozenset())
    picker.list_control.mouse_handler(down)
    assert picker.model.selected == 1
    picker.preview_control.viewport = (30, 3)
    picker.model.preview_offset = 0
    picker.preview_control.mouse_handler(down)
    assert picker.model.selected == 1
    assert picker.model.preview_offset > 0


def test_mouse_click_maps_two_line_rows_to_session() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row(str(index), f"title {index}") for index in range(4)], app)
    click = MouseEvent(Point(1, 3), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
    picker.list_control.mouse_handler(click)
    assert picker.model.selected == 1


def test_render_has_three_separated_areas_and_fixed_help() -> None:
    model = ResumePickerModel([row("one", "Visible title")])
    frame = render_resume_picker(model, 90, 24)
    assert "Resume a previous session" in frame
    assert "[Search:" in frame
    assert "Sessions  ·  updated   state     title" not in frame  # layout-only heading
    assert "Preview · Visible title" in frame
    assert frame.count("─" * 90) == 3
    assert frame.splitlines()[-1].endswith("Esc cancel")


def test_row_metadata_is_sanitized_without_breaking_two_line_layout() -> None:
    model = ResumePickerModel(
        [
            row(
                "unsafe",
                "Title\nwith\x1b[31m controls",
                turns=(ResumePickerTurn("agent", "Reply\rwith\x1b[2J controls"),),
            )
        ]
    )
    frame = render_resume_picker(model, 80, 20)
    assert "\x1b" not in frame
    assert "Title with controls" in frame
    assert r"Agent: Reply\rwith\x1b[2J controls" in frame


def test_clip_uses_terminal_cells_and_preserves_graphemes() -> None:
    assert _clip("界界界", 5) == "界界…"
    assert _clip("ééé", 3) == "ééé"
    assert _clip("👩‍💻👩‍💻", 3) == "👩‍💻…"


def test_required_terminal_frames_have_truthful_floor() -> None:
    model = ResumePickerModel([row("needle-id", "会議 👩‍💻 é session")])
    for width, height in ((120, 30), (80, 24), (60, 20), (48, 13)):
        frame = render_resume_picker(model, width, height)
        assert "会議" in frame
        assert len(frame.splitlines()) <= height
    assert render_resume_picker(model, 47, 12).splitlines() == [
        "Terminal too small",
        "Need 48 x 13; now 47 x 12",
    ]


@pytest.mark.asyncio
async def test_session_resume_without_id_opens_dedicated_picker() -> None:
    from nooa_cli.tui.commands import SessionCommand

    frontend = MagicMock()
    frontend.open_session_resume_dialog = AsyncMock(return_value=None)
    cmd = SessionCommand(frontend, MagicMock(), MagicMock())
    assert cmd.validate_args(["resume"]) == (True, None)
    result = await cmd.execute(["resume"])
    frontend.open_session_resume_dialog.assert_awaited_once()
    assert result.success


@pytest.mark.asyncio
async def test_resume_alias_opens_the_same_picker() -> None:
    from nooa_cli.tui.commands import ResumeCommand

    frontend = MagicMock()
    frontend.open_session_resume_dialog = AsyncMock(return_value=None)
    cmd = ResumeCommand(frontend, MagicMock(), MagicMock())
    assert cmd.validate_args([]) == (True, None)
    result = await cmd.execute([])
    frontend.open_session_resume_dialog.assert_awaited_once()
    assert result.success


def test_resume_commands_reject_extra_ids() -> None:
    from nooa_cli.tui.commands import ResumeCommand, SessionCommand

    frontend = MagicMock()
    assert SessionCommand(frontend, MagicMock(), MagicMock()).validate_args(
        ["resume", "one", "two"]
    ) == (False, "Usage: /session resume [session_id]")
    assert ResumeCommand(frontend, MagicMock(), MagicMock()).validate_args(["one", "two"]) == (
        False,
        "Usage: /resume [session_id]",
    )


@pytest.mark.asyncio
async def test_picker_excludes_empty_sessions(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    empty = SimpleNamespace(
        id="empty",
        name="empty",
        model="m",
        agent="A",
        working_dir=str(Path.cwd()),
        last_active=1,
        turn_count=0,
    )
    resumable = SimpleNamespace(
        id="resumable",
        name="kept",
        model="m",
        agent="A",
        working_dir=str(Path.cwd()),
        last_active=2,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [empty, resumable])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "recent_turns",
        classmethod(
            lambda cls, value, limit=12: [
                SimpleNamespace(role="user", content=f"question for {value}"),
                SimpleNamespace(role="agent", content=f"preview for {value}"),
            ]
        ),
    )
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        assert [item.id for item in harness.app._resume_picker.model.rows] == ["resumable"]
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
async def test_real_prompt_toolkit_routes_search_navigation_and_cancel(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    sessions = [
        SimpleNamespace(
            id="session-1",
            name="qjk",
            model="m",
            agent="A",
            working_dir=str(Path.cwd()),
            started_at=1,
            last_active=2,
            turn_count=1,
        ),
        SimpleNamespace(
            id="session-2",
            name="other",
            model="m",
            agent="A",
            working_dir="/another/project",
            started_at=3,
            last_active=1,
            turn_count=1,
        ),
    ]
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: sessions)
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "recent_turns",
        classmethod(
            lambda cls, value, limit=12: [
                SimpleNamespace(role="user", content=f"question for {value}"),
                SimpleNamespace(role="agent", content=f"preview for {value}"),
            ]
        ),
    )
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        await harness.type_keys("qjk")
        await harness.wait_for(lambda: harness.app._resume_picker.model.query == "qjk")
        picker = harness.app._resume_picker
        assert [match.row.id for match in picker.model.matches] == ["session-1"]
        picker.buffer.text = ""
        await harness.wait_for(lambda: picker.model.query == "")
        await harness.press("tab")
        await harness.wait_for(lambda: picker.active_control == "filter")
        await harness.type_keys(" ")
        await harness.wait_for(lambda: picker.model.filter_cwd)
        assert [match.row.id for match in picker.model.matches] == ["session-1"]
        await harness.press("tab")
        await harness.wait_for(lambda: picker.active_control == "sort")
        await harness.type_keys(" ")
        await harness.wait_for(lambda: not picker.model.sort_updated)
        assert [match.row.id for match in picker.model.matches] == ["session-1"]
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "height", "usable"),
    [(120, 30, True), (80, 24, True), (60, 20, True), (48, 13, True), (47, 12, False)],
)
async def test_full_application_screen_keeps_picker_help_visible(
    monkeypatch, width: int, height: int, usable: bool
) -> None:
    """Capture the complete Application, including Float/Box/Frame allocation."""
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import MutableRecordingOutput, TUIHarness

    sessions = [
        SimpleNamespace(
            id=f"session-{index}",
            name=f"Populated session {index:02d}",
            model="provider/model",
            agent="Agent",
            working_dir=str(Path.cwd()),
            started_at=1_700_000_000 + index,
            last_active=1_700_000_100 + index,
            turn_count=index + 1,
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: sessions)
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "recent_turns",
        classmethod(
            lambda cls, value, limit=12: [
                SimpleNamespace(role="user", content=f"question for {value}"),
                SimpleNamespace(role="agent", content=f"preview for {value}"),
            ]
        ),
    )
    output = MutableRecordingOutput(columns=width, rows=height)
    async with TUIHarness(output=output, full_screen=True) as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        harness.app._app.invalidate()
        await harness.wait_for(
            lambda: (
                harness.app._app.renderer.last_rendered_screen is not None
                and any(
                    cell.char.strip()
                    for line in harness.app._app.renderer.last_rendered_screen.data_buffer.values()
                    for cell in line.values()
                )
            )
        )
        screen = harness.app._app.renderer.last_rendered_screen
        visible = [
            "".join(screen.data_buffer[y][x].char for x in range(width)).rstrip()
            for y in range(height)
        ]
        if usable:
            joined = "\n".join(visible)
            assert "20 sessions" in joined
            assert "Search" in joined
            assert "Filter: All sessions" in joined
            assert "Sort: Recent activity" in joined
            assert "Conversation preview" in joined
            assert joined.count("─") >= width * 3
            assert "preview for session-19" in joined
            assert "❯" in joined and "y ago" in joined
            assert any(("Enter" in line or "↵" in line) and "Esc" in line for line in visible)
            assert all(len(line) <= width for line in visible)
        else:
            assert any("Terminal too small" in line for line in visible)
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None
