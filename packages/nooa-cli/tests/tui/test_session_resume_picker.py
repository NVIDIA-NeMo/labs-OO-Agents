import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui.resume_picker import (
    ResumePicker,
    ResumePickerModel,
    ResumePickerRow,
    _clip,
    fuzzy_match,
    render_resume_picker,
)
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl


def row(id: str, title: str, **kw) -> ResumePickerRow:
    return ResumePickerRow(id, title, "provider/model", "Agent", "/work", 1, 3, **kw)


def test_model_fuzzy_ranks_and_blocks_attached() -> None:
    model = ResumePickerModel([row("1", "alpha"), row("2", "resume feature"), row("3", "ransom")])
    model.set_query("rsm")
    assert [item.title for item, _ in model.matches][0] == "resume feature"
    assert fuzzy_match("rsm", "resume feature") is not None
    blocked = ResumePickerModel([row("1", "active", attached=True), row("2", "other")])
    assert blocked.current.id == "2"


def test_casefold_expansion_maps_highlights_to_original_source() -> None:
    result = fuzzy_match("ss", "Maße")
    assert result is not None
    assert result[1] == (2,)
    app = MagicMock()
    app.output.get_size.return_value = MagicMock(columns=60, rows=20)
    picker = ResumePicker([row("1", "Maße")], app)
    picker.buffer.text = "ss"
    highlighted = "".join(
        text for style, text in picker.list_control.text() if "resume-picker.match" in style
    )
    assert highlighted == "ß"


def test_picker_control_marks_fuzzy_match_fragments() -> None:
    app = MagicMock()
    app.output.get_size.return_value = MagicMock(columns=80, rows=20)
    picker = ResumePicker([row("1", "resume")], app)
    picker.buffer.text = "rsm"
    fragments = picker.container.body.body.children[1].content.text()
    assert any("class:resume-picker.match" in style for style, _ in fragments)


def test_rendered_frames_are_responsive_and_truthful() -> None:
    model = ResumePickerModel([row("12345678-full", "migration", current=True)])
    assert "provider/model" in render_resume_picker(model, 80, 16)
    assert "3 turns" in render_resume_picker(model, 60, 12)
    assert "provider/model" not in render_resume_picker(model, 50, 10)
    assert "at least 40 columns" in render_resume_picker(model, 39, 9)


def test_clip_uses_terminal_cells_and_preserves_graphemes() -> None:
    assert _clip("界界界", 5) == "界界…"
    assert _clip("ééé", 3) == "ééé"
    assert _clip("👩‍💻👩‍💻", 3) == "👩‍💻…"


def test_match_fragments_belong_to_fields_not_rendered_chrome() -> None:
    app = MagicMock()
    app.output.get_size.return_value = MagicMock(columns=60, rows=20)
    picker = ResumePicker([row("needle-id", "ordinary")], app)
    picker.buffer.text = "needle"
    fragments = picker.list_control.text()
    highlighted = "".join(text for style, text in fragments if "resume-picker.match" in style)
    assert highlighted == "needle"
    assert "id: needle-id" in "".join(text for _, text in fragments)


def test_semantic_states_survive_combination() -> None:
    app = MagicMock()
    app.output.get_size.return_value = MagicMock(columns=80, rows=24)
    picker = ResumePicker([row("1", "match me")], app)
    picker.buffer.text = "match"
    styles = " ".join(style for style, _ in picker.list_control.text())
    assert "class:resume-picker.selected" in styles
    assert "class:resume-picker.match" in styles


def test_required_terminal_frames_have_readable_floor() -> None:
    model = ResumePickerModel([row("needle-id", "会議 👩‍💻 é session")])
    model.set_query("needle")
    for width, height in ((100, 30), (80, 24), (60, 20), (40, 10)):
        frame = render_resume_picker(model, width, height)
        assert "Search:" not in frame
        assert "needle-id" in frame
        assert len(frame.splitlines()) <= height - 3
    assert render_resume_picker(model, 39, 9).splitlines() == [
        "Terminal too small",
        "Need at least 40 columns",
        "and 10 rows; now 39 x 9",
    ]


def _screen_for(picker: ResumePicker, width: int, height: int):
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    picker.app.output.get_size.return_value = MagicMock(columns=width, rows=height)
    # Freeze the size-specific fragments in a fresh control to avoid prompt_toolkit's
    # render cache while still exercising its real Window -> Screen cell path.
    control = FormattedTextControl(picker.list_control._text())
    screen = Screen()
    Window(control, wrap_lines=False).write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(xpos=0, ypos=0, width=width, height=height),
        parent_style="",
        erase_bg=False,
        z_index=None,
    )
    return screen


def test_actual_prompt_toolkit_screen_frames_and_semantic_cells() -> None:
    rows = [
        row("current-id", "current row", current=True),
        row("attached-id", "attached row", attached=True),
        row("needle-id", "selectable row"),
    ]
    app = MagicMock()
    picker = ResumePicker(rows, app)
    picker.buffer.text = "id"
    for width, height in ((100, 30), (80, 24), (60, 20), (40, 10), (39, 9)):
        screen = _screen_for(picker, width, height)
        visible = [
            "".join(screen.data_buffer[y][x].char for x in range(width)).rstrip()
            for y in range(height)
        ]
        assert any(visible)
        assert all(
            cell.width >= 0 for line in screen.data_buffer.values() for cell in line.values()
        )
        if (width, height) == (39, 9):
            assert visible[:3] == [
                "Terminal too small",
                "Need at least 40 columns",
                "and 10 rows; now 39 x 9",
            ]
        else:
            assert "sessions" in visible[0]
            assert any("Esc cancel" in line for line in visible)

    screen = _screen_for(picker, 80, 24)
    cells = [cell for line in screen.data_buffer.values() for cell in line.values()]
    assert any("class:resume-picker.selected" in cell.style for cell in cells)
    assert any("class:resume-picker.unavailable" in cell.style for cell in cells)
    assert any("class:resume-picker.match" in cell.style for cell in cells)


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

    empty = MagicMock(
        id="empty",
        name="empty",
        model="m",
        agent="A",
        working_dir="/w",
        last_active=1,
        turn_count=0,
    )
    resumable = MagicMock(
        id="resumable",
        name="kept",
        model="m",
        agent="A",
        working_dir="/w",
        last_active=2,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [empty, resumable])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
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

    meta = MagicMock(
        id="session-1",
        name="qjk",
        model="m",
        agent="A",
        working_dir="/w",
        last_active=1,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [meta])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        await harness.type_keys("qjk")
        await harness.wait_for(lambda: harness.app._resume_picker.model.query == "qjk")
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "height", "usable"),
    [(80, 24, True), (60, 20, True), (40, 10, True), (39, 9, False)],
)
async def test_full_application_screen_keeps_picker_help_visible(
    monkeypatch, width: int, height: int, usable: bool
) -> None:
    """Capture the complete Application, including Float/Box/Frame allocation."""
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import MutableRecordingOutput, TUIHarness

    sessions = [
        MagicMock(
            id=f"session-{index}",
            name=f"Populated session {index:02d}",
            model="provider/model",
            agent="Agent",
            working_dir="/work",
            last_active=index + 1,
            turn_count=index + 1,
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: sessions)
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
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
            assert any("20 sessions" in line for line in visible)
            assert any("Enter resume" in line and "Esc cancel" in line for line in visible)
        else:
            assert any("Terminal too small" in line for line in visible)
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None
