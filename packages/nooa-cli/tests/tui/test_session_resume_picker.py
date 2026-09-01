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
    _row_fragments,
    _semantic_preview_selection,
    literal_match,
    render_resume_picker,
)
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.cells import cell_len


def test_semantic_preview_selection_removes_user_and_agent_chrome() -> None:
    rendered_selection = (
        "▔▔▔▔▔▔▔▔\n"
        " ❯ hello world   \n"
        "   continued   \n"
        "▁▁▁▁▁▁▁▁\n"
        "OO:\n"
        "answer text"
    )

    assert _semantic_preview_selection(rendered_selection) == (
        "hello world\ncontinued\nanswer text"
    )


def test_semantic_preview_selection_preserves_partial_plain_text() -> None:
    assert _semantic_preview_selection("partial OO: text") == "partial OO: text"


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
    model.set_query("resume")
    assert model.matches[0].row.title == "resume feature"
    assert literal_match("RESUME", "resume feature") is not None
    assert literal_match("rsm", "resume feature") is None
    model.set_query("deep needle")
    assert [match.row.id for match in model.matches] == ["2"]


def test_search_excludes_noncontiguous_and_hidden_metadata_matches() -> None:
    model = ResumePickerModel(
        [
            row("hidden", "ordinary", working_directory="/work/grep-project"),
            row("scattered", "g___r___e___p"),
            row("visible", "grep results"),
        ]
    )

    model.set_query("grep")

    assert [match.row.id for match in model.matches] == ["visible"]
    assert model.matches[0].field == "title"


def test_filter_and_sort_reuse_cached_query_matches(monkeypatch) -> None:
    import nooa_cli.tui.resume_picker as picker_module

    model = ResumePickerModel([row("one", "needle"), row("two", "other", attached=True)])
    model.set_query("needle")

    def unexpected_match(query: str, value: str):
        raise AssertionError("filter/sort must not rescan transcript search fields")

    monkeypatch.setattr(picker_module, "literal_match", unexpected_match)
    model.toggle_filter()
    model.toggle_sort()
    model.toggle_filter()


def test_state_filter_defaults_to_detached_and_cycles_all_states() -> None:
    detached = row("detached", "Detached")
    attached = row("attached", "Attached", attached=True, last_active=30)
    model = ResumePickerModel([detached, attached])
    assert [match.row.id for match in model.matches] == ["detached"]
    assert "Filter: ✓ Not attached" in render_resume_picker(model, 80, 20)
    model.toggle_filter()
    assert [match.row.id for match in model.matches] == ["attached"]
    assert "Filter: ✗ Attached" in render_resume_picker(model, 80, 20)
    model.toggle_filter()
    assert {match.row.id for match in model.matches} == {"detached", "attached"}
    assert "Filter: ✓/✗ All" in render_resume_picker(model, 80, 20)


def test_sort_labels_explain_updated_versus_created() -> None:
    older_created = row("updated", "Recently active", last_active=30, created_at=1)
    newer_created = row("created", "Recently created", last_active=20, created_at=10)
    model = ResumePickerModel([older_created, newer_created])
    assert [match.row.id for match in model.matches] == ["updated", "created"]
    assert "Sort: Recent activity" in render_resume_picker(model, 80, 20)
    model.toggle_sort()
    assert [match.row.id for match in model.matches] == ["created", "updated"]
    assert "Sort: Creation date" in render_resume_picker(model, 80, 20)


def test_creation_date_sort_is_primary_with_a_search_query() -> None:
    older_exact = row("old", "needle", last_active=30, created_at=1)
    newer_weaker = row("new", "prefix needle", last_active=20, created_at=10)
    model = ResumePickerModel([older_exact, newer_weaker])
    model.set_query("needle")

    assert [match.row.id for match in model.matches] == ["old", "new"]
    model.toggle_sort()
    assert [match.row.id for match in model.matches] == ["new", "old"]


def test_rows_are_one_line_with_state_title_and_latest_agent_message() -> None:
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
    model.state_filter = "all"
    model.set_query("")
    frame = render_resume_picker(model, 80, 16).splitlines()
    row_line = next(line for line in frame if "Important title" in line)
    assert "✗" in row_line
    assert "a very long reply" in row_line
    assert sum("Important title" in line for line in frame) == 2  # row plus preview heading


def test_selection_can_inspect_attached_but_cannot_resume_it() -> None:
    model = ResumePickerModel([row("1", "active", attached=True), row("2", "other")])
    model.state_filter = "all"
    model.set_query("")
    model.select(next(index for index, match in enumerate(model.matches) if match.row.id == "1"))
    assert model.current.id == "1"
    assert model.can_select is False
    assert "✗  active" in render_resume_picker(model, 80, 20)
    model.move(1)
    assert model.current.id == "2"
    assert model.can_select is True
    assert "✓  other" in render_resume_picker(model, 80, 20)


def test_resume_picker_specializes_shared_explorer_browser() -> None:
    from nooa_cli.tui.fullscreen_browser import ExplorerBrowser

    assert issubclass(ResumePicker, ExplorerBrowser)
    assert ResumePicker.preview_selection is ExplorerBrowser.preview_selection
    assert ResumePicker.focus_initial is ExplorerBrowser.focus_initial


def test_tab_cycles_only_list_and_preview() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one")], app)
    assert picker.active_control == "list"
    picker.focus_next()
    assert picker.active_control == "preview"
    picker.focus_next()
    assert picker.active_control == "list"
    picker.focus_previous()
    assert picker.active_control == "preview"


def test_resume_picker_shows_copy_status_in_its_footer() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker(
        [row("1", "one")], app, selection_status=lambda: "Copied 12 characters"
    )

    assert picker._help_text() == "Copied 12 characters"


def test_options_mode_selects_and_changes_filter_and_sort() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one")], app)
    picker.toggle_options()
    assert picker.option_cursor == "filter"
    picker.change_option()
    assert picker.model.state_filter == "attached"
    picker.move_option(1)
    assert picker.option_cursor == "sort"
    picker.change_option()
    assert picker.model.sort_updated is False
    assert picker.close_options()
    assert picker.option_cursor is None
    assert picker.active_control == "list"


def test_search_text_and_brackets_share_active_highlight() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one")], app)
    picker.buffer.text = "needle"
    label_style = picker._search_label()[0][0]
    close_style = picker._search_close()[0][0]
    assert "control-focused" in label_style
    assert "control-focused" in close_style
    assert picker.query_window.style() == "class:fullscreen-browser.control-focused"
    picker.activate_control("preview")
    assert "control-focused" not in picker._search_label()[0][0]
    assert picker.query_window.style() == ""


def test_only_selected_option_is_highlighted_in_options_mode() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one")], app)
    assert "control-focused" not in picker.filter_control._text()[0][0]
    assert "control-focused" not in picker.sort_control._text()[0][0]
    picker.toggle_options()
    assert "control-focused" not in picker._search_label()[0][0]
    assert "control-focused" not in picker._search_close()[0][0]
    assert "control-focused" not in picker.query_window.style()
    assert "control-focused" in picker.filter_control._text()[0][0]
    assert "control-focused" not in picker.sort_control._text()[0][0]
    picker.move_option(1)
    assert "control-focused" not in picker.filter_control._text()[0][0]
    assert "control-focused" in picker.sort_control._text()[0][0]


def test_active_rail_marks_only_current_area() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one")], app)
    list_rail = picker._active_rail("list")[0][1].splitlines()
    assert list_rail[0] == "▌"
    assert len(list_rail) == 24
    assert picker._active_rail("preview")[0][1].splitlines()[0] == "│"
    picker.activate_control("preview")
    assert picker._active_rail("preview")[0][1].splitlines()[0] == "▌"
    assert picker._active_rail("list")[0][1].splitlines()[0] == "│"


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
    picker.list_control.viewport = (80, 2)
    picker.list_control.mouse_handler(down)
    assert picker.model.selected == 0
    assert picker.model.list_offset == 3
    picker.preview_control.viewport = (30, 3)
    transcript = picker._preview_model(30)
    assert transcript is not None and transcript.viewport.follows_tail
    picker.preview_control.mouse_handler(
        MouseEvent(Point(0, 0), MouseEventType.SCROLL_UP, MouseButton.NONE, frozenset())
    )
    assert picker.model.selected == 0
    assert transcript.viewport.follows_tail is False


def test_mouse_click_maps_one_line_rows_to_session() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row(str(index), f"title {index}") for index in range(4)], app)
    click = MouseEvent(Point(1, 1), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
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
    assert "Esc cancel" in frame.splitlines()[2]


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
    assert r"Reply\rwith\x1b[2J controls" in frame


def test_clip_uses_terminal_cells_and_preserves_graphemes() -> None:
    assert _clip("界界界", 5) == "界界…"
    assert _clip("ééé", 3) == "ééé"
    assert _clip("👩‍💻👩‍💻", 3) == "👩‍💻…"


def test_header_columns_align_with_row_columns() -> None:
    model = ResumePickerModel([row("one", "Aligned title")])
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker(model.rows, app)
    header = "".join(text for _style, text in picker._list_header())
    first = "".join(text for _style, text in _row_fragments(picker.model.matches[0], True, 80)[0])
    assert header.index("updated") == 3
    assert header.index("st") == first.index("✓")
    assert cell_len(header[: header.index("title")]) == cell_len(
        first[: first.index("Aligned title")]
    )
    assert cell_len(header[: header.index("last agent message")]) == cell_len(
        first[: first.index("answer for one")]
    )


def test_selection_marker_moves_before_viewport_scrolls() -> None:
    model = ResumePickerModel([row(str(index), f"title {index}") for index in range(5)])
    first = [
        "".join(text for _style, text in line)
        for _, match in model.visible(3)
        for line in _row_fragments(match, match.row.id == model.current.id, 80)
    ]
    model.move(1)
    second = [
        "".join(text for _style, text in line)
        for _, match in model.visible(3)
        for line in _row_fragments(match, match.row.id == model.current.id, 80)
    ]
    assert first[0].startswith("❯")
    assert second[0].startswith("  ")
    assert second[1].startswith("❯")
    assert model.list_offset == 0


def test_live_preview_reports_empty_conversation() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("empty", "Empty", turns=())], app)
    assert picker.preview_text(40, 5) == [
        ("class:fullscreen-browser.empty", "No conversation preview")
    ]


def test_preview_uses_live_scrollback_visual_language() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("one", "One")], app)
    plain = "".join(text for _style, text, *_ in picker.preview_text(40, 12))
    assert "❯ question for one" in plain
    assert "OO:" in plain
    assert "You:" not in plain
    assert "Agent:" not in plain


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
    session = SessionCommand(frontend, MagicMock(), MagicMock())
    assert session.validate_args(["list"]) == (False, "Unknown subcommand `list`")
    assert "/session list" not in session.help_text()
    assert ResumeCommand(frontend, MagicMock(), MagicMock()).validate_args(["one", "two"]) == (
        False,
        "Usage: /resume [session_id]",
    )


@pytest.mark.asyncio
async def test_concurrent_picker_open_is_rejected_while_rows_load(monkeypatch) -> None:
    import threading

    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    loading = threading.Event()
    release = threading.Event()

    def slow_list(cls, limit=None):
        loading.set()
        release.wait(timeout=1)
        return []

    monkeypatch.setattr(sm.SessionManager, "list_sessions", classmethod(slow_list))
    async with TUIHarness() as harness:
        first = asyncio.create_task(harness.app.open_session_resume_dialog())
        await asyncio.to_thread(loading.wait, 1)
        assert await harness.app.open_session_resume_dialog() is None
        release.set()
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        await harness.press("escape")
        assert await asyncio.wait_for(first, 1) is None


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
        "load_turns",
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
async def test_resume_picker_f2_temporarily_restores_native_selection(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    session = SimpleNamespace(
        id="resumable",
        name="kept",
        model="m",
        agent="A",
        working_dir=str(Path.cwd()),
        last_active=2,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [session])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "load_turns",
        classmethod(lambda cls, value, limit=12: [SimpleNamespace(role="agent", content="copy")]),
    )
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        assert bool(harness.app._app.mouse_support()) is True

        await harness.press("f2")
        await harness.wait_for(lambda: not bool(harness.app._app.mouse_support()))

        await harness.press("f2")
        await harness.wait_for(lambda: bool(harness.app._app.mouse_support()))
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
async def test_filter_change_prepares_new_preview_without_blocking(monkeypatch) -> None:
    import threading

    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("attached", "attached", attached=True)], app)
    picker.preview_control.viewport = (40, 5)
    started = threading.Event()
    release = threading.Event()

    def build_preview(selected, width, height):
        from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

        started.set()
        assert release.wait(timeout=1)
        return FullscreenTranscriptModel(show_trailing_blank=False)

    monkeypatch.setattr(ResumePicker, "_build_preview_model", staticmethod(build_preview))
    picker.cycle_filter()

    key = ("attached", 40)
    assert key in picker._preview_tasks
    assert key not in picker._preview_models
    await asyncio.wait_for(asyncio.to_thread(started.wait), 1)
    assert not picker._preview_tasks[key].done()
    release.set()
    await picker._preview_tasks[key]
    assert key in picker._preview_models


def test_preview_search_highlights_and_cycles_transcript_matches() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker(
        [
            row(
                "1",
                "one",
                turns=(ResumePickerTurn("agent", "before needle middle needle after"),),
            )
        ],
        app,
    )
    picker.preview_control.viewport = (12, 2)
    picker.buffer.text = "needle"

    fragments = picker.preview_text(12, 2)
    assert picker.preview_search_position() == (1, 2)
    assert any("transcript-search-current" in style for style, _text in fragments)
    picker.activate_control("preview")
    picker.navigate_vertical(1)
    assert picker.preview_search_position() == (2, 2)
    assert "match 2/2" in picker._preview_header()[0][1]
    picker.navigate_vertical(-1)
    assert picker.preview_search_position() == (1, 2)


def test_preview_measurement_redraw_does_not_cancel_drag() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker([row("1", "one", turns=(ResumePickerTurn("agent", "alpha beta"),))], app)
    picker.preview_control.create_content(20, 3)
    picker.preview_control.mouse_handler(
        MouseEvent(Point(0, 1), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
    )

    # Focus changes invalidate the application after mouse-down. prompt_toolkit
    # asks controls for their preferred height before the next mouse packet.
    picker.preview_control.create_content(20, None)

    assert picker.preview_control.viewport == (20, 3)
    assert picker.preview_control.dragging


def test_preview_mouse_drag_selects_and_copies() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    copied: list[str] = []
    picker = ResumePicker(
        [row("1", "one", turns=(ResumePickerTurn("agent", "alpha beta"),))],
        app,
        selection_copy_callback=copied.append,
    )
    picker.preview_control.create_content(20, 3)
    down = MouseEvent(Point(0, 1), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
    move = MouseEvent(Point(4, 1), MouseEventType.MOUSE_MOVE, MouseButton.LEFT, frozenset())
    up = MouseEvent(Point(4, 1), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset())

    picker.preview_control.mouse_handler(down)
    picker.preview_control.mouse_handler(move)
    picker.preview_control.mouse_handler(up)

    assert copied
    assert app.clipboard.set_text.call_args.args[0] == copied[0]
    assert picker._preview_model(20).selected_text() == ""


@pytest.mark.asyncio
async def test_real_prompt_toolkit_preview_drag_survives_redraw_between_packets(
    monkeypatch,
) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import MutableRecordingOutput, TUIHarness

    session = SimpleNamespace(
        id="session-1",
        name="one",
        model="m",
        agent="A",
        working_dir=str(Path.cwd()),
        started_at=1,
        last_active=2,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [session])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "load_turns",
        classmethod(
            lambda cls, value, limit=12: [SimpleNamespace(role="agent", content="alpha beta gamma")]
        ),
    )

    async with TUIHarness(output=MutableRecordingOutput(80, 24), full_screen=True) as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        picker = harness.app._resume_picker
        await harness.wait_for(lambda: picker.preview_control.viewport[1] >= 2)
        width = picker.preview_control.viewport[0]
        # The preview renders off-thread; wait for the model before dragging so
        # the packets can never race the async preview preparation.
        await harness.wait_for(lambda: picker._preview_model(width) is not None)

        # Locate the screen row that actually shows the message text: the
        # bottom-aligned preview can end with blank rows and pane origins
        # shift with layout changes (e.g. upstream separator rows), so SGR
        # rows must not be hardcoded.

        def rendered_message_row():
            # The session's LIST row also shows the newest agent message as a
            # preview column, so take the LAST screen row containing the text —
            # that is the conversation-preview pane's copy.
            screen = harness.app._app.renderer.last_rendered_screen
            if screen is None:
                return None
            found = None
            for row_index, line in screen.data_buffer.items():
                chars = "".join(line[column].char for column in sorted(line))
                if "alpha beta gamma" in chars:
                    found = row_index
            return found

        await harness.wait_for(lambda: rendered_message_row() is not None, timeout=5.0)
        # SGR coordinates are one-based. Send each packet separately so the
        # focus-changing mouse-down is followed by a real render measurement
        # before the drag and release packets arrive.
        down = rendered_message_row() + 1
        harness._pipe.send_text(f"\x1b[<0;3;{down}M")
        await harness.wait_for(lambda: picker.preview_control.dragging, timeout=5.0)
        harness._pipe.send_text(f"\x1b[<32;12;{down}M")
        await harness.wait_for(
            lambda: bool(picker._preview_model(width).selected_text()), timeout=5.0
        )
        harness._pipe.send_text(f"\x1b[<0;12;{down}m")
        await harness.wait_for(lambda: not picker.preview_control.dragging, timeout=5.0)

        selected = harness.app._app.clipboard.get_data().text
        assert selected
        assert picker._preview_model(width).selected_text() == ""
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
async def test_resume_f2_native_selection_cancels_edge_autoscroll() -> None:
    app = MagicMock()
    app.output.get_size.return_value = SimpleNamespace(columns=80, rows=24)
    picker = ResumePicker(
        [row("1", "one", turns=(ResumePickerTurn("agent", "\n".join(map(str, range(20)))),))],
        app,
    )
    picker.preview_control.create_content(20, 3)
    scrolls: list[tuple[str, int]] = []
    picker.mouse_scroll = lambda pane, delta: scrolls.append((pane, delta))  # type: ignore[method-assign]
    picker.preview_control.mouse_handler(
        MouseEvent(Point(0, 1), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
    )
    picker.preview_control.mouse_handler(
        MouseEvent(Point(3, 2), MouseEventType.MOUSE_MOVE, MouseButton.LEFT, frozenset())
    )

    picker.toggle_native_selection()
    assert picker.mouse_support is False
    await asyncio.sleep(0.4)
    assert scrolls == []


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
        "load_turns",
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
        await harness.press("tab")
        await harness.wait_for(lambda: picker.active_control == "preview")
        await harness.press("s-tab")
        await harness.wait_for(lambda: picker.active_control == "list")
        picker.buffer.text = ""
        await harness.wait_for(lambda: picker.model.query == "")
        await harness.press("c-o")
        await harness.wait_for(lambda: picker.option_cursor == "filter")
        await harness.type_keys("x")
        await harness.press("option-backspace")
        await asyncio.sleep(0)
        assert picker.model.query == ""
        await harness.press("down")
        await harness.wait_for(lambda: picker.model.state_filter == "attached")
        assert picker.model.matches == []
        await harness.press("down")
        await harness.wait_for(lambda: picker.model.state_filter == "all")
        assert {match.row.id for match in picker.model.matches} == {"session-1", "session-2"}
        await harness.press("right")
        await harness.wait_for(lambda: picker.option_cursor == "sort")
        await harness.type_keys(" ")
        await harness.wait_for(lambda: not picker.model.sort_updated)
        await harness.press("enter")
        await harness.wait_for(lambda: picker.option_cursor is None)
        assert [match.row.id for match in picker.model.matches] == ["session-2", "session-1"]
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
async def test_ctrl_c_cancels_picker_while_options_are_open(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    sessions = [
        SimpleNamespace(
            id="session-1",
            name="one",
            model="m",
            agent="A",
            working_dir=str(Path.cwd()),
            started_at=1,
            last_active=2,
            turn_count=1,
        )
    ]
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: sessions)
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "load_turns",
        classmethod(lambda cls, value, limit=12: []),
    )
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        await harness.press("c-o")
        await harness.wait_for(lambda: harness.app._resume_picker.option_cursor == "filter")
        await harness.press("c-c")
        assert await asyncio.wait_for(opened, 1) is None
        assert harness.app._resume_picker is None


@pytest.mark.asyncio
async def test_option_backspace_edits_search_without_closing_picker(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    session = SimpleNamespace(
        id="session-1",
        name="alpha",
        model="m",
        agent="A",
        working_dir=str(Path.cwd()),
        started_at=1,
        last_active=2,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [session])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(sm.SessionManager, "load_turns", classmethod(lambda cls, value: []))

    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        await harness.type_keys("alpha beta")
        await harness.press("option-backspace")
        await harness.wait_for(lambda: harness.app._resume_picker.buffer.text == "alpha ")
        assert harness.app._resume_picker is not None
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None


@pytest.mark.asyncio
async def test_enter_resumes_selected_session_from_preview(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import TUIHarness

    session = SimpleNamespace(
        id="session-1",
        name="Session 1",
        model="m",
        agent="A",
        working_dir=str(Path.cwd()),
        started_at=1,
        last_active=2,
        turn_count=1,
    )
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: [session])
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "load_turns",
        classmethod(lambda cls, value: [SimpleNamespace(role="agent", content="answer")]),
    )
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)
        await harness.press("tab")
        await harness.wait_for(
            lambda: (
                harness.app._resume_picker is not None
                and harness.app._resume_picker.active_control == "preview"
            )
        )
        await harness.press("enter")
        assert await asyncio.wait_for(opened, 1) == "session-1"


@pytest.mark.asyncio
async def test_full_application_selection_marker_moves_down_the_visible_list(monkeypatch) -> None:
    from nooa_cli.tui import session_manager as sm

    from .tui_app_harness import MutableRecordingOutput, TUIHarness

    sessions = [
        SimpleNamespace(
            id=f"session-{index}",
            name=f"Session {index}",
            model="m",
            agent="A",
            working_dir=str(Path.cwd()),
            started_at=index,
            last_active=100 - index,
            turn_count=1,
        )
        for index in range(5)
    ]
    monkeypatch.setattr(
        sm.SessionManager, "list_sessions", classmethod(lambda cls, limit=None: sessions)
    )
    monkeypatch.setattr(sm.SessionManager, "is_active", classmethod(lambda cls, value: False))
    monkeypatch.setattr(
        sm.SessionManager,
        "load_turns",
        classmethod(
            lambda cls, value: [
                SimpleNamespace(role="user", content=f"question for {value}"),
                SimpleNamespace(role="agent", content=f"answer for {value}"),
            ]
        ),
    )
    output = MutableRecordingOutput(columns=80, rows=24)
    async with TUIHarness(output=output, full_screen=True) as harness:
        opened = asyncio.create_task(harness.app.open_session_resume_dialog())
        await harness.wait_for(lambda: harness.app._resume_picker is not None)

        def marker_row() -> int:
            screen = harness.app._app.renderer.last_rendered_screen
            return next(
                y
                for y in range(24)
                if "❯" in "".join(screen.data_buffer[y][x].char for x in range(80))
                and "✓" in "".join(screen.data_buffer[y][x].char for x in range(80))
            )

        await harness.wait_for(lambda: harness.app._app.renderer.last_rendered_screen is not None)
        await harness.wait_for(lambda: harness.app._resume_picker.active_control == "list")
        before = marker_row()
        await harness.press("down")
        await harness.wait_for(lambda: harness.app._resume_picker.model.selected == 1)
        await harness.wait_for(lambda: marker_row() > before)
        assert marker_row() == before + 1
        assert harness.app._resume_picker.model.list_offset == 0
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
        "load_turns",
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
        if usable:
            await harness.wait_for(
                lambda: any(
                    session_id == "session-19"
                    for session_id, _width in harness.app._resume_picker._preview_models
                )
            )
        harness.app._app.invalidate()

        def rendered_lines() -> list[str]:
            screen = harness.app._app.renderer.last_rendered_screen
            if screen is None:
                return []
            return [
                "".join(screen.data_buffer[y][x].char for x in range(width)).rstrip()
                for y in range(height)
            ]

        if usable:
            # The preview renders off-thread; wait until the prepared replay is
            # actually on screen so a stale "Preparing…" frame can't be read.
            await harness.wait_for(
                lambda: any("preview for session-19" in line for line in rendered_lines()),
                timeout=5.0,
            )
        else:
            await harness.wait_for(lambda: any(line.strip() for line in rendered_lines()))
        visible = rendered_lines()
        if usable:
            joined = "\n".join(visible)
            assert "20 sessions" in joined
            assert "Search" in joined
            assert "Filter:" in joined and "✓" in joined
            assert "Sort:" in joined
            assert "Conversation preview" in joined
            assert "Ctrl-O" in joined and "options" in joined
            assert joined.count("─") >= width * 2
            assert "preview for session-19" in joined
            assert "❯" in joined and "y ago" in joined
            assert any(("Enter" in line or "↵" in line) and "Esc" in line for line in visible)
            assert all(len(line) <= width for line in visible)
        else:
            assert any("Terminal too small" in line for line in visible)
        await harness.press("escape")
        assert await asyncio.wait_for(opened, 1) is None
