# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Resume-style full-screen browser shell."""

import asyncio
from unittest.mock import MagicMock

import pytest
from nooa_cli.tui.explorer_base import ExplorerConfig, ExplorerModel, ExplorerView
from nooa_cli.tui.fullscreen_browser import ExplorerBrowser
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output import DummyOutput


def _browser() -> ExplorerBrowser:
    rows = [
        MagicMock(search_text="alpha", title="Alpha"),
        MagicMock(search_text="beta", title="Beta"),
    ]
    view = ExplorerView(ExplorerModel(rows), ExplorerConfig(title="Todo Explorer"))
    view.configure_row_options(
        filters=(
            ("all", "All", lambda _row: True),
            ("alpha", "Alpha", lambda row: row.title == "Alpha"),
        ),
        sorts=(
            ("new", "Newest", lambda row: row.title, True),
            ("old", "Oldest", lambda row: row.title, False),
        ),
    )
    output = DummyOutput()
    output.get_size = lambda: Size(rows=24, columns=100)  # type: ignore[method-assign]
    app = Application(output=output, full_screen=True)
    browser = ExplorerBrowser(view, app)
    app.layout = app.layout.__class__(browser.container)
    return browser


def test_explorer_browser_uses_resume_layout_and_search() -> None:
    browser = _browser()
    browser.on_open()
    assert browser.container is not None
    assert browser.explorer_list_height.max > 5
    assert browser.explorer_list_height.weight == 1
    assert "Type search" in browser._help_text()
    browser.handle_key("text", "alpha")
    assert browser.model.query == "alpha"
    assert len(browser.model.matches) == 1
    assert browser.handle_key("escape") == "close"


def test_explorer_browser_options_and_two_pane_focus() -> None:
    browser = _browser()
    browser.handle_key("options")
    assert browser.option_cursor == 0
    browser.handle_key("text", "x")
    browser.handle_key("quit")
    assert browser.buffer.text == ""
    browser.handle_key("down")
    assert len(browser.model.matches) == 1
    browser.handle_key("right")
    browser.handle_key("space")
    assert browser.view.options[1].value == "old"
    browser.handle_key("tab")
    assert browser.option_cursor is None
    assert browser.active_control == "preview"
    browser.handle_key("s-tab")
    assert browser.active_control == "list"

    browser.handle_key("options")
    browser.handle_key("enter")
    assert browser.option_cursor is None


def test_explorer_browser_mouse_wheel_scrolls_list_without_moving_selection() -> None:
    browser = _browser()
    browser.model.rows.extend(
        MagicMock(search_text=f"item {index}", title=f"Item {index}") for index in range(10)
    )
    browser.model.set_query("")
    browser.list_control.viewport = (80, 4)

    browser.mouse_scroll("list", 3)

    assert browser.model.cursor == 0
    assert browser.list_offset == 3
    visible = browser._visible(4)
    assert visible[0][0] == 3
    assert browser.list_offset == 3

    browser.navigate_vertical(1)
    assert browser.model.cursor == 1
    browser._visible(4)
    assert browser.list_offset == 1


def test_explorer_browser_search_cursor_uses_left_and_right() -> None:
    browser = _browser()
    browser.buffer.text = "ab"
    browser.buffer.cursor_position = 2
    browser.handle_key("left")
    assert browser.buffer.cursor_position == 1
    browser.handle_key("right")
    assert browser.buffer.cursor_position == 2


@pytest.mark.asyncio
async def test_todo_explorer_is_wrapped_in_shared_fullscreen_browser() -> None:
    from nooa_cli.tui.todo_explorer import TodoExplorerView

    from .tui_app_harness import TUIHarness

    row = MagicMock(
        id="abc12345",
        title="Ship reusable browser",
        status="open",
        deps=[],
        created_at="2026-08-27 18:00",
        notes="Use the Resume Picker shell.",
        comments=[],
        search_text="Ship reusable browser",
    )
    view = TodoExplorerView([row])
    async with TUIHarness() as harness:
        opened = asyncio.create_task(harness.app.open_subview(view))
        await harness.wait_for(lambda: isinstance(harness.app.active_subview, ExplorerBrowser))
        browser = harness.app.active_subview
        assert browser.view is view
        assert browser.container is not None
        assert "Type search" in browser._help_text()
        assert len(browser.option_controls) == 2

        await harness.type_keys("Ship")
        await harness.wait_for(lambda: browser.model.query == "Ship")
        await harness.press("c-o")
        await harness.wait_for(lambda: browser.option_cursor == 0)
        await harness.press("escape")
        await harness.wait_for(lambda: browser.option_cursor is None)
        await harness.press("escape")
        await asyncio.wait_for(opened, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "job"])
async def test_remaining_explorers_are_wrapped_in_shared_fullscreen_browser(kind: str) -> None:
    from nooa_cli.tui.job_explorer import JobExplorerView, JobSnapshot
    from nooa_cli.tui.memory_explorer import MemoryExplorerRow, MemoryExplorerView

    from .tui_app_harness import TUIHarness

    if kind == "memory":
        view = MemoryExplorerView(
            [
                MemoryExplorerRow(
                    id="memory-12345678",
                    type="info",
                    status=None,
                    title="Shared browser memory",
                    content="Rendered through the reusable shell.",
                    owner="tester",
                    importance="MEDIUM",
                    tags=[],
                    created_at=1.0,
                    last_accessed_at=1.0,
                    usage={
                        "fetches": 0,
                        "last_ts": None,
                        "recalled": 0,
                        "searched": 0,
                        "injected": 0,
                        "reinforced": 0,
                        "deref": 0,
                        "last_channel": None,
                        "last_session_ref": None,
                        "mean_rank": None,
                        "mean_score": None,
                        "injected_never_used": False,
                        "prune_eta": None,
                        "retention": "keep",
                        "strength": 1.0,
                    },
                    reference_lines=[],
                    edges=[],
                    search_text="Shared browser memory",
                )
            ],
            forget=lambda _memory_id: None,
            mark_done=lambda _memory_id: None,
        )
    else:
        view = JobExplorerView([JobSnapshot("job-1", "worker", "running", 0, ())])

    async with TUIHarness() as harness:
        if kind == "memory":
            from nooa_cli.tui.host_services import TUIHostServices

            harness.app._host_services = TUIHostServices(open_memory_view=lambda: view)
            opened = asyncio.create_task(harness.app.open_memory_explorer())
        else:
            opened = asyncio.create_task(harness.app.open_job_explorer())
        await harness.wait_for(lambda: isinstance(harness.app.active_subview, ExplorerBrowser))
        browser = harness.app.active_subview
        if kind == "memory":
            assert browser.view is view
        else:
            assert isinstance(browser.view, JobExplorerView)
        assert browser.title == ("Memory Explorer" if kind == "memory" else "Job Explorer")
        assert browser.container is not None
        assert len(browser.option_controls) == 2

        await harness.press("c-c")
        await asyncio.wait_for(opened, timeout=1)


def test_explorer_browser_list_text_replaces_newlines_in_summary() -> None:
    """Newlines in row summaries are replaced with spaces to prevent rendering corruption."""

    class _MultiLineRow:
        search_text = "Multi-line Item\nsecond line\rthird line"
        title = "Multi-line Item"
        tag = "1"

    browser = _browser()
    browser.model.rows.clear()
    browser.model.rows.append(_MultiLineRow())
    browser.view.format_row = lambda row, w: f"{row.tag} {row.search_text}"
    browser.model.set_query("")
    browser.list_control.viewport = (80, 4)

    fragments = browser.list_text(80, 4)
    joined = "".join(text for _style, text in fragments)

    assert "\n" not in joined
    assert "\r" not in joined
    assert "Multi-line Item second line third line" in joined


def test_explorer_browser_has_separator_under_title() -> None:
    """Separators follow the title, list header, and preview area."""
    from prompt_toolkit.layout import FloatContainer, HSplit, VSplit, Window

    browser = _browser()
    main = browser.container._get_container()
    assert isinstance(main, FloatContainer)
    body = main.content
    assert isinstance(body, HSplit)

    assert body.children[0].content is browser.title_control
    assert isinstance(body.children[1], Window)
    assert body.children[1].char == "─"

    list_area = body.children[3]
    assert isinstance(list_area, VSplit)
    list_body = list_area.children[1]
    assert isinstance(list_body, HSplit)
    assert list_body.children[1].content is browser.list_header_control
    assert isinstance(list_body.children[2], Window)
    assert list_body.children[2].char == "─"

    preview_area = body.children[5]
    assert isinstance(preview_area, VSplit)
    assert isinstance(body.children[6], Window)
    assert body.children[6].char == "─"


def test_explorer_browser_handle_key_scroll_down_scrolls_list() -> None:
    """scroll_down action routes to the active pane via handle_key."""
    browser = _browser()
    browser.model.rows.extend(
        MagicMock(search_text=f"item {i}", title=f"Item {i}") for i in range(10)
    )
    browser.model.set_query("")
    browser.list_control.viewport = (80, 4)
    browser.active_control = "list"

    browser.handle_key("scroll_down")

    assert browser.list_offset == 3
    assert browser.model.cursor == 0  # selection doesn't move


def test_explorer_browser_handle_key_scroll_up_detail() -> None:
    """scroll_up in the detail pane scrolls detail, not the list."""
    browser = _browser()
    browser.model.rows.extend(
        MagicMock(search_text=f"item {i}", title=f"Item {i}") for i in range(3)
    )
    browser.model.set_query("")
    browser.list_control.viewport = (80, 4)
    browser.preview_control.viewport = (80, 4)
    browser.active_control = "preview"
    browser.model._last_detail_line_count = 20
    browser.model.detail_offset = 10

    browser.handle_key("scroll_up")

    assert browser.model.detail_offset == 7


def test_explorer_browser_navigate_vertical_jumps_search_matches() -> None:
    """When search is active, up/down jumps between matches instead of moving list selection."""

    class _MatchRow:
        search_text = "alpha beta alpha"
        title = "Match Row"
        tag = "1"

    browser = _browser()
    browser.model.rows.clear()
    browser.model.rows.append(_MatchRow())
    browser.model.set_query("")
    browser.list_control.viewport = (80, 4)
    browser.preview_control.viewport = (80, 4)
    browser.active_control = "list"

    # Simulate a search query being typed
    browser.buffer.text = "alpha"
    browser._query_changed()

    assert browser.model.search_active is True

    # Populate match lines (as detail_lines would)
    browser.model._last_detail_match_lines = [0, 2]
    browser.model._last_detail_visible_lines = 4

    # Down should jump to match, not move cursor
    browser.navigate_vertical(1)
    assert browser.model.search_line_cursor == 1
    assert browser.model.cursor == 0  # selection stays


def test_explorer_browser_search_match_boundary_moves_to_next_row() -> None:
    browser = _browser()
    browser.model.rows.append(MagicMock(search_text="gamma", title="Gamma"))
    browser.model.set_query("")
    browser.model.search_active = True
    browser.model._last_detail_match_lines = [1, 3]
    browser.model.search_line_cursor = 1

    browser.navigate_vertical(1)

    assert browser.model.cursor == 1
    assert browser.model._last_detail_match_lines == []
    assert browser.model.search_line_cursor == 0

    # A second move before rendering must not reuse the previous row's matches.
    browser.navigate_vertical(1)
    assert browser.model.cursor == 2
