# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ownership tests for the alternate-screen transcript renderer."""

from __future__ import annotations

import asyncio
import io

import pytest
from nooa_cli.tui.config import DisplayMode
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from .tui_app_harness import TUIHarness


def _make_fullscreen_app():
    from nooa_cli.tui.tui_application import TUIApplication

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        return TUIApplication(display_mode=DisplayMode.FULLSCREEN)


def test_resize_application_prompt_toolkit_private_api_contract() -> None:
    """Fail clearly if a prompt-toolkit upgrade changes private resize hooks."""
    import inspect

    from nooa_cli.tui.tui_application import _ResizeAwareApplication
    from prompt_toolkit.application import Application

    on_resize = inspect.signature(Application._on_resize)
    assert tuple(on_resize.parameters) == ("self",)

    redraw = inspect.signature(Application._redraw)
    assert tuple(redraw.parameters) == ("self", "render_as_done")
    assert redraw.parameters["render_as_done"].default is False

    request_cursor = inspect.signature(Application._request_absolute_cursor_position)
    assert tuple(request_cursor.parameters) == ("self",)

    assert tuple(inspect.signature(_ResizeAwareApplication._on_resize).parameters) == ("self",)
    assert tuple(inspect.signature(_ResizeAwareApplication._redraw).parameters) == (
        "self",
        "render_as_done",
    )

    app = _make_fullscreen_app()
    assert isinstance(app._app._running_in_terminal, bool)


def test_fullscreen_shell_owns_alternate_screen_and_transcript_window() -> None:
    app = _make_fullscreen_app()

    assert app.display_mode is DisplayMode.FULLSCREEN
    assert app._app.full_screen is True
    assert app.full_screen is False  # legacy destructive native replay flag
    assert app._output_window is not None


def test_fullscreen_copy_and_return_actions_share_status_region() -> None:
    from prompt_toolkit.layout import VSplit

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._show_transient_status("Copied 5 characters", style="class:return-to-tail")
    app._scroll_fullscreen_transcript(-4)

    status_container = app._status_region_container
    assert status_container is not None
    status_region = status_container.content
    assert isinstance(status_region, VSplit)
    assert app._transient_status_container in status_region.children
    assert app._return_to_tail_container in status_region.children
    assert bool(status_container.filter())
    assert bool(app._transient_status_container.filter())
    assert bool(app._return_to_tail_container.filter())


def test_fullscreen_agent_message_notice_appears_only_while_scrolled_up() -> None:
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._scroll_fullscreen_transcript(-4)

    app.emit_block("tool activity\n")
    text = fragment_list_to_text(to_formatted_text(app._return_to_tail_control.text()))
    assert "New agent message" not in text
    return_position = text.index("Return to bottom")

    app.emit_block("agent reply\n", agent_message=True)
    fragments = to_formatted_text(app._return_to_tail_control.text())
    text = fragment_list_to_text(fragments)
    assert text.index("New agent message") < text.index("Return to bottom")
    assert text.index("Return to bottom") == return_position
    notice_style = next(style for style, value, *_ in fragments if "New agent message" in value)
    return_style = next(style for style, value, *_ in fragments if "Return to bottom" in value)
    assert notice_style == return_style
    assert app._has_unseen_agent_message

    app._jump_fullscreen_to_tail()
    assert not app._has_unseen_agent_message
    assert not bool(app._return_to_tail_container.filter())


def test_fullscreen_clear_removes_unseen_agent_message_notice() -> None:
    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._scroll_fullscreen_transcript(-4)
    app.emit_block("agent reply\n", agent_message=True)
    assert app._has_unseen_agent_message

    app.clear_transcript()

    assert not app._has_unseen_agent_message
    assert app._fullscreen_transcript.viewport.follows_tail


def test_fullscreen_agent_message_at_tail_does_not_create_notice() -> None:
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("agent reply\n", agent_message=True)

    assert app._fullscreen_transcript.viewport.follows_tail
    assert not app._has_unseen_agent_message
    assert "New agent message" not in fragment_list_to_text(
        to_formatted_text(app._return_to_tail_control.text())
    )


@pytest.mark.asyncio
async def test_fullscreen_input_composer_is_capped_at_ten_rows() -> None:
    from prompt_toolkit.application.current import set_app

    app = _make_fullscreen_app()
    app.input_buffer.text = "\n".join(f"line {index}" for index in range(20))

    with set_app(app._app):
        dimension = app._input_window.preferred_height(80, 100)

    assert dimension.min == 1
    assert dimension.max == 10
    assert dimension.preferred == 10


def test_fullscreen_status_removes_visual_transcript_gap() -> None:
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("answer\n")

    assert not bool(app._status_region_container.filter())
    assert (
        fragment_list_to_text(
            to_formatted_text(app._fullscreen_transcript.formatted_text(width=20, height=2))
        )
        == "answer\n"
    )

    app._command_status_text = "thinking..."
    app._before_render(app._app)

    assert bool(app._status_region_container.filter())
    assert app._fullscreen_transcript.text == "answer\n"
    assert (
        fragment_list_to_text(
            to_formatted_text(app._fullscreen_transcript.formatted_text(width=20, height=1))
        )
        == "answer"
    )


def test_native_hyperlink_marker_vocabulary_is_bounded() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append(
        " ".join(
            f"\x1b]8;;https://example.test/{index}\x1b\\x\x1b]8;;\x1b\\" for index in range(64)
        )
    )

    def marker_classes(render_counter: int) -> set[str]:
        return {
            token
            for style, _text, *_ in model.formatted_text(
                width=4_096,
                height=1,
                render_counter=render_counter,
            )
            for token in style.split()
            if token.startswith("class:native-hyperlink-")
        }

    assert marker_classes(0) == {"class:native-hyperlink-0"}
    assert marker_classes(1) == {"class:native-hyperlink-1"}
    model._formatted_cache.clear()
    assert marker_classes(2) == {"class:native-hyperlink-0"}


def test_fullscreen_status_occupancy_is_sampled_once_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_fullscreen_app()
    app.emit_block("answer\n")
    calls = 0

    def status_rows(*, include_transient: bool = True):
        del include_transient
        nonlocal calls
        calls += 1
        return [[("class:status", "busy")]]

    monkeypatch.setattr(app, "_status_rows", status_rows)
    app._before_render(app._app)

    assert calls == 1
    app._fullscreen_transcript.formatted_text(width=20, height=2)
    app._fullscreen_transcript.cursor_position(width=20, height=2)
    assert bool(app._status_region_container.filter())
    assert calls == 1


def test_fullscreen_bootstrap_output_only_mutates_renderer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_fullscreen_app()
    stdout = io.StringIO()
    real_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.__stdout__", real_stdout)

    app.emit_block("hello \x1b[31mred\x1b[0m\n")

    assert app.output_buffer.text == ""
    assert app._fullscreen_transcript.text == "hello red\n"
    assert stdout.getvalue() == ""
    assert real_stdout.getvalue() == ""


@pytest.mark.asyncio
async def test_fullscreen_live_output_never_suspends_renderer_or_writes_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def forbidden_run_in_terminal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("fullscreen transcript must remain application-owned")

    stdout = io.StringIO()
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", forbidden_run_in_terminal)
    monkeypatch.setattr("sys.__stdout__", stdout)

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("first\n")
        await asyncio.sleep(0)
        app.clear_transcript()
        app.emit_block("second\n")
        if app._block_queue is not None:
            await app._block_queue.join()

        assert app.output_buffer.text == ""
        assert app._fullscreen_transcript.text == "second\n"

    assert calls == 0
    assert stdout.getvalue() == ""


def test_fullscreen_drops_accepted_off_thread_callback_after_queue_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback accepted before teardown cannot write after screen restoration."""
    app = _make_fullscreen_app()
    callbacks: list[tuple[object, tuple[object, ...]]] = []

    class DelayedLoop:
        def call_soon_threadsafe(self, callback, *args) -> None:
            callbacks.append((callback, args))

    stdout = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", stdout)
    app._loop = DelayedLoop()  # type: ignore[assignment]
    app._block_queue = asyncio.Queue()

    app.emit_block("late fullscreen output\n")
    assert len(callbacks) == 1

    app._block_queue = None
    callback, args = callbacks.pop()
    callback(*args)  # type: ignore[operator]

    assert "late fullscreen output" not in app.output_buffer.text
    assert "late fullscreen output" not in app._fullscreen_transcript.text
    assert stdout.getvalue() == ""


@pytest.mark.asyncio
async def test_fullscreen_resize_reflows_without_destructive_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .tui_app_harness import MutableRecordingOutput

    async def forbidden_run_in_terminal(*_args, **_kwargs):
        raise AssertionError("fullscreen resize must remain renderer-owned")

    stdout = io.StringIO()
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", forbidden_run_in_terminal)
    monkeypatch.setattr("sys.__stdout__", stdout)
    output = MutableRecordingOutput(columns=8, rows=20)

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("abcdefghijklmnop\n")
        assert app._fullscreen_transcript.text == "abcdefghijklmnop\n"
        output.events.clear()
        render_count = app._app.render_counter
        await harness.resize_from_terminal(5, 20)
        assert app._app.render_counter == render_count
        await harness.wait_for(lambda: app._fullscreen_invalidate_count == 1)
        assert app._fullscreen_transcript.text == "abcdefghijklmnop\n"
        assert app._app.render_counter == render_count + 1
        flushes = [index for index, event in enumerate(output.events) if event == ("flush",)]
        renderer_writes = [
            index for index, event in enumerate(output.events) if event[0] == "write"
        ]
        assert len(flushes) == 1
        assert renderer_writes
        assert max(renderer_writes) < flushes[0]

    assert stdout.getvalue() == ""


def test_resize_deferred_flush_restores_output_and_flushes_on_callback_error() -> None:
    """A failed PTK mutation must not strand buffered terminal output."""
    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput()
    with create_app_session(input=DummyInput(), output=output):
        from nooa_cli.tui.tui_application import TUIApplication

        app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

    original_flush = output.flush
    output.events.clear()

    def failing_callback() -> None:
        output.write_raw("partial mutation")
        output.flush()
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        app._app._run_with_deferred_flush(failing_callback)

    assert output.flush == original_flush
    assert output.events[-1] == ("flush",)
    assert output.events.count(("flush",)) == 1


@pytest.mark.asyncio
async def test_fullscreen_resize_while_subview_open_rebuilds_on_close() -> None:
    """A modal view must defer, not consume, a semantic transcript resize."""
    from .tui_app_harness import MutableRecordingOutput

    class View:
        mouse_support = True

        def on_open(self) -> None:
            pass

        def on_close(self) -> None:
            pass

        def render(self, _width: int, _height: int) -> str:
            return "temporary view"

        def handle_key(self, key: str, _value: str | None = None) -> str:
            return "close" if key == "quit" else "ignored"

    output = MutableRecordingOutput(columns=10, rows=20)
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        app = harness.app
        assert app is not None
        widths: list[int] = []

        def semantic_projection() -> str:
            width = app.transcript_columns()
            widths.append(width)
            return "wide transcript\n" if width >= 8 else "narrow\ntranscript\n"

        app.emit_block(semantic_projection(), replay=semantic_projection)
        opened = asyncio.create_task(app.open_subview(View()))  # type: ignore[arg-type]
        await harness.wait_for(lambda: app.active_subview is not None)

        output.set_size(6, 20)
        app._app._on_resize()
        await asyncio.sleep(0)

        assert app._fullscreen_transcript.text == "wide transcript\n"
        assert app._resize_reflow.has_pending_replay is True
        assert app._fullscreen_invalidate_count == 0

        await harness.press("q")
        await asyncio.wait_for(opened, timeout=1)

        assert app._fullscreen_transcript.text == "narrow\ntranscript\n"
        assert widths == [9, 5]
        assert app._resize_reflow.replayed_width == 6
        assert app._resize_reflow.has_pending_replay is False
        assert app._fullscreen_invalidate_count == 1


@pytest.mark.asyncio
async def test_fullscreen_row_only_resize_publishes_one_settled_frame() -> None:
    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=80, rows=40)
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        app = harness.app
        assert app is not None
        output.events.clear()
        render_count = app._app.render_counter

        output.set_size(80, 30)
        app._app._on_resize()

        assert app._app.render_counter == render_count
        await harness.wait_for(lambda: app._fullscreen_invalidate_count == 1)
        assert app._app.render_counter == render_count + 1
        assert sum(event == ("flush",) for event in output.events) == 1
        assert app._resize_reflow.has_pending_replay is False


@pytest.mark.asyncio
async def test_fullscreen_duplicate_sigwinch_after_settle_has_no_delayed_repaint() -> None:
    """Re-exposing a settled tmux pane may repaint once, never again later."""
    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=80, rows=40)
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        app = harness.app
        assert app is not None

        output.set_size(60, 30)
        app._app._on_resize()
        await harness.wait_for(lambda: app._fullscreen_invalidate_count == 1)
        settled_render_count = app._app.render_counter
        output.events.clear()

        app._app._on_resize()
        assert app._app.render_counter == settled_render_count + 1
        assert app._fullscreen_rebuild_timer is None
        await asyncio.sleep(0.1)

        assert app._app.render_counter == settled_render_count + 1
        assert app._fullscreen_invalidate_count == 1
        assert app._fullscreen_rebuild_timer is None
        assert sum(event == ("flush",) for event in output.events) == 1


@pytest.mark.asyncio
async def test_fullscreen_unreadable_size_uses_prompt_toolkit_resize() -> None:
    """A transient geometry read failure must not suppress PTK's safe fallback."""
    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=80, rows=40)
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        app = harness.app
        assert app is not None
        render_count = app._app.render_counter
        original_read_terminal_size = app._read_terminal_size
        app._read_terminal_size = lambda: None  # type: ignore[method-assign]
        try:
            app._app._on_resize()
        finally:
            app._read_terminal_size = original_read_terminal_size  # type: ignore[method-assign]

        assert app._app.render_counter == render_count + 1
        assert app._fullscreen_rebuild_timer is None
        assert app._resize_redraw_deferred is False


@pytest.mark.asyncio
async def test_fullscreen_transient_sigwinches_publish_only_final_frame() -> None:
    """A live resize burst must not expose prompt_toolkit's intermediate sizes."""
    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=80, rows=40)
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("stable transcript\n")
        output.events.clear()
        render_count = app._app.render_counter

        output.set_size(70, 35)
        app._app._on_resize()
        output.set_size(60, 30)
        app._app._on_resize()
        app._app._on_resize()  # Duplicate final SIGWINCH from tmux exposure.

        assert app._app.render_counter == render_count
        await harness.wait_for(lambda: app._fullscreen_invalidate_count == 1)
        assert app._resize_reflow.observed_size == (60, 30)
        assert app._app.render_counter == render_count + 1
        assert sum(event == ("flush",) for event in output.events) == 1
        assert app._fullscreen_rebuild_timer is None


def test_fullscreen_theme_refresh_recolors_retained_semantic_scrollback() -> None:
    from nooa_cli.tui import theme
    from nooa_cli.tui.session import Session
    from rich.text import Text

    original_theme = theme.get_theme()
    try:
        theme.set_theme("mocha")
        app = _make_fullscreen_app()
        session = Session.__new__(Session)
        session._app = app
        session._emit_text(Text("hello", style="agent.response"))
        before = app._transcript_blocks[0].fullscreen_rendered
        assert before is not None

        theme.set_theme("latte")
        app.refresh_style()
        after = app._transcript_blocks[0].fullscreen_rendered

        assert after is not None
        assert after != before
        assert app._transcript_blocks[0].replay_cache
        assert app._fullscreen_transcript.text.count("hello") == 1
    finally:
        theme.set_theme(original_theme)


def test_fullscreen_resize_preserves_history_beyond_native_replay_tail() -> None:
    app = _make_fullscreen_app()
    for index in range(app._untagged_replay_tail + 7):
        app.emit_block(f"block {index}\n")

    before_resize = app._fullscreen_transcript.text
    assert before_resize.startswith("block 0\n")
    assert len(app._transcript_blocks) == app._untagged_replay_tail + 7

    app._rebuild_fullscreen_transcript()

    assert app._fullscreen_transcript.text == before_resize
    assert app.output_buffer.text == ""


def test_fullscreen_transcript_window_follows_appended_tail() -> None:
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\nthree\nfour\n")
    assert app._output_window is not None

    screen = Screen()
    app._output_window.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(xpos=0, ypos=0, width=10, height=2),
        parent_style="",
        erase_bg=False,
        z_index=None,
    )
    visible = ["".join(screen.data_buffer[y][x].char for x in range(10)).rstrip() for y in range(2)]

    assert visible == ["four", ""]
    # The model virtualizes exactly the visible rows, so Window itself never
    # scrolls over retained history.
    assert app._output_window.vertical_scroll == 0


def test_fullscreen_subview_projection_does_not_leak_unsupported_ansi() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    class UnsafeSubview:
        def render(self, _width: int, _height: int) -> str:
            return (
                "\x1b]8;;https://example.test\x1b\\label\x1b]8;;\x1b\\ "
                "\x1b[38:2::1:2:3mcolored\x1b[0m"
            )

    app = _make_fullscreen_app()
    app._active_subview = UnsafeSubview()  # type: ignore[assignment]
    rendered = "".join(fragment[1] for fragment in to_formatted_text(app._subview_control.text()))

    assert rendered == "label colored"
    assert "example.test" not in rendered
    assert "8;;" not in rendered
    assert "38:2" not in rendered


def test_fullscreen_sanitizes_terminal_commands_without_native_width_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_fullscreen_app()
    monkeypatch.setattr(app, "transcript_columns", lambda: 3)

    app.emit_block("abcdef\x1b[2J")

    assert app._fullscreen_transcript.text == r"abcdef\x1b[2J" + "\n"


def test_fullscreen_hyperlink_hit_testing_survives_projection_and_wrapping() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append(
        "prefix \x1b]8;id=42;https://example.test/docs\x1b\\linked text\x1b]8;;\x1b\\ suffix"
    )

    assert model.hyperlink_at(x=0, y=0, width=8, height=3) == "https://example.test/docs"
    assert model.hyperlink_at(x=1, y=1, width=8, height=3) == "https://example.test/docs"
    assert model.hyperlink_at(x=4, y=1, width=8, height=3) is None

    blank = FullscreenTranscriptModel()
    blank.append("\x1b]8;;https://example.test/docs\x1b\\foo\n\nbar\x1b]8;;\x1b\\")
    assert blank.hyperlink_at(x=0, y=1, width=20, height=3) is None
    assert blank.hyperlink_at(x=19, y=1, width=20, height=3) is None


def test_fullscreen_projection_wraps_prose_at_word_boundaries() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("alpha beta gamma-delta")

    rows = _projected_row_texts(model, 11)

    assert rows == ["alpha beta ", "gamma-delta"]
    assert "".join(rows) == model.text


def test_fullscreen_projection_keeps_an_exactly_full_row() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("alpha beta next")

    assert _projected_row_texts(model, 10) == ["alpha ", "beta next"]


def test_fullscreen_projection_folds_one_long_word_without_losing_text() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("supercalifragilistic")

    rows = _projected_row_texts(model, 6)

    assert rows == ["superc", "alifra", "gilist", "ic"]
    assert "".join(rows) == model.text


def test_fullscreen_hyperlink_hit_testing_matches_combining_grapheme_rendering() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    # The OSC-8 boundary lies inside one displayed grapheme (e + combining acute).
    model.append("e\x1b]8;;https://example.test\x1b\\́\x1b]8;;\x1b\\")

    assert model.hyperlink_at(x=0, y=0, width=20, height=1) == "https://example.test"


def test_fullscreen_projection_keeps_osc8_only_as_zero_width_metadata() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;https://example.test\x1b\\label\x1b]8;;\x1b\\\n")

    fragments = to_formatted_text(app._fullscreen_transcript.formatted_text())
    rendered = "".join(text for style, text, *_ in fragments if "[ZeroWidthEscape]" not in style)
    raw = "".join(text for style, text, *_ in fragments if "[ZeroWidthEscape]" in style)

    assert rendered == "label\n"
    assert raw == "\x1b]8;;https://example.test\x1b\\" * len("label")


def test_fullscreen_does_not_emit_native_metadata_for_unsafe_link_target() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;javascript:alert(1)\x1b\\label\x1b]8;;\x1b\\")

    fragments = to_formatted_text(app._fullscreen_transcript.formatted_text())

    assert "".join(text for _style, text, *_ in fragments) == "label\n"
    assert all("[ZeroWidthEscape]" not in style for style, _text, *_ in fragments)


def test_fullscreen_projection_drops_unsupported_colon_sgr_without_leaking_parameters() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    app = _make_fullscreen_app()
    app.emit_block("\x1b[38:2::1:2:3mcolored\x1b[0m\n")

    rendered = "".join(
        fragment[1] for fragment in to_formatted_text(app._fullscreen_transcript.formatted_text())
    )
    assert rendered == "colored\n"
    assert "38:2" not in rendered


@pytest.mark.asyncio
async def test_fullscreen_edit_returns_command_error_without_terminal_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from types import SimpleNamespace

    from nooa_cli.tui.commands import EditCommand
    from nooa_cli.tui.frontend import TerminalFrontend

    frontend = TerminalFrontend(SimpleNamespace(tui=SimpleNamespace(vi_mode=False)))
    frontend.bind_app(_make_fullscreen_app())
    command = EditCommand(frontend, SimpleNamespace(), object())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fullscreen /edit must not launch a process")

    monkeypatch.setattr("subprocess.run", forbidden)
    result = await command.execute([str(tmp_path / "example.py")])

    assert result.success is False
    assert len(result.outputs) == 1
    assert "native or native-replay" in result.outputs[0].content


def test_fullscreen_transcript_never_projects_past_physical_width() -> None:
    from nooa_cli.tui.tui_application import TUIApplication

    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=20, rows=12)
    with create_app_session(input=DummyInput(), output=output):
        app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

    control = app._output_window.content
    control._render_width = 80
    control._render_height = 7

    assert app._transcript_viewport_size() == (20, 7)


def test_fullscreen_width_change_reprojects_semantic_blocks_and_unwraps() -> None:
    from nooa_cli.tui.tui_application import TUIApplication

    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=8, rows=20)
    with create_app_session(input=DummyInput(), output=output):
        app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

    calls: list[int] = []

    def semantic_projection() -> str:
        width = app.transcript_columns()
        calls.append(width)
        return "abcdefgh\n" if width >= 8 else "abcd\nefgh\n"

    output.set_size(columns=5, rows=20)
    app._before_render(app._app)
    app.emit_block(semantic_projection(), replay=semantic_projection)
    assert app._fullscreen_transcript.text == "abcd\nefgh\n"

    output.set_size(columns=9, rows=20)
    app._before_render(app._app)

    assert app._fullscreen_transcript.text == "abcdefgh\n"
    assert app.output_buffer.text == ""
    assert calls == [4, 8]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [DisplayMode.NATIVE, DisplayMode.NATIVE_REPLAY])
async def test_native_modes_preserve_external_editor_handoff(
    mode: DisplayMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from nooa_cli.tui.frontend import TerminalFrontend

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        from nooa_cli.tui.tui_application import TUIApplication

        app = TUIApplication(display_mode=mode)
    frontend = TerminalFrontend(SimpleNamespace(tui=SimpleNamespace(vi_mode=False)))
    frontend.bind_app(app)
    handoffs = 0

    async def run_in_terminal(callback, *, in_executor):
        nonlocal handoffs
        handoffs += 1
        assert in_executor is True
        callback()

    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", run_in_terminal)
    monkeypatch.setattr("subprocess.run", lambda _args: None)

    assert await frontend.open_editor("example.py", "unchanged") == "unchanged"
    assert handoffs == 1


def test_fullscreen_session_markdown_keeps_copyable_source_unwrapped() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nooa_cli.tui.session import Session
    from nooa_cli.tui.tui_application import TUIApplication
    from rich.markdown import Markdown

    from .tui_app_harness import MutableRecordingOutput

    message = (
        "I left a brief closing note that the interaction design and implementation "
        "need a more fundamental rethink. The local diagnostic mouse fix remains "
        "uncommitted and was not pushed."
    )
    output = MutableRecordingOutput(columns=60, rows=20)
    with create_app_session(input=DummyInput(), output=output):
        app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

    session = Session.__new__(Session)
    session._app = app
    session._renderer = Mock()
    session.config = SimpleNamespace(tui=SimpleNamespace(full_screen=False))

    session._emit_text(Markdown(message))

    assert app._fullscreen_transcript.text == f"{message}\n"


@pytest.mark.parametrize("producer", ["rich", "user-bar"])
def test_fullscreen_session_production_rendering_reprojects_narrow_then_wide(
    producer: str,
) -> None:
    """Session retains semantics, not the initially width-bound ANSI bytes.

    This deliberately uses the production Rich renderer and production user-bar
    builder.  A narrow projection must wrap, and widening must remove that wrap.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nooa_cli.tui.session import Session
    from nooa_cli.tui.tui_application import TUIApplication
    from rich.text import Text

    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=6, rows=20)
    with create_app_session(input=DummyInput(), output=output):
        app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

    session = Session.__new__(Session)
    session._app = app
    session._renderer = Mock()
    session.config = SimpleNamespace(tui=SimpleNamespace(full_screen=False))
    # Establish the initial settled width so the later render observes a change.
    app._before_render(app._app)

    if producer == "rich":
        semantic = Text("abcdefgh", style="bold red")
        expected_narrow = session._render_to_ansi(semantic)
        session._emit_text(semantic)
    else:
        from nooa_cli.tui.session import _build_user_bar

        expected_narrow = _build_user_bar("abcdefgh", app, session._colors)
        session._on_user_message_ui("abcdefgh")

    narrow = app._fullscreen_transcript.text
    # Fullscreen has one renderer-owned history; the legacy native buffer is
    # deliberately not mirrored on every stream delta.
    assert app.output_buffer.text == ""
    assert expected_narrow != ""
    assert narrow.count("\n") >= 2

    output.set_size(columns=14, rows=20)
    app._before_render(app._app)

    if producer == "rich":
        expected_wide = session._render_to_ansi(semantic)
    else:
        expected_wide = _build_user_bar("abcdefgh", app, session._colors)
    wide = app._fullscreen_transcript.text
    assert app.output_buffer.text == ""
    assert expected_wide != expected_narrow
    expected_visible = (
        "abcdefgh\n" if producer == "rich" else "▔" * 13 + "\n ❯ abcdefgh  \n" + "▁" * 13 + "\n"
    )
    assert wide == expected_visible
    assert wide != narrow


@pytest.mark.parametrize("mode", [DisplayMode.NATIVE, DisplayMode.NATIVE_REPLAY])
def test_native_modes_preserve_transcript_source_exactly(
    mode: DisplayMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compatibility modes retain the exact producer bytes without reprojection."""
    from nooa_cli.tui.tui_application import TUIApplication

    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    source = "prefix \x1b[1;31mstyled\x1b[0m  tail\nsecond line\n"

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        app = TUIApplication(display_mode=mode)
    app.emit_block(source)

    assert len(app._transcript_blocks) == 1
    block = app._transcript_blocks[0]
    assert block.source == source
    assert block.replay is None
    # Native rendering preserves the producer payload exactly and applies only
    # the renderer's documented final reset when writing it to the terminal.
    assert stdout.getvalue() == app._render_transcript_source(source)
    assert stdout.getvalue() == source + "\x1b[0m"
    assert app.output_buffer.text == "prefix styled  tail\nsecond line\n"


def test_trailing_blank_line_anchors_are_monotonic_and_each_row_is_navigable() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("\n\n\n")
    rows = model._projection(2)

    assert [row.anchor.source_offset for row in rows] == [0, 1, 2, 3]
    model.jump_to_start(width=2)
    visited = [model.top_row(width=2, height=1)]
    for _ in range(3):
        model.scroll_visual_lines(1, width=2, height=1)
        visited.append(model.top_row(width=2, height=1))
    assert visited == [0, 1, 2, 3]


def test_fullscreen_streaming_does_not_rebuild_legacy_output_buffer() -> None:
    app = _make_fullscreen_app()

    class ForbiddenLegacyBuffer:
        @property
        def text(self) -> str:
            raise AssertionError("fullscreen must not read the legacy output buffer")

        @property
        def document(self):
            raise AssertionError("fullscreen must not read the legacy output buffer")

        @document.setter
        def document(self, _value) -> None:
            raise AssertionError("fullscreen must not rebuild the legacy output buffer")

        def set_document(self, *_args, **_kwargs) -> None:
            raise AssertionError("fullscreen must not rebuild the legacy output buffer")

    app.output_buffer = ForbiddenLegacyBuffer()  # type: ignore[assignment]
    for index in range(100):
        app.emit_block(f"delta {index}\n")

    assert app._fullscreen_transcript.text.startswith("delta 0\n")
    assert app._fullscreen_transcript.text.endswith("delta 99\n")


def _projected_row_texts(model, width: int) -> list[str]:
    return ["".join(text for _, text in row.fragments) for row in model._projection(width)]


def test_viewport_anchor_survives_append_width_change_and_semantic_replace() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    sources = ["abcdefgh\n", "ijklmnop\n", "qrstuvwx\n"]
    for source in sources:
        model.append(source)
    model.jump_to_start(width=4)
    model.scroll_visual_lines(1, width=4, height=2)
    anchor = model.viewport.anchor
    assert anchor is not None

    model.append("late output\n")
    assert model.viewport.anchor == anchor
    assert model.top_row(width=4, height=2) == 1

    # Width changes select the visual row containing the same logical source
    # location rather than retaining a width-dependent row number.
    wider_top = model.top_row(width=6, height=2)
    wider_rows = model._projection(6)
    assert wider_rows[wider_top].anchor.record_id == anchor.record_id
    assert wider_rows[wider_top].anchor.source_offset <= anchor.source_offset

    model.replace(sources + ["late output\n"])
    assert model.viewport.anchor == anchor


def test_semantic_replace_remaps_flag_prefix_in_projection_coordinates() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("🇨🇭\nTARGET\nlater\nlast", record_id=101)
    model.jump_to_start(width=10)
    model.scroll_visual_lines(1, width=10, height=2)
    assert model.viewport.anchor is not None
    assert model.viewport.anchor.record_id == 101
    assert _projected_row_texts(model, 10)[model.top_row(width=10, height=2)] == "TARGET"

    model.replace(
        ["🇨🇭\n  TARGET\nlater\nlast"],
        record_ids=[101],
    )

    assert model.viewport.anchor is not None
    assert model.viewport.anchor.record_id == 101
    top = model.top_row(width=10, height=2)
    assert _projected_row_texts(model, 10)[top] == "  TARGET"


def test_semantic_replace_preserves_ids_by_explicit_record_identity() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("alpha\n", record_id=101)
    model.append("bravo long line\n", record_id=202)
    model.append("charlie\n", record_id=303)
    model.jump_to_start(width=6)
    model.scroll_visual_lines(1, width=6, height=2)
    anchor = model.viewport.anchor
    assert anchor is not None and anchor.record_id == 202

    model.replace(
        ["inserted\n", "bravo changed rendering\n", "charlie\n"],
        record_ids=[404, 202, 303],
    )

    assert model.viewport.anchor is not None
    assert model.viewport.anchor.record_id == 202
    top = model.top_row(width=6, height=2)
    assert model._projection(6)[top].anchor.record_id == 202


def test_projection_caches_are_bounded_and_cleared() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("one two three\n")
    for width in range(1, 20):
        model.formatted_text(width=width)

    assert len(model._projection_cache) <= 2
    assert len(model._formatted_cache) <= 2 * 2  # two marker variants per retained geometry

    model.clear()
    assert not model._projection_cache
    assert not model._formatted_cache


def test_anchored_append_preserves_visible_rows_while_tail_append_follows() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    for line in ("one\n", "two\n", "three\n", "four\n"):
        model.append(line)
    model.jump_to_start(width=8)
    before_top = model.top_row(width=8, height=2)
    before = _projected_row_texts(model, 8)[before_top : before_top + 2]

    model.append("five\n")
    after_top = model.top_row(width=8, height=2)
    after = _projected_row_texts(model, 8)[after_top : after_top + 2]
    assert after == before == ["one", "two"]

    model.jump_to_tail()
    model.append("six\n")
    tail_top = model.top_row(width=8, height=2)
    assert _projected_row_texts(model, 8)[tail_top : tail_top + 2] == ["six", ""]


def test_unicode_graphemes_are_never_split_across_visual_rows() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    clusters = ["e\u0301", "界", "🇨🇭", "👩\u200d💻", "1️⃣", "가"]
    model = FullscreenTranscriptModel()
    model.append("".join(clusters))

    # Canonically composable graphemes become one code point. Every valid
    # extended grapheme remains intact at exact-fit and wrap boundaries.
    assert _projected_row_texts(model, 4) == ["é界", "🇨🇭👩‍💻", "1️⃣가"]
    assert _projected_row_texts(model, 2) == [
        "é",
        "界",
        "🇨🇭",
        "👩‍💻",
        "1️⃣",
        "가",
    ]
    assert "".join(_projected_row_texts(model, 2)) == "é界🇨🇭👩‍💻1️⃣가"


def test_grapheme_wider_than_viewport_is_clipped_without_text_mutation() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("界")

    assert _projected_row_texts(model, 1) == ["界"]
    assert model.text == "界"
    assert "�" not in "".join(_projected_row_texts(model, 1))


def test_clear_while_anchored_resets_to_follow_tail() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("one\ntwo\nthree\n")
    model.jump_to_start(width=10)
    assert model.viewport.follows_tail is False

    model.clear()

    assert model.text == ""
    assert model.viewport.follows_tail is True
    assert model.viewport.anchor is None
    assert model.top_row(width=10, height=2) == 0


def test_projection_cache_extends_incrementally_on_stream_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("first record\n")
    first_projection = model._projection(8)
    calls: list[int] = []
    original = model._project_record

    def counted(record, width):
        calls.append(record.record_id)
        return original(record, width)

    monkeypatch.setattr(model, "_project_record", counted)
    model.append("stream delta\n")
    extended = model._projection(8)

    assert calls == [1]
    assert extended[: len(first_projection) - 1] == first_projection[:-1]
    assert _projected_row_texts(model, 8)[-3:] == ["stream ", "delta", ""]


def test_adjacent_records_have_one_separator_without_blank_row() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("abc", record_id=1)
    model.append("def", record_id=2)

    assert model.text == "abc\ndef"
    assert _projected_row_texts(model, 20) == ["abc", "def"]


def test_anchor_survives_semantic_replay_whitespace_reflow() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("status: alpha beta", record_id=7)
    model.jump_to_start(width=8)
    model.scroll_visual_lines(1, width=8, height=1)
    assert model.viewport.anchor is not None
    source_offset = model.viewport.anchor.source_offset
    semantic_offset = model.viewport.anchor.semantic_offset

    model.replace(["status:   alpha\n    beta"], record_ids=[7])

    assert model.viewport.anchor is not None
    assert model.viewport.anchor.source_offset != source_offset
    assert model.viewport.anchor.semantic_offset == semantic_offset
    top = model.top_row(width=8, height=1)
    assert "alpha" in _projected_row_texts(model, 8)[top]


def test_first_append_discards_cached_empty_projection_and_index() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from prompt_toolkit.formatted_text import to_formatted_text

    model = FullscreenTranscriptModel()
    assert model.top_row(width=10, height=5) == 0
    assert (
        "".join(
            text for _style, text in to_formatted_text(model.formatted_text(width=10, height=5))
        )
        == "\n" * 4
    )
    model.jump_to_start(width=10)

    model.append("abcdefgh\n", record_id=0)

    assert _projected_row_texts(model, 4) == ["abcd", "efgh", ""]
    model.jump_to_start(width=4)
    assert model.top_row(width=4, height=1) == 0
    assert (
        "".join(text for _style, text in to_formatted_text(model.formatted_text(width=4, height=3)))
        == "abcd\nefgh\n"
    )


@pytest.mark.parametrize("empty_source", ["", "\x1b[31m\x1b[0m"])
def test_cached_synthetic_empty_row_is_removed_before_first_projectable_append(
    empty_source: str,
) -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    appended = FullscreenTranscriptModel()
    appended.append(empty_source, record_id=10)
    assert _projected_row_texts(appended, 8) == [""]  # Prime synthetic cache.
    appended.jump_to_start(width=8)
    appended.append("x", record_id=20)

    replaced = FullscreenTranscriptModel()
    replaced.replace([empty_source, "x"], record_ids=[10, 20])

    assert _projected_row_texts(appended, 8) == ["x"]
    assert _projected_row_texts(appended, 8) == _projected_row_texts(replaced, 8)
    assert appended.viewport.follows_tail


def test_empty_record_joining_is_stable_across_semantic_replace() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("", record_id=10)
    model.append("x\n", record_id=20)
    before_text = model.text
    before_rows = _projected_row_texts(model, 10)

    model.replace(["", "x\n"], record_ids=[10, 20])

    assert model.text == before_text == "\nx\n"
    assert _projected_row_texts(model, 10) == before_rows


def test_nonempty_then_empty_append_matches_semantic_replace() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    appended = FullscreenTranscriptModel()
    appended.append("x", record_id=10)
    # Prime the incremental projection/index before the empty append.
    assert _projected_row_texts(appended, 3) == ["x"]
    appended.append("", record_id=20)

    rebuilt = FullscreenTranscriptModel()
    rebuilt.replace(["x", ""], record_ids=[10, 20])

    assert appended.text == rebuilt.text == "x\n"
    assert _projected_row_texts(appended, 3) == _projected_row_texts(rebuilt, 3) == ["x", ""]
    appended_rows = appended._projection(3)
    rebuilt_rows = rebuilt._projection(3)
    assert [
        appended._row_index_for_anchor(3, appended_rows, row.anchor) for row in appended_rows
    ] == [rebuilt._row_index_for_anchor(3, rebuilt_rows, row.anchor) for row in rebuilt_rows]

    appended.append("y", record_id=30)
    rebuilt.replace(["x", "", "y"], record_ids=[10, 20, 30])
    assert appended.text == rebuilt.text == "x\ny"
    assert _projected_row_texts(appended, 3) == _projected_row_texts(rebuilt, 3) == ["x", "y"]


def test_newline_then_ansi_only_record_preserves_joining_state() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    appended = FullscreenTranscriptModel()
    appended.append("a\n", record_id=10)
    appended.append("\x1b[31m\x1b[0m", record_id=20)
    appended.append("y", record_id=30)

    rebuilt = FullscreenTranscriptModel()
    rebuilt.replace(
        ["a\n", "\x1b[31m\x1b[0m", "y"],
        record_ids=[10, 20, 30],
    )

    assert appended.text == rebuilt.text == "a\ny"
    assert _projected_row_texts(appended, 10) == _projected_row_texts(rebuilt, 10) == ["a", "y"]


def test_newline_ansi_only_newline_append_matches_semantic_replace() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    sources = ["\n\n", "\x1b[31m\x1b[0m", "\n"]
    record_ids = [10, 20, 30]

    appended = FullscreenTranscriptModel()
    for source, record_id in zip(sources[:2], record_ids[:2], strict=True):
        appended.append(source, record_id=record_id)
    # Prime the incremental projection before the final newline arrives.
    assert _projected_row_texts(appended, 4) == ["", "", ""]
    appended.append(sources[2], record_id=record_ids[2])

    rebuilt = FullscreenTranscriptModel()
    rebuilt.replace(sources, record_ids=record_ids)

    assert appended.text == rebuilt.text == "\n\n\n"
    assert (
        _projected_row_texts(appended, 4)
        == _projected_row_texts(rebuilt, 4)
        == [
            "",
            "",
            "",
            "",
        ]
    )


def test_ansi_only_record_uses_plain_joining_semantics() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("x", record_id=10)
    model.append("\x1b[31m\x1b[0m", record_id=20)
    model.append("y", record_id=30)

    assert model.text == "x\ny"
    assert _projected_row_texts(model, 10) == ["x", "y"]


def test_synthetic_empty_anchor_is_not_reused_as_record_identity() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.jump_to_start(width=10)
    assert model.viewport.anchor is not None
    assert model.viewport.anchor.record_id == 0

    model.replace(["x"], record_ids=[0])

    assert model.viewport.follows_tail
    assert model.viewport.anchor is None
    assert _projected_row_texts(model, 10) == ["x"]


def test_blank_and_indented_rows_have_unique_navigable_anchors() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    blank = FullscreenTranscriptModel()
    blank.append("\n\n\n", record_id=1)
    blank.jump_to_start(width=2)
    assert blank.top_row(width=2, height=1) == 0
    blank.scroll_visual_lines(1, width=2, height=1)
    assert blank.top_row(width=2, height=1) == 1

    indented = FullscreenTranscriptModel()
    indented.append("    abc", record_id=2)
    indented.jump_to_start(width=2)
    assert indented.top_row(width=2, height=1) == 0
    assert _projected_row_texts(indented, 2)[:3] == ["  ", "  ", "ab"]
    indented.scroll_visual_lines(1, width=2, height=1)
    assert indented.top_row(width=2, height=1) == 1


def test_anchored_top_row_uses_bounded_projection_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("".join(f"line-{index}\n" for index in range(10_000)))
    model.jump_to_start(width=12)
    model.scroll_visual_lines(5_000, width=12, height=4)
    rows = model._projection(12)
    anchor = model.viewport.anchor
    assert anchor is not None

    class NoIterationRows:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

        def __iter__(self):
            raise AssertionError("indexed anchor lookup must not scan projected rows")

    # The width index was built on the first anchored lookup. A redraw resolves
    # the same anchor by bisect without iterating the retained projection.
    assert model._row_index_for_anchor(12, NoIterationRows(), anchor) == 5_000


def test_multiline_projection_and_cached_stream_append_are_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("\n".join(f"line-{index}" for index in range(2_000)) + "\n")
    count_calls = 0
    original = model._grapheme_spans

    def counted(chars):
        nonlocal count_calls
        count_calls += 1
        return original(chars)

    monkeypatch.setattr(FullscreenTranscriptModel, "_grapheme_spans", staticmethod(counted))
    rows = model._projection(20)
    # One pass projects rows and one computes the newline-safe synthetic tail.
    assert count_calls == 2
    prefix_identity = id(rows)

    for index in range(200):
        model.append(f"stream-{index}\n")

    assert id(model._projection(20)) == prefix_identity
    # Each append performs the same two bounded passes over only the delta.
    assert count_calls == 402
    assert _projected_row_texts(model, 20)[-2:] == ["stream-199", ""]


def _fullscreen_window_cells(app, *, width: int, height: int) -> list[str]:
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    assert app._output_window is not None
    app._transcript_viewport_size = lambda: (width, height)
    # A real Application increments this per render. This helper drives the
    # Window directly, so advance the same generation-backed fragment cache.
    from prompt_toolkit.application.current import set_app

    app._app.render_counter += 1
    screen = Screen()
    with set_app(app._app):
        app._output_window.write_to_screen(
            screen,
            MouseHandlers(),
            WritePosition(xpos=0, ypos=0, width=width, height=height),
            parent_style="",
            erase_bg=False,
            z_index=None,
        )
    return [
        "".join(
            screen.data_buffer[y][x].char
            for x in range(width)
            if screen.data_buffer[y][x].width > 0
        ).rstrip()
        for y in range(height)
    ]


def test_fullscreen_screen_preserves_native_osc8_metadata_across_wrapped_rows() -> None:
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    app = _make_fullscreen_app()
    app.emit_block(
        "plain \x1b]8;id=docs;https://example.test/docs\x1b\\linked text\x1b]8;;\x1b\\ tail"
    )
    app._transcript_viewport_size = lambda: (8, 5)
    assert app._output_window is not None
    app._app.render_counter += 1
    screen = Screen()
    with set_app(app._app):
        app._output_window.write_to_screen(
            screen,
            MouseHandlers(),
            WritePosition(xpos=0, ypos=0, width=8, height=5),
            parent_style="",
            erase_bg=False,
            z_index=None,
        )

    target = "\x1b]8;;https://example.test/docs\x1b\\"
    close = "\x1b]8;;\x1b\\"
    assert screen.zero_width_escapes[1][0] == target
    assert screen.zero_width_escapes[2][0] == target
    assert screen.zero_width_escapes[2][4] == close
    assert all(
        "id=docs" not in sequence
        for row in screen.zero_width_escapes.values()
        for sequence in row.values()
    )


@pytest.mark.asyncio
async def test_fullscreen_link_at_exact_row_end_closes_before_lower_chrome() -> None:
    from prompt_toolkit.application.current import set_app

    app = _make_fullscreen_app()
    target = "https://example.test/docs"
    app.emit_block(f"\x1b]8;;{target}\x1b\\12345678\x1b]8;;\x1b\\")
    app._command_status_text = "STATUS"
    app._before_render(app._app)
    writes: list[tuple[str, str]] = []
    app._app.output.write_raw = lambda value: writes.append(("raw", value))  # type: ignore[method-assign]
    app._app.output.write = lambda value: writes.append(("text", value))  # type: ignore[method-assign]

    with set_app(app._app):
        app._app.renderer.render(app._app, app._app.layout)

    opening = f"\x1b]8;;{target}\x1b\\"
    close = "\x1b]8;;\x1b\\"
    opening_index = next(index for index, item in enumerate(writes) if item == ("raw", opening))
    last_link_text = max(
        index
        for index, (kind, value) in enumerate(writes)
        if kind == "text" and value in set("12345678")
    )
    close_index = next(
        index
        for index, item in enumerate(writes)
        if index > last_link_text and item == ("raw", close)
    )
    status_index = next(
        index for index, item in enumerate(writes) if item == ("text", "S") and index > close_index
    )
    assert opening_index < last_link_text < close_index < status_index


@pytest.mark.asyncio
async def test_fullscreen_renderer_emits_native_osc8_through_raw_output() -> None:
    from prompt_toolkit.application.current import set_app

    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;https://example.test/docs\x1b\\link\x1b]8;;\x1b\\")
    raw_writes: list[str] = []
    plain_writes: list[str] = []
    app._app.output.write_raw = raw_writes.append  # type: ignore[method-assign]
    app._app.output.write = plain_writes.append  # type: ignore[method-assign]

    with set_app(app._app):
        app._app.renderer.render(app._app, app._app.layout)

    opening = "\x1b]8;;https://example.test/docs\x1b\\"
    assert opening in "".join(raw_writes)
    assert opening not in "".join(plain_writes)


def test_fullscreen_after_render_closes_native_hyperlink_state() -> None:
    app = _make_fullscreen_app()
    writes: list[str] = []
    flushes: list[None] = []
    app._app.output.write_raw = writes.append  # type: ignore[method-assign]
    app._app.output.flush = lambda: flushes.append(None)  # type: ignore[method-assign]

    app._after_render(app._app)

    assert writes == ["\x1b]8;;\x1b\\"]
    assert flushes == [None]


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_fullscreen_after_render_tolerates_broken_output(failure: str) -> None:
    app = _make_fullscreen_app()

    def fail() -> None:
        raise OSError("closed output")

    if failure == "write":
        app._app.output.write_raw = lambda _text: fail()  # type: ignore[method-assign]
    else:
        app._app.output.flush = fail  # type: ignore[method-assign]

    app._after_render(app._app)


def test_short_fullscreen_transcript_is_bottom_aligned_in_viewport() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from prompt_toolkit.formatted_text import to_formatted_text

    model = FullscreenTranscriptModel()
    model.append("agent reply\n")

    fragments = to_formatted_text(model.formatted_text(width=16, height=5))
    rendered = "".join(text for _style, text, *_ in fragments).split("\n")

    assert rendered == ["", "", "", "agent reply", ""]
    assert model.cursor_position(width=16, height=5).y == 4


def test_short_fullscreen_transcript_screen_cells_touch_viewport_bottom() -> None:
    app = _make_fullscreen_app()
    app.emit_block("agent reply\n")

    assert _fullscreen_window_cells(app, width=16, height=5) == [
        "",
        "",
        "",
        "agent reply",
        "",
    ]


def test_short_anchored_fullscreen_transcript_stays_bottom_aligned() -> None:
    app = _make_fullscreen_app()
    app.emit_block("agent reply\n")
    app._fullscreen_transcript.jump_to_start(width=16)

    assert not app._fullscreen_transcript.viewport.follows_tail
    assert _fullscreen_window_cells(app, width=16, height=5) == [
        "",
        "",
        "",
        "agent reply",
        "",
    ]


def test_short_anchored_transcript_stays_bottom_aligned_after_taller_resize() -> None:
    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\n")
    app._fullscreen_transcript.jump_to_start(width=16)

    assert _fullscreen_window_cells(app, width=16, height=2) == ["one", "two"]
    assert _fullscreen_window_cells(app, width=16, height=6) == [
        "",
        "",
        "",
        "one",
        "two",
        "",
    ]


def test_anchored_mid_history_reveals_complete_page_after_taller_resize() -> None:
    app = _make_fullscreen_app()
    app.emit_block("".join(f"{index}\n" for index in range(10)))
    app._fullscreen_transcript.scroll_visual_lines(-6, width=16, height=3)

    assert not app._fullscreen_transcript.viewport.follows_tail
    assert _fullscreen_window_cells(app, width=16, height=3) == ["2", "3", "4"]
    assert _fullscreen_window_cells(app, width=16, height=12) == [
        "",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "",
    ]


def test_fullscreen_screen_cells_stay_fixed_when_output_arrives_while_anchored() -> None:
    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\nthree\nfour\n")
    app._fullscreen_transcript.jump_to_start(width=8)

    before = _fullscreen_window_cells(app, width=8, height=2)
    app.emit_block("five\n")
    after = _fullscreen_window_cells(app, width=8, height=2)

    assert before == after == ["one", "two"]


def test_fullscreen_screen_cells_follow_streaming_tail_and_height_resize() -> None:
    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\nthree\n")
    assert _fullscreen_window_cells(app, width=8, height=2) == ["three", ""]

    app.emit_block("four\n")
    assert _fullscreen_window_cells(app, width=8, height=2) == ["four", ""]
    assert _fullscreen_window_cells(app, width=8, height=3) == ["three", "four", ""]


def test_fullscreen_screen_cells_reflow_unicode_and_clear() -> None:
    app = _make_fullscreen_app()
    app.emit_block("é界🇨🇭👩‍💻1️⃣가")

    narrow_projection = [row for row in _projected_row_texts(app._fullscreen_transcript, 2) if row]
    narrow_screen = [row for row in _fullscreen_window_cells(app, width=2, height=8) if row]
    assert (
        narrow_projection
        == narrow_screen
        == [
            "é",
            "界",
            "🇨🇭",
            "👩‍💻",
            "1️⃣",
            "가",
        ]
    )

    wide_projection = [row for row in _projected_row_texts(app._fullscreen_transcript, 4) if row]
    wide_screen = [row for row in _fullscreen_window_cells(app, width=4, height=5) if row]
    assert wide_projection == wide_screen == ["é界", "🇨🇭👩‍💻", "1️⃣가"]

    app.clear_transcript()
    assert _fullscreen_window_cells(app, width=4, height=3) == ["", "", ""]


def test_one_column_screen_uses_narrow_placeholder_without_mutating_source() -> None:
    app = _make_fullscreen_app()
    app.emit_block("界")

    assert app._fullscreen_transcript.text == "界\n"
    assert _projected_row_texts(app._fullscreen_transcript, 1) == ["界", ""]
    assert _fullscreen_window_cells(app, width=1, height=2) == ["…", ""]


def test_cap_scale_static_resize_does_not_scan_retained_blocks() -> None:
    """The render callback stays O(1) for the 10,000-record common case."""
    from types import SimpleNamespace

    class NoIteration(list):
        def __iter__(self):
            raise AssertionError("static resize scanned retained transcript blocks")

    app = _make_fullscreen_app()
    app._transcript_blocks = NoIteration([object()] * 10_000)  # type: ignore[list-item]
    app._fullscreen_semantic_replay_count = 0
    app._resize_reflow.observe((80, 24))
    app._read_terminal_size = lambda: (79, 24)
    app._app = SimpleNamespace(invalidate=lambda: None)

    app._before_render(app._app)

    assert app._fullscreen_invalidate_count == 1


def test_fullscreen_semantic_replay_cache_avoids_duplicate_width_work() -> None:
    app = _make_fullscreen_app()
    calls = 0

    def replay() -> str:
        nonlocal calls
        calls += 1
        return "semantic"

    app.emit_block("initial", replay=replay)
    app._rebuild_fullscreen_transcript()
    app._rebuild_fullscreen_transcript()

    assert calls == 1
    assert app._fullscreen_transcript.text == "semantic\n"


def test_fullscreen_queue_window_measures_wrapped_multiline_message() -> None:
    from nooa_cli.tui.tui_application import _PendingInputHandoff

    app = _make_fullscreen_app()
    app._pending_input_handoff = [_PendingInputHandoff("first line\n" + "wrapped " * 12)]
    queue_window = app._main_container.children[2].content

    assert bool(queue_window.wrap_lines()) is True
    assert queue_window.preferred_height(20, 100).preferred > 2


def test_fullscreen_wheel_is_guarded_by_explicit_mouse_navigation_policy() -> None:
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    assert app._output_window is not None
    control = app._output_window.content
    event = MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.SCROLL_UP,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )

    assert app._fullscreen_mouse_navigation
    assert bool(app._app.mouse_support()) is True
    control.mouse_handler(event)
    assert not app._fullscreen_transcript.viewport.follows_tail

    app._fullscreen_transcript.jump_to_tail()
    app._fullscreen_mouse_navigation = False
    control.mouse_handler(event)
    assert app._fullscreen_transcript.viewport.follows_tail


def test_fullscreen_wheel_over_short_composer_scrolls_transcript() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    app.input_buffer.text = "draft\n" * 8
    app.input_buffer.cursor_position = len(app.input_buffer.text)
    before_text = app.input_buffer.text
    before_cursor = app.input_buffer.cursor_position

    result = app._input_window.content.mouse_handler(
        _mouse_event(MouseEventType.SCROLL_UP, button=None)
    )

    assert result is None
    assert not app._fullscreen_transcript.viewport.follows_tail
    assert app.input_buffer.text == before_text
    assert app.input_buffer.cursor_position == before_cursor


def test_fullscreen_wheel_over_status_scrolls_transcript() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)

    result = app._status_control.mouse_handler(_mouse_event(MouseEventType.SCROLL_UP, button=None))

    assert result is None
    assert not app._fullscreen_transcript.viewport.follows_tail


@pytest.mark.parametrize("region", ["queue", "session", "input-padding", "completions"])
def test_fullscreen_wheel_over_bottom_chrome_scrolls_transcript(region: str) -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    controls = {
        "queue": app._main_container.children[2].content.content,
        "session": app._main_container.children[3].content,
        "input-padding": app._input_container.children[0].content,
        "completions": app._main_container.children[5].content.content,
    }

    result = controls[region].mouse_handler(_mouse_event(MouseEventType.SCROLL_UP, button=None))

    assert result is None
    assert not app._fullscreen_transcript.viewport.follows_tail


def test_fullscreen_wheel_over_transient_notice_scrolls_transcript() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    app._show_transient_status("Copied 5 characters")
    control = app._transient_status_container.content.content

    result = control.mouse_handler(_mouse_event(MouseEventType.SCROLL_UP, button=None))

    assert result is None
    assert not app._fullscreen_transcript.viewport.follows_tail


@pytest.mark.asyncio
async def test_fullscreen_drag_release_over_transient_notice_finishes_copy(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("zero\none\ntwo")
    app._transcript_viewport_size = lambda: (8, 3)
    copied = []

    async def copy_locally(text: str) -> _ClipboardResult:
        copied.append(text)
        return _ClipboardResult(True, "test")

    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
    app._show_transient_status("Copying...")
    app._before_render(app._app)
    assert app._output_window is not None
    transcript = app._output_window.content
    transcript.create_content(8, 3)
    transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))

    control = app._transient_status_container.content.content
    assert control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=2, y=0)) is None
    if app._clipboard_task is not None:
        await app._clipboard_task

    assert not transcript.dragging
    assert copied == ["zero\none\ntwo"]


def test_transcript_control_reformats_when_frame_height_changes_after_measurement() -> None:
    from prompt_toolkit.application.current import set_app

    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\n")
    assert app._output_window is not None
    control = app._output_window.content

    def lines(content) -> list[str]:
        return [
            "".join(text for _style, text in content.get_line(index))
            for index in range(content.line_count)
        ]

    with set_app(app._app):
        app._app.render_counter += 1
        assert lines(control.create_content(20, 3)) == ["one", "two", " "]

        # A newly occupied status row shrinks the transcript in the same frame.
        # prompt_toolkit measures with height=None before painting at height=2.
        app._status_region_occupied = True
        app._app.render_counter += 1
        control.preferred_height(20, 3, False, None)
        assert lines(control.create_content(20, 2)) == ["one", "two"]

        # Expanding the multiline composer shrinks the transcript once more;
        # typing on that line keeps the same geometry on the following frame.
        app._app.render_counter += 1
        control.preferred_height(20, 2, False, None)
        assert lines(control.create_content(20, 1)) == ["two"]
        app._app.render_counter += 1
        control.preferred_height(20, 1, False, None)
        assert lines(control.create_content(20, 1)) == ["two"]

        # When status chrome disappears, the synthetic trailing row returns in
        # the same frame rather than one repaint later.
        app._status_region_occupied = False
        app._app.render_counter += 1
        control.preferred_height(20, 1, False, None)
        assert lines(control.create_content(20, 2)) == ["two", " "]


def test_transcript_control_uses_current_frame_height_before_formatting() -> None:
    from prompt_toolkit.application.current import set_app

    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\nthree\nfour\n")
    assert app._output_window is not None
    control = app._output_window.content

    def content_lines(height: int) -> list[str]:
        app._app.render_counter += 1
        with set_app(app._app):
            content = control.create_content(8, height)
        return [
            "".join(text for _style, text in content.get_line(index)).rstrip()
            for index in range(content.line_count)
        ]

    # create_content receives current-frame geometry before asking the model
    # for fragments. A height-only change must not reuse prior render_info.
    assert content_lines(2) == ["four", ""]
    assert content_lines(3) == ["three", "four", ""]


@pytest.mark.asyncio
async def test_fullscreen_wheel_scrolls_overflowing_composer_instead_of_transcript() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("".join(f"transcript {index}\n" for index in range(30)))
        app.input_buffer.text = "\n".join(f"draft {index}" for index in range(20))
        app.input_buffer.cursor_position = len(app.input_buffer.text)
        app._app.invalidate()
        await harness.wait_for(
            lambda: (
                app._input_window.render_info is not None
                and app._input_window.render_info.window_height == 10
                and app._input_window.vertical_scroll > 0
            )
        )
        before_scroll = app._input_window.vertical_scroll
        before_cursor_row = app.input_buffer.document.cursor_position_row

        result = app._input_window.content.mouse_handler(
            _mouse_event(MouseEventType.SCROLL_UP, button=None)
        )
        assert app._input_window.vertical_scroll < before_scroll

        assert result is None
        assert app._fullscreen_transcript.viewport.follows_tail
        assert app.input_buffer.document.cursor_position_row == before_cursor_row
        assert app._input_window.manual_cursor_is_offscreen()
        after_up_scroll = app._input_window.vertical_scroll

        result = app._input_window.content.mouse_handler(
            _mouse_event(MouseEventType.SCROLL_DOWN, button=None)
        )
        assert app._input_window.vertical_scroll > after_up_scroll

        assert result is None
        assert app._fullscreen_transcript.viewport.follows_tail
        assert app.input_buffer.document.cursor_position_row == before_cursor_row

        app._input_window.content.mouse_handler(_mouse_event(MouseEventType.SCROLL_UP, button=None))
        assert app._input_window.manual_cursor_is_offscreen()
        await harness.type_keys("!")
        await harness.wait_for(lambda: not app._input_window._manual_wheel_scroll)
        await harness.wait_for(lambda: app._input_window.vertical_scroll == before_scroll)

        assert app.input_buffer.document.cursor_position_row == before_cursor_row
        assert not app._input_window.manual_cursor_is_offscreen()


@pytest.mark.asyncio
async def test_fullscreen_wheel_scrolls_wrapped_draft_without_moving_cursor() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.input_buffer.text = "wrapped " * 300
        app.input_buffer.cursor_position = len(app.input_buffer.text)
        app._app.invalidate()
        await harness.wait_for(
            lambda: (
                app._input_window.render_info is not None
                and app._input_window.render_info.window_height == 10
                and app._input_window.vertical_scroll_2 > 0
            )
        )
        before_cursor = app.input_buffer.cursor_position
        before_scroll = app._input_window.vertical_scroll_2

        app._input_window.content.mouse_handler(_mouse_event(MouseEventType.SCROLL_UP, button=None))

        assert app.input_buffer.cursor_position == before_cursor
        assert app._input_window.vertical_scroll_2 < before_scroll
        assert app._input_window.manual_cursor_is_offscreen()

        await harness.type_keys("!")
        await harness.wait_for(lambda: not app._input_window._manual_wheel_scroll)

        assert app.input_buffer.cursor_position == before_cursor + 1
        assert not app._input_window.manual_cursor_is_offscreen()


@pytest.mark.asyncio
async def test_fullscreen_page_keys_scroll_overflowing_composer_instead_of_transcript() -> None:
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("".join(f"transcript {index}\n" for index in range(30)))
        app.input_buffer.text = "\n".join(f"draft {index}" for index in range(20))
        app.input_buffer.cursor_position = len(app.input_buffer.text)
        app._app.invalidate()
        await harness.wait_for(
            lambda: (
                app._input_window.render_info is not None
                and app._input_window.render_info.window_height == 10
            )
        )
        before_cursor_row = app.input_buffer.document.cursor_position_row

        await harness.press("pageup")
        await harness.wait_for(
            lambda: app.input_buffer.document.cursor_position_row < before_cursor_row
        )

        assert app._fullscreen_transcript.viewport.follows_tail
        assert app._input_window.vertical_scroll < 10
        after_up_cursor_row = app.input_buffer.document.cursor_position_row

        after_up_scroll = app._input_window.vertical_scroll
        await harness.press("pagedown")
        await harness.wait_for(
            lambda: (
                app.input_buffer.document.cursor_position_row > after_up_cursor_row
                or app._input_window.vertical_scroll > after_up_scroll
            )
        )

        assert app._fullscreen_transcript.viewport.follows_tail


@pytest.mark.asyncio
async def test_fullscreen_navigation_preserves_composer_draft_focus_and_mouse_policy() -> None:
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("".join(f"line {index}\n" for index in range(80)))
        await harness.type_keys("draft text")
        await harness.wait_input_equals("draft text")
        assert app._app.layout.current_buffer is app.input_buffer
        assert bool(app._app.mouse_support()) is True

        await harness.press("pageup")
        await harness.wait_for(lambda: not app._fullscreen_transcript.viewport.follows_tail)
        assert harness.capture_input() == "draft text"
        assert app._app.layout.current_buffer is app.input_buffer

        await harness.press("f6")
        await harness.wait_for(lambda: not bool(app._app.mouse_support()))
        assert harness.capture_input() == "draft text"
        await harness.press("f6")
        await harness.wait_for(lambda: bool(app._app.mouse_support()))

        await harness.press("c-end")
        await harness.wait_for(lambda: app._fullscreen_transcript.viewport.follows_tail)
        assert harness.capture_input() == "draft text"
        assert app._app.layout.current_buffer is app.input_buffer


@pytest.mark.asyncio
async def test_fullscreen_enter_submission_returns_scrolled_transcript_to_tail() -> None:
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("".join(f"line {index}\n" for index in range(80)))
        await harness.press("pageup")
        await harness.wait_for(lambda: not app._fullscreen_transcript.viewport.follows_tail)
        app.emit_block("agent reply\n", agent_message=True)
        assert app._has_unseen_agent_message

        await harness.type_keys("follow up")
        await harness.press("enter")
        await harness.wait_for(lambda: harness.agent.messages_received == ["follow up"])

        assert app._fullscreen_transcript.viewport.follows_tail
        assert not app._has_unseen_agent_message


@pytest.mark.asyncio
async def test_fullscreen_renderer_requests_mouse_reporting_on_start_and_f6_toggles() -> None:
    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput()
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN, output=output) as harness:
        await harness.wait_for(lambda: ("enable_mouse_support",) in output.events)

        await harness.press("f6")
        await harness.wait_for(lambda: ("disable_mouse_support",) in output.events)

        enable_count = output.events.count(("enable_mouse_support",))
        await harness.press("f6")
        await harness.wait_for(
            lambda: output.events.count(("enable_mouse_support",)) > enable_count
        )


def test_vt100_mouse_reporting_uses_xterm_modes_expected_by_tmux() -> None:
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    stream = io.StringIO()
    output = Vt100_Output(
        stream,
        get_size=lambda: Size(rows=24, columns=80),
        term="xterm-256color",
        enable_cpr=False,
    )

    output.enable_mouse_support()
    output.flush()

    written = stream.getvalue()
    assert "\x1b[?1000h" in written
    assert "\x1b[?1003h" in written
    assert "\x1b[?1006h" in written


@pytest.mark.asyncio
async def test_fullscreen_subview_round_trip_preserves_logical_viewport_anchor() -> None:
    class View:
        mouse_support = False

        def on_open(self) -> None:
            pass

        def on_close(self) -> None:
            pass

        def render(self, _width: int, _height: int) -> str:
            return "temporary view"

        def handle_key(self, key: str, _value: str | None = None) -> str:
            return "close" if key == "quit" else "ignored"

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.emit_block("one\ntwo\nthree\nfour\n")
        app._fullscreen_transcript.jump_to_start(width=8)
        anchor = app._fullscreen_transcript.viewport.anchor
        cells_before = _fullscreen_window_cells(app, width=8, height=2)

        opened = asyncio.create_task(app.open_subview(View()))  # type: ignore[arg-type]
        await harness.wait_for(lambda: app.active_subview is not None)
        await harness.press("q")
        await asyncio.wait_for(opened, timeout=1)

        assert app._fullscreen_transcript.viewport.anchor == anchor
        assert _fullscreen_window_cells(app, width=8, height=2) == cells_before == ["one", "two"]
        assert app._app.layout.current_buffer is app.input_buffer


def test_fullscreen_selection_copies_logical_text_without_soft_wraps_or_ansi() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("\x1b[31mabcdef\x1b[0m\nsecond")

    model.begin_selection(x=2, y=0, width=4, height=4)
    model.update_selection(x=2, y=2, width=4, height=4)

    assert model.selected_text() == "cdef\nsec"


def test_fullscreen_selection_preserves_unicode_graphemes_and_reverse_drag() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("Aé界Z")

    model.begin_selection(x=4, y=0, width=8, height=1)
    model.update_selection(x=1, y=0, width=8, height=1)

    assert model.selected_text() == "é界Z"


def test_fullscreen_selection_extends_across_autoscrolled_pages() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("zero\none\ntwo\nthree\nfour\nfive")

    # Tail viewport initially shows three/four/five. Start on four, then scroll
    # upward while dragging and extend to the top visible row.
    model.begin_selection(x=0, y=1, width=8, height=3)
    model.scroll_visual_lines(-2, width=8, height=3)
    model.update_selection(x=0, y=0, width=8, height=3)

    assert model.selected_text() == "one\ntwo\nthree\nf"


def test_fullscreen_selection_can_be_cancelled_without_mutating_viewport() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("alpha beta")
    before = model.viewport
    model.begin_selection(x=0, y=0, width=20, height=1)
    model.update_selection(x=4, y=0, width=20, height=1)
    assert model.selected_text() == "alpha"

    model.clear_selection()

    assert model.selected_text() == ""
    assert model.viewport == before


def _mouse_event(event_type, *, x=0, y=0, button=None, shift=False, alt=False):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseModifier

    return MouseEvent(
        position=Point(x=x, y=y),
        event_type=event_type,
        button=button or MouseButton.LEFT,
        modifiers=frozenset(
            modifier
            for modifier, enabled in (
                (MouseModifier.SHIFT, shift),
                (MouseModifier.ALT, alt),
            )
            if enabled
        ),
    )


@pytest.mark.asyncio
async def test_fullscreen_mouse_drag_selects_autoscrolls_and_copies(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("zero\none\ntwo\nthree\nfour\nfive")
    app._transcript_viewport_size = lambda: (8, 3)
    copied = []

    async def copy_locally(text: str) -> _ClipboardResult:
        copied.append(text)
        return _ClipboardResult(True, "test")

    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(8, 3)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=1))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=0, y=0))
    await asyncio.sleep(0.62)
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=0, y=0))
    assert app._fullscreen_transcript.selected_text() == ""
    if app._clipboard_task is not None:
        await app._clipboard_task

    assert copied == ["one\ntwo\nthree\nfour\nf"]
    assert app._transient_status_text == "Copied 20 characters"


@pytest.mark.asyncio
async def test_fullscreen_drag_recovers_release_outside_tmux_on_reentry(monkeypatch) -> None:
    """A no-button motion closes the drag whose release tmux could not report."""
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseButton, MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("zero\none\ntwo")
    app._transcript_viewport_size = lambda: (8, 3)
    copied = []

    async def copy_locally(text: str) -> _ClipboardResult:
        copied.append(text)
        return _ClipboardResult(True, "test")

    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
    assert app._output_window is not None
    transcript = app._output_window.content
    transcript.create_content(8, 3)

    transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
    transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=2, y=1))
    assert transcript.dragging
    assert app._fullscreen_transcript.selected_text() == "one\ntwo"

    # No MOUSE_UP arrives while the pointer is outside the tmux pane.  With
    # all-motion reporting, re-entry after release is the first no-button move.
    transcript.mouse_handler(
        _mouse_event(MouseEventType.MOUSE_MOVE, x=4, y=1, button=MouseButton.NONE)
    )
    assert not transcript.dragging
    assert app._fullscreen_transcript.selected_text() == ""
    if app._clipboard_task is not None:
        await app._clipboard_task

    assert copied == ["one\ntwo"]
    assert app._transient_status_text == "Copied 7 characters"


@pytest.mark.asyncio
async def test_fullscreen_drag_release_over_status_finishes_and_copies(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("zero\none\ntwo")
    app._transcript_viewport_size = lambda: (8, 3)
    copied = []

    async def copy_locally(text: str) -> _ClipboardResult:
        copied.append(text)
        return _ClipboardResult(True, "test")

    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
    assert app._output_window is not None
    transcript = app._output_window.content
    transcript.create_content(8, 3)

    transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
    assert transcript.dragging
    assert (
        app._status_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=2, y=0)) is None
    )
    assert (
        app._status_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=2, y=0)) is None
    )
    assert app._fullscreen_transcript.selected_text() == ""
    if app._clipboard_task is not None:
        await app._clipboard_task

    assert not transcript.dragging
    assert copied == ["one\ntwo\n"]
    assert app._transient_status_text == "Copied 8 characters"


@pytest.mark.asyncio
async def test_fullscreen_drag_recovers_outside_release_reentering_over_status(
    monkeypatch,
) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseButton, MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("zero\none\ntwo")
    app._transcript_viewport_size = lambda: (8, 3)
    copied = []

    async def copy_locally(text: str) -> _ClipboardResult:
        copied.append(text)
        return _ClipboardResult(True, "test")

    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
    assert app._output_window is not None
    transcript = app._output_window.content
    transcript.create_content(8, 3)

    transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
    transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=2, y=1))
    assert transcript.dragging

    result = app._status_control.mouse_handler(
        _mouse_event(MouseEventType.MOUSE_MOVE, x=4, y=0, button=MouseButton.NONE)
    )
    assert result is None
    assert not transcript.dragging
    if app._clipboard_task is not None:
        await app._clipboard_task

    # Bottom chrome is clamped to the final transcript row.
    assert copied == ["one\ntwo\n"]
    assert app._fullscreen_transcript.selected_text() == ""


@pytest.mark.asyncio
async def test_fullscreen_drag_release_over_composer_finishes_and_copies(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("zero\none\ntwo")
    app._transcript_viewport_size = lambda: (8, 3)
    app.input_buffer.text = "composer text"
    app.input_buffer.cursor_position = len(app.input_buffer.text)
    copied = []

    async def copy_locally(text: str) -> _ClipboardResult:
        copied.append(text)
        return _ClipboardResult(True, "test")

    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
    assert app._output_window is not None
    transcript = app._output_window.content
    transcript.create_content(8, 3)
    composer = app._input_window.content
    composer.create_content(20, 1)

    with set_app(app._app):
        transcript.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
        assert transcript.dragging
        assert composer.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=2, y=0)) is None
        assert composer.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=2, y=0)) is None
    assert app._fullscreen_transcript.selected_text() == ""
    if app._clipboard_task is not None:
        await app._clipboard_task

    assert not transcript.dragging
    assert copied == ["one\ntwo\n"]
    assert app.input_buffer.text == "composer text"
    assert app.input_buffer.cursor_position == len(app.input_buffer.text)
    assert app.input_buffer.selection_state is None
    assert app._transient_status_text == "Copied 8 characters"


def test_fullscreen_modifier_drag_and_disabled_mouse_defer_to_native_selection() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("select me")
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(20, 2)
    app._fullscreen_transcript.begin_selection(x=0, y=0, width=20, height=2)
    app._fullscreen_transcript.update_selection(x=4, y=0, width=20, height=2)
    selected = app._fullscreen_transcript.selected_text()
    viewport = app._fullscreen_transcript.viewport

    assert (
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, alt=True)) is NotImplemented
    )
    assert app._fullscreen_transcript.selected_text() == selected
    assert app._fullscreen_transcript.viewport == viewport

    # Keep Shift as a compatibility escape for terminal/tmux defaults.
    assert (
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, shift=True)) is NotImplemented
    )
    assert app._fullscreen_transcript.selected_text() == selected
    assert app._fullscreen_transcript.viewport == viewport

    app._fullscreen_mouse_navigation = False
    assert control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN)) is NotImplemented
    assert app._fullscreen_transcript.selected_text() == selected


def test_fullscreen_selection_is_visibly_styled() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("abcdef")
    model.begin_selection(x=1, y=0, width=10, height=1)
    model.update_selection(x=3, y=0, width=10, height=1)

    fragments = model.formatted_text(width=10, height=1)
    selected = "".join(text for style, text, *_ in fragments if "selected" in style)

    assert selected == "bcd"


def test_fullscreen_click_without_drag_activates_link_but_drag_does_not() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;https://example.test/docs\x1b\\link\x1b]8;;\x1b\\ plain")
    app._transcript_viewport_size = lambda: (20, 1)
    opened: list[tuple[int, int]] = []
    app._open_fullscreen_link_at = lambda x, y: opened.append((x, y)) or True
    assert app._output_window is not None
    control = app._output_window.content
    control._link_callback = app._open_fullscreen_link_at
    control.create_content(20, 1)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=1, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=1, y=0))
    assert opened == [(1, 0)]

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=1, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=2, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=2, y=0))
    assert opened == [(1, 0)]

    # A terminal may omit an intermediate motion report; a changed release
    # coordinate still belongs to drag selection, never link activation.
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=1, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=2, y=0))
    assert opened == [(1, 0)]


@pytest.mark.asyncio
async def test_local_browser_timeout_counts_as_dispatched_without_termination(monkeypatch) -> None:
    app = _make_fullscreen_app()

    class Process:
        returncode = None

        async def wait(self) -> None:
            return None

    process = Process()
    terminated: list[object] = []

    async def create_process(*_args, **_kwargs):
        return process

    async def time_out(awaitable, *, timeout):
        assert timeout == 5
        awaitable.close()
        raise TimeoutError

    async def terminate(candidate) -> None:
        terminated.append(candidate)

    monkeypatch.setattr(app, "_browser_open_command", lambda _url: ("open", "url"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(asyncio, "wait_for", time_out)
    monkeypatch.setattr(app, "_terminate_clipboard_process", terminate)

    assert await app._open_local_url("https://example.test/docs") is True
    assert terminated == []


@pytest.mark.asyncio
async def test_fullscreen_link_click_opens_safe_http_url(monkeypatch) -> None:
    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;https://example.test/docs\x1b\\link\x1b]8;;\x1b\\")
    app._transcript_viewport_size = lambda: (20, 2)
    calls: list[str] = []

    async def open_browser(url: str) -> bool:
        calls.append(url)
        return True

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(app, "_open_local_url", open_browser)

    assert app._open_fullscreen_link_at(1, 0) is True
    assert app._link_task is not None
    await app._link_task
    assert calls == ["https://example.test/docs"]
    assert app._open_fullscreen_link_at(8, 0) is False


@pytest.mark.asyncio
async def test_fullscreen_file_link_click_uses_platform_opener(monkeypatch) -> None:
    app = _make_fullscreen_app()
    url = "file:///path/to/file"
    app.emit_block(f"\x1b]8;;{url}\x1b\\link\x1b]8;;\x1b\\")
    app._transcript_viewport_size = lambda: (20, 2)
    calls: list[str] = []

    async def open_file(target: str) -> bool:
        calls.append(target)
        return True

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(app, "_open_local_url", open_file)

    assert app._open_fullscreen_link_at(1, 0) is True
    assert app._link_task is not None
    await app._link_task
    assert calls == [url]


@pytest.mark.asyncio
async def test_fullscreen_link_open_failure_copies_url(monkeypatch) -> None:
    app = _make_fullscreen_app()
    url = "https://example.test/docs"
    app.emit_block(f"\x1b]8;;{url}\x1b\\link\x1b]8;;\x1b\\")
    app._transcript_viewport_size = lambda: (20, 2)
    copied: list[str] = []

    async def fail_to_open(_url: str) -> bool:
        return False

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(app, "_open_local_url", fail_to_open)
    monkeypatch.setattr(app, "_start_fullscreen_selection_copy", copied.append)

    assert app._open_fullscreen_link_at(1, 0) is True
    task = app._link_task
    assert task is not None
    await task
    assert copied == [url]


@pytest.mark.asyncio
async def test_fullscreen_remote_link_click_copies_without_launching(monkeypatch) -> None:
    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;https://example.test/docs\x1b\\link\x1b]8;;\x1b\\")
    app._transcript_viewport_size = lambda: (20, 2)
    copied: list[str] = []

    monkeypatch.setenv("SSH_CONNECTION", "client 1 server 2")
    monkeypatch.setattr(
        app,
        "_start_fullscreen_selection_copy",
        lambda text: copied.append(text),
    )

    async def forbidden_open(_url: str) -> bool:
        raise AssertionError("remote clicks must not launch a host browser")

    monkeypatch.setattr(app, "_open_local_url", forbidden_open)

    assert app._open_fullscreen_link_at(1, 0) is True
    assert copied == ["https://example.test/docs"]
    assert app._link_task is None


@pytest.mark.asyncio
async def test_fullscreen_rapid_link_clicks_do_not_duplicate_launch(monkeypatch) -> None:
    app = _make_fullscreen_app()
    app.emit_block("\x1b]8;;https://example.test/docs\x1b\\link\x1b]8;;\x1b\\")
    app._transcript_viewport_size = lambda: (20, 2)
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)

    async def blocked_open(url: str) -> bool:
        calls.append(url)
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(app, "_open_local_url", blocked_open)

    assert app._open_fullscreen_link_at(1, 0) is True
    await started.wait()
    assert app._open_fullscreen_link_at(1, 0) is True
    assert calls == ["https://example.test/docs"]
    release.set()
    assert app._link_task is not None
    await app._link_task


def test_fullscreen_click_without_drag_clears_selection_without_copy(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("select me")
    app._transcript_viewport_size = lambda: (20, 1)
    copied = []
    monkeypatch.setattr(
        app,
        "_copy_to_clipboard_result",
        lambda text: copied.append(text) or _ClipboardResult(True, "test"),
    )
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(20, 1)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=2, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=2, y=0))

    assert copied == []
    assert app._fullscreen_transcript.selected_text() == ""


def test_fullscreen_copy_failure_explains_native_selection_escape() -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult

    app = _make_fullscreen_app()
    app._report_fullscreen_copy("copy", _ClipboardResult(False, reason="terminal rejected OSC 52"))

    assert "terminal rejected OSC 52" in app._transient_status_text
    assert "Option/Alt-drag" in app._transient_status_text
    assert "F6" in app._transient_status_text


@pytest.mark.asyncio
async def test_fullscreen_stationary_edge_drag_repeats_autoscroll() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(12, 4)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=2))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=0, y=0))
    first_top = app._fullscreen_transcript.top_row(width=12, height=4)
    await asyncio.sleep(0.62)
    later_top = app._fullscreen_transcript.top_row(width=12, height=4)
    control.cancel_drag()

    assert later_top <= first_top - 2


def test_fullscreen_selection_style_reaches_prompt_toolkit_screen_cells() -> None:
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    app = _make_fullscreen_app()
    app.emit_block("abcdef\n")
    app._fullscreen_transcript.begin_selection(x=1, y=0, width=10, height=2)
    app._fullscreen_transcript.update_selection(x=3, y=0, width=10, height=2)
    assert app._output_window is not None
    screen = Screen()

    app._output_window.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(xpos=0, ypos=0, width=10, height=2),
        parent_style="",
        erase_bg=False,
        z_index=None,
    )

    assert "class:selected" not in screen.data_buffer[0][0].style
    assert all("class:selected" in screen.data_buffer[0][x].style for x in range(1, 4))
    assert "class:selected" not in screen.data_buffer[0][4].style


@pytest.mark.asyncio
async def test_fullscreen_copy_notice_expires_without_touching_command_status() -> None:
    app = _make_fullscreen_app()
    app._command_status_text = "command remains"

    app._show_transient_status(
        "Copied 5 characters",
        seconds=0.01,
        style="class:return-to-tail",
    )
    assert app._transient_status_text == "Copied 5 characters"
    assert bool(app._transient_status_container.filter())
    assert (
        "class:return-to-tail",
        "Copied 5 characters",
    ) in app._transient_status_container.content.content.text()
    await asyncio.sleep(0.03)

    assert app._transient_status_text == ""
    assert app._transient_status_timer is None
    assert app._command_status_text == "command remains"


def test_copy_notice_style_is_structural_with_mixed_status_rows() -> None:
    from nooa_cli.tui.host_services import TUIHostServices
    from nooa_cli.tui.tui_application import TUIApplication, _ClipboardResult

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        app = TUIApplication(
            display_mode=DisplayMode.FULLSCREEN,
            host_services=TUIHostServices(auxiliary_status=lambda: "Earlier: Copied 5 characters"),
        )
    app._command_status_text = "command remains"
    app.set_session_label("session")
    app._report_fullscreen_copy("12345", _ClipboardResult(True, "test"))

    assert app._status_rows() == [
        [("class:status", "Earlier: Copied 5 characters")],
        [("class:return-to-tail", "Copied 5 characters")],
        [
            ("class:status", "command remains"),
            ("class:status", "   [session]"),
        ],
    ]

    app._report_fullscreen_copy(
        "12345",
        _ClipboardResult(False, reason="terminal rejected OSC 52"),
    )
    assert all(style == "class:status" for row in app._status_rows() for style, _text in row)


def test_fullscreen_selection_ignores_blank_top_padding() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("only")

    model.begin_selection(x=0, y=0, width=10, height=4)

    assert model.selected_text() == ""


def test_fullscreen_reflow_cancels_active_drag_and_selection() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("one\ntwo\nthree\nfour")
    app._transcript_viewport_size = lambda: (10, 2)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(10, 2)
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=2, y=0))
    assert app._fullscreen_transcript.selected_text()

    app._rebuild_fullscreen_transcript()

    assert app._fullscreen_transcript.selected_text() == ""
    assert control._dragging is False
    assert control._autoscroll_timer is None


def test_fullscreen_entering_edge_row_does_not_scroll_immediately() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(12, 4)
    before = app._fullscreen_transcript.top_row(width=12, height=4)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=2))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=3, y=0))
    after = app._fullscreen_transcript.top_row(width=12, height=4)
    control.cancel_drag()

    assert after == before


@pytest.mark.asyncio
async def test_fullscreen_opening_subview_cancels_drag_autoscroll() -> None:
    from nooa_cli.tui.subapp import TextPromptView
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(12, 4)
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=2))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=0, y=0))
    assert control._autoscroll_timer is not None

    opening = asyncio.create_task(app.open_subview(TextPromptView("Title", "Prompt")))
    await asyncio.sleep(0)

    assert control._dragging is False
    assert control._autoscroll_timer is None
    assert app._fullscreen_transcript.selected_text() == ""
    app._close_subview()
    await opening


@pytest.mark.asyncio
async def test_fullscreen_second_edge_row_arms_autoscroll_without_restarting_timer() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 8)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(12, 8)
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=4))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=0, y=1))
    timer = control._autoscroll_timer

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=4, y=1))

    assert timer is not None
    assert control._autoscroll_timer is timer
    control.cancel_drag()


def test_fullscreen_shift_escape_stops_drag_without_clearing_selection() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("select me")
    app._transcript_viewport_size = lambda: (20, 2)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(20, 2)
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=4, y=0))
    assert app._fullscreen_transcript.selected_text()

    result = control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=5, y=0, shift=True))

    assert result is NotImplemented
    assert app._fullscreen_transcript.selected_text() == "selec"
    assert control._dragging is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("height", "y", "direction"),
    [(1, 0, 0), (2, 0, -1), (2, 1, 1), (3, 0, -1), (3, 1, 0), (3, 2, 1)],
)
async def test_fullscreen_short_viewport_edge_direction(
    height: int, y: int, direction: int
) -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, height)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(12, height)
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=y))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=1, y=y))

    assert control._autoscroll_direction == direction
    assert (control._autoscroll_timer is not None) is (direction != 0)
    control.cancel_drag()


@pytest.mark.asyncio
async def test_superseded_clipboard_copy_does_not_run_after_its_own_cancellation(
    monkeypatch,
) -> None:
    app = _make_fullscreen_app()
    predecessor_started = asyncio.Event()
    release_predecessor = asyncio.Event()
    copied: list[str] = []

    async def previous_copy() -> None:
        predecessor_started.set()
        try:
            await release_predecessor.wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise

    async def local_copy(text: str):
        copied.append(text)
        return None

    previous = asyncio.create_task(previous_copy())
    await predecessor_started.wait()
    previous.cancel()
    monkeypatch.setattr(app, "_copy_to_local_clipboard_async", local_copy)
    current = asyncio.create_task(app._copy_fullscreen_selection("stale", previous=previous))
    current.cancel()

    with pytest.raises(asyncio.CancelledError):
        await current
    assert copied == []


def test_fullscreen_return_to_tail_affordance_is_visible_and_clickable() -> None:
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)

    assert not bool(app._return_to_tail_container.filter())
    app._scroll_fullscreen_transcript(-4)

    assert bool(app._return_to_tail_container.filter())
    assert "↓ Return to bottom (Ctrl+End)" in fragment_list_to_text(
        to_formatted_text(app._return_to_tail_control.text())
    )
    down = MouseEvent(
        position=Point(x=3, y=0),
        event_type=MouseEventType.MOUSE_DOWN,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )
    up = MouseEvent(
        position=Point(x=3, y=0),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )
    assert app._return_to_tail_control.mouse_handler(down) is None
    assert app._return_to_tail_control.mouse_handler(up) is None
    assert app._fullscreen_transcript.viewport.follows_tail
    assert not bool(app._return_to_tail_container.filter())


def test_fullscreen_wheel_over_return_to_tail_scrolls_transcript() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(20)))
    app._transcript_viewport_size = lambda: (12, 4)
    app._scroll_fullscreen_transcript(-4)
    before = app._fullscreen_transcript.viewport

    result = app._return_to_tail_control.mouse_handler(
        _mouse_event(MouseEventType.SCROLL_UP, button=None)
    )

    assert result is None
    assert app._fullscreen_transcript.viewport != before
    assert not app._fullscreen_transcript.viewport.follows_tail


def test_fullscreen_return_to_tail_affordance_preserves_native_mouse_escape() -> None:
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import (
        MouseButton,
        MouseEvent,
        MouseEventType,
        MouseModifier,
    )

    app = _make_fullscreen_app()
    app.emit_block("".join(f"line {index}\n" for index in range(30)))
    app._transcript_viewport_size = lambda: (20, 4)
    app._scroll_fullscreen_transcript(-4)
    events = [
        MouseEvent(
            position=Point(x=3, y=0),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.RIGHT,
            modifiers=frozenset(),
        ),
        MouseEvent(
            position=Point(x=3, y=0),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset({MouseModifier.ALT}),
        ),
        MouseEvent(
            position=Point(x=3, y=0),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset({MouseModifier.SHIFT}),
        ),
    ]

    assert all(
        app._return_to_tail_control.mouse_handler(event) is NotImplemented for event in events
    )
    assert not app._fullscreen_transcript.viewport.follows_tail


def test_fullscreen_window_preserves_blank_row_mouse_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank rendered row must not make prompt_toolkit route its click to row zero."""
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.emit_block("top\n\nbottom")
    app._transcript_viewport_size = lambda: (12, 3)
    assert app._output_window is not None
    screen = Screen()
    handlers = MouseHandlers()
    app._output_window.write_to_screen(
        screen,
        handlers,
        WritePosition(xpos=0, ypos=0, width=12, height=3),
        parent_style="",
        erase_bg=False,
        z_index=None,
    )

    # Dispatch through Window's screen-coordinate adapter, which used to map
    # empty rows to the control fallback coordinate (0, 0).
    from prompt_toolkit.application.current import set_app

    # This isolated render has no running Application to establish the modal
    # walk. Keep Window's production coordinate adapter under test while
    # supplying the same active window that the running layout would expose.
    monkeypatch.setattr(app._app.layout, "walk_through_modal_area", lambda: [app._output_window])
    handler = handlers.mouse_handlers[0][0]
    with set_app(app._app):
        handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
        # Selection invalidation redraws before the next movement event. The
        # blank row must remain addressable in that selected rendering too.
        handlers = MouseHandlers()
        app._output_window.write_to_screen(
            Screen(),
            handlers,
            WritePosition(xpos=0, ypos=0, width=12, height=3),
            parent_style="",
            erase_bg=False,
            z_index=None,
        )
        handlers.mouse_handlers[1][0](_mouse_event(MouseEventType.MOUSE_MOVE, x=0, y=1))

    assert app._fullscreen_transcript.selected_text() == "\nb"


@pytest.mark.asyncio
async def test_fullscreen_input_mouse_click_moves_cursor_and_drag_selects() -> None:
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.input_buffer.text = "alpha beta"
    app.input_buffer.cursor_position = len(app.input_buffer.text)
    control = app._input_window.content
    # Populate BufferControl's processed-line mapping. The two display columns
    # occupied by the prompt must not shift cursor/selection source offsets.
    control.create_content(20, 1)

    with set_app(app._app):
        app._app.layout.current_control = control
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=5, y=0))
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=5, y=0))

        assert app.input_buffer.cursor_position == 3
        assert app.input_buffer.selection_state is None

    drag_app = _make_fullscreen_app()
    copied: list[str] = []
    drag_app._start_fullscreen_selection_copy = copied.append
    drag_app.input_buffer.text = "alpha beta"
    drag_control = drag_app._input_window.content
    drag_control.create_content(20, 1)
    with set_app(drag_app._app):
        drag_app._app.layout.current_control = drag_control
        drag_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=3, y=0))
        drag_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=7, y=0))
        drag_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=7, y=0))

    assert drag_app.input_buffer.document.selection_range() == (1, 5)
    assert drag_app.input_buffer.copy_selection().text == "lpha"
    assert copied == ["lpha"]
    assert drag_app._app.clipboard.get_data().text == "lpha"


@pytest.mark.asyncio
async def test_fullscreen_input_shift_arrows_select_and_delete() -> None:
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.input_buffer.text = "alpha beta"
        app.input_buffer.cursor_position = len(app.input_buffer.text)

        await harness.press("s-left")
        await harness.press("s-left")
        await harness.wait_for(lambda: app.input_buffer.document.selection_range() == (8, 10))
        await harness.press("delete")
        await harness.wait_for(lambda: app.input_buffer.text == "alpha be")

        assert app.input_buffer.selection_state is None
        assert app.input_buffer.cursor_position == len("alpha be")


@pytest.mark.asyncio
async def test_fullscreen_input_shift_up_selects_and_typing_replaces() -> None:
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        app.input_buffer.text = "abcde\n12345"
        app.input_buffer.cursor_position = len(app.input_buffer.text)

        await harness.press("s-up")
        await harness.wait_for(lambda: app.input_buffer.document.selection_range() == (5, 11))
        await harness.type_keys("!")
        await harness.wait_for(lambda: app.input_buffer.text == "abcde!")

        assert app.input_buffer.selection_state is None
        assert app.input_buffer.cursor_position == len("abcde!")


def test_fullscreen_composer_mouse_selection_is_mirrored_non_destructively(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _NativeSelectionBufferControl
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.selection import SelectionState

    app = _make_fullscreen_app()
    copied: list[str] = []
    app._start_fullscreen_selection_copy = copied.append
    app.input_buffer.text = "copy this"
    app.input_buffer.cursor_position = 4
    app.input_buffer.selection_state = SelectionState(original_cursor_position=0)
    control = app._input_window.content
    assert isinstance(control, _NativeSelectionBufferControl)

    monkeypatch.setattr(BufferControl, "mouse_handler", lambda _self, _event: None)
    with set_app(app._app):
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=4))

    assert copied == ["copy"]
    assert app._app.clipboard.get_data().text == "copy"
    assert app.input_buffer.text == "copy this"
    assert app.input_buffer.selection_state is not None


@pytest.mark.asyncio
async def test_fullscreen_input_selection_ctrl_c_clears_and_ctrl_x_cuts(monkeypatch) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.selection import SelectionState

    copied: list[str] = []
    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None

        async def copy_locally(text: str) -> _ClipboardResult:
            copied.append(text)
            return _ClipboardResult(True, "test")

        monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
        app.input_buffer.text = "copy and cut"
        app.input_buffer.cursor_position = 4
        app.input_buffer.selection_state = SelectionState(original_cursor_position=0)
        await harness.press("c-c")
        await harness.wait_input_equals("")

        assert copied == []
        assert app.input_buffer.selection_state is None
        assert app._ctrl_c_exit_armed is False

        app.input_buffer.text = "copy and cut"
        app.input_buffer.cursor_position = 12
        app.input_buffer.selection_state = SelectionState(original_cursor_position=9)
        await harness.press("c-x")
        await harness.wait_for(lambda: copied == ["cut"])
        if app._clipboard_task is not None:
            await app._clipboard_task

        assert copied == ["cut"]
        assert app.input_buffer.text == "copy and "
        assert app.input_buffer.selection_state is None
        assert app._app.clipboard.get_data().text == "cut"


@pytest.mark.asyncio
@pytest.mark.parametrize("display_mode", [DisplayMode.NATIVE, DisplayMode.NATIVE_REPLAY])
async def test_native_input_selection_ctrl_c_clears_without_copying(
    monkeypatch, display_mode: DisplayMode
) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.selection import SelectionState

    copied: list[str] = []
    async with TUIHarness(display_mode=display_mode) as harness:
        app = harness.app
        assert app is not None

        async def copy_locally(text: str) -> _ClipboardResult:
            copied.append(text)
            return _ClipboardResult(True, "test")

        monkeypatch.setattr(app, "_copy_to_local_clipboard_async", copy_locally)
        app.input_buffer.text = "copy safely"
        app.input_buffer.cursor_position = 4
        app.input_buffer.selection_state = SelectionState(original_cursor_position=0)
        await harness.press("c-c")
        await harness.wait_input_equals("")

        assert copied == []
        assert app.input_buffer.selection_state is None
        assert app._ctrl_c_exit_armed is False


@pytest.mark.asyncio
async def test_fullscreen_cut_retains_recoverable_copy_when_system_clipboard_fails(
    monkeypatch,
) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.selection import SelectionState

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None

        async def fail_copy(_text: str) -> _ClipboardResult:
            return _ClipboardResult(False, reason="test failure")

        monkeypatch.setattr(app, "_copy_to_local_clipboard_async", fail_copy)
        app.input_buffer.text = "keep recoverable"
        app.input_buffer.cursor_position = len(app.input_buffer.text)
        app.input_buffer.selection_state = SelectionState(original_cursor_position=5)
        await harness.press("c-x")
        await harness.wait_for(lambda: app.input_buffer.text == "keep ")
        if app._clipboard_task is not None:
            await app._clipboard_task

        assert "Copy failed" in app._transient_status_text
        clipboard_data = app._app.clipboard.get_data()
        assert clipboard_data.text == "recoverable"
        app.input_buffer.paste_clipboard_data(clipboard_data)
        assert app.input_buffer.text == "keep recoverable"


@pytest.mark.asyncio
async def test_fullscreen_alt_drag_is_not_consumed_by_composer_or_completion_menu() -> None:
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    app.input_buffer.text = "keep me"
    app.input_buffer.cursor_position = len(app.input_buffer.text)
    input_control = app._input_window.content
    input_control.create_content(20, 1)

    with set_app(app._app):
        assert (
            input_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0, alt=True))
            is NotImplemented
        )

    assert app.input_buffer.cursor_position == len(app.input_buffer.text)
    assert app.input_buffer.selection_state is None

    completion_control = app._main_container.children[-1].content.content
    assert (
        completion_control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=0, y=0, alt=True))
        is NotImplemented
    )


def test_fullscreen_alt_mouse_is_not_consumed_by_subview_control() -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    assert (
        app._subview_control.mouse_handler(
            _mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0, alt=True)
        )
        is NotImplemented
    )


@pytest.mark.asyncio
async def test_f6_native_selection_fallback_is_available_while_subview_is_open() -> None:
    class View:
        mouse_support = True

        def on_open(self) -> None:
            pass

        def on_close(self) -> None:
            pass

        def render(self, _width: int, _height: int) -> str:
            return "temporary view"

        def handle_key(self, key: str, _value: str | None = None) -> str:
            return "close" if key == "quit" else "ignored"

    async with TUIHarness(display_mode=DisplayMode.FULLSCREEN) as harness:
        app = harness.app
        assert app is not None
        opened = asyncio.create_task(app.open_subview(View()))  # type: ignore[arg-type]
        await harness.wait_for(lambda: app.active_subview is not None)
        assert bool(app._app.mouse_support()) is True

        await harness.press("f6")
        await harness.wait_for(lambda: not bool(app._app.mouse_support()))
        assert "Native terminal selection enabled" in app._command_status_text

        await harness.press("f6")
        await harness.wait_for(lambda: bool(app._app.mouse_support()))
        await harness.press("q")
        await asyncio.wait_for(opened, timeout=1)


@pytest.mark.asyncio
async def test_fullscreen_input_buffer_mouse_drag_selects_text() -> None:
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    control = app._input_window.content
    assert isinstance(control, BufferControl)
    app.input_buffer.text = "select me"
    control.create_content(20, 1)

    with set_app(app._app):
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=0))
        # Two display cells are occupied by the visible ``❯ `` prompt.
        control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=8, y=0))

    _document, clipboard_data = app.input_buffer.document.cut_selection()
    assert clipboard_data.text == "select"


def test_append_projectability_bookkeeping_does_not_scan_retained_records() -> None:
    """Appending remains O(1) in retained record count."""
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    class AppendOnlyProbe(list):
        def __iter__(self):
            raise AssertionError("append scanned retained transcript records")

    model = FullscreenTranscriptModel()
    model.append("first")
    model._records = AppendOnlyProbe(model._records)

    model.append("second")

    assert model._projectable_record_count == 2


def test_prefix_eviction_preserves_surviving_anchor_and_selection() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    for record_id, text in enumerate(("old", "middle", "new"), start=10):
        model.append(text, record_id=record_id)
    model.jump_to_start(width=80)
    model.scroll_visual_lines(1, width=80, height=1)
    assert model.viewport.anchor is not None
    assert model.viewport.anchor.record_id == 11
    model.begin_selection(x=0, y=0, width=80, height=1)
    model.update_selection(x=3, y=0, width=80, height=1)
    assert model.selected_text()

    model.evict_prefix(1)

    assert model.text == "middle\nnew"
    assert not model.viewport.follows_tail
    assert model.viewport.anchor is not None
    assert model.viewport.anchor.record_id == 11
    assert model.selected_text() == "midd"


def test_prefix_eviction_clears_selection_when_an_endpoint_is_evicted() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("old", record_id=1)
    model.append("new", record_id=2)
    model.jump_to_start(width=80)
    model.begin_selection(x=0, y=0, width=80, height=2)
    model.update_selection(x=2, y=1, width=80, height=2)
    assert model.selected_text() == "old\nnew"

    model.evict_prefix(1)

    assert model.selected_text() == ""


def test_prefix_eviction_of_viewport_anchor_falls_back_to_tail() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    model = FullscreenTranscriptModel()
    model.append("old", record_id=1)
    model.append("new", record_id=2)
    model.jump_to_start(width=80)

    model.evict_prefix(1)

    assert model.text == "new"
    assert model.viewport.follows_tail
    assert model.viewport.anchor is None


def test_fullscreen_retention_bounds_source_and_model_transactionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nooa_cli.tui.tui_application as tui_application

    monkeypatch.setattr(tui_application, "_FULLSCREEN_TRANSCRIPT_MAX_RECORDS", 3)
    monkeypatch.setattr(tui_application, "_FULLSCREEN_TRANSCRIPT_MAX_BYTES", 80)
    app = _make_fullscreen_app()

    app.emit_block("aa")
    app.emit_block("bbb")
    app.emit_block("cccc")

    assert [block.source for block in app._transcript_blocks] == ["bbb", "cccc"]
    assert app._fullscreen_transcript_bytes == sum(
        block.resident_bytes for block in app._transcript_blocks
    )
    assert app._fullscreen_transcript_bytes == 70
    assert app._fullscreen_transcript.text == "bbb\ncccc\n"


def test_fullscreen_resize_replay_expansion_is_charged_and_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nooa_cli.tui.tui_application as tui_application

    monkeypatch.setattr(tui_application, "_FULLSCREEN_TRANSCRIPT_MAX_BYTES", 120)
    app = _make_fullscreen_app()
    app.emit_block("small", replay=lambda: "expanded-" * 20)
    assert len(app._transcript_blocks) == 1

    app._rebuild_fullscreen_transcript()

    assert app._transcript_blocks == []
    assert app._fullscreen_transcript_bytes == 0
    assert app._fullscreen_transcript.text == ""


def test_fullscreen_retention_hard_byte_cap_can_evict_oversized_newest_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nooa_cli.tui.tui_application as tui_application

    monkeypatch.setattr(tui_application, "_FULLSCREEN_TRANSCRIPT_MAX_RECORDS", 3)
    monkeypatch.setattr(tui_application, "_FULLSCREEN_TRANSCRIPT_MAX_BYTES", 40)
    app = _make_fullscreen_app()

    app.emit_block("five!")

    assert app._transcript_blocks == []
    assert app._fullscreen_transcript_bytes == 0
    assert app._fullscreen_transcript.text == ""


def _copyable_markdown_ansi(markdown: str, *, width: int = 60):
    from io import StringIO

    from nooa_cli.tui.copyable_markdown import CopyableMarkdown
    from nooa_cli.tui.theme import CATPPUCCIN_THEME
    from rich.console import Console

    renderable = CopyableMarkdown(markdown)
    buffer = StringIO()
    Console(
        file=buffer,
        force_terminal=True,
        color_system="256",
        width=width,
        height=1,
        theme=CATPPUCCIN_THEME,
    ).print(renderable, soft_wrap=True)
    return renderable, buffer.getvalue()


def _copy_label_position(model, *, width: int, height: int) -> tuple[int, int]:
    formatted = model.formatted_text(width=width, height=height)
    lines = "".join(text for _style, text, *_rest in formatted).splitlines()
    for y, line in enumerate(lines):
        if "Copy" in line:
            return line.index("Copy"), y
    raise AssertionError("copy label was not rendered")


def test_copyable_markdown_preserves_long_list_item_for_fullscreen_reflow() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    suffix = "the complete suffix remains visible after an intelligent wrap"
    markdown = (
        "- **[#205 — Trace Explorer](https://example.test/issues/205)** — "
        "a long explanation that reaches beyond the initial Rich console width; " + suffix
    )
    _renderable, ansi = _copyable_markdown_ansi(markdown, width=60)

    plain = strip_safe_ansi(ansi)
    assert suffix in plain

    model = FullscreenTranscriptModel()
    model.append(ansi)
    rows = [row for row in _projected_row_texts(model, 40) if row]

    assert "".join(rows) == plain.strip("\n")
    assert all(not row.endswith(("explan", "intellig")) for row in rows[:-1])
    assert any(row.endswith(" ") for row in rows[:-1])


@pytest.mark.parametrize(
    "markup",
    [
        "ordinary prose {suffix}",
        "## heading text {suffix}",
        "- bullet text {suffix}",
        "1. numbered text {suffix}",
        "> quoted text {suffix}",
        "| key | value |\n| --- | --- |\n| row | table text {suffix} |",
    ],
)
def test_copyable_markdown_preserves_long_structural_text(markup: str) -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    suffix = "finalword"
    _renderable, ansi = _copyable_markdown_ansi(
        markup.format(suffix=("word " * 12) + suffix),
        width=24,
    )

    assert suffix in strip_safe_ansi(ansi)


def test_copyable_markdown_wraps_complete_block_quotes_without_cropping() -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi
    from rich.cells import cell_len

    sentence = (
        "Before changing anything, inspect the relevant context and applicable "
        "repository instructions. Preserve unrelated work. Verify outcomes."
    )
    _renderable, ansi = _copyable_markdown_ansi(f"> {sentence}", width=30)
    lines = [line for line in strip_safe_ansi(ansi).splitlines() if line]

    assert "".join(line.removeprefix("▌ ").rstrip() + " " for line in lines).split() == (
        sentence.split()
    )
    assert len(lines) > 1
    assert all(line.startswith("▌ ") for line in lines)
    assert all(cell_len(line) <= 30 for line in lines)


def test_copyable_markdown_preserves_a_long_list_nested_in_a_quote() -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    suffix = "complete-end"
    _renderable, ansi = _copyable_markdown_ansi(
        "> - " + "word " * 20 + suffix,
        width=20,
    )
    plain = strip_safe_ansi(ansi)

    assert suffix in plain
    assert all(line.startswith("▌ ") for line in plain.splitlines() if line)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 10, 21])
def test_copyable_markdown_keeps_quotes_within_pathological_widths(width: int) -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi
    from rich.cells import cell_len

    _renderable, ansi = _copyable_markdown_ansi("> - alpha beta END", width=width)
    plain = strip_safe_ansi(ansi)

    lines = plain.splitlines()
    content = "".join(line.removeprefix("▌ ") for line in lines)
    assert "".join(content.split()) == "•alphabetaEND"
    assert all(cell_len(line) <= width for line in lines)


@pytest.mark.parametrize(
    "markup",
    [
        "> plain prose alpha beta END",
        "> | heading |\n> | --- |\n> | cell-END |",
        "> ```text\n> code-END\n> ```",
    ],
)
@pytest.mark.parametrize("width", [1, 2, 3, 4, 10])
def test_copyable_markdown_preserves_nested_quote_content_at_narrow_widths(
    markup: str, width: int
) -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi
    from rich.cells import cell_len

    _renderable, ansi = _copyable_markdown_ansi(markup, width=width)
    plain = strip_safe_ansi(ansi)

    content = "".join(line.removeprefix("▌ ") for line in plain.splitlines())
    assert "END" in content
    assert all(cell_len(line) <= width for line in plain.splitlines())


def test_copyable_markdown_preserves_safe_markdown_links() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from nooa_cli.tui.terminal_safety import safe_hyperlink_spans, strip_safe_ansi

    _renderable, ansi = _copyable_markdown_ansi(
        "Read the [documentation](https://example.test/docs) for details.", width=60
    )

    plain = strip_safe_ansi(ansi)
    label_start = plain.index("documentation")
    assert safe_hyperlink_spans(ansi) == (
        (label_start, label_start + len("documentation"), "https://example.test/docs"),
    )

    model = FullscreenTranscriptModel()
    model.append(ansi)
    assert model.hyperlink_at(x=label_start, y=0, width=60, height=2) == (
        "https://example.test/docs"
    )


def test_copyable_markdown_renders_and_hit_tests_file_links() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from nooa_cli.tui.terminal_safety import safe_hyperlink_spans, strip_safe_ansi

    _renderable, ansi = _copyable_markdown_ansi(
        "Open [the file](file://path/to/file) locally.", width=60
    )

    plain = strip_safe_ansi(ansi)
    assert "[the file](file://path/to/file)" not in plain
    label_start = plain.index("the file")
    assert safe_hyperlink_spans(ansi) == (
        (label_start, label_start + len("the file"), "file:///path/to/file"),
    )

    model = FullscreenTranscriptModel()
    model.append(ansi)
    assert model.hyperlink_at(x=label_start, y=0, width=60, height=2) == ("file:///path/to/file")


def test_copyable_markdown_preserves_nested_and_multiline_list_structure() -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    _renderable, ansi = _copyable_markdown_ansi(
        "- parent item\n"
        "  - nested item one\n"
        "  - nested item two\n"
        "- first paragraph\n\n"
        "  second paragraph in the same item\n\n"
        "  > quoted child\n"
        "98. ninety eight\n"
        "99. ninety nine\n"
        "100. one hundred",
        width=30,
    )

    lines = [line.rstrip() for line in strip_safe_ansi(ansi).splitlines() if line.strip()]
    assert lines == [
        " • parent item",
        "    • nested item one",
        "    • nested item two",
        " • first paragraph",
        "   second paragraph in the same item",
        "   ▌ quoted child",
        "  98 ninety eight",
        "  99 ninety nine",
        " 100 one hundred",
    ]


def test_copyable_markdown_exposes_exact_fenced_code_payloads() -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    renderable, ansi = _copyable_markdown_ansi(
        "before\n\n```python title=demo\nprint('one')\n```\n\n```\n  exact spacing  \n```"
    )

    plain = strip_safe_ansi(ansi)
    assert "Copy" in ansi
    assert "\x1b[" in ansi
    assert " print('one')" in plain
    assert " " * 60 in plain
    assert list(renderable.copy_actions.values()) == ["print('one')", "  exact spacing  "]


def test_copyable_markdown_places_copy_action_in_top_code_padding() -> None:
    from nooa_cli.tui.terminal_safety import sanitize_transcript_ansi, strip_safe_ansi
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    _renderable, ansi = _copyable_markdown_ansi("```python\nprint('one')\n```", width=30)
    lines = strip_safe_ansi(ansi).splitlines()

    assert lines == [
        "python · Copy ".rjust(30),
        " print('one')".ljust(30),
        " " * 30,
    ]
    fragments = to_formatted_text(ANSI(sanitize_transcript_ansi(ansi)))
    copy_style = next(style for style, text, *_rest in fragments if text == "C" and "bg:" in style)
    code_style = next(style for style, text, *_rest in fragments if text == "p" and "bg:" in style)
    assert copy_style.split("bg:", 1)[1].split()[0] == code_style.split("bg:", 1)[1].split()[0]


@pytest.mark.parametrize(
    ("markdown", "width"),
    [
        ("```python\nx\n```", 8),
        ("```averyverylonglexername\nx\n```", 12),
    ],
)
def test_copyable_markdown_keeps_complete_copy_label_at_narrow_width(
    markdown: str, width: int
) -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    _renderable, ansi = _copyable_markdown_ansi(markdown, width=width)

    assert "Copy" in strip_safe_ansi(ansi).splitlines()[0]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_copyable_markdown_omits_ambiguous_copy_suffix_at_tiny_width(width: int) -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    _renderable, ansi = _copyable_markdown_ansi("```text\nx\n```", width=width)
    first_line = strip_safe_ansi(ansi).splitlines()[0]

    assert not any(label in first_line for label in ("y", "py", "opy"))
    assert "nooa-copy://" not in ansi


def test_copyable_markdown_keeps_full_copy_label_at_minimum_action_width() -> None:
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    _renderable, ansi = _copyable_markdown_ansi("```text\nx\n```", width=4)

    assert strip_safe_ansi(ansi).splitlines()[0] == "Copy"
    assert "nooa-copy://" in ansi


def test_copyable_markdown_does_not_offer_copy_for_empty_fence() -> None:
    renderable, ansi = _copyable_markdown_ansi("```python\n```")

    assert renderable.copy_actions == {}
    assert "Copy" not in ansi


def test_model_authored_internal_link_cannot_forge_code_copy_action() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    renderable, ansi = _copyable_markdown_ansi(
        "[Copy](nooa-copy://code-0)\n\n```python\nSECRET\n```", width=60
    )
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    formatted = model.formatted_text(width=60, height=20)
    lines = "".join(text for _style, text, *_rest in formatted).splitlines()
    copy_positions = [(line.index("Copy"), y) for y, line in enumerate(lines) if "Copy" in line]

    assert len(copy_positions) == 2
    assert (
        model.copy_action_at(x=copy_positions[0][0], y=copy_positions[0][1], width=60, height=20)
        is None
    )
    assert (
        model.copy_action_at(x=copy_positions[1][0], y=copy_positions[1][1], width=60, height=20)
        == "SECRET"
    )


def test_identical_code_blocks_keep_distinct_copy_targets() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    renderable, ansi = _copyable_markdown_ansi(
        "```python\nsame()\n```\n\n```python\nsame()\n```", width=60
    )
    assert len(renderable.copy_actions) == 2
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    formatted = model.formatted_text(width=60, height=30)
    lines = "".join(text for _style, text, *_rest in formatted).splitlines()
    positions = [(line.index("Copy"), y) for y, line in enumerate(lines) if "Copy" in line]

    assert len(positions) == 2
    assert [model.copy_action_at(x=x, y=y, width=60, height=30) for x, y in positions] == [
        "same()",
        "same()",
    ]


def test_fullscreen_drag_copy_projects_decorated_code_to_exact_source() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    source = "\tif ready:\n\t\tprint('one')  "
    renderable, ansi = _copyable_markdown_ansi(
        f"Before\n\n```python\n{source}\n```\n\nAfter", width=60
    )
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=60, height=20)
    ).splitlines()
    header_y = next(index for index, line in enumerate(lines) if "Copy" in line)
    after_y = next(index for index, line in enumerate(lines) if "After" in line)

    # Drag over the complete visual panel, including its Copy header, gutter,
    # full-width background fill, and blank padding rows.
    model.begin_selection(x=0, y=header_y, width=60, height=20)
    model.update_selection(x=59, y=after_y - 1, width=60, height=20)

    assert model.selected_text() == source


def test_fullscreen_code_selection_exposes_controls_but_copies_original_source() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from nooa_cli.tui.terminal_safety import safe_hyperlink_spans, strip_safe_ansi

    source = "a\x1b[31mb\x1b]8;;https://evil.test\x1b\\c"
    renderable, ansi = _copyable_markdown_ansi(f"```text\n{source}\n```", width=160)
    plain = strip_safe_ansi(ansi)

    assert r"a\x1b[31mb\x1b]8;;https://evil.test\x1b\c" in plain
    assert safe_hyperlink_spans(ansi) == ()

    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=160, height=20)
    ).splitlines()
    header_y = next(index for index, line in enumerate(lines) if "Copy" in line)
    last_panel_y = max(index for index, line in enumerate(lines) if line and not line.isspace()) + 1
    model.begin_selection(x=0, y=header_y, width=160, height=20)
    model.update_selection(x=159, y=last_panel_y, width=160, height=20)

    assert model.selected_text() == source


def test_copyable_markdown_wraps_long_code_without_clipping() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from nooa_cli.tui.terminal_safety import strip_safe_ansi

    source = "x" * 100
    renderable, ansi = _copyable_markdown_ansi(f"```text\n{source}\n```", width=32)

    assert strip_safe_ansi(ansi).count("x") == len(source)

    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=32, height=20)
    ).splitlines()
    header_y = next(index for index, line in enumerate(lines) if "Copy" in line)
    last_panel_y = max(index for index, line in enumerate(lines) if "x" in line) + 1
    model.begin_selection(x=0, y=header_y, width=32, height=20)
    model.update_selection(x=31, y=last_panel_y, width=32, height=20)

    assert model.selected_text() == source


def test_fullscreen_code_selection_preserves_blank_line_endpoints() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    renderable, ansi = _copyable_markdown_ansi("```text\na\n\nb\n```", width=30)
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=30, height=20)
    ).splitlines()
    a_y = next(index for index, line in enumerate(lines) if line.strip() == "a")
    blank_y = a_y + 1
    b_y = a_y + 2

    model.begin_selection(x=1, y=a_y, width=30, height=20)
    model.update_selection(x=1, y=blank_y, width=30, height=20)
    assert model.selected_text() == "a\n\n"

    model.begin_selection(x=1, y=blank_y, width=30, height=20)
    model.update_selection(x=1, y=b_y, width=30, height=20)
    assert model.selected_text() == "\nb"

    model.begin_selection(x=1, y=blank_y, width=30, height=20)
    model.update_selection(x=1, y=blank_y, width=30, height=20)
    assert model.selected_text() == "\n"

    model.begin_selection(x=1, y=b_y, width=30, height=20)
    model.update_selection(x=1, y=blank_y, width=30, height=20)
    assert model.selected_text() == "\nb"


def test_fullscreen_code_selection_preserves_trailing_blank_lines() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    source = "abc\n\n"
    renderable, ansi = _copyable_markdown_ansi(f"```text\n{source}\n```", width=30)
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=30, height=20)
    ).splitlines()
    header_y = next(index for index, line in enumerate(lines) if "Copy" in line)
    last_source_y = max(index for index, line in enumerate(lines) if index > header_y) - 1

    model.begin_selection(x=0, y=header_y, width=30, height=20)
    model.update_selection(x=1, y=last_source_y, width=30, height=20)

    assert model.selected_text() == source


def test_fullscreen_partial_code_selection_excludes_unselected_trailing_spaces() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    renderable, ansi = _copyable_markdown_ansi("```text\nabc   \n```", width=30)
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=30, height=20)
    ).splitlines()
    code_y = next(index for index, line in enumerate(lines) if "abc" in line)

    model.begin_selection(x=3, y=code_y, width=30, height=20)
    model.update_selection(x=3, y=code_y, width=30, height=20)

    assert model.selected_text() == "c"


def test_fullscreen_code_selection_maps_wide_character_tabs_by_cell() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel
    from rich.cells import cell_len

    renderable, ansi = _copyable_markdown_ansi("```text\n界\tX\n```", width=30)
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=30, height=20)
    ).splitlines()
    code_y = next(index for index, line in enumerate(lines) if "界" in line)
    x_position = cell_len(lines[code_y][: lines[code_y].index("X")])

    model.begin_selection(x=x_position, y=code_y, width=30, height=20)
    model.update_selection(x=x_position, y=code_y, width=30, height=20)

    assert model.selected_text() == "X"


def test_fullscreen_drag_copy_source_mapping_survives_resize_replay() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    source = "first_call()\n    second_call()"
    initial, initial_ansi = _copyable_markdown_ansi(f"```python\n{source}\n```", width=60)
    model = FullscreenTranscriptModel()
    model.append(initial_ansi, record_id=7, copy_actions=initial.copy_actions)

    resized, resized_ansi = _copyable_markdown_ansi(f"```python\n{source}\n```", width=32)
    model.replace(
        [resized_ansi],
        record_ids=[7],
        copy_actions=[resized.copy_actions],
    )
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=32, height=20)
    ).splitlines()
    header_y = next(index for index, line in enumerate(lines) if "Copy" in line)
    last_panel_y = max(index for index, line in enumerate(lines) if line and not line.isspace()) + 1

    model.begin_selection(x=0, y=header_y, width=32, height=20)
    model.update_selection(x=31, y=last_panel_y, width=32, height=20)

    assert model.selected_text() == source


def test_fullscreen_drag_copy_mixes_prose_and_semantic_code() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    renderable, ansi = _copyable_markdown_ansi(
        'Before\n\n```python\nif ready:\n    print("one")\n```\n\nAfter', width=60
    )
    model = FullscreenTranscriptModel()
    model.append(ansi, copy_actions=renderable.copy_actions)
    lines = "".join(
        fragment[1] for fragment in model.formatted_text(width=60, height=20)
    ).splitlines()
    before_y = next(index for index, line in enumerate(lines) if "Before" in line)
    after_y = next(index for index, line in enumerate(lines) if "After" in line)

    model.begin_selection(x=0, y=before_y, width=60, height=20)
    model.update_selection(x=len("After") - 1, y=after_y, width=60, height=20)

    assert model.selected_text() == 'Before\n\nif ready:\n    print("one")\n\nAfter'


def test_fullscreen_copy_action_hit_testing_survives_resize() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    renderable, ansi = _copyable_markdown_ansi("```python\nprint('exact')\n```", width=60)
    model = FullscreenTranscriptModel()
    model.append(ansi, record_id=7, copy_actions=renderable.copy_actions)

    x, y = _copy_label_position(model, width=60, height=20)
    assert model.copy_action_at(x=x, y=y, width=60, height=20) == "print('exact')"
    assert model.copy_action_at(x=max(0, x - 1), y=y, width=60, height=20) is None

    renderable, resized = _copyable_markdown_ansi("```python\nprint('exact')\n```", width=42)
    model.replace(
        [resized],
        record_ids=[7],
        copy_actions=[renderable.copy_actions],
    )
    x, y = _copy_label_position(model, width=42, height=20)
    assert model.copy_action_at(x=x, y=y, width=42, height=20) == "print('exact')"


def test_fullscreen_code_copy_click_copies_source_without_starting_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.tui_application import _ClipboardResult
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    renderable, ansi = _copyable_markdown_ansi("```python\nprint('exact')\n```", width=60)
    app.emit_block(ansi, code_copy_actions=renderable.copy_actions)
    app._transcript_viewport_size = lambda: (60, 20)
    copied: list[str] = []
    monkeypatch.setattr(
        app,
        "_copy_to_clipboard_result",
        lambda text: copied.append(text) or _ClipboardResult(True, "test"),
    )
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(60, 20)
    x, y = _copy_label_position(app._fullscreen_transcript, width=60, height=20)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=x, y=y))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=x, y=y))

    assert copied == ["print('exact')"]
    assert app._fullscreen_transcript.selected_text() == ""
    assert app._transient_status_text == "Copied 14 characters"


def test_fullscreen_drag_over_decorated_code_copies_exact_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    source = "\tif ready:\n\t\tprint('one')  "
    renderable, ansi = _copyable_markdown_ansi(f"```python\n{source}\n```", width=60)
    app.emit_block(ansi, code_copy_actions=renderable.copy_actions)
    app._transcript_viewport_size = lambda: (60, 20)
    copied: list[str] = []
    monkeypatch.setattr(app, "_start_fullscreen_selection_copy", copied.append)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(60, 20)
    lines = "".join(
        fragment[1] for fragment in app._fullscreen_transcript.formatted_text(width=60, height=20)
    ).splitlines()
    header_y = next(index for index, line in enumerate(lines) if "Copy" in line)
    last_panel_y = max(index for index, line in enumerate(lines) if line and not line.isspace()) + 1

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=0, y=header_y))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=59, y=last_panel_y))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=59, y=last_panel_y))

    assert copied == [source]


def test_model_authored_code_source_link_cannot_change_drag_copy() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    forged = "\x1b]8;;nooa-code-source://forged/0\x1b\\SECRET\x1b]8;;\x1b\\"
    model = FullscreenTranscriptModel()
    model.append(forged, copy_actions={"real": "safe()"})

    model.begin_selection(x=0, y=0, width=20, height=1)
    model.update_selection(x=5, y=0, width=20, height=1)

    assert model.selected_text() == "SECRET"


def test_code_copy_metadata_tracks_prefix_eviction() -> None:
    from nooa_cli.tui.fullscreen_transcript import FullscreenTranscriptModel

    first, first_ansi = _copyable_markdown_ansi("```python\nfirst()\n```", width=60)
    second, second_ansi = _copyable_markdown_ansi("```python\nsecond()\n```", width=60)
    model = FullscreenTranscriptModel()
    model.append(first_ansi, record_id=1, copy_actions=first.copy_actions)
    model.append(second_ansi, record_id=2, copy_actions=second.copy_actions)
    model.evict_prefix(1)

    x, y = _copy_label_position(model, width=60, height=20)
    assert model.copy_action_at(x=x, y=y, width=60, height=20) == "second()"


def test_fullscreen_code_copy_drag_away_does_not_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prompt_toolkit.mouse_events import MouseEventType

    app = _make_fullscreen_app()
    renderable, ansi = _copyable_markdown_ansi("```python\nprint('exact')\n```", width=60)
    app.emit_block(ansi, code_copy_actions=renderable.copy_actions)
    app._transcript_viewport_size = lambda: (60, 20)
    copied: list[str] = []
    monkeypatch.setattr(app, "_start_fullscreen_selection_copy", copied.append)
    assert app._output_window is not None
    control = app._output_window.content
    control.create_content(60, 20)
    x, y = _copy_label_position(app._fullscreen_transcript, width=60, height=20)

    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_DOWN, x=x, y=y))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_MOVE, x=0, y=y))
    control.mouse_handler(_mouse_event(MouseEventType.MOUSE_UP, x=x, y=y))

    assert copied == []


async def test_fullscreen_theme_refresh_recolors_mixed_retained_history_for_all_themes() -> None:
    from types import SimpleNamespace

    from nooa_cli.tui import theme
    from nooa_cli.tui.frontend import TerminalFrontend
    from nooa_cli.tui.output import (
        AgentMessage,
        HelpOutput,
        HistoryReplay,
        HistoryTurn,
        StartupInfo,
        TableOutput,
        TextOutput,
    )
    from nooa_cli.tui.session import _EmitStream
    from rich.console import Console

    original_theme = theme.get_theme()
    try:
        theme.set_theme("mocha")
        app = _make_fullscreen_app()
        frontend = TerminalFrontend(SimpleNamespace(tui=SimpleNamespace(theme="mocha")))
        frontend.bind_app(app)
        stream = _EmitStream(
            app.emit_block,
            replay_width=lambda: 72,
            layout_width=lambda: 72,
            supports_code_copy_actions=True,
        )
        frontend._console.replace_console(
            Console(
                file=stream,
                force_terminal=True,
                color_system="256",
                width=72,
                theme=theme.create_theme(),
            )
        )
        app.emit_block("raw retained output\n", event_id="raw", tags={"raw"}, keep=True)
        outputs = [
            *(
                TextOutput(f"retained {level}", level)
                for level in ("info", "error", "warning", "success", "status")
            ),
            TableOutput(["kind", "value"], [["theme", "live"]], title="retained table"),
            HelpOutput({"/theme": "Switch theme"}),
            AgentMessage("Retained prose with `inline code` and:\n\n```python\nprint('live')\n```"),
            StartupInfo(
                model="provider/model",
                short_model="model",
                working_dir="/work",
                vi_mode=False,
            ),
            HistoryReplay(
                turns=[HistoryTurn(role="agent", content="Earlier retained response with `code`")],
                session_id="theme-history",
                show_header=True,
                show_footer=True,
            ),
        ]
        with frontend.batch_render():
            for output in outputs:
                await frontend.render(output)

        assert len(app._transcript_blocks) == 3
        raw_block, batched_block, history_block = app._transcript_blocks
        assert raw_block.replay is None
        assert raw_block.event_id == "raw"
        assert raw_block.tags == frozenset({"raw"})
        assert raw_block.keep is True
        assert batched_block.replay is not None
        assert history_block.replay is not None
        record_ids = [block.transcript_record_id for block in app._transcript_blocks]
        copy_actions = [dict(block.code_copy_actions) for block in app._transcript_blocks]
        raw_rendered = raw_block.fullscreen_rendered
        rendered_by_theme = {}
        for name in theme.THEMES:
            theme.set_theme(name)
            frontend.refresh_theme()
            rendered_by_theme[name] = "".join(
                block.fullscreen_rendered or "" for block in app._transcript_blocks
            )
            plain = app._fullscreen_transcript.text
            assert raw_block.fullscreen_rendered == raw_rendered
            assert [block.transcript_record_id for block in app._transcript_blocks] == record_ids
            assert [block.code_copy_actions for block in app._transcript_blocks] == copy_actions
            for expected in (
                "raw retained output",
                *(
                    f"retained {level}"
                    for level in ("info", "error", "warning", "success", "status")
                ),
                "retained table",
                "/theme",
                "Retained prose",
                "NOOA ready",
                "Earlier retained response",
            ):
                assert plain.count(expected) == 1

        assert len(set(rendered_by_theme.values())) == len(theme.THEMES)
    finally:
        theme.set_theme(original_theme)


def test_fullscreen_theme_refresh_recolors_retained_agent_event_syntax() -> None:
    from nooa_cli.tui import theme
    from nooa_cli.tui.session import Session
    from nooa_cli.tui.theme import ThemeSyntax

    original_theme = theme.get_theme()
    try:
        theme.set_theme("mocha")
        app = _make_fullscreen_app()
        session = Session.__new__(Session)
        session._app = app
        syntax = ThemeSyntax(
            "def greet(name: str) -> str:\n    return name",
            "python",
            background_color="default",
        )
        session._emit_text(syntax)
        before = app._transcript_blocks[0].fullscreen_rendered
        assert before is not None

        theme.set_theme("vslight")
        app.refresh_style()
        after = app._transcript_blocks[0].fullscreen_rendered

        assert after is not None
        assert after != before
        assert app._fullscreen_transcript.text.count("def greet") == 1
    finally:
        theme.set_theme(original_theme)


@pytest.mark.asyncio
async def test_native_replay_theme_refresh_requests_same_width_replay() -> None:
    from nooa_cli.tui.tui_application import (
        TranscriptBlock,
        TUIApplication,
        _ResizeReplayQueueItem,
    )

    from .tui_app_harness import MutableRecordingOutput

    output = MutableRecordingOutput(columns=80, rows=24)
    with create_app_session(input=DummyInput(), output=output):
        app = TUIApplication(display_mode=DisplayMode.NATIVE_REPLAY)
    app._loop = asyncio.get_running_loop()
    app._block_queue = asyncio.Queue()
    app._resize_replays_enabled = True
    app._resize_reflow.observe((80, 24))
    app.emit_block("old theme\n", replay=lambda: "new theme\n")

    app.refresh_style()

    assert app._resize_reflow.replay_required is True
    assert app._resize_reflow.pending_width == 80
    await asyncio.sleep(0.1)
    queued = app._block_queue.get_nowait()
    assert isinstance(queued, TranscriptBlock)
    queued = app._block_queue.get_nowait()
    assert isinstance(queued, _ResizeReplayQueueItem)
    assert queued.request.required is True
    assert queued.request.width == 80
    app._cancel_resize_replay_work()
