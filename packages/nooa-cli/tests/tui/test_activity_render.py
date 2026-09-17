# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /activity rendering: no flicker (batched output), no empty header band."""

import asyncio

import pytest
from nooa_cli.tui.session import _EmitStream


def test_emit_stream_hold_coalesces_into_one_block():
    """Multiple flush()es inside hold() emit a single block on release."""
    blocks: list[str] = []
    stream = _EmitStream(blocks.append)

    def render(text: str) -> None:
        stream.write(text)
        stream.flush()

    # Without hold: one block per render.
    render("a\n")
    render("b\n")
    assert blocks == ["a\n", "b\n"]

    blocks.clear()
    with stream.hold():
        render("table\n")
        render("code\n")
        render("footer\n")
    assert blocks == ["table\ncode\nfooter\n"]


def test_emit_stream_hold_is_reentrant():
    """Nested holds only release on the outermost exit."""
    blocks: list[str] = []
    stream = _EmitStream(blocks.append)
    with stream.hold():
        stream.write("x")
        stream.flush()
        with stream.hold():
            stream.write("y")
            stream.flush()
        assert blocks == []  # inner exit must not flush
    assert blocks == ["xy"]


def test_table_output_show_header_false_drops_header():
    """A header-less key/value table renders no empty header band."""
    import io

    from nooa_cli.tui.console import TUIConsole
    from rich.console import Console

    tc = TUIConsole()
    buf = io.StringIO()
    tc.console = Console(file=buf, width=60, force_terminal=False)
    tc.print_table("", ["", ""], [["Phase", "Idle"]], show_header=False)
    out = buf.getvalue()
    # No title line, and the only content row is the data row.
    assert "Agent Activity" not in out
    assert "Phase" in out and "Idle" in out
    # The box should have no header separator (┡/╇) since show_header=False.
    assert "╇" not in out


@pytest.mark.asyncio
async def test_activity_idle_is_one_liner():
    """Idle /activity is a single status line, not a table."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import ActivityCommand
    from nooa_cli.tui.output import TableOutput, TextOutput

    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())

    async def _idle_locate():
        return None

    cmd._locate_agent = _idle_locate  # type: ignore[method-assign]

    result = await cmd.execute([])
    assert len(result.outputs) == 1
    out = result.outputs[0]
    assert isinstance(out, TextOutput)
    assert not isinstance(out, TableOutput)


@pytest.mark.asyncio
async def test_activity_table_has_no_header():
    """The activity table is emitted header-less (show_header=False, no title)."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import ActivityCommand
    from nooa_cli.tui.output import TableOutput

    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())

    async def tool_layer():
        await asyncio.sleep(5)

    async def handle():
        await tool_layer()

    task = asyncio.ensure_future(handle())
    await asyncio.sleep(0.02)
    try:
        result = await cmd.execute([])
        table = next(o for o in result.outputs if isinstance(o, TableOutput))
        assert table.show_header is False
        assert table.title == ""
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_batch_render_ctx_tolerates_mock_frontend():
    """_batch_render_ctx falls back to a no-op for non-context batch_render().

    Regression: ``with self.frontend.batch_render()`` raised ``TypeError:
    'coroutine' object does not support the context manager protocol`` when the
    frontend was an AsyncMock (its ``batch_render()`` returns a coroutine). The
    helper must return an enter/exit-able context in every case.
    """
    from unittest.mock import AsyncMock

    from nooa_cli.tui.commands import _batch_render_ctx

    # AsyncMock.batch_render() returns a coroutine — must not be entered.
    ctx = _batch_render_ctx(AsyncMock())
    with ctx:
        pass  # no raise

    # A frontend without batch_render at all → still a usable context.
    class _NoBatch:
        pass

    with _batch_render_ctx(_NoBatch()):
        pass


def test_batch_render_ctx_uses_real_context():
    """When batch_render returns a real context manager, the helper passes it through."""
    from contextlib import contextmanager

    entered = []

    @contextmanager
    def _real():
        entered.append("enter")
        yield
        entered.append("exit")

    class _Frontend:
        def batch_render(self):
            return _real()

    from nooa_cli.tui.commands import _batch_render_ctx

    with _batch_render_ctx(_Frontend()):
        pass
    assert entered == ["enter", "exit"]


def test_activity_overlay_renders_table_code_and_scrolls():
    from nooa_cli.tui.activity_overlay import ActivityOverlayView, render_activity_overlay
    from nooa_cli.tui.output import CodeExecution, TableOutput

    view = ActivityOverlayView(
        [
            TableOutput(
                columns=["", ""],
                rows=[["Phase", "Executing Python"], ["  python (1.2s)", "print('hello')"]],
                footer="In a code cell — not waiting on the model.",
                show_header=False,
            ),
            CodeExecution(
                tool_call_id="activity:test.py:2",
                code="line1\nline2\nline3",
                start_line=1,
                highlight_line=2,
            ),
        ]
    )

    rendered = render_activity_overlay(view, width=80, height=10, ansi=False)

    assert "Activity" in rendered
    assert "Executing Python" in rendered
    assert "print('hello')" in rendered
    assert "Enter/Esc/q close" in rendered
    assert view.handle_key("down") == "handled"
    assert view.offset >= 0
    assert view.handle_key("quit") == "close"


@pytest.mark.asyncio
async def test_tui_app_opens_and_closes_activity_overlay() -> None:
    from nooa_cli.tui.output import TextOutput

    from .tui_app_harness import TUIHarness

    async with TUIHarness() as h:
        task = asyncio.create_task(h.app.open_activity_overlay([TextOutput("Agent idle")]))
        await h.wait_for(lambda: h.app.active_subview is not None)

        rendered = h.app.active_subview.render(80, 10)
        assert "Activity" in rendered
        assert "Agent idle" in rendered

        await h.press("q")
        await asyncio.wait_for(task, timeout=1)
        assert h.app.active_subview is None
