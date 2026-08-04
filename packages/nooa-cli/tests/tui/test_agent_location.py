# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the agent-location probe and its use in /activity."""

import asyncio
import linecache

import pytest
from nooa_cli.tui.agent_location import (
    AgentLocation,
    FrameInfo,
    _source_context,
    locate_agent_on_loop,
)
from nooa_cli.tui.commands import ActivityCommand
from nooa_cli.tui.output import CodeExecution, TableOutput


@pytest.mark.asyncio
async def test_locate_walks_full_await_chain():
    """The probe follows cr_await down to the actual suspend point."""

    async def tool_layer():
        await asyncio.sleep(5)

    async def cell_layer():
        await tool_layer()

    # The probe matches tasks whose coroutine/name contains the turn method.
    async def handle():  # noqa: ANN202 - name is what the probe matches on
        await cell_layer()

    task = asyncio.ensure_future(handle())
    await asyncio.sleep(0.02)  # let it suspend inside asyncio.sleep
    try:
        loc = locate_agent_on_loop()
        assert isinstance(loc, AgentLocation)
        quals = [f.qualname for f in loc.stack]
        # outermost-first: handle -> cell_layer -> tool_layer -> sleep
        assert any("handle" in q for q in quals)
        assert any("cell_layer" in q for q in quals)
        assert any("tool_layer" in q for q in quals)
        # innermost is the suspend point
        assert loc.innermost is loc.stack[-1]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_locate_descends_into_awaited_task():
    """A directly-awaited child Task is followed, not stopped at."""

    async def child():
        await asyncio.sleep(5)

    async def handle():
        await asyncio.ensure_future(child())

    task = asyncio.ensure_future(handle())
    await asyncio.sleep(0.02)
    try:
        loc = locate_agent_on_loop()
        assert loc is not None
        quals = [f.qualname for f in loc.stack]
        assert any("handle" in q for q in quals)
        assert any("child" in q for q in quals)
    finally:
        task.cancel()
        # handle awaits a child task; cancelling parent cancels the await
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_locate_returns_none_when_no_turn_running():
    """With no matching turn task, the probe returns None (idle)."""
    loc = locate_agent_on_loop(turn_method="__no_such_method__")
    assert loc is None


@pytest.mark.asyncio
async def test_activity_command_includes_suspend_location():
    """/activity surfaces the await stack when a turn is in flight."""
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
        assert result.success
        table = result.outputs[0]
        assert isinstance(table, TableOutput)
        flat = "\n".join(cell for row in table.rows for cell in row)
        assert "Suspended at" in flat
        assert "tool_layer" in flat
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_innermost_frame_has_source_context():
    """The suspend-point frame carries ±3 lines of source (this test file)."""

    async def handle():
        await asyncio.sleep(5)

    task = asyncio.ensure_future(handle())
    await asyncio.sleep(0.02)
    try:
        loc = locate_agent_on_loop()
        assert loc is not None and loc.innermost is not None
        ctx = loc.innermost.context
        # asyncio.sleep is C-adjacent but resolves to the sleep() coroutine in
        # the stdlib, which has real source — so context should be non-empty.
        assert ctx, "expected source context for the suspend frame"
        current = [s for s in ctx if s.is_current]
        assert len(current) == 1
        assert current[0].lineno == loc.innermost.lineno
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_activity_command_emits_highlighted_code(monkeypatch):
    """/activity emits the suspend-point source as a CodeExecution block.

    Simulates a CodeAct/REPL cell by registering fake source in linecache under
    the frame's co_filename and pointing the probe's innermost frame at it. The
    code block is a separate output so the frontend can syntax-highlight it.
    """
    cell_name = "Cell In[42]"
    cell_src = "a = 1\nb = 2\nc = await tool()\nd = 4\ne = 5\n"
    linecache.cache[cell_name] = (
        len(cell_src),
        None,
        cell_src.splitlines(keepends=True),
        cell_name,
    )

    fake = AgentLocation(
        task_name="TUIAgent.handle",
        stack=[FrameInfo("handle", 3, cell_name)],
    )
    fake.stack[-1].context = _source_context(cell_name, 3)

    from unittest.mock import AsyncMock, MagicMock

    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())

    async def _fake_locate():
        return fake

    monkeypatch.setattr(cmd, "_locate_agent", _fake_locate)

    try:
        result = await cmd.execute([])
        # First output is the activity table; second is the code block.
        assert isinstance(result.outputs[0], TableOutput)
        code = next(o for o in result.outputs if isinstance(o, CodeExecution))
        # the snippet covers the suspend line plus context, in source order
        assert "c = await tool()" in code.code
        assert "a = 1" in code.code
        assert code.code.splitlines() == ["a = 1", "b = 2", "c = await tool()", "d = 4", "e = 5"]
        # numbered from the real file offset, with the suspend line highlighted
        assert code.start_line == 1  # context starts at file line 1 here
        assert code.highlight_line == 3  # "c = await tool()" is file line 3
    finally:
        linecache.cache.pop(cell_name, None)


@pytest.mark.asyncio
async def test_activity_idle_is_one_liner():
    """When idle with no turn running, /activity is a single status line."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import ActivityCommand
    from nooa_cli.tui.output import TableOutput, TextOutput

    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())

    async def _idle_locate():  # probe returns None when idle
        return None

    cmd._locate_agent = _idle_locate  # type: ignore[method-assign]

    result = await cmd.execute([])
    assert result.success
    assert len(result.outputs) == 1
    out = result.outputs[0]
    assert isinstance(out, TextOutput)
    assert not isinstance(out, TableOutput)
    assert "idle" in out.content.lower()


@pytest.mark.asyncio
async def test_innermost_cell_prefers_agent_cell_frame():
    """innermost_cell returns the agent's own ``Cell In[N]`` frame, not plumbing.

    The real suspend point is usually framework/stdlib code the cell is blocked
    in (e.g. asyncio.to_thread -> run_in_executor). The agent's own line lives in
    a ``Cell In[N]`` frame further up the chain; that is what /activity should
    highlight when a user asks "which line of my cell is running?".
    """
    cell_name = "Cell In[7]"
    loc = AgentLocation(
        task_name="TUIAgent.handle",
        stack=[
            FrameInfo("TUIAgent.handle", 1, "actor.py"),
            FrameInfo("__repl_wrapper__", 9, cell_name),
            FrameInfo("to_thread", 25, "/usr/lib/python3.12/asyncio/threads.py"),
        ],
    )
    assert loc.innermost.qualname == "to_thread"  # true suspend point
    assert loc.innermost_cell is loc.stack[1]  # the cell frame
    assert loc.innermost_cell.filename == cell_name
    assert loc.highlight is loc.innermost_cell  # highlight prefers the cell


@pytest.mark.asyncio
async def test_innermost_cell_none_without_cell_frame():
    """No cell frame on the stack -> innermost_cell is None, highlight falls back."""
    loc = AgentLocation(
        task_name="TUIAgent.handle",
        stack=[
            FrameInfo("TUIAgent.handle", 1, "actor.py"),
            FrameInfo("sleep", 3, "/usr/lib/python3.12/asyncio/tasks.py"),
        ],
    )
    assert loc.innermost_cell is None
    assert loc.highlight is loc.innermost


@pytest.mark.asyncio
async def test_activity_highlights_cell_line_below_plumbing(monkeypatch):
    """/activity highlights the agent's cell line even when suspended in plumbing.

    Mirrors the real bug: the cell did ``await asyncio.to_thread(...)`` so the
    suspend point is to_thread's run_in_executor, but the user wants the cell
    line. The code block must highlight the cell line and the stack must mark it.
    """
    cell_name = "Cell In[7]"
    cell_src = "wt = setup()\nimport os\ntool = await asyncio.to_thread(_connect)\nprint(tool)\n"
    linecache.cache[cell_name] = (
        len(cell_src),
        None,
        cell_src.splitlines(keepends=True),
        cell_name,
    )

    fake = AgentLocation(
        task_name="TUIAgent.handle",
        stack=[
            FrameInfo("TUIAgent.handle", 1, "actor.py"),
            FrameInfo("__repl_wrapper__", 3, cell_name),
            FrameInfo("to_thread", 25, "/usr/lib/python3.12/asyncio/threads.py"),
        ],
    )
    fake.stack[1].context = _source_context(cell_name, 3)

    from unittest.mock import AsyncMock, MagicMock

    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())

    async def _fake_locate():
        return fake

    monkeypatch.setattr(cmd, "_locate_agent", _fake_locate)

    try:
        result = await cmd.execute([])
        code = next(o for o in result.outputs if isinstance(o, CodeExecution))
        # highlight is the cell's await line, not to_thread's line 25
        assert code.highlight_line == 3
        assert "await asyncio.to_thread(_connect)" in code.code
        # the stack still lists the deeper plumbing frame, marked as such
        table = result.outputs[0]
        flat = "\n".join(cell for row in table.rows for cell in row)
        assert "to_thread" in flat
        assert "\u2192 __repl_wrapper__" in flat  # cell frame marked with arrow
    finally:
        linecache.cache.pop(cell_name, None)


@pytest.mark.asyncio
async def test_locate_attaches_context_to_cell_frame_not_plumbing():
    """The real probe attaches source to the highlight (cell) frame, not plumbing.

    Regression guard for the bug where ``locate_agent_on_loop`` attached source
    context only to ``stack[-1]`` (the framework suspend point), leaving the
    cell frame — which the renderer highlights — without source, so the code
    block silently vanished. Here a cell frame (registered in linecache) sits
    above an ``asyncio.sleep`` suspend point; the cell frame must carry context.
    """
    cell_name = "Cell In[99]"

    # A coroutine whose frame reports the cell filename, awaiting sleep. Only
    # ``co_filename`` is rewritten — we deliberately do NOT touch
    # ``co_firstlineno``, since the suspended frame's ``f_lineno`` is computed
    # relative to it and remapping makes the reported line version-fragile.
    async def cell_coro():
        await asyncio.sleep(5)

    cell_coro.__code__ = cell_coro.__code__.replace(co_filename=cell_name)

    async def handle():
        await cell_coro()

    task = asyncio.ensure_future(handle())
    await asyncio.sleep(0.02)
    try:
        loc = locate_agent_on_loop()
        assert loc is not None
        cell = loc.innermost_cell
        assert cell is not None, "expected the Cell In[N] frame in the stack"
        assert cell.filename == cell_name
        assert loc.highlight is cell

        # Register enough linecache lines to cover the frame's *real* f_lineno
        # (whatever the running interpreter reports), then re-run the probe so
        # the source-context attach has source to find. Robust across CPython
        # line-number-table changes — we never assume a specific line.
        n = cell.lineno + 5
        src = "".join(f"line {i}\n" for i in range(1, n + 1))
        linecache.cache[cell_name] = (
            len(src),
            None,
            src.splitlines(keepends=True),
            cell_name,
        )
        loc = locate_agent_on_loop()
        cell = loc.innermost_cell
        # The bug: context was attached to stack[-1] (sleep), not the cell frame.
        assert cell.context, "highlight (cell) frame must carry source context"
        current = [s for s in cell.context if s.is_current]
        assert len(current) == 1 and current[0].lineno == cell.lineno
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        linecache.cache.pop(cell_name, None)
