# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: /activity must not freeze the UI loop while the agent is busy.

Root cause of the reported flicker: ``ActivityCommand`` probed the agent
location via the *blocking* ``agent_run`` (``future.result(timeout=30)``),
which froze the prompt_toolkit UI loop while the agent loop was busy —
starving the block-queue consumer (``self.message()`` output stopped
appearing) and wedging the prompt redraw (status-bar / input flicker).

The fix routes the probe through the awaitable ``agent_run_async`` so the UI
loop keeps spinning. These tests pin that behaviour at the unit level.
"""

import asyncio
import threading

import pytest


def _make_busy_agent_loop():
    """Start an event loop on its own thread with a long-running task hogging it.

    Returns (loop, stop_event, thread). The loop is "busy" in the sense that a
    blocking ``future.result`` against it would have to wait for the hog task
    to yield — modelling the agent mid-LLM-call / mid-cell.
    """
    loop = asyncio.new_event_loop()
    started = threading.Event()
    stop = asyncio.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    started.wait(2.0)
    return loop, stop, t


@pytest.mark.asyncio
async def test_agent_run_async_does_not_block_calling_loop():
    """``agent_run_async`` yields to the calling loop while the agent loop works.

    We schedule a probe that only completes after a delay, and concurrently run
    a "UI heartbeat" coroutine on the calling loop. If ``agent_run_async``
    blocked the loop (the old ``future.result`` behaviour), the heartbeat would
    not tick until the probe returned. We assert it ticks meanwhile.
    """
    from nooa_cli.tui.tui_application import TUIApplication

    app = TUIApplication.__new__(TUIApplication)  # avoid full prompt_toolkit init
    loop, _stop, _t = _make_busy_agent_loop()
    app._agent_loop = loop

    probe_done = asyncio.Event()

    def slow_probe():
        # Runs on the agent loop; sleeps to model an in-flight turn.
        async def _inner():
            await asyncio.sleep(0.3)
            return "located"

        return _inner()

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while not probe_done.is_set():
            ticks += 1
            await asyncio.sleep(0.02)

    hb = asyncio.ensure_future(heartbeat())

    result = await app.agent_run_async(slow_probe)
    probe_done.set()
    await hb

    loop.call_soon_threadsafe(loop.stop)

    assert result == "located"
    # The UI loop kept ticking during the 0.3s probe — proof it was not blocked.
    assert ticks >= 5


@pytest.mark.asyncio
async def test_agent_run_async_inline_without_loop():
    """With no agent loop (tests / pre-startup) it runs inline, no await needed."""
    from nooa_cli.tui.tui_application import TUIApplication

    app = TUIApplication.__new__(TUIApplication)
    app._agent_loop = None

    assert await app.agent_run_async(lambda: 42) == 42


@pytest.mark.asyncio
async def test_activity_probe_uses_async_dispatch():
    """ActivityCommand routes the location probe through agent_run_async.

    Guards against a regression to the blocking ``agent_run`` call, which is
    what froze the UI loop. We record which dispatcher the command used.
    """
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import ActivityCommand

    cmd = ActivityCommand(frontend=AsyncMock(), config=MagicMock(), agent=MagicMock())

    used = {"sync": False, "async": False}

    def sync_run(fn):
        used["sync"] = True
        return None

    async def async_run(fn):
        used["async"] = True
        return None

    cmd._agent_run = sync_run
    cmd._agent_run_async = async_run

    await cmd.execute([])

    assert used["async"] is True
    assert used["sync"] is False


@pytest.mark.asyncio
async def test_agent_mutating_commands_use_async_dispatch():
    """UI-loop commands must not call blocking agent_run while the agent is busy."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from nooa_cli.tui.commands import (
        CompactCommand,
        ModelCommand,
    )

    async def assert_uses_async_dispatch(cmd, execute_args):
        used = {"sync": False, "async": False}

        def sync_run(fn):
            used["sync"] = True
            return fn()

        async def async_run(fn):
            used["async"] = True
            return fn()

        cmd._agent_run = sync_run
        cmd._agent_run_async = async_run
        result = await cmd.execute(execute_args)
        assert result.success
        assert used["async"] is True
        assert used["sync"] is False

    config = MagicMock()
    config.default_model = "old/model"
    agent = MagicMock()
    agent.event_manager.keys.return_value = ["1", "2"]
    agent.context_stats = None

    from nooa_cli.tui.health_check import HealthCheckResult

    with (
        patch("nooa_cli.tui.config.get_llm_for_model", return_value=MagicMock()),
        patch(
            "nooa_cli.tui.health_check.probe_llm",
            new=AsyncMock(return_value=HealthCheckResult(ok=True)),
        ),
    ):
        await assert_uses_async_dispatch(ModelCommand(AsyncMock(), config, agent), ["new/model"])

    compact = CompactCommand(AsyncMock(), config, agent)
    agent._summarizers = []
    await assert_uses_async_dispatch(compact, [])


@pytest.mark.asyncio
async def test_session_swap_uses_async_agent_dispatch():
    """Session manager swaps run agent mutations without blocking the UI loop."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session.agent = MagicMock()
    session.agent.queue_manager = None
    session.agent.event_manager = MagicMock()
    session.registry = MagicMock()
    session.registry.commands.return_value = []
    old_storage = MagicMock()
    session.agent._storage = old_storage
    old_manager = MagicMock()
    old_manager.close = MagicMock()
    session._session_manager = old_manager
    new_manager = MagicMock()
    new_manager._storage = SimpleNamespace(event_backend=object())

    used = {"async": 0, "sync": 0}

    async def async_run(fn):
        used["async"] += 1
        return fn()

    def sync_run(fn):
        used["sync"] += 1
        return fn()

    session._app = SimpleNamespace(agent_run_async=async_run, agent_run=sync_run)

    await session._swap_session_manager(new_manager)

    assert used["async"] == 2
    assert used["sync"] == 0


@pytest.mark.asyncio
async def test_python_skill_slash_commands_use_async_agent_dispatch():
    """@slash_command methods run on the agent loop, not directly on the UI loop.

    Mesh slash commands mutate the live AgentMesh client/QueueManager and can do
    slow connect/list/cert work. Running those methods directly in
    CommandHandler ties the work to prompt_toolkit's UI loop; routing them
    through agent_run_async keeps input/redraw/output draining responsive while
    the agent loop performs the operation.
    """
    from unittest.mock import AsyncMock

    from nooa_cli.tui.commands import CommandHandler, _UserSkill

    class Registry:
        def __init__(self):
            self.skill_called = False

        def get_user_skill(self, name):
            if name != "mesh-list":
                return None

            def mesh_list(args: str = ""):
                self.skill_called = True
                return f"mesh roster for {args}"

            return _UserSkill(
                name="mesh-list",
                body="",
                description="Show the live mesh roster",
                _method=mesh_list,
            )

        def get_command(self, name):
            return None

        def get_all_command_classes(self):
            return {}

    registry = Registry()
    used = {"async": False}

    async def agent_run_async(fn):
        used["async"] = True
        return fn()

    handler = CommandHandler(
        registry=registry,
        frontend=AsyncMock(),
        agent_run_async=agent_run_async,
    )

    result = await handler.handle("/mesh-list now")

    assert result.success is True
    assert used["async"] is True
    assert registry.skill_called is True
    assert result.slash_result is not None
    assert result.slash_result.value == "mesh roster for now"
