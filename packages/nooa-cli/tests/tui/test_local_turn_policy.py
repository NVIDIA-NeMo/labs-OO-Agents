# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared turn completion and reflection lifecycle for both hosts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from nooa_cli.interactive.local_turn_policy import LocalTurnPolicy as SharedTurnPolicy
from nooa_cli.tui.local_turn_policy import LocalTurnPolicy as NativeTurnPolicy


@pytest.fixture(params=[SharedTurnPolicy, NativeTurnPolicy])
def policy_state(request):
    # Simulate a resumed snapshot with the retired feature enabled.
    reflection = Mock(interrupt=AsyncMock())
    agent = SimpleNamespace(
        vars={"tui_keep_going": True, "tui_keep_going_model": "obsolete-judge"},
        _tui_reflection_runner=reflection,
    )
    runtime = Mock()

    async def run_async(fn):
        return await fn()

    runtime.run_async = AsyncMock(side_effect=run_async)
    emit = AsyncMock()
    policy = request.param(agent, runtime, emit_output=emit, invalidate=Mock())
    return policy, agent, runtime, reflection, emit


async def test_done_schedules_reflection_without_continuing_old_snapshot(policy_state):
    policy, agent, runtime, reflection, emit = policy_state
    await policy.before_handle(agent)
    reflection.interrupt.assert_awaited_once()
    await policy.after_handle(agent, SimpleNamespace(kind="DONE", explanation="Finished."))
    reflection.on_response_done.assert_called_once()
    assert runtime.mock_calls == []
    emit.assert_awaited_once()
    assert emit.call_args.args[0].kind == "DONE"
    assert emit.call_args.args[0].explanation == "Finished."
    await policy.shutdown()


async def test_shutdown_stops_reflection_on_owner_and_blocks_late_turn(policy_state):
    policy, agent, runtime, reflection, emit = policy_state
    await policy.shutdown()
    await policy.shutdown()
    runtime.run_async.assert_awaited_once()
    reflection.interrupt.assert_awaited_once()
    reflection.teardown.assert_called_once()

    await policy.before_handle(agent)
    await policy.after_handle(agent, SimpleNamespace(kind="DONE", explanation="Late result"))
    reflection.interrupt.assert_awaited_once()
    reflection.on_response_done.assert_not_called()
    emit.assert_not_awaited()
