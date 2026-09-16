# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal notices adapt the shared completion policy without lifecycle extras."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nooa_cli.interactive.local_turn_policy import LocalTurnPolicy as SharedTurnPolicy
from nooa_cli.tui.local_turn_policy import LocalTurnPolicy as NativeTurnPolicy


@pytest.mark.parametrize("policy_type", [SharedTurnPolicy, NativeTurnPolicy])
async def test_turn_status_stops_after_shutdown(policy_type):
    emit = AsyncMock()
    policy = policy_type(emit_output=emit)
    agent = SimpleNamespace()
    await policy.after_handle(agent, SimpleNamespace(kind="DONE", explanation="Finished."))
    emit.assert_awaited_once()
    assert emit.call_args.args[0].kind == "DONE"
    assert emit.call_args.args[0].explanation == "Finished."
    await policy.shutdown()
    await policy.shutdown()
    await policy.after_handle(agent, SimpleNamespace(kind="DONE", explanation="Late result"))
    emit.assert_awaited_once()
