# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox cells preserve REPL state across recoverable exits."""

from __future__ import annotations

import pytest

from nooa.events import ExecutionSignal
from nooa.runtime.sandbox.cell_core import run_cell_source


@pytest.mark.asyncio
async def test_assignments_persist_after_cell_error():
    namespace: dict = {}

    failed = await run_cell_source(
        "answer = 41\nraise ValueError('boom')",
        namespace,
        execution_count=1,
    )
    assert isinstance(failed.error, ValueError)

    resumed = await run_cell_source("answer + 1", namespace, execution_count=2)
    assert resumed.error is None
    assert resumed.returned_value == 42


@pytest.mark.asyncio
async def test_assignments_persist_after_execution_signal():
    class ReturnResultSignal(ExecutionSignal):
        pass

    def return_result() -> None:
        raise ReturnResultSignal

    namespace: dict = {"return_result": return_result}

    signaled = await run_cell_source(
        "answer = 41\nreturn_result()",
        namespace,
        execution_count=1,
    )
    assert isinstance(signaled.signal, ReturnResultSignal)

    resumed = await run_cell_source("answer + 1", namespace, execution_count=2)
    assert resumed.error is None
    assert resumed.returned_value == 42
