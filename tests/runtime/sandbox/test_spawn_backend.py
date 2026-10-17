# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spawn backend: the worker rebuilds everything from a blueprint.

``fork`` inherits the live agent and the ``return_result`` closure, so a bug in
the broker can hide behind inherited state. ``spawn`` inherits nothing, which is
what makes this the real completeness test for the broker — and the only start
method Windows has.

The agent class is imported by name in the child, so it lives in
``spawn_agent_module`` rather than inside this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor

from .spawn_agent_module import SpawnProbeAgent

pytestmark = pytest.mark.sandbox

# The spawned child inherits sys.path, so the agent module must be findable.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _executor(*, timeout: float = 30.0) -> SandboxedExecutor:
    return SandboxedExecutor(
        SpawnProbeAgent(),
        # require=False: the guards are pruned on hosts without a mechanism, so
        # this test measures the process/blueprint path independent of Landlock.
        SandboxConfig(require=False, start_method="spawn"),
        cell_timeout=timeout,
    )


async def test_spawn_worker_runs_a_cell():
    ex = _executor()
    try:
        result = await ex.run_cell("x = 6 * 7\nx", execution_count=1)
        assert result.error is None, result.error
        # A trailing expression is the cell's implicit return, not stdout.
        assert result.returned_value == 42
    finally:
        await ex.aclose()


async def test_spawn_worker_brokers_self_method_calls():
    """``self.*`` must reach the live parent — nothing was inherited."""
    ex = _executor()
    try:
        result = await ex.run_cell("print(self.label('probe'))", execution_count=1)
        assert result.error is None, result.error
        assert "value:probe" in result.stdout
    finally:
        await ex.aclose()


async def test_spawn_worker_sees_live_parent_state():
    """A parent-side mutation must be visible on the next attribute read."""
    ex = _executor()
    try:
        first = await ex.run_cell("self.bump()", execution_count=1)
        assert first.error is None, first.error
        second = await ex.run_cell("print(self.prefix)", execution_count=2)
        assert second.error is None, second.error
        assert "changed" in second.stdout
    finally:
        await ex.aclose()


async def test_spawn_worker_brokers_introspection():
    """``doc(self)`` has no local instance to render, so it must broker."""
    ex = _executor()
    try:
        result = await ex.run_cell("print(doc(self))", execution_count=1)
        assert result.error is None, result.error
        assert "label" in result.stdout
    finally:
        await ex.aclose()


async def test_spawn_worker_keeps_namespace_across_cells():
    ex = _executor()
    try:
        first = await ex.run_cell("carried = [1, 2, 3]", execution_count=1)
        assert first.error is None, first.error
        second = await ex.run_cell("print(sum(carried))", execution_count=2)
        assert second.error is None, second.error
        assert "6" in second.stdout
    finally:
        await ex.aclose()


async def test_spawn_requires_importable_agent_class():
    """A locally defined class cannot be rebuilt; the error must say so."""
    from nooa.runtime.sandbox.errors import SandboxUnavailable

    class LocalAgent(SpawnProbeAgent):  # defined inside a function -> <locals>
        pass

    # Fail fast at construction: this is a configuration error, not a cell error.
    with pytest.raises(SandboxUnavailable, match="importable by name"):
        SandboxedExecutor(
            LocalAgent(),
            SandboxConfig(require=False, start_method="spawn"),
            cell_timeout=10.0,
        )
