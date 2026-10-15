# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped ordinary child agents join the active ATIF trajectory without network I/O."""

from __future__ import annotations

import asyncio

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, atif_scope
from nooa.events import Task
from nooa.standalone import _atif_exporter_var
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


class _Worker(Agent, llm=FakeLLMClient()):
    @strategy(PredictStrategy())
    async def answer(self) -> int:
        """Return seven."""
        ...

    async def fail(self) -> None:
        self.event_manager.add(Task(prompt="failing child", generation_id="failure"))
        raise ValueError("child failed")

    async def descend(self, depth: int) -> int:
        if depth == 0:
            return await self.answer()
        child = _worker()
        try:
            async with atif_scope(child, inherit_parent=True):
                return await child.descend(depth - 1)
        finally:
            await child.aclose()


def _worker() -> _Worker:
    return _Worker(llm=FakeLLMClient(scripted_responses=[LLMResponse(content='{"value": 7}')]))


def _handoffs(trajectory: Trajectory) -> list:
    return [
        step
        for step in trajectory.steps
        if step.extra and step.extra.get("event_kind") == "subagent_handoff"
    ]


async def test_inherit_parent_without_active_export_is_noop(tmp_path):
    child = _worker()
    before = list(child.event_manager._handlers["*"])
    try:
        assert _atif_exporter_var.get() is None
        async with atif_scope(child, inherit_parent=True, path=tmp_path / "child.json") as scope:
            assert scope is None
            assert child.event_manager._handlers["*"] == before
            assert await child.answer() == 7
        assert not list(tmp_path.iterdir())
        assert _atif_exporter_var.get() is None
    finally:
        await child.aclose()


async def test_unused_inherited_scope_does_not_create_child(tmp_path):
    parent, child = _worker(), _worker()
    out = tmp_path / "parent.json"
    try:
        async with atif_scope(parent, path=out):
            async with atif_scope(child, inherit_parent=True):
                pass
            await parent.answer()
        trajectory = Trajectory.model_validate_json(out.read_text())
        assert not trajectory.subagent_trajectories
        assert not _handoffs(trajectory)
        assert list(tmp_path.iterdir()) == [out]
    finally:
        await child.aclose()
        await parent.aclose()


async def test_inherited_scope_preserves_parent_binding_and_embeds_once(tmp_path):
    parent, child = _worker(), _worker()
    out = tmp_path / "parent.json"
    before = list(child.event_manager._handlers["*"])
    try:
        async with atif_scope(parent, path=out, session_id="shared") as root:
            async with atif_scope(child, inherit_parent=True) as nested:
                assert _atif_exporter_var.get() is root
                assert nested.path == root.path
                assert await child.answer() == 7
                assert _atif_exporter_var.get() is root
            assert child.event_manager._handlers["*"] == before
            assert _atif_exporter_var.get() is root
            await parent.answer()
        trajectory = Trajectory.model_validate_json(out.read_text())
        children = trajectory.subagent_trajectories or []
        assert len(children) == 1
        assert children[0].session_id == "shared"
        assert any(step.source == "agent" for step in children[0].steps)
        assert len(_handoffs(trajectory)) == 1
        assert list(tmp_path.iterdir()) == [out]
        assert _atif_exporter_var.get() is None
    finally:
        await child.aclose()
        await parent.aclose()


async def test_inherited_scope_exception_preserves_child_state_and_cleans_up(tmp_path):
    parent, child = _worker(), _worker()
    out = tmp_path / "parent.json"
    before = list(child.event_manager._handlers["*"])
    try:
        async with atif_scope(parent, path=out) as root:
            with pytest.raises(ValueError, match="child failed"):
                async with atif_scope(child, inherit_parent=True):
                    await child.fail()
            assert _atif_exporter_var.get() is root
            assert child.event_manager._handlers["*"] == before
            await parent.answer()
        trajectory = Trajectory.model_validate_json(out.read_text())
        children = trajectory.subagent_trajectories or []
        assert len(children) == 1
        assert children[0].extra["crashed"] is True
        assert children[0].extra["exception_type"] == "ValueError"
        assert any(step.message == "failing child" for step in children[0].steps)
        assert len(_handoffs(trajectory)) == 1
        assert list(tmp_path.iterdir()) == [out]
        assert _atif_exporter_var.get() is None
    finally:
        await child.aclose()
        await parent.aclose()


async def test_inherited_scope_recursive_children_preserve_tree(tmp_path):
    parent = _worker()
    out = tmp_path / "parent.json"
    try:
        async with atif_scope(parent, path=out):
            assert await parent.descend(2) == 7
        trajectory = Trajectory.model_validate_json(out.read_text())
        for _ in range(2):
            children = trajectory.subagent_trajectories or []
            assert len(children) == 1
            assert len(_handoffs(trajectory)) == 1
            trajectory = children[0]
        assert any(step.source == "agent" for step in trajectory.steps)
        assert list(tmp_path.iterdir()) == [out]
        assert _atif_exporter_var.get() is None
    finally:
        await parent.aclose()


async def test_inherited_scope_concurrent_children_stay_separate(tmp_path):
    parent = _worker()
    out = tmp_path / "parent.json"
    barrier = asyncio.Barrier(3)

    class ConcurrentWorker(_Worker):
        async def run(self) -> int:
            await barrier.wait()
            return await self.answer()

    async def run_child():
        child = ConcurrentWorker(
            llm=FakeLLMClient(scripted_responses=[LLMResponse(content='{"value": 7}')])
        )
        try:
            async with atif_scope(child, inherit_parent=True):
                return await child.run()
        finally:
            await child.aclose()

    try:
        async with atif_scope(parent, path=out) as root:
            assert await asyncio.gather(*(run_child() for _ in range(3))) == [7, 7, 7]
            assert _atif_exporter_var.get() is root
            await parent.answer()
        trajectory = Trajectory.model_validate_json(out.read_text())
        children = trajectory.subagent_trajectories or []
        assert len(children) == 3
        assert len({child.trajectory_id for child in children}) == 3
        assert len(_handoffs(trajectory)) == 3
        assert all(not child.subagent_trajectories for child in children)
        assert list(tmp_path.iterdir()) == [out]
        assert _atif_exporter_var.get() is None
    finally:
        await parent.aclose()
