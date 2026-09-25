# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The headless host: one in-process tree, closed on exit."""

import asyncio

import nooa_coder
import pytest
from coder_test_agents import cell
from nooa_coder import SessionOptions, SessionStore, TaskResult, open_tree
from nooa_coder.session.store import sessions_root

from nooa.interactive import Done
from nooa.unifiedllm import FakeLLMClient

_RESULT = (
    "return_result(Done(explanation='solved', result=TaskResult("
    "solution_description='fixed the bug', evidence='tests pass', how_to_verify='pytest')))"
)


async def test_open_tree_runs_a_batch_prompt_to_a_task_result(tmp_path):
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="coder_test_agents:BatchAgent",
        turn_method="handle_batch",
        llm=FakeLLMClient([cell(_RESULT)], strict_exhaustion=True),
    )
    async with open_tree(options) as tree:
        outcome = await asyncio.wait_for(tree.root.prompt("fix the bug"), 20)
        assert tree.registry.get(tree.root.id) is tree.root
    assert outcome == Done(
        explanation="solved",
        result=TaskResult(
            solution_description="fixed the bug", evidence="tests pass", how_to_verify="pytest"
        ),
    )
    assert tree.registry.sessions == {}


async def test_an_exception_in_the_block_leaves_no_live_claim(tmp_path, sessions_dir):
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="coder_test_agents:EchoAgent",
        llm=FakeLLMClient([], strict_exhaustion=True),
    )
    with pytest.raises(RuntimeError, match="benchmark crashed"):
        async with open_tree(options) as tree:
            root_id = tree.root.id
            assert SessionStore(sessions_dir).is_active(root_id)
            raise RuntimeError("benchmark crashed")
    assert not SessionStore(sessions_dir).is_active(root_id)
    assert tree.registry.sessions == {}


def _echo_options(workspace, **values):
    workspace.mkdir(parents=True, exist_ok=True)
    return SessionOptions(
        workspace=workspace,
        agent_spec="coder_test_agents:EchoAgent",
        llm=FakeLLMClient([], strict_exhaustion=True),
        **values,
    )


async def test_each_workspace_keeps_its_own_sessions(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    async with open_tree(_echo_options(one)) as first, open_tree(_echo_options(two)) as second:
        child = await first.registry.create(
            first.root.options.inherit(name="child"), parent_id=first.root.id
        )
        assert first.root.handle.path.parent == one / ".nooa" / "sessions"
        assert child.handle.path.parent == first.root.handle.path.parent
        assert second.root.handle.path.parent == two / ".nooa" / "sessions"
    assert [i.id for i in SessionStore(sessions_root(one)).list()] == [first.root.id]
    assert [i.id for i in SessionStore(sessions_root(two)).list()] == [second.root.id]


async def test_nooa_sessions_dir_puts_every_workspace_in_one_directory(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("NOOA_SESSIONS_DIR", str(shared))
    one, two = tmp_path / "one", tmp_path / "two"
    async with open_tree(_echo_options(one)) as first, open_tree(_echo_options(two)) as second:
        assert first.root.handle.path.parent == shared
        assert second.root.handle.path.parent == shared
    store = SessionStore(shared)
    assert {i.id for i in store.list()} == {first.root.id, second.root.id}
    assert [i.id for i in store.list(workspace=one)] == [first.root.id]
    assert not (one / ".nooa").exists()


async def test_the_sessions_dir_option_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("NOOA_SESSIONS_DIR", str(tmp_path / "shared"))
    given = tmp_path / "given"
    async with open_tree(_echo_options(tmp_path / "ws", sessions_dir=given)) as tree:
        assert tree.root.handle.path.parent == given


def test_public_names_are_exported():
    for name in (
        "Session",
        "SessionRegistry",
        "SessionPort",
        "SessionOptions",
        "SessionStore",
        "ChildRef",
        "Receipt",
        "open_tree",
        "Tree",
    ):
        assert hasattr(nooa_coder, name), name
