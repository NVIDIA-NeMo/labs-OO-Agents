# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The headless host: one in-process tree, closed on exit."""

import asyncio

import nooa_coder
import pytest
from coder_test_agents import cell
from nooa_coder import SessionOptions, SessionStore, TaskResult, open_tree

from nooa.interactive import Done
from nooa.unifiedllm import FakeLLMClient

_RESULT = (
    "return_result(Done(explanation='solved', result=TaskResult("
    "solution_description='fixed the bug', evidence='tests pass', how_to_verify='pytest')))"
)


async def test_open_tree_runs_a_batch_prompt_to_a_task_result(tmp_path, sessions_dir):
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="coder_test_agents:BatchAgent",
        turn_method="handle_batch",
        llm=FakeLLMClient([cell(_RESULT)], strict_exhaustion=True),
        sessions_dir=sessions_dir,
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
        sessions_dir=sessions_dir,
    )
    with pytest.raises(RuntimeError, match="benchmark crashed"):
        async with open_tree(options) as tree:
            root_id = tree.root.id
            assert SessionStore(sessions_dir).is_active(root_id)
            raise RuntimeError("benchmark crashed")
    assert not SessionStore(sessions_dir).is_active(root_id)
    assert tree.registry.sessions == {}


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
