# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single-tool (CodeActV2) coding agent on the Session layer."""

import asyncio
import json

import pytest
from nooa_coder.session.items import TaskResult
from nooa_coder.session.loader import load_agent_class
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

from nooa.interactive import Done
from nooa.llm_types import LLMResponse
from nooa.unifiedllm import FakeLLMClient, ToolCall

TIMEOUT = 30
SPEC = "nooa_coder.coding.experimental_agent:ExperimentalCodingAgent"


def python_cell(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name="python_cell", arguments=json.dumps({"code": code}))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


@pytest.mark.parametrize(
    "spec",
    [
        "nooa_cli.tui.experimental_agent:ExperimentalTUIAgent",
        "nooa_cli.coding.experimental_agent:ExperimentalTUIAgent",
        "nooa_cli.coding.experimental_agent:ExperimentalCodingAgent",
    ],
)
def test_legacy_experimental_specs_load_the_rewritten_class(spec):
    assert load_agent_class(spec).__name__ == "ExperimentalCodingAgent"
    assert load_agent_class(spec) is load_agent_class(SPEC)


async def test_an_experimental_batch_turn_must_carry_a_task_result(tmp_path, sessions_dir):
    llm = FakeLLMClient(
        [
            python_cell("return_result(Done(explanation='finished'))", "c1"),
            python_cell(
                "return_result(Done(explanation='finished', result=TaskResult("
                "solution_description='a', evidence='b', how_to_verify='c')))",
                "c2",
            ),
        ],
        strict_exhaustion=True,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = SessionRegistry(SessionStore(sessions_dir))
    try:
        root = await registry.create(
            SessionOptions(
                workspace=workspace,
                agent_spec=SPEC,
                llm=llm,
                turn_method="handle_batch",
                sessions_dir=sessions_dir,
            )
        )
        outcome = await asyncio.wait_for(root.prompt("do it"), TIMEOUT)
        assert outcome == Done(
            explanation="finished",
            result=TaskResult(solution_description="a", evidence="b", how_to_verify="c"),
        )
        assert [tool.name for tool in llm.last_tools or []] == ["python_cell"]
        assert "return_result validation error" in str(llm.calls[1].messages)
    finally:
        await registry.close_all()
