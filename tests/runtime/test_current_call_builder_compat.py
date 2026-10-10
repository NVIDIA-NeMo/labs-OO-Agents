# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CurrentCall construction must preserve agent-runtime call state."""

import json
from typing import Annotated

from nooa import Agent, spec, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class SetupAgent(Agent):
    instruction: str = "expanded-runtime-instruction"

    async def run(self, value: int, stack: dict) -> dict:
        parent_id = self.runtime._agent_call_id
        result = await self.answer(value, _session_locals=stack)
        assert result["parent_id"] == parent_id
        return result

    @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
    async def answer(self, value: Annotated[int, spec(max_string=100_000)]) -> dict:
        """Use {self.instruction}."""
        setup_value = value + 1  # noqa: F841 - consumed by generated code
        ...


async def test_runtime_builder_preserves_setup_identity_parent_docs_and_caller_locals():
    code = (
        "call = self.runtime.current_call\n"
        "assert call.session_locals is not None\n"
        "assert call.kwargs['value'] == value\n"
        "assert call.param_names == ['value']\n"
        "assert call.param_specs['value']['max_string'] == 100000\n"
        "assert call.return_type is dict and call.is_async\n"
        "assert call.decorator == 'plan' and call.method_name == 'answer'\n"
        "assert 'setup_value = value + 1' in call.pre_ellipsis_code\n"
        "assert call.id == self.runtime._agent_call_id\n"
        "caller_after = setup_value + caller_before\n"
        "return_result({'value': caller_after, 'parent_id': call.parent_id, "
        "'docstring': call.docstring, 'id': call.id})"
    )
    llm = FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="finish", name="python_cell", arguments=json.dumps({"code": code}))
                ],
                finish_reason="tool_calls",
            )
        ],
        strict_exhaustion=True,
    )
    agent = SetupAgent(llm=llm)
    stack = {"caller_before": 10}
    try:
        result = await agent.run(4, stack)
        assert result["value"] == 15
        assert result["docstring"] == "Use expanded-runtime-instruction."
        assert stack["setup_value"] == 5 and stack["caller_after"] == 15
        task = next(event for event in agent.events.query(type="Task"))
        assert task.metadata["call_id"] == result["id"]
    finally:
        await agent.aclose()
