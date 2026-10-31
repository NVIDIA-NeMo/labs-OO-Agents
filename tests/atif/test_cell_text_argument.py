# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF export keeps python_cell's raw ``text`` argument in the tool call."""

from __future__ import annotations

import json
from pathlib import Path

from nooa import Agent, strategy
from nooa.atif import Trajectory, atif_scope
from nooa.config import CodeActConfig
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.atif.normative import assert_atif_normative

TEXTS = {"app.py": 'print("hi")\r\n\'\'\' """ \\ 🤖'}


async def test_atif_tool_call_arguments_include_the_cell_text(tmp_path: Path) -> None:
    code = "return_result(len(text['app.py']))"
    fake = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                parts=(
                    ToolCall(
                        id="cell",
                        name="python_cell",
                        arguments=json.dumps({"code": code, "text": TEXTS}),
                    ),
                ),
                finish_reason="tool_calls",
                usage={"prompt_tokens": 5, "completion_tokens": 1},
            )
        ]
    )

    class TextAgent(Agent, llm=fake):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def run(self) -> int:
            """Count."""
            ...

    agent = TextAgent()
    out = tmp_path / "t.json"
    async with atif_scope(agent, path=out, session_id="s1", agent_name="TextAgent"):
        assert await agent.run() == len(TEXTS["app.py"])
    traj = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(traj)
    calls = [tc for s in traj.steps for tc in (s.tool_calls or []) if tc.tool_call_id == "cell"]
    assert len(calls) == 1
    assert calls[0].arguments == {"code": code, "text": TEXTS}
