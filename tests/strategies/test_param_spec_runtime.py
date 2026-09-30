# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime regressions for per-parameter prefill truncation overrides."""

from __future__ import annotations

import json
from typing import Annotated

import pytest

from nooa import Agent, CodeActStrategy, spec, strategy
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


def _messages(llm: FakeLLMClient) -> str:
    """Serialize provider-shaped messages for content-only assertions."""
    return json.dumps(llm.last_messages)


def _payload(label: str, size: int = 2_400) -> str:
    return f"{label}-BEGIN-" + "x" * size + f"-{label}-END"


class _ExplicitCodeActAgent(Agent, llm=_TEST_LLM):
    @strategy(CodeActStrategy())
    async def solve(self, instruction: Annotated[str, spec(max_string=None)]) -> str:
        """Solve the instruction."""
        ...


class _DefaultCodeActAgent(Agent, llm=_TEST_LLM):
    async def solve(self, instruction: Annotated[str, spec(max_string=None)]) -> str:
        """Solve the instruction."""
        ...


class _UnannotatedAgent(Agent, llm=_TEST_LLM):
    async def solve(self, instruction: str) -> str:
        """Solve the instruction."""
        ...


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [_ExplicitCodeActAgent, _DefaultCodeActAgent])
async def test_runtime_honors_unlimited_string_override(agent_type):
    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})
    agent = agent_type(llm=llm)
    instruction = _payload(agent_type.__name__)

    assert await agent.solve(instruction) == "done"

    messages = _messages(llm)
    assert instruction in messages
    assert f"str(len={len(instruction)}" not in messages
    assert "Output too large" not in messages


@pytest.mark.asyncio
async def test_unannotated_string_keeps_default_formatter_limit():
    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})
    agent = _UnannotatedAgent(llm=llm)
    instruction = _payload("DEFAULT")

    assert await agent.solve(instruction) == "done"

    messages = _messages(llm)
    assert instruction not in messages
    assert f"str(len={len(instruction)}, [:1000]=" in messages
    assert "Output too large" not in messages


@pytest.mark.asyncio
async def test_finite_param_override_wins_over_agent_and_method_defaults():
    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})

    class FiniteOverrideAgent(
        Agent,
        llm=llm,
        truncation=TruncationConfig(prefill_format=FormatConfig(max_string=40)),
    ):
        @strategy(
            CodeActStrategy(),
            truncation=TruncationConfig(prefill_format=FormatConfig(max_string=80)),
        )
        async def solve(self, instruction: Annotated[str, spec(max_string=120)]) -> str:
            """Solve the instruction."""
            ...

    instruction = _payload("FINITE", size=240)
    assert await FiniteOverrideAgent().solve(instruction) == "done"

    messages = _messages(llm)
    assert instruction not in messages
    assert f"str(len={len(instruction)}, [:60]=" in messages
    assert "[:40]=" not in messages
    assert "[:20]=" not in messages


@pytest.mark.asyncio
async def test_stdout_capture_limit_remains_independent_of_param_override():
    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})
    agent = _DefaultCodeActAgent(llm=llm)
    instruction = _payload("CAPTURE", size=55_000)

    assert await agent.solve(instruction) == "done"

    messages = _messages(llm)
    assert instruction not in messages
    assert "Output too large" in messages
    assert f"str(len={len(instruction)}" not in messages
