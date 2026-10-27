# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for inspected versus live prefill truncation settings."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nooa import Agent, CodeActStrategy, strategy
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.prompts import build_prompt_data
from nooa.runtime.actor import _current_truncation_config_var
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_type", [CodeActStrategy, PurePythonStrategy])
@pytest.mark.parametrize("method_limit", [None, 90_000], ids=["agent-default", "method-override"])
async def test_live_prefill_matches_inspected_truncation(strategy_type, method_limit, monkeypatch):
    """Live prefill must use the same resolved configuration as prompt inspection.

    Run the real prefill path, mocking only code execution to avoid starting a
    REPL or calling an LLM. Activate the method config as ActorRuntime does.
    """
    strategy_obj = strategy_type()
    method_config = (
        TruncationConfig(prefill_format=FormatConfig(max_string=method_limit))
        if method_limit is not None
        else None
    )

    class DemoAgent(Agent, llm=FakeLLMClient()):
        @strategy(strategy_obj, truncation=method_config)
        async def analyze(self, data: str) -> str:
            """Analyze the data."""
            ...

    agent = DemoAgent()
    token = None
    try:
        payload = "BEGIN-" + "x" * 3000 + "-END"
        preview = await build_prompt_data(agent.analyze, payload)
        wrapper = agent.analyze.__func__
        call = CurrentCall.from_method(getattr(wrapper, "_original", wrapper), args=(payload,))
        resolved = agent._truncation.merge_with(method_config)
        token = _current_truncation_config_var.set(resolved)

        result = SimpleNamespace(
            captured_locals={},
            error=None,
            stdout="",
            stderr="",
            returned_value=None,
            has_return=False,
            explicit_return=False,
        )
        execute = AsyncMock(return_value=result)
        executor_name = (
            "_execute_prefill_step"
            if isinstance(strategy_obj, CodeActStrategy)
            else "_execute_code"
        )
        monkeypatch.setattr(strategy_obj, executor_name, execute)

        await strategy_obj._run_prefill(
            agent.runtime,
            call,
            {},
            SimpleNamespace(session_locals={}),
        )

        execute.assert_awaited_once()
        live_code = execute.await_args.args[1]
        assert f"max_string={resolved.prefill_format.max_string}" in preview.inspect_prefill
        assert live_code == preview.inspect_prefill
    finally:
        if token is not None:
            _current_truncation_config_var.reset(token)
        await agent.aclose()
