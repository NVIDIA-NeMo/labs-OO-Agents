# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual model requests preserve configured inputs and warn for omissions."""

import warnings
from typing import Annotated

import pytest

from nooa import Agent, CodeActStrategy, spec, strategy
from nooa.config.strategy_config import CodeActConfig
from nooa.config.truncation_config import CaptureConfig, FormatConfig, TruncationConfig
from nooa.prompts import build_prompt_data
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategies.prefill import PrefillTruncationWarning
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


def _sandbox_codeact():
    """Exercise worker transport without filesystem restrictions on test imports."""
    return CodeActStrategy(
        config=CodeActConfig(
            execution_backend="sandbox",
            sandbox=SandboxConfig(filesystem=False),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy_type", "mode"),
    [
        (factory, mode)
        for factory in (CodeActStrategy, PurePythonStrategy)
        for mode in ("method", "parameter", "format", "capture")
    ]
    + [(_sandbox_codeact, mode) for mode in ("method", "parameter", "format")],
)
async def test_prefill_request_and_warning(strategy_type, mode):
    """Check the real REPL, developer warning, and first model request together."""
    strategy_obj = strategy_type()
    llm = (
        FakeLLMClient.with_tool_call(
            "execute_python", {"code": "return_result(len(data))"}, strict_exhaustion=True
        )
        if isinstance(strategy_obj, CodeActStrategy)
        else FakeLLMClient.with_code_responses(["return len(data)"], strict_exhaustion=True)
    )
    method_limit = 40 if mode == "format" else 90_000
    config = TruncationConfig(prefill_format=FormatConfig(max_string=method_limit))

    class DemoAgent(Agent, llm=llm):
        @strategy(strategy_obj, truncation=config)
        async def bounded(self, data: str) -> int:
            """Return the length of the complete data variable."""
            ...

        @strategy(strategy_obj)
        async def full(self, data: Annotated[str, spec(max_string=None)]) -> int:
            """Return the length of the complete data variable."""
            ...

    agent = DemoAgent(
        truncation=TruncationConfig(
            capture=CaptureConfig(max_stdout=500 if mode == "capture" else 50_000)
        )
    )
    payload = "BEGIN|" + "A" * 1500 + "|UNIQUE_MIDDLE_SENTINEL|" + "B" * 1500 + "|END"
    method = agent.full if mode in {"parameter", "capture"} else agent.bounded
    try:
        preview = await build_prompt_data(method, payload)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PrefillTruncationWarning)
            assert await method(payload) == len(payload)

        notices = [item for item in caught if issubclass(item.category, PrefillTruncationWarning)]
        request = str(llm.calls[0].messages)
        assert "Input inspection prefill for " not in request
        if mode in {"format", "capture"}:
            assert len(notices) == 1
            assert "UNIQUE_MIDDLE_SENTINEL" not in request
            reason = "capture.max_stdout" if mode == "capture" else "max_string"
            assert reason in str(notices[0].message)
            if mode == "capture":
                assert "prefill_format" not in str(notices[0].message)
                assert "Output too large" in request
            else:
                assert "Output too large" not in request
        else:
            assert not notices
            assert payload in request
        expected = "None" if mode in {"parameter", "capture"} else str(method_limit)
        assert f"max_string={expected}" in preview.inspect_prefill
    finally:
        await agent.aclose()
