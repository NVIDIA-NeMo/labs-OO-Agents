# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #331.

PredictConfig / ReflexionConfig sampling params (max_tokens/temperature/top_p)
were declared and documented but never forwarded to ``runtime.generate()``.
These tests pin the wiring: configured sampling params must reach the LLM call,
and unset (None) params must be omitted.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from nooa.config.strategy_config import PredictConfig, ReflexionConfig
from nooa.strategies.predict import PredictStrategy
from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy


class _Resp(BaseModel):
    value: int


def test_predict_build_sampling_kwargs_omits_none():
    strat = PredictStrategy(PredictConfig(max_tokens=256, top_p=0.9))
    kwargs = strat._build_sampling_kwargs()
    assert kwargs == {"max_tokens": 256, "top_p": 0.9}  # temperature (None) omitted


def test_predict_build_sampling_kwargs_empty_when_unset():
    assert PredictStrategy(PredictConfig())._build_sampling_kwargs() == {}


@pytest.mark.asyncio
async def test_predict_forwards_sampling_params_to_generate():
    strat = PredictStrategy(PredictConfig(max_tokens=128, temperature=0.2, top_p=0.5))
    runtime = MagicMock()
    runtime.generate = AsyncMock(return_value=(MagicMock(content=_Resp(value=1)), "evt"))

    await strat._call_llm_raw(runtime, _Resp)

    _, kwargs = runtime.generate.call_args
    assert kwargs["max_tokens"] == 128
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.5


@pytest.mark.asyncio
async def test_predict_omits_unset_sampling_params():
    strat = PredictStrategy(PredictConfig())
    runtime = MagicMock()
    runtime.generate = AsyncMock(return_value=(MagicMock(content=_Resp(value=1)), "evt"))

    await strat._call_llm_raw(runtime, _Resp)

    _, kwargs = runtime.generate.call_args
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_reflexion_build_sampling_kwargs():
    strat = ReflexionStrategy(config=ReflexionConfig(temperature=0.7))
    assert strat._build_sampling_kwargs() == {"temperature": 0.7}


@pytest.mark.asyncio
async def test_reflexion_forwards_sampling_params_to_generate():
    strat = ReflexionStrategy(config=ReflexionConfig(max_tokens=64, temperature=0.3))
    runtime = MagicMock()
    runtime.event_manager = MagicMock()
    runtime.truncation_config = MagicMock()
    runtime.generate = AsyncMock(
        return_value=(
            MagicMock(content=ReflectionOutput(is_satisfactory=True, reasoning="ok")),
            "evt",
        )
    )
    strat.reflection_prompt = AsyncMock(return_value="reflect")

    call = MagicMock(method_name="m", signature="(self) -> int", docstring="d")
    await strat._reflect(runtime, call, 42)

    _, kwargs = runtime.generate.call_args
    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == 0.3
