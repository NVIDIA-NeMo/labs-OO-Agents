# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import BaseModel, ValidationError

from nooa.config.execution_config import ExecutionConfig
from nooa.unifiedllm import FakeLLMClient


def test_execution_config_is_pydantic_model():
    assert issubclass(ExecutionConfig, BaseModel)


def test_execution_config_defaults():
    c = ExecutionConfig()
    assert c.max_nesting_depth == 10


def test_execution_config_frozen():
    c = ExecutionConfig()
    with pytest.raises(ValidationError):
        c.max_nesting_depth = 5


def test_merge_with_overrides_only_explicit_fields():
    base = ExecutionConfig()
    override = ExecutionConfig(max_nesting_depth=5)
    merged = base.merge_with(override)
    assert merged.max_nesting_depth == 5


def test_merge_with_rejects_empty_fields_set():
    """merge_with() must reject configs with no explicitly-set fields (e.g. ExecutionConfig())."""
    base = ExecutionConfig()
    no_overrides = ExecutionConfig()
    assert not no_overrides.model_fields_set
    with pytest.raises(
        ValueError, match="merge_with\\(\\) received a config with no model_fields_set"
    ):
        base.merge_with(no_overrides)


@pytest.mark.asyncio
async def test_max_nesting_depth_enforced():
    """Agent calls that exceed max_nesting_depth should raise RuntimeError.

    Regression test: ExecutionConfig.max_nesting_depth was stored but never
    checked during execution.
    """
    from nooa import Agent

    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"})

    class ShallowAgent(
        Agent,
        llm=llm,
        execution=ExecutionConfig(max_nesting_depth=1),
    ):
        async def outer(self) -> str:
            return await self.inner()

        async def inner(self) -> str: ...

    agent = ShallowAgent()
    # inner() is the 2nd nesting level — exceeds max_nesting_depth=1
    with pytest.raises(RuntimeError, match="nesting depth"):
        await agent.outer()
