# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test all models in models.yaml with agents.

These tests verify that each model configuration works with nooa agents.
Tests are skipped if the required API key is not set.

SKIPPED BY DEFAULT: These tests make real API calls. Run explicitly:
    RUN_MODEL_TESTS=1 pytest util/eval_pipeline/tests/test_models.py -v
"""

import os

import pytest

from eval_pipeline.model_factory import client, get, list_models
from nooa import Agent

# Skip entire module by default - only run when explicitly requested
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_TESTS"),
    reason="Model tests skipped by default. Set RUN_MODEL_TESTS=1 to run.",
)


def get_skip_reason(model_id: str) -> str | None:
    """Return skip reason if model should be skipped, None otherwise."""
    cfg = get(model_id)
    if not os.getenv(cfg.api_key_env):
        return f"API key {cfg.api_key_env} not set"
    return None


# Get all model IDs from models.yaml
ALL_MODELS = list_models()


# Simple test agent that works with any model
class TestAgent(Agent):
    """A simple test agent for validating model connectivity."""

    async def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        ...


@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_model_config_valid(model_id: str):
    """Test that model config can be loaded."""
    cfg = get(model_id)
    assert cfg.model_id == model_id
    assert cfg.name  # Should have a name
    assert cfg.api_key_env  # Should have an API key env var


@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_model_client_creation(model_id: str):
    """Test that a client can be created (doesn't make API call)."""
    skip_reason = get_skip_reason(model_id)
    if skip_reason:
        pytest.skip(skip_reason)

    llm = client(model_id)
    assert llm is not None
    assert llm.model  # Should have a model name


@pytest.mark.parametrize("model_id", ALL_MODELS)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_model_in_agent(model_id: str):
    """Test that a model works with an agent.

    This test creates an agent with each model and runs a simple task.
    Marked as slow because it makes actual API calls.
    """
    skip_reason = get_skip_reason(model_id)
    if skip_reason:
        pytest.skip(skip_reason)

    llm = client(model_id)
    test_agent = TestAgent(llm=llm)

    try:
        result = await test_agent.multiply(3, 4)
        print(f"\n{model_id}: 3 * 4 = {result}")
        assert result == 12, f"Expected 12, got {result}"
    except Exception as e:
        pytest.fail(f"{model_id} failed: {e}")
