# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for strategy tests."""

import pytest


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.agent_id = "test_agent_123"
    agent.__class__.__name__ = "TestAgent"
    agent.event_manager = MagicMock()
    agent.events = []
    return agent


@pytest.fixture
def mock_event_manager():
    """Create a mock event manager for testing."""
    from unittest.mock import MagicMock

    em = MagicMock()
    em.add = MagicMock(return_value="event_123")  # Returns event_id
    em.get = MagicMock(return_value=None)
    em.update = MagicMock(return_value=True)
    em.filter = MagicMock(return_value=[])
    return em


@pytest.fixture
def mock_runtime(mock_agent, mock_event_manager):
    """Create a mock runtime that satisfies RuntimeServices protocol."""

    class MockRuntime:
        def __init__(self, agent, event_manager):
            self._agent = agent
            self._events = event_manager

        @property
        def agent(self):
            return self._agent

        @property
        def event_manager(self):
            """Event manager."""
            return self._events

        @property
        def truncation_config(self):
            """Truncation configuration."""
            from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

            return DEFAULT_TRUNCATION_CONFIG

        async def generate(self, *, tools=None, **kwargs):
            response = MagicMock(content="generated code", reasoning=None, usage={})
            return response, "event_123"

        async def execute_code(self, code, extra_globals=None):
            return None

        async def execute_nested(self, strategy, call):
            """Execute nested strategy."""
            return await strategy.execute(self, call)

        def get_generation_id(self) -> str | None:
            """Get current generation ID."""
            return "mock-generation-id"

        def get_parent_generation_id(self) -> str | None:
            """Get parent generation ID."""
            return None

        async def expand_variables(
            self, text: str, extra_context=None, error_mode: str = "show"
        ) -> str:
            """Mock expand_variables — returns text unchanged."""
            return text

    from unittest.mock import MagicMock

    return MockRuntime(mock_agent, mock_event_manager)
