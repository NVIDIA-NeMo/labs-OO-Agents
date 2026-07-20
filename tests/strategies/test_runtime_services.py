# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RuntimeServices protocol.

TDD: Write these tests first, then implement the protocol.
"""

from typing import Protocol


class TestRuntimeServicesProtocol:
    """Tests for RuntimeServices protocol definition."""

    def test_protocol_exists(self):
        """RuntimeServices protocol should exist."""
        from nooa.strategies.base import RuntimeServices

        # Should be a Protocol
        assert hasattr(RuntimeServices, "__protocol_attrs__") or issubclass(
            RuntimeServices, Protocol
        )

    def test_protocol_is_runtime_checkable(self):
        """RuntimeServices should be runtime checkable."""
        from nooa.strategies.base import RuntimeServices

        # Should work with isinstance()
        class MockRuntime:
            @property
            def agent(self):
                return None

            @property
            def event_manager(self):
                return None

            @property
            def truncation_config(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

            async def execute_nested(self, strategy, call):
                return None

            def get_generation_id(self):
                return None

            def get_parent_generation_id(self):
                return None

            async def expand_variables(self, text, extra_context=None, error_mode="show"):
                return text

        mock = MockRuntime()
        assert isinstance(mock, RuntimeServices)

    def test_protocol_requires_agent_property(self):
        """RuntimeServices should require agent property."""
        from nooa.strategies.base import RuntimeServices

        class MissingAgent:
            @property
            def event_manager(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

        mock = MissingAgent()
        assert not isinstance(mock, RuntimeServices)

    def test_protocol_requires_event_manager_property(self):
        """RuntimeServices should require event_manager property."""
        from nooa.strategies.base import RuntimeServices

        class MissingEventManager:
            @property
            def agent(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

        mock = MissingEventManager()
        assert not isinstance(mock, RuntimeServices)

    def test_protocol_requires_generate_method(self):
        """RuntimeServices should require generate() method."""
        from nooa.strategies.base import RuntimeServices

        class MissingGenerate:
            @property
            def agent(self):
                return None

            @property
            def event_manager(self):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

        mock = MissingGenerate()
        assert not isinstance(mock, RuntimeServices)

    def test_protocol_requires_execute_code_method(self):
        """RuntimeServices should require execute_code() method."""
        from nooa.strategies.base import RuntimeServices

        class MissingExecuteCode:
            @property
            def agent(self):
                return None

            @property
            def event_manager(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_nested(self, strategy, call):
                return None

        mock = MissingExecuteCode()
        assert not isinstance(mock, RuntimeServices)

    def test_protocol_requires_execute_nested_method(self):
        """RuntimeServices should require execute_nested() method."""
        from nooa.strategies.base import RuntimeServices

        class MissingExecuteNested:
            @property
            def agent(self):
                return None

            @property
            def event_manager(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

        mock = MissingExecuteNested()
        assert not isinstance(mock, RuntimeServices)


class TestRuntimeServicesComplete:
    """Tests for complete RuntimeServices implementation."""

    def test_complete_implementation_passes_check(self):
        """Complete implementation should pass isinstance check."""
        from nooa.strategies.base import RuntimeServices

        class CompleteRuntime:
            @property
            def agent(self):
                return None

            @property
            def event_manager(self):
                return None

            @property
            def truncation_config(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

            async def execute_nested(self, strategy, call):
                return None

            def get_generation_id(self):
                return None

            def get_parent_generation_id(self):
                return None

            async def expand_variables(self, text, extra_context=None, error_mode="show"):
                return text

        runtime = CompleteRuntime()
        assert isinstance(runtime, RuntimeServices)

    def test_extra_methods_are_allowed(self):
        """RuntimeServices implementations can have extra methods."""
        from nooa.strategies.base import RuntimeServices

        class ExtendedRuntime:
            @property
            def agent(self):
                return None

            @property
            def event_manager(self):
                return None

            @property
            def truncation_config(self):
                return None

            async def generate(self, messages, tools=None, **kwargs):
                return None

            async def execute_code(self, code, extra_globals=None):
                return None

            async def execute_nested(self, strategy, call):
                return None

            def get_generation_id(self):
                return None

            def get_parent_generation_id(self):
                return None

            def extra_method(self):
                """Extra method not in protocol."""
                return "extra"

            async def expand_variables(self, text, extra_context=None, error_mode="show"):
                return text

        runtime = ExtendedRuntime()
        assert isinstance(runtime, RuntimeServices)
