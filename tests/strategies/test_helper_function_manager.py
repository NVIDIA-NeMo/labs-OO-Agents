# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HelperFunctionManager class vs instance guard.

Under, helpers are never attached to the agent. They live as plain
callables in ``session_locals`` (and the exec namespace) so the LLM calls them
as ``helper(self, x)`` — not ``self.helper(x)``. These tests verify that
contract.
"""

import pytest

from nooa.strategies.generated_code import (
    ExecutionNamespaceBuilder,
    HelperFunctionManager,
)
from nooa.unifiedllm import FakeLLMClient


class TestHelperFunctionManagerGuard:
    """Tests for HelperFunctionManager class vs instance guard."""

    def test_rejects_class_instead_of_instance(self):
        """HelperFunctionManager should raise TypeError if passed a class."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        namespace = ExecutionNamespaceBuilder.build(FakeAgent)  # Note: class, not instance

        with pytest.raises(TypeError, match="instance but received a class"):
            manager.apply(
                code="def helper(self): pass",
                agent=FakeAgent,  # CLASS, not instance - should be rejected!
                session_locals={},
                namespace=namespace,
            )

    def test_accepts_instance(self):
        """HelperFunctionManager should accept agent instances and install helpers as plain callables."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        instance = FakeAgent()
        namespace = ExecutionNamespaceBuilder.build(instance)
        session_locals: dict = {}

        result = manager.apply(
            code="def helper(self): return 42",
            agent=instance,
            session_locals=session_locals,
            namespace=namespace,
        )

        assert "helper" in result.installed
        # Helper is a plain callable in session_locals — never attached to the agent.
        assert not hasattr(instance, "helper")
        helper = session_locals["helper"]
        assert callable(helper)
        assert helper(instance) == 42

    def test_method_does_not_leak_to_class(self):
        """Helpers are plain callables in session_locals, never attached to any instance."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        instance1 = FakeAgent()
        namespace = ExecutionNamespaceBuilder.build(instance1)

        manager.apply(
            code="def helper(self): return 42",
            agent=instance1,
            session_locals={},
            namespace=namespace,
        )

        # Neither instance ever has the helper attached.
        assert not hasattr(instance1, "helper")
        instance2 = FakeAgent()
        assert not hasattr(instance2, "helper")

    def test_method_does_not_leak_to_class_definition(self):
        """Helper methods must not appear in the agent class __dict__."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        instance = FakeAgent()
        namespace = ExecutionNamespaceBuilder.build(instance)

        manager.apply(
            code="def helper(self): return 42",
            agent=instance,
            session_locals={},
            namespace=namespace,
        )

        assert "helper" not in FakeAgent.__dict__, "Helper method leaked to class __dict__!"

    def test_async_helper_method_binding(self):
        """Async helper methods should be installed as plain callables in session_locals."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        instance = FakeAgent()
        namespace = ExecutionNamespaceBuilder.build(instance)
        session_locals: dict = {}

        result = manager.apply(
            code="async def async_helper(self): return await asyncio.sleep(0) or 42",
            agent=instance,
            session_locals=session_locals,
            namespace=namespace,
        )

        assert "async_helper" in result.installed
        assert not hasattr(instance, "async_helper")
        assert callable(session_locals["async_helper"])


class TestHelperFunctionManagerSessionLocals:
    """Tests for session_locals handling in HelperFunctionManager."""

    def test_helper_added_to_session_locals(self):
        """Helper methods should be added to session_locals for reuse."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        instance = FakeAgent()
        namespace = ExecutionNamespaceBuilder.build(instance)
        session_locals: dict = {}

        manager.apply(
            code="def helper(self): return 42",
            agent=instance,
            session_locals=session_locals,
            namespace=namespace,
        )

        assert "helper" in session_locals
        # session_locals stores plain callables (never bound methods).
        assert callable(session_locals["helper"])


class TestHelperFunctionManagerErrors:
    """Tests for error handling in HelperFunctionManager."""

    def test_records_decorator_validation_errors(self):
        """Errors from decorator validation should be recorded."""
        from nooa.agent import Agent

        class FakeAgent(Agent, llm=FakeLLMClient()):
            async def process(self) -> dict:
                """Process something."""
                ...

        manager = HelperFunctionManager()
        instance = FakeAgent()
        namespace = ExecutionNamespaceBuilder.build(instance)

        # Body references undefined var — no compile-time error, only at call time.
        result = manager.apply(
            code="def helper(self): return undefined_variable",
            agent=instance,
            session_locals={},
            namespace=namespace,
        )

        # Decorator-application failure.
        result = manager.apply(
            code="@nonexistent_decorator\ndef broken_helper(self): pass",
            agent=instance,
            session_locals={},
            namespace=namespace,
        )

        assert len(result.errors) > 0 or "broken_helper" not in result.installed
