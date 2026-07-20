# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that has_ellipsis_body works for dynamically exec'd functions.

This tests the critical path where PurePythonStrategy generates code with
@strategy decorators that need to be detected as needing generation.
"""

from nooa.decorators import strategy
from nooa.ellipsis_detection import has_ellipsis_body
from nooa.strategies import PredictStrategy
from nooa.strategies.generated_code import HelperFunctionManager


class TestHasEllipsisBodyExec:
    """Test has_ellipsis_body for functions created via exec."""

    def test_raw_exec_function_no_source(self):
        """Raw exec'd functions CAN be detected via bytecode heuristic.

        Even without _generated_source, the bytecode heuristic (checking
        bytecode length ≤12 bytes) successfully detects ellipsis-only functions.
        """
        code = '''
async def my_method(self, x: int) -> str:
    """A method."""
    ...
'''
        namespace = {}
        exec(code, namespace)
        func = namespace["my_method"]

        # Bytecode heuristic detects ellipsis even without _generated_source
        assert has_ellipsis_body(func) is True

    def test_raw_exec_function_implemented(self):
        """Test that exec'd function with implementation is NOT detected as ellipsis."""
        code = '''
async def my_method(self, x: int) -> str:
    """A method."""
    return str(x)
'''
        namespace = {}
        exec(code, namespace)
        func = namespace["my_method"]

        # Should NOT detect as ellipsis
        assert has_ellipsis_body(func) is False

    def test_exec_function_with_generated_source_ellipsis(self):
        """Test that _generated_source attribute enables ellipsis detection."""
        code = '''
async def my_method(self, x: int) -> str:
    """A method."""
    ...
'''
        namespace = {}
        exec(code, namespace)
        func = namespace["my_method"]

        # Set _generated_source (this is what HelperFunctionManager does)
        func._generated_source = code

        # Should detect ellipsis body via _generated_source
        assert has_ellipsis_body(func) is True


class TestHelperFunctionManagerSourceTracking:
    """Test that HelperFunctionManager sets _generated_source correctly."""

    def test_undecorated_method_gets_generated_source(self):
        """HelperFunctionManager should set _generated_source on undecorated methods."""
        code = '''
async def _helper(self, x: int) -> str:
    """A helper method."""
    ...
'''

        # Create a mock agent instance
        class MockAgent:
            pass

        agent = MockAgent()
        manager = HelperFunctionManager()

        session_locals: dict = {}
        result = manager.apply(
            code,
            agent,
            session_locals=session_locals,
            namespace={},
        )

        assert "_helper" in result.installed
        # helper lives in session_locals as a plain function, not on the agent.
        assert not hasattr(agent, "_helper")

        func = session_locals["_helper"]
        assert hasattr(func, "_generated_source")
        assert has_ellipsis_body(func) is True

    def test_decorated_method_needs_generation(self):
        """Test that @strategy decorator on method via HelperFunctionManager works.

        This is the critical test for the PurePython nested method pattern.
        """
        code = '''
@strategy(PredictStrategy())
async def _summarize(self, doc: str) -> str:
    """Summarize a document."""
    ...
'''

        class MockAgent:
            pass

        agent = MockAgent()
        manager = HelperFunctionManager()

        session_locals: dict = {}
        result = manager.apply(
            code,
            agent,
            session_locals=session_locals,
            namespace={
                "strategy": strategy,
                "PredictStrategy": PredictStrategy,
            },
        )

        assert "_summarize" in result.installed
        # helper is in session_locals as a plain decorated callable.
        assert not hasattr(agent, "_summarize")

        decorated = session_locals["_summarize"]
        assert getattr(decorated, "_needs_generation", None) is True, (
            f"Expected _needs_generation=True but got "
            f"{getattr(decorated, '_needs_generation', 'MISSING')}"
        )

    def test_decorated_implemented_method_no_generation(self):
        """Test that implemented @strategy method has _needs_generation=False."""
        code = '''
@strategy(PredictStrategy())
async def _helper(self, doc: str) -> str:
    """Process a document."""
    return f"Processed: {doc}"
'''

        class MockAgent:
            pass

        agent = MockAgent()
        manager = HelperFunctionManager()

        session_locals: dict = {}
        result = manager.apply(
            code,
            agent,
            session_locals=session_locals,
            namespace={
                "strategy": strategy,
                "PredictStrategy": PredictStrategy,
            },
        )

        assert "_helper" in result.installed
        decorated = session_locals["_helper"]
        assert getattr(decorated, "_needs_generation", None) is False


class TestRawExecDecorator:
    """Test raw exec with @strategy decorator (without HelperFunctionManager).

    These tests document the expected behavior: raw exec doesn't work
    for decorated functions without _generated_source.
    """

    def test_raw_exec_decorated_detects_via_bytecode(self):
        """Raw exec'd @strategy function detects ellipsis via bytecode heuristic."""
        code = '''
@strategy(PredictStrategy())
async def _helper(self, doc: str) -> str:
    """Summarize a document."""
    ...
'''
        namespace = {
            "strategy": strategy,
            "PredictStrategy": PredictStrategy,
        }
        exec(code, namespace)
        func = namespace["_helper"]

        # Bytecode heuristic allows detection even without _generated_source
        # The @strategy decorator will correctly set _needs_generation=True
        assert getattr(func, "_needs_generation", None) is True

    def test_raw_exec_decorated_implemented(self):
        """Raw exec'd implemented @strategy function correctly has _needs_generation=False."""
        code = '''
@strategy(PredictStrategy())
async def _helper(self, doc: str) -> str:
    """Summarize a document."""
    return f"Summary of: {doc}"
'''
        namespace = {
            "strategy": strategy,
            "PredictStrategy": PredictStrategy,
        }
        exec(code, namespace)
        func = namespace["_helper"]

        # Should NOT need generation since it has implementation
        assert getattr(func, "_needs_generation", None) is False
