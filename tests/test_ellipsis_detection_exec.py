# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that has_ellipsis_body works for dynamically exec'd functions.

This tests the critical path where PurePythonStrategy generates code with
@strategy decorators that need to be detected as needing generation.
"""

import pytest

from nooa.decorators import strategy
from nooa.ellipsis_detection import has_ellipsis_body, has_ellipsis_marker
from nooa.strategies import PredictStrategy


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

        # Set _generated_source (this is what the cell compiler does)
        func._generated_source = code

        # Should detect ellipsis body via _generated_source
        assert has_ellipsis_body(func) is True
        assert has_ellipsis_body(func, require_source=True) is True

    @pytest.mark.parametrize(
        "body", ["...", "pass", "return", "return None", "return 1", '"Doc only."']
    )
    def test_source_confirmation_does_not_guess_from_short_bytecode(self, body):
        """Strict standalone detection must not promote ambiguous bytecode to proof."""
        code = f"async def fn():\n    {body}\n"
        namespace = {}
        exec(compile(code, "<strict-ellipsis-confirmation>", "exec"), namespace)
        func = namespace["fn"]

        assert has_ellipsis_body(func, require_source=True) is False
        # Preserve the existing fallback for callers such as Agent methods.
        assert has_ellipsis_body(func) is True

        func._generated_source = code
        assert has_ellipsis_body(func, require_source=True) is (body == "...")


class TestHasEllipsisMarkerExec:
    """Test source-less generator markers without ordinary-Ellipsis false positives."""

    def test_yield_ellipsis_is_detected_from_bytecode(self):
        namespace = {}
        exec("async def generated():\n    yield ...", namespace)

        assert has_ellipsis_marker(namespace["generated"]) is True

    def test_yield_from_ellipsis_is_detected_from_bytecode(self):
        namespace = {}
        exec("def generated():\n    yield from ...", namespace)

        assert has_ellipsis_marker(namespace["generated"]) is True

    def test_ellipsis_index_is_not_a_generation_marker(self):
        namespace = {}
        exec("async def deterministic(items):\n    yield items[...]", namespace)

        assert has_ellipsis_marker(namespace["deterministic"]) is False


class TestRawExecDecorator:
    """Test raw exec with @strategy decorator (outside the cell compiler).

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
