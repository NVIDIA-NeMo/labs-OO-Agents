# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test: _pformat respects Pydantic exclude=True fields.

Regression test for TUI hang caused by _pformat recursing into
ExecutionResult.captured_locals which contains arbitrary user objects.
The captured_locals field has exclude=True in Pydantic but _pformat
was ignoring that annotation, walking into the dict, and triggering
expensive property access (importlib_metadata .files → stat loop).
"""

import signal
from typing import Any

import pytest
from pydantic import BaseModel, Field


class FakeExecutionResult(BaseModel):
    """Mimics ExecutionResult with an excluded field containing expensive objects."""

    stdout: str = ""
    returned_value: Any = None
    captured_locals: dict[str, Any] = Field(
        default_factory=dict,
        description="Local variables — should NOT be serialized",
        exclude=True,
    )


def _require_sigalrm():
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("requires signal.SIGALRM (Unix-only)")


class ExpensiveObject:
    """Object that records if its attributes were accessed."""

    def __init__(self):
        self.accessed = False
        self.name = "pkg"

    @property
    def files(self):
        self.accessed = True
        return ["would_do_stat_on_thousands_of_files"]


def test_pformat_skips_excluded_pydantic_fields():
    """_pformat must not recurse into fields with exclude=True."""
    from nooa.agentdoc import truncating_pformat

    expensive = ExpensiveObject()
    result = FakeExecutionResult(
        stdout="hello",
        returned_value=None,
        captured_locals={"eps": expensive},
    )

    # Format the result — should NOT touch captured_locals
    output = truncating_pformat(result, max_depth=4)

    assert not expensive.accessed, (
        "_pformat recursed into a Pydantic exclude=True field and triggered "
        "a property on an object inside it. This causes the TUI hang."
    )
    # stdout should still appear (it's not excluded)
    assert "hello" in output


def test_pformat_still_includes_non_excluded_fields():
    """Non-excluded fields should still be formatted normally."""
    from nooa.agentdoc import truncating_pformat

    result = FakeExecutionResult(
        stdout="test output",
        returned_value=42,
        captured_locals={"x": 1},
    )

    output = truncating_pformat(result, max_depth=4)
    # Non-excluded fields should be present
    assert "test output" in output or "stdout" in output
    assert "42" in output


def test_pformat_no_hang_on_importlib_metadata_entrypoints():
    """End-to-end: pformat on ExecutionResult with entry_points must not hang."""
    from importlib.metadata import entry_points

    from nooa.agentdoc import truncating_pformat

    eps = entry_points(group="nooa.skills")

    result = FakeExecutionResult(
        stdout="found skills",
        returned_value=None,
        captured_locals={"eps": eps},
    )

    _require_sigalrm()

    def timeout_handler(signum, frame):
        raise TimeoutError(
            "pformat hung! It recursed into captured_locals containing "
            "importlib_metadata objects that trigger expensive I/O."
        )

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout
    try:
        output = truncating_pformat(result, max_depth=4)
        assert isinstance(output, str)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_exact_regression_real_execution_result():
    """Exact regression test for the TUI hang reported in #214.

    The agent generated code:
        from importlib.metadata import entry_points
        eps = entry_points(group="nooa.skills")
        print(...)
        for ep in eps: print(...)

    The tracing hook called truncating_pformat(ExecutionResult) where
    ExecutionResult.captured_locals contained the EntryPoints. pformat
    recursed into captured_locals, found EntryPoint objects, called
    hasattr(ep, 'files') which triggered Distribution.files property
    → thousands of os.stat() calls → blocked the event loop.
    """
    _require_sigalrm()

    from importlib.metadata import entry_points

    from nooa.agentdoc import truncating_pformat

    # Use FakeExecutionResult which has the same model_fields structure
    # as the real ExecutionResult (captured_locals with exclude=True).
    # The real ExecutionResult import is too heavy for a unit test.
    eps = entry_points(group="nooa.skills")

    result = FakeExecutionResult(
        stdout="Registered skill entry points:\n  nemo.shell → ...",
        captured_locals={"eps": eps, "entry_points": entry_points},
    )

    # Verify the model has the same exclude=True contract
    assert result.model_fields["captured_locals"].exclude is True

    def timeout_handler(signum, frame):
        raise TimeoutError(
            "REGRESSION: pformat hung on ExecutionResult with importlib_metadata "
            "objects in captured_locals. This is the exact bug from #214."
        )

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    try:
        output = truncating_pformat(result, max_depth=4)
        assert isinstance(output, str)
        # captured_locals should NOT appear in the output (exclude=True)
        assert "eps" not in output
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
