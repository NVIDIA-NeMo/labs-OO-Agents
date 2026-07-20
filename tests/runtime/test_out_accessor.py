# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OutAccessor - Jupyter-style access to execution outputs."""

from typing import Any

import pytest

from nooa.context_blocks import ResultStatus
from nooa.events import PythonOutput


class MockEventManager:
    """Mock EventManager for testing OutAccessor.

    This provides a simple in-memory implementation that mimics
    how PythonOutput events are stored and retrieved.
    """

    def __init__(self):
        self._events: list[PythonOutput] = []
        self._tool_call_counter = 0

    def add_output(self, execution_count: int, value: Any) -> None:
        """Helper to add an execution output."""
        self._tool_call_counter += 1
        event = PythonOutput(
            tool_call_id=f"test_call_{self._tool_call_counter}",
            execution_status=ResultStatus.COMPLETE,
            execution_count=execution_count,
            value=value,
        )
        self._events.append(event)

    def filter(self, type: str | None = None, **kwargs) -> list:
        """Mock filter method - returns events matching type."""
        if type == "python_output":
            return list(self._events)
        return self._events


@pytest.fixture
def mockevent_manager():
    """Create a fresh MockEventManager for each test."""
    return MockEventManager()


class TestOutAccessorBasics:
    """Tests for basic OutAccessor functionality."""

    def test_empty_accessor(self, mockevent_manager):
        """OutAccessor should start empty."""
        from nooa.runtime.out_accessor import OutAccessor

        out = OutAccessor(event_manager=mockevent_manager)
        assert len(out) == 0

    def test_single_output(self, mockevent_manager):
        """OutAccessor should return stored output."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, {"data": "value"})
        out = OutAccessor(event_manager=mockevent_manager)

        assert len(out) == 1
        assert out[1] == {"data": "value"}

    def test_multiple_outputs(self, mockevent_manager):
        """OutAccessor should handle multiple outputs."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "first")
        mockevent_manager.add_output(2, "second")
        mockevent_manager.add_output(3, "third")
        out = OutAccessor(event_manager=mockevent_manager)

        assert len(out) == 3
        assert out[1] == "first"
        assert out[2] == "second"
        assert out[3] == "third"


class TestOutAccessorIndexing:
    """Tests for OutAccessor indexing."""

    def test_positive_indexing(self, mockevent_manager):
        """Out[n] should return output for execution n."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, {"result": 100})
        mockevent_manager.add_output(2, {"result": 200})
        out = OutAccessor(event_manager=mockevent_manager)

        assert out[1] == {"result": 100}
        assert out[2] == {"result": 200}

    def test_negative_indexing(self, mockevent_manager):
        """Out[-n] should return from end like list."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "first")
        mockevent_manager.add_output(2, "second")
        mockevent_manager.add_output(3, "third")
        out = OutAccessor(event_manager=mockevent_manager)

        assert out[-1] == "third"
        assert out[-2] == "second"
        assert out[-3] == "first"

    def test_index_error_empty(self, mockevent_manager):
        """Indexing empty OutAccessor should raise IndexError."""
        from nooa.runtime.out_accessor import OutAccessor

        out = OutAccessor(event_manager=mockevent_manager)

        # Positive index on empty should raise IndexError (no outputs recorded yet)
        with pytest.raises(IndexError, match="No outputs recorded yet"):
            _ = out[1]

        with pytest.raises(IndexError, match="No outputs recorded yet"):
            _ = out[-1]

    def test_key_error_missing_positive(self, mockevent_manager):
        """Indexing non-existent positive index should raise KeyError."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "exists")
        out = OutAccessor(event_manager=mockevent_manager)

        with pytest.raises(KeyError, match="No output for execution 99"):
            _ = out[99]

    def test_index_error_out_of_range_negative(self, mockevent_manager):
        """Negative index out of range should raise IndexError."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "only one")
        out = OutAccessor(event_manager=mockevent_manager)

        with pytest.raises(IndexError, match="Out index -5 out of range"):
            _ = out[-5]


class TestOutAccessorContains:
    """Tests for 'in' operator on OutAccessor."""

    def test_contains_existing(self, mockevent_manager):
        """'in' should return True for existing execution count."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "value")
        mockevent_manager.add_output(3, "value")
        out = OutAccessor(event_manager=mockevent_manager)

        assert 1 in out
        assert 3 in out

    def test_contains_missing(self, mockevent_manager):
        """'in' should return False for missing execution count."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "value")
        out = OutAccessor(event_manager=mockevent_manager)

        assert 2 not in out
        assert 99 not in out


class TestOutAccessorLast:
    """Tests for OutAccessor.last property."""

    def test_last_property(self, mockevent_manager):
        """last should return the most recent output."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, "first")
        out = OutAccessor(event_manager=mockevent_manager)
        assert out.last == "first"

        mockevent_manager.add_output(2, "second")
        assert out.last == "second"

    def test_last_empty_raises(self, mockevent_manager):
        """last should raise IndexError when empty."""
        from nooa.runtime.out_accessor import OutAccessor

        out = OutAccessor(event_manager=mockevent_manager)

        with pytest.raises(IndexError, match="No outputs recorded yet"):
            _ = out.last


class TestOutAccessorRepr:
    """Tests for OutAccessor string representation."""

    def test_repr_empty(self, mockevent_manager):
        """repr() should indicate no outputs."""
        from nooa.runtime.out_accessor import OutAccessor

        out = OutAccessor(event_manager=mockevent_manager)
        assert "no outputs" in repr(out).lower()

    def test_repr_with_outputs(self, mockevent_manager):
        """repr() should show output summaries."""
        from nooa.runtime.out_accessor import OutAccessor

        mockevent_manager.add_output(1, 42)
        mockevent_manager.add_output(2, "hello")
        out = OutAccessor(event_manager=mockevent_manager)

        result = repr(out)
        assert "Out[1]" in result
        assert "Out[2]" in result


class TestOutAccessorRealUseCases:
    """Tests for realistic OutAccessor usage patterns."""

    def test_jupyter_style_workflow(self, mockevent_manager):
        """Test Jupyter-style workflow: access previous outputs."""
        from nooa.runtime.out_accessor import OutAccessor

        out = OutAccessor(event_manager=mockevent_manager)

        # First execution: load data
        data = {"North": 15000, "South": 12000, "East": 18000}
        mockevent_manager.add_output(1, data)

        # Second execution: use Out[-1] to access previous result
        total = sum(out[-1].values())
        mockevent_manager.add_output(2, total)

        assert out[2] == 45000

        # Third execution: reference specific execution
        avg = out[2] / len(out[1])
        mockevent_manager.add_output(3, avg)

        assert out[3] == 15000.0

    def test_non_contiguous_execution_counts(self, mockevent_manager):
        """OutAccessor should handle non-contiguous execution counts."""
        from nooa.runtime.out_accessor import OutAccessor

        out = OutAccessor(event_manager=mockevent_manager)

        # Some executions may not produce output (e.g., assignments)
        mockevent_manager.add_output(2, "second")
        mockevent_manager.add_output(5, "fifth")

        assert out[2] == "second"
        assert out[5] == "fifth"
        assert 1 not in out
        assert 3 not in out

        # Negative indexing should still work based on order recorded
        assert out[-1] == "fifth"
        assert out[-2] == "second"
