# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OutAccessor: Jupyter-style access to execution outputs.

Provides `Out[n]` syntax for accessing previous execution results:
- Out[n] returns the actual Python object from execution n
- Out[-1] returns the last output (negative indexing supported)
- Only executions with non-None return values are stored

Backed by EventManager, which stores PythonOutput records.

Example in LLM-generated code:
    data = load_csv("sales.csv")
    data.groupby("region")["amount"].sum()
    # Out[1]: {'North': 15000, 'South': 12000}

    sum(Out[-1].values())  # Access last result
    # Out[2]: 45000
"""

from typing import TYPE_CHECKING, Any

from nooa.agentdoc import truncating_pformat
from nooa.events import PythonOutput

if TYPE_CHECKING:
    from nooa.runtime.event_manager import EventManager


class OutAccessor:
    """Jupyter-style accessor for execution outputs.

    Queries PythonOutput from EventManager to provide Out[n] access.
    Both CodeActStrategy and PurePythonStrategy use this unified approach.

    Supports both positive indexing (Out[1], Out[2]) and negative indexing (Out[-1]).

    Attributes:
        event_manager: EventManager reference for querying execution outputs.
    """

    def __init__(self, event_manager: "EventManager | None"):
        """Initialize OutAccessor.

        Args:
            event_manager: EventManager for querying PythonOutput records.
        """
        self.event_manager = event_manager

    def _get_output_events(self) -> list[PythonOutput]:
        """Get all PythonOutput with non-None values from events.

        Returns:
            List of PythonOutput with actual output values, in order of occurrence.
        """
        if self.event_manager is None:
            return []
        events = self.event_manager.filter(type="PythonOutput")
        return [e for e in events if isinstance(e, PythonOutput) and e.value is not None]

    def _get_event_by_index(self, index: int, events: list[PythonOutput]) -> PythonOutput:
        """Get event by index (positive or negative).

        Indexing semantics (Jupyter-style):
        - Out[n] (positive): Get execution #n by its execution_count field (sparse-safe)
        - Out[-n] (negative): Get nth-from-last output by list position

        Both use the same underlying lookup logic, but:
        - Positive indices require iteration because execution_count may be sparse
        - Negative indices can use direct list access since we want "nth from end"

        Args:
            index: Positive (1-based execution count) or negative (from end).
            events: List of PythonOutput with non-None values.

        Returns:
            The matching event.

        Raises:
            IndexError: If negative index is out of range.
            KeyError: If no output exists for the given execution_count.
        """
        if index < 0:
            # Negative: direct list access (Out[-1] = last, Out[-2] = second-to-last)
            try:
                return events[index]
            except IndexError:
                raise IndexError(
                    f"Out index {index} out of range (have {len(events)} outputs)"
                ) from None
        else:
            # Positive: search by execution_count (Out[3] = execution #3)
            # Must iterate because execution_count can be sparse (e.g., 1, 3, 5)
            for event in events:
                if event.execution_count == index:
                    return event
            raise KeyError(f"No output for execution {index}")

    def __getitem__(self, index: int) -> Any:
        """Get execution output by index.

        Indexing semantics (Jupyter-style):
        - Out[n] (positive): Get output from execution #n (by execution_count field)
        - Out[-n] (negative): Get nth-from-last output (by list position)

        Note: Positive indices must search by execution_count because execution
        counts may be sparse (e.g., 1, 3, 5 if some executions returned None).
        Negative indices use direct list position for intuitive "last N" access.

        Args:
            index: Positive (1-based execution count) or negative (from end).

        Returns:
            The stored execution result.

        Raises:
            IndexError: If index is out of range.
            KeyError: If no output exists for that execution count.
        """
        events = self._get_output_events()
        if not events:
            raise IndexError("No outputs recorded yet")

        event = self._get_event_by_index(index, events)
        return event.value

    def __len__(self) -> int:
        """Number of recorded outputs."""
        return len(self._get_output_events())

    def __contains__(self, index: int) -> bool:
        """Check if an execution count has recorded output."""
        events = self._get_output_events()
        return any(event.execution_count == index for event in events)

    def __repr__(self) -> str:
        events = self._get_output_events()
        if not events:
            return "Out (no outputs yet)"

        items = []
        for event in events:
            v = event.value
            items.append(f"Out[{event.execution_count}]: {truncating_pformat(v, max_chars=500)}")
        return "\n".join(items)

    @property
    def last(self) -> Any:
        """Get the last recorded output (convenience property)."""
        events = self._get_output_events()
        if not events:
            raise IndexError("No outputs recorded yet")
        return events[-1].value
