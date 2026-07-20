# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded value rendering in plain_event_content and PlainBlockFormatter.

These cover the Out[n] code path in codeact_lite.py and the field renderer
in plain_formatter.py — both must be bounded to prevent OOM when agent code
returns a huge object.
"""

from typing import Any

from nooa.events import PythonOutput


class TestPlainEventContentBounded:
    """plain_event_content must bound the Out[n] value representation."""

    def _make_output(self, value: Any, execution_count: int = 1) -> PythonOutput:
        from nooa.events import ResultStatus

        return PythonOutput(
            tool_call_id="test_tc",
            execution_status=ResultStatus.COMPLETE,
            execution_count=execution_count,
            stdout="",
            stderr="",
            value=value,
        )

    def test_large_list_value_is_bounded(self) -> None:
        """Out[n] with a 1 M-element list must be bounded when event_format is set."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
        from nooa.strategies.codeact_lite import plain_event_content

        event = self._make_output(list(range(1_000_000)))
        result = plain_event_content(event, event_format=DEFAULT_TRUNCATION_CONFIG.event_format)
        assert "Out[1]:" in result
        assert len(result) < 1_000_000  # full repr would be ~7 MB

    def test_large_string_value_passes_through(self) -> None:
        """Block-level string truncation has been removed; large string Out[n]
        values now pass through verbatim. Per-field bounds for events come
        from spec() annotations on PythonOutput.value once wired (issue !158).
        """
        from nooa.strategies.codeact_lite import plain_event_content

        event = self._make_output("y" * 2_000_000)
        result = plain_event_content(event)
        assert "Out[1]:" in result
        # Full string passes through
        assert len(result) >= 2_000_000

    def test_small_value_preserved(self) -> None:
        """Normal small values are not affected."""
        from nooa.strategies.codeact_lite import plain_event_content

        event = self._make_output({"answer": 42})
        result = plain_event_content(event)
        assert "Out[1]:" in result
        assert "42" in result
        assert "truncation" not in result.lower()

    def test_none_value_not_shown(self) -> None:
        """None value produces no Out[n] line."""
        from nooa.events import ResultStatus
        from nooa.strategies.codeact_lite import plain_event_content

        event = PythonOutput(
            tool_call_id="tc",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="some output",
            stderr="",
            value=None,
        )
        result = plain_event_content(event)
        assert "Out[1]:" not in result


class TestPlainBlockFormatterBounded:
    """PlainBlockFormatter.format_event must bound non-string field values."""

    def test_large_non_string_value_is_bounded(self) -> None:
        """A PythonOutput whose value is a huge list must produce bounded output."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
        from nooa.events import PythonOutput, ResultStatus
        from nooa.plain_formatter import PlainBlockFormatter

        fmt = PlainBlockFormatter()
        event = PythonOutput(
            tool_call_id="tc",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="",
            stderr="",
            value=list(range(500_000)),
        )
        result = fmt.format_event(event, event_format=DEFAULT_TRUNCATION_CONFIG.event_format)
        assert len(result) < 1_000_000
