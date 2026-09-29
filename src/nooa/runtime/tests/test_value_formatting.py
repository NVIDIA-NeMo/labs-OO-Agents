# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded PlainBlockFormatter values must not expand huge agent objects."""


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
