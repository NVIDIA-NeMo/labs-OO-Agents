# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that PythonOutput.stdout and .stderr are NOT repr-truncated by pformat.

These fields are already truncated at capture time by TruncatingIO.
Repr-level truncation double-truncates them, losing content the LLM needs to see.
"""

from nooa.agentdoc import pformat
from nooa.context_blocks import ResultStatus
from nooa.events import PythonOutput


def _make_event(stdout: str = "", stderr: str = "") -> PythonOutput:
    return PythonOutput(
        tool_call_id="tc_test",
        execution_status=ResultStatus.COMPLETE,
        execution_count=1,
        stdout=stdout,
        stderr=stderr,
    )


class TestPythonOutputNotReprTruncated:
    """PythonOutput stdout/stderr must survive pformat without string truncation."""

    def test_long_stdout_not_truncated(self):
        """A 15k-char stdout must appear in full in pformat output."""
        long_output = "x" * 15_000
        event = _make_event(stdout=long_output)
        rendered = pformat(event, max_string=10_000)
        # The full stdout must be present — no str(len=...) truncation marker
        assert long_output in rendered, (
            f"stdout was truncated by pformat: got {len(rendered)} chars, "
            f"expected full 15000-char string"
        )
        assert "str(len=" not in rendered

    def test_long_stderr_not_truncated(self):
        """A 15k-char stderr must appear in full in pformat output."""
        long_output = "y" * 15_000
        event = _make_event(stderr=long_output)
        rendered = pformat(event, max_string=10_000)
        assert long_output in rendered, (
            f"stderr was truncated by pformat: got {len(rendered)} chars, "
            f"expected full 15000-char string"
        )

    def test_other_string_fields_still_truncated(self):
        """Non-exempt string fields should still respect max_string."""
        event_with_long_error = PythonOutput(
            tool_call_id="tc_test",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            error="z" * 15_000,
        )
        rendered = pformat(event_with_long_error, max_string=10_000)
        # error field should still be truncated
        assert "z" * 15_000 not in rendered
