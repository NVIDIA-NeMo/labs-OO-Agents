# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that certain event fields are never truncated by pformat max_string.

Fields annotated with spec(max_string=None) should render their full content
regardless of the max_string kwarg passed to pformat(). This matches the
existing behavior of PythonOutput.stdout/stderr.
"""

from nooa.agentdoc import pformat
from nooa.events import LLMOutput, PythonOutput, ResultStatus, Summary, Task

LONG_STRING = "x" * 20_000
MAX_STRING = 100  # Aggressively low to verify the override works


class TestEventMaxStringOverride:
    """Fields with spec(max_string=None) must not be truncated."""

    def test_summary_text_not_truncated(self):
        """Summary.summary_text is already condensed — truncating defeats its purpose."""
        event = Summary(
            summary_tag="1..50",
            replaced_range=(1, 50),
            summary_text=LONG_STRING,
        )
        rendered = pformat(event, max_string=MAX_STRING)
        assert LONG_STRING in rendered

    def test_task_prompt_not_truncated(self):
        """Task.prompt carries critical instructions — must never be truncated."""
        event = Task(prompt=LONG_STRING)
        rendered = pformat(event, max_string=MAX_STRING)
        assert LONG_STRING in rendered

    def test_llm_output_content_not_truncated(self):
        """LLMOutput.content is bounded by token limits — truncating hides previous code."""
        event = LLMOutput(content=LONG_STRING)
        rendered = pformat(event, max_string=MAX_STRING)
        assert LONG_STRING in rendered

    def test_python_output_stdout_not_truncated(self):
        """Existing behavior — regression guard."""
        event = PythonOutput(
            tool_call_id="test",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout=LONG_STRING,
        )
        rendered = pformat(event, max_string=MAX_STRING)
        assert LONG_STRING in rendered

    def test_python_output_stderr_not_truncated(self):
        """Existing behavior — regression guard."""
        event = PythonOutput(
            tool_call_id="test",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stderr=LONG_STRING,
        )
        rendered = pformat(event, max_string=MAX_STRING)
        assert LONG_STRING in rendered


class TestOtherFieldsStillTruncated:
    """Fields without spec(max_string=None) should still respect the limit."""

    def test_summary_doc_is_truncated(self):
        event = Summary(
            summary_tag="1..50",
            replaced_range=(1, 50),
            summary_text="short",
            doc=LONG_STRING,
        )
        rendered = pformat(event, max_string=MAX_STRING)
        # doc field should be truncated — full string should NOT appear
        assert LONG_STRING not in rendered
