# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PlainBlockFormatter — plain text event serialization."""

from nooa.context_blocks.formatter import FormatType
from nooa.context_blocks.models import ResolvedBlock


class TestPlainBlockFormatterFormatType:
    def test_format_type_is_plain(self):
        from nooa.plain_formatter import PlainBlockFormatter

        assert PlainBlockFormatter().format_type == FormatType.PLAIN


class TestPlainBlockFormatterFormatSystemBlocks:
    """System blocks (format()) use XML — same as XMLBlockFormatter."""

    def test_format_delegates_to_xml(self):
        from nooa.plain_formatter import PlainBlockFormatter

        formatter = PlainBlockFormatter()
        blocks = [ResolvedBlock(key="persona", content="You are helpful.")]
        result = formatter.format(blocks)

        assert len(result) == 1
        assert "<persona>" in result[0].content
        assert "You are helpful." in result[0].content


class TestPlainBlockFormatterFormatEvent:
    """format_event() renders each event type as clean plain text."""

    def test_task_renders_prompt(self):
        from nooa.events import Task
        from nooa.plain_formatter import PlainBlockFormatter

        event = Task(prompt="Analyze the data.")
        result = PlainBlockFormatter().format_event(event)

        assert result == "Analyze the data."
        assert "Task(" not in result  # no pformat repr

    def test_error_renders_content(self):
        from nooa.events import Error
        from nooa.plain_formatter import PlainBlockFormatter

        event = Error(content="NameError: name 'x' is not defined")
        result = PlainBlockFormatter().format_event(event)

        assert result == "NameError: name 'x' is not defined"

    def test_python_output_stdout_only(self):
        from nooa.context_blocks.events import ResultStatus
        from nooa.events import PythonOutput
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="42\n",
        )
        result = PlainBlockFormatter().format_event(event)

        assert "42" in result
        assert "PythonOutput(" not in result

    def test_python_output_with_error_only(self):
        """Error-only output shows fields as XML tags."""
        from nooa.context_blocks.events import ResultStatus
        from nooa.events import PythonOutput
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            stdout="",
            error="NameError: name 'foo' is not defined",
        )
        result = PlainBlockFormatter().format_event(event)

        assert "<execution_status>error</execution_status>" in result
        assert "<tool_call_id>tc_1</tool_call_id>" in result
        assert "<error>NameError" in result
        assert "[status]" not in result  # old format gone

    def test_python_output_with_stdout_and_error(self):
        """Multiple non-empty fields → <field>value</field> XML tags."""
        from nooa.context_blocks.events import ResultStatus
        from nooa.events import PythonOutput
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            stdout="partial output\n",
            error="NameError: name 'foo' is not defined",
        )
        result = PlainBlockFormatter().format_event(event)

        assert "<stdout>partial output" in result
        assert "<error>NameError" in result
        assert "[stdout]" not in result  # old format gone

    def test_python_output_with_value(self):
        from nooa.context_blocks.events import ResultStatus
        from nooa.events import PythonOutput
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="done\n",
            value=[1, 2, 3],
        )
        result = PlainBlockFormatter().format_event(event)

        assert "<stdout>done" in result
        assert "<value>" in result
        assert "1" in result and "2" in result and "3" in result

    def test_python_output_no_output(self):
        """No stdout/value/error — still shows status and tool_call_id as XML tags."""
        from nooa.context_blocks.events import ResultStatus
        from nooa.events import PythonOutput
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
        )
        result = PlainBlockFormatter().format_event(event)

        assert "<execution_status>complete</execution_status>" in result
        assert "<tool_call_id>tc_1</tool_call_id>" in result

    def test_generic_event_with_content_field(self):
        """Events with a 'content' field fall back to returning content directly."""
        from nooa.context_blocks.events import UserEvent
        from nooa.plain_formatter import PlainBlockFormatter

        event = UserEvent(content="Hello there")
        result = PlainBlockFormatter().format_event(event)

        assert result == "Hello there"
        assert "UserEvent(" not in result

    def test_python_output_captured_locals_not_shown(self):
        """captured_locals is infrastructure — must never appear in plain output."""
        from nooa.context_blocks.events import ResultStatus
        from nooa.events import PythonOutput
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="hello\n",
            captured_locals="x = 1, df = DataFrame(shape=(3,2))",
        )
        result = PlainBlockFormatter().format_event(event)

        assert "captured_locals" not in result
        assert "DataFrame" not in result
        assert "hello" in result  # stdout still shown
