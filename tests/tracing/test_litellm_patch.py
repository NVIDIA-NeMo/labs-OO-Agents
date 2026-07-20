# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the litellm null content patch."""

from unittest.mock import MagicMock, patch

from litellm.types.utils import Choices
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

from nooa.tracing._litellm_patch import (
    _patched_set_output_message_value,
    apply_litellm_patch,
)


def make_mock_result(content, reasoning_content=None):
    """Create a mock litellm result with the given content."""
    mock_message = MagicMock()
    mock_message.content = content
    # Explicitly set reasoning attributes to None to avoid MagicMock returning truthy values
    mock_message.reasoning_content = reasoning_content
    mock_message.reasoning = None

    # Use a MagicMock that passes isinstance check for Choices
    mock_choice = MagicMock(spec=Choices)
    mock_choice.message = mock_message

    mock_result = MagicMock()
    mock_result.choices = [mock_choice]

    return mock_result


class TestPatchedSetOutputMessageValue:
    """Tests for _patched_set_output_message_value."""

    def test_content_with_value(self):
        """When content is a string, span should get that value."""
        mock_span = MagicMock()
        mock_result = make_mock_result("hello world")

        _patched_set_output_message_value(mock_span, mock_result)

        mock_span.set_attribute.assert_called_once_with(SpanAttributes.OUTPUT_VALUE, "hello world")

    def test_content_empty_string(self):
        """When content is empty string, span should get empty string (not JSON dump)."""
        mock_span = MagicMock()
        mock_result = make_mock_result("")

        _patched_set_output_message_value(mock_span, mock_result)

        # Should be called with empty string, NOT the full JSON
        mock_span.set_attribute.assert_called_once_with(SpanAttributes.OUTPUT_VALUE, "")
        # Should NOT have called model_dump_json
        mock_result.model_dump_json.assert_not_called()

    def test_content_none(self):
        """When content is None, span should get empty string (not JSON dump)."""
        mock_span = MagicMock()
        mock_result = make_mock_result(None)

        _patched_set_output_message_value(mock_span, mock_result)

        # Should be called with empty string, NOT the full JSON
        mock_span.set_attribute.assert_called_once_with(SpanAttributes.OUTPUT_VALUE, "")
        # Should NOT have called model_dump_json
        mock_result.model_dump_json.assert_not_called()

    def test_no_choices_falls_back_to_json(self):
        """When there are no choices, should fall back to JSON dump."""
        mock_span = MagicMock()
        mock_result = MagicMock()
        mock_result.choices = []
        mock_result.model_dump_json.return_value = '{"test": "json"}'

        _patched_set_output_message_value(mock_span, mock_result)

        # Should have called model_dump_json for fallback
        mock_result.model_dump_json.assert_called_once()
        assert mock_span.set_attribute.call_count == 2
        mock_span.set_attribute.assert_any_call(SpanAttributes.OUTPUT_VALUE, '{"test": "json"}')
        mock_span.set_attribute.assert_any_call(
            SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
        )

    def test_choices_none_falls_back_to_json(self):
        """When choices is None, should fall back to JSON dump."""
        mock_span = MagicMock()
        mock_result = MagicMock()
        mock_result.choices = None
        mock_result.model_dump_json.return_value = '{"error": "no choices"}'

        _patched_set_output_message_value(mock_span, mock_result)

        mock_result.model_dump_json.assert_called_once()

    def test_reasoning_content_captured(self):
        """When reasoning_content is present, it should be captured in span."""
        mock_span = MagicMock()
        mock_result = make_mock_result("hello", reasoning_content="thinking step by step...")

        _patched_set_output_message_value(mock_span, mock_result)

        # Should have two calls: OUTPUT_VALUE and llm.reasoning_content
        assert mock_span.set_attribute.call_count == 2
        mock_span.set_attribute.assert_any_call(SpanAttributes.OUTPUT_VALUE, "hello")
        mock_span.set_attribute.assert_any_call("llm.reasoning_content", "thinking step by step...")

    def test_reasoning_not_captured_when_none(self):
        """When reasoning_content is None, only OUTPUT_VALUE should be set."""
        mock_span = MagicMock()
        mock_result = make_mock_result("hello", reasoning_content=None)

        _patched_set_output_message_value(mock_span, mock_result)

        # Should have only one call: OUTPUT_VALUE
        mock_span.set_attribute.assert_called_once_with(SpanAttributes.OUTPUT_VALUE, "hello")


class TestApplyLitellmPatch:
    """Tests for apply_litellm_patch."""

    def test_patch_is_applied(self):
        """Verify the patch replaces the original function."""
        with patch.dict("sys.modules", {"openinference.instrumentation.litellm": MagicMock()}):
            from openinference.instrumentation import litellm as mock_module

            apply_litellm_patch()

            assert mock_module._set_output_message_value == _patched_set_output_message_value

    def test_patch_handles_missing_module(self):
        """Verify the patch silently skips if litellm instrumentation is not available."""
        with patch.dict("sys.modules", {"openinference.instrumentation.litellm": None}):
            # Should not raise - silently skips
            apply_litellm_patch()
