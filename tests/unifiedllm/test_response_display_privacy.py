# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public formatting must not turn nested provider state into readable text."""

import pytest
from pydantic import BaseModel, Field

from nooa.agentdoc import pformat
from nooa.context_blocks import MarkdownBlockFormatter, XMLBlockFormatter
from nooa.events import PythonOutput, ResultStatus
from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, ToolCall
from nooa.runtime.event_manager import EventManager
from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.tracing._secret_scrubber import scrub_value

SECRET = "private-provider-state-12345"


def response():
    return LLMResponse(
        parts=(
            AssistantReasoning(text="readable reasoning", native={"encrypted_content": SECRET}),
            AssistantText(text="readable answer"),
            ToolCall(id="call", name="public_tool", arguments="{}", native={"signature": SECRET}),
        )
    )


def python_output(value):
    return PythonOutput(
        value=value, tool_call_id="outer", execution_count=1, execution_status=ResultStatus.COMPLETE
    )


@pytest.mark.parametrize(
    "wrap", [lambda r: r, lambda r: [r], lambda r: {"a": r}, lambda r: r.parts, python_output]
)
def test_nested_display_keeps_public_text_only(wrap):
    value = wrap(response())
    for rendered in (pformat(value), OpenInferenceHooks._safe_serialize(value)):
        assert SECRET not in rendered
        assert "readable answer" in rendered
        assert "readable reasoning" in rendered
        assert "public_tool" in rendered


@pytest.mark.parametrize("formatter", [MarkdownBlockFormatter, XMLBlockFormatter])
def test_python_output_does_not_reintroduce_native_state_into_model_context(formatter):
    rendered = formatter().format_event(python_output(response()))
    assert SECRET not in rendered
    assert "readable answer" in rendered


def test_search_keeps_nested_models_public():
    text = EventManager()._get_searchable_text(python_output(response()))
    assert SECRET not in text
    assert "readable answer" in text


@pytest.mark.parametrize("wrap", [lambda r: r, lambda r: [r]])
def test_public_content_retains_no_truncation_rule(wrap):
    text = "answer" * 1000
    assert text in pformat(wrap(LLMResponse(parts=(AssistantText(text=text),))), max_string=20)


def test_nested_pydantic_fields_obey_repr_and_exclude():
    class Value(BaseModel):
        public: str = "visible"
        private: str = Field(default=SECRET, repr=False)
        excluded: str = Field(default=SECRET, exclude=True)

    rendered = pformat([Value()])
    assert SECRET not in rendered
    assert "visible" in rendered


@pytest.mark.parametrize(
    "value",
    [
        {"reasoningText": {"text": "visible", "signature": SECRET}},
        {"text": "visible", "inline_thought_signature": SECRET},
    ],
)
def test_scrubber_covers_nested_and_inline_signatures(value):
    scrubbed = str(scrub_value(value))
    assert SECRET not in scrubbed
    assert "visible" in scrubbed
