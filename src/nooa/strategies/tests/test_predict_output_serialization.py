# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PredictStrategy output serialization modes."""

from types import SimpleNamespace

from pydantic import BaseModel

from nooa.config.strategy_config import PredictConfig
from nooa.context_blocks import ResultStatus, ToolCallEvent
from nooa.events import LLMOutput
from nooa.runtime.event_manager import EventManager
from nooa.strategies.predict import PredictStrategy


class Payload(BaseModel):
    value: str


def test_tool_call_mode_replaces_llm_output_with_predict_return_result():
    """Verify tool_call mode replaces LLMOutput with synthetic return_result."""
    strategy = PredictStrategy(PredictConfig(output_serialization="tool_call"))
    event_manager = EventManager()
    event_id = event_manager.add(LLMOutput(content='{"value":"hello"}'))

    strategy._replace_with_tool_call(
        SimpleNamespace(event_manager=event_manager),
        event_id,
        Payload(value="hello"),
    )

    events = event_manager.values()
    assert len(events) == 1

    event = events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.name == "return_result"
    assert event.tool_call_id.startswith("predict_")
    assert event.arguments == {"result": {"value": "hello"}}
    assert event.result is not None
    assert event.result.tool_call_id == event.tool_call_id
    assert event.result.content == "Result accepted."
    assert event.result.result_status == ResultStatus.COMPLETE
    assert event.metadata == {"synthetic": True, "synthetic_type": "predict_return_result"}


def test_jsonable_converts_nested_predict_results_for_tool_arguments():
    """Verify nested Predict results become JSON-compatible tool arguments."""
    strategy = PredictStrategy(PredictConfig(output_serialization="tool_call"))

    assert strategy._jsonable({"payloads": [Payload(value="a"), Payload(value="b")]}) == {
        "payloads": [{"value": "a"}, {"value": "b"}]
    }


def test_jsonable_sorts_sets_for_deterministic_tool_arguments():
    """Verify set-valued Predict results serialize with deterministic ordering."""
    strategy = PredictStrategy(PredictConfig(output_serialization="tool_call"))

    assert strategy._jsonable({"items": {"b", "a", "c"}}) == {"items": ["a", "b", "c"]}


def test_default_output_serialization_is_existing_event_behavior():
    """Verify Predict keeps existing LLMOutput event serialization by default."""
    assert PredictConfig().output_serialization == "event"
    assert PredictStrategy().config.output_serialization == "event"
