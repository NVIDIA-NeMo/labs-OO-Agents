# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF v1.7 schema validation.

Loads the §IV example trajectory from the spec into our Pydantic
models and asserts no validation errors. Also tests the structural
conditional-field rules (ContentPart, SubagentTrajectoryRef,
agent-only-fields, deterministic-dispatch).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nooa.atif import (
    AgentSchema,
    ContentPart,
    ImageSource,
    MetricsSchema,
    StepObject,
    SubagentTrajectoryRef,
    ToolCallSchema,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Spec §IV — the canonical example trajectory must validate end-to-end
# ---------------------------------------------------------------------------

SPEC_IV_EXAMPLE: dict = {
    "schema_version": "ATIF-v1.7",  # spec example shows v1.5; we upgrade for our validator
    "session_id": "025B810F-B3A2-4C67-93C0-FE7A142A947A",
    "agent": {
        "name": "harbor-agent",
        "version": "1.0.0",
        "model_name": "gemini-2.5-flash",
        "tool_definitions": [
            {
                "type": "function",
                "function": {
                    "name": "financial_search",
                    "description": "Search for financial data for a given stock ticker",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "description": "Stock ticker symbol"},
                            "metric": {
                                "type": "string",
                                "description": "The financial metric to retrieve (e.g., price, volume)",
                            },
                        },
                        "required": ["ticker", "metric"],
                    },
                },
            }
        ],
        "extra": {},
    },
    "notes": "Initial test trajectory for financial data retrieval using a single-hop ReAct pattern.",
    "extra": {},
    "final_metrics": {
        "total_prompt_tokens": 1120,
        "total_completion_tokens": 124,
        "total_cached_tokens": 200,
        "total_cost_usd": 0.00078,
        "total_steps": 3,
        "extra": {},
    },
    "steps": [
        {
            "step_id": 1,
            "timestamp": "2025-10-11T10:30:00Z",
            "source": "user",
            "message": "What is the current trading price of Alphabet (GOOGL)?",
            "extra": {},
        },
        {
            "step_id": 2,
            "timestamp": "2025-10-11T10:30:02Z",
            "source": "agent",
            "model_name": "gemini-2.5-flash",
            "reasoning_effort": "medium",
            "message": "I will search for the current trading price and volume for GOOGL.",
            "reasoning_content": (
                "The request requires two data points: the current stock price and the latest volume data."
            ),
            "tool_calls": [
                {
                    "tool_call_id": "call_price_1",
                    "function_name": "financial_search",
                    "arguments": {"ticker": "GOOGL", "metric": "price"},
                },
                {
                    "tool_call_id": "call_volume_2",
                    "function_name": "financial_search",
                    "arguments": {"ticker": "GOOGL", "metric": "volume"},
                },
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "call_price_1",
                        "content": "GOOGL is currently trading at $185.35 (Close: 10/11/2025)",
                    },
                    {
                        "source_call_id": "call_volume_2",
                        "content": "GOOGL volume: 1.5M shares traded.",
                    },
                ]
            },
            "metrics": {
                "prompt_tokens": 520,
                "completion_tokens": 80,
                "cached_tokens": 200,
                "cost_usd": 0.00045,
            },
        },
        {
            "step_id": 3,
            "timestamp": "2025-10-11T10:30:05Z",
            "source": "agent",
            "model_name": "gemini-2.5-flash",
            "reasoning_effort": "low",
            "message": (
                "As of October 11, 2025, Alphabet (GOOGL) is trading at $185.35 "
                "with a volume of 1.5M shares traded."
            ),
            "reasoning_content": (
                "The previous step retrieved all necessary data. "
                "I will now format this into a final conversational response."
            ),
            "metrics": {
                "prompt_tokens": 600,
                "completion_tokens": 44,
                "completion_token_ids": [1722, 310, 5533],
                "logprobs": [-0.1, -0.05, -0.02],
                "cost_usd": 0.00033,
                "extra": {"reasoning_tokens": 12},
            },
        },
    ],
}


class TestSpecExample:
    """The spec §IV example must round-trip through our schema."""

    def test_validates(self) -> None:
        traj = Trajectory.model_validate(SPEC_IV_EXAMPLE)
        assert traj.schema_version == "ATIF-v1.7"
        assert traj.agent.name == "harbor-agent"
        assert len(traj.steps) == 3
        agent_step = traj.steps[1]
        assert agent_step.source == "agent"
        assert agent_step.tool_calls is not None
        assert len(agent_step.tool_calls) == 2
        assert agent_step.observation is not None
        assert len(agent_step.observation.results) == 2

    def test_round_trips_through_json(self) -> None:
        original = Trajectory.model_validate(SPEC_IV_EXAMPLE)
        as_json = original.model_dump_json(exclude_none=True)
        reloaded = Trajectory.model_validate_json(as_json)
        assert reloaded == original


# ---------------------------------------------------------------------------
# ContentPart conditional fields
# ---------------------------------------------------------------------------


class TestContentPart:
    def test_text_requires_text_field(self) -> None:
        with pytest.raises(ValidationError, match="requires `text`"):
            ContentPart(type="text")  # no text

    def test_text_forbids_source(self) -> None:
        with pytest.raises(ValidationError, match="must omit `source`"):
            ContentPart(
                type="text",
                text="x",
                source=ImageSource(media_type="image/png", path="a.png"),
            )

    def test_image_requires_source(self) -> None:
        with pytest.raises(ValidationError, match="requires `source`"):
            ContentPart(type="image")

    def test_image_forbids_text(self) -> None:
        with pytest.raises(ValidationError, match="must omit `text`"):
            ContentPart(
                type="image",
                text="caption",
                source=ImageSource(media_type="image/png", path="a.png"),
            )

    def test_text_happy_path(self) -> None:
        cp = ContentPart(type="text", text="hello")
        assert cp.text == "hello"

    def test_image_happy_path(self) -> None:
        cp = ContentPart(
            type="image", source=ImageSource(media_type="image/png", path="images/x.png")
        )
        assert cp.source is not None and cp.source.media_type == "image/png"


# ---------------------------------------------------------------------------
# SubagentTrajectoryRef resolution
# ---------------------------------------------------------------------------


class TestSubagentTrajectoryRef:
    def test_requires_at_least_one_resolver(self) -> None:
        with pytest.raises(ValidationError, match="MUST set at least one"):
            SubagentTrajectoryRef(session_id="run-1")

    def test_trajectory_id_alone_is_valid(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_id="child-1")
        assert ref.trajectory_id == "child-1"

    def test_trajectory_path_alone_is_valid(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_path="/tmp/child.json")
        assert ref.trajectory_path == "/tmp/child.json"

    def test_both_resolvers_is_valid(self) -> None:
        ref = SubagentTrajectoryRef(trajectory_id="x", trajectory_path="/y")
        assert ref.trajectory_id == "x" and ref.trajectory_path == "/y"

    def test_session_id_is_informational_only(self) -> None:
        # session_id alone is NOT enough (v1.7 breaking change vs v1.6).
        with pytest.raises(ValidationError):
            SubagentTrajectoryRef(session_id="run-1")


# ---------------------------------------------------------------------------
# Agent-only step fields
# ---------------------------------------------------------------------------


class TestAgentOnlyFields:
    @pytest.mark.parametrize(
        "field, value",
        [
            ("model_name", "gpt"),
            ("reasoning_effort", "high"),
            ("reasoning_content", "thinking..."),
            (
                "tool_calls",
                [{"tool_call_id": "x", "function_name": "f", "arguments": {}}],
            ),
            ("metrics", {"prompt_tokens": 10}),
        ],
    )
    def test_rejected_on_user_step(self, field: str, value) -> None:
        kwargs = {"step_id": 1, "source": "user", "message": "hi", field: value}
        with pytest.raises(ValidationError, match=f"StepObject.{field}"):
            StepObject(**kwargs)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("model_name", "gpt"),
            ("reasoning_effort", "high"),
            ("reasoning_content", "thinking..."),
            (
                "tool_calls",
                [{"tool_call_id": "x", "function_name": "f", "arguments": {}}],
            ),
            ("metrics", {"prompt_tokens": 10}),
        ],
    )
    def test_rejected_on_system_step(self, field: str, value) -> None:
        kwargs = {"step_id": 1, "source": "system", "message": "hi", field: value}
        with pytest.raises(ValidationError, match=f"StepObject.{field}"):
            StepObject(**kwargs)

    def test_allowed_on_agent_step(self) -> None:
        step = StepObject(
            step_id=1,
            source="agent",
            message="ok",
            model_name="gpt",
            reasoning_effort="high",
            reasoning_content="thinking...",
            metrics=MetricsSchema(prompt_tokens=10),
        )
        assert step.model_name == "gpt"


# ---------------------------------------------------------------------------
# llm_call_count = 0 (deterministic dispatch) constraints
# ---------------------------------------------------------------------------


class TestDeterministicDispatch:
    def test_zero_llm_calls_forbids_metrics(self) -> None:
        with pytest.raises(ValidationError, match="deterministic dispatch"):
            StepObject(
                step_id=1,
                source="agent",
                message="dispatched",
                llm_call_count=0,
                metrics=MetricsSchema(prompt_tokens=5),
            )

    def test_zero_llm_calls_forbids_reasoning(self) -> None:
        with pytest.raises(ValidationError, match="deterministic dispatch"):
            StepObject(
                step_id=1,
                source="agent",
                message="dispatched",
                llm_call_count=0,
                reasoning_content="why",
            )

    def test_zero_llm_calls_allows_tool_calls(self) -> None:
        step = StepObject(
            step_id=1,
            source="agent",
            message="dispatched",
            llm_call_count=0,
            tool_calls=[ToolCallSchema(tool_call_id="x", function_name="f", arguments={})],
        )
        assert step.llm_call_count == 0


# ---------------------------------------------------------------------------
# Trajectory-level subagent uniqueness
# ---------------------------------------------------------------------------


class TestSubagentEmbeddingUniqueness:
    def _make_child(self, *, trajectory_id: str | None) -> Trajectory:
        return Trajectory(
            trajectory_id=trajectory_id,
            agent=AgentSchema(name="child", version="0.1"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
        )

    def test_embedded_subagent_requires_trajectory_id(self) -> None:
        with pytest.raises(ValidationError, match="trajectory_id"):
            Trajectory(
                trajectory_id="root-1",
                agent=AgentSchema(name="parent", version="0.1"),
                steps=[StepObject(step_id=1, source="user", message="hi")],
                subagent_trajectories=[self._make_child(trajectory_id=None)],
            )

    def test_unique_trajectory_ids_required(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate trajectory_id"):
            Trajectory(
                trajectory_id="root-1",
                agent=AgentSchema(name="parent", version="0.1"),
                steps=[StepObject(step_id=1, source="user", message="hi")],
                subagent_trajectories=[
                    self._make_child(trajectory_id="child-A"),
                    self._make_child(trajectory_id="child-A"),
                ],
            )

    def test_distinct_ids_validate(self) -> None:
        parent = Trajectory(
            trajectory_id="root-1",
            agent=AgentSchema(name="parent", version="0.1"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
            subagent_trajectories=[
                self._make_child(trajectory_id="child-A"),
                self._make_child(trajectory_id="child-B"),
            ],
        )
        assert parent.subagent_trajectories is not None
        assert len(parent.subagent_trajectories) == 2


# ---------------------------------------------------------------------------
# JSON round-trip + exclude_none cleanliness
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_minimal_trajectory_round_trips(self) -> None:
        original = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
        )
        as_json = original.model_dump_json(exclude_none=True)
        # No null fields in serialized form.
        parsed = json.loads(as_json)
        assert "session_id" not in parsed
        assert "trajectory_id" not in parsed
        reloaded = Trajectory.model_validate_json(as_json)
        assert reloaded == original

    def test_extra_fields_forbidden(self) -> None:
        # Round-trip safety: schema rejects unknown fields so consumers can't
        # accidentally rely on producer-specific extensions outside `extra`.
        with pytest.raises(ValidationError, match="Extra inputs"):
            Trajectory.model_validate(
                {
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": "a", "version": "0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
                    "unknown_field": "should-fail",
                }
            )
