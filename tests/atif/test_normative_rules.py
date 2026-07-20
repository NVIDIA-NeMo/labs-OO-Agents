# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF v1.7 normative-rule unit tests.

One test per rule in :mod:`tests.atif.normative` (N1–N9).
"""

from __future__ import annotations

import pytest

from nooa.atif import (
    AgentSchema,
    FinalMetricsSchema,
    MetricsSchema,
    ObservationResultSchema,
    ObservationSchema,
    StepObject,
    SubagentTrajectoryRef,
    ToolCallSchema,
    Trajectory,
)
from tests.atif.normative import (
    NormativeRuleError,
    assert_atif_normative,
    check_copied_context_propagation,
    check_final_metrics_sum,
    check_iso_timestamp,
    check_joinability,
    check_sequential_step_ids,
    check_subagent_ref_resolvability,
    check_subagent_trajectory_id_uniqueness,
)


def _minimal_trajectory(**kwargs) -> Trajectory:
    defaults: dict = {
        "agent": AgentSchema(name="a", version="0"),
        "steps": [StepObject(step_id=1, source="user", message="hi")],
    }
    defaults.update(kwargs)
    return Trajectory(**defaults)


# ---------------------------------------------------------------------------
# N1 — sequential step_ids
# ---------------------------------------------------------------------------


class TestN1SequentialStepIds:
    def test_sequential_ids_pass(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(step_id=1, source="user", message="x"),
                StepObject(step_id=2, source="agent", message="y"),
                StepObject(step_id=3, source="user", message="z"),
            ],
        )
        check_sequential_step_ids(t)

    def test_gap_fails(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(step_id=1, source="user", message="x"),
                StepObject(step_id=3, source="user", message="z"),
            ],
        )
        with pytest.raises(NormativeRuleError, match="N1"):
            check_sequential_step_ids(t)

    def test_starts_at_zero_fails(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[StepObject(step_id=0, source="user", message="x")],
        )
        with pytest.raises(NormativeRuleError, match="N1"):
            check_sequential_step_ids(t)


# ---------------------------------------------------------------------------
# N2 — joinability
# ---------------------------------------------------------------------------


class TestN2Joinability:
    def test_paired_result_passes(self) -> None:
        step = StepObject(
            step_id=1,
            source="agent",
            message="",
            tool_calls=[ToolCallSchema(tool_call_id="call_x", function_name="f", arguments={})],
            observation=ObservationSchema(
                results=[ObservationResultSchema(source_call_id="call_x", content="ok")]
            ),
        )
        check_joinability(step)

    def test_orphan_source_call_id_fails(self) -> None:
        step = StepObject(
            step_id=1,
            source="agent",
            message="",
            tool_calls=[ToolCallSchema(tool_call_id="call_x", function_name="f", arguments={})],
            observation=ObservationSchema(
                results=[ObservationResultSchema(source_call_id="call_other", content="")]
            ),
        )
        with pytest.raises(NormativeRuleError, match="N2"):
            check_joinability(step)

    def test_orphan_tool_call_passes(self) -> None:
        """A tool_call without a matching observation result is allowed (in flight)."""
        step = StepObject(
            step_id=1,
            source="agent",
            message="",
            tool_calls=[ToolCallSchema(tool_call_id="call_x", function_name="f", arguments={})],
        )
        check_joinability(step)

    def test_result_with_null_source_call_id_passes(self) -> None:
        """Observation result without source_call_id (non-tool action) is allowed."""
        step = StepObject(
            step_id=1,
            source="agent",
            message="",
            observation=ObservationSchema(
                results=[ObservationResultSchema(source_call_id=None, content="env feedback")]
            ),
        )
        check_joinability(step)


# ---------------------------------------------------------------------------
# N3 — subagent trajectory_id uniqueness (recursive)
# ---------------------------------------------------------------------------


class TestN3SubagentUniqueness:
    def _child(self, tid: str) -> Trajectory:
        return Trajectory(
            trajectory_id=tid,
            agent=AgentSchema(name="c", version="0"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
        )

    def test_unique_ids_pass(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
            subagent_trajectories=[self._child("c1"), self._child("c2")],
        )
        check_subagent_trajectory_id_uniqueness(t)

    def test_nested_duplicate_fails(self) -> None:
        # Build via model_construct at every level to bypass Pydantic's
        # model_validator (which would catch the duplicate at construction
        # time). This verifies the recursive walker independently of the
        # built-in validator.
        inner = Trajectory.model_construct(
            schema_version="ATIF-v1.7",
            trajectory_id="parent",
            agent=AgentSchema(name="p", version="0"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
            subagent_trajectories=[self._child("grand-1"), self._child("grand-1")],
        )
        outer = Trajectory.model_construct(
            schema_version="ATIF-v1.7",
            agent=AgentSchema(name="a", version="0"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
            subagent_trajectories=[inner],
        )
        with pytest.raises(NormativeRuleError, match="N3"):
            check_subagent_trajectory_id_uniqueness(outer)


# ---------------------------------------------------------------------------
# N4 — is_copied_context propagation
# ---------------------------------------------------------------------------


class TestN4CopiedContext:
    def test_no_boundary_passes(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(step_id=1, source="user", message="x"),
                StepObject(step_id=2, source="agent", message="y"),
            ],
        )
        check_copied_context_propagation(t)

    def test_boundary_with_copied_pre_steps_passes(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(step_id=1, source="user", message="x", is_copied_context=True),
                StepObject(step_id=2, source="agent", message="y", is_copied_context=True),
                StepObject(
                    step_id=3,
                    source="system",
                    message="Context compaction performed",
                    extra={
                        "context_management": {
                            "type": "compaction",
                            "boundary": "replace",
                        }
                    },
                ),
                StepObject(step_id=4, source="user", message="continue"),
            ],
        )
        check_copied_context_propagation(t)

    def test_boundary_with_uncopied_pre_step_fails(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(step_id=1, source="user", message="x"),  # missing is_copied_context
                StepObject(
                    step_id=2,
                    source="system",
                    message="Context compaction performed",
                    extra={
                        "context_management": {
                            "type": "compaction",
                            "boundary": "replace",
                        }
                    },
                ),
            ],
        )
        with pytest.raises(NormativeRuleError, match="N4"):
            check_copied_context_propagation(t)


# ---------------------------------------------------------------------------
# N5 — deterministic dispatch (also enforced by schema, restated here)
# ---------------------------------------------------------------------------


class TestN5DeterministicDispatch:
    def test_zero_with_no_metrics_passes(self) -> None:
        # llm_call_count=0 + no metrics/reasoning = valid deterministic dispatch.
        step = StepObject(
            step_id=1,
            source="agent",
            message="",
            llm_call_count=0,
            tool_calls=[ToolCallSchema(tool_call_id="x", function_name="f", arguments={})],
        )
        # No NormativeRuleError expected (also passes Pydantic).
        from tests.atif.normative import check_deterministic_dispatch

        check_deterministic_dispatch(step)


# ---------------------------------------------------------------------------
# N6 — ISO 8601 timestamps
# ---------------------------------------------------------------------------


class TestN6IsoTimestamps:
    def test_z_suffix_passes(self) -> None:
        step = StepObject(step_id=1, source="user", message="x", timestamp="2025-10-11T10:30:00Z")
        check_iso_timestamp(step)

    def test_offset_passes(self) -> None:
        step = StepObject(
            step_id=1, source="user", message="x", timestamp="2025-10-11T10:30:00+00:00"
        )
        check_iso_timestamp(step)

    def test_malformed_fails(self) -> None:
        step = StepObject(step_id=1, source="user", message="x", timestamp="yesterday")
        with pytest.raises(NormativeRuleError, match="N6"):
            check_iso_timestamp(step)


# ---------------------------------------------------------------------------
# N8 — final_metrics totals
# ---------------------------------------------------------------------------


class TestN8FinalMetricsSum:
    def test_matching_totals_pass(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(
                    step_id=1,
                    source="agent",
                    message="x",
                    metrics=MetricsSchema(prompt_tokens=100, completion_tokens=20, cost_usd=0.001),
                ),
                StepObject(
                    step_id=2,
                    source="agent",
                    message="y",
                    metrics=MetricsSchema(prompt_tokens=50, completion_tokens=10, cost_usd=0.0005),
                ),
            ],
            final_metrics=FinalMetricsSchema(
                total_prompt_tokens=150,
                total_completion_tokens=30,
                total_cost_usd=0.0015,
            ),
        )
        check_final_metrics_sum(t)

    def test_mismatch_fails(self) -> None:
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(
                    step_id=1,
                    source="agent",
                    message="x",
                    metrics=MetricsSchema(prompt_tokens=100),
                ),
            ],
            final_metrics=FinalMetricsSchema(total_prompt_tokens=999),
        )
        with pytest.raises(NormativeRuleError, match="N8"):
            check_final_metrics_sum(t)

    def test_documented_mismatch_passes(self) -> None:
        """If `notes` documents a discrepancy, the rule tolerates it."""
        t = Trajectory(
            agent=AgentSchema(name="a", version="0"),
            steps=[
                StepObject(
                    step_id=1,
                    source="agent",
                    message="x",
                    metrics=MetricsSchema(prompt_tokens=100),
                ),
            ],
            notes="Some steps omitted from steps[] for size; totals reflect all turns.",
            final_metrics=FinalMetricsSchema(total_prompt_tokens=999),
        )
        check_final_metrics_sum(t)


# ---------------------------------------------------------------------------
# N9 — subagent_trajectory_ref resolvability
# ---------------------------------------------------------------------------


class TestN9RefResolvability:
    def test_embedded_ref_resolves(self) -> None:
        child = Trajectory(
            trajectory_id="child-1",
            agent=AgentSchema(name="c", version="0"),
            steps=[StepObject(step_id=1, source="user", message="hi")],
        )
        parent = Trajectory(
            agent=AgentSchema(name="p", version="0"),
            steps=[
                StepObject(
                    step_id=1,
                    source="agent",
                    message="",
                    observation=ObservationSchema(
                        results=[
                            ObservationResultSchema(
                                subagent_trajectory_ref=[
                                    SubagentTrajectoryRef(trajectory_id="child-1")
                                ]
                            )
                        ]
                    ),
                ),
            ],
            subagent_trajectories=[child],
        )
        check_subagent_ref_resolvability(parent)

    def test_dangling_ref_fails(self) -> None:
        parent = Trajectory(
            agent=AgentSchema(name="p", version="0"),
            steps=[
                StepObject(
                    step_id=1,
                    source="agent",
                    message="",
                    observation=ObservationSchema(
                        results=[
                            ObservationResultSchema(
                                subagent_trajectory_ref=[
                                    SubagentTrajectoryRef(trajectory_id="does-not-exist")
                                ]
                            )
                        ]
                    ),
                ),
            ],
        )
        with pytest.raises(NormativeRuleError, match="N9"):
            check_subagent_ref_resolvability(parent)

    def test_file_ref_form_skipped(self) -> None:
        """trajectory_path (external file) is not validated in-document."""
        parent = Trajectory(
            agent=AgentSchema(name="p", version="0"),
            steps=[
                StepObject(
                    step_id=1,
                    source="agent",
                    message="",
                    observation=ObservationSchema(
                        results=[
                            ObservationResultSchema(
                                subagent_trajectory_ref=[
                                    SubagentTrajectoryRef(trajectory_path="child.json")
                                ]
                            )
                        ]
                    ),
                ),
            ],
        )
        check_subagent_ref_resolvability(parent)


# ---------------------------------------------------------------------------
# Composite assert
# ---------------------------------------------------------------------------


class TestAssertComposite:
    def test_minimal_trajectory_passes(self) -> None:
        t = _minimal_trajectory()
        assert_atif_normative(t)

    def test_spec_iv_example_passes(self) -> None:
        from tests.atif.test_schema_validation import SPEC_IV_EXAMPLE

        # The spec example has metrics summing to 1120 prompt / 124 completion;
        # final_metrics matches by construction.
        t = Trajectory.model_validate(SPEC_IV_EXAMPLE)
        assert_atif_normative(t)
