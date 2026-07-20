# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normative-rule validators for ATIF v1.7 trajectories.

These rules are MUST/MUST-NOT requirements from the spec that are not
captured by the Pydantic structural schema (which validates only field
shapes). They are:

- N1: ``step_id`` sequential from 1 (no gaps, no duplicates).
- N2: Joinability — every ``tool_calls[i].tool_call_id`` in a step is
      referenceable as ``observation.results[j].source_call_id`` in
      the SAME step (orphan tool_call is allowed; orphan
      source_call_id MUST point back).
- N3: ``trajectory_id`` unique within each ``subagent_trajectories[]``
      array (verified for embedded subagents; root validator handles
      this for top-level — but we also recurse).
- N4: ``is_copied_context = True`` propagation — when a system step
      with ``extra.context_management.boundary == "replace"`` exists,
      every step BEFORE it (in step_id order) MUST have
      ``is_copied_context == True``.
- N5: ``llm_call_count = 0`` on ``source="agent"`` ⇒ ``metrics`` and
      ``reasoning_content`` absent (also enforced by schema, but
      restated here as a normative-rule check).
- N6: Timestamps parse as ISO 8601 (Python datetime).
- N7: ``message`` field is present on every step (Pydantic enforces
      it's not None; we also check that empty-string is preserved).
- N8: ``final_metrics.total_*_tokens`` equals the sum of per-step
      ``metrics`` (when present) — OR ``notes`` documents otherwise.
- N9: Sub-trajectory ``trajectory_id`` resolvability — every
      ``subagent_trajectory_ref.trajectory_id`` in observation results
      MUST point at an entry in some ancestor's
      ``subagent_trajectories[]``.

Used by:
- Producer tests (exporter outputs satisfy all rules).
- The exporter itself (may call these to fail fast on programming
  errors).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from nooa.atif import StepObject, Trajectory


class NormativeRuleError(AssertionError):
    """A trajectory violates an ATIF v1.7 normative rule."""


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------


def check_sequential_step_ids(traj: Trajectory) -> None:
    """N1: step_id is sequential starting at 1."""
    for i, step in enumerate(traj.steps, start=1):
        if step.step_id != i:
            raise NormativeRuleError(
                f"N1: step_id at index {i - 1} is {step.step_id}, expected {i} "
                "(sequential from 1, no gaps)."
            )


def check_joinability(step: StepObject) -> None:
    """N2: every source_call_id in an observation joins back to a tool_call_id.

    Orphan tool_calls (no observation result) are permitted — they
    represent tool calls whose result is not yet known or not
    captured. Orphan source_call_ids (point at a non-existent tool
    call) are NOT permitted within the same step.
    """
    if step.observation is None:
        return
    tool_call_ids = {tc.tool_call_id for tc in (step.tool_calls or [])}
    for j, result in enumerate(step.observation.results):
        if result.source_call_id is None:
            continue
        if result.source_call_id not in tool_call_ids:
            raise NormativeRuleError(
                f"N2: step_id={step.step_id} observation.results[{j}].source_call_id="
                f"{result.source_call_id!r} does not match any tool_calls[].tool_call_id "
                f"in the same step. tool_call_ids in this step: {sorted(tool_call_ids)}."
            )


def check_subagent_trajectory_id_uniqueness(traj: Trajectory) -> None:
    """N3: trajectory_id unique within each subagent_trajectories[] array (recursive)."""
    # Top-level uniqueness is validated by the Pydantic model_validator;
    # we recurse here to make sure nested embeddings also satisfy it.
    if traj.subagent_trajectories:
        seen: set[str] = set()
        for child in traj.subagent_trajectories:
            if child.trajectory_id is None:
                raise NormativeRuleError("N3: embedded subagent missing trajectory_id.")
            if child.trajectory_id in seen:
                raise NormativeRuleError(
                    f"N3: duplicate trajectory_id {child.trajectory_id!r} "
                    f"in {traj.trajectory_id!r}.subagent_trajectories[]."
                )
            seen.add(child.trajectory_id)
            check_subagent_trajectory_id_uniqueness(child)


def check_copied_context_propagation(traj: Trajectory) -> None:
    """N4: every step before a replace-boundary system step is copied."""
    boundary_index: int | None = None
    for i, step in enumerate(traj.steps):
        if step.source != "system":
            continue
        extra = step.extra or {}
        cm = extra.get("context_management") if isinstance(extra, dict) else None
        if isinstance(cm, dict) and cm.get("boundary") == "replace":
            boundary_index = i
            break
    if boundary_index is None:
        return
    for i in range(boundary_index):
        step = traj.steps[i]
        if not step.is_copied_context:
            raise NormativeRuleError(
                f"N4: step_id={step.step_id} precedes a replace-boundary "
                f"(step_id={traj.steps[boundary_index].step_id}) but is_copied_context "
                f"is not True (got {step.is_copied_context!r}). Spec §VII normative."
            )


def check_deterministic_dispatch(step: StepObject) -> None:
    """N5: llm_call_count=0 on source='agent' ⇒ no metrics, no reasoning_content."""
    if step.source != "agent":
        return
    if step.llm_call_count != 0:
        return
    if step.metrics is not None:
        raise NormativeRuleError(
            f"N5: step_id={step.step_id} has llm_call_count=0 but `metrics` is present."
        )
    if step.reasoning_content is not None:
        raise NormativeRuleError(
            f"N5: step_id={step.step_id} has llm_call_count=0 but `reasoning_content` is present."
        )


def check_iso_timestamp(step: StepObject) -> None:
    """N6: timestamps parse as ISO 8601."""
    if step.timestamp is None:
        return
    try:
        # Strip trailing 'Z' which Python <3.11 didn't accept in fromisoformat.
        ts = step.timestamp.rstrip("Z").replace("Z", "")
        datetime.fromisoformat(ts)
    except ValueError as exc:
        raise NormativeRuleError(
            f"N6: step_id={step.step_id} timestamp {step.timestamp!r} is not ISO 8601: {exc}"
        ) from exc


def check_message_present(step: StepObject) -> None:
    """N7: every step has a message field (may be empty)."""
    # Pydantic enforces message is not None (required field), but a non-string
    # falsy value (e.g. None) wouldn't validate. This is a belt-and-braces
    # check so producer tests fail with a clear message rather than a
    # ValidationError from Pydantic.
    if step.message is None:
        raise NormativeRuleError(f"N7: step_id={step.step_id} has no message field (None).")


def check_final_metrics_sum(traj: Trajectory) -> None:
    """N8: final_metrics.total_*_tokens equals sum of per-step metrics, or notes documents otherwise.

    Tolerates rounding for cost_usd (within 1e-6 USD).
    """
    if traj.final_metrics is None:
        return
    fm = traj.final_metrics
    if traj.notes:
        # Producer documented a discrepancy; trust them.
        return
    agent_metrics = [s.metrics for s in traj.steps if s.source == "agent" and s.metrics is not None]

    def _sum(key: str) -> int | None:
        values = [getattr(m, key) for m in agent_metrics if getattr(m, key) is not None]
        return sum(values) if values else None

    def _sum_cost() -> float | None:
        values = [m.cost_usd for m in agent_metrics if m.cost_usd is not None]
        return sum(values) if values else None

    for fm_field, per_step_field in [
        ("total_prompt_tokens", "prompt_tokens"),
        ("total_completion_tokens", "completion_tokens"),
        ("total_cached_tokens", "cached_tokens"),
    ]:
        fm_value = getattr(fm, fm_field)
        if fm_value is None:
            continue
        expected = _sum(per_step_field)
        if expected is not None and fm_value != expected:
            raise NormativeRuleError(
                f"N8: final_metrics.{fm_field}={fm_value} != sum of per-step "
                f"metrics.{per_step_field}={expected}. Document with `notes` if "
                "intentional."
            )
    if fm.total_cost_usd is not None:
        expected_cost = _sum_cost()
        if expected_cost is not None and abs(fm.total_cost_usd - expected_cost) > 1e-6:
            raise NormativeRuleError(
                f"N8: final_metrics.total_cost_usd={fm.total_cost_usd} != "
                f"sum of per-step cost_usd={expected_cost}."
            )


def check_subagent_ref_resolvability(traj: Trajectory) -> None:
    """N9: subagent_trajectory_ref.trajectory_id must resolve within the document.

    Only checks embedded refs (trajectory_id form). File-ref form
    (trajectory_path) is resolved by an external loader, out of scope
    for in-document validation.
    """
    embedded_ids: set[str] = set()

    def _collect(t: Trajectory) -> None:
        for child in t.subagent_trajectories or []:
            if child.trajectory_id:
                embedded_ids.add(child.trajectory_id)
            _collect(child)

    _collect(traj)

    def _check_step(step: StepObject) -> None:
        if step.observation is None:
            return
        for result in step.observation.results:
            for ref in result.subagent_trajectory_ref or []:
                if ref.trajectory_id is None:
                    # file-ref form; not our concern
                    continue
                if ref.trajectory_id not in embedded_ids:
                    raise NormativeRuleError(
                        f"N9: subagent_trajectory_ref.trajectory_id="
                        f"{ref.trajectory_id!r} (step_id={step.step_id}) does not "
                        f"match any embedded subagent. Embedded ids: {sorted(embedded_ids)}."
                    )

    def _walk(t: Trajectory) -> None:
        for s in t.steps:
            _check_step(s)
        for child in t.subagent_trajectories or []:
            _walk(child)

    _walk(traj)


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------


def assert_atif_normative(traj: Trajectory) -> None:
    """Run every rule. Raises NormativeRuleError on first failure.

    Per-trajectory rules (N1 sequential step_ids, N4 copied-context
    propagation, N8 final_metrics sums) run recursively on every
    embedded subagent trajectory — nested trajectories MUST satisfy
    the same invariants as the root.
    """
    for t in _walk_all_trajectories(traj):
        check_sequential_step_ids(t)
        check_copied_context_propagation(t)
        check_final_metrics_sum(t)
    for step in _walk_all_steps(traj):
        check_joinability(step)
        check_deterministic_dispatch(step)
        check_iso_timestamp(step)
        check_message_present(step)
    check_subagent_trajectory_id_uniqueness(traj)
    check_subagent_ref_resolvability(traj)


def _walk_all_trajectories(traj: Trajectory) -> Iterable[Trajectory]:
    """Yield the root trajectory plus every nested subagent_trajectory."""
    yield traj
    for child in traj.subagent_trajectories or []:
        yield from _walk_all_trajectories(child)


def _walk_all_steps(traj: Trajectory) -> Iterable[StepObject]:
    yield from traj.steps
    for child in traj.subagent_trajectories or []:
        yield from _walk_all_steps(child)
