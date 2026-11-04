# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe records: the stored result of one Connect test call ("probe").

A probe record holds sanitized facts about one test call (outcome, finish
reason, token counts, what was observed in the request and reply). It is used
by both the library and the user interfaces. Every field is optional; a
missing field means the fact is unknown, not zero or unsupported. The
``request`` field keeps the request that was sent so a later check can tell
whether exactly the same request would be sent again; it is never shown in
diagnostic output.

This module also holds the shared rules for turning records into the verdicts
shown to the user ("passed", "attention", "skipped").
"""

from typing import Any, TypedDict


class ProbeRecord(TypedDict, total=False):
    """Known fields of a probe record; all optional (see the module docstring)."""

    outcome: str
    reason: str
    error: str
    detail: str
    checked_at: str
    elapsed_seconds: float
    error_chain: list[str]
    timeout_kind: str
    request_shape: dict[str, Any]
    status_code: int
    reasoning_observed: bool
    reasoning_encrypted: bool
    reasoning_encrypted_bytes: int | None
    reasoning_text_chars: int | None
    answer_correct: bool
    tool_observed: bool
    state_retained: bool
    settings_retained: bool
    settings_sent: bool | None
    configured_reply_tokens: int
    tested_reply_tokens: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    readings: list[dict[str, Any]]
    finish_reason: str
    tokens_charged_to_budget: int
    attempts: list[dict[str, Any]]
    client: str
    transport: str
    tested_reasoning_level: str | None
    reported_tokens: int
    resolved_model: str
    probability_true: float
    stable_prefix: bool
    marker_count: int
    explicit_mode: bool
    reasoning_observed_by_turn: list[bool]
    request: dict[str, Any]
    baseline_kind: str
    typed_result_validated: bool
    typed_result: Any
    final_answer_observed: bool
    request_accepted: bool
    access_reported: bool
    attachment_observed: bool
    implementation_supported: bool  # Derived implementation ceiling, not model support.
    output_settings_observed: bool
    structured_output_mode: str | None
    structured_output_carrier: str
    structured_output_schema_observed: bool
    structured_output_conflicts_observed: bool
    verification_config: dict[str, Any]
    verification_fingerprint: str
    verification_current: bool
    verification_execution: dict[str, Any]


def public_record(record: dict) -> dict:
    """Return the publicly shown fields of ``record``.

    Keeps only fields declared in ``ProbeRecord``, minus the private
    ``request`` and ``verification_config``. Unknown fields are dropped.
    """
    return {
        k: v
        for k, v in record.items()
        if k in ProbeRecord.__annotations__ and k not in {"request", "verification_config"}
    }


TYPED_TEXT_BASELINE = "predict_int_v1"


def text_baseline_passed(record: dict) -> bool:
    """Return True if ``record`` is a current, fully passing typed text test call.

    This single rule decides the text check verdict in the UI and whether
    media test calls may run. All of these must hold: the verification is
    current, the call was the typed baseline, it was accepted, validated, gave
    the right answer with a final answer and ``stop``, and the configured
    settings were seen in the sent request. For Anthropic-style structured
    output the JSON schema must also have been sent in ``output_config.format``
    without conflicting settings.
    """
    return (
        isinstance(record.get("verification_fingerprint"), str)
        and record.get("verification_current") is True
        and record.get("baseline_kind") == TYPED_TEXT_BASELINE
        and record.get("outcome") == "accepted"
        and record.get("typed_result_validated") is True
        and record.get("answer_correct") is True
        and record.get("final_answer_observed") is True
        and record.get("settings_sent") is True
        and record.get("finish_reason") == "stop"
        and (
            record.get("structured_output_mode") != "anthropic"
            or record.get("structured_output_schema_observed") is True
            and record.get("structured_output_conflicts_observed") is False
            and record.get("structured_output_carrier") == "output_config.format"
        )
    )


def text_baseline_reason(record: dict) -> str:
    """Return a fixed sentence explaining the text baseline result in ``record``.

    Checks the same conditions as ``text_baseline_passed`` and names the first
    one that failed. Model output and provider error text are never included.
    """
    if record.get("outcome") == "not_probed":
        if record.get("reason") == "budget exhausted":
            return "Text baseline was not tested: approved budget exhausted"
        return "Text baseline was not tested"
    if record.get("typed_result_validated") is not True and "finish_reason" not in record:
        return "Text answer check failed: no validated integer response was available"
    if record.get("finish_reason") != "stop":
        return "Text baseline reply was incomplete: did not complete with a final stop"
    if record.get("final_answer_observed") is not True:
        return "Text baseline returned no final answer (reasoning alone is insufficient)"
    if record.get("typed_result_validated") is not True:
        return "Text baseline did not return a validated integer"
    if record.get("answer_correct") is not True:
        return "Text answer check failed: request accepted but returned the wrong integer"
    if record.get("settings_sent") is not True:
        return "Configured text baseline settings were not observed on the actual request"
    return "Validated integer 323 with a complete final answer and configured settings observed"


def check_status(
    name: str, record: dict, *, entry: dict | None = None, missing_reasoning=False
) -> str:
    """Return the verdict for check ``name``: "passed", "attention" or "skipped".

    Used everywhere a check result is shown (progress rows, JSON stages and
    final summaries) so they always agree. ``entry`` is the model entry the
    check belongs to; with it, the record's verification is compared with the
    entry's current settings. ``missing_reasoning`` marks a reasoning-level
    check whose reply showed no reasoning.

    A record that is not current (the settings changed since it ran) always
    needs attention, even if it once passed.
    """
    outcome = record.get("outcome")
    if name.startswith("media:") and entry is not None:
        from nooa.media_policy import entry_input_interface, implemented_input_kinds

        if name[6:] not in implemented_input_kinds(entry_input_interface(entry)):
            return "skipped" if outcome == "not_probed" else "attention"
    current = False
    if outcome not in {"not_probed", "running"}:
        from ._verification import verification_is_current

        current = (
            verification_is_current(record, entry, name)
            if entry is not None
            else isinstance(record.get("verification_fingerprint"), str)
            and record.get("verification_current") is True
        )
        if not current:
            return "attention"  # A result for older settings no longer counts.
    # Decision models have no text baseline, so their routing check is judged
    # by the generic rules below.
    if name == "routing" and outcome != "not_probed" and record.get("client") != "decision":
        if record.get("baseline_kind") != "interface_acceptance":
            # Use the freshly computed "current" value rather than a flag
            # stored by an earlier edit; the same rule decides whether the
            # entry may be saved.
            baseline = {**record, "verification_current": current}
            return "passed" if text_baseline_passed(baseline) else "attention"
    if name.startswith("media:") and outcome == "passed":
        return "passed"
    if outcome == "not_probed" and not record.get("error"):
        return "skipped"
    if outcome not in {"accepted", "confirmed", "completed", "warning"}:
        return "attention"
    if record.get("finish_reason") in {"length", "error", "content_filter"}:
        return "attention"
    if name == "tools" and not record.get("tool_observed"):
        return "attention"
    # The reasoning-level puzzle is there to make the model reason, not to
    # test whether it can solve it: a wrong answer with reasoning observed
    # still passes. Only missing reasoning needs attention.
    if name.startswith("level:") and missing_reasoning:
        return "attention"
    return "passed"
