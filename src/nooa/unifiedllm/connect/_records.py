# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitized, optional probe evidence shared by library and frontends.

Missing fields mean unavailable evidence, not zero or unsupported. ``request``
is retained privately for exact reuse checks; it is never diagnostic output.
"""

from typing import Any, TypedDict


class ProbeRecord(TypedDict, total=False):
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
    final_answer_observed: bool
    request_accepted: bool
    access_reported: bool
    attachment_observed: bool
    output_settings_observed: bool


def public_record(record: dict) -> dict:
    """Select documented evidence; never expose private requests or new raw fields."""
    return {k: v for k, v in record.items() if k in ProbeRecord.__annotations__ and k != "request"}


TYPED_TEXT_BASELINE = "predict_int_v1"


def text_baseline_passed(record: dict) -> bool:
    """One configured baseline criterion for UI verdicts and media gating."""
    return (
        record.get("baseline_kind") == TYPED_TEXT_BASELINE
        and record.get("outcome") == "accepted"
        and record.get("typed_result_validated") is True
        and record.get("answer_correct") is True
        and record.get("final_answer_observed") is True
        and record.get("settings_sent") is True
        and record.get("finish_reason") == "stop"
    )


def text_baseline_reason(record: dict) -> str:
    """Sanitized facts only; never interpolate model output or provider errors."""
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


def check_status(name: str, record: dict, *, missing_reasoning=False) -> str:
    """One verdict policy for progress rows, JSON stages and final summaries."""
    outcome = record.get("outcome")
    # Decision models have no text baseline; their routing check stands alone.
    if name == "routing" and outcome != "not_probed" and record.get("client") != "decision":
        if record.get("baseline_kind") != "interface_acceptance":
            return "passed" if text_baseline_passed(record) else "attention"
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
    # The puzzle exists to elicit reasoning, not to prove the model can solve
    # it — a wrong answer with reasoning genuinely observed is not a check
    # failure. Only a missing reasoning signal (the thing this check actually
    # verifies) gates "attention" here.
    if name.startswith("level:") and missing_reasoning:
        return "attention"
    return "passed"
