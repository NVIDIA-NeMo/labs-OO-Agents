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
    stable_prefix: bool
    marker_count: int
    explicit_mode: bool
    reasoning_observed_by_turn: list[bool]
    request: dict[str, Any]


def public_record(record: dict) -> dict:
    """Select documented evidence; never expose private requests or new raw fields."""
    return {k: v for k, v in record.items() if k in ProbeRecord.__annotations__ and k != "request"}


def check_status(name: str, record: dict, *, missing_reasoning=False) -> str:
    """One verdict policy for progress rows, JSON stages and final summaries."""
    outcome = record.get("outcome")
    if outcome == "not_probed" and not record.get("error"):
        return "skipped"
    if outcome not in {"accepted", "confirmed", "completed", "warning"}:
        return "attention"
    if record.get("finish_reason") in {"length", "error", "content_filter"}:
        return "attention"
    if name == "tools" and not record.get("tool_observed"):
        return "attention"
    if name.startswith("level:") and (missing_reasoning or record.get("answer_correct") is False):
        return "attention"
    return "passed"
