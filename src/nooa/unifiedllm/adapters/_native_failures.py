# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure evidence from actual core SDK exception identities, never text guesses."""

import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import anthropic
import openai

from nooa.unifiedllm.errors import FailureKind, LLMFailure


def _retry_after(headers):
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError):
        try:
            date = parsedate_to_datetime(raw)
            if date.tzinfo is None:
                return None
            value = max(0.0, (date - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None
    return value if math.isfinite(value) and value >= 0 else None


def _body_facts(exc):
    """Only explicit structured provider facts; never scan prose/cause chains."""
    body = exc.body
    if not isinstance(body, dict):
        return None, {}
    body = body.get("error", body)
    if not isinstance(body, dict):
        return None, {}
    code = body.get("code") or body.get("type")
    kind = (
        {
            "context_length_exceeded": FailureKind.CONTEXT_WINDOW,
            "max_tokens_exceeded": FailureKind.OUTPUT_LIMIT,
            "image_count_exceeded": FailureKind.MEDIA_LIMIT,
            "invalid_image": FailureKind.MEDIA_REJECTED,
            "invalid_image_format": FailureKind.MEDIA_REJECTED,
            "unsupported_media_type": FailureKind.MEDIA_REJECTED,
        }.get(code)
        if isinstance(code, str)
        else None
    )
    facts = {}
    for name, minimum in (
        ("prompt_tokens", 0),
        ("context_window", 1),
        ("media_count", 0),
        ("media_count_limit", 0),
    ):
        value = body.get(name)
        if type(value) is int and value >= minimum:
            facts[name] = value
    if body.get("media_kind") in ("image", "pdf", "audio", "video"):
        facts["media_kind"] = body["media_kind"]
    return kind, facts


def _failure(exc):
    if isinstance(exc, LLMFailure):
        return exc
    status = None
    retry_after = None
    facts = {}
    # The outer trusted status dominates even a misleading inner cause.
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        candidate = exc.status_code
        status = candidate if type(candidate) is int and 100 <= candidate <= 599 else None
        retry_after = _retry_after(exc.response.headers)
        if status == 429:
            kind = FailureKind.RATE_LIMIT
        elif status == 408:
            kind = FailureKind.TIMEOUT
        elif status is not None and status >= 500:
            kind = FailureKind.TRANSIENT
        elif status in (401, 403):
            kind = FailureKind.AUTH
        else:
            body_kind, body_facts = _body_facts(exc)
            kind = body_kind or FailureKind.BAD_REQUEST
            if body_kind is not None:
                facts = body_facts
    elif isinstance(exc, (TimeoutError, openai.APITimeoutError, anthropic.APITimeoutError)):
        kind = FailureKind.TIMEOUT
    elif isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        kind = FailureKind.TRANSIENT
    elif isinstance(exc, (openai.APIResponseValidationError, anthropic.APIResponseValidationError)):
        kind = FailureKind.BAD_REQUEST
    else:
        kind = FailureKind.CONFIG
    # Do not expose provider bodies, signed content, keys or private routes.
    return LLMFailure(
        kind,
        "Native provider attempt failed",
        status_code=status,
        retry_after=retry_after,
        cause=exc,
        **facts,
    )
