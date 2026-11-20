# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-local provider usage projection into private numeric records only."""

import math
from collections.abc import Mapping

from nooa.unifiedllm._observations import _NumericUsage


def _field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        return None
    return int(value) if integer else float(value)


def _first(value, *names):
    for name in names:
        if (result := _field(value, name)) is not None:
            return result
    return None


def _numeric_usage(raw_response):
    """Best effort: a malformed usage object must not mask provider/parse errors.

    Only explicit numeric fields are known. Exclude unset model defaults so an
    LLMUsage's default cost=0 cannot be mistaken for a reported free response.
    Keep the legacy normalized total fallback when input and output are known.
    """
    try:
        hidden = getattr(raw_response, "_hidden_params", None)
        if _field(hidden, "cache_hit") is True:
            # Cache evidence is independent of the final response's usage. This
            # invocation dispatched no paid request; public usage stays intact.
            return _NumericUsage(0, 0, 0, 0, 0, 0, 0.0, "cache_hit")
        metadata_cost = _number(_field(hidden, "response_cost"))
        headers = _field(hidden, "additional_headers")
        if not isinstance(headers, Mapping):
            headers = _field(hidden, "response_headers")
        if not isinstance(headers, Mapping):
            headers = getattr(raw_response, "_response_headers", None)
        header_cost = None
        if isinstance(headers, Mapping):
            for key, raw in headers.items():
                if str(key).lower().endswith("x-litellm-response-cost"):
                    try:
                        header_cost = _number(float(raw)) if not isinstance(raw, bool) else None
                    except (TypeError, ValueError, OverflowError):
                        pass
                    break
        value = getattr(raw_response, "usage", None)
        if value is None:
            if metadata_cost is None and header_cost is None:
                return None
            value = {}
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_unset=True)
        elif hasattr(value, "_asdict"):
            value = value._asdict()
        prompt = _first(value, "prompt_tokens_details", "input_tokens_details")
        completion = _first(value, "completion_tokens_details", "output_tokens_details")
        input_tokens = _number(_first(value, "input_tokens", "prompt_tokens"), integer=True)
        output_tokens = _number(_first(value, "output_tokens", "completion_tokens"), integer=True)
        total = _number(_field(value, "total_tokens"), integer=True)
        if total is None and input_tokens is not None and output_tokens is not None:
            total = input_tokens + output_tokens
        cached = _first(value, "cached_input_tokens", "cached_tokens", "cache_read_input_tokens")
        if cached is None:
            cached = _first(prompt, "cached_tokens", "cache_read_input_tokens")
        written = _first(value, "cache_write_input_tokens", "cache_creation_input_tokens")
        if written is None:
            written = _first(
                prompt,
                "cache_write_tokens",
                "cache_write_input_tokens",
                "cache_creation_input_tokens",
            )
        reasoning = _field(value, "reasoning_tokens")
        if reasoning is None:
            reasoning = _field(completion, "reasoning_tokens")
        cost = _number(_first(value, "cost_usd", "cost"))
        source = "provider" if cost is not None else "unknown"
        if metadata_cost is not None:
            cost, source = metadata_cost, "response_metadata"
        if header_cost is not None:
            cost, source = header_cost, "gateway_header"
        return _NumericUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=_number(cached, integer=True),
            cache_write_input_tokens=_number(written, integer=True),
            reasoning_tokens=_number(reasoning, integer=True),
            total_tokens=total,
            cost_usd=cost,
            cost_source=source,
        )
    except Exception:
        return None
