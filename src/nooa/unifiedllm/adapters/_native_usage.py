# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numeric evidence from native replies; no pricing tables or default-zero proof."""

import math
from collections.abc import Mapping

from nooa.unifiedllm._observations import _NumericUsage


def _field(value, key):
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        return None
    return int(value) if integer else float(value)


def _numeric_usage(family, reply, headers=None, *, cost_header=None):
    usage = _field(reply, "usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_unset=True)

    def get(key):
        return _number(_field(usage, key), integer=True)

    if family == "openai-chat":
        input_tokens, output_tokens = get("prompt_tokens"), get("completion_tokens")
        input_details = _field(usage, "prompt_tokens_details")
        output_details = _field(usage, "completion_tokens_details")
    else:
        input_tokens, output_tokens = get("input_tokens"), get("output_tokens")
        input_details = _field(usage, "input_tokens_details")
        output_details = _field(usage, "output_tokens_details")
    cached = _number(_field(input_details, "cached_tokens"), integer=True)
    written = None
    if family == "anthropic-messages":
        cached, written = get("cache_read_input_tokens"), get("cache_creation_input_tokens")
        # Anthropic's input_tokens is UNCACHED. Cache creation breakdowns are
        # subsets of creation, not additions; do not double add them.
        if input_tokens is not None and cached is not None and written is not None:
            input_tokens += cached + written
        else:
            input_tokens = None  # Missing/malformed cache counts are not proven zero.
    reasoning = _number(_field(output_details, "reasoning_tokens"), integer=True)
    total = get("total_tokens") if family != "anthropic-messages" else None
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    cost = _number(_field(usage, "cost_usd"))
    if cost is None:
        cost = _number(_field(usage, "cost"))
    source = "provider" if cost is not None else "unknown"
    # A route declaration names ONE exact header. No suffix/prefix matching.
    if cost_header is not None and headers is not None:
        raw = headers.get(cost_header)
        if raw is not None and not isinstance(raw, bool):
            try:
                candidate = _number(float(raw))
            except (ValueError, TypeError, OverflowError):
                candidate = None
            if candidate is not None:
                cost, source = candidate, "gateway_header"
    return _NumericUsage(
        input_tokens, output_tokens, cached, written, reasoning, total, cost, source
    )
