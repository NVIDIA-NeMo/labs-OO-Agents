# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey patches for openinference-instrumentation-litellm bugs.

Fixes:
1. Bug where null/empty content causes entire JSON response to be logged
   in output.value instead of the actual content value.
2. Bug where tool_call.id is not captured in span attributes (despite being
   defined in OpenInference semantic conventions).
3. Missing reasoning_content capture for reasoning models (DeepSeek, o1, Nemotron, etc.)
4. Extract <think> tags from content for models that embed reasoning (Nemotron, QwQ)

Bug: https://github.com/Arize-ai/openinference/issues (to be filed)
Affected version: openinference-instrumentation-litellm v0.1.28+
"""

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel


def _extract_think_tags(content: str) -> tuple[str, str | None]:
    """Extract <think>...</think> tags from content.

    Returns (cleaned_content, reasoning) where:
    - cleaned_content is the content with think tags removed
    - reasoning is the extracted thinking content (or None if no tags found)

    Handles both complete tags and malformed tags (missing opening tag due to litellm bug).
    """
    if not content:
        return content, None

    # Pattern for complete <think>...</think> tags
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, content, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        cleaned = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        return cleaned, reasoning

    # Handle malformed case: content starts with thinking and ends with </think>
    # (litellm bug strips opening <think> but leaves closing </think>)
    if "</think>" in content:
        parts = content.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            cleaned = parts[1].strip()
            return cleaned, reasoning

    return content, None


# Gateway response-cost headers (set by the LiteLLM proxy/gateway) mapped to the
# span attribute they populate. ``x-litellm-response-cost`` is the authoritative
# total; the rest are gateway-specific extras kept for completeness. Header keys
# are matched by suffix so a provider prefix (e.g. ``llm_provider-...``) is tolerated.
_COST_HEADER_ATTRS: tuple[tuple[str, str], ...] = (
    ("x-litellm-response-cost", SpanAttributes.LLM_COST_TOTAL),
    ("x-litellm-response-cost-original", "llm.cost.total.original"),
    ("x-litellm-response-cost-discount-amount", "llm.cost.discount_amount"),
    ("x-litellm-response-cost-margin-amount", "llm.cost.margin_amount"),
    ("x-litellm-response-cost-margin-percent", "llm.cost.margin_percent"),
)


def _response_headers(result: Any) -> Mapping[str, Any]:
    """Collect any response headers litellm attached to *result* (needs
    ``litellm.return_response_headers = True``)."""
    hidden = getattr(result, "_hidden_params", None) or {}
    headers = hidden.get("additional_headers")
    if not isinstance(headers, Mapping):
        headers = getattr(result, "_response_headers", None)
    return headers if isinstance(headers, Mapping) else {}


def _header_value(headers: Mapping[str, Any], suffix: str) -> Any:
    """Return the header whose (lowercased) name ends with *suffix*, or None."""
    suffix = suffix.lower()
    for key, value in headers.items():
        if str(key).lower().endswith(suffix):
            return value
    return None


def _set_cost_attributes(span: trace_api.Span, result: Any) -> None:
    """Stamp the OpenInference ``llm.cost.*`` (USD) attributes on the LLM span.

    The litellm OpenInference instrumentor records token counts but not the
    dollar cost, so cost is otherwise absent from our traces. Two sources:

    1. **Gateway headers** (preferred): when the LiteLLM proxy/gateway returns
       ``x-litellm-response-cost`` (+ optional original/discount/margin extras),
       map them onto ``llm.cost.total`` etc. This is the only source for models
       with no local pricing (internal endpoints).
    2. **Local computation** (fallback for direct API calls): litellm's
       per-response ``response_cost`` → ``llm.cost.total``, with a
       prompt/completion split via ``cost_per_token`` when derivable.

    Cost is only stamped when a positive value is available, so models with no
    known pricing are left unset rather than reporting a misleading $0.
    """
    try:
        import litellm

        # 1. Gateway cost headers (authoritative when present).
        headers = _response_headers(result)
        total_from_header = False
        for suffix, attr in _COST_HEADER_ATTRS:
            raw = _header_value(headers, suffix)
            if raw is None:
                continue
            try:
                span.set_attribute(attr, float(raw))
                if attr == SpanAttributes.LLM_COST_TOTAL:
                    total_from_header = True
            except (TypeError, ValueError):
                pass
        if total_from_header:
            return

        # 2. Local computation fallback (direct API calls with known pricing).
        hidden = getattr(result, "_hidden_params", None) or {}
        total = hidden.get("response_cost")
        if total is None:
            try:
                total = litellm.completion_cost(completion_response=result)
            except Exception:
                total = None
        if not total or total <= 0:
            return
        span.set_attribute(SpanAttributes.LLM_COST_TOTAL, float(total))

        usage = getattr(result, "usage", None)
        model = getattr(result, "model", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        if model and prompt_tokens is not None and completion_tokens is not None:
            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                if prompt_cost:
                    span.set_attribute(SpanAttributes.LLM_COST_PROMPT, float(prompt_cost))
                if completion_cost:
                    span.set_attribute(SpanAttributes.LLM_COST_COMPLETION, float(completion_cost))
            except Exception:
                pass
    except Exception:
        # Cost is best-effort enrichment — never break the instrumentor.
        pass


def _patched_set_output_message_value(span: trace_api.Span, result: Any) -> Any:
    """Fixed version of _set_output_message_value that handles null/empty content.

    The original bug: when content is None or "", the walrus operator evaluates
    to falsy, causing the condition to fail and fall through to dumping the
    entire JSON response.

    Fix: Check explicitly for None and use span.set_attribute directly
    (the _set_span_attribute helper filters out empty strings).

    Enhancements: capture reasoning_content for reasoning models, and stamp the
    OpenInference ``llm.cost.*`` attributes (not emitted by the instrumentor).
    """
    # Stamp cost first so it lands regardless of the output-message branch below.
    _set_cost_attributes(span, result)

    from litellm.types.utils import Choices

    if result.choices and isinstance(result.choices[-1], Choices):
        message = result.choices[-1].message
        content = message.content
        # Use direct set_attribute to preserve empty strings
        # (the original _set_span_attribute helper filters them out)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, content if content is not None else "")

        # Capture reasoning_content for reasoning models (DeepSeek, o1, Nemotron, etc.)
        # This is returned by models like deepseek-reasoner, o1-*, and some NVIDIA models
        reasoning_content = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )

        # Also extract <think> tags from content for models that embed reasoning
        # (Nemotron, QwQ, etc.) - these don't use reasoning_content field
        if not reasoning_content and content:
            _, think_reasoning = _extract_think_tags(content)
            if think_reasoning:
                reasoning_content = think_reasoning

        if reasoning_content:
            # Use a custom attribute since OpenInference doesn't have a standard one yet
            span.set_attribute("llm.reasoning_content", reasoning_content)
    else:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.model_dump_json())
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)


# Store reference to original function for the tool_call.id patch
_original_get_attributes_from_message_param: Any = None


def _patched_get_attributes_from_message_param(
    message: Mapping[str, Any],
) -> Iterator[tuple[str, AttributeValue]]:
    """Fixed version that includes tool_call.id in span attributes.

    The original bug: The LiteLLM instrumentation captures tool_call.function.name
    and tool_call.function.arguments but NOT tool_call.id, even though:
    1. The OpenInference semantic conventions define TOOL_CALL_ID
    2. The core openinference library (_attributes.py) captures it correctly
    3. The LiteLLM-specific code reimplements this but forgot the ID

    Fix: Yield tool_call.id for each tool call in addition to function details.
    """
    # First, yield all attributes from the original function
    if _original_get_attributes_from_message_param is not None:
        yield from _original_get_attributes_from_message_param(message)

    # Convert to dict if needed (message might be a Message object for output messages)
    if isinstance(message, BaseModel):
        message = message.model_dump()

    # Then, add the missing tool_call.id attributes
    if (tool_calls := message.get("tool_calls")) and isinstance(tool_calls, Iterable):
        for tool_call_index, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, Mapping) and (tool_call_id := tool_call.get("id")):
                yield (
                    f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}.{ToolCallAttributes.TOOL_CALL_ID}",
                    tool_call_id,
                )

    # Also add message.tool_call_id for tool result messages (role=tool)
    if (tool_call_id := message.get("tool_call_id")) and isinstance(tool_call_id, str):
        yield (MessageAttributes.MESSAGE_TOOL_CALL_ID, tool_call_id)


def apply_litellm_patch() -> None:
    """Apply monkey patches to fix litellm instrumentation bugs.

    This patches:
    1. _set_output_message_value - fixes null/empty content handling, adds
       reasoning_content, and stamps llm.cost.* (not emitted by the instrumentor)
    2. _get_attributes_from_message_param - adds missing tool_call.id capture

    It also enables ``litellm.return_response_headers`` so the gateway's
    ``x-litellm-response-cost`` headers are retained on the response for cost
    capture (see :func:`_set_cost_attributes`).
    """
    global _original_get_attributes_from_message_param

    try:
        import litellm as _litellm_mod

        # Retain provider/gateway response headers so cost headers are available.
        _litellm_mod.return_response_headers = True
    except Exception:
        pass

    try:
        from openinference.instrumentation import litellm

        # Patch 1: Fix null/empty content handling
        litellm._set_output_message_value = _patched_set_output_message_value

        # Patch 2: Fix missing tool_call.id
        # Store the original function so our patch can call it
        # CRITICAL: Only patch if not already patched (prevents infinite recursion)
        if (
            litellm._get_attributes_from_message_param
            is not _patched_get_attributes_from_message_param
        ):
            _original_get_attributes_from_message_param = litellm._get_attributes_from_message_param
            litellm._get_attributes_from_message_param = _patched_get_attributes_from_message_param

    except (ImportError, AttributeError):
        # If litellm instrumentation is not available, silently skip
        pass
