# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize actual native SDK replies; parsing never hides numeric receipts."""

import json

from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, LLMUsage, ToolCall
from nooa.unifiedllm.errors import FailureKind, LLMFailure

from ._native_projection import _envelope, _valid_replay_item


def _normalize_response(family, model, reply, numeric, *, replay_scope=None):
    data = reply.model_dump(exclude_unset=True)
    parts = []
    finish = "stop"
    if family == "openai-chat":
        choices = data.get("choices", [])
        if not choices:
            raise LLMFailure(FailureKind.BAD_REQUEST, "Native Chat response contains no choices")
        choice = choices[0]
        message = choice.get("message", {})
        reason = choice.get("finish_reason")
        finish = (
            "length"
            if reason == "length"
            else "tool_calls"
            if reason == "tool_calls"
            else "error"
            if reason == "content_filter"
            else "stop"
        )
        if message.get("reasoning_content"):
            parts.append(AssistantReasoning(text=message["reasoning_content"]))
        parts.append(AssistantText(text=message.get("content") or ""))
        for call in message.get("tool_calls", []):
            function = call["function"]
            parts.append(
                ToolCall(id=call["id"], name=function["name"], arguments=function["arguments"])
            )
        # Opaque Chat extensions are one complete assistant message. Ordinary
        # empty text never receives a native carrier (paid-empty retry remains).
        opaque = {
            key: value
            for key, value in message.items()
            if key not in {"role", "content", "tool_calls", "refusal", "annotations"}
            and value is not None
        }
        if opaque:
            parts[0] = parts[0].model_copy(update={"native": _envelope(family, message)})
        if message.get("refusal"):
            finish = "error"
    elif family == "openai-responses":
        finish = (
            "length"
            if data.get("status") == "incomplete"
            and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
            else "error"
            if data.get("status") in {"failed", "cancelled", "incomplete"}
            else "stop"
        )
        for item in data.get("output", []):
            kind = item.get("type")
            native = _envelope(family, item)
            if kind == "message":
                # Reject the whole unsupported/malformed inner turn NOW, after
                # recording its numeric receipt, not on its next replay.
                if not _valid_replay_item(family, item):
                    raise LLMFailure(
                        FailureKind.BAD_REQUEST, "Unsupported or malformed native Responses message"
                    )
                content = item.get("content", [])
                text = "".join(
                    block.get("text", "") for block in content if block.get("type") == "output_text"
                )
                if any(block.get("type") == "refusal" for block in content):
                    finish = "error"
                parts.append(AssistantText(text=text, native=native if text.strip() else None))
            elif kind == "function_call":
                parts.append(
                    ToolCall(
                        id=item["call_id"],
                        name=item["name"],
                        arguments=item["arguments"],
                        native=native,
                    )
                )
                if finish == "stop":
                    finish = "tool_calls"
            elif kind == "reasoning":
                text = "\n".join(block.get("text", "") for block in item.get("summary", []))
                parts.append(AssistantReasoning(text=text, native=native))
            else:
                raise LLMFailure(
                    FailureKind.BAD_REQUEST, "Unsupported native Responses output item"
                )
    else:
        reason = data.get("stop_reason")
        finish = (
            "length"
            if reason == "max_tokens"
            else "tool_calls"
            if reason == "tool_use"
            else "error"
            if reason == "refusal"
            else "stop"
        )
        for block in data.get("content", []):
            native = _envelope(family, block)
            kind = block.get("type")
            if kind == "text":
                text = block.get("text", "")
                parts.append(AssistantText(text=text, native=native if text.strip() else None))
            elif kind == "tool_use":
                parts.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=json.dumps(
                            block["input"], separators=(",", ":"), ensure_ascii=False
                        ),
                        native=native,
                    )
                )
            elif kind in {"thinking", "redacted_thinking"}:
                parts.append(AssistantReasoning(text=block.get("thinking", ""), native=native))
            else:
                raise LLMFailure(
                    FailureKind.BAD_REQUEST, "Unsupported native Anthropic output block"
                )
    if family != "openai-chat" and any(part.native is not None for part in parts):
        wires = data.get("output", []) if family == "openai-responses" else data.get("content", [])
        # One part per native item/block. Empty text is retained when this turn
        # has other replayable content; an entirely empty answer remains empty.
        parts = [
            part.model_copy(update={"native": _envelope(family, wire)})
            if part.native is None
            else part
            for part, wire in zip(parts, wires, strict=True)
        ]
    usage = LLMUsage(
        input_tokens=numeric.input_tokens or 0,
        output_tokens=numeric.output_tokens or 0,
        cached_input_tokens=numeric.cached_input_tokens or 0,
        cache_write_input_tokens=numeric.cache_write_input_tokens or 0,
        reasoning_tokens=numeric.reasoning_tokens or 0,
        total_tokens=numeric.total_tokens or 0,
        cost_usd=numeric.cost_usd or 0.0,
    )
    return LLMResponse(
        parts=tuple(parts),
        raw_response=reply,
        model_name=model,
        replay_scope=replay_scope,
        usage=usage,
        finish_reason=finish,
    )


def _parse_typed(response, output_type):
    if output_type is None:
        return response
    try:
        parsed = output_type.model_validate_json(response.content)
    except Exception as exc:
        kind = (
            FailureKind.OUTPUT_LIMIT
            if response.finish_reason == "length"
            else FailureKind.BAD_REQUEST
        )
        raise LLMFailure(kind, "Native structured response did not validate", cause=exc) from exc
    return response.model_copy(update={"parsed": parsed})
