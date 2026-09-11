# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture and project LiteLLM Chat turns without detached replay carriers.

The normalized Chat protocol groups thinking, content, and tool calls. Signatures
stay on their owning part; scope is checked once on the owning turn. Only this
adapter knows the names of LiteLLM's native extension fields.
"""

from typing import Any

from nooa._immutable_json import json_containers
from nooa.llm_types import AssistantPart, AssistantReasoning, AssistantText, LLMResponse, ToolCall

from .replay_state import (
    ReasoningReplayError,
    _field,
    _scope_provider,
    _tool_call_state,
    _warn_unknown_fields,
    logger,
    opaque_item,
)
from .response_parts import _capture_summary, _require_encrypted_reasoning, _restore_summary


def capture_chat_parts(message: Any, scope: str | None) -> tuple[AssistantPart, ...]:
    """Capture one normalized Chat response as the durable assistant-turn record.

    Strategies and renderers consume public text and tool-call views; they must
    not reconstruct signed thinking or know the gateway's extension fields.
    Capture therefore stores reasoning, answer text, and calls as ordered parts,
    with each signature or encrypted item attached to its owning part. Chat's
    normalized response already groups these fields; we preserve that ordering,
    not an interleaving the transport has discarded. Readable reasoning is kept
    once even when LiteLLM exposes it both as a block and reasoning_content.

    Public fields are moved out of detached native containers before the parts
    freeze them. Unknown routes and incomplete native sequences keep portable
    text but lose native state together; malformed supported shapes raise rather
    than turning a successful-looking capture into silently incomplete replay.
    The LLMResponse stores the compatibility scope once for the whole turn.
    """
    provider = _scope_provider(scope)
    parts: list[AssistantPart] = []
    portable_only = False
    for field in ("thinking_blocks", "reasoning_items"):
        blocks = _field(message, field)
        if blocks is None:
            blocks = []
        if not isinstance(blocks, list):
            raise ReasoningReplayError(f"Malformed provider response field {field!r}.")
        if (
            blocks
            and provider is not None
            and (field == "reasoning_items" and provider not in {"openai", "azure"})
        ):
            raise ReasoningReplayError(f"Cannot retain {field} for this provider route.")
        for block in blocks:
            native = opaque_item(block)
            if not isinstance(native, dict):
                raise ReasoningReplayError(f"Malformed {field} block: expected a mapping.")
            kind = native.get("type")
            if field == "thinking_blocks":
                if kind == "thinking":
                    text = native.pop("thinking", "")
                    if (
                        not isinstance(text, str)
                        or not isinstance(native.get("signature"), str)
                        or not native["signature"]
                    ):
                        raise ReasoningReplayError("Malformed signed thinking block.")
                elif (
                    kind == "redacted_thinking"
                    and isinstance(native.get("data"), str)
                    and native["data"]
                ):
                    text = ""
                else:
                    raise ReasoningReplayError(f"Unsupported thinking block {kind!r}.")
            else:
                if kind != "reasoning":
                    raise ReasoningReplayError(f"Unsupported reasoning item {kind!r}.")
                if native.get("encrypted_content") is not None:
                    _require_encrypted_reasoning(native)
                else:
                    portable_only = True
                text = _capture_summary(native)
                if native.get("encrypted_content") is None:
                    parts.append(AssistantReasoning(text=text))
                    continue
            parts.append(AssistantReasoning(text=text, native={field: native}))

    reasoning = _field(message, "reasoning") or _field(message, "reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ReasoningReplayError("Provider reasoning text must be a string.")
    # LiteLLM commonly exposes the same signed thinking as reasoning_content.
    # Keep the readable text once, on its authoritative reasoning parts.
    if reasoning and reasoning not in {
        "".join(part.text for part in parts),
        "\n".join(part.text for part in parts if part.text),
    }:
        parts.append(AssistantReasoning(text=reasoning))

    fields = _field(message, "provider_specific_fields")
    native_text: dict[str, Any] = {}
    if fields is not None and not isinstance(fields, dict):
        raise ReasoningReplayError("Malformed provider_specific_fields in provider response.")
    if fields:
        _warn_unknown_fields(fields, {"thought_signatures"}, "a provider message")
        if "thought_signatures" in fields:
            signatures = fields["thought_signatures"]
            if (
                (provider is not None and provider not in {"openai", "azure", "gemini"})
                or not isinstance(signatures, list)
                or not signatures
                or not all(isinstance(s, str) and s for s in signatures)
            ):
                raise ReasoningReplayError("Malformed or unsupported message thought signatures.")
            native_text["provider_specific_fields"] = {"thought_signatures": signatures}
    content = _field(message, "content")
    if content is not None and not isinstance(content, str):
        raise ReasoningReplayError("Chat assistant content must be a string or null.")
    parts.append(AssistantText(text=content or "", native=native_text or None))

    calls = _field(message, "tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ReasoningReplayError("Malformed tool_calls in provider response.")
    ids: set[str] = set()
    for call in calls:
        native = _tool_call_state(call, scope)
        if native and provider is not None and provider not in {"openai", "azure", "gemini"}:
            raise ReasoningReplayError("Cannot retain tool signatures for this provider route.")
        call_id = _field(call, "id")
        if native and "inline_thought_signature" in native:
            call_id = call_id.split("__thought__", 1)[0]
        if not isinstance(call_id, str) or (call_id and call_id in ids):
            raise ReasoningReplayError(
                "Chat tool call ids must be strings; nonempty ids must be unique."
            )
        ids.add(call_id)
        function = _field(call, "function")
        parts.append(
            ToolCall(
                id=call_id,
                name=_field(function, "name") or "",
                arguments=_field(function, "arguments") or "",
                native=native,
            )
        )
    if portable_only:
        if any(part.native for part in parts):
            logger.warning("Incomplete native reasoning sequence; replaying the turn portably.")
        return tuple(part.model_copy(update={"native": None}) for part in parts)
    if scope is None and any(part.native for part in parts):
        logger.warning(
            "Unknown provider route: dropping opaque reasoning state; keeping readable text."
        )
        return tuple(part.model_copy(update={"native": None}) for part in parts)
    if any(isinstance(part, ToolCall) and not part.id for part in parts) and any(
        part.native for part in parts if not isinstance(part, AssistantText)
    ):
        raise ReasoningReplayError("Native reasoning requires nonempty tool call ids.")
    return tuple(parts)


def project_chat_turn(turn: LLMResponse, scope: str | None) -> tuple[dict, dict[str, str]]:
    """Build request-owned Chat fields from an immutable assistant turn.

    Projection is the only outbound layer that interprets native part data.
    Matching non-null scopes restore provider fields beside their original
    public text/calls; incompatible scopes omit native state and render readable
    reasoning as ordinary assistant text. This portable fallback belongs to a
    retained LLMResponse, not arbitrary dictionaries supplied by a caller.

    Native containers are allocated for the request while immutable string
    leaves remain shared with the archive. The returned id map reconnects Gemini
    tool results to ids containing restored signatures; callers never need to
    know how those ids encode private state. No stored part is mutated.
    """
    compatible = scope is not None and turn.replay_scope == scope
    provider = _scope_provider(scope)
    if turn.replay_scope and not compatible:
        logger.warning(
            "Incompatible assistant turn: replaying portable parts without native state."
        )
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    text: list[str] = []
    call_ids: dict[str, str] = {}
    for part in turn.parts:
        native = json_containers(part.native) if compatible and part.native else {}
        if isinstance(part, ToolCall):
            call_id = part.id
            if signature := native.pop("inline_thought_signature", None):
                call_id = f"{call_id}__thought__{signature}"
                call_ids[part.id] = call_id
            message.setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": part.name, "arguments": part.arguments},
                    **native,
                }
            )
        elif isinstance(part, AssistantText):
            message.update(native)
            if part.text:
                text.append(part.text)
        elif native:
            field, block = next(iter(native.items()))
            if field == "thinking_blocks" and block["type"] == "thinking":
                block["thinking"] = part.text
            elif field == "reasoning_items":
                if provider not in {"openai", "azure"}:
                    raise ReasoningReplayError(
                        "Encrypted Chat reasoning is unsupported for this provider."
                    )
                _require_encrypted_reasoning(block)
                _restore_summary(block, part.text)
            elif field != "thinking_blocks":
                raise ReasoningReplayError("Unknown native Chat reasoning field.")
            message.setdefault(field, []).append(block)
        elif part.text:
            text.append(part.text)
    if text:
        message["content"] = "\n\n".join(text)
    elif message.get("tool_calls"):
        message["content"] = None
    return message, call_ids
