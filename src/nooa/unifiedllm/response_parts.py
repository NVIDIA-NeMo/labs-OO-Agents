# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless Responses projection: ordered parts in, ordered wire items out.

Native JSON is immutable and contains only fields not held in public parts.
There is no flattened-turn order ledger or public-content fingerprint.
"""

import logging
from typing import Any

from nooa._immutable_json import freeze, json_containers
from nooa.llm_types import AssistantPart, AssistantReasoning, AssistantText, LLMResponse, ToolCall

from .replay_state import (
    ReasoningReplayError,
    _scope_provider,
    opaque_item,
    unsupported_responses_parts,
)

logger = logging.getLogger(__name__)


def _require_encrypted_reasoning(item: dict) -> None:
    if (
        not isinstance(item, dict)
        or item.get("type") != "reasoning"
        or not isinstance(item.get("encrypted_content"), str)
        or not item["encrypted_content"]
    ):
        raise ReasoningReplayError("Malformed encrypted reasoning item.")


def _capture_text(blocks: list[dict], separator: str) -> str:
    """Move text into the public part; retain only block metadata and lengths."""
    texts = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ReasoningReplayError("Responses text blocks must be mappings.")
        text = block.pop("text", None)
        if not isinstance(text, str):
            raise ReasoningReplayError("Responses text must be a string.")
        texts.append(text)
        block["_text_length"] = len(text)
    return separator.join(texts)


def _capture_summary(native: dict) -> str:
    summary = native.get("summary", [])
    if isinstance(summary, str):
        native["summary"] = ""  # Preserve the wire shape, not a second copy of public text.
        return summary
    if not isinstance(summary, list):
        raise ReasoningReplayError("Reasoning summary must be text or a list of text blocks.")
    return _capture_text(summary, "\n")


def _restore_summary(native: dict, text: str) -> None:
    if isinstance(native.get("summary"), str):
        native["summary"] = text
    else:
        _restore_text(native.get("summary", []), text, "\n")


def _restore_text(blocks: list[dict], text: str, separator: str) -> None:
    offset = 0
    for block in blocks:
        length = block.pop("_text_length")
        block["text"] = text[offset : offset + length]
        offset += length + len(separator)
    if blocks and offset - len(separator) != len(text):
        raise ReasoningReplayError("Malformed native text-block lengths in ordered archive.")


def capture_parts(output: list[Any], scope: str | None) -> tuple[AssistantPart, ...]:
    """Capture Responses output order without flattening away replay information.

    One assistant turn can contain multiple reasoning items, text messages and
    function calls. The ordered parts are authoritative: public text/arguments
    live once, and native fields retain message boundaries, phase, signatures
    and text-block lengths needed to reconstruct the original items. Consumers
    see the derived public views, not provider-specific reconstruction details.

    opaque_item detaches mutable provider containers; frozen native data then
    survives storage and repeated requests without copying its string payloads.
    Unrecognized routes retain portable parts only. A summary-only reasoning
    item makes the entire turn portable, since mixing a rewritten public turn
    with retained native state would violate replay consistency. Missing or
    malformed supported fields raise the terminal ReasoningReplayError.
    """
    unsupported = unsupported_responses_parts(output)
    if unsupported:
        raise ReasoningReplayError("Unsupported Responses output parts: " + ", ".join(unsupported))
    supported = _scope_provider(scope) in {"openai", "azure"}
    parts: list[AssistantPart] = []
    ids: set[str] = set()
    portable_only = False
    for item in output:
        native = opaque_item(item)
        kind = native["type"]
        if kind == "message":
            content = native.get("content")
            if not isinstance(content, list):
                raise ReasoningReplayError("Responses message content must be a list of blocks.")
            text = _capture_text(content, "")
            part = AssistantText(text=text)
        elif kind == "function_call":
            call_id = native.pop("call_id", None)
            if not isinstance(call_id, str) or (call_id and call_id in ids):
                raise ReasoningReplayError(
                    "Responses tool call ids must be strings; nonempty ids must be unique."
                )
            ids.add(call_id)
            name = native.pop("name", None)
            arguments = native.pop("arguments", None)
            if not isinstance(name, str) or not isinstance(arguments, str):
                raise ReasoningReplayError(
                    "Responses tool call name and arguments must be strings."
                )
            part = ToolCall(id=call_id, name=name, arguments=arguments)
        else:
            encrypted = native.get("encrypted_content")
            if encrypted is not None:
                _require_encrypted_reasoning(native)
            if encrypted is not None and not supported:
                logger.warning(
                    "Unknown provider route: dropping opaque reasoning state; keeping readable text."
                )
            text = _capture_summary(native)
            part = AssistantReasoning(text=text)
            if encrypted is None:
                portable_only = True
                parts.append(part)
                continue
        if supported:
            part = part.model_copy(update={"native": freeze(native)})
        parts.append(part)
    if any(isinstance(part, ToolCall) and not part.id for part in parts) and any(
        isinstance(part, AssistantReasoning) and part.native for part in parts
    ):
        raise ReasoningReplayError("Native reasoning requires nonempty tool call ids.")
    if portable_only:
        # A summary-only reasoning item cannot be replayed natively. Its text
        # demotion edits the turn, so no other part may keep native authority.
        if any(part.native is not None for part in parts):
            logger.warning("Incomplete native reasoning sequence; replaying the turn portably.")
        return tuple(part.model_copy(update={"native": None}) for part in parts)
    return tuple(parts)


def project_turn(turn: LLMResponse, scope: str | None) -> list[dict[str, Any]]:
    """Reconstruct ordered Responses wire items, gated by the turn's replay scope.

    Compatible native parts restore the original item boundaries and metadata,
    filling public text and arguments back into their original positions. This
    preserves the history prefix needed for cache reuse; it does not itself
    select cache breakpoints or guarantee a hit. Incompatible/edited turns keep
    portable text and calls, never the old provider's opaque state.

    Only this adapter opens native data. It allocates request-owned containers
    and shares immutable string leaves; projecting a turn cannot mutate its
    archive. Edits already discard native state at the LLMResponse boundary,
    so projection needs no content fingerprint or reconstructed order ledger.
    """
    compatible = scope is not None and turn.replay_scope == scope
    if compatible and _scope_provider(scope) not in {"openai", "azure"}:
        raise ReasoningReplayError("Native Responses replay only supports OpenAI and Azure.")
    if turn.replay_scope and not compatible:
        logger.warning(
            "Incompatible assistant turn: replaying portable parts without native state."
        )
    result: list[dict[str, Any]] = []
    for part in turn.parts:
        native = json_containers(part.native) if compatible and part.native is not None else None
        if isinstance(part, ToolCall):
            item = native or {"type": "function_call"}
            item.update(call_id=part.id, name=part.name, arguments=part.arguments)
        elif isinstance(part, AssistantText):
            if native:
                _restore_text(native["content"], part.text, "")
                item = native
            elif part.text:
                item = {"role": "assistant", "content": part.text}
            else:
                continue
        elif native is not None:
            _require_encrypted_reasoning(native)
            _restore_summary(native, part.text)
            item = native
        elif part.text:
            item = {"role": "assistant", "content": part.text}
        else:
            continue
        result.append(item)
    return result
