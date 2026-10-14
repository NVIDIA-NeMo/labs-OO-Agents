# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

from collections.abc import Mapping
from typing import Any

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.media_inputs import MediaInput, attachment_kind
from nooa.media_inputs import MediaInputs as MediaInputs
from nooa.media_inputs import validate_media_inputs as validate_media_inputs
from nooa.unifiedllm.errors import ReasoningReplayError


def filter_media_messages(messages, enabled: tuple[MediaInput, ...]):
    """Filter public attachment slots on a new request view, keeping turn objects.

    Does not visit tool arguments, reasoning, vendor extensions or arbitrary
    nested dictionaries. Projection still owns native LLMResponse containers.
    """
    result = []
    for message in messages:
        if isinstance(message, (LLMResponse, CacheBoundary)) or not isinstance(message, Mapping):
            result.append(message)
            continue
        item = dict(message)
        slot = "output" if item.get("type") == "function_call_output" else "content"
        if (
            item.get("role") == "assistant"
            and item.get("audio") is not None
            and "audio" not in enabled
        ):
            item.pop("audio")
            notice = "[Attachment not inspected: audio input disabled.]"
            if isinstance(item.get(slot), str):
                item[slot] += "\n" + notice
            elif isinstance(item.get(slot), list):
                item[slot] = [*item[slot], {"type": "text", "text": notice}]
            else:
                item[slot] = notice
        content = item.get(slot)
        if isinstance(content, list):
            blocks = []
            for block in content:
                kind = attachment_kind(block) if isinstance(block, Mapping) else None
                if kind is not None and kind not in enabled:
                    blocks.append(
                        {
                            "type": "input_text" if slot == "output" else "text",
                            "text": f"[Attachment not inspected: {kind} input disabled.]",
                        }
                    )
                else:
                    blocks.append(block)
            item[slot] = blocks
        result.append(item)
    return result


def guard_native_media(native: Mapping[str, Any], enabled: tuple[MediaInput, ...] | None) -> None:
    """Adapter-owned compatible envelopes are indivisible; never recurse."""
    if enabled is None:
        return  # Direct projection helpers retain their existing contract.
    content = native.get("content")
    kinds = ["audio"] if native.get("audio") is not None else []
    if isinstance(content, (list, tuple)):
        kinds.extend(attachment_kind(block) for block in content if isinstance(block, Mapping))
    if any(kind is not None and kind not in enabled for kind in kinds):
        raise ReasoningReplayError(
            "Cannot remove disabled media from opaque assistant state; "
            "supply a portable assistant turn instead."
        )
