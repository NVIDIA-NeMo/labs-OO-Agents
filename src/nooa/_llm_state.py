# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private in-memory transport for reasoning replay metadata.

The mapping stays provider-wire-safe: JSON serializers see only its ordinary
message fields. UnifiedLLM consumes the private attributes before dispatch,
demotes portable text reasoning, and drops opaque state unless a later
provider-specific layer explicitly recognizes it.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

LLM_STATE_KEY = "_nooa_llm_state"


class ReplayCarryingMessage(dict[str, Any]):
    """A public wire message with replay metadata outside its mapping."""

    __slots__ = ("llm_state", "reasoning", "replay_batch_id", "replay_batch_size")

    def __init__(
        self,
        message: dict[str, Any],
        llm_state: dict[str, Any] | None = None,
        reasoning: str | None = None,
        *,
        replay_batch_id: str | None = None,
        replay_batch_size: int = 0,
    ):
        super().__init__(message)
        self.llm_state = copy.deepcopy(llm_state)
        self.reasoning = reasoning
        self.replay_batch_id = replay_batch_id
        self.replay_batch_size = replay_batch_size


def carried_state(message: Any) -> dict[str, Any] | None:
    """Read opaque state attached outside a rendered message mapping."""
    state = getattr(message, "llm_state", None)
    return state if isinstance(state, dict) else None


def carried_reasoning(message: Any) -> str | None:
    """Read provider-exposed text reasoning from a rendered message."""
    reasoning = getattr(message, "reasoning", None)
    return reasoning if isinstance(reasoning, str) and reasoning else None


def carry_replay_batch(
    messages: list[dict[str, Any]],
    llm_state: dict[str, Any] | None,
    reasoning: str | None,
) -> list[dict[str, Any]]:
    """Attach replay metadata and one identity to a Responses item batch."""
    batch_id = uuid4().hex
    size = len(messages)
    return [
        ReplayCarryingMessage(
            message,
            llm_state if index == 0 else None,
            reasoning if index == 0 else None,
            replay_batch_id=batch_id,
            replay_batch_size=size,
        )
        for index, message in enumerate(messages)
    ]


def carried_replay_batch(message: Any) -> tuple[str, int] | None:
    """Return a rendered Responses batch identity stored outside its mapping."""
    batch_id = getattr(message, "replay_batch_id", None)
    batch_size = getattr(message, "replay_batch_size", 0)
    if isinstance(batch_id, str) and isinstance(batch_size, int) and batch_size > 0:
        return batch_id, batch_size
    return None


def _merge_reasoning_text(message: dict[str, Any], reasoning: str | None) -> None:
    """Demote portable reasoning onto an assistant message without duplication."""
    if not reasoning or message.get("role") != "assistant":
        return
    content = message.get("content")
    if isinstance(content, str):
        if content.strip() == reasoning.strip():
            return
        message["content"] = f"{reasoning}\n\n{content}" if content else reasoning
    elif isinstance(content, list):
        message["content"] = [{"type": "text", "text": reasoning}, *content]
    elif content is None:
        message["content"] = reasoning


def demote_chat_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build public Chat messages, withholding opaque state by default."""
    prepared: list[dict[str, Any]] = []
    for original in messages:
        state = carried_state(original)
        reasoning = carried_reasoning(original)
        message = copy.deepcopy(dict(original))
        message.pop(LLM_STATE_KEY, None)
        _merge_reasoning_text(message, reasoning)
        if (
            state is not None
            and not reasoning
            and message.get("role") == "assistant"
            and not message.get("content")
            and not message.get("tool_calls")
        ):
            continue
        prepared.append(message)
    return prepared


def demote_responses_batch(
    batch: list[dict[str, Any]],
    llm_state: dict[str, Any] | None,
    reasoning: str | None,
) -> list[dict[str, Any]]:
    """Build public Responses items, withholding opaque state by default."""
    clean = [copy.deepcopy(dict(item)) for item in batch]
    for item in clean:
        item.pop(LLM_STATE_KEY, None)

    if not reasoning:
        if (
            llm_state is not None
            and len(clean) == 1
            and clean[0].get("role") == "assistant"
            and not clean[0].get("content")
            and not clean[0].get("tool_calls")
        ):
            return []
        return clean

    message = next((item for item in clean if item.get("role") == "assistant"), None)
    if message is None:
        return [{"role": "assistant", "content": reasoning}, *clean]
    if isinstance(message.get("content"), list):
        index = clean.index(message)
        return [
            *clean[:index],
            {"role": "assistant", "content": reasoning},
            *clean[index:],
        ]
    _merge_reasoning_text(message, reasoning)
    return clean
