# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small provider-neutral helpers for public message dictionaries."""

from __future__ import annotations

from typing import Any

LLM_STATE_KEY = "_nooa_llm_state"


def demote_reasoning_text(message: dict[str, Any], reasoning: str | None) -> None:
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


def carried_cache_boundary(message: Any) -> bool:
    """Return whether the volatile suffix begins at this rendered message."""
    return isinstance(message, dict) and message.get("nooa_cache_boundary") is True
