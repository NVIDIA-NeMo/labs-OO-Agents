# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ContextVar sideband carrying a journal skeleton + content-addressed blocks.

The nooa runtime sets this before each litellm call so
``MessageJournalCallback`` can ship a *skeleton* (the wire message list
with each block reference replaced by its content hash) and the
underlying block bodies (content-addressed so repeat blocks across
turns reuse their hash and don't retransmit).

The skeleton is ``list[SkeletonMessage]``:

* ``{"role": "system", "parts": [{"text": "..."} | {"block_hash": "sha256:..."}, ...]}``
* ``{"role": "user", "content": "..."}`` — when no blocks are present.
* ``{"role": "assistant", "tool_calls": [...]}``.
* ``{"role": "tool", "content": "...", "tool_call_id": "..."}``.

Block bodies are ``dict[str, str]`` keyed by hash.

**Async-boundary note:** ``ContextVar.set()`` only modifies the value in the
*current* async task's context copy. The consumer
(``MessageJournalCallback.log_pre_api_call`` in ``_litellm_journal.py``)
calls ``set_journal_payload(None)`` to consume the sideband after reading it.
This works correctly as long as the litellm call and the callback both
execute in the *same* async task as the ``set_journal_payload`` call —
which is true for nooa's normal execution model
(``_build_messages`` → ``litellm.acompletion`` → callback, all in the
same coroutine).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JournalPayload:
    """Skeleton + block bodies for a single LLM call."""

    skeleton: list[dict[str, Any]]
    """Wire message list with block references as ``{"block_hash": ...}``."""

    blocks: dict[str, str]
    """``hash -> rendered_content`` for every block referenced in ``skeleton``."""


_current_payload: ContextVar[JournalPayload | None] = ContextVar(
    "_nooa_journal_payload", default=None
)


def set_journal_payload(payload: JournalPayload | None) -> None:
    """Called by the runtime just before each litellm call.

    Pass ``None`` to clear. Consumers (the journal callback) clear the
    sideband immediately after reading so later calls within the same
    task don't pick up a stale payload.
    """
    _current_payload.set(payload)


def get_journal_payload() -> JournalPayload | None:
    """Called by the journal callback inside ``log_pre_api_call``."""
    return _current_payload.get()
