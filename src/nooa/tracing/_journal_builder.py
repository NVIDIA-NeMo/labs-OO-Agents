# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a :class:`JournalPayload` from a runtime ``RenderedMessage`` list.

Lives in the tracing layer (not the runtime) because the
hashing/skeleton-construction logic is a pure tracing concern.  The
runtime hands its rendered messages to :func:`set_journal_payload_from_messages`
and is otherwise oblivious to ``JournalPayload``, ``block_hash``, and
SHA-256 — those are wire-protocol details.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nooa.tracing._context_sideband import (
    JournalPayload,
    set_journal_payload,
)


def _hash(content: str) -> str:
    """SHA-256 a UTF-8 string the same way the litellm callback does."""
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _encode_image(img: Any) -> str:
    """Canonicalize an image value to a stable string for hashing.

    String pass-through (URLs); everything else serializes via canonical
    JSON.  Must match :func:`nooa.tracing._litellm_journal._skeleton_dict_message`
    so the same image hashes identically whether it arrives via the
    runtime sideband (input messages) or via the assistant's reply
    (output messages).  Otherwise dedup misses across the in/out boundary.
    """
    if isinstance(img, str):
        return img
    return json.dumps(img, sort_keys=True, separators=(",", ":"))


def build_journal_payload(messages: list[Any]) -> JournalPayload:
    """Build a :class:`JournalPayload` from a rendered message list.

    Walks each ``RenderedMessage.parts`` (populated by block-aware
    formatters), hashes every ``BlockPart.content`` with SHA-256, and
    constructs a skeleton that mirrors the wire message shape but with
    block references replaced by their hashes.  Messages without
    ``parts`` are carried through as-is (plain ``content``).

    The shape returned here must match what
    :class:`nooa.tracing._litellm_journal.MessageJournalCallback`
    consumes from the sideband -- and what the viewer's
    ``_resolve_message`` reverses on the read side.  Keep the three in
    sync.
    """
    blocks: dict[str, str] = {}
    skeleton: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.role
        # Skip roles that never reach the LLM
        role_s = role.value if hasattr(role, "value") else str(role)
        if role_s in ("metadata", "runtime_event"):
            continue

        entry: dict[str, Any] = {"role": role_s}

        if msg.parts:
            parts_repr: list[dict[str, Any]] = []
            for part in msg.parts:
                if part.kind == "block":
                    h = _hash(part.content)
                    blocks[h] = part.content
                    parts_repr.append({"block_hash": h, "key": part.key})
                else:  # text
                    parts_repr.append({"text": part.text})
            entry["parts"] = parts_repr
        elif msg.content is not None:
            # Content-address plain content too -- otherwise an N-turn
            # conversation re-ships every previous message body in full
            # on each /v1/journal/calls POST.  ``_send_new_blocks`` ships
            # the same hash at most once per session, collapsing the
            # per-call journal record to O(delta).
            content_s = str(msg.content)
            h = _hash(content_s)
            blocks[h] = content_s
            entry["parts"] = [{"block_hash": h}]

        if msg.tool_call is not None:
            tc = msg.tool_call
            args = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments)
            ah = _hash(args)
            blocks[ah] = args
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments_hash": ah},
                }
            ]
        if msg.tool_call_id is not None:
            entry["tool_call_id"] = msg.tool_call_id
        if msg.images:
            img_hashes: list[str] = []
            for img in msg.images:
                s = _encode_image(img)
                h = _hash(s)
                blocks[h] = s
                img_hashes.append(h)
            entry["image_hashes"] = img_hashes

        skeleton.append(entry)

    return JournalPayload(skeleton=skeleton, blocks=blocks)


def set_journal_payload_from_messages(messages: list[Any]) -> None:
    """Public entry point for the runtime: build + publish in one call.

    Pass an iterable of ``RenderedMessage``-shaped objects (anything with
    ``role``, ``parts`` / ``content``, ``tool_call``, ``tool_call_id``,
    ``images``).  The resulting :class:`JournalPayload` is written into
    the tracing sideband ``ContextVar``; the journal callback consumes
    it on the next ``log_pre_api_call``.

    Safe to call when tracing is disabled -- the sideband is just a
    ``ContextVar`` set; no I/O happens until the litellm callback fires.
    """
    payload = build_journal_payload(messages)
    set_journal_payload(payload)
