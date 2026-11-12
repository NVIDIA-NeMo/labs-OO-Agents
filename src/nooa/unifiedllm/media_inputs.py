# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.media_inputs import MediaInput, attachment_kind
from nooa.media_inputs import MediaInputs as MediaInputs
from nooa.media_inputs import validate_media_inputs as validate_media_inputs
from nooa.unifiedllm.adapters import _legacy_error_text
from nooa.unifiedllm.errors import FailureKind, LLMFailure, ReasoningReplayError

_NOTICE_SUBJECT = {
    "image": ("An image", "images"),
    "audio": ("An audio clip", "audio"),
    "video": ("A video", "video"),
    "pdf": ("A PDF", "PDF files"),
}


def removed_media_notice(kind: str) -> str:
    """The text the model sees in place of media it cannot receive."""
    if kind not in _NOTICE_SUBJECT:  # "file": any file that is not a PDF
        return "[A file was removed here. Only PDF files can be sent to a model.]"
    subject, plural = _NOTICE_SUBJECT[kind]
    return f"[{subject} was removed here because this model is not set up to receive {plural}.]"


def rejected_media_notice(kind: str) -> str:
    """The text the model sees in place of media the provider could not read."""
    subject = _NOTICE_SUBJECT.get(kind, ("A file", "files"))[0]
    return f"[{subject} was removed here because the model provider could not read it.]"


def image_limit(exc: BaseException) -> int | None:
    """Typed image-count facts first; raw fallback for unmigrated clients only."""
    if isinstance(exc, LLMFailure):
        if exc.kind == FailureKind.MEDIA_LIMIT and exc.media_kind == "image":
            return exc.media_count_limit
        return None
    return _legacy_error_text.image_limit(exc)


def explicitly_rejected(exc, kind):
    """Typed kind is authoritative; never inspect a typed failure's cause."""
    if isinstance(exc, LLMFailure):
        return exc.kind == FailureKind.MEDIA_REJECTED and exc.media_kind == kind
    return _legacy_error_text.explicitly_rejected(exc, kind)


def _occurrences(messages):
    """Yield (message index, slot, block index, kind, key) for public attachment slots.

    ``key`` names one attachment occurrence: its kind, its block's content and
    how many identical blocks came before it in this request.
    """
    seen: dict[tuple[str, str], int] = {}
    for index, message in enumerate(messages):
        if isinstance(message, (LLMResponse, CacheBoundary)) or not isinstance(message, Mapping):
            continue
        slot = "output" if message.get("type") == "function_call_output" else "content"
        content = message.get(slot)
        if not isinstance(content, (list, tuple)):
            continue
        for position, block in enumerate(content):
            kind = attachment_kind(block) if isinstance(block, Mapping) else None
            if kind is None:
                continue
            payload = {k: v for k, v in block.items() if k != "cache_control"}
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=repr).encode()
            ).hexdigest()
            count = seen.get((kind, digest), 0)
            seen[(kind, digest)] = count + 1
            yield index, slot, position, kind, (kind, digest, count)


def media_occurrences(messages, enabled: tuple[MediaInput, ...]):
    """The attachment occurrences a request sends, oldest first, as (kind, key)."""
    return [(kind, key) for *_, kind, key in _occurrences(messages) if kind in enabled]


def filter_media_messages(messages, enabled: tuple[MediaInput, ...], excluded=frozenset()):
    """Return a copy of ``messages`` with media that is not ``enabled`` replaced.

    Each disabled attachment is replaced by a short text notice (see
    ``removed_media_notice``) so the model knows something was removed. Three
    places are checked: a message that is itself an attachment item (Responses
    input allows these at the top level), an assistant ``audio`` field, and the
    blocks of a ``content`` list (``output`` for ``function_call_output`` items).
    ``excluded`` holds keys from ``media_occurrences`` that the current
    connection rejected; each such block becomes a ``rejected_media_notice``.
    Tool arguments, reasoning and vendor extensions are not inspected.
    ``LLMResponse`` and ``CacheBoundary`` objects are passed through unchanged;
    their native content is checked later by ``check_replayed_native_media``. The input
    list and its messages are not modified.
    """
    rejected = {
        (index, position)
        for index, _, position, _, key in _occurrences(messages)
        if key in excluded
    }
    result = []
    for message_index, message in enumerate(messages):
        if isinstance(message, (LLMResponse, CacheBoundary)) or not isinstance(message, Mapping):
            result.append(message)
            continue
        item = dict(message)
        kind = attachment_kind(item)
        if kind is not None and kind not in enabled:
            # The whole message is an attachment item, so there is no content
            # list to filter: replace the item with a user notice.
            result.append({"role": "user", "content": removed_media_notice(kind)})
            continue
        slot = "output" if item.get("type") == "function_call_output" else "content"
        if isinstance(item.get(slot), tuple):
            # A tuple of blocks is sent like a list, so it must be filtered like one.
            item[slot] = list(item[slot])
        if (
            item.get("role") == "assistant"
            and item.get("audio") is not None
            and "audio" not in enabled
        ):
            item.pop("audio")
            notice = removed_media_notice("audio")
            if isinstance(item.get(slot), str):
                item[slot] += "\n" + notice
            elif isinstance(item.get(slot), list):
                item[slot] = [*item[slot], {"type": "text", "text": notice}]
            else:
                item[slot] = notice
        content = item.get(slot)
        if isinstance(content, list):
            blocks = []
            for position, block in enumerate(content):
                kind = attachment_kind(block) if isinstance(block, Mapping) else None
                if kind is not None and (
                    kind not in enabled or (message_index, position) in rejected
                ):
                    blocks.append(
                        {
                            "type": "input_text" if slot == "output" else "text",
                            "text": removed_media_notice(kind)
                            if kind not in enabled
                            else rejected_media_notice(kind),
                        }
                    )
                else:
                    blocks.append(block)
            item[slot] = blocks
        result.append(item)
    return result


def check_replayed_native_media(
    native: Mapping[str, Any], enabled: tuple[MediaInput, ...] | None, *, interface: str = "chat"
) -> None:
    """Check the native content of a replayed assistant turn before it is sent.

    ``native`` is the provider's own message for an earlier assistant turn. It is
    replayed as one unit and cannot be partly edited, so media cannot be removed
    from it the way ``filter_media_messages`` removes it from ordinary messages.

    ``enabled=None`` skips both checks: direct projection helpers that do not
    apply media permissions keep their existing contract. Otherwise two checks
    run in order:

    1. If ``native`` contains a media kind that is not in ``enabled``, raise
       ``ReasoningReplayError``. The caller must replay a portable version of the
       turn instead.
    2. Raise ``ValueError`` if ``native`` contains media that ``interface`` cannot
       send (see ``nooa.media_policy``).
    """
    if enabled is None:
        return
    content = native.get("content")
    kinds = ["audio"] if native.get("audio") is not None else []
    if isinstance(content, (list, tuple)):
        kinds.extend(attachment_kind(block) for block in content if isinstance(block, Mapping))
    if any(kind is not None and kind not in enabled for kind in kinds):
        raise ReasoningReplayError(
            "Cannot remove disabled media from opaque assistant state; "
            "supply a portable assistant turn instead."
        )
    from nooa.media_interfaces import reject_unsupported_media

    reject_unsupported_media([native], interface)
