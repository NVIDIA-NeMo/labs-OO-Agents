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


def filter_media_messages(messages, enabled: tuple[MediaInput, ...]):
    """Return a copy of ``messages`` with media that is not ``enabled`` replaced.

    Each disabled attachment is replaced by a short text notice (see
    ``removed_media_notice``) so the model knows something was removed. Three
    places are checked: a message that is itself an attachment item (Responses
    input allows these at the top level), an assistant ``audio`` field, and the
    blocks of a ``content`` list (``output`` for ``function_call_output`` items).
    Tool arguments, reasoning and vendor extensions are not inspected.
    ``LLMResponse`` and ``CacheBoundary`` objects are passed through unchanged;
    their native content is checked later by ``guard_native_media``. The input
    list and its messages are not modified.
    """
    result = []
    for message in messages:
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
            for block in content:
                kind = attachment_kind(block) if isinstance(block, Mapping) else None
                if kind is not None and kind not in enabled:
                    blocks.append(
                        {
                            "type": "input_text" if slot == "output" else "text",
                            "text": removed_media_notice(kind),
                        }
                    )
                else:
                    blocks.append(block)
            item[slot] = blocks
        result.append(item)
    return result


def guard_native_media(
    native: Mapping[str, Any], enabled: tuple[MediaInput, ...] | None, *, interface: str = "chat"
) -> None:
    """Check the native content of a replayed assistant turn before it is sent.

    ``native`` is the provider's own message for an earlier assistant turn. It is
    replayed as one unit and cannot be partly edited, so media cannot be removed
    from it the way ``filter_media_messages`` removes it from ordinary messages.

    Two checks run in order:

    1. If ``enabled`` is not None and ``native`` contains a media kind that is not
       enabled, raise ``ReasoningReplayError``. The caller must replay a portable
       version of the turn instead. ``enabled=None`` skips this check; direct
       projection helpers use it when they do not apply media permissions.
    2. Raise ``ValueError`` if ``native`` contains media that ``interface`` cannot
       send (see ``nooa.media_policy``).
    """
    if enabled is not None:
        content = native.get("content")
        kinds = ["audio"] if native.get("audio") is not None else []
        if isinstance(content, (list, tuple)):
            kinds.extend(attachment_kind(block) for block in content if isinstance(block, Mapping))
        if any(kind is not None and kind not in enabled for kind in kinds):
            raise ReasoningReplayError(
                "Cannot remove disabled media from opaque assistant state; "
                "supply a portable assistant turn instead."
            )
    from nooa.media_policy import reject_unsupported_media

    reject_unsupported_media([native], interface)
