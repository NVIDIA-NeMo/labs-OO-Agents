# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal, get_args

from pydantic import BeforeValidator

MediaInput = Literal["image", "audio", "video", "pdf"]
"""A kind of media a model can be allowed to receive (its ``media_inputs``)."""

AttachmentKind = Literal["image", "audio", "video", "pdf", "file"]
"""What an attachment is: a ``MediaInput``, or ``"file"`` for any file that is not a PDF."""

_MEDIA_INPUTS: frozenset[str] = frozenset(get_args(MediaInput))


def validate_media_inputs(value: object) -> tuple[MediaInput, ...]:
    """Absent/null/empty is text-only; reject strings, unknowns and duplicates."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("media_inputs must be a list of image, audio, video, pdf")
    if any(not isinstance(item, str) or item not in _MEDIA_INPUTS for item in value):
        raise ValueError("media_inputs values must be image, audio, video, pdf")
    if len(set(value)) != len(value):
        raise ValueError("media_inputs must not contain duplicates")
    return tuple(value)  # type: ignore[return-value]  # checked against MediaInput above


MediaInputs = Annotated[tuple[MediaInput, ...], BeforeValidator(validate_media_inputs)]

# Content-block ``type`` -> attachment kind. Every ``file`` kind is narrowed to
# ``pdf`` or ``file`` by its MIME type in attachment_kind().
_KINDS: dict[str, Literal["image", "audio", "video", "file"]] = {
    "image_url": "image",
    "input_image": "image",
    "image": "image",
    "input_audio": "audio",
    "audio": "audio",
    "output_audio": "audio",
    "video_url": "video",
    "input_video": "video",
    "video": "video",
    "file": "file",
    "input_file": "file",
    "document": "file",
}


def _mime_hints(block: Mapping[str, Any]) -> set[str]:
    # Compare ALL documented hints/data URLs: a PDF label must not hide a
    # conflicting Word data URL. Unknown bytes are never fetched or decoded.
    nodes = [block]
    kind = block.get("type")
    keys = {
        "image_url": ("image_url",),
        "image": ("source",),
        "video_url": ("video_url",),
        "video": ("source",),
        "input_audio": ("input_audio",),
        "audio": ("source",),
        "file": ("file",),
        "document": ("source",),
    }.get(kind, ())
    for key in keys:
        if isinstance(block.get(key), Mapping):
            nodes.append(block[key])
    hints = set()
    for node in nodes:
        for key in ("mime_type", "media_type", "format"):
            value = node.get(key)
            if isinstance(value, str) and "/" in value:
                hints.add(value.lower())
        for key in ("url", "file_data", "image_url", "file_url", "video_url"):
            value = node.get(key)
            if isinstance(value, str) and value.startswith("data:"):
                hints.add(value[5:].split(";", 1)[0].split(",", 1)[0].lower())
    return hints


def attachment_kind(block: Mapping[str, Any]) -> AttachmentKind | None:
    """Return what a content block attaches, or None for text and unknown blocks.

    ``block`` is a provider message block (JSON), hence ``Mapping[str, Any]``.
    A file counts as ``"pdf"`` only when its MIME type is exactly
    ``application/pdf``; a ``.pdf`` name or a file ID alone is not enough.
    """
    kind = _KINDS.get(block.get("type"))
    if kind is None:
        return None
    hints = _mime_hints(block)
    if kind == "file":
        # Every hint must say application/pdf: a PDF label cannot hide a Word data URL.
        return "pdf" if hints == {"application/pdf"} else "file"
    if any(not hint.startswith(kind + "/") for hint in hints):
        return "file"  # a hint of another kind never broadens permissions
    return kind  # e.g. image/jpeg format with an image/png data URL is still an image
