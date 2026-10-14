# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator

MediaInput = Literal["image", "audio", "video", "pdf"]


def validate_media_inputs(value: Any) -> tuple[MediaInput, ...]:
    """Absent/null/empty is text-only; reject strings, unknowns and duplicates."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("media_inputs must be a list of image, audio, video, pdf")
    if any(
        not isinstance(item, str) or item not in {"image", "audio", "video", "pdf"}
        for item in value
    ):
        raise ValueError("media_inputs values must be image, audio, video, pdf")
    if len(set(value)) != len(value):
        raise ValueError("media_inputs must not contain duplicates")
    return tuple(value)


MediaInputs = Annotated[tuple[MediaInput, ...], BeforeValidator(validate_media_inputs)]

_KINDS = {
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


def _mime(block: Mapping[str, Any]) -> str | None:
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
    return next(iter(hints)) if len(hints) == 1 else "conflicting/unsupported" if hints else None


def attachment_kind(block: Mapping[str, Any]) -> str | None:
    """Known wire kinds; PDF requires exactly application/pdf, not a suffix/id."""
    kind = _KINDS.get(block.get("type"))
    if kind is None:
        return None
    mime = _mime(block)
    if kind == "file":
        return "pdf" if mime == "application/pdf" else "file"
    if mime is not None and not mime.startswith(kind + "/"):
        return "file"  # mismatched declarations never broaden permissions
    return kind
