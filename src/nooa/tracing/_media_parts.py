# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe ``show()`` media content blocks for traces and the trace viewer.

``show()`` attaches LiteLLM-shaped content blocks (see
:mod:`nooa.runtime.media_capture`). The message journal stores each block as
a content-addressed string (``_journal_builder._encode_image``) and the
message skeleton refers to it from ``image_hashes``. This module gives both
sides one way to name a block (:func:`media_block_hash`) and to read its kind,
MIME type and payload (:func:`describe_media_block`), so the code-execution
span can point at the same block the LLM call stored, and the viewer can serve
it without inlining multi-megabyte payloads into span listings.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from nooa.tracing._journal_builder import _encode_image, _hash

_AUDIO_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4"}


def media_block_hash(block: Any) -> str:
    """Return the journal block hash of one ``show()`` content block."""
    return _hash(_encode_image(block))


def _parse_data_url(url: str) -> tuple[str, str] | None:
    """``data:<mime>;base64,<b64>`` -> ``(mime, b64)``; None for other URLs."""
    if not url.startswith("data:"):
        return None
    header, sep, payload = url[5:].partition(",")
    if not sep or not header.endswith(";base64"):
        return None
    return header[: -len(";base64")].split(";")[0].strip().lower(), payload


def describe_media_block(block: Any) -> dict[str, Any] | None:
    """Describe one content block, or None if it is not a known media block.

    Returns ``kind`` (``image``/``audio``/``video``/``file``), ``mime_type``
    (may be empty), ``filename`` (files only), and either ``url`` (a remote
    ``http(s)`` URL, served as is) or ``base64`` (the inline payload).
    """
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    url: Any = None
    mime = ""
    raw_b64: str | None = None
    filename: str | None = None
    if kind == "image_url":
        inner = block.get("image_url")
        url = inner.get("url") if isinstance(inner, dict) else inner
        mime = str(inner.get("format") or "") if isinstance(inner, dict) else ""
        kind = "image"
    elif kind == "video_url":
        inner = block.get("video_url")
        url = inner.get("url") if isinstance(inner, dict) else inner
        kind = "video"
    elif kind == "file":
        inner = block.get("file") or {}
        if not isinstance(inner, dict):
            return None
        url = inner.get("file_data") or inner.get("file_url") or inner.get("url")
        mime = str(inner.get("format") or "")
        filename = inner.get("filename") if isinstance(inner.get("filename"), str) else None
    elif kind == "input_audio":
        inner = block.get("input_audio") or {}
        if not isinstance(inner, dict) or not isinstance(inner.get("data"), str):
            return None
        fmt = str(inner.get("format") or "wav").lower()
        mime = _AUDIO_MIME.get(fmt, f"audio/{fmt}")
        raw_b64 = inner["data"]
        kind = "audio"
    else:
        return None

    out: dict[str, Any] = {"kind": kind, "mime_type": mime.lower()}
    if filename:
        out["filename"] = filename
    if raw_b64 is not None:
        out["base64"] = raw_b64
        return out
    if not isinstance(url, str) or not url:
        return None
    parsed = _parse_data_url(url)
    if parsed is not None:
        data_mime, payload = parsed
        out["mime_type"] = data_mime or out["mime_type"]
        out["base64"] = payload
    elif url.startswith(("http://", "https://")):
        out["url"] = url
    else:
        return None
    return out


def media_data_url(desc: dict[str, Any]) -> str | None:
    """The original URL of a described block: a data URL or a remote URL."""
    if "url" in desc:
        return desc["url"]
    if "base64" in desc:
        mime = desc.get("mime_type") or "application/octet-stream"
        return f"data:{mime};base64,{desc['base64']}"
    return None


def media_bytes(desc: dict[str, Any]) -> bytes | None:
    """Decode the inline payload of a described block; None if remote or invalid."""
    payload = desc.get("base64")
    if not isinstance(payload, str):
        return None
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None


def media_summary(block: Any) -> dict[str, Any] | None:
    """Small, payload-free reference to a block: hash, kind, MIME type, size."""
    desc = describe_media_block(block)
    if desc is None:
        return None
    ref: dict[str, Any] = {
        "hash": media_block_hash(block),
        "kind": desc["kind"],
        "mime_type": desc.get("mime_type", ""),
    }
    if desc.get("filename"):
        ref["filename"] = desc["filename"]
    if "url" in desc:
        ref["url"] = desc["url"]
    if "base64" in desc:
        ref["bytes"] = len(desc["base64"]) * 3 // 4
    return ref
