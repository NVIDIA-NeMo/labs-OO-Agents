# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media capture for CodeAct execution via show().

Uses a ContextVar-based collector (same pattern as stdout/stderr capture)
to accumulate multimodal content blocks during execute_python() calls.

All content blocks follow LiteLLM's format conventions:
- Images:  {"type": "image_url", "image_url": {"url": ..., "format": ...}}
- Audio:   {"type": "input_audio", "input_audio": {"data": ..., "format": ...}}
- Video:   {"type": "video_url", "video_url": {"url": ...}}
- Files:   {"type": "file", "file": {"file_data": "data:...;base64,...", "filename": ...}}

LiteLLM automatically converts these to provider-native formats.
See: https://docs.litellm.ai/docs/completion/vision
"""

import base64
import contextvars
import io
from typing import Any


class _MediaBuffer:
    """Bounded collector of media content blocks captured by ``show()``.

    Same self-enforcing principle as :class:`TruncatingStringIO` (the cap
    lives with the buffer), but signals overflow via a ``bool`` return from
    :meth:`append` rather than by silently truncating the payload —
    media blocks are atomic and cannot be partially kept.

    ``ActorRuntime`` constructs one per ``execute_python`` cell from
    ``MediaCaptureConfig.max_attachments_per_execution``.
    """

    def __init__(self, max_attachments: int):
        self.blocks: list[dict[str, Any]] = []
        self.max_attachments = max_attachments

    def append(self, block: dict[str, Any]) -> bool:
        """Append ``block``. Returns ``False`` if the buffer is full."""
        if len(self.blocks) >= self.max_attachments:
            return False
        self.blocks.append(block)
        return True


# Task-local media buffer for async-safe capture.
_media_buffer_var: contextvars.ContextVar[_MediaBuffer | None] = contextvars.ContextVar(
    "media_buffer", default=None
)


def media_to_content_block(media: Any) -> dict[str, Any]:
    """Convert a Media object to a LiteLLM-compatible content block.

    Uses LiteLLM's universal formats. Provider-specific conversion
    (e.g. Anthropic image source format) is handled by LiteLLM.
    See: https://docs.litellm.ai/docs/completion/vision
    """
    from nooa.media import Audio, File, Image, Media, Video

    if not isinstance(media, Media):
        raise TypeError(f"Expected Media (Image/Audio/Video/File), got {type(media).__name__}")

    if isinstance(media, Image):
        image_url_dict: dict[str, Any] = {"url": media.data_url}
        # Add format hint — used by Anthropic/Bedrock/Vertex AI, ignored by OpenAI
        if media.media_type and media.media_type != "application/octet-stream":
            image_url_dict["format"] = media.media_type
        # Merge vendor metadata (e.g. detail="high" for OpenAI)
        if media.vendor_metadata:
            image_url_dict.update(media.vendor_metadata)
        return {"type": "image_url", "image_url": image_url_dict}

    if isinstance(media, Audio):
        # LiteLLM input_audio format: {"type": "input_audio", "input_audio": {"data": base64, "format": "wav"}}
        fmt = media.media_type.split("/")[-1] if media.media_type else "wav"
        return {
            "type": "input_audio",
            "input_audio": {"data": media._base64_data(), "format": fmt},
        }

    if isinstance(media, Video):
        # LiteLLM video_url format: {"type": "video_url", "video_url": {"url": ...}}
        video_url_dict: dict[str, Any] = {"url": media.data_url}
        if media.vendor_metadata:
            video_url_dict.update(media.vendor_metadata)
        return {"type": "video_url", "video_url": video_url_dict}

    if isinstance(media, File):
        # LiteLLM file format
        return {
            "type": "file",
            "file": {"file_data": media.data_url, "filename": "attachment"},
        }

    # Generic fallback for unknown Media subclasses
    return {"type": "image_url", "image_url": {"url": media.data_url}}


# Backward-compatible alias
image_to_content_block = media_to_content_block


def show(obj: Any) -> None:
    """Display an image, audio clip, video, or file so you can see/hear it.

    Call show() on any Image, Audio, Video, or File object to perceive its contents.
    Also accepts PIL images and matplotlib figures (auto-converted to PNG).
    """
    from nooa.media import Media

    block: dict[str, Any]
    if isinstance(obj, Media):
        block = media_to_content_block(obj)
    else:
        converted = _try_auto_convert(obj)
        if converted is None:
            raise TypeError(
                f"show() expects Image, Audio, Video, File, PIL.Image, or matplotlib Figure, "
                f"got {type(obj).__name__}"
            )
        block = converted

    buf = _media_buffer_var.get()
    if buf is not None:
        if not buf.append(block):
            print(f"[show() limit reached ({buf.max_attachments}), attachment not added]")
            return
        print(f"[shown: {obj}]")
    else:
        print(f"[show() called outside execution context, not captured: {obj}]")


def _try_auto_convert(obj: Any) -> dict[str, Any] | None:
    """Try to convert a non-Media object to a content block.

    Supports PIL.Image.Image and matplotlib.figure.Figure via lazy imports.
    Returns None if the object is not a recognized type.
    """
    block = _try_pil_to_content_block(obj)
    if block is not None:
        return block
    return _try_matplotlib_to_content_block(obj)


# Alias used by tests covering line 113 (the try_auto_convert return path).
_to_content_block = _try_auto_convert


def _try_pil_to_content_block(obj: Any) -> dict[str, Any] | None:
    """Convert a PIL Image to an image_url content block, or None."""
    try:
        from PIL import Image as PILImage  # type: ignore[import-not-found]

        if isinstance(obj, PILImage.Image):
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "format": "image/png"},
            }
    except ImportError:
        pass
    return None


def _try_matplotlib_to_content_block(obj: Any) -> dict[str, Any] | None:
    """Convert a matplotlib Figure to an image_url content block, or None."""
    try:
        from matplotlib.figure import Figure  # type: ignore[import-not-found]

        if isinstance(obj, Figure):
            buf = io.BytesIO()
            obj.savefig(buf, format="png", bbox_inches="tight")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "format": "image/png"},
            }
    except ImportError:
        pass
    return None
