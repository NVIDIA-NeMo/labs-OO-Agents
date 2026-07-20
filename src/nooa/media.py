# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media types for multimodal agent interactions.

Provides media classes for passing rich content (images, audio, video, files)
to LLMs via the show() builtin in CodeAct or as parameters to generation methods.

Media classes are LLM-agnostic — they store raw data and metadata.
Conversion to LLM-specific content blocks happens in runtime/media_capture.py.
"""

import base64
import hashlib
import mimetypes
from pathlib import Path


class Media:
    """Base class for all media attachments.

    Stores raw data (as a data URL or regular URL) plus MIME type.
    Subclasses define the modality (image, audio, video, file) which determines
    how the content block is rendered for the LLM.

    Use the subclasses directly: Image, Audio, Video, File.
    """

    __slots__ = ("_data_url", "_media_type", "_vendor_metadata")

    # Subclasses override this
    _modality: str = "unknown"

    def __init__(
        self,
        data_url: str,
        media_type: str,
        vendor_metadata: dict[str, object] | None = None,
    ):
        self._data_url = data_url
        self._media_type = media_type
        self._vendor_metadata = vendor_metadata or {}

    @property
    def data_url(self) -> str:
        """The data URL or regular URL for this media."""
        return self._data_url

    @property
    def media_type(self) -> str:
        """MIME type (e.g. 'image/png', 'audio/wav', 'application/pdf')."""
        return self._media_type

    @property
    def modality(self) -> str:
        """Modality: 'image', 'audio', 'video', or 'file'."""
        return self._modality

    @property
    def vendor_metadata(self) -> dict[str, object]:
        """Provider-specific hints (e.g. ``{"detail": "high"}`` for OpenAI images)."""
        return self._vendor_metadata

    @classmethod
    def from_file(cls, path: str | Path, **vendor_metadata: object) -> "Media":
        """Load media from a file path."""
        p = Path(path)
        data = p.read_bytes()
        mime, _ = mimetypes.guess_type(str(p))
        media_type = mime or "application/octet-stream"
        return cls.from_bytes(data, media_type=media_type, **vendor_metadata)

    @classmethod
    def from_bytes(cls, data: bytes, *, media_type: str, **vendor_metadata: object) -> "Media":
        """Create media from raw bytes with explicit media type."""
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:{media_type};base64,{b64}"
        return cls(
            data_url=data_url, media_type=media_type, vendor_metadata=vendor_metadata or None
        )

    @classmethod
    def from_url(cls, url: str, *, media_type: str = "", **vendor_metadata: object) -> "Media":
        """Create media from a URL (no download — URL is passed directly to the LLM)."""
        return cls(data_url=url, media_type=media_type, vendor_metadata=vendor_metadata or None)

    def _base64_data(self) -> str:
        """Extract raw base64 data from a data URL."""
        if self._data_url.startswith("data:"):
            return self._data_url.split(",", 1)[1]
        return self._data_url

    @property
    def content_hash(self) -> str:
        """Short SHA-256 hash (8 hex chars) identifying this media's content."""
        return hashlib.sha256(self._data_url.encode()).hexdigest()[:8]

    @property
    def size_bytes(self) -> int | None:
        """Approximate size in bytes of the payload, or None for URL references."""
        if self._data_url.startswith("data:"):
            b64 = self._data_url.split(",", 1)[1]
            return len(b64) * 3 // 4  # base64 → bytes approximation
        return None

    def __repr__(self) -> str:
        size = self.size_bytes
        size_str = f", {size} bytes" if size is not None else ", url"
        return f"{type(self).__name__}({self._media_type}, {self.content_hash}{size_str})"

    def __str__(self) -> str:
        return self.__repr__()


class Image(Media):
    """An image that can be shown to a vision-capable LLM via show().

    Supports three creation patterns:
        Image.from_file("photo.png")           # Load from disk
        Image.from_bytes(data, media_type=...) # From raw bytes
        Image.from_url("https://...")          # From URL (no download)
    """

    _modality = "image"

    @classmethod
    def from_url(
        cls, url: str, *, media_type: str = "image/jpeg", **vendor_metadata: object
    ) -> "Image":
        """Create an image from a URL (no download — URL is passed directly to the LLM)."""
        return cls(data_url=url, media_type=media_type, vendor_metadata=vendor_metadata or None)


class Audio(Media):
    """Audio that can be shown to an audio-capable LLM via show().

    Supports three creation patterns:
        Audio.from_file("clip.wav")            # Load from disk
        Audio.from_bytes(data, media_type=...) # From raw bytes
        Audio.from_url("https://...")          # From URL
    """

    _modality = "audio"

    @classmethod
    def from_url(
        cls, url: str, *, media_type: str = "audio/wav", **vendor_metadata: object
    ) -> "Audio":
        """Create audio from a URL."""
        return cls(data_url=url, media_type=media_type, vendor_metadata=vendor_metadata or None)


class Video(Media):
    """Video that can be shown to a video-capable LLM via show().

    The model perceives both visual frames and (where the model supports it)
    the audio track. Tested with Qwen-VL family models served on
    OpenAI-compatible endpoints (vLLM, NVIDIA inference API), which accept
    ``video_url`` content blocks natively.

    Provider support varies. LiteLLM transforms ``video_url`` blocks for
    providers that support video; on providers that do not, the request
    will be rejected — same contract as Audio.

    Supports three creation patterns:
        Video.from_file("clip.mp4")              # Load from disk
        Video.from_bytes(data, media_type=...)   # From raw bytes
        Video.from_url("https://...")            # From URL (no download)

    Large videos consume significant context (base64 expansion ~4:3 plus
    per-frame token cost on the model side). Prefer URL references over
    inline base64 when the endpoint can fetch the URL.
    """

    _modality = "video"

    @classmethod
    def from_url(
        cls, url: str, *, media_type: str = "video/mp4", **vendor_metadata: object
    ) -> "Video":
        """Create a video reference from a URL (no download — URL is passed directly to the LLM)."""
        return cls(data_url=url, media_type=media_type, vendor_metadata=vendor_metadata or None)


class File(Media):
    """A file (PDF, etc.) that can be shown to an LLM via show().

    Supports three creation patterns:
        File.from_file("report.pdf")           # Load from disk
        File.from_bytes(data, media_type=...) # From raw bytes
        File.from_url("https://...")          # From URL
    """

    _modality = "file"

    @classmethod
    def from_url(
        cls, url: str, *, media_type: str = "application/pdf", **vendor_metadata: object
    ) -> "File":
        """Create a file reference from a URL."""
        return cls(data_url=url, media_type=media_type, vendor_metadata=vendor_metadata or None)
