# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-model media capabilities: accepted MIME types, transports and size limits.

``media_inputs`` says which media kinds a model may receive. ``media_capabilities``
narrows each kind further, for one model entry:

- ``mime_types``: the MIME types the model accepts for this kind. Absent: any.
- ``transports``: ``inline`` (base64 data in the request) and/or ``url`` (a
  remote URL the provider downloads). Absent: both.
- ``max_bytes``: the largest inline attachment, in decoded bytes. Absent: no limit.

Each request interface has a provider default in ``PROVIDER_MEDIA_DEFAULTS``. A
model entry overrides it per kind and per field, for example::

    media_capabilities:
      image: {mime_types: [image/png, image/jpeg], transports: [inline], max_bytes: 5242880}

Before a request is sent, an attachment the capabilities rule out is replaced
by a short text note (``nooa.unifiedllm.media_inputs.filter_media_messages``),
so the provider never sees it. An attachment whose MIME type, transport or size
cannot be read from the request (a file ID, a URL without a type hint) is sent:
an unknown fact is not a refusal. Rejections that no capability declares are
still handled by error-based recovery after the provider answers.

Like ``media_inputs``, these settings describe one route: a call that goes to
another model or endpoint does not inherit them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

from nooa.media_inputs import MediaInput, _mime_hints

Transport = Literal["inline", "url"]
TRANSPORTS: tuple[Transport, ...] = get_args(Transport)
_KINDS: tuple[MediaInput, ...] = get_args(MediaInput)
_FIELDS = frozenset({"mime_types", "transports", "max_bytes"})
_MIME_ALIASES = {"image/jpg": "image/jpeg"}
_MIB = 1024 * 1024


def canonical_mime_type(value: str) -> str:
    """``value`` in lower case without parameters, with ``image/jpg`` as ``image/jpeg``."""
    value = value.split(";", 1)[0].strip().lower()
    return _MIME_ALIASES.get(value, value)


@dataclass(frozen=True)
class MediaKindCapabilities:
    """What a model accepts for one media kind. The defaults restrict nothing."""

    mime_types: frozenset[str] | None = None  # None: any MIME type
    transports: frozenset[Transport] = frozenset(TRANSPORTS)
    max_bytes: int | None = None  # largest inline attachment; None: no limit

    @property
    def unrestricted(self) -> bool:
        return (
            self.mime_types is None
            and self.transports == frozenset(TRANSPORTS)
            and self.max_bytes is None
        )


@dataclass(frozen=True)
class MediaCapabilities:
    """What a model accepts for each media kind (see the module docstring)."""

    image: MediaKindCapabilities = field(default_factory=MediaKindCapabilities)
    audio: MediaKindCapabilities = field(default_factory=MediaKindCapabilities)
    video: MediaKindCapabilities = field(default_factory=MediaKindCapabilities)
    pdf: MediaKindCapabilities = field(default_factory=MediaKindCapabilities)

    def for_kind(self, kind: str) -> MediaKindCapabilities | None:
        """The capabilities of ``kind``, or None for a kind that is not a MediaInput."""
        return getattr(self, kind) if kind in _KINDS else None

    @property
    def unrestricted(self) -> bool:
        """True when no kind declares any limit (requests are sent unchanged)."""
        return all(getattr(self, kind).unrestricted for kind in _KINDS)

    def refusal(self, block: Mapping[str, Any], kind: str) -> str | None:
        """Why the model cannot take this attachment block, or None if it can.

        The reason completes "this model does not accept ...", for example
        ``"image/heic images"``. Only facts the block shows are checked.
        """
        caps = self.for_kind(kind)
        if caps is None or caps.unrestricted:
            return None
        plural = _PLURAL[kind]
        if caps.mime_types is not None:
            refused = sorted(block_mime_types(block) - caps.mime_types)
            if refused:
                return f"{refused[0]} {plural}"
        transport = block_transport(block)
        if transport is not None and transport not in caps.transports:
            return f"{plural} sent as {'remote URLs' if transport == 'url' else 'inline data'}"
        if caps.max_bytes is not None and transport == "inline":
            size = inline_size(block)
            if size is not None and size > caps.max_bytes:
                return f"inline {plural} larger than {_size_text(caps.max_bytes)}"
        return None


_PLURAL = {"image": "images", "audio": "audio clips", "video": "videos", "pdf": "PDF files"}


def _size_text(size: int) -> str:
    return f"{size / _MIB:g} MiB" if size % (_MIB // 4) == 0 else f"{size} bytes"


UNRESTRICTED = MediaCapabilities()

# The image types, transports and size limit native decision requests accepted
# before per-model capabilities existed (nooa.decisions.media). Kept as the
# decision default so existing decision entries behave exactly as before.
DECISION_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
DECISION_MAX_IMAGE_BYTES = 20 * _MIB

# Provider defaults, by request interface (nooa.request_interface). Only the
# native decision interface declares limits. Every other interface (``chat``:
# OpenAI-compatible Chat Completions for every LiteLLM provider but Anthropic;
# ``anthropic``: native Anthropic Messages; ``responses``) defaults to
# UNRESTRICTED: NOOA has never filtered by MIME type, transport or size on
# these routes and no provider limit is verified here, so their requests stay
# unchanged. A model entry declares what it knows.
PROVIDER_MEDIA_DEFAULTS: dict[str, MediaCapabilities] = {
    "decision": MediaCapabilities(
        image=MediaKindCapabilities(
            mime_types=DECISION_IMAGE_MIME_TYPES, max_bytes=DECISION_MAX_IMAGE_BYTES
        )
    ),
}


def validate_media_capabilities(value: object) -> dict[str, dict[str, Any]] | None:
    """Check a ``media_capabilities`` entry setting and return it normalized.

    Absent/None means "provider default". The result maps each declared kind
    to the declared fields only: ``mime_types`` (sorted, canonical),
    ``transports`` (in ``TRANSPORTS`` order) and ``max_bytes``.

    Raises:
        ValueError: For unknown kinds or fields, malformed values, an empty
            ``mime_types`` or ``transports`` list (leave the kind out of
            ``media_inputs`` instead), or MIME types of another kind.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("media_capabilities must map image, audio, video or pdf to settings")
    result: dict[str, dict[str, Any]] = {}
    for kind, settings in value.items():
        if kind not in _KINDS:
            raise ValueError(f"media_capabilities kinds must be image, audio, video, pdf: {kind!r}")
        if not isinstance(settings, Mapping) or set(settings) - _FIELDS:
            raise ValueError(
                f"media_capabilities.{kind} may set only mime_types, transports and max_bytes"
            )
        entry: dict[str, Any] = {}
        if (mimes := settings.get("mime_types")) is not None:
            entry["mime_types"] = _validate_mime_types(kind, mimes)
        if (transports := settings.get("transports")) is not None:
            if (
                not isinstance(transports, (list, tuple))
                or not transports
                or any(t not in TRANSPORTS for t in transports)
                or len(set(transports)) != len(transports)
            ):
                raise ValueError(
                    f"media_capabilities.{kind}.transports must be a nonempty list of inline, url"
                )
            entry["transports"] = [t for t in TRANSPORTS if t in transports]
        if (limit := settings.get("max_bytes")) is not None:
            if type(limit) is not int or limit <= 0:
                raise ValueError(f"media_capabilities.{kind}.max_bytes must be a positive integer")
            entry["max_bytes"] = limit
        result[kind] = entry
    return result


def _validate_mime_types(kind: str, value: object) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) for item in value)
    ):
        raise ValueError(f"media_capabilities.{kind}.mime_types must be a nonempty list of strings")
    mimes = sorted({canonical_mime_type(item) for item in value})
    for mime in mimes:
        if kind == "pdf":
            valid = mime == "application/pdf"
        else:
            valid = mime.startswith(kind + "/") and len(mime) > len(kind) + 1
        if not valid:
            expected = "application/pdf" if kind == "pdf" else f"{kind}/<subtype>"
            raise ValueError(
                f"media_capabilities.{kind}.mime_types must be {expected} values: {mime!r}"
            )
    return mimes


def resolve_media_capabilities(interface: str, override: object = None) -> MediaCapabilities:
    """The provider default for ``interface`` with an entry's ``media_capabilities`` applied.

    Each field the entry declares replaces the default's field for that kind;
    undeclared fields keep the default. Interfaces without an entry in
    ``PROVIDER_MEDIA_DEFAULTS`` default to ``UNRESTRICTED``.
    """
    caps = PROVIDER_MEDIA_DEFAULTS.get(interface, UNRESTRICTED)
    for kind, settings in (validate_media_capabilities(override) or {}).items():
        current = getattr(caps, kind)
        updates: dict[str, Any] = {}
        if "mime_types" in settings:
            updates["mime_types"] = frozenset(settings["mime_types"])
        if "transports" in settings:
            updates["transports"] = frozenset(settings["transports"])
        if "max_bytes" in settings:
            updates["max_bytes"] = settings["max_bytes"]
        caps = replace(caps, **{kind: replace(current, **updates)})
    return caps


# Nested nodes that carry an attachment's URL or data, by block key.
_NODE_KEYS = ("image_url", "video_url", "audio_url", "input_audio", "file", "source")
_URL_KEYS = ("url", "image_url", "file_url", "video_url", "audio_url", "file_data")


def _nodes(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [block, *(block[k] for k in _NODE_KEYS if isinstance(block.get(k), Mapping))]


def block_mime_types(block: Mapping[str, Any]) -> set[str]:
    """Every MIME type the block states (type hints and data URLs), canonical.

    LiteLLM ``input_audio`` blocks name their format without a slash (``wav``);
    that counts as ``audio/wav``.
    """
    hints = {canonical_mime_type(hint) for hint in _mime_hints(block)}
    audio = block.get("input_audio")
    if block.get("type") == "input_audio" and isinstance(audio, Mapping):
        fmt = audio.get("format")
        if isinstance(fmt, str) and fmt and "/" not in fmt:
            hints.add(canonical_mime_type("audio/" + fmt))
    return hints


def block_transport(block: Mapping[str, Any]) -> Transport | None:
    """``inline`` if the block carries its bytes, ``url`` for a remote URL, else None."""
    remote = False
    for node in _nodes(block):
        if node.get("type") == "base64" or (
            node is not block and isinstance(node.get("data"), str) and node["data"]
        ):
            return "inline"
        for key in _URL_KEYS:
            value = node.get(key)
            if isinstance(value, str):
                scheme = value[:8].lower()
                if scheme.startswith("data:"):
                    return "inline"
                if scheme.startswith(("http://", "https://")):
                    remote = True
    return "url" if remote else None


def inline_size(block: Mapping[str, Any]) -> int | None:
    """Decoded size of the block's inline base64 payload, or None if it has none."""
    for node in _nodes(block):
        candidates = [node.get(key) for key in _URL_KEYS]
        if node is not block:
            candidates.append(node.get("data"))
        for value in candidates:
            if not isinstance(value, str) or not value:
                continue
            if value[:5].lower() == "data:":
                head, _, value = value.partition(",")
                if ";base64" not in head.lower():
                    return len(value.encode())
            elif value[:8].lower().startswith(("http://", "https://")):
                continue
            value = value.strip()
            return len(value) * 3 // 4 - len(value) + len(value.rstrip("="))
    return None


__all__ = [
    "DECISION_IMAGE_MIME_TYPES",
    "DECISION_MAX_IMAGE_BYTES",
    "PROVIDER_MEDIA_DEFAULTS",
    "TRANSPORTS",
    "UNRESTRICTED",
    "MediaCapabilities",
    "MediaKindCapabilities",
    "Transport",
    "block_mime_types",
    "block_transport",
    "canonical_mime_type",
    "inline_size",
    "resolve_media_capabilities",
    "validate_media_capabilities",
]
