# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect and validate image attachments for native decision-model requests.

A decision method may take ``nooa.media.Image`` arguments. This module turns
those arguments into ``DecisionAttachment`` objects that a decision client can
send next to the JSON state, and checks them before any HTTP request is made.

Terms used here:

- *Named input*: a top-level argument of the decision method, by parameter name.
  Only Images passed directly as named inputs become attachments. Images nested
  inside other values (models, dicts, lists, context blocks, events) are
  rejected, not collected.
- *Alias*: the placeholder text ``<decision-image-N>`` written into the JSON
  state instead of the image, so the model can tell which input an image is.
- *Media permission*: the model entry's ``media_inputs`` setting. Images are
  allowed only when it lists ``"image"``. The permission is checked separately
  from how the image is encoded on the wire (JSON or multipart).

Remote image URLs are validated but never fetched here, and limits on image
count and size are enforced per request.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

from nooa.media import Image, Media
from nooa.media_inputs import MediaInput, validate_media_inputs

MAX_DECISION_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DECISION_TOTAL_IMAGE_BYTES = 50 * 1024 * 1024
MAX_DECISION_IMAGES = 16
MAX_DECISION_IMAGE_URL_LENGTH = 8192
_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


@dataclass(frozen=True, slots=True)
class DecisionAttachment:
    """One image to send with a decision request.

    ``alias`` is the placeholder written into the JSON state in place of the
    image. ``input_names`` lists every named input that passed this same Image
    object. ``image_url`` is a ``data:`` URL or an HTTP(S) URL; it is hidden
    from ``repr`` and never copied into the JSON state. ``detail`` is the
    optional image detail hint (``low``, ``high``, ``auto`` or ``original``);
    other vendor metadata on the Image is not supported.
    """

    alias: str
    input_names: tuple[str, ...]
    image_url: str = field(repr=False)
    media_type: str
    detail: str | None = None

    def decoded_bytes(self) -> bytes | None:
        """Validate the image URL and MIME type and return the image bytes.

        Returns:
            The decoded bytes for a base64 ``data:`` URL, or ``None`` for an
            HTTP(S) URL (remote images are checked but not downloaded).

        Raises:
            ValueError: If the MIME type is unsupported, the data URL is
                malformed or larger than 20 MiB, or the HTTP(S) URL is too
                long, contains credentials or a fragment, or is otherwise
                invalid.

        Only the URL and the base64 encoding are checked; the image content
        itself is not decoded.
        """
        if self.media_type not in _IMAGE_MIMES:
            raise ValueError(
                "Decision image MIME must be image/png, image/jpeg, image/webp or image/gif"
            )
        url = self.image_url
        if not isinstance(url, str):
            raise ValueError("Decision image URI must be a string")
        if url.startswith("data:"):
            prefix = f"data:{self.media_type};base64,"
            if not url.startswith(prefix):
                raise ValueError(
                    "Decision image data URI must have matching MIME and base64 encoding"
                )
            encoded = url[len(prefix) :]
            if len(encoded) > 4 * ((MAX_DECISION_IMAGE_BYTES + 2) // 3):
                raise ValueError("Decision image exceeds the 20 MiB size limit")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Decision image contains invalid base64") from exc
            if not data or len(data) > MAX_DECISION_IMAGE_BYTES:
                raise ValueError("Decision image must be nonempty and at most 20 MiB")
            return data
        if len(url) > MAX_DECISION_IMAGE_URL_LENGTH or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 or char == "\\" for char in url
        ):
            raise ValueError("Decision image HTTP(S) URI is invalid or too long")
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
        except ValueError as exc:
            raise ValueError("Decision image HTTP(S) URI is invalid") from exc
        if (
            not valid
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("Decision image URI must be a data URL or credential-free HTTP(S) URL")
        return None


def decision_media_inputs(value: object) -> tuple[MediaInput, ...]:
    """Validate a ``media_inputs`` setting for a decision client.

    Decision clients support images only, so any other media kind (audio,
    video, pdf) raises ``ValueError`` instead of being silently accepted.
    Returns the validated permissions.
    """
    permissions = validate_media_inputs(value)
    if any(kind != "image" for kind in permissions):
        raise ValueError(
            "The NOOA decision client currently supports image attachments only. Use media_inputs: [image], or choose a chat model for audio, video or PDFs."
        )
    return permissions


def validate_attachments(
    attachments: tuple[DecisionAttachment, ...], media_inputs: object
) -> tuple[bytes | None, ...]:
    """Check attachments against the media permission and the request limits.

    This runs before any HTTP request, also for requests built by hand
    rather than through ``collect_decision_attachments``. It checks that
    images are permitted, that there are at most 16, that aliases and input
    names are unique, that the detail hint is valid, and that the total size
    stays within 50 MiB.

    Returns:
        The decoded bytes of each attachment, in order (``None`` for HTTP(S)
        URLs).

    Raises:
        ValueError: If any check fails.
    """
    permissions = decision_media_inputs(media_inputs)
    if attachments and "image" not in permissions:
        raise ValueError(
            "Decision image attachments require explicit media_inputs: [image] for this route"
        )
    if len(attachments) > MAX_DECISION_IMAGES:
        raise ValueError("Decision requests support at most 16 images")
    aliases: set[str] = set()
    names: set[str] = set()
    decoded: list[bytes | None] = []
    total = 0
    for attachment in attachments:
        if not isinstance(attachment, DecisionAttachment):
            raise ValueError("Decision attachments must be normalized DecisionAttachment images")
        if (
            not isinstance(attachment.alias, str)
            or not re.fullmatch(r"<decision-image-[0-9]+>", attachment.alias)
            or attachment.alias in aliases
        ):
            raise ValueError("Decision image aliases must be unique <decision-image-N> tokens")
        aliases.add(attachment.alias)
        if (
            not isinstance(attachment.input_names, tuple)
            or not attachment.input_names
            or any(
                not isinstance(name, str) or not name or name in names
                for name in attachment.input_names
            )
            or len(set(attachment.input_names)) != len(attachment.input_names)
        ):
            raise ValueError("Decision images need unique nonempty named input associations")
        names.update(attachment.input_names)
        if attachment.detail not in (None, "low", "high", "auto", "original"):
            raise ValueError("Decision image detail must be low, high, auto or original")
        data = attachment.decoded_bytes()
        total += len(data) if data is not None else 0
        if total > MAX_DECISION_TOTAL_IMAGE_BYTES:
            raise ValueError("Decision images exceed the 50 MiB total size limit")
        decoded.append(data)
    return tuple(decoded)


def reject_nested_media(value: Any, seen: set[int] | None = None) -> None:
    """Raise ``TypeError`` if ``value`` contains a Media object anywhere inside.

    Pydantic models, dataclasses, mappings (keys and values) and sequences are
    walked recursively. This runs before serialization because a custom
    serializer could otherwise turn a nested Image into plain text that would
    reach the model unnoticed. ``seen`` guards against reference cycles.
    """
    if isinstance(value, Media):
        raise TypeError("Nested Media is unsupported; pass a permitted top-level named Image input")
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            reject_nested_media(getattr(value, name), seen)
        for item in (value.model_extra or {}).values():
            reject_nested_media(item, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            reject_nested_media(getattr(value, field.name), seen)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            reject_nested_media(key, seen)
            reject_nested_media(item, seen)
    elif isinstance(value, (Sequence, set, frozenset)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            reject_nested_media(item, seen)


def collect_decision_attachments(
    parameters: Mapping[str, object], media_inputs: object
) -> tuple[dict[str, str], tuple[DecisionAttachment, ...]]:
    """Turn the Image arguments of a decision call into attachments.

    ``parameters`` are the call's arguments after binding to the method
    signature (as in Predict), so each value appears once under its parameter
    name whether it was passed by position or by keyword. When the same
    Image object is passed under several names, it becomes one attachment
    that lists all of those names. Two different Image objects with equal
    content stay two attachments.

    Returns:
        A mapping from input name to alias, for writing into the JSON state,
        and the validated attachments.

    Raises:
        TypeError: If an argument is a Media object other than an Image.
        ValueError: If an Image carries vendor metadata other than ``detail``,
            or the attachments fail ``validate_attachments``.
    """
    references: dict[str, str] = {}
    attachments: list[DecisionAttachment] = []
    seen: dict[int, int] = {}
    for name, value in parameters.items():
        if not isinstance(value, Media):
            continue
        if not isinstance(value, Image):
            raise TypeError(
                f"Decision input {name!r}: unsupported Media {type(value).__name__}; only native Image inputs are supported"
            )
        if id(value) in seen:
            index = seen[id(value)]
            attachments[index] = replace(
                attachments[index], input_names=(*attachments[index].input_names, name)
            )
        else:
            index = len(attachments)
            seen[id(value)] = index
            metadata = value.vendor_metadata
            if set(metadata) - {"detail"}:
                raise ValueError(
                    f"Decision input {name!r}: unsupported image vendor metadata; only detail is supported"
                )
            attachments.append(
                DecisionAttachment(
                    alias=f"<decision-image-{index}>",
                    input_names=(name,),
                    image_url=value.data_url,
                    media_type=value.media_type,
                    detail=metadata.get("detail"),  # type: ignore[arg-type]
                )
            )
        references[name] = attachments[index].alias
    result = tuple(attachments)
    validate_attachments(result, media_inputs)
    return references, result
