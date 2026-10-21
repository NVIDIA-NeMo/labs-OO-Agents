# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, request-local image attachments for native decision transports.

Only direct named Image inputs are collected. Nothing fetches remote URLs or
walks hidden object state. Permission is separate from the wire transport.
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
    """One image, associated with all its direct named inputs.

    ``alias`` is the token written in JSON state. Encoded bytes/URL are kept
    out of repr, and never embedded in JSON state. Unknown vendor metadata is
    not forwarded; only the documented image detail hint is supported.
    """

    alias: str
    input_names: tuple[str, ...]
    image_url: str = field(repr=False)
    media_type: str
    detail: str | None = None

    def decoded_bytes(self) -> bytes | None:
        """Validate URI and MIME, returning local bytes or None for HTTP(S).

        Signature checks below are deliberately not a complete image decoder.
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
    """Validate permissions without claiming unsupported decision modalities."""
    permissions = validate_media_inputs(value)
    if any(kind != "image" for kind in permissions):
        raise ValueError(
            "The NOOA decision client currently supports image attachments only. Use media_inputs: [image], or choose a chat model for audio, video or PDFs."
        )
    return permissions


def validate_attachments(
    attachments: tuple[DecisionAttachment, ...], media_inputs: object
) -> tuple[bytes | None, ...]:
    """Preflight even manually constructed requests before any HTTP attempt."""
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
    """Prevent model serializers from hiding unsupported native Media."""
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
    """Collect bound top-level Media; one repeated object has several names.

    As in Predict, binding happens first, avoiding positional/keyword overlap.
    Distinct Image objects are not content-hash deduplicated.
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
