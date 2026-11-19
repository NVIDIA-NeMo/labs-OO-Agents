# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

import hashlib
import json
import re
from collections.abc import Collection, Mapping
from typing import Any

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.media_inputs import MediaInput, _mime_hints, attachment_kind
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


# Inline image formats the common model APIs accept (OpenAI, Anthropic and
# Bedrock). This is a common floor, not a per-provider list (there is none
# yet): Gemini does not take GIF and OpenAI refuses animated GIF. Other types,
# such as SVG, BMP or TIFF, are refused by some providers; kept in history,
# such an image would fail every later request.
SENDABLE_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_IMAGE_TYPE_ALIASES = {"image/jpg": "image/jpeg"}


def _canonical_image_type(media_type: str) -> str:
    """``media_type`` in lower case without parameters, with ``image/jpg`` as ``image/jpeg``."""
    media_type = media_type.split(";", 1)[0].strip().lower()
    return _IMAGE_TYPE_ALIASES.get(media_type, media_type)


def unsendable_image_type(block: Mapping[str, Any]) -> str | None:
    """The MIME type of an inline image block models do not accept, else None.

    Every MIME hint of the block is checked (data URL, ``format``,
    ``media_type``, ``mime_type``; see ``nooa.media_inputs._mime_hints``),
    after lower-casing and dropping parameters. One hint outside
    ``SENDABLE_IMAGE_TYPES`` makes the image unsendable; two accepted hints
    that disagree do not. Only inline images are checked: for a remote URL the
    provider reads the real type from the server it downloads from, so a
    ``format`` hint next to it is not trusted to be wrong.
    """
    if attachment_kind(block) != "image" or not _is_inline(block):
        return None
    refused = sorted(
        {_canonical_image_type(hint) for hint in _mime_hints(block)} - SENDABLE_IMAGE_TYPES
    )
    return refused[0] if refused else None


def _is_inline(block: Mapping[str, Any]) -> bool:
    """True if ``block`` carries its bytes: a ``data:`` URL or a base64 ``source``."""
    nodes = [
        block,
        *(block[key] for key in ("image_url", "source") if isinstance(block.get(key), Mapping)),
    ]
    for node in nodes:
        if node.get("type") == "base64":
            return True
        for key in ("url", "image_url"):
            value = node.get(key)
            if isinstance(value, str) and value[:5].lower() == "data:":
                return True
    return False


def _sendable_image_block(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """``block`` with its accepted MIME hints in canonical form, for the wire.

    ``image/JPG; q=1`` is sent as ``image/jpeg`` and ``DATA:image/PNG;...`` as
    ``data:image/png;...``. The caller's block is not modified; an unchanged
    block is returned as is.
    """

    def canonical(node: Mapping[str, Any]) -> Mapping[str, Any]:
        changed = {}
        for key in ("mime_type", "media_type", "format"):
            value = node.get(key)
            if isinstance(value, str) and "/" in value and _canonical_image_type(value) != value:
                changed[key] = _canonical_image_type(value)
        for key in ("url", "image_url"):
            value = node.get(key)
            if isinstance(value, str) and value[:5].lower() == "data:":
                end = min((i for i in (value.find(";"), value.find(",")) if i >= 0), default=5)
                url = "data:" + _canonical_image_type(value[5:end]) + value[end:]
                if url != value:
                    changed[key] = url
        return {**node, **changed} if changed else node

    result = canonical(block)
    for key in ("image_url", "source"):
        if isinstance(block.get(key), Mapping):
            node = canonical(block[key])
            if node is not block[key]:
                result = {**result, key: node}
    return result


def unsendable_image_notice(media_type: str) -> str:
    """The text the model sees in place of an image in a format models do not accept."""
    return (
        f"[An image was removed here: {media_type} images cannot be sent to a model. "
        "Only PNG, JPEG, GIF and WebP are accepted; convert it (for example render "
        "an SVG to PNG) and show it again.]"
    )


def rejected_media_notice(kind: str) -> str:
    """The text the model sees in place of media the provider could not read."""
    subject = _NOTICE_SUBJECT.get(kind, ("A file", "files"))[0]
    return f"[{subject} was removed here because the model provider could not read it.]"


# "Too many images: 7 (max 5)" (GLM 5.3 hub) and generic phrasings of a
# per-request image count limit. The number captured is the limit M.
_IMAGE_LIMIT_RES = (
    re.compile(r"too many images\b.{0,40}?\b(?:max(?:imum)?|limit)\D{0,12}(\d+)"),
    re.compile(
        r"\b(?:at most|maximum of|max(?:imum)?(?: number of)?|up to|no more than|limit of)"
        r"\s*:?\s*(\d+)\s*(?:images?|image inputs?|image_url)\b"
    ),
    re.compile(
        r"\b(?:images?|image inputs?) (?:per request |count |number )?"
        r"(?:limit|maximum|max)\s*(?:is|of|=|:)?\s*(\d+)\b"
    ),
    re.compile(r"\bexceeds? the (?:maximum|max|limit) of (\d+) images?\b"),
)


def image_limit(exc: BaseException) -> int | None:
    """The per-request image limit M a provider rejection names, else None.

    Not a rejection of one image: the request carried more images than the
    model takes (``Too many images: 7 (max 5)``). Looks through the
    exception's body message and its cause chain.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        body = getattr(cur, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else None
        texts = [str(cur)]
        if isinstance(error, dict) and error.get("message"):
            texts.append(str(error["message"]))
        for text in texts:
            lowered = text.lower()
            if "image" not in lowered:
                continue
            for pattern in _IMAGE_LIMIT_RES:
                match = pattern.search(lowered)
                if match:
                    return int(match.group(1))
        cur = cur.__cause__ or cur.__context__
    return None


_FORMAT_NOUN = {
    "image": "image",
    "audio": "audio",
    "video": "video",
    "pdf": "(?:pdf|file|document)",
}


def _error_message(exc) -> str:
    """The provider's error message in lower case: ``body.error.message`` if set, else ``str(exc)``."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", str(exc))).lower()
    return str(exc).lower()


def _unsupported_format(exc, kind) -> bool:
    """True if ``exc`` says the attachment's format is unsupported.

    The NVIDIA Inference Hub (a LiteLLM proxy) returns this as HTTP 500:
    "Unsupported image format: svg+xml. Supported formats: [...]". The error
    is about the request, so retrying it unchanged fails every time. The
    phrase must name a format value (after a colon, ``=`` or a quote) or be
    followed by the formats that are supported, so "unsupported image format
    conversion plugin" does not count.
    """
    noun = _FORMAT_NOUN.get(kind, "file")
    return bool(
        re.search(
            rf"\bunsupported {noun} (?:format|type)s?"
            r"(?:\s*[:=]|\s+['\"`]|\b[^.]{0,80}\b(?:supported|allowed|accepted) (?:formats|types)\b)",
            _error_message(exc),
        )
    )


def unsupported_attachment_format(exc) -> bool:
    """True if ``exc`` is an HTTP 500 that says an attachment's format is unsupported.

    Such an error is about the request, not the server: the retry layer must
    not resend it unchanged, so the client's media recovery can replace the
    attachment with a note at once.
    """
    status = getattr(exc, "_connect_http_status", getattr(exc, "status_code", None))
    return status == 500 and any(_unsupported_format(exc, kind) for kind in _FORMAT_NOUN)


def explicitly_rejected(exc, kind):
    """Return True if ``exc`` is the provider refusing this kind of media.

    Requires HTTP 400/415/422 and an error message that is specifically about
    the media (or the content part used to send it), for example "image input
    is not supported" or "Unsupported content part type: 'file'". A generic
    400, or an error about another parameter (model, max_tokens, ...), does
    not count. A per-request image limit (``image_limit``) is not a rejection
    of one image: the runtime collapses the oldest images instead.
    """
    status = getattr(exc, "_connect_http_status", getattr(exc, "status_code", None))
    if status not in {400, 415, 422} and not (status == 500 and _unsupported_format(exc, kind)):
        return False
    if image_limit(exc) is not None:
        return False
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    text = _error_message(exc)
    if isinstance(param, str) and param.split(".")[0] in {
        "model",
        "include",
        "store",
        "max_tokens",
        "max_output_tokens",
        "api_key",
    }:
        return False
    # Match a statement about the media/input itself. A media-looking model
    # name plus an unrelated 'invalid' parameter is not explicit rejection.
    names = {
        "image": r"images?|vision inputs?",
        "audio": r"audio|wav",
        "video": r"videos?",
        "pdf": r"pdfs?|documents?|files?",
    }.get(kind, r"files?")
    # "media" names whichever attachment the request carries (one gateway
    # says "Media could not be processed ... Failed to load image").
    noun = rf"(?:{names}|media)(?: (?:input|content|format|type|data|url|attachment|processing|file)s?)?"
    # Text-only models: the whole request is refused because it has media.
    if re.search(r"\bnot a multimodal model\b|\bdoes(?: not|n't) support multimodal", text):
        return True
    # "Unsupported content part type: 'file'" names the wire part for this format.
    part = re.search(
        r"unsupported (?:chat )?(?:content(?: part)?|media) type:?\s*[\\'\"]*([a-z_]+)", text
    )
    parts = {
        "image": {"image_url", "input_image"},
        "audio": {"input_audio", "audio_url"},
        "video": {"video_url"},
        "pdf": {"file", "input_file", "document"},
    }.get(kind, set())
    if part and part.group(1) in parts:
        return True
    # A strict server whose message schema has no variant for this media part.
    if "did not match any variant of untagged enum" in text and "messagecontent" in text:
        return True
    return bool(
        re.search(
            rf"\b{noun}\b (?:is |are |was |were )?(?:not supported|unsupported|not allowed|rejected|invalid"
            r"|corrupt(?:ed)?|truncated|malformed)\b",
            text,
        )
        or re.search(rf"\b(?:unsupported|invalid|rejected) {noun}\b(?![-\w])", text)
        or re.search(
            r"\b(?:does not|doesn't|cannot|can't|could not|couldn't|unable to|failed to) "
            rf"(?:support|accept|process|decode|load|read) {noun}\b",
            text,
        )
        # OpenAI: "The image data you provided does not represent a valid image."
        or re.search(rf"\bnot (?:\w+ ){{0,2}}an? valid {noun}\b", text)
        # Bedrock: "Could not process image"; gateways: "Media could not be processed".
        or re.search(
            rf"\b{noun} (?:could not|couldn't|cannot|can't) be (?:processed|decoded|loaded|read)\b",
            text,
        )
    )


def _occurrences(messages, enabled: tuple[MediaInput, ...]):
    """Yield (message index, block index, kind, key) for each attachment a request sends.

    A message that is itself an attachment item has block index None.
    Attachments the filter replaces anyway (a kind not in ``enabled``, an
    image in a format models do not accept) are skipped, so a rejection can
    never be blamed on one of them. ``key`` names one attachment occurrence:
    its kind, its block's content and how many identical blocks came before it
    in this request.
    """
    seen: dict[tuple[str, str], int] = {}

    def occurrence(index, position, block):
        kind = attachment_kind(block)
        if kind is None or kind not in enabled or unsendable_image_type(block) is not None:
            return None
        payload = {k: v for k, v in block.items() if k != "cache_control"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=repr).encode()
        ).hexdigest()
        count = seen.get((kind, digest), 0)
        seen[(kind, digest)] = count + 1
        return index, position, kind, (kind, digest, count)

    for index, message in enumerate(messages):
        if isinstance(message, (LLMResponse, CacheBoundary)) or not isinstance(message, Mapping):
            continue
        if attachment_kind(message) is not None:
            found = occurrence(index, None, message)
            if found:
                yield found
            continue
        slot = "output" if message.get("type") == "function_call_output" else "content"
        content = message.get(slot)
        if not isinstance(content, (list, tuple)):
            continue
        for position, block in enumerate(content):
            found = occurrence(index, position, block) if isinstance(block, Mapping) else None
            if found:
                yield found


def media_occurrences(messages, enabled: tuple[MediaInput, ...]):
    """The attachment occurrences a request sends, oldest first, as (kind, key)."""
    return [(kind, key) for *_, kind, key in _occurrences(messages, enabled)]


def _replacement(kind, block, rejected: bool, enabled) -> str | None:
    """The notice that replaces an attachment block on the wire, or None to send it."""
    if kind not in enabled:
        return removed_media_notice(kind)
    bad_type = unsendable_image_type(block) if kind == "image" else None
    if bad_type is not None:
        return unsendable_image_notice(bad_type)
    return rejected_media_notice(kind) if rejected else None


def filter_media_messages(
    messages, enabled: tuple[MediaInput, ...], excluded: Collection[Any] = frozenset()
):
    """Return a copy of ``messages`` with media that is not ``enabled`` replaced.

    Each disabled attachment is replaced by a short text notice (see
    ``removed_media_notice``) so the model knows something was removed. Three
    places are checked: a message that is itself an attachment item (Responses
    input allows these at the top level), an assistant ``audio`` field, and the
    blocks of a ``content`` list (``output`` for ``function_call_output`` items).
    ``excluded`` holds keys from ``media_occurrences`` that the current
    connection rejected; each such attachment becomes a ``rejected_media_notice``.
    An enabled inline image in a format models do not accept (such as SVG)
    becomes an ``unsendable_image_notice``; an accepted one is sent with its
    MIME type in canonical form (``image/jpg`` as ``image/jpeg``).
    Tool arguments, reasoning and vendor extensions are not inspected.
    ``LLMResponse`` and ``CacheBoundary`` objects are passed through unchanged;
    their native content is checked later by ``check_replayed_native_media``. The input
    list and its messages are not modified.
    """
    rejected = {
        (index, position)
        for index, position, _, key in _occurrences(messages, enabled)
        if key in excluded
    }
    result = []
    for message_index, message in enumerate(messages):
        if isinstance(message, (LLMResponse, CacheBoundary)) or not isinstance(message, Mapping):
            result.append(message)
            continue
        item = dict(message)
        kind = attachment_kind(item)
        if kind is not None:
            # The whole message is an attachment item, so there is no content
            # list to filter: replace the item with a user notice.
            notice = _replacement(kind, item, (message_index, None) in rejected, enabled)
            result.append(
                {"role": "user", "content": notice}
                if notice is not None
                else _sendable_image_block(item)
                if kind == "image"
                else item
            )
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
                notice = (
                    None
                    if kind is None
                    else _replacement(kind, block, (message_index, position) in rejected, enabled)
                )
                if notice is not None:
                    text_type = "input_text" if slot == "output" else "text"
                    blocks.append({"type": text_type, "text": notice})
                else:
                    blocks.append(_sendable_image_block(block) if kind == "image" else block)
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
    kinds: list[str | None] = ["audio"] if native.get("audio") is not None else []
    if isinstance(content, (list, tuple)):
        kinds.extend(attachment_kind(block) for block in content if isinstance(block, Mapping))
    if any(kind is not None and kind not in enabled for kind in kinds):
        raise ReasoningReplayError(
            "Cannot remove disabled media from opaque assistant state; "
            "supply a portable assistant turn instead."
        )
    from nooa.media_policy import reject_unsupported_media

    reject_unsupported_media([native], interface)
