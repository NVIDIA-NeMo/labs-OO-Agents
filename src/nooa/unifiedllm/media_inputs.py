# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

import hashlib
import json
import re
from collections.abc import Mapping
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

# The words that name each kind of attachment in a provider's error message.
_KIND_NAMES = {
    "image": r"images?|vision inputs?",
    "audio": r"audio|wav",
    "video": r"videos?",
    "pdf": r"pdfs?|documents?|files?",
}

# The wire part (or its field) that carries each kind; an error ``param`` such
# as ``messages[0].content[1].input_audio.format`` names it.
_KIND_PARTS = {
    "image": {"image_url", "input_image"},
    "audio": {"input_audio", "audio_url"},
    "video": {"video_url"},
    "pdf": {"file", "input_file", "document"},
}

MEDIA_KINDS: tuple[str, ...] = ("image", "audio", "video", "pdf")


def _error_message(exc) -> str:
    """The provider's error message in lower case: ``body.error.message`` if set, else ``str(exc)``."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", str(exc))).lower()
    return str(exc).lower()


def _error_texts(exc) -> str:
    """``str(exc)`` and the body's message, lower case, for pattern matching."""
    texts = [str(exc).lower()]
    message = _error_message(exc)
    if message not in texts[0]:
        texts.append(message)
    return "\n".join(texts)


def _error_param(exc) -> str | None:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    return param if isinstance(param, str) else None


def _http_status(exc) -> int | None:
    status = getattr(exc, "_connect_http_status", getattr(exc, "status_code", None))
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _unsupported_format(exc, kind) -> bool:
    """True if ``exc`` says the attachment's format is unsupported.

    The NVIDIA Inference Hub (a LiteLLM proxy) returns this as HTTP 500:
    "Unsupported image format: svg+xml. Supported formats: [...]". The error
    is about the request, so retrying it unchanged fails every time. The
    phrase must name a format value (after a colon, ``=`` or a quote) or be
    followed by the formats that are supported, so "unsupported image format
    conversion plugin" does not count.
    """
    return _unsupported_format_text(_error_texts(exc), kind)


def _unsupported_format_text(text: str, kind) -> bool:
    noun = _FORMAT_NOUN.get(kind, "file")
    return bool(
        re.search(
            rf"\bunsupported {noun} (?:format|type)s?"
            r"(?:\s*[:=]|\s+['\"`]|\b[^.]{0,80}\b(?:supported|allowed|accepted) (?:formats|types)\b)",
            text,
        )
    )


# The request as a whole is too long: never one attachment's fault.
_CONTEXT_LENGTH_RE = re.compile(
    r"context[ _](?:length|window)|maximum context|too many (?:input )?tokens|prompt is too long"
    r"|tokens? (?:exceed|limit)|reduce the length of the messages"
)
# A 500 with these words is the server being busy, not the attachment.
_TRANSIENT_RE = re.compile(
    r"\btemporar(?:y|ily)\b|\boverloaded\b|\btry again later\b|\bunavailable\b|\brate.?limit"
    r"|\bcapacity\b"
)


def _size_rejected(text: str, kind) -> bool:
    """The attachment is too big: per-item byte, pixel or page limits.

    Wording examples from Anthropic's documentation and the #597 reviews (not
    checked against a live provider): "image exceeds 5 MB maximum", "image
    dimensions exceed max allowed size: 8000 pixels", "A maximum of 100 PDF
    pages may be provided". Request-wide limits (context length, too many
    images) are handled elsewhere and are not matched here.
    """
    names = _KIND_NAMES.get(kind, r"files?")
    if re.search(
        rf"\b(?:{names})(?: (?:size|dimensions?|file|data|payload|resolution|width|height))?\b"
        r".{0,60}?\b(?:exceeds?|exceeded|exceeding|too large|too big|larger than)\b",
        text,
    ):
        return True
    return kind == "pdf" and bool(
        re.search(
            r"\bpdf (?:pages?|page count)\b.{0,40}?\b(?:exceed|maximum|limit|too many)"
            r"|\bmaximum of \d+ pdf pages\b|\btoo many (?:pdf )?pages\b",
            text,
        )
    )


def _media_type_enum_kind(text: str) -> str | None:
    """Anthropic's schema error for a ``media_type`` it does not take.

    "messages.0.content.1.image.source.base64.media_type: Input should be
    'image/jpeg', 'image/png', 'image/gif' or 'image/webp'" (wording from the
    #597 issue; not captured from a live provider).
    """
    match = re.search(
        r"\bmedia_type\b.{0,20}?input should be\s*['\"]?(image|application|audio|video)/", text
    )
    if not match:
        return None
    return {"application": "pdf"}.get(match.group(1), match.group(1))


def _mime_type_kind(text: str) -> str | None:
    """'Unsupported MIME type: image/gif' (Gemini-style wording from the #597 review, unverified)."""
    match = re.search(
        r"\bunsupported mime[ _-]?type\s*:?\s*['\"`]?(image|audio|video|application/pdf)\b", text
    )
    if not match:
        return None
    return "pdf" if match.group(1) == "application/pdf" else match.group(1)


def _processing_failed(text: str, kind) -> bool:
    """The server could not read the attachment's bytes ("Could not process image")."""
    names = _KIND_NAMES.get(kind, r"files?")
    noun = rf"(?:{names}|media)(?: (?:input|content|format|type|data|url|attachment|file)s?)?"
    return bool(
        re.search(
            r"\b(?:does not|doesn't|cannot|can't|could not|couldn't|unable to|failed to) "
            rf"(?:process|decode|load|read|parse) (?:the )?{noun}\b",
            text,
        )
        # OpenAI: "The image data you provided does not represent a valid image."
        or re.search(rf"\bnot (?:\w+ ){{0,2}}an? valid {noun}\b", text)
        # Anthropic (docs/review wording): "The PDF specified was not valid."
        or re.search(rf"\b{noun}(?: \w+)? (?:is|was) not valid\b", text)
        # Bedrock: "Could not process image"; gateways: "Media could not be processed".
        or re.search(
            rf"\b{noun} (?:could not|couldn't|cannot|can't) be (?:processed|decoded|loaded|read|parsed)\b",
            text,
        )
        or re.search(rf"\b(?:corrupt(?:ed)?|malformed|truncated) {noun}\b", text)
        or re.search(rf"\b{noun}\b (?:is |was )?(?:corrupt(?:ed)?|truncated|malformed)\b", text)
        or (
            kind == "pdf"
            and re.search(
                r"\b(?:password[- ]protected|encrypted) (?:pdf|document|file)s?\b"
                r"|\b(?:pdf|document|file)s?\b.{0,40}?\b(?:password[- ]protected|encrypted)\b",
                text,
            )
        )
    )


def _fetch_failed(text: str) -> bool:
    """A remote image URL could not be downloaded (dead link, expired presigned URL).

    LiteLLM downloads image URLs itself for Bedrock and Anthropic routes and
    raises ``ImageFetchError`` ("Error: Unable to fetch image from URL. Status
    code: 404, url=..."). OpenAI's server-side fetch says "Error while
    downloading https://..." or "Timeout while downloading ..." (wording from
    the #597 review, not captured live). The image stays in history, so every
    later request would fail the same way.
    """
    return bool(
        re.search(
            r"\bunable to (?:fetch|download|retrieve) (?:the )?image\b"
            r"|\b(?:error|timeout|timed out) while downloading\b"
            r"|\bfailed to (?:fetch|download) (?:the )?image\b",
            text,
        )
    )


def _is_image_fetch_error(exc) -> bool:
    """``litellm.ImageFetchError`` (a 400 ``BadRequestError``), without importing LiteLLM."""
    return any(cls.__name__ == "ImageFetchError" for cls in type(exc).__mro__)


def _param_kind(param: str | None) -> str | None:
    """The kind of the wire part a structured error ``param`` points at, if any."""
    if not param:
        return None
    fields = set(re.split(r"[.\[\]]+", param.lower()))
    for kind in MEDIA_KINDS:
        if fields & (_KIND_PARTS[kind] - {"file", "document"}) or (
            kind == "pdf" and "file_data" in fields
        ):
            return kind
    return None


def _strict_media_statement(text: str, kind) -> bool:
    """Wording a provider uses only when it refuses this kind of media (HTTP 4xx)."""
    names = _KIND_NAMES.get(kind, r"files?")
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
    if part and part.group(1) in _KIND_PARTS.get(kind, set()):
        return True
    # A strict server whose message schema has no variant for this media part.
    if "did not match any variant of untagged enum" in text and "messagecontent" in text:
        return True
    return bool(
        re.search(
            rf"\b{noun}\b (?:is |are |was |were )?(?:not supported|unsupported|not allowed|rejected|invalid)\b",
            text,
        )
        or re.search(rf"\b(?:unsupported|invalid|rejected) {noun}\b(?![-\w])", text)
        or re.search(
            r"\b(?:does not|doesn't|cannot|can't|could not|couldn't|unable to|failed to) "
            rf"(?:support|accept) {noun}\b",
            text,
        )
        # OpenAI audio (wording from the #597 review): "audio format mpeg is not supported".
        or (
            kind == "audio"
            and re.search(r"\baudio format\b.{0,40}?\b(?:not supported|unsupported)\b", text)
        )
    )


def rejected_media_kinds(exc) -> frozenset[str]:
    """The media kinds ``exc`` says the provider refused; empty if it is not a media rejection.

    The one predicate both the endpoint retry layer (``retry.py``: never resend
    a refused attachment unchanged) and the client's media recovery (replace it
    with a note) use. It is narrow and depends on the HTTP status:

    - 400/415/422: any wording that is specifically about the media (or the
      content part that carries it), its size, its format, or a structured
      ``param`` that points at a media part.
    - 413: only when the message says the attachment itself is too large.
    - 500: only wording that says the server could not read this attachment
      (unsupported format, "Could not process image", a size limit, a failed
      image download) and nothing that sounds transient ("temporarily", "overloaded").
      Other 5xx statuses (502/503/504: gateways, outages) never count.
    - No status: only a local ``ValueError`` (LiteLLM's Bedrock adapter raises
      ``ValueError("Unsupported image format: ...")`` before sending).

    ``litellm.ImageFetchError`` (a remote image URL LiteLLM could not
    download) is always an image rejection. A generic error, one about another parameter (model, max_tokens, ...),
    a per-request image limit (``image_limit``) or a context-length error does
    not count.
    """
    status = _http_status(exc)
    text = _error_texts(exc)
    if image_limit(exc) is not None or _CONTEXT_LENGTH_RE.search(text):
        return frozenset()
    if _is_image_fetch_error(exc):
        return frozenset({"image"})
    param = _error_param(exc)
    if param is not None and param.split(".")[0] in {
        "model",
        "include",
        "store",
        "max_tokens",
        "max_output_tokens",
        "api_key",
        "tools",
        "tool_choice",
    }:
        return frozenset()
    if status in {400, 415, 422}:
        strict, server = True, True
    elif status == 413:
        strict, server = False, False
    elif status == 500 and not _TRANSIENT_RE.search(text):
        strict, server = False, True
    elif status is None and isinstance(exc, ValueError):
        return frozenset(k for k in MEDIA_KINDS if _unsupported_format_text(text, k))
    else:
        return frozenset()
    kinds = set()
    if status != 413:
        for found in (_media_type_enum_kind(text), _mime_type_kind(text)):
            if found:
                kinds.add(found)
        if strict:
            # A structured ``param`` or a field path in the message, e.g. OpenAI's
            # "Invalid value: 'mpeg'. Supported values are: 'wav' and 'mp3'." with
            # param ``messages[0].content[1].input_audio.format``.
            for path in (
                param,
                *re.findall(
                    r"[\w\[\].]*\b(?:input_audio|audio_url|image_url|input_image|video_url|input_file|file_data)\b[\w\[\].]*",
                    text,
                ),
            ):
                if found := _param_kind(path):
                    kinds.add(found)
    for kind in MEDIA_KINDS:
        if (
            _size_rejected(text, kind)
            or (server and (_unsupported_format_text(text, kind) or _processing_failed(text, kind)))
            or (server and kind == "image" and _fetch_failed(text))
            or (strict and _strict_media_statement(text, kind))
        ):
            kinds.add(kind)
    return frozenset(kinds)


def is_media_rejection(exc) -> bool:
    """True if ``exc`` is a provider refusing an attachment (see ``rejected_media_kinds``).

    Such an error is about the request, not the server: the retry layer must
    not resend it unchanged, so the client's media recovery can replace the
    attachment with a note at once.
    """
    return bool(rejected_media_kinds(exc))


def unsupported_attachment_format(exc) -> bool:
    """True if ``exc`` is an HTTP 5xx media rejection (kept for callers of the #595 API)."""
    status = _http_status(exc)
    return status is not None and status >= 500 and is_media_rejection(exc)


def explicitly_rejected(exc, kind):
    """Return True if ``exc`` is the provider refusing this kind of media.

    ``kind in rejected_media_kinds(exc)``; see there for the status rules. A
    per-request image limit (``image_limit``) is not a rejection of one image:
    the runtime collapses the oldest images instead.
    """
    return kind in rejected_media_kinds(exc)


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


def filter_media_messages(messages, enabled: tuple[MediaInput, ...], excluded=frozenset()):
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
