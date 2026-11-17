# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, route-local media permissions; no discovery or payload inspection."""

import hashlib
import json
import re
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.media_inputs import (
    INPUT_AUDIO_FORMATS,
    MediaInput,
    _mime_hints,
    attachment_kind,
    audio_format_name,
)
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
# yet), and not universal: Gemini does not take GIF but does take HEIC/HEIF,
# and OpenAI refuses animated GIF. A provider that refuses one of these
# still triggers the media recovery in the client. Other types,
# such as SVG, BMP or TIFF, are refused by some providers; kept in history,
# such an image would fail every later request.
SENDABLE_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_IMAGE_TYPE_ALIASES = {"image/jpg": "image/jpeg"}


def _canonical_image_type(media_type: str) -> str:
    """``media_type`` in lower case without parameters, with ``image/jpg`` as ``image/jpeg``."""
    media_type = media_type.split(";", 1)[0].strip().lower()
    return _IMAGE_TYPE_ALIASES.get(media_type, media_type)


# Remote image types that no common model API accepts. A remote URL with one
# of these (by its file extension or an explicit type) gets the same note as
# an inline image; any other or unknown remote type is sent unchanged, since
# the provider reads the real type when it downloads the image.
CLEARLY_UNSENDABLE_IMAGE_TYPES = frozenset(
    {"image/svg+xml", "image/bmp", "image/x-ms-bmp", "image/tiff", "image/heic", "image/heif"}
)
_REMOTE_EXTENSION_TYPES = {
    "svg": "image/svg+xml",
    "svgz": "image/svg+xml",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "heic": "image/heic",
    "heif": "image/heif",
}


def _remote_url(block: Mapping[str, Any]) -> str | None:
    """The http(s) URL a remote image block points at, else None."""
    candidates = [block.get("image_url"), block.get("url")]
    for key in ("image_url", "source"):
        if isinstance(block.get(key), Mapping):
            candidates.extend([block[key].get("url"), block[key].get("image_url")])
    for value in candidates:
        if isinstance(value, str) and value[:5].lower() != "data:" and "://" in value:
            return value
    return None


def remote_url_image_type(url: str) -> str | None:
    """The image type a URL's file extension names (``.svg`` -> ``image/svg+xml``), else None."""
    from urllib.parse import urlsplit

    path = urlsplit(url).path
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return None
    return _REMOTE_EXTENSION_TYPES.get(name.rsplit(".", 1)[-1].lower())


def unsendable_image_type(block: Mapping[str, Any]) -> str | None:
    """The MIME type of an image block models do not accept, else None.

    Inline images (a data URL or base64 ``source``): every MIME hint of the
    block is checked (data URL, ``format``, ``media_type``, ``mime_type``; see
    ``nooa.media_inputs._mime_hints``), after lower-casing and dropping
    parameters. One hint outside ``SENDABLE_IMAGE_TYPES`` makes the image
    unsendable; two accepted hints that disagree do not.

    Remote images (an http(s) URL): only a type that is clearly unsendable
    (``CLEARLY_UNSENDABLE_IMAGE_TYPES``: SVG, BMP, TIFF, HEIC/HEIF), named by
    an explicit hint or by the URL's file extension, counts. A remote URL with
    a sendable or unknown type is sent unchanged: the provider reads the real
    type from the server.
    """
    if attachment_kind(block) != "image":
        return None
    hints = {_canonical_image_type(hint) for hint in _mime_hints(block)}
    if _is_inline(block):
        refused = sorted(hints - SENDABLE_IMAGE_TYPES)
        return refused[0] if refused else None
    url = _remote_url(block)
    if url is not None and (by_extension := remote_url_image_type(url)):
        hints.add(by_extension)
    refused = sorted(hints & CLEARLY_UNSENDABLE_IMAGE_TYPES)
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


def unsendable_audio(block: Mapping[str, Any]) -> str | None:
    """Why an ``input_audio`` block cannot be sent, else None.

    ``"url"``: its ``data`` is a URL, not base64 (``Audio.from_url``); a URL
    must never be sent as base64 audio. Otherwise its format name, if it is
    not one ``input_audio`` accepts (``wav``, ``mp3``; aliases such as
    ``mpeg`` and ``x-wav`` count as those). Other audio blocks are not checked.
    """
    if block.get("type") != "input_audio" or not isinstance(block.get("input_audio"), Mapping):
        return None
    node = block["input_audio"]
    data = node.get("data")
    if isinstance(data, str) and re.match(r"\s*[a-z][a-z0-9+.-]*://", data, re.I):
        return "url"
    fmt = node.get("format")
    if isinstance(fmt, str) and fmt and audio_format_name(fmt) not in INPUT_AUDIO_FORMATS:
        return audio_format_name(fmt)
    return None


def _sendable_audio_block(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """``block`` with its ``input_audio`` format as the provider name (``mpeg`` -> ``mp3``)."""
    node = block.get("input_audio")
    if block.get("type") != "input_audio" or not isinstance(node, Mapping):
        return block
    fmt = node.get("format")
    if not isinstance(fmt, str) or audio_format_name(fmt) == fmt:
        return block
    return {**block, "input_audio": {**node, "format": audio_format_name(fmt)}}


def unsendable_audio_notice(reason: str) -> str:
    """The text the model sees in place of audio it cannot be sent."""
    if reason == "url":
        return (
            "[An audio clip was removed here: audio given as a URL cannot be sent to a model. "
            "Load it with Audio.from_file or Audio.from_bytes and show it again.]"
        )
    return (
        f"[An audio clip was removed here: {reason} audio cannot be sent to a model. "
        "Only WAV and MP3 are accepted; convert it and show it again.]"
    )


def unsendable_image_notice(media_type: str) -> str:
    """The text the model sees in place of an image in a format models do not accept."""
    return (
        f"[An image was removed here: {media_type} images cannot be sent to a model. "
        "Convert it to PNG, JPEG, GIF or WebP (for example render an SVG to PNG) "
        "and show it again.]"
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


def _kind_names(kind, narrow: bool = False) -> str:
    """The ``_KIND_NAMES`` pattern of ``kind``; ``narrow`` (HTTP 5xx): a PDF only as "PDF".

    A server error about "the file" or "the document" is usually about the
    server's own files (model weights, caches), not an attachment.
    """
    if narrow and kind == "pdf":
        return r"pdfs?"
    return _KIND_NAMES.get(kind, r"files?")


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
# A server or network problem, not the attachment, at any status: such an
# error is never a media rejection, even when it names one ("Could not
# process image: upstream request timed out"). Checked before any media rule.
_TRANSIENT_RE = re.compile(
    r"\btime ?outs?\b|\btimed out\b|\bconnection (?:reset|refused|aborted|closed)\b"
    # "try again later" / "in a few seconds", but not OpenAI's "The image data
    # you provided does not represent a valid image. Please check your input
    # and try again.", which is a real rejection.
    r"|\bunavailable\b|\boverloaded\b|\btemporar(?:y|ily)\b|\brate.?limit"
    r"|\btry again (?:later|shortly|soon|in\b)|\b(?:retry|try) (?:the request )?later\b"
    r"|\bcapacity\b"
)


def _size_rejected(text: str, kind, narrow: bool = False) -> bool:
    """The attachment is too big: per-item byte, pixel or page limits.

    Wording examples from Anthropic's documentation and the #597 reviews (not
    checked against a live provider): "image exceeds 5 MB maximum", "image
    dimensions exceed max allowed size: 8000 pixels", "A maximum of 100 PDF
    pages may be provided". Request-wide limits (context length, too many
    images) are handled elsewhere and are not matched here.
    """
    names = _kind_names(kind, narrow)
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


def _processing_failed(text: str, kind, narrow: bool = False) -> bool:
    """The server could not read the attachment's bytes ("Could not process image")."""
    names = _kind_names(kind, narrow)
    docs = "pdf" if narrow else "(?:pdf|document|file)"
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
                rf"\b(?:password[- ]protected|encrypted) {docs}s?\b"
                rf"|\b{docs}s?\b.{{0,40}}?\b(?:password[- ]protected|encrypted)\b",
                text,
            )
        )
    )


def _fetch_failed(text: str) -> bool:
    """A remote image URL could not be downloaded (dead link, expired presigned URL).

    LiteLLM downloads image URLs itself for Bedrock and Anthropic routes and
    raises ``ImageFetchError`` ("Error: Unable to fetch image from URL. Status
    code: 404, url=..."). OpenAI's server-side fetch says "Error while
    downloading https://..." (wording from the #597 review, not captured
    live). The image stays in history, so every later request would fail the
    same way. A download error must name an image or a URL: "Error while
    downloading model weights" is the server's own problem. A timeout is
    transient (``_TRANSIENT_RE``), never a rejection.
    """
    return bool(
        re.search(
            r"\bunable to (?:fetch|download|retrieve) (?:the )?image\b"
            r"|\berror while downloading (?:the )?(?:image\b|https?://|url\b)"
            r"|\bfailed to (?:fetch|download) (?:the )?image\b",
            text,
        )
    )


def _is_image_fetch_error(exc) -> bool:
    """``litellm.ImageFetchError`` (a 400 ``BadRequestError``), without importing LiteLLM."""
    return any(cls.__name__ == "ImageFetchError" for cls in type(exc).__mro__)


def _param_root(param: str) -> str:
    """The top-level request field a ``param`` path names: ``tools[0].function`` -> ``tools``."""
    return re.split(r"[.\[\]]", param.strip().lower(), maxsplit=1)[0]


# Request fields that hold the conversation (and so the attachments).
_CONTENT_ROOTS = frozenset({"messages", "input", "contents", "content", "system", "prompt"})
# A media field inside a content path of the message text, such as
# "messages[0].content[1].input_audio.format"; a bare "image_url" (a function
# name, say) does not count.
_MEDIA_FIELD_PATH_RE = re.compile(
    r"\b(?:messages|input|contents?)\b[\w\[\].]*?[.\]]"
    r"(?:input_audio|audio_url|image_url|input_image|video_url|input_file|file_data)\b[\w\[\].]*"
)


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
      image download). A PDF counts only when the message says "PDF": a 500
      about "the file" or "the document" is usually the server's own file.
      Other 5xx statuses (502/503/504: gateways, outages) never count.

    Wording that sounds transient (a timeout, a connection reset, "unavailable",
    "overloaded", "temporarily", "try again"; ``_TRANSIENT_RE``) never counts,
    at any status: a later identical request may succeed. A structured
    ``param`` outside the conversation (``tools[0].function.parameters``,
    ``max_tokens``) wins over media words in the text, and a media field name
    in the text counts only inside a content path
    (``messages[0].content[1].image_url``), not as a function name.
    - No status: only a local ``ValueError`` (LiteLLM's Bedrock adapter raises
      ``ValueError("Unsupported image format: ...")`` before sending).

    ``litellm.ImageFetchError`` (a remote image URL LiteLLM could not
    download) is always an image rejection. A generic error, one about another parameter (model, max_tokens, ...),
    a per-request image limit (``image_limit``) or a context-length error does
    not count.
    """
    status = _http_status(exc)
    text = _error_texts(exc)
    if _TRANSIENT_RE.search(text):
        return frozenset()  # a server or network problem, even if it names an image
    if image_limit(exc) is not None or _CONTEXT_LENGTH_RE.search(text):
        return frozenset()
    if _is_image_fetch_error(exc):
        return frozenset({"image"})
    param = _error_param(exc)
    if param and _param_root(param) not in _CONTENT_ROOTS and _param_kind(param) is None:
        # The provider named another parameter (tools[0].function.parameters,
        # max_tokens, model, ...): that wins over media words in the text.
        return frozenset()
    if status in {400, 415, 422}:
        strict, server = True, True
    elif status == 413:
        strict, server = False, False
    elif status == 500:
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
            for path in (param, *_MEDIA_FIELD_PATH_RE.findall(text)):
                if found := _param_kind(path):
                    kinds.add(found)
    narrow = status is not None and status >= 500  # a PDF only as "PDF" (_kind_names)
    for kind in MEDIA_KINDS:
        if (
            _size_rejected(text, kind, narrow)
            or (
                server
                and (
                    # Names the refused format value: specific even as a 500.
                    _unsupported_format_text(text, kind) or _processing_failed(text, kind, narrow)
                )
            )
            or (server and kind == "image" and _fetch_failed(text))
            or (strict and _strict_media_statement(text, kind))
        ):
            kinds.add(kind)
    return frozenset(kinds)


# The media kinds the request being sent carries after filtering, set by the
# client for each request it prepares. The retry layer stops retrying a
# media-worded 5xx only if the request really sent that kind of media.
sent_media_kinds: ContextVar[frozenset[str]] = ContextVar("sent_media_kinds", default=frozenset())


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
        if kind is None or kind not in enabled or _unsendable(kind, block) is not None:
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
            if not isinstance(block, Mapping):
                continue
            nested = _tool_result_blocks(block)
            if nested is not None:
                # Anthropic's user ``tool_result`` block carries its own content list.
                for inner, nested_block in enumerate(nested):
                    found = (
                        occurrence(index, (position, inner), nested_block)
                        if isinstance(nested_block, Mapping)
                        else None
                    )
                    if found:
                        yield found
                continue
            found = occurrence(index, position, block)
            if found:
                yield found


def _tool_result_blocks(block: Mapping[str, Any]):
    """The content list of an Anthropic ``tool_result`` block, else None.

    Documented shape: ``{"type": "tool_result", "tool_use_id": ..., "content":
    [{"type": "text", ...}, {"type": "image", "source": {...}}]}``. Only this
    shape is looked into; tool arguments are never scanned.
    """
    if block.get("type") != "tool_result":
        return None
    content = block.get("content")
    return content if isinstance(content, (list, tuple)) else None


def unsendable_occurrences(messages, enabled: tuple[MediaInput, ...]):
    """The enabled attachments the filter replaces before sending, as (reason, key).

    ``reason`` is the refused MIME type of an image (``image/svg+xml``) or the
    audio problem (``url``, a format name). ``key`` identifies the occurrence
    like ``media_occurrences`` keys do, so a caller can count each one once
    although the filter runs on every request.
    """
    found = []
    seen: dict[tuple[str, str], int] = {}
    for message in messages:
        if isinstance(message, (LLMResponse, CacheBoundary)) or not isinstance(message, Mapping):
            continue
        blocks = [message] if attachment_kind(message) is not None else []
        if not blocks:
            slot = "output" if message.get("type") == "function_call_output" else "content"
            content = message.get(slot)
            for block in content if isinstance(content, (list, tuple)) else ():
                if isinstance(block, Mapping):
                    nested = _tool_result_blocks(block)
                    blocks.extend(nested if nested is not None else [block])
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            kind = attachment_kind(block)
            if kind not in enabled:
                continue
            reason = (
                unsendable_image_type(block)
                if kind == "image"
                else unsendable_audio(block)
                if kind == "audio"
                else None
            )
            if reason is None:
                continue
            digest = hashlib.sha256(
                json.dumps(dict(block), sort_keys=True, default=repr).encode()
            ).hexdigest()
            count = seen.get((kind, digest), 0)
            seen[(kind, digest)] = count + 1
            found.append((f"{kind}:{reason}", (kind, digest, count)))
    return found


def media_occurrences(messages, enabled: tuple[MediaInput, ...]):
    """The attachment occurrences a request sends, oldest first, as (kind, key)."""
    return [(kind, key) for *_, kind, key in _occurrences(messages, enabled)]


def unknown_type_notice(part_kind: str, media_type: str) -> str:
    """The text the model sees in place of an image (audio, video) whose type is unknown."""
    subject = _NOTICE_SUBJECT[part_kind][0]
    described = f" ({media_type})" if media_type else ""
    advice = " Save it as PNG, JPEG, GIF or WebP and show it again." if part_kind == "image" else ""
    return f"[{subject} of unknown type{described} was removed here.{advice}]"


def _part_kind(block: Mapping[str, Any]) -> str | None:
    """The kind the block's wire part carries (``image_url`` -> image), ignoring MIME hints."""
    from nooa.media_inputs import _KINDS

    kind = _KINDS.get(block.get("type"))
    return kind if kind in _NOTICE_SUBJECT else None


def _replacement(kind, block, rejected: bool, enabled) -> str | None:
    """The notice that replaces an attachment block on the wire, or None to send it."""
    if kind not in enabled:
        part = _part_kind(block)
        if kind == "file" and part is not None:
            # An image part whose MIME type is not an image type (for example
            # application/octet-stream): say so instead of "Only PDF files".
            hints = sorted(_mime_hints(block))
            return unknown_type_notice(part, hints[0] if hints else "")
        return removed_media_notice(kind)
    notice = _unsendable(kind, block)
    if notice is not None:
        return notice
    return rejected_media_notice(kind) if rejected else None


def _unsendable(kind, block) -> str | None:
    """The notice for an enabled attachment no model can be sent in this form, else None."""
    if kind == "image" and (bad_type := unsendable_image_type(block)) is not None:
        return unsendable_image_notice(bad_type)
    if kind == "audio" and (reason := unsendable_audio(block)) is not None:
        return unsendable_audio_notice(reason)
    return None


def _sendable_block(kind, block):
    """An accepted attachment block in canonical wire form (MIME type, audio format name)."""
    if kind == "image":
        return _sendable_image_block(block)
    if kind == "audio":
        return _sendable_audio_block(block)
    return block


def filter_media_messages(messages, enabled: tuple[MediaInput, ...], excluded=frozenset()):
    """Return a copy of ``messages`` with media that is not ``enabled`` replaced.

    Each disabled attachment is replaced by a short text notice (see
    ``removed_media_notice``) so the model knows something was removed. Three
    places are checked: a message that is itself an attachment item (Responses
    input allows these at the top level), an assistant ``audio`` field, and the
    blocks of a ``content`` list (``output`` for ``function_call_output`` items),
    including the content list of an Anthropic ``tool_result`` block.
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
                else _sendable_block(kind, item)
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
            text_type = "input_text" if slot == "output" else "text"

            def filtered(block, where, message_index=message_index, text_type=text_type):
                kind = attachment_kind(block) if isinstance(block, Mapping) else None
                notice = (
                    None
                    if kind is None
                    else _replacement(kind, block, (message_index, where) in rejected, enabled)
                )
                if notice is not None:
                    return {"type": text_type, "text": notice}
                return _sendable_block(kind, block) if kind is not None else block

            for position, block in enumerate(content):
                nested = _tool_result_blocks(block) if isinstance(block, Mapping) else None
                if nested is not None:
                    inner_blocks = [
                        filtered(inner, (position, index)) for index, inner in enumerate(nested)
                    ]
                    changed = any(a is not b for a, b in zip(inner_blocks, nested, strict=True))
                    blocks.append({**block, "content": inner_blocks} if changed else block)
                    continue
                blocks.append(filtered(block, position))
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

    1. If ``native`` contains a media kind that is not in ``enabled``, or an
       attachment in a form models do not accept (an SVG image, URL audio; see
       ``_unsendable``), raise ``ReasoningReplayError``. The caller must replay
       a portable version of the turn instead.
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
    if isinstance(content, (list, tuple)) and any(
        isinstance(block, Mapping)
        and (kind := attachment_kind(block)) is not None
        and _unsendable(kind, block) is not None
        for block in content
    ):
        # An SVG (or URL audio) inside opaque state cannot be replaced by a
        # note either: the caller falls back to the portable replay, whose
        # media the filter can replace.
        raise ReasoningReplayError(
            "Opaque assistant state holds media in a form models do not accept; "
            "supply a portable assistant turn instead."
        )
    from nooa.media_policy import reject_unsupported_media

    reject_unsupported_media([native], interface)
