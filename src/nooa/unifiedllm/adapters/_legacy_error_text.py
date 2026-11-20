# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure historical provider-text fallbacks for deliberately unmigrated clients.

No SDK imports, including lazy imports. Strict typed callers never search a
failure cause; these cycle-safe helpers exist solely for raw legacy contracts.
"""

import re

# Provider/LiteLLM error substrings to fall back on when an error only carries a
# stringified form (e.g. raw gateway HTML) without a usable ``.status_code`` or a
# ``status {code}`` token. Keyed by HTTP status code so a phrase only triggers a
# retry when that code is present in ``RetryConfig.retryable_status_codes``.
# ``error_str`` is lowercased before matching, so class-name forms (e.g.
# ``BadGatewayError``) match too.
_RETRYABLE_STATUS_PHRASES: dict[int, tuple[str, ...]] = {
    500: ("internal server error", "internalservererror"),
    502: ("bad gateway", "badgatewayerror"),
    503: ("service unavailable", "serviceunavailableerror", "server unavailable"),
    504: ("gateway timeout", "gateway time-out"),
}

# Endpoint failures often arrive without a useful HTTP status. Match exception
# class names through the MRO so optional provider/httpx/aiohttp exception types
# do not become import-time dependencies; the names below cover transport,
# timeout, disconnected-server, and provider APIConnectionError-style failures.
_RETRYABLE_ENDPOINT_CLASS_NAMES = {
    "apiconnectionerror",
    "apitimeouterror",
    "connecterror",
    "connecttimeout",
    "connectionerror",
    "connectiontimeouterror",
    "networkerror",
    "readtimeout",
    "remoteprotocolerror",
    "serverdisconnectederror",
    "timeout",
    "timeouterror",
    "transporterror",
    "writeerror",
    "writetimeout",
}

# When an exception is just a generic wrapper with text, fall back to phrases
# that indicate the endpoint/transport failed before a usable model response was
# produced. Structured non-retryable HTTP statuses return before this list is used.
_RETRYABLE_ENDPOINT_PHRASES = (
    "api connection error",
    "cannot connect",
    "can't connect",
    "connection aborted",
    "connection closed",
    "connection reset",
    "connection refused",
    "connection terminated",
    "connection timed out",
    "endpoint not reachable",
    "failed to connect",
    "max retries exceeded",
    "name resolution",
    "network is unreachable",
    "not reachable",
    "nodename nor servname",
    "remote disconnected",
    "remote end closed",
    "server disconnected",
    "temporary failure in name resolution",
    "timed out",
    "timeout",
    "transport error",
)


def _is_retryable_endpoint_error(error: Exception, error_str: str) -> bool:
    """Return True for transient network/transport failures without HTTP status."""
    for cls in type(error).__mro__:
        if cls.__name__.lower() in _RETRYABLE_ENDPOINT_CLASS_NAMES:
            return True

    if any(phrase in error_str for phrase in _RETRYABLE_ENDPOINT_PHRASES):
        return True

    return "connection" in error_str and any(
        phrase in error_str
        for phrase in (
            "abort",
            "closed",
            "disconnect",
            "failed",
            "lost",
            "reset",
            "refused",
            "terminated",
            "timed out",
            "unreachable",
        )
    )


def legacy_retryable(error, config):
    # Inspect structured status data before falling back to message heuristics.
    # LiteLLM and OpenAI APIStatusError subclasses expose ``.status_code``
    # directly, while HTTPX stores it on ``HTTPStatusError.response.status_code``.
    # A known non-retryable status must remain terminal: if broad text matching
    # ran first, an endpoint or provider message could accidentally make a
    # permanent 4xx authentication/model error look transient and retry it.
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429:
            return (True, True) if 429 in config.retryable_status_codes else (False, False)
        if status_code in config.retryable_status_codes:
            return True, False
        return False, False

    error_str = str(error).lower()

    # Check for rate limit (429) — only retryable if 429 is in the configured set.
    if "status 429" in error_str or "rate limit" in error_str:
        return (True, True) if 429 in config.retryable_status_codes else (False, False)

    # Check for retryable status codes
    for code in config.retryable_status_codes:
        if f"status {code}" in error_str:
            return True, False
        # Fall back to provider/LiteLLM phrasing (e.g. "502 Bad Gateway",
        # "BadGatewayError") when the "status {code}" token is absent.
        if any(phrase in error_str for phrase in _RETRYABLE_STATUS_PHRASES.get(code, ())):
            return True, False

    # Check for timeout/connection exceptions configured by the caller.
    if isinstance(error, config.retryable_exceptions):
        return True, False

    # Check for endpoint-level transient failures that may not expose a status
    # code, including LiteLLM APIConnectionError and httpx transport errors.
    if _is_retryable_endpoint_error(error, error_str):
        return True, False

    return False, False


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
    if status not in {400, 415, 422}:
        return False
    if image_limit(exc) is not None:
        return False
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    text = (
        str(error.get("message", str(exc))).lower() if isinstance(error, dict) else str(exc).lower()
    )
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


_PROMPT_TOKENS_RE = re.compile(
    r"(?:prompt|request)[^0-9]*"
    # "contains/has N tokens" (OpenAI/NVIDIA) OR
    # "is too long: N tokens" (Anthropic / Azure-Anthropic gateway)
    r"(?:contains?|has|is\s+too\s+long\s*:?)\s+(?:at\s+least\s+)?(\d[\d,]*)\s*(?:input\s+)?tokens",
    re.IGNORECASE,
)

_CONTEXT_WINDOW_TOKENS_RE = re.compile(
    r"maximum\s+context\s+length\s+is\s+(\d[\d,]*)\s+tokens"
    r"|context\s+window\s+of\s+(?:this\s+model\s+is\s+)?(\d[\d,]*)\s+tokens"
    r"|(\d[\d,]*)\s+tokens\s*>\s*(\d[\d,]*)\s+maximum",
    re.IGNORECASE,
)


def _is_context_window_error(exc: BaseException) -> bool:
    """Recognize context-window errors using typed and provider-normalized signals."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__.lower()
        msg = str(cur).lower()
        if (
            "contextwindowexceeded" in name
            or "context length" in msg
            or "context_length_exceeded" in msg
            or "context window" in msg
            # Anthropic / Azure-Anthropic gateway phrasing, surfaced as a
            # BadRequestError rather than a typed ContextWindowExceededError:
            #   "prompt is too long: 1017198 tokens > 1000000 maximum"
            or "prompt is too long" in msg
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _parse_prompt_tokens(exc: BaseException) -> int | None:
    """Extract prompt token count from a ContextWindowExceededError chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        m = _PROMPT_TOKENS_RE.search(str(cur))
        if m:
            return int(m.group(1).replace(",", ""))
        cur = cur.__cause__ or cur.__context__
    return None


def _parse_context_window_tokens(exc: BaseException) -> int | None:
    """Extract model context-window size from a ContextWindowExceededError chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        m = _CONTEXT_WINDOW_TOKENS_RE.search(str(cur))
        if m:
            # Alternatives capture either the window directly or "input > maximum".
            for group in reversed(m.groups()):
                if group:
                    return int(group.replace(",", ""))
        cur = cur.__cause__ or cur.__context__
    return None
