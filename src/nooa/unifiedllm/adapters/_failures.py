# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict built-in SDK failure boundary; one send remains one wire attempt.

Only installed LiteLLM/OpenAI types are recognized. Historical raw heuristics
live in the pure compatibility helper, not in retry or runtime policy.
"""

import math
import time
from collections.abc import Mapping
from contextlib import contextmanager
from email.utils import parsedate_to_datetime

import httpx
import openai
from litellm import exceptions as sdk
from litellm.llms.base_llm.chat.transformation import BaseLLMException

from nooa.unifiedllm.admission import AdmissionError
from nooa.unifiedllm.errors import FailureKind as K
from nooa.unifiedllm.errors import LLMFailure, _ReplyParseError

from . import _legacy_error_text as legacy


def _integer(value, minimum=0, maximum=None):
    return (
        value
        if type(value) is int and value >= minimum and (maximum is None or value <= maximum)
        else None
    )


def _body(error):
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return {}
    body = body.get("error", body)
    return body if isinstance(body, Mapping) else {}


def _text(value):
    return value if isinstance(value, str) else None


def _sdk_metadata(error):
    """Recover concrete LiteLLM's lossy mapping, not arbitrary cause chains.

    1.97 maps vendor text before HTTP status and manufactures a new response.
    Its immediate BaseLLMException context is the installed dispatch wrapper;
    Responses keeps the actual response in that wrapper's HTTPStatusError
    context. Only these concrete, bounded links license overriding synthetic
    status/body. The raised failure's cause remains the original SDK object.
    """
    context = error.__context__
    if not isinstance(error, openai.APIError) or type(error).__module__ != "litellm.exceptions":
        return error, _body(error)
    if isinstance(context, openai.APIStatusError) and type(context).__module__.startswith("openai"):
        return context, _body(context)
    if not isinstance(context, BaseLLMException):
        return error, _body(error)
    body = _body(context)
    wire = context.__context__
    if isinstance(wire, httpx.HTTPStatusError):
        response = wire.response
        try:
            payload = response.json()
        except (ValueError, RuntimeError):
            payload = None
        if isinstance(payload, Mapping):
            payload = payload.get("error", payload)
            if isinstance(payload, Mapping):
                body = payload
        return wire, body
    return context, body


def retry_after(error, *, now=None):
    """Real response or documented SDK headers; finite seconds/HTTP date only."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if not isinstance(headers, Mapping):
        headers = getattr(error, "headers", None)
    value = None
    if isinstance(headers, Mapping):
        value = next(
            (v for k, v in headers.items() if isinstance(k, str) and k.lower() == "retry-after"),
            None,
        )
    if value is None:
        # LiteLLM documents .headers separately from its synthetic response.
        headers = getattr(error, "headers", None)
        if isinstance(headers, Mapping):
            value = next(
                (
                    v
                    for k, v in headers.items()
                    if isinstance(k, str) and k.lower() == "retry-after"
                ),
                None,
            )
    if value is None:
        # Installed LiteLLM exception mapping explicitly preserves vendor headers here.
        headers = getattr(error, "litellm_response_headers", None)
        if isinstance(headers, Mapping):
            value = next(
                (
                    v
                    for k, v in headers.items()
                    if isinstance(k, str) and k.lower() == "retry-after"
                ),
                None,
            )
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
    except (ValueError, OverflowError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                return None
            seconds = max(0.0, date.timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _slot(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def truncated(reply):
    """Only trusted finish/incomplete evidence; filtering is not truncation."""
    choices = _slot(reply, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        if _slot(choices[0], "finish_reason") == "length":
            return True
    details = _slot(reply, "incomplete_details")
    return (
        _slot(reply, "status") == "incomplete" and _slot(details, "reason") == "max_output_tokens"
    )


def normalize(error, phase="wire", *, reply=None):
    """Bounded installed SDK facts, never generic cause-chain or MRO guesses.

    Preserve original typed failures and original paid-parse exception causes.
    """
    if isinstance(error, LLMFailure):
        return error
    if isinstance(error, _ReplyParseError):
        error = error.error
        phase = "parse"
        if isinstance(error, LLMFailure):
            return error
    if phase == "wire" and not isinstance(
        error,
        (
            openai.APIError,
            openai.LengthFinishReasonError,
            sdk.BudgetExceededError,
            sdk.UnsupportedParamsError,
            sdk.LiteLLMUnknownProvider,
            TimeoutError,
            httpx.HTTPError,
        ),
    ):
        # An unknown wrapper's status attributes are not SDK evidence either.
        return LLMFailure(K.CONFIG, str(error), cause=error)
    metadata, body = _sdk_metadata(error) if phase == "wire" else (error, _body(error))
    status = _integer(getattr(metadata, "status_code", None), 100, 599)
    if status is None:
        status = _integer(
            getattr(getattr(metadata, "response", None), "status_code", None), 100, 599
        )
    outer_status = _integer(getattr(error, "status_code", None), 100, 599)
    if outer_status is not None and 400 <= outer_status < 500 and outer_status not in (408, 429):
        status = outer_status  # An explicitly terminal outer SDK status always wins.
    code = _text(body.get("code")) or _text(getattr(error, "code", None))
    param = _text(body.get("param")) or _text(getattr(error, "param", None))
    facts = {}
    # Response validation and budget objects have synthetic HTTP statuses.
    if isinstance(error, openai.LengthFinishReasonError):
        kind = K.OUTPUT_LIMIT  # This SDK type itself certifies a length finish.
        status = None
    elif phase == "parse" or isinstance(error, openai.APIResponseValidationError):
        kind = K.OUTPUT_LIMIT if truncated(reply) else K.BAD_REQUEST
        status = None
    elif isinstance(
        error, (sdk.BudgetExceededError, sdk.UnsupportedParamsError, sdk.LiteLLMUnknownProvider)
    ) or (
        isinstance(error, sdk.RateLimitError)
        and getattr(error, "rate_limit_type", None) in ("budget", "max_iterations")
    ):
        kind = K.CONFIG
        status = None
    elif status in (401, 403) or isinstance(
        error, (openai.AuthenticationError, openai.PermissionDeniedError)
    ):
        kind = K.AUTH
    elif status is not None and 400 <= status < 500 and status not in (408, 429):
        # Outer terminal status dominates a misleading transient cause or name.
        kind = _request_kind(error, body, code, param, status, facts)
    elif (
        isinstance(
            error, (TimeoutError, openai.APITimeoutError, sdk.Timeout, httpx.TimeoutException)
        )
        or status == 408
    ):
        kind = K.TIMEOUT
    elif isinstance(error, (openai.APIConnectionError, httpx.TransportError)):
        kind = K.TRANSIENT
        status = None  # LiteLLM manufactures 500 for connection errors.
    elif status == 429:
        kind = K.RATE_LIMIT
    elif status is not None and 500 <= status <= 599:
        kind = K.TRANSIENT
    elif phase == "config":
        kind = K.CONFIG
    elif phase == "prepare":
        kind = K.BAD_REQUEST if isinstance(error, (TypeError, ValueError)) else K.CONFIG
    else:
        # Unknown SDK/custom raw exceptions must not acquire retries by wording.
        kind = K.CONFIG
    if kind == K.CONTEXT_WINDOW:
        facts["prompt_tokens"] = _integer(
            body.get("prompt_tokens", getattr(error, "prompt_tokens", None))
        )
        facts["context_window"] = _integer(
            body.get("context_window", getattr(error, "context_window", None)), 1
        )
        # Known SDK context classification licenses narrowly documented wording,
        # but only the outer message, never an unrelated cause's facts.
        message = Exception(str(error))
        if facts["prompt_tokens"] is None:
            facts["prompt_tokens"] = _integer(legacy._parse_prompt_tokens(message))
        if facts["context_window"] is None:
            facts["context_window"] = _integer(legacy._parse_context_window_tokens(message), 1)
    hint = None
    if kind in (K.RATE_LIMIT, K.TRANSIENT, K.TIMEOUT):
        hint = retry_after(metadata)
        if hint is None:
            hint = retry_after(error)
    return LLMFailure(kind, str(error), status_code=status, retry_after=hint, cause=error, **facts)


def _request_kind(error, body, code, param, status, facts):
    if isinstance(error, sdk.ContextWindowExceededError) or code == "context_length_exceeded":
        return K.CONTEXT_WINDOW
    if code in (
        "model_not_found",
        "invalid_model",
        "invalid_provider",
        "invalid_api_base",
    ) or param in ("model", "api_base", "base_url", "custom_llm_provider"):
        return K.CONFIG
    # Media rejection requires a request status, not image-looking transient text.
    if status in (400, 415, 422):
        media = _text(body.get("media_kind"))
        if media not in ("image", "pdf", "audio", "video"):
            media = (
                "image"
                if code
                in (
                    "image_limit_exceeded",
                    "too_many_images",
                    "invalid_image",
                    "invalid_image_url",
                    "image_parse_error",
                    "unsupported_image",
                )
                else None
            )
        limit = _integer(body.get("media_count_limit"))
        # Existing narrow count wording is retained at the adapter boundary.
        outer = Exception(str(error))
        outer.body = body
        outer.status_code = status
        text_limit = legacy.image_limit(outer)
        if (
            code in ("media_limit_exceeded", "image_limit_exceeded", "too_many_images")
            or text_limit is not None
        ):
            facts.update(
                media_kind=media or ("image" if text_limit is not None else None),
                media_count=_integer(body.get("media_count")),
                media_count_limit=limit if limit is not None else text_limit,
            )
            return K.MEDIA_LIMIT
        if (
            code
            in (
                "invalid_image",
                "invalid_image_url",
                "image_parse_error",
                "unsupported_image",
                "media_rejected",
            )
            and media is not None
        ):
            facts["media_kind"] = media
            return K.MEDIA_REJECTED
        for media in ("image", "pdf", "audio", "video"):
            if legacy.explicitly_rejected(outer, media):
                facts["media_kind"] = media
                return K.MEDIA_REJECTED
    return K.BAD_REQUEST


@contextmanager
def failure_boundary(strict, phase, *, reply=None):
    """Invocation-local policy; legacy views keep exact raw SDK/parse identity.

    Admission is outside the provider and stays raw until the public executor's
    final terminal wrapper, after retry has rejected it. BaseException escapes.
    """
    try:
        yield
    except Exception as error:
        if not strict or isinstance(error, (LLMFailure, AdmissionError)):
            raise
        normalized = normalize(error, phase, reply=reply)
        raise normalized from normalized.__cause__
