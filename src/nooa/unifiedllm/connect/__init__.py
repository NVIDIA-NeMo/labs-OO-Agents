# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect: set up and test a model entry, shared by the CLI and TUI.

This module holds the logic only; it never prompts the user or prints.

Terms used throughout:

- *Model entry*: one alias in the model registry file (``models.yaml``), a
  dict such as ``{"model_name": ..., "api_base": ..., "api_style": ...}``.
- *API style* (or interface): the request format used to talk to the
  endpoint: ``chat`` (Chat Completions), ``responses`` (OpenAI Responses),
  ``anthropic`` (Anthropic Messages), or a decision style (``systemone``,
  ``openai``) whose endpoint is a complete decisions URL.
- *Model listing*: the endpoint's ``GET /models`` response. The *catalogue* is
  the public OpenRouter model list, used for estimates.
- *Probe* or *test call*: one small, capped request that checks part of the
  configuration (``routing`` is the basic request; others test tools,
  reasoning levels, media, ...).
- *Provenance*: the ``entry["provenance"]`` dict that records where each
  configured value came from and the results of the test calls.
- *Reply cap*: the maximum number of reply tokens a request may use.

Main steps: ``discover`` lists models; ``plan`` builds the test calls for the
user to approve; ``run`` sends only the approved calls, each with a reply cap
and without retries on network errors; ``write`` saves one alias into a
registry file. Limits from catalogues are estimates and record their source.
A request being accepted does not prove a reasoning setting was applied. At
runtime NOOA reads only the saved entry, not the templates used here.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import math
import os
import re
import stat
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

from nooa.unifiedllm.limits import REPLY_CAP_KEYS

from ._advertised_media import advertised_media_suggestions, media_metadata
from ._context_limits import apply_context_window, bound_context_window, scoped_endpoint_limits
from ._records import (
    TYPED_TEXT_BASELINE,
    ProbeRecord,
    check_status,
    public_record,
    text_baseline_passed,
    text_baseline_reason,
)
from ._server_details import ModelDetails as ModelDetails
from ._server_details import bounded_json, openrouter_facts, token_limit
from ._server_details import server_details as server_details
from ._verification import (
    effective_config_fingerprint as effective_config_fingerprint,
)
from ._verification import (
    effective_config_snapshot as effective_config_snapshot,
)
from ._verification import (
    effective_route_scope,
    invalidate_verification,
    routing_allows_save,
    stamp_verification,
    verification_is_current,
)

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
# OpenRouter lists only text-output models by default; also ask for decision
# models.
OPENROUTER_MODEL_OUTPUTS = "text,decisions"
# Token budget used when --budget-tokens is not set. It is far above any real
# test plan, so no test is skipped for budget reasons. A large int is used
# instead of float("inf") so arithmetic, type hints and the JSON form of a
# ConnectPlan keep working with ints.
DEFAULT_CHECK_BUDGET = 10**15
DEFAULT_REASONING_OUTPUT_TOKENS = 4096
# Reply cap for the interface test calls (one short routing request per API
# style). Reasoning models spend part of the cap on reasoning before the
# visible reply, so a small cap (it was 200) made them run out of reply
# tokens before answering. Must stay within plan()'s allowed range 1..4096.
DISCOVERY_OUTPUT_TOKENS = 2048
REASONING_CHECK_PROMPT = """Eight jobs—A, B, C, D, E, F, G and H—must run one at a time.
Each job runs exactly once.

Rules:
- A runs exactly three positions after B.
- C runs immediately before E.
- F runs immediately after E.
- G runs immediately before D.
- E runs after A.
- H runs last.

Find the order that satisfies every rule.
Reply only with the eight letters in order, without explanation."""
ENCRYPTED_REASONING_EXPLANATION = (
    "Responses setup asks the server not to store replies. Connect requests encrypted "
    "reasoning so NOOA can carry the model's reasoning context into later turns and tool steps."
)


def _include_rejected(exc: Exception) -> bool:
    """Return True if ``exc`` says the endpoint rejected the ``include`` field.

    Used to detect servers that refuse ``include: ["reasoning.encrypted_content"]``
    (encrypted reasoning) on the Responses API. Requires HTTP 400/422 and
    either a structured ``param`` naming ``include``, or an error message that
    mentions include/encrypted_content together with a word like
    "unsupported". Only a True/False answer is returned; the provider's error
    text is not kept.
    """
    status = getattr(exc, "status_code", None)
    if status not in {400, 422}:
        return False
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    message = str(exc).lower()
    wrapped_param = re.search(r"""["']param["']\s*:\s*["']([^"']+)""", message)
    if param is None and wrapped_param:
        param = wrapped_param.group(1)
    if isinstance(param, str):
        # A structured param is enough even if the message is generic. A
        # rejection of another field (e.g. the model or message history) must
        # not turn off encrypted reasoning.
        return (
            param == "include"
            or param.startswith("include[")
            or param == "reasoning.encrypted_content"
        )
    return bool(
        re.search(r"\b(include|encrypted_content)\b", message)
        and re.search(
            r"unsupported|unknown|unrecognized|not supported|not allowed|invalid|reject", message
        )
    )


REASONING_DISCOVERY_TIMEOUT = 90

CACHE_PARAMS = ("prompt_cache_options", "prompt_cache_breakpoint", "cache_control")


def _cache_options_rejected(exc: Exception) -> bool:
    """Return True if ``exc`` says the endpoint rejected the prompt-cache fields.

    Requires HTTP 400/422 and a ``param`` or message naming one of
    ``CACHE_PARAMS``.
    """
    if getattr(exc, "status_code", None) not in {400, 422}:
        return False
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    message = str(exc).lower()
    if isinstance(param, str) and param in CACHE_PARAMS:
        return True
    return any(name in message for name in CACHE_PARAMS) and bool(
        re.search(
            r"unsupported|unknown|unrecognized|not supported|not allowed|invalid|extra", message
        )
    )


def _disable_explicit_cache(entry: dict, status: int) -> dict:
    """Turn off explicit prompt caching in ``entry`` and record why; return the record."""
    entry["cache_breakpoint"] = None
    record = {
        "source": "connect",
        "outcome": "rejected",
        "status_code": status,
        "reason": "The endpoint rejected explicit prompt caching; it is turned off for this model.",
        "checked_at": datetime.now(UTC).isoformat(),
    }
    entry["provenance"]["explicit_cache"] = record
    return record


def _disable_encrypted_reasoning(entry: dict, status: int) -> dict:
    """Stop requesting encrypted reasoning in ``entry`` and record why; return the record."""
    # An explicit empty list also stops the runtime from adding its own
    # default include for native OpenAI endpoints.
    entry["include"] = []
    record = {
        "source": "connect",
        "outcome": "rejected",
        "status_code": status,
        "reason": "The endpoint rejected encrypted reasoning; omitted from requests. Reasoning replay may be unavailable.",
        "checked_at": datetime.now(UTC).isoformat(),
    }
    entry["provenance"]["encrypted_reasoning"] = record
    return record


# Default for structured-output arguments: "not given" keeps the saved value,
# while an explicit None clears it.
STRUCTURED_OUTPUT_UNSET = object()


def _validate_structured_output(entry: dict, override=STRUCTURED_OUTPUT_UNSET):
    """Check the entry's ``structured_output_mode`` setting (and ``override``).

    Runs before any API key is read or request is built. Raises ValueError if
    the mode is invalid, if it appears inside ``extra_body`` or
    ``reasoning_levels`` (it is a connection setting, not a request field), or
    if it is set for anything other than a Completion client with the chat or
    anthropic API style.
    """
    from nooa.unifiedllm.structured_output import validate_structured_output_mode

    def reject_payload(value):
        if isinstance(value, dict):
            if "structured_output_mode" in value:
                raise ValueError(
                    "structured_output_mode belongs on the connection, not request settings"
                )
            for child in value.values():
                reject_payload(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                reject_payload(child)

    reject_payload(entry.get("extra_body"))
    reject_payload(entry.get("reasoning_levels"))
    if "structured_output_mode" in entry:
        validate_structured_output_mode(entry["structured_output_mode"])
    if override is not STRUCTURED_OUTPUT_UNSET:
        validate_structured_output_mode(override)
    declared = "structured_output_mode" in entry or override is not STRUCTURED_OUTPUT_UNSET
    if declared and (
        entry.get("client_type", "completion") != "completion"
        or entry.get("api_style", "chat") not in {"chat", "anthropic"}
    ):
        raise ValueError(
            "structured_output_mode is a Completion-only profile control (chat or anthropic)"
        )


_RESERVED = {
    "structured_output_mode",
    "media_inputs",
    "model",
    "messages",
    "input",
    "api_base",
    "base_url",
    "api_key",
    "custom_llm_provider",
    "extra_body",
    "client",
    "transport",
    "api_style",
    "replay_vendor",
}
_PATHS = {
    "chat": "chat/completions",
    "responses": "responses",
    "anthropic": "messages",
    "systemone": "",
    "openai": "",
}
# Decision API styles: their endpoint is the complete decisions URL, used as
# given (no path is appended).
DECISION_API_STYLES = ("systemone", "openai")


@dataclass(frozen=True)
class ProviderPreset:
    """Default connection settings for a well-known provider.

    Frontends offer these as starting points; the list of models is always
    read from the provider, not stored here.
    """

    label: str
    api_base: str
    api_style: str
    api_key_env: str


PROVIDERS = {
    "nvidia": ProviderPreset(
        "NVIDIA (build.nvidia.com)", "https://integrate.api.nvidia.com/v1", "chat", "NVIDIA_API_KEY"
    ),
    "hub": ProviderPreset(
        "NVIDIA Inference Hub (inference-api.nvidia.com)",
        "https://inference-api.nvidia.com/v1",
        "chat",
        "NVIDIA_INFERENCE_API_KEY",
    ),
    "openai": ProviderPreset("OpenAI", "https://api.openai.com/v1", "responses", "OPENAI_API_KEY"),
    "anthropic": ProviderPreset(
        "Anthropic", "https://api.anthropic.com/v1", "anthropic", "ANTHROPIC_API_KEY"
    ),
    "google": ProviderPreset(
        "Google (Gemini)",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "chat",
        "GEMINI_API_KEY",
    ),
    "openrouter": ProviderPreset(
        "OpenRouter", "https://openrouter.ai/api/v1", "chat", "OPENROUTER_API_KEY"
    ),
}


@dataclass(frozen=True)
class Probe:
    """One planned test call: a request body to POST, plus its limits.

    ``body`` never contains credentials. ``token_estimate`` is the tokens
    reserved from the budget, ``uses_configured_cap`` says whether the reply
    cap comes from the entry, ``expected_answer`` is the answer checked for (if
    any), and ``skip_reason`` is set when the call will not be sent.
    """

    name: str
    body: dict[str, Any]
    token_estimate: int
    timeout_seconds: float = 30
    uses_configured_cap: bool = True
    input_reservation: int = 512
    expected_answer: str | None = None
    skip_reason: str | None = None


@dataclass(frozen=True)
class ConnectPlan:
    """The test calls planned for one model entry, shown to the user before ``run``.

    The token and price estimates help the user decide; they do not limit
    what the provider bills.
    """

    alias: str
    entry: dict[str, Any]
    probes: tuple[Probe, ...]
    budget_tokens: int
    token_estimate: int
    price_estimate: float | None
    session_checks: bool = False


@dataclass(frozen=True)
class ConnectResult:
    """A model entry after its test calls, with results under ``entry["provenance"]``."""

    alias: str
    entry: dict[str, Any]


@dataclass(frozen=True)
class InterfaceResult:
    """Results of trying several API styles against one model (``check_interfaces``).

    These describe what happened in this run, not a lasting statement of
    what the provider supports. ``results`` maps API style -> result.
    """

    results: dict[str, ConnectResult]
    tokens_charged_to_budget: int

    @property
    def accepted(self) -> tuple[str, ...]:
        """API styles whose routing test call was accepted, in test order."""
        return tuple(
            style
            for style, result in self.results.items()
            if result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"
        )

    @property
    def accepted_client_types(self) -> tuple[str, ...]:
        """Client types (completion, responses, decision) of the accepted styles, no repeats."""
        return tuple(
            dict.fromkeys(self.results[style].entry["client_type"] for style in self.accepted)
        )

    @property
    def reasoning_observed(self) -> dict[str, bool]:
        """For each API style, whether a reasoning-level test call returned reasoning.

        All False unless ``check_interfaces`` was given ``reasoning_levels``.
        A style can accept requests and still never return reasoning.
        """
        observed: dict[str, bool] = {}
        for style, result in self.results.items():
            probes = result.entry["provenance"]["probes"]
            observed[style] = any(
                record.get("reasoning_observed")
                for name, record in probes.items()
                if name.startswith("level:")
            )
        return observed

    @property
    def recommended_style(self) -> str | None:
        """The API style to suggest to the user, or None if none was accepted.

        If both a decision client and another client type were accepted, the
        first accepted style wins. Otherwise the first accepted style that
        returned reasoning wins, else the first accepted style.
        """
        accepted = self.accepted
        if "decision" in self.accepted_client_types and len(self.accepted_client_types) > 1:
            return accepted[0]
        observed = self.reasoning_observed
        for style in accepted:
            if observed.get(style):
                return style
        return accepted[0] if accepted else None


@dataclass(frozen=True)
class ProbeUpdate:
    """Progress event from a running test: test ``name`` started or finished.

    ``outcome`` is the test's probe record (``{"outcome": "running"}`` while
    it runs).
    """

    name: str
    outcome: ProbeRecord


@dataclass(frozen=True)
class Verdict:
    """Test results of a model entry grouped by verdict (see ``verdict``).

    Tests the user did not approve are "skipped"; tests that ran but did not
    show the expected result "need attention". ``unobserved_levels`` lists
    reasoning levels whose test call returned no reasoning.
    """

    passed: tuple[str, ...]
    needs_attention: tuple[str, ...]
    skipped: tuple[str, ...]
    unobserved_levels: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True if at least one test passed and none was skipped or needs attention."""
        return bool(self.passed) and not self.needs_attention and not self.skipped


def verdict(entry: dict) -> Verdict:
    """Group the test results stored in ``entry`` by verdict.

    Uses only the stored records (``check_status``); makes no requests and
    assumes nothing about tests that did not run.
    """
    provenance = entry.get("provenance", {})
    missing = tuple(unobserved_reasoning_levels(entry))
    groups = {"passed": [], "attention": [], "skipped": []}
    for name, record in {
        **provenance.get("probes", {}),
        **provenance.get("session_checks", {}),
    }.items():
        status = check_status(
            name, record, entry=entry, missing_reasoning=name.removeprefix("level:") in missing
        )
        groups[status].append(name)
    return Verdict(
        tuple(groups["passed"]), tuple(groups["attention"]), tuple(groups["skipped"]), missing
    )


@dataclass(frozen=True)
class Discovery:
    """Result of ``discover``: the models an endpoint lists.

    Being listed does not mean a model works with a given API style.
    ``api_base`` is the base URL that requests are sent to; when the user gave
    a server root and only ``/v1/models`` answered, it ends in ``/v1``.
    ``models_url`` is the URL the listing was actually read from, which may
    differ from the request URL. ``request_count`` counts the GETs made.
    """

    api_base: str
    models: tuple[dict[str, Any], ...]
    models_url: str | None = None
    request_count: int = 0


class DiscoveryError(ValueError):
    """Model listing failed; the message contains no provider text or secrets.

    ``status_code`` is the HTTP status if there was one (frontends can offer
    to enter a new key on 401 or 403); ``request_count`` counts the GETs made.
    """

    def __init__(self, message: str, *, status_code: int | None = None, request_count: int = 0):
        super().__init__(message)
        self.status_code = status_code
        self.request_count = request_count


def normalize_endpoint(endpoint: str) -> str:
    """Return the API base URL for ``endpoint`` (an API base or its ``/models`` URL).

    Trailing slashes and a trailing ``/models`` are removed. Raises ValueError
    unless it is an HTTP(S) URL without user info, query or fragment; plain
    HTTP is allowed only for localhost.
    """
    address = urlsplit(endpoint.strip().rstrip("/"))
    if (
        address.scheme not in {"http", "https"}
        or not address.netloc
        or address.username
        or address.password
        or address.query
        or address.fragment
    ):
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query or fragment")
    if address.scheme == "http" and address.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for a remote endpoint")
    path = address.path.removesuffix("/models")
    return urlunsplit((address.scheme, address.netloc, path, "", ""))


def _headers(style: str, key: str | None) -> dict[str, str]:
    """Return the auth headers for a model listing request in API ``style``."""
    if style == "anthropic":
        return {"anthropic-version": "2023-06-01", **({"x-api-key": key} if key else {})}
    return {"Authorization": f"Bearer {key}"} if key else {}


def validate_models_url(models_url: str) -> str:
    """Check a user-given model listing URL and return it, with its path unchanged.

    Applies the same rules as ``normalize_endpoint`` (raises ValueError).
    """
    normalize_endpoint(models_url)  # Same HTTPS/credential/query restrictions.
    return models_url.strip()


async def discover(
    endpoint: str,
    *,
    api_style: str = "chat",
    api_key: str | None = None,
    models_url: str | None = None,
    listing_style: str | None = None,
) -> Discovery:
    """Read the model listing of ``endpoint`` and return the models it lists.

    Sends only GET requests (no model calls) and stores no credentials.

    ``models_url`` is a full listing URL to use instead of
    ``<endpoint>/models``; it is required for decision API styles, does not
    change where requests are sent, and is not retried under ``/v1``.
    ``listing_style`` selects how the listing is authenticated and paged
    (chat, responses or anthropic); it does not affect requests. ``api_key``
    is used for the listing only; the caller decides which key that is.

    For a server root URL, ``/models`` is tried first and ``/v1/models`` only
    if that returns 404. Anthropic listings are read page by page (at most 50
    pages). Limits apply to the whole call: 30 seconds, 5 MiB and 5,000
    entries; redirects and automatic retries are disabled.

    Each model keeps only known, validated fields (ID, token limits, names,
    modality lists, media and reasoning capabilities, and for the OpenRouter
    catalogue a ``public_model`` record). Models are sorted by ID.

    Raises ValueError for invalid arguments and ``DiscoveryError`` for any
    listing failure, duplicate IDs, or an empty listing.
    """
    if api_style in DECISION_API_STYLES and models_url is None:
        raise ValueError(
            "Decision endpoints do not list decision models themselves; supply models_url "
            "for a shared catalogue or the exact decision model ID"
        )
    if api_style not in _PATHS:
        raise ValueError("Unknown api_style")
    style = listing_style or ("chat" if api_style in DECISION_API_STYLES else api_style)
    if style not in {"chat", "responses", "anthropic"}:
        raise ValueError("listing_style must be chat, responses or anthropic")
    base = normalize_endpoint(endpoint)
    url = validate_models_url(models_url) if models_url is not None else base + "/models"
    root = models_url is None and not urlsplit(base).path
    models: dict[str, dict] = {}
    cursors: set[str] = set()
    params = {"limit": "100"} if style == "anthropic" else {}
    if url == CATALOGUE_URL:
        params["output_modalities"] = OPENROUTER_MODEL_OUTPUTS
    size = 0
    count = 0
    request_count = 0
    try:
        async with (
            asyncio.timeout(30),
            httpx.AsyncClient(timeout=15, follow_redirects=False) as client,
        ):
            while True:
                request_count += 1
                async with client.stream(
                    "GET", url, headers=_headers(style, api_key), params=params
                ) as response:
                    if response.status_code == 404 and root:
                        base += "/v1"
                        url = base + "/models"
                        root = False
                        continue
                    if not response.is_success:
                        raise DiscoveryError(
                            f"Model discovery returned HTTP {response.status_code}",
                            status_code=response.status_code,
                            request_count=request_count,
                        )
                    chunks = []
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 5 * 1024 * 1024:
                            raise DiscoveryError(
                                "Model discovery exceeds 5 MiB", request_count=request_count
                            )
                        chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list):
                    raise DiscoveryError(
                        "Model discovery must contain a data list", request_count=request_count
                    )
                count += len(data)
                if count > 5000:
                    raise DiscoveryError(
                        "Model discovery exceeds 5,000 entries", request_count=request_count
                    )
                for item in data:
                    name = item.get("id") if isinstance(item, dict) else None
                    if (
                        not isinstance(name, str)
                        or not name.strip()
                        or len(name) > 500
                        or any(ord(c) < 32 for c in name)
                    ):
                        continue
                    model = {"id": name}
                    for field, candidates in (
                        (
                            "context_window",
                            (
                                "context_window",
                                "context_length",
                                "max_model_len",
                            ),
                        ),
                        (
                            "max_output_tokens",
                            ("max_output_tokens", "max_completion_tokens", "max_tokens")
                            if style == "anthropic"
                            else ("max_output_tokens", "max_completion_tokens"),
                        ),
                        ("max_input_tokens", ("max_input_tokens",)),
                    ):
                        for candidate in candidates:
                            value = item.get(candidate)
                            if token_limit(value) is not None:
                                model[field] = token_limit(value)
                                if field != candidate:
                                    model.setdefault("metadata_fields", {})[field] = (
                                        "data[]." + candidate
                                    )
                                break
                    # Kept as information only; these never enable an API
                    # style or media input.
                    for field in (
                        "name",
                        "display_name",
                        "description",
                        "client_type",
                        "model_type",
                    ):
                        value = item.get(field)
                        if (
                            isinstance(value, str)
                            and len(value) <= 4096
                            and not any(ord(c) < 32 and c not in "\n\t" for c in value)
                        ):
                            model[field] = value
                    for field in (
                        "supported_interfaces",
                        "input_modalities",
                        "output_modalities",
                        "supported_parameters",
                    ):
                        value = item.get(field)
                        if (
                            isinstance(value, list)
                            and len(value) <= 100
                            and all(
                                isinstance(v, str)
                                and len(v) <= 200
                                and not any(ord(c) < 32 for c in v)
                                for v in value
                            )
                        ):
                            model[field] = value
                    architecture = item.get("architecture")
                    if isinstance(architecture, dict):
                        hints = {
                            k: v
                            for k, v in architecture.items()
                            if k in {"input_modalities", "output_modalities"}
                            and isinstance(v, list)
                            and len(v) <= 100
                            and all(
                                isinstance(s, str)
                                and len(s) <= 200
                                and not any(ord(c) < 32 for c in s)
                                for s in v
                            )
                        }
                        if hints:
                            model["architecture"] = hints
                    # OpenRouter's top_provider limits can only lower the
                    # listed limits, never raise them.
                    top = item.get("top_provider")
                    if isinstance(top, dict):
                        for raw, field in (
                            ("context_length", "context_window"),
                            ("max_completion_tokens", "max_output_tokens"),
                        ):
                            value = top.get(raw)
                            if token_limit(value) is not None:
                                value = token_limit(value)
                                old = model.get(field)
                                model[field] = min(old, value) if isinstance(old, int) else value
                                if old is None or value < old:
                                    model.setdefault("metadata_fields", {})[field] = (
                                        "data[].top_provider." + raw
                                    )
                    media = media_metadata(item)
                    if "architecture" in media:
                        media["architecture"] = {
                            **model.get("architecture", {}),
                            **media["architecture"],
                        }
                    model.update(media)
                    from ._reasoning import reasoning_metadata

                    reasoning_facts = reasoning_metadata(item)
                    if reasoning_facts:
                        model.setdefault("capabilities", {}).update(reasoning_facts["capabilities"])
                    if media:
                        model["media_source"] = url
                    if url == CATALOGUE_URL:
                        model["public_model"] = openrouter_facts(item)
                    if name in models:
                        raise DiscoveryError(
                            "Model listing contains duplicate model IDs",
                            request_count=request_count,
                        )
                    models[name] = model
                if style != "anthropic" or not payload.get("has_more"):
                    break
                cursor = payload.get("last_id")
                if (
                    not isinstance(cursor, str)
                    or not cursor
                    or cursor in cursors
                    or len(cursors) >= 50
                ):
                    raise DiscoveryError(
                        "Invalid or excessive model discovery pagination",
                        request_count=request_count,
                    )
                cursors.add(cursor)
                params["after_id"] = cursor
    except (httpx.HTTPError, TimeoutError):
        raise DiscoveryError(
            "Model discovery connection failed or timed out", request_count=request_count
        ) from None
    except (ValueError, UnicodeDecodeError) as exc:
        if isinstance(exc, DiscoveryError):
            raise
        raise DiscoveryError(
            "Model discovery did not return valid JSON", request_count=request_count
        ) from None
    if not models:
        raise DiscoveryError(
            "Endpoint advertised no usable model IDs; supply a model explicitly",
            request_count=request_count,
        )
    return Discovery(base, tuple(models[name] for name in sorted(models)), url, request_count)


async def catalogue() -> list[dict]:
    """Fetch the public OpenRouter catalogue (no API key is sent).

    Returns the validated catalogue records (see ``openrouter_facts``). Raises
    ValueError or an httpx error if the catalogue cannot be read.
    """
    async with asyncio.timeout(30), httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        payload = await bounded_json(
            client, CATALOGUE_URL, params={"output_modalities": OPENROUTER_MODEL_OUTPUTS}
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ValueError("Catalogue response must contain a list of model objects")
        if len(data) > 5000:
            raise ValueError("Catalogue exceeds 5,000 entries")
        data = [openrouter_facts(item) for item in data]
        if len({item["id"] for item in data}) != len(data):
            raise ValueError("Catalogue contains duplicate model IDs")
        for item in data:
            if media_metadata(item):
                item["media_source"] = CATALOGUE_URL
        return data


def _normalized_model_id(value: str) -> str:
    """Lower-case a model ID and remove ``-``, ``_`` and ``.`` for comparison.

    Shared by ``match_models`` and ``fuzzy_match_models`` so both agree on
    which IDs count as the same.
    """
    return re.sub(r"[-_.]", "", value.lower())


def match_models(model: str, models: list[dict]) -> list[dict]:
    """Return up to three listed models whose IDs match ``model``, best first.

    A match is an equal normalised ID, an ID that ``model`` ends with after a
    ``/`` (``model`` has an extra prefix), or an equal last path segment. These
    are suggestions; the frontend must have the user confirm one.
    """
    normalized = _normalized_model_id
    target = normalized(model)
    matches = []
    for item in models:
        name = item.get("id")
        if not isinstance(name, str):
            continue
        candidate = normalized(name)
        if target == candidate or target.endswith("/" + candidate):
            matches.append((0, name, item))
        elif target.rsplit("/", 1)[-1] == candidate.rsplit("/", 1)[-1]:
            matches.append((1, name, item))
    return [item for _, _, item in sorted(matches, key=lambda item: item[:2])[:3]]


_FUZZY_MATCH_CUTOFF = 0.6


def fuzzy_match_models(model: str, models: list[dict]) -> list[dict]:
    """Return up to three similar model IDs ("did you mean"), most similar first.

    Used when ``match_models`` finds nothing. A model ID routed through a
    gateway (e.g. ``aws/anthropic/bedrock-claude-opus-5``) contains path parts
    the catalogue does not use, so exact matching can fail even though the
    model is listed under its own name. Similarity is measured on the whole ID
    and on its last path segment, with a cutoff of 0.6. The result is never
    selected automatically; the frontend must have the user confirm one.
    """
    normalized = _normalized_model_id
    target = normalized(model)
    target_tail = target.rsplit("/", 1)[-1]
    scored = []
    for item in models:
        name = item.get("id")
        if not isinstance(name, str):
            continue
        candidate = normalized(name)
        candidate_tail = candidate.rsplit("/", 1)[-1]
        ratio = max(
            difflib.SequenceMatcher(None, target, candidate).ratio(),
            difflib.SequenceMatcher(None, target_tail, candidate_tail).ratio(),
        )
        if ratio >= _FUZZY_MATCH_CUTOFF:
            scored.append((ratio, name, item))
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, _, item in scored[:3]]


def reasoning_settings(template: str, style: str, level: str, *, budget: int = 4096) -> dict:
    """Return the request settings for reasoning ``level`` in a given ``template``.

    ``template`` names a known way providers control reasoning: ``effort``
    (``reasoning_effort``, or ``reasoning.effort`` for the Responses API),
    ``adaptive`` (Anthropic adaptive thinking with an effort), ``budget``
    (thinking with a token budget of ``budget``), ``toggle``
    (``chat_template_kwargs.enable_thinking``) or ``thinking`` (on/off).
    Levels "none", "off" and "disabled" turn reasoning off. The result is a
    setting to test, not a claim that the model supports it. Raises
    ValueError for an unknown template.

    For the Responses API, ``reasoning.summary`` is always requested with the
    effort. Some endpoints (seen with OpenAI, Azure and DeepSeek) return a
    summary anyway, but others (seen with Kimi and Qwen on the NVIDIA
    Inference Hub) leave out the whole reasoning item unless a summary is
    requested, even though they still report reasoning tokens. Requesting it
    is harmless where it is the default and needed to see reasoning at all
    elsewhere.
    """
    if template == "effort":
        return (
            {"reasoning": {"effort": level, "summary": "auto"}}
            if style == "responses"
            else {"reasoning_effort": level}
        )
    enabled = level not in {"none", "off", "disabled"}
    if template == "adaptive":
        return (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
            if enabled
            else {"thinking": {"type": "disabled"}}
        )
    if template == "budget":
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}, "max_tokens": budget + 1024}
            if enabled
            else {"thinking": {"type": "disabled"}}
        )
    if template == "toggle":
        return {"chat_template_kwargs": {"enable_thinking": enabled}}
    if template == "thinking":
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    raise ValueError(f"Unknown reasoning template: {template}")


def unobserved_reasoning_levels(entry: dict) -> list[str]:
    """Return the reasoning levels that should reason but returned no reasoning.

    A level is listed if its test call was accepted, no reasoning was seen in
    the reply, and its settings turn reasoning on. Whether reasoning is "on"
    is read from the settings (as produced by ``reasoning_settings``), not
    from the level's name or the provider; custom settings with unknown
    meaning are not listed. Seeing no reasoning does not prove reasoning was
    off. Frontends use this to show one warning; the stored test results are
    not changed.
    """
    missing = []
    records = entry.get("provenance", {}).get("probes", {})
    for label, params in entry.get("reasoning_levels", {}).items():
        record = records.get(f"level:{label}", {})
        if record.get("outcome") != "accepted" or record.get("reasoning_observed"):
            continue
        reasoning = params.get("reasoning")
        thinking = params.get("thinking")
        template = params.get("chat_template_kwargs")
        efforts = (
            params.get("reasoning_effort"),
            reasoning.get("effort") if isinstance(reasoning, dict) else None,
        )
        enabled = any(
            isinstance(effort, str)
            and effort.strip().lower() not in {"", "none", "off", "disabled"}
            for effort in efforts
        )
        enabled |= isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
        enabled |= isinstance(template, dict) and template.get("enable_thinking") is True
        if enabled:
            missing.append(label)
    return missing


def configure_entry(
    entry: dict,
    *,
    reply_tokens: int | None = None,
    cache_enabled: bool | None = None,
    structured_output_mode=STRUCTURED_OUTPUT_UNSET,
) -> dict:
    """Validate ``entry`` and return a copy with Connect's defaults filled in.

    Does no I/O. Raises ValueError for invalid or conflicting settings,
    including literal credentials anywhere in the entry (use ``api_key_env``).

    - ``max_tokens`` is the reply cap sent on every request. ``reply_tokens``
      sets it; otherwise the saved value is kept, or a default is chosen (at
      most 32768 and at most half the context window). Limits published by
      the endpoint or catalogue are recorded in ``provenance`` and only bound
      the cap; they are never used as the cap themselves.
    - For the Responses API with ``store: false``, encrypted reasoning is
      requested (``include: ["reasoning.encrypted_content"]``) so reasoning can
      be sent back on later turns; ``include: []`` explicitly opts out.
    - ``structured_output_mode`` applies only to Completion clients. Not
      passing it keeps the saved value, None clears it, "auto" keeps the
      older way of requesting JSON output, and only "anthropic" uses
      Anthropic's ``output_config.format``.
    - ``cache_enabled=False`` turns off explicit prompt-cache options and
      markers (the provider's automatic caching is unaffected); True selects
      the runtime's automatic policy; None keeps the entry's setting.

    Also records the active reasoning settings and warnings in
    ``provenance`` and recomputes which test results are still current.
    Decision entries are only validated, since they have no reply cap,
    reasoning or cache settings.
    """

    def reject_credentials(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key.lower().replace("-", "_") in {
                    "api_key",
                    "apikey",
                    "x_api_key",
                    "authorization",
                    "proxy_authorization",
                    "access_token",
                    "client_secret",
                    "password",
                }:
                    raise ValueError(
                        "Use api_key_env, never literal credentials in model configuration"
                    )
                reject_credentials(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                reject_credentials(child)

    if cache_enabled is not None and not isinstance(cache_enabled, bool):
        raise ValueError("cache_enabled must be a boolean or None")
    _validate_structured_output(entry, structured_output_mode)
    reject_credentials(entry)
    from nooa.unifiedllm.media_inputs import validate_media_inputs

    result = deepcopy(entry)
    effective_mode = (
        entry.get("structured_output_mode")
        if structured_output_mode is STRUCTURED_OUTPUT_UNSET
        else structured_output_mode
    )
    provenance = result.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    snapshot = provenance.get("structured_output_profile")
    if (
        "structured_output_mode" in entry
        or structured_output_mode is not STRUCTURED_OUTPUT_UNSET
        or snapshot is not None
    ):
        result.setdefault("provenance", {})["structured_output_profile"] = (
            {**deepcopy(snapshot), "mode": effective_mode}
            if isinstance(snapshot, dict) and snapshot.get("mode") == effective_mode
            else {"mode": effective_mode}
        )
    if structured_output_mode is not STRUCTURED_OUTPUT_UNSET:
        result["structured_output_mode"] = structured_output_mode
    from nooa.media_policy import entry_input_interface, validate_implemented_media_inputs

    interface = entry_input_interface(result)
    result["media_inputs"] = list(
        validate_implemented_media_inputs(result.get("media_inputs"), interface)
        if interface in {"anthropic", "responses"}
        else validate_media_inputs(result.get("media_inputs"))
    )
    for patch in (result.get("reasoning_levels") or {}).values():
        if "media_inputs" in patch or "media_inputs" in (patch.get("extra_body") or {}):
            raise ValueError("media_inputs belongs on the connection, not a reasoning level")
    result.setdefault("transport", "direct")
    if result.get("client_type") == "decision":
        if cache_enabled is not None:
            raise ValueError("Decision models do not configure chat caching")
        from nooa.decisions.media import decision_media_inputs

        result["media_inputs"] = list(decision_media_inputs(result["media_inputs"]))
        result.setdefault("decision_image_transport", "json")
        result.setdefault("typesafe_danger_zone_images", False)
        image_transport = result["decision_image_transport"]
        optin = result["typesafe_danger_zone_images"]
        if image_transport not in {"json", "typesafe_multipart"}:
            raise ValueError("decision_image_transport must be json or typesafe_multipart")
        if not isinstance(optin, bool):
            raise ValueError("typesafe_danger_zone_images must be an explicit boolean")
        if image_transport == "typesafe_multipart" and result.get("api_style") != "systemone":
            raise ValueError("typesafe_multipart requires api_style=systemone")
        if optin and image_transport != "typesafe_multipart":
            raise ValueError("typesafe_danger_zone_images requires typesafe_multipart transport")
        if result.get("api_style") not in DECISION_API_STYLES:
            raise ValueError("Decision entries require api_style: systemone or openai")
        if not isinstance(result.get("endpoint"), str) or not result["endpoint"].strip():
            raise ValueError("Decision entries require a complete endpoint URL")
        if reply_tokens is not None:
            raise ValueError("Decision models do not use reply-token budgets")
        forbidden = {
            "max_tokens",
            "max_output_tokens",
            "reasoning_levels",
            "reasoning_default",
            "reasoning_level",
            "store",
            "include",
            "cache_breakpoint",
        }
        present = sorted(forbidden & result.keys())
        if present:
            raise ValueError(
                "Decision entries cannot configure chat-only fields: " + ", ".join(present)
            )
        provenance = result.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a mapping")
        provenance.setdefault("probes", {})
        provenance.setdefault(
            "not_probed",
            ["context_window", "reasoning", "tools", "sessions", "reply_limit"],
        )
        return invalidate_verification(result)
    provenance = result.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    if cache_enabled is False:

        def reject_cache_options(value):
            if isinstance(value, dict):
                if {
                    "prompt_cache_options",
                    "prompt_cache_breakpoint",
                    "cache_control",
                    "nooa_cache_boundary",
                } & value.keys():
                    raise ValueError(
                        "cache_enabled=False conflicts with explicit cache options/markers; "
                        "remove them from the profile settings first"
                    )
                for child in value.values():
                    reject_cache_options(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    reject_cache_options(child)

        # Raise instead of silently removing the user's own settings.
        reject_cache_options({k: v for k, v in result.items() if k != "provenance"})
    if cache_enabled is not None:
        cache_breakpoint = "auto" if cache_enabled else None
        result["cache_breakpoint"] = cache_breakpoint
    from nooa.unifiedllm.reasoning import ReasoningConfig

    declaration = ReasoningConfig(
        levels=result.get("reasoning_levels"), default=result.get("reasoning_default")
    )
    if result.get("reasoning_level") is not None:
        declaration.settings(result["reasoning_level"])
    extra = result.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise ValueError("extra_body must be a mapping")
    responses = result.get("api_style") == "responses" or result.get("client_type") == "responses"
    managed = {"max_tokens", "max_output_tokens", "max_completion_tokens", "media_inputs"}
    if responses:
        managed |= {"include", "store"}
    if managed & extra.keys():
        raise ValueError(
            "Reply limits and Responses retention settings belong on the entry, not in extra_body"
        )
    if "max_completion_tokens" in result:
        raise ValueError("Use max_tokens for the saved reply limit, not max_completion_tokens")
    provenance = result.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    window = result.get("context_window")
    if window is not None:
        model = result.get("model_name", "").split("/", 1)[-1]
        endpoint_limits = scoped_endpoint_limits(result)
        context_evidence = bound_context_window(window, model=model, endpoint_model=endpoint_limits)
        result["context_window"] = context_evidence["effective"]
        previous = provenance.get("context_window") or {}
        if previous and provenance.get("endpoint_limits") and not endpoint_limits:
            # The endpoint changed, so the stored endpoint limit no longer
            # applies: keep the value declared now, not the old limit or an
            # older, larger declaration.
            provenance["context_window"] = context_evidence
            previous = context_evidence
        if previous.get("effective", window) != window:
            previous = {
                **previous,
                "declared": window,
                "value": window,
                "effective_source": context_evidence["effective_source"],
            }
        if previous or endpoint_limits.get("id") == model:
            # Reconfiguring or importing may lower the effective value, but
            # must not bring back an older, larger declaration after the user
            # deliberately changed the value.
            provenance["context_window"] = {
                **context_evidence,
                "source": provenance.get("limit_sources", {}).get("context_length", "entry"),
                **previous,
                "effective": context_evidence["effective"],
            }
            if context_evidence["effective"] < window:
                provenance["context_window"]["effective_source"] = context_evidence[
                    "effective_source"
                ]
    ceiling = result.pop("max_output_tokens", None)
    limits = provenance.setdefault("catalogue_limits", {})
    if not isinstance(limits, dict):
        raise ValueError("catalogue_limits must be a mapping")
    if isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0:
        limits["max_completion_tokens"] = ceiling
    supplied_ceiling = ceiling
    ceiling = limits.get("max_completion_tokens")
    if (
        supplied_ceiling is None
        and str(provenance.get("limit_sources", {}).get("max_completion_tokens", "")).startswith(
            "endpoint"
        )
        and not scoped_endpoint_limits(result)
    ):
        # An output limit read from a different endpoint no longer applies.
        # A limit passed in this call (initial planning) still does.
        ceiling = None
    known_limits = [
        v
        for v in (ceiling, result.get("context_window"))
        if isinstance(v, int) and not isinstance(v, bool) and v > 0
    ]
    bound = min(known_limits) if known_limits else None
    cap = reply_tokens if reply_tokens is not None else result.get("max_tokens")
    source = "user" if reply_tokens is not None else "entry"
    if cap is None:
        window = result.get("context_window")
        default_bound = (
            min(bound, max(1, window // 2)) if isinstance(window, int) and bound else bound
        )
        cap = min(32768, default_bound) if default_bound else 32768
        source = "connect_default"
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError("Reply limit max_tokens must be a positive integer")
    window = result.get("context_window")
    if (
        reply_tokens is None
        and provenance.get("reply_limit", {}).get("source")
        in {"connect_default", "catalogue_recommendation"}
        and isinstance(window, int)
        and cap >= window
    ):
        cap = min(cap, max(1, window // 2))
        provenance["reply_limit"]["value"] = cap
    if isinstance(window, int) and cap >= window:
        raise ValueError(
            "Reply limit leaves no room for input; set max_tokens below context_window"
        )
    if (
        bound is not None
        and cap > bound
        and reply_tokens is None
        and provenance.get("reply_limit", {}).get("source")
        in {"connect_default", "catalogue_recommendation"}
    ):
        cap = bound
        provenance["reply_limit"]["value"] = cap
    if bound is not None and cap > bound:
        raise ValueError(f"Reply limit {cap} exceeds the configured model limit {bound}")
    result["max_tokens"] = cap
    if reply_tokens is not None or provenance.get("reply_limit", {}).get("value") != cap:
        provenance["reply_limit"] = {"source": source, "value": cap}
    for label, patch in (result.get("reasoning_levels") or {}).items():
        thinking = patch.get("thinking") or {}
        budget = thinking.get("budget_tokens") if isinstance(thinking, dict) else None
        cap_names = {"max_tokens", "max_output_tokens", "max_completion_tokens"} & patch.keys()
        if len(cap_names) > 1:
            raise ValueError(f"Reasoning level {label!r} must set only one reply limit")
        cap_name = next(iter(cap_names), "max_tokens")
        level_cap = patch.get(cap_name, cap)
        if not isinstance(level_cap, int) or isinstance(level_cap, bool) or level_cap <= 0:
            raise ValueError(f"Reasoning level {label!r} has an invalid reply limit")
        if isinstance(budget, int) and not isinstance(budget, bool) and budget >= level_cap:
            level_cap = budget + 1024
            patch[cap_name] = level_cap
            provenance.setdefault("level_reply_limits", {})[label] = {
                "source": "thinking_budget",
                "value": level_cap,
            }
        if bound is not None and level_cap > bound:
            raise ValueError(
                f"Reasoning level {label!r} reply limit exceeds the model limit {bound}"
            )
        if isinstance(window, int) and level_cap >= window:
            raise ValueError(f"Reasoning level {label!r} reply limit leaves no room for input")
    if responses:
        result.setdefault("store", False)
        if not isinstance(result["store"], bool):
            raise ValueError("Responses store must be true or false")
        if result["store"] is False:
            if "include" not in result:
                result["include"] = ["reasoning.encrypted_content"]
                provenance["encrypted_reasoning"] = {"source": "connect", "outcome": "not_probed"}
            elif not isinstance(result["include"], list) or not all(
                isinstance(v, str) for v in result["include"]
            ):
                raise ValueError("include must be a list; use [] to opt out of encrypted reasoning")
            elif result["include"] and "reasoning.encrypted_content" not in result["include"]:
                result["include"].append("reasoning.encrypted_content")
                provenance["encrypted_reasoning"] = {"source": "connect", "outcome": "not_probed"}
    active = result.get("reasoning_level")
    controls = {
        "thinking",
        "reasoning",
        "reasoning_effort",
        "output_config",
        "chat_template_kwargs",
    }
    base = {k: deepcopy(result[k]) for k in controls if k in result}
    extra_controls = {
        k: deepcopy(v) for k, v in (result.get("extra_body") or {}).items() if k in controls
    }
    effective = {**extra_controls, **base}
    if active is not None:
        effective.update((result.get("reasoning_levels") or {})[active])
    snapshot = {"level": active, "settings": effective, "reply_cap": configured_reply_cap(result)}
    provenance["active_reasoning"] = snapshot
    provenance["warnings"] = entry_warnings(result)
    return invalidate_verification(result)


def model_metadata(
    model: str, catalogue: dict | None = None, endpoint_model: dict | None = None
) -> dict:
    """Combine catalogue and endpoint limits for ``model`` into one metadata dict.

    ``catalogue`` is the model's catalogue record and ``endpoint_model`` its
    record from the endpoint's listing. Endpoint values win, but only when
    the endpoint record has the same ID (records without an ID are accepted
    for older callers). ``limit_sources`` records where each limit came from.

    The context length is never computed by adding input and output limits.
    An input-only limit is used as a cautious context bound, not as the
    model's total context size. The endpoint's raw limits are kept in
    ``endpoint_limits``, and its media fields replace the catalogue's.
    """
    result = deepcopy(catalogue or {"id": model})
    sources = result.setdefault("limit_sources", {})
    route_context = (result.get("top_provider") or {}).get("context_length")
    if token_limit(route_context) is not None:
        context = result.get("context_length")
        result["context_length"] = (
            min(context, token_limit(route_context))
            if token_limit(context) is not None
            else token_limit(route_context)
        )
    for field, value in (
        ("context_length", result.get("context_length")),
        ("max_completion_tokens", (result.get("top_provider") or {}).get("max_completion_tokens")),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            sources.setdefault(field, "catalogue")
    endpoint_model = endpoint_model or {}
    # Older callers may omit the ID; a record for a different model is ignored.
    if "id" in endpoint_model and endpoint_model["id"] != model:
        endpoint_model = {}
    reported = {}
    for key in ("context_window", "context_length", "max_input_tokens", "max_output_tokens"):
        value = endpoint_model.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            reported[key] = value
    context = reported.get("context_window", reported.get("context_length"))
    input_limit = reported.get("max_input_tokens")
    if context or input_limit:
        result["context_length"] = min(v for v in (context, input_limit) if v)
        sources["context_length"] = (
            "endpoint_input_limit"
            if input_limit and (context is None or input_limit < context)
            else "endpoint"
        )
    if "max_output_tokens" in reported:
        result["top_provider"] = {
            **(result.get("top_provider") or {}),
            "max_completion_tokens": reported["max_output_tokens"],
        }
        sources["max_completion_tokens"] = "endpoint"
    if reported:
        result["endpoint_limits"] = reported
    # Media fields are stored apart from limits and settings. The endpoint's
    # fields replace the catalogue's only when its record has the same ID,
    # never for another model listed on the same hub.
    media = media_metadata(endpoint_model) if endpoint_model.get("id") == model else {}
    if media:
        for key in ("architecture", "input_modalities", "capabilities"):
            result.pop(key, None)
        result.update(media)
        result["media_source"] = endpoint_model.get("media_source", "endpoint listing")
        result["media_model_id"] = endpoint_model["id"]
    return result


def decision_metadata(
    model: str, catalogue: dict | None = None, endpoint_model: dict | None = None
) -> dict:
    """Return the description and media fields for a decision model.

    ``catalogue`` is an OpenRouter catalogue record the user explicitly chose
    for this decision model; ``endpoint_model`` is used only if its ID equals
    ``model``. Descriptions are never read as capabilities, and no chat
    settings are copied. The source labels (``media_source``) are set here,
    not taken from the records. Endpoint media fields replace the
    catalogue's. Raises ValueError for a missing or invalid record ID.
    """
    result = {}
    for record, source in (
        (catalogue, CATALOGUE_URL),
        (
            endpoint_model if endpoint_model and endpoint_model.get("id") == model else None,
            "endpoint listing",
        ),
    ):
        if not isinstance(record, dict):
            continue
        name = record.get("id")
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 500
            or any(ord(c) < 32 for c in name)
        ):
            raise ValueError("Decision metadata requires a bounded explicit model ID")
        result.setdefault("id", name)
        result.setdefault("media_source", source)
        result.setdefault("media_model_id", name)
        description = record.get("description")
        if (
            isinstance(description, str)
            and len(description) <= 4096
            and not any(ord(c) < 32 for c in description)
        ):
            result["description"] = description
        media = media_metadata(record)
        if media:
            for key in ("architecture", "input_modalities", "capabilities"):
                result.pop(key, None)
            result.update(media)
            result["media_source"] = source
            result["media_model_id"] = name
            result["id"] = name
    return result


def entry_warnings(entry: dict) -> list[str]:
    """Return the warnings to show for ``entry`` (wizard, library and stage reports).

    For decision entries: images enabled where they cannot be sent. For
    other entries: no known context window, or a reply cap that no accepted
    test call has used with the saved settings.
    """
    if entry.get("client_type") == "decision":
        if (
            "image" in entry.get("media_inputs", [])
            and entry.get("api_style") == "systemone"
            and not (
                entry.get("decision_image_transport") == "typesafe_multipart"
                and entry.get("typesafe_danger_zone_images") is True
            )
        ):
            reason = (
                "TypeSafe multipart requires explicit experimental image opt-in before sending"
                if entry.get("decision_image_transport") == "typesafe_multipart"
                else "System One JSON blocks image requests before sending"
            )
            return [
                f"Images are enabled, but {reason}. Use OpenAI Decisions or an approved direct TypeSafe multipart connection, or turn images off."
            ]
        return []
    warnings = (
        []
        if entry.get("context_window")
        else [
            "Context window not listed. NOOA will use its default; supply --context-window or your server model details if you know the limit."
        ]
    )
    cap = entry.get("max_tokens")
    evidence = entry.get("provenance", {})
    records = {**evidence.get("probes", {}), **evidence.get("session_checks", {})}.values()
    if cap and not any(
        r.get("outcome") == "accepted"
        and r.get("tested_reply_tokens") == cap
        and r.get("settings_sent") is True
        for r in records
        if isinstance(r, dict)
    ):
        warnings.append(
            "Reply limit not tested with the saved settings. Run a connection check to test it."
        )
    return warnings


def same_media_route(previous: dict, entry: dict) -> bool:
    """Return True if two entries send media to the same model, endpoint and interface.

    Media permissions are tied to that combination (plus the decision image
    transport settings), not to the API key, so they may be kept when only
    the key or other settings change.
    """

    def route(value):
        style = value.get("api_style") or (
            "responses" if value.get("client_type") == "responses" else "chat"
        )
        endpoint = (
            value.get("endpoint")
            if value.get("client_type") == "decision"
            else value.get("api_base")
        )
        if not endpoint or not value.get("model_name"):
            return None
        endpoint = normalize_endpoint(endpoint)
        if style == "anthropic":
            endpoint = endpoint.removesuffix("/v1")
        return (
            value["model_name"],
            endpoint,
            style,
            value.get("client_type", "responses" if style == "responses" else "completion"),
            value.get("decision_image_transport", "json"),
            value.get("typesafe_danger_zone_images", False),
        )

    old = route(previous)
    return old is not None and old == route(entry)


def configured_reply_cap(entry: dict, level: str | None = None) -> int:
    """Return the reply cap requests use: the level's own cap if it sets one, else ``max_tokens``.

    ``level`` defaults to the entry's selected reasoning level.
    """
    if level is None:
        level = entry.get("reasoning_level")
    settings = (entry.get("reasoning_levels") or {}).get(level, {})
    return next((settings[k] for k in REPLY_CAP_KEYS if k in settings), entry["max_tokens"])


def refresh_plan(proposal: ConnectPlan) -> ConnectPlan:
    """Return ``proposal`` with its test calls rebuilt for the edited entry.

    Re-runs ``configure_entry`` and rebuilds every request body and token
    reservation from the current settings. Every test call uses the saved
    reply cap (or the reasoning level's cap); only the interface test calls,
    made before settings are chosen, use the small ``DISCOVERY_OUTPUT_TOKENS``.
    """
    entry = configure_entry(proposal.entry)
    if entry.get("client_type") == "decision":
        return replace(
            proposal,
            entry=entry,
            token_estimate=sum(probe.token_estimate for probe in proposal.probes),
            price_estimate=None,
            session_checks=False,
        )
    token_key = "max_output_tokens" if entry["api_style"] == "responses" else "max_tokens"
    probes = []
    for probe in proposal.probes:
        body = deepcopy(probe.body)
        if probe.uses_configured_cap:
            level = (
                probe.name[6:] if probe.name.startswith("level:") else entry.get("reasoning_level")
            )
            settings = (entry.get("reasoning_levels") or {}).get(level, {})
            # The body may still hold settings of a previously selected level.
            # Remove every field any level (or the old selection) sets before
            # applying the new one; e.g. switching from adaptive to Off must
            # not keep the old effort block or thinking budget.
            controlled = {
                key for patch in (entry.get("reasoning_levels") or {}).values() for key in patch
            }
            controlled.update(
                (proposal.entry.get("provenance") or {})
                .get("active_reasoning", {})
                .get("settings", {})
            )
            for key in controlled:
                body.pop(key, None)
            body.update(settings)
            cap = configured_reply_cap(
                entry, probe.name[6:] if probe.name.startswith("level:") else None
            )
            for k in REPLY_CAP_KEYS:
                body.pop(k, None)
            body[token_key] = cap
            for k in ("store", "include"):
                if k in entry:
                    body[k] = deepcopy(entry[k])
            probe = replace(
                probe, body=body, token_estimate=probe.input_reservation + cap, timeout_seconds=120
            )
        probes.append(probe)
    # Skipped test calls keep their full reservations and caps in the plan, so
    # if a skip is later removed the call still goes through the budget and
    # approved-cap checks.
    estimate = sum(p.token_estimate for p in probes if not p.skip_reason)
    if proposal.session_checks:
        from ._session import reply_budget

        estimate += 3 * reply_budget(entry, max(0, proposal.budget_tokens - estimate))[2]
    return replace(
        proposal,
        entry=entry,
        probes=tuple(probes),
        token_estimate=estimate,
        price_estimate=proposal.price_estimate if tuple(probes) == proposal.probes else None,
    )


REASONING_SELECTION_UNSET = object()


def plan(
    alias: str,
    model: str,
    api_style: str,
    api_base: str,
    api_key_env: str,
    *,
    catalogue: dict | None = None,
    catalogue_settings: bool = False,
    endpoint_model: dict | None = None,
    endpoint_models_url: str | None = None,
    context_window: int | None = None,
    reasoning_levels: dict | None = None,
    reasoning_level=REASONING_SELECTION_UNSET,
    budget_tokens: int = DEFAULT_CHECK_BUDGET,
    output_tokens: int = DISCOVERY_OUTPUT_TOKENS,
    reasoning_output_tokens: int = DEFAULT_REASONING_OUTPUT_TOKENS,
    existing_entry: dict | None = None,
    session_checks: bool = False,
    reply_tokens: int | None = None,
    cache_enabled: bool | None = None,
    structured_output_mode=STRUCTURED_OUTPUT_UNSET,
    media_inputs: list[str] | tuple[str, ...] | None = None,
    test_media_inputs: list[str] | tuple[str, ...] | None = None,
    decision_image_transport: str | None = None,
    typesafe_danger_zone_images: bool | None = None,
    decision_checks: bool = True,
    checked_levels: list[str] | tuple[str, ...] | None = None,
) -> ConnectPlan:
    """Build the model entry and the test calls for the user to approve.

    Reads no API key, file or network resource; ``run`` sends the calls.
    Returns a ``ConnectPlan``. Raises ValueError for invalid arguments.

    Model and endpoint:

    - ``model`` is the model ID exactly as the endpoint knows it, without a
      LiteLLM provider prefix; the returned entry adds the prefix the runtime
      currently needs (``openai/`` or ``anthropic/``).
    - ``endpoint_model`` is the model's record from the endpoint's listing and
      ``endpoint_models_url`` the URL it was read from (default
      ``<api_base>/models``). Only a record whose ID equals ``model`` counts.
    - ``catalogue`` is the model's catalogue record. A record from the
      OpenRouter catalogue only supplies reasoning levels, default reply
      cap, prices and OpenRouter-specific limits when the entry actually
      points at OpenRouter for the same model ID. ``catalogue_settings=True``
      marks a catalogue dict written by the user (e.g. in a settings editor),
      which is used as given. For decision styles, ``catalogue`` is an
      OpenRouter record the user mapped to the decision model; only its
      description and media fields are kept, never chat settings.
    - ``context_window`` is the user's declared context window, lowered to
      the endpoint's limit if smaller (see ``apply_context_window``). It does
      not set the reply cap.

    Settings (for each, "not given" keeps the value of ``existing_entry`` when
    it points at the same model, endpoint and interface):

    - ``reasoning_level``: the reasoning level selected for use (not
      ``reasoning_default``, which only records the provider's default); None
      clears it. ``checked_levels`` limits the level test calls to these
      labels (for example only the chosen level); None tests every level.
    - ``structured_output_mode``: "anthropic" uses Anthropic's
      ``output_config.format`` for typed output on Completion chat/anthropic
      entries; None and "auto" keep the older JSON output method. Native
      support is never guessed, except that a listing record from the same
      Anthropic endpoint reporting structured output support makes
      "anthropic" the recommended default. The saved value is also kept only
      when the API key variable is the same.
    - ``cache_enabled``: False saves ``cache_breakpoint: null`` (no explicit
      cache options or markers; the provider's automatic caching still
      applies); True selects "auto". This does not test whether caching works.
    - ``media_inputs`` are the media kinds to permit; ``test_media_inputs``
      the kinds to test with generated attachments. Media fields from
      listings only produce suggestions and never change these settings.
      Media kinds an interface cannot send are planned as skipped.

    Budget: each test call reserves its reply cap plus an input estimate of
    512 tokens (4,096 for media test calls). Usage reported by the server can
    add to that. Servers may ignore caps, so the budget is not a billing limit.
    """
    if cache_enabled is not None and not isinstance(cache_enabled, bool):
        raise ValueError("cache_enabled must be a boolean or None")
    if api_style in DECISION_API_STYLES and reasoning_level is not REASONING_SELECTION_UNSET:
        raise ValueError("Decision models do not configure chat reasoning")
    if api_style in DECISION_API_STYLES and cache_enabled is not None:
        raise ValueError("Decision models do not configure chat caching")
    _validate_structured_output({"api_style": api_style}, structured_output_mode)
    if existing_entry is not None:
        _validate_structured_output(existing_entry)
    if api_style not in _PATHS:
        raise ValueError("api_style must be chat, responses, anthropic, systemone or openai")
    api_base = normalize_endpoint(api_base)
    if (
        not catalogue_settings
        and catalogue is not None
        and "id" in catalogue
        and (api_base + "/models" != CATALOGUE_URL or catalogue["id"] != model)
        and api_style not in DECISION_API_STYLES
    ):
        # An OpenRouter record describing a different endpoint or model may
        # supply general facts, but not its routing-specific settings. Dicts
        # without an ID are older hand-written settings and are used as given.
        catalogue = deepcopy(catalogue)
        for field in ("reasoning", "default_parameters", "pricing", "top_provider"):
            catalogue.pop(field, None)
        # The general context length is still used as a fallback, but the
        # output limit of OpenRouter's provider applies only via OpenRouter.
        if isinstance(catalogue.get("limit_sources"), dict):
            catalogue["limit_sources"].pop("max_completion_tokens", None)
    if endpoint_models_url is not None:
        endpoint_models_url = validate_models_url(endpoint_models_url)
    if (
        not alias.strip()
        or not model.strip()
        or api_key_env
        and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env)
    ):
        raise ValueError(
            "An alias and model ID are required; api_key_env must be empty or a variable name"
        )
    if not 1 <= output_tokens <= 4096 or budget_tokens < 1:
        raise ValueError("output_tokens must be 1..4096 and budget_tokens must be positive")
    if not 1 <= reasoning_output_tokens <= 32768:
        raise ValueError("reasoning_output_tokens must be 1..32768")
    from nooa.unifiedllm.media_inputs import validate_media_inputs

    tested_media = validate_media_inputs(test_media_inputs)
    decision_facts = (
        decision_metadata(model, catalogue, endpoint_model)
        if api_style in DECISION_API_STYLES
        else None
    )
    selected_media_model = (
        decision_facts
        if api_style in DECISION_API_STYLES
        else (
            endpoint_model
            if endpoint_model is not None
            and endpoint_model.get("id") == model
            and media_metadata(endpoint_model)
            else catalogue
        )
    )
    endpoint_media_selected = (
        endpoint_model is not None
        and endpoint_model.get("id") == model
        and bool(media_metadata(endpoint_model))
    )
    if endpoint_media_selected:
        # Set the source labels here; labels already in the record are not
        # trusted as coming from the listing.
        selected_media_model = {
            **selected_media_model,
            "media_model_id": model,
            "media_source": endpoint_models_url or "endpoint listing",
        }
    media_suggestions = advertised_media_suggestions(
        selected_media_model,
        source=(selected_media_model or {}).get(
            "media_source",
            "endpoint listing"
            if endpoint_media_selected
            else CATALOGUE_URL
            if selected_media_model
            else "unknown",
        ),
        api_style=api_style,
        decision_image_transport=decision_image_transport
        or (existing_entry or {}).get("decision_image_transport", "json"),
        typesafe_danger_zone_images=typesafe_danger_zone_images
        if typesafe_danger_zone_images is not None
        else (existing_entry or {}).get("typesafe_danger_zone_images", False),
    )
    if api_style in DECISION_API_STYLES:
        if reasoning_levels:
            raise ValueError("Decision models do not configure reasoning levels")
        if session_checks:
            raise ValueError("Decision models are stateless and do not use session checks")
        if reply_tokens is not None:
            raise ValueError("Decision models do not use reply-token budgets")
        from nooa.decisions.media import decision_media_inputs

        decision_media_inputs(tested_media)
        decision_media_inputs(media_inputs)
        entry = {
            "model_name": model,
            "client_type": "decision",
            "media_inputs": list(decision_media_inputs(media_inputs)),
            "decision_image_transport": decision_image_transport or "json",
            "typesafe_danger_zone_images": typesafe_danger_zone_images
            if typesafe_danger_zone_images is not None
            else False,
            "api_style": api_style,
            "endpoint": api_base,
            "api_key_env": api_key_env,
            "provenance": {
                "media_suggestions": media_suggestions,
                "probes": {},
                "requests_accepted": [],
                "not_probed": [
                    "context_window",
                    "reasoning",
                    "tools",
                    "sessions",
                    "reply_limit",
                ],
            },
        }
        if catalogue is not None:
            entry["provenance"]["catalogue"] = {
                "url": CATALOGUE_URL,
                "id": catalogue["id"],
                "model_id": model,
            }
            entry["provenance"]["catalogue_metadata"] = decision_metadata(model, catalogue)
        if decision_facts and "description" in decision_facts:
            entry["provenance"]["model_description"] = decision_facts["description"]
        # Keep the old permissions and settings only if the existing entry
        # uses the same model, endpoint and interface.
        if existing_entry:
            candidate = {**entry}
            if decision_image_transport is None:
                candidate["decision_image_transport"] = existing_entry.get(
                    "decision_image_transport", "json"
                )
            if typesafe_danger_zone_images is None:
                candidate["typesafe_danger_zone_images"] = existing_entry.get(
                    "typesafe_danger_zone_images", False
                )
            if same_media_route(existing_entry, candidate):
                entry = candidate
                if media_inputs is None:
                    entry["media_inputs"] = list(
                        decision_media_inputs(existing_entry.get("media_inputs"))
                    )
        entry["provenance"]["media_suggestions"] = advertised_media_suggestions(
            selected_media_model,
            source=media_suggestions["source"],
            api_style=api_style,
            decision_image_transport=entry["decision_image_transport"],
            typesafe_danger_zone_images=entry["typesafe_danger_zone_images"],
        )
        if (
            tested_media
            and api_style == "systemone"
            and not (
                entry["decision_image_transport"] == "typesafe_multipart"
                and entry["typesafe_danger_zone_images"] is True
            )
        ):
            raise ValueError(
                "Image testing isn't available with System One JSON. Use --decision-image-transport typesafe_multipart and --typesafe-danger-zone-images for an approved direct TypeSafe connection, or leave images off"
            )
        if endpoint_model is not None and endpoint_model.get("id") == model:
            # Keep the listing record for reference only; it never becomes
            # chat settings.
            entry["provenance"]["advertised_model"] = deepcopy(endpoint_model)
        from ._decision import planned_body

        body = {"model": model, **planned_body("routing")}
        probes = [Probe("routing", body, 512, uses_configured_cap=False)]
        if decision_checks:
            probes.extend(
                Probe(name, planned_body(name), 512, uses_configured_cap=False)
                for name in ("decision:typed", "decision:positive", "decision:negative")
            )
        if tested_media:
            probes.append(
                Probe(
                    "media:image",
                    planned_body("media:image"),
                    4096,
                    uses_configured_cap=False,
                    input_reservation=4096,
                )
            )
        proposal = ConnectPlan(
            alias,
            configure_entry(entry),
            tuple(probes),
            budget_tokens,
            sum(p.token_estimate for p in probes),
            None,
        )
        if existing_entry:
            keys = (
                "model_name",
                "endpoint",
                "api_key_env",
                "client_type",
                "api_style",
                "decision_image_transport",
                "typesafe_danger_zone_images",
            )
            if all(existing_entry.get(key) == proposal.entry.get(key) for key in keys):
                previous = existing_entry.get("provenance", {}).get("probes", {})
                if "routing" in previous:
                    proposal.entry["provenance"]["probes"]["routing"] = deepcopy(
                        previous["routing"]
                    )
        return proposal
    if decision_image_transport is not None or typesafe_danger_zone_images is not None:
        raise ValueError("Decision image transport options require a decision interface")
    vendor = "anthropic" if api_style == "anthropic" else "openai"
    entry: dict[str, Any] = {
        "model_name": f"{vendor}/{model}",
        "media_inputs": list(validate_media_inputs(media_inputs)),
        "client_type": "responses" if api_style == "responses" else "completion",
        "api_style": api_style,
        # The Anthropic client appends /v1/messages itself, so store the root;
        # Chat and Responses clients expect the API base including /v1.
        "api_base": api_base.removesuffix("/v1") if api_style == "anthropic" else api_base,
        "api_key_env": api_key_env,
    }
    if media_inputs is None and existing_entry and same_media_route(existing_entry, entry):
        entry["media_inputs"] = list(validate_media_inputs(existing_entry.get("media_inputs")))
    if existing_entry and same_media_route(existing_entry, entry):
        if "cache_breakpoint" in existing_entry:
            entry["cache_breakpoint"] = deepcopy(existing_entry["cache_breakpoint"])
        if (
            existing_entry.get("api_key_env", "") == entry.get("api_key_env", "")
            and "structured_output_mode" in existing_entry
        ):
            entry["structured_output_mode"] = existing_entry["structured_output_mode"]
    provenance: dict[str, Any] = {
        "media_suggestions": media_suggestions,
        "probes": {},
        "requests_accepted": [],
        "reasoning_observed": [],
        "not_probed": [
            "underlying_model",
            "context_window",
            "max_output_tokens",
            "reasoning_default",
            "cache_breakpoint",
            "encrypted_reasoning",
        ],
    }
    if catalogue is not None or endpoint_model is not None:
        catalogue = model_metadata(model, catalogue, endpoint_model)
        provenance["limit_sources"] = deepcopy(catalogue.get("limit_sources", {}))
        if catalogue.get("endpoint_limits"):
            if (endpoint_model or {}).get("id") == model:
                provenance["endpoint_model_id"] = model
            provenance["endpoint_limits"] = {
                **catalogue["endpoint_limits"],
                "url": endpoint_models_url or api_base + "/models",
            }
            if (endpoint_model or {}).get("id") == model:
                provenance["endpoint_limit_scope"] = effective_route_scope(entry)
        if any(
            v == "catalogue" for v in catalogue.get("limit_sources", {}).values()
        ) or catalogue.get("reasoning"):
            provenance["catalogue"] = {"url": CATALOGUE_URL, "id": catalogue["id"]}
        for field, value in (
            ("context_window", catalogue.get("context_length")),
            (
                "max_output_tokens",
                (catalogue.get("top_provider") or {}).get("max_completion_tokens"),
            ),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                entry[field] = value
    reasoning = (catalogue or {}).get("reasoning") or {}
    if reasoning_levels is None and reasoning.get("supported_efforts"):
        template = "adaptive" if api_style == "anthropic" else "effort"
        reasoning_levels = {
            label: reasoning_settings(template, api_style, label)
            for label in reasoning["supported_efforts"]
        }
        provenance["reasoning_levels"] = {"source": "catalogue", "template": template}
    if reasoning_levels is not None and not isinstance(reasoning_levels, dict):
        raise ValueError("reasoning_levels must be a mapping of labels to request settings")
    levels = deepcopy(reasoning_levels or {})
    for label, params in levels.items():
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(params, dict)
            or not params
        ):
            raise ValueError(
                "Reasoning levels must map string labels to request dictionaries; quote YAML boolean labels"
            )
        if any(
            not isinstance(key, str)
            or key in _RESERVED
            or key.startswith("reasoning_")
            and key != "reasoning_effort"
            for key in params
        ):
            raise ValueError(
                "Reasoning levels cannot set routing, credentials or framework declarations"
            )
    if reasoning_levels is not None:
        entry["reasoning_levels"] = levels
        # The user asked for these fields; they are not guessed capabilities.
        # Allow them explicitly so LiteLLM's parameter filtering does not drop
        # them for a model missing from its own model list.
        entry["allowed_openai_params"] = sorted({key for patch in levels.values() for key in patch})
    if (default := reasoning.get("default_effort")) in levels:
        entry["reasoning_default"] = default
    if existing_entry and same_media_route(existing_entry, entry):
        if reasoning_levels is None and "reasoning_levels" in existing_entry:
            entry["reasoning_levels"] = deepcopy(existing_entry["reasoning_levels"])
            levels = entry["reasoning_levels"] or {}
        if existing_entry.get("reasoning_default") in levels:
            entry["reasoning_default"] = existing_entry["reasoning_default"]
        if existing_entry.get("reasoning_level") in levels:
            entry["reasoning_level"] = existing_entry["reasoning_level"]
        retained = existing_entry.get("allowed_openai_params") or []
        if retained:
            entry["allowed_openai_params"] = sorted(
                set(entry.get("allowed_openai_params", [])) | set(retained)
            )
    # Levels kept from the existing entry need the same allowed parameters as
    # new ones; otherwise their settings could be filtered out.
    if levels:
        entry["allowed_openai_params"] = sorted(
            set(entry.get("allowed_openai_params", []))
            | {key for patch in levels.values() for key in patch}
        )
    if reasoning_level is not REASONING_SELECTION_UNSET:
        if reasoning_level is None:
            entry.pop("reasoning_level", None)
        else:
            entry["reasoning_level"] = reasoning_level
    if api_style == "responses":
        entry["store"] = False
        entry["include"] = ["reasoning.encrypted_content"]
        provenance["encrypted_reasoning"] = {"source": "connect", "outcome": "not_probed"}
        if (
            existing_entry
            and same_media_route(existing_entry, entry)
            and existing_entry.get("api_key_env") == entry.get("api_key_env")
        ):
            # Keep the user's store/include values; saves drop provenance, so
            # the values themselves must be carried over.
            for field in ("store", "include"):
                if field in existing_entry:
                    entry[field] = deepcopy(existing_entry[field])
            previous = existing_entry.get("provenance", {}).get("encrypted_reasoning")
            if entry["include"] == [] and previous and previous.get("outcome") == "rejected":
                provenance["encrypted_reasoning"] = deepcopy(previous)
    token_key = "max_output_tokens" if api_style == "responses" else "max_tokens"
    body = {
        "model": model,
        token_key: output_tokens,
        "input" if api_style == "responses" else "messages": [
            {"role": "user", "content": "Compute 17 * 19. Reply with the number."}
        ],
    }
    if api_style == "responses":
        body["store"] = entry["store"]
        body["include"] = list(entry["include"])
    probes = [Probe("routing", body, output_tokens + 512)]
    tool_body = deepcopy(body)
    tool_body["input" if api_style == "responses" else "messages"][0]["content"] = (
        "Call probe_tool with value 'ok'."
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    function = {"name": "probe_tool", "description": "Echo a value", "parameters": schema}
    tool_body["tools"] = (
        [{"name": "probe_tool", "description": "Echo a value", "input_schema": schema}]
        if api_style == "anthropic"
        else [{"type": "function", **function}]
        if api_style == "responses"
        else [{"type": "function", "function": function}]
    )
    probes.append(Probe("tools", tool_body, output_tokens + 512))
    for label, params in levels.items():
        if checked_levels is not None and label not in checked_levels:
            continue
        level_body = deepcopy(body)
        level_body[token_key] = reasoning_output_tokens
        level_body["input" if api_style == "responses" else "messages"][0]["content"] = (
            REASONING_CHECK_PROMPT
        )
        level_body.update(params)
        probes.append(Probe(f"level:{label}", level_body, reasoning_output_tokens + 512, 120))
    from ._media import media_content

    for kind in tested_media:
        content, expected, skip = media_content(kind, api_style)
        media_body = deepcopy(body)
        media_body["input" if api_style == "responses" else "messages"][0]["content"] = (
            content or ""
        )
        probes.append(
            Probe(
                f"media:{kind}",
                media_body,
                output_tokens + 4096,
                input_reservation=4096,
                expected_answer=expected,
                skip_reason=skip,
            )
        )
    entry["provenance"] = provenance
    if context_window is not None:
        entry = apply_context_window(
            entry, context_window, model=model, endpoint_model=endpoint_model
        )
        provenance = entry["provenance"]
    if reply_tokens is None:
        recommendation = ((catalogue or {}).get("default_parameters") or {}).get("max_tokens")
        if (
            isinstance(recommendation, int)
            and not isinstance(recommendation, bool)
            and recommendation > 0
        ):
            ceiling = entry.get("max_output_tokens")
            window = entry.get("context_window")
            entry["max_tokens"] = min(
                [recommendation] + [v for v in (ceiling, window) if isinstance(v, int) and v > 0]
            )
            provenance["reply_limit"] = {
                "source": "catalogue_recommendation",
                "value": entry["max_tokens"],
            }
    # If the Anthropic endpoint's own listing says this model supports
    # structured output, recommend Anthropic's native format. Other gateways
    # and explicitly given values (including None/"auto") are left alone.
    from ._reasoning import native_metadata_matches

    native_format = (
        structured_output_mode is STRUCTURED_OUTPUT_UNSET
        and "structured_output_mode" not in entry
        and native_metadata_matches(
            endpoint_model,
            model_id=model,
            api_style=api_style,
            source=endpoint_models_url,
            endpoint=api_base,
        )
        and endpoint_model.get("capabilities", {}).get("structured_outputs", {}).get("supported")
        is True
    )
    if native_format:
        structured_output_mode = "anthropic"
    entry = configure_entry(
        entry,
        reply_tokens=reply_tokens,
        cache_enabled=cache_enabled,
        structured_output_mode=structured_output_mode,
    )
    if native_format:
        entry["provenance"]["structured_output_profile"].update(
            source=endpoint_models_url,
            model_id=model,
            field="data[].capabilities.structured_outputs.supported",
            recommended=True,
        )
    provenance = entry["provenance"]
    for probe in probes:
        if probe.name.startswith("level:"):
            probe.body.update(entry["reasoning_levels"][probe.name.removeprefix("level:")])
    if existing_entry:
        # Reuse earlier results when the connection is unchanged; run_steps
        # also compares each result with its exact request. Adding a level
        # should not discard unchanged routing or tool results.
        keys = (
            "model_name",
            "api_base",
            "api_key_env",
            "api_style",
            "client_type",
            "structured_output_mode",
            "reasoning_level",
            "reasoning_levels",
        )
        if all(existing_entry.get(key) == entry.get(key) for key in keys) and existing_entry.get(
            "cache_breakpoint", "auto"
        ) == entry.get("cache_breakpoint", "auto"):
            previous = existing_entry.get("provenance", {}).get("probes", {})
            provenance["probes"] = {
                probe.name: deepcopy(previous[probe.name])
                for probe in probes
                if probe.name in previous
            }
    refreshed = refresh_plan(ConnectPlan(alias, entry, tuple(probes), budget_tokens, 0, None))
    entry, probes = refreshed.entry, refreshed.probes
    estimate = refreshed.token_estimate
    price = None
    if catalogue and catalogue.get("pricing"):
        try:
            rates = [float(catalogue["pricing"][key]) for key in ("prompt", "completion")]
            if all(math.isfinite(rate) and rate >= 0 for rate in rates):
                price = (
                    sum(probe.input_reservation for probe in probes if not probe.skip_reason)
                    * rates[0]
                    + sum(
                        probe.token_estimate - probe.input_reservation
                        for probe in probes
                        if not probe.skip_reason
                    )
                    * rates[1]
                )
        except (ValueError, TypeError, KeyError):
            pass  # Invalid prices mean "unknown", never zero.
    if session_checks:
        from nooa.unifiedllm.connect._session import reply_budget

        estimate += 3 * reply_budget(entry, max(0, budget_tokens - estimate))[2]
        price = None  # Session cost depends on the first reply's length.
    return ConnectPlan(alias, entry, tuple(probes), budget_tokens, estimate, price, session_checks)


async def _run_probe(alias: str, entry: dict, probe: Probe, api_key: str | None):
    """Send one test call for the unsaved ``entry``; return its probe record.

    The client is built by ``client_from_config``, the same factory agents use,
    so the test exercises the real request path. Retries are off and the
    probe's timeout applies; probes that use the configured cap send the
    saved reply cap. Runtime modules are imported only here so that
    ``discover`` and ``plan`` stay lightweight. Tool calls in the reply are
    inspected as data; Connect never executes them.
    """
    from nooa.unifiedllm import RetryConfig, Tool
    from nooa.unifiedllm.http_config import HttpConfig
    from nooa.unifiedllm.registry import client_from_config

    def probe_tool(value: str):
        # Offered to the model, but Connect never runs tools during tests.
        raise RuntimeError("Setup checks never execute tools")

    params = deepcopy(probe.body)
    params.pop("model")  # The entry selects the model, not the planned body.
    messages = params.pop("input" if entry["api_style"] == "responses" else "messages")
    if params.pop("tools", None):
        params["tools"] = [Tool(name="probe_tool", description="Echo a value", callable=probe_tool)]
    if probe.name.startswith("level:"):
        label = probe.name.removeprefix("level:")
        settings = entry["reasoning_levels"][label]
        if any(
            probe.body.get(key) != value
            for key, value in settings.items()
            if key not in REPLY_CAP_KEYS
        ):
            raise ValueError("Reasoning settings changed after planning; make a new plan")
        for key in settings:
            params.pop(key, None)
        if {"max_tokens", "max_completion_tokens", "max_output_tokens"} & settings.keys():
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                params.pop(key, None)
        # The client applies the level's settings (which may set the cap), so
        # no conflicting value is passed here.
        params["reasoning_level"] = label
    elif selected := entry.get("reasoning_level"):
        settings = (entry.get("reasoning_levels") or {}).get(selected, {})
        for key in settings:
            params.pop(key, None)
        if REPLY_CAP_KEYS & settings.keys():
            for key in REPLY_CAP_KEYS:
                params.pop(key, None)
    client = cast(
        Any,
        client_from_config(
            alias,
            {**entry, "media_inputs": [probe.name[6:]]}
            if probe.name.startswith("media:")
            else entry,
            api_key=api_key,
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
            num_retries=0,
            timeout=probe.timeout_seconds,
            http_config=HttpConfig(read_timeout=probe.timeout_seconds),
        ),
    )
    typed_media_check = probe.name.startswith("media:")
    typed = (probe.name == "routing" and probe.uses_configured_cap) or typed_media_check
    settings_sent = []
    attachments_sent = []
    output_settings_sent = []
    response_status = []
    reply_evidence = {}
    schema_carrier = None
    schema_completed = False
    schema_value = None
    send_started = False
    native = typed and entry.get("structured_output_mode") == "anthropic"
    expected_format = None
    carrier_evidence = {}

    def capture_output_model(output_model):
        nonlocal expected_format
        from nooa.unifiedllm.structured_output import anthropic_output_format

        # The JSON schema format Predict actually generated, including strict
        # fields and titles, to compare with what was sent.
        expected_format = anthropic_output_format(output_model)

    async def capture_status(response):
        nonlocal schema_completed, schema_value
        response_status.append(response.status_code)
        if (typed_media_check or native) and response.is_success:
            # Read usage and finish facts from the raw response here, so they
            # are kept even if the typed result fails validation later. The
            # reply text is neither stored nor judged. The schema used is only
            # compared with the exact requested output format; Predict still
            # has to validate and return the typed result itself.
            await response.aread()
            try:
                body = response.json()
                from nooa.llm_types import LLMUsage

                usage = LLMUsage.from_provider(body.get("usage"))
                if usage:
                    reply_evidence.update(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        reasoning_tokens=usage.reasoning_tokens,
                        reported_tokens=usage.input_tokens + usage.output_tokens,
                    )
                if "choices" in body:
                    choice = body["choices"][0]
                    finish = choice.get("finish_reason")
                    final = choice.get("message", {}).get("content")
                elif body.get("type") == "message":
                    blocks = body.get("content", [])
                    carriers = [b for b in blocks if b.get("type") == "tool_use"]
                    schema_completed = bool(
                        schema_carrier
                        and body.get("stop_reason") == "tool_use"
                        and len(carriers) == 1
                        and carriers[0].get("name") == schema_carrier
                        and isinstance(carriers[0].get("input"), dict)
                        and set(carriers[0]["input"]) == {"value"}
                        and type(carriers[0]["input"]["value"]) is bool
                    )
                    if schema_completed:
                        schema_value = carriers[0]["input"]["value"]
                    finish = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                    }.get(body.get("stop_reason"), "error")
                    final = "".join(
                        b.get("text", "")
                        for b in body.get("content", [])
                        if b.get("type") == "text"
                    )
                elif body.get("object") == "response":
                    finish = (
                        "stop"
                        if body.get("status") == "completed"
                        else "length"
                        if body.get("status") == "incomplete"
                        else "error"
                    )
                    final = "".join(
                        b.get("text", "")
                        for m in body.get("output", [])
                        if m.get("type") == "message" and m.get("status") == "completed"
                        for b in m.get("content", [])
                        if b.get("type") == "output_text"
                    )
                else:
                    raise ValueError("Unrecognized response envelope")
                reply_evidence["finish_reason"] = (
                    finish
                    if finish in {"stop", "length", "tool_calls", "content_filter"}
                    else "error"
                )
                reply_evidence["final_answer_observed"] = isinstance(final, str) and bool(
                    final.strip()
                )
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                pass

    async def capture(request):
        nonlocal send_started, schema_carrier
        from nooa.unifiedllm.connect._session import settings_on_wire

        if typed and send_started:
            raise RuntimeError("Configured typed check permits only one outbound send")
        send_started = True

        # Work out the settings this request should carry, the same way the
        # runtime does when sending (including the selected level and
        # extra_body merged into the top level). Typed calls use only the
        # client's defaults; other test calls add their planned parameters.
        resolved = client._prepare_call_config({} if typed else params)
        selected_level = params.get(
            "reasoning_level", entry.get("reasoning_level", client.reasoning_level)
        )
        selected_settings = (entry.get("reasoning_levels") or {}).get(selected_level, {})
        control_keys = {
            "temperature",
            "top_p",
            "reasoning",
            "reasoning_effort",
            "thinking",
            "chat_template_kwargs",
            "output_config",
            "store",
            "include",
            *REPLY_CAP_KEYS,
        } | (selected_settings.keys() - {"extra_body"})
        expected = {
            **{key: entry[key] for key in control_keys - REPLY_CAP_KEYS if key in entry},
            **{key: resolved[key] for key in control_keys if key in resolved},
            # If a buggy adapter drops a selected level setting while
            # resolving, it must still be expected in the sent request.
            **{
                key: resolved.get(key, value)
                for key, value in selected_settings.items()
                if key not in REPLY_CAP_KEYS and key != "extra_body"
            },
            **(resolved.get("extra_body") or {}),
        }
        wire = json.loads(request.content)
        if typed_media_check and entry["api_style"] == "anthropic" and not native:
            tools = wire.get("tools", [])
            choice = wire.get("tool_choice", {})
            if len(tools) == 1 and choice.get("type") == "tool":
                tool = tools[0]
                schema = tool.get("input_schema", {})
                if (
                    tool.get("name") == choice.get("name")
                    and schema.get("type") == "object"
                    and schema.get("required") == ["value"]
                    and set(schema.get("properties", {})) == {"value"}
                    and schema["properties"]["value"].get("type") == "boolean"
                ):
                    schema_carrier = tool["name"]
        # Responses SDKs leave out an empty include list. A missing include
        # then means the same as [], but a non-empty include still fails.
        controls_wire = (
            {**wire, "include": []}
            if expected.get("include") == [] and "include" not in wire
            else wire
        )
        observed = settings_on_wire(expected, controls_wire)
        if native:
            output = wire.get("output_config")
            exact = (
                expected_format is not None
                and isinstance(output, dict)
                and output.get("format") == expected_format
            )
            # Typed Predict calls send no tools, so tools/tool_choice here would
            # mean the schema was sent as a tool instead of natively. (The tools
            # test call has no output model and is not checked here.) Format
            # fields left inside extra_body are not the native request shape.
            conflicts = any(
                k in wire
                for k in ("response_format", "text.format", "output_format", "tools", "tool_choice")
            )
            conflicts |= isinstance(wire.get("text"), dict) and "format" in wire["text"]
            conflicts |= isinstance(wire.get("extra_body"), dict) and any(
                k in wire["extra_body"]
                for k in (
                    "output_config",
                    "output_format",
                    "response_format",
                    "tools",
                    "tool_choice",
                    "text",
                )
            )
            carrier_evidence.update(
                structured_output_mode="anthropic",
                structured_output_carrier="output_config.format" if exact else "unconfirmed",
                structured_output_schema_observed=bool(exact),
                structured_output_conflicts_observed=bool(conflicts),
            )
            observed = observed and exact and not conflicts
        output_settings_sent.append(observed)
        if probe.name.startswith("media:"):
            from ._media import media_on_wire

            attachment = media_on_wire(probe.name[6:], probe.body, wire)
            attachments_sent.append(attachment)
            observed = observed and attachment
        settings_sent.append(observed)

    hooks = client._http.httpx_async.event_hooks["request"]
    response_hooks = client._http.httpx_async.event_hooks["response"]
    hooks.append(capture)
    response_hooks.append(capture_status)
    try:
        params["timeout"] = probe.timeout_seconds
        if typed:
            from ._baseline import typed_baseline

            resolved = client._prepare_call_config({})
            effective = {**resolved, **(resolved.get("extra_body") or {})}
            caps = [effective[k] for k in REPLY_CAP_KEYS if k in effective]
            if caps != [configured_reply_cap(entry)]:
                raise ValueError("Configured baseline reply cap differs from the approved cap")
            if effective.get("stream") or any(effective.get(k, 1) != 1 for k in ("n", "best_of")):
                raise ValueError("Configured baseline requires one non-streaming generation")
            if typed_media_check:
                from ._typed_media import typed_media

                response, result = await typed_media(
                    client,
                    probe.name[6:],
                    **({"observe_output_model": capture_output_model} if native else {}),
                )
            else:
                response, result = await typed_baseline(
                    client, **({"observe_output_model": capture_output_model} if native else {})
                )
            response.parsed = result  # Typed result, used here only; never saved.
        else:
            response = await client.acall(messages=messages, **params)
        if (
            typed_media_check
            and schema_completed
            and type(response.parsed) is bool
            and response.parsed is schema_value
            and not response.tool_calls
            and response.finish_reason == "stop"
        ):
            # A tool call carrying exactly the requested schema output is the
            # typed answer, not a tool request, so count it as a final answer.
            # Other test calls are not affected.
            reply_evidence.update(finish_reason="stop", final_answer_observed=True)
        transport = getattr(client, "transport", "litellm")
        observed = (len(settings_sent) == 1 and all(settings_sent)) if settings_sent else None
        response.metadata["connect_evidence"] = {
            **reply_evidence,
            **carrier_evidence,
            "request_accepted": bool(response_status and 200 <= response_status[-1] < 300),
            "attachment_observed": len(attachments_sent) == 1 and all(attachments_sent),
            "output_settings_observed": len(output_settings_sent) == 1
            and all(output_settings_sent),
        }
        return response, observed, transport
    except Exception as exc:
        exc._connect_evidence = {
            **reply_evidence,
            **carrier_evidence,
            "settings_sent": (len(settings_sent) == 1 and all(settings_sent))
            if settings_sent
            else None,
            "request_accepted": bool(response_status and 200 <= response_status[-1] < 300),
            "attachment_observed": len(attachments_sent) == 1 and all(attachments_sent),
            "output_settings_observed": len(output_settings_sent) == 1
            and all(output_settings_sent),
        }
        if response_status:
            # LiteLLM can report a malformed HTTP 200 response as a 422 error,
            # so also record the status the server actually returned.
            exc._connect_http_status = response_status[-1]  # type: ignore[attr-defined]
        raise
    finally:
        hooks.remove(capture)
        response_hooks.remove(capture_status)
        await client.aclose()


async def _run_decision_probe(alias: str, entry: dict, probe: Probe, api_key: str | None):
    """Send one decision test call; return ``(response, http_status, facts)``.

    Uses an own httpx client whose hooks allow only one request, read token
    usage from the raw response, and, for the image test, check that the
    generated PNG really was in the request (as an ``input_image`` data URL
    for OpenAI Decisions, or a multipart part with the experimental-image
    header for TypeSafe). On failure the status, usage and attachment facts
    are attached to the exception for the caller; nothing else is kept.
    """
    import json
    from email import policy
    from email.parser import BytesParser

    from nooa.unifiedllm import RetryConfig
    from nooa.unifiedllm.registry import client_from_config

    from ._decision import typed_decision
    from ._media import png_fixture

    statuses = []
    observed_usage = {}
    wire = {"attachment_observed": False}
    sent = False

    async def capture_request(request):
        nonlocal sent
        if sent:
            raise RuntimeError("Decision check permits only one outbound request")
        sent = True
        if probe.name != "media:image":
            return
        body = await request.aread()
        if entry["api_style"] == "openai":
            from ._media import _data_url

            payload = json.loads(body)
            blocks = payload.get("input", [{}])[0].get("content", [])
            wire["attachment_observed"] = any(
                block.get("type") == "input_image"
                and block.get("image_url") == _data_url("image/png", png_fixture())
                for block in blocks
            )
        else:
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + request.headers.get("content-type", "") + "\r\n\r\n").encode()
                + body
            )
            parts = list(message.iter_parts())
            wire["attachment_observed"] = request.headers.get(
                "TypeSafe-Danger-Zone-Images"
            ) == "true" and any(
                part.get_content_type() == "image/png"
                and part.get_payload(decode=True) == png_fixture()
                for part in parts
            )

    async def capture_status(response):
        statuses.append(response.status_code)
        # Read usage even if the reply later fails validation; keep only the counts.
        await response.aread()
        try:
            usage = response.json().get("usage", {})
            if isinstance(usage, dict):
                for field in ("input_tokens", "output_tokens"):
                    value = usage.get(field)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        observed_usage[field] = value
        except (ValueError, AttributeError):
            pass

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    http_client = httpx.AsyncClient(
        timeout=probe.timeout_seconds,
        headers=headers,
        event_hooks={"request": [capture_request], "response": [capture_status]},
    )
    client = None
    try:
        client = client_from_config(
            alias,
            {**entry, **({"media_inputs": ["image"]} if probe.name == "media:image" else {})},
            api_key=api_key,
            client=http_client,
            media_inputs=["image"]
            if probe.name == "media:image"
            else entry.get("media_inputs", []),
            decision_image_transport=entry.get("decision_image_transport", "json"),
            typesafe_danger_zone_images=entry.get("typesafe_danger_zone_images", False),
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        )
        response, facts = await typed_decision(client, probe.name)
        if probe.name == "media:image":
            facts.update(wire)
        return response, statuses[-1] if statuses else None, facts
    except Exception as exc:
        if statuses:
            exc._connect_http_status = statuses[-1]
        exc._connect_usage = observed_usage
        exc._connect_wire = wire if probe.name == "media:image" else {}
        raise
    finally:
        if client is not None:
            await client.aclose()
        await http_client.aclose()


async def run(
    proposal: ConnectPlan,
    *,
    approved: Literal["all", "minimal", "none"],
    api_key: str | None = None,
) -> ConnectResult:
    """Run ``run_steps`` to the end and return its final ``ConnectResult``."""
    async for event in run_steps(proposal, approved=approved, api_key=api_key):
        if isinstance(event, ConnectResult):
            return event
    raise RuntimeError("Probe run ended without a result")


async def check_stage(
    proposal: ConnectPlan, stage: str, *, api_key: str | None = None
) -> ConnectResult:
    """Run the test calls of one stage again and return the updated entry.

    ``stage`` is routing, decision, tools, reasoning, media, session or all.
    Calling this counts as the user's approval for that stage, within the
    plan's existing limits. It does not prompt or write the registry. Results
    of other stages are kept; the selected stage's old results are discarded
    and never reused. The API key is read from ``api_key_env`` if not given.

    Raises ValueError for an unknown stage, a stage the plan has no test calls
    for, or a missing API key.
    """
    if stage not in {"routing", "tools", "reasoning", "media", "decision", "session", "all"}:
        raise ValueError("Unknown check stage")
    if proposal.entry.get("client_type") == "decision" and stage not in {
        "routing",
        "decision",
        "media",
        "all",
    }:
        raise ValueError("Decision models support routing, decision, media and all checks")
    if stage == "decision" and proposal.entry.get("client_type") != "decision":
        raise ValueError("Decision stage requires a native decision model")
    # If the selected reasoning level is in the approved plan with the same
    # cap, the routing/media calls may use that cap. Running a stage must not
    # approve a changed cap; such edits need refresh_plan and a new review.
    level = proposal.entry.get("reasoning_level")
    planned_level = next((p for p in proposal.probes if p.name == f"level:{level}"), None)
    if planned_level and configured_reply_cap(proposal.entry) == (
        planned_level.token_estimate - planned_level.input_reservation
    ):
        refreshed = refresh_plan(proposal)
        proposal = replace(
            refreshed,
            # Level test calls keep the caps they were approved with.
            probes=tuple(
                old if old.name.startswith("level:") else new
                for old, new in zip(proposal.probes, refreshed.probes, strict=True)
            ),
        )
    probes = tuple(
        p
        for p in proposal.probes
        if stage == "all"
        or p.name == stage
        or stage == "decision"
        and (p.name == "routing" or p.name.startswith("decision:"))
        or stage == "reasoning"
        and p.name.startswith("level:")
        or stage == "media"
        and (
            p.name == "routing"
            or p.name.startswith("media:")
            or proposal.entry.get("client_type") == "decision"
            and p.name.startswith("decision:")
        )
    )
    if stage == "decision" and not any(p.name.startswith("decision:") for p in probes):
        raise ValueError("Decision stage requires a fresh typed decision plan")
    if stage == "media" and not any(p.name.startswith("media:") for p in probes):
        raise ValueError("Media stage requires explicit test_media_inputs")
    if stage == "reasoning" and not probes:
        raise ValueError("No reasoning levels configured; provide a reasoning_levels mapping")
    entry = configure_entry(proposal.entry)
    for probe in probes:
        entry["provenance"]["probes"].pop(probe.name, None)
    if api_key is None and entry.get("api_key_env"):
        api_key = os.environ.get(entry["api_key_env"])
        if not api_key:
            raise ValueError("The configured credential variable is unset or empty")
    selected = replace(
        proposal,
        entry=entry,
        probes=probes,
        session_checks=stage in {"session", "all"} and entry.get("client_type") != "decision",
    )
    return await run(selected, approved="all", api_key=api_key)


def diagnostic_prompt(
    stage: str,
    entry: dict,
    checks: dict,
    *,
    run_context: dict | None = None,
    api_key: str | None = None,
) -> str:
    """Return a text a user can copy to a person or agent to diagnose a failed stage.

    Contains instructions, the connection settings, the public test results
    and installation details (see ``installation_context``), plus allowed
    ``run_context`` keys. API keys are removed, and no raw request or reply
    bodies are included.
    """
    from nooa.unifiedllm.connect._diagnostics import installation_context, scrub_report

    installation = installation_context()
    context = {
        key: entry[key]
        for key in (
            "model_name",
            "api_style",
            "client_type",
            "api_base",
            "endpoint",
            "api_key_env",
            "transport",
            "max_tokens",
            "store",
            "include",
            "cache_breakpoint",
            "reasoning_default",
        )
        if key in entry
    }
    # The endpoint may come from user input and contain credentials or a query;
    # normalising rejects those, and an invalid value is left out.
    endpoint_key = "endpoint" if "endpoint" in context else "api_base"
    if endpoint_key in context:
        try:
            context[endpoint_key] = normalize_endpoint(context[endpoint_key])
        except (TypeError, ValueError):
            context[endpoint_key] = "[invalid endpoint omitted]"
    outcomes = {name: public_record(record) for name, record in checks.items()}
    text = (
        f"Diagnose and fix NOOA Connect stage {stage!r}. "
        "First read the nooa-connect skill for permission and headless operation, then "
        "the model-configuration reference and companion docs located by the "
        "installation references below. "
        "Inspect the configuration, credential lookup, and actual request construction. "
        "Test the key with a connection check. Distinguish rejection, missing "
        "evidence, and unsupported features by reviewing the answer and settings sent. "
        "Use nooa.unifiedllm.connect library functions or nooa connect --stage to isolate the failure, "
        "then rerun the affected checks within configured limits. Never print or copy key "
        "values, reasoning text, opaque state, or raw error bodies. Preserve unrelated aliases "
        "and inspect layer precedence before editing the target file. Confirm permission for "
        "additional paid calls, model configuration or writes, and confirm the scope and "
        "remaining allowance before rerunning. "
        "Report the cause, fix, and evidence.\n"
        + json.dumps(
            {
                "connection": context,
                "checks": outcomes,
                "run_context": installation
                | {
                    key: value
                    for key, value in (run_context or {}).items()
                    if key
                    in {
                        "target_file",
                        "working_directory",
                        "alias",
                        "wire_model",
                        "registry_files",
                        "effective_alias_source",
                        "nooa_version",
                        "transport_override",
                        "approved_budget_tokens",
                        "remaining_budget_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                        "credential_source",
                        "credential_available",
                        "interface_timeout_seconds",
                        "reasoning_timeout_seconds",
                        "rerun_command",
                        "discovery_succeeded",
                        "proxy_variables_set",
                        "target_in_registry_chain",
                        "target_load_note",
                        "credential_note",
                    }
                },
            },
            indent=2,
        )
    )

    return scrub_report(text, api_key=api_key, api_key_env=entry.get("api_key_env"))


async def run_steps(
    proposal: ConnectPlan,
    *,
    approved: Literal["all", "minimal", "none"],
    api_key: str | None = None,
) -> AsyncIterator[ProbeUpdate | ConnectResult]:
    """Send the approved test calls of ``proposal`` and yield progress and the result.

    ``approved`` is "all", "minimal" (only the routing call) or "none" (no
    calls; every test is recorded as not approved). Calls are not retried
    and there is no fallback to another provider. If the routing call fails,
    the remaining calls are skipped. An earlier result is reused when it is
    current and was made with exactly the same request.

    An HTTP 400 is recorded as a rejection, not as proof the feature is
    unsupported; authentication and temporary failures leave the feature
    untested. Errors record status codes, never server bodies or keys.

    Yields a ``ProbeUpdate`` before and after each call, and finally the
    ``ConnectResult`` to save. The library never prints. Consume the iterator
    fully, or close it with ``contextlib.aclosing`` when stopping early, so
    HTTP clients are closed. Raises ValueError if the API key is needed but
    its environment variable is unset.
    """
    if approved not in {"all", "minimal", "none"}:
        raise ValueError("approved must be all, minimal or none")
    if proposal.entry.get("client_type") == "decision":
        async for update in _run_decision_steps(proposal, approved=approved, api_key=api_key):
            yield update
        return
    entry = configure_entry(proposal.entry)
    provenance = entry["provenance"]
    records = provenance["probes"]
    spent = 0
    stopped = False
    key = api_key
    if approved != "none" and key is None and entry["api_key_env"]:
        key = os.environ.get(entry["api_key_env"])
        if not key:
            raise ValueError(f"Set {entry['api_key_env']} before probing, or approve none")
    pending = deque(proposal.probes)
    while pending:
        probe = pending.popleft()
        if entry.get("include") == [] and "include" in probe.body:
            probe = replace(probe, body={**probe.body, "include": []})
        previous = records.get(probe.name, {})
        if (
            verification_is_current(previous, entry, probe.name)
            and previous.get("outcome") == "accepted"
            and previous.get("request") == probe.body
            and previous.get("client") == "unifiedllm"
            and previous.get("settings_sent") is True
            and previous.get("structured_output_mode") == entry.get("structured_output_mode")
            and (
                probe.name != "routing"
                or not probe.uses_configured_cap
                and previous.get("baseline_kind") == "interface_acceptance"
                or probe.uses_configured_cap
                and text_baseline_passed(previous)
            )
        ):
            if probe.name == "routing" and entry.get("include"):
                provenance.setdefault("encrypted_reasoning", {})["outcome"] = "accepted"
            yield ProbeUpdate(
                probe.name, {**deepcopy(previous), "reason": "previous result reused"}
            )
            continue
        record: dict[str, Any] = {"outcome": "not_probed"}
        if "structured_output_mode" in entry:
            record["structured_output_mode"] = entry["structured_output_mode"]
        records[probe.name] = record
        if approved == "none" or approved == "minimal" and probe.name != "routing":
            record["reason"] = "not approved"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        if probe.name.startswith("media:"):
            from nooa.media_policy import entry_input_interface, implemented_input_kinds

            if probe.name[6:] not in implemented_input_kinds(entry_input_interface(entry)):
                record["implementation_supported"] = False
            baseline = records.get("routing", {})
            if probe.skip_reason or not routing_allows_save(entry, entry):
                record["reason"] = (
                    probe.skip_reason or text_baseline_reason(baseline) + "; media not tested"
                )
                yield ProbeUpdate(probe.name, deepcopy(record))
                continue
        if stopped or spent + probe.token_estimate > proposal.budget_tokens:
            record["reason"] = "previous check failed" if stopped else "budget exhausted"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        if (
            not isinstance(probe.timeout_seconds, (int, float))
            or not 1 <= probe.timeout_seconds <= 120
        ):
            record["reason"] = "probe timeout must be between 1 and 120 seconds"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        if probe.body.get("stream") or any(
            probe.body.get(name, 1) != 1 for name in ("n", "best_of")
        ):
            record["reason"] = "probes require one non-streaming generation"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        style = entry["api_style"]
        caps = [
            probe.body[name]
            for name in ("max_tokens", "max_output_tokens", "max_completion_tokens")
            if name in probe.body
        ]
        required_cap = "max_output_tokens" if style == "responses" else "max_tokens"
        expected_cap = configured_reply_cap(
            entry, probe.name[6:] if probe.name.startswith("level:") else None
        )
        if probe.uses_configured_cap and probe.body.get(required_cap) != expected_cap:
            record["reason"] = "Configured reply cap changed after planning; make a new plan"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        if required_cap not in probe.body or any(
            not isinstance(cap, int)
            or isinstance(cap, bool)
            or cap < 1
            or cap > probe.token_estimate - probe.input_reservation
            for cap in caps
        ):
            record["reason"] = "declared output cap exceeds the approved probe cap"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        spent += probe.token_estimate
        yield ProbeUpdate(probe.name, {"outcome": "running"})
        stamp_verification(record, entry, probe.name, timeout_seconds=probe.timeout_seconds)
        started = time.monotonic()
        deadline = asyncio.timeout(probe.timeout_seconds)
        record["request_shape"] = {
            "api_style": style,
            "timeout_seconds": probe.timeout_seconds,
            "output_tokens": probe.body[required_cap],
            **{k: deepcopy(probe.body[k]) for k in ("store", "include") if k in probe.body},
        }
        try:
            async with deadline:
                response, settings_sent, transport = await _run_probe(
                    proposal.alias, entry, probe, key
                )
            usage = response.usage
            # Look for reasoning parts directly instead of using
            # response.reasoning, which joins only non-empty reasoning text.
            # Some providers return a signed reasoning part with empty text
            # (Claude Sonnet 5/Opus 5 via Azure or Bedrock), which would
            # otherwise look like no reasoning. _session.py does the same.
            reasoning_parts = [part for part in response.parts if part.kind == "reasoning"]
            # usage.reasoning_tokens alone does not count: in live tests
            # Kimi/Qwen reported reasoning tokens on endpoints that return no
            # reasoning content at all. Only a reasoning part in the reply
            # counts, with or without visible text.
            reasoning = bool(reasoning_parts)
            # Detect reasoning whose readable text was withheld ("encrypted").
            # Anthropic does this in two forms, both kept in part.native by
            # chat_parts.py: a redacted_thinking block (an opaque "data"
            # string, no text), and a signed "thinking" block with empty text
            # (seen live with Claude Sonnet 5/Opus 5 via Bedrock/Azure). With
            # no text, litellm cannot estimate reasoning tokens, so report
            # "withheld" rather than a misleading 0.
            reasoning_encrypted = False
            reasoning_encrypted_bytes = None

            def _encrypted_blob_size(data: str) -> int:
                # Decoded size of an encrypted reasoning string: a rough
                # indication of how much reasoning it holds, not a token
                # count (providers give no way to convert it).
                try:
                    return len(base64.b64decode(data, validate=False))
                except ValueError:  # binascii.Error is a ValueError subclass
                    return len(data)

            for part in reasoning_parts:
                if not isinstance(part.native, Mapping):
                    continue
                block = part.native.get("thinking_blocks")
                if isinstance(block, Mapping):
                    kind = block.get("type")
                    if kind == "redacted_thinking" and isinstance(block.get("data"), str):
                        reasoning_encrypted = True
                        data = block["data"]
                        reasoning_encrypted_bytes = (reasoning_encrypted_bytes or 0) + (
                            _encrypted_blob_size(data)
                        )
                    elif (
                        kind == "thinking"
                        and isinstance(block.get("signature"), str)
                        and not part.text
                    ):
                        # chat_parts.py moved the "thinking" text out of this
                        # dict into part.text, so check part.text for empty.
                        # A signature has a fixed length regardless of how
                        # much was reasoned, so no size is reported.
                        reasoning_encrypted = True
                    continue
                # OpenAI/Azure Chat Completions endpoints (chat_parts.py) keep
                # their reasoning item under "reasoning_items", with the same
                # encrypted_content field as the Responses form below.
                #
                # Such an item can have a visible summary (part.text) and
                # encrypted_content at the same time; the encrypted part is
                # state to send back, not proof the text was withheld. So only
                # mark it withheld when there is no visible text.
                block = part.native.get("reasoning_items")
                if isinstance(block, Mapping):
                    encrypted_content = block.get("encrypted_content")
                    if isinstance(encrypted_content, str) and encrypted_content and not part.text:
                        reasoning_encrypted = True
                        reasoning_encrypted_bytes = (reasoning_encrypted_bytes or 0) + (
                            _encrypted_blob_size(encrypted_content)
                        )
                    continue
                # Responses API endpoints (response_parts.py) keep the raw
                # output item directly in part.native, e.g. {"type":
                # "reasoning", "encrypted_content": "..."}. Same rule about
                # visible summaries as above.
                encrypted_content = part.native.get("encrypted_content")
                if isinstance(encrypted_content, str) and encrypted_content and not part.text:
                    reasoning_encrypted = True
                    reasoning_encrypted_bytes = (reasoning_encrypted_bytes or 0) + (
                        _encrypted_blob_size(encrypted_content)
                    )
            # litellm estimates reasoning tokens from text length only for
            # Anthropic/Bedrock; for other providers it reports nothing even
            # when reasoning text came back (seen live with Qwen and DeepSeek).
            # Record the length of that text instead; the text itself is never
            # stored.
            reasoning_text_chars = (
                sum(len(part.text) for part in reasoning_parts if part.text) or None
            )
            tool = any(call.name == "probe_tool" for call in response.tool_calls)
            tokens = usage.input_tokens + usage.output_tokens if usage else 0
        except Exception as exc:
            from nooa.unifiedllm.connect._diagnostics import timeout_details

            status = getattr(exc, "_connect_http_status", getattr(exc, "status_code", None))
            record["error"] = type(exc).__name__
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if isinstance(status, int):
                record["status_code"] = status
                record["outcome"] = "rejected" if status in {400, 422} else "not_probed"
                if 200 <= status < 300:
                    record.update(
                        outcome="not_confirmed", reason="Reply not understood by the runtime"
                    )
            record.update(timeout_details(exc, deadline_expired=deadline.expired()))
            if type(exc).__name__ == "ReasoningReplayError":
                record.update(
                    outcome="not_confirmed", reason="Reply not understood (ReasoningReplayError)"
                )
            if probe.name == "routing" and probe.uses_configured_cap:
                if entry.get("structured_output_mode") == "anthropic":
                    # Keep what was observed about the request and raw response
                    # (schema field used, settings sent, finish) even though the
                    # typed result failed validation or did not finish with stop.
                    record.update(public_record(getattr(exc, "_connect_evidence", {})))
                record.update(
                    baseline_kind=TYPED_TEXT_BASELINE,
                    typed_result_validated=False,
                    answer_correct=False,
                )
                if status is None or 200 <= status < 300:
                    record["outcome"] = "not_confirmed"
                record.setdefault("reason", "Text baseline did not return a validated integer")
            if probe.name.startswith("media:"):
                from ._media import exception_reason, explicitly_rejected

                record.update(getattr(exc, "_connect_evidence", {}))
                if record.get("timeout_kind"):
                    record["request_accepted"] = False
                record.update(typed_result_validated=False)
                record.setdefault("final_answer_observed", False)
                record["outcome"] = (
                    "rejected" if explicitly_rejected(exc, probe.name[6:]) else "inconclusive"
                )
                record["reason"] = (
                    "Provider explicitly rejected this media request"
                    if record["outcome"] == "rejected"
                    else exception_reason(record, probe.name[6:], exc)
                )
            if (
                probe.name == "routing"
                and not probe.uses_configured_cap
                and entry.get("include") == ["reasoning.encrypted_content"]
                and _include_rejected(exc)
            ):
                rejection = _disable_encrypted_reasoning(entry, status)
                yield ProbeUpdate("encrypted_reasoning", deepcopy(rejection))
                # Retry once without encrypted reasoning (the entry has changed);
                # the retry is charged as another capped request.
                pending.appendleft(probe)
                yield ProbeUpdate(probe.name, deepcopy(record))
                continue
            if (
                probe.name == "routing"
                and entry.get("cache_breakpoint", "auto") is not None
                and "explicit_cache" not in entry["provenance"]
                and _cache_options_rejected(exc)
            ):
                rejection = _disable_explicit_cache(entry, status)
                # Retry once without cache fields; the saved entry keeps explicit
                # caching off. The rejected attempt is not reported as a failure.
                yield ProbeUpdate("explicit_cache", deepcopy(rejection))
                pending.appendleft(probe)
                continue
            spent += max(0, record.get("reported_tokens", 0) - probe.token_estimate)
            stopped = probe.name == "routing" or status in {401, 403}
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        record.update(
            outcome="accepted",
            elapsed_seconds=round(time.monotonic() - started, 3),
            client="unifiedllm",
            transport=transport,
            request=deepcopy(probe.body),
            reasoning_observed=reasoning,
            reasoning_encrypted=reasoning_encrypted,
            reasoning_encrypted_bytes=reasoning_encrypted_bytes,
            reasoning_text_chars=reasoning_text_chars,
            tool_observed=tool,
            reported_tokens=tokens,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None,
            finish_reason=response.finish_reason,
            checked_at=datetime.now(UTC).isoformat(),
            configured_reply_tokens=configured_reply_cap(
                entry, probe.name[6:] if probe.name.startswith("level:") else None
            ),
            tested_reply_tokens=probe.body[required_cap],
            settings_sent=settings_sent,
        )
        if not settings_sent:
            record.update(
                outcome="not_confirmed",
                reason="Could not observe the actual request; the runtime used an uninstrumented HTTP client"
                if settings_sent is None
                else "Configured reply cap or settings missing from the actual request",
            )
        if probe.name == "routing":
            record.update(response.metadata.get("connect_evidence", {}))
            if probe.uses_configured_cap:
                record.update(
                    baseline_kind=TYPED_TEXT_BASELINE,
                    typed_result_validated=type(response.parsed) is int,
                    answer_correct=type(response.parsed) is int and response.parsed == 323,
                    final_answer_observed=bool((response.content or "").strip()),
                )
                record["reason"] = text_baseline_reason(record)
                if not text_baseline_passed(record):
                    record["outcome"] = "not_confirmed"
                    stopped = True
            else:
                record["baseline_kind"] = "interface_acceptance"
        if probe.name.startswith("media:"):
            from ._media import reply_failure_reason

            record.update(response.metadata.get("connect_evidence", {}))
            record.update(
                typed_result_validated=type(response.parsed) is bool,
                final_answer_observed=bool((response.content or "").strip())
                and record.get("final_answer_observed", True),
            )
            if type(response.parsed) is bool:
                record["access_reported"] = response.parsed
            complete = (
                record["typed_result_validated"]
                and settings_sent is True
                and record["finish_reason"] == "stop"
                and record["final_answer_observed"]
                and record.get("request_accepted") is True
                and record.get("attachment_observed") is True
                and record.get("output_settings_observed") is True
            )
            if complete:
                record["outcome"] = "passed" if record["access_reported"] else "unavailable"
                record["reason"] = (
                    "Model reported access to this input"
                    if record["access_reported"]
                    else "Model cannot access this input"
                )
            else:
                record["outcome"] = "inconclusive"
                record["reason"] = reply_failure_reason(probe.name[6:], record)
        if probe.name.startswith("level:"):
            # Check only the final answer; reasoning and reply text are not
            # stored. A correct answer does not prove the setting was applied,
            # and a wrong one does not fail the check.
            record["answer_correct"] = (
                re.sub(r"[\s,]+", "", response.content or "").upper() == "BGDACEFH"
            )
            record["settings_sent"] = settings_sent
            if settings_sent is False:
                record["outcome"] = "not_confirmed"
                record["reason"] = (
                    "The client did not send the requested settings; check parameter filtering"
                )
        spent += max(0, tokens - probe.token_estimate)
        yield ProbeUpdate(probe.name, deepcopy(record))
        if probe.name == "routing" and entry.get("include"):
            provenance.setdefault("encrypted_reasoning", {})["outcome"] = "accepted"
    provenance["requests_accepted"] = [
        name for name, item in records.items() if item["outcome"] == "accepted"
    ]
    provenance["reasoning_observed"] = [
        name for name, item in records.items() if item.get("reasoning_observed")
    ]
    provenance["tokens_charged_to_budget"] = spent
    if proposal.session_checks:
        from nooa.unifiedllm.connect._session import session_steps

        checks = {}
        provenance["session_checks"] = checks
        if approved != "all" or stopped:
            checks["session"] = {
                "outcome": "not_probed",
                "reason": "not approved or earlier check failed",
            }
            yield ProbeUpdate("session", deepcopy(checks["session"]))
        else:
            async with aclosing(
                session_steps(
                    proposal.alias,
                    entry,
                    api_key=key,
                    budget_tokens=max(0, proposal.budget_tokens - spent),
                )
            ) as steps:
                async for update in steps:
                    if update.outcome["outcome"] != "running":
                        checks[update.name] = update.outcome
                    if update.outcome.get("include_rejected"):
                        rejection = _disable_encrypted_reasoning(
                            entry, update.outcome["status_code"]
                        )
                        yield ProbeUpdate("encrypted_reasoning", deepcopy(rejection))
                    yield update
            provenance["tokens_charged_to_budget"] += checks.get("session", {}).get(
                "tokens_charged_to_budget", 0
            )
    provenance["warnings"] = entry_warnings(entry)
    yield ConnectResult(proposal.alias, invalidate_verification(entry))


async def _run_decision_steps(
    proposal: ConnectPlan, *, approved: Literal["all", "minimal", "none"], api_key: str | None
) -> AsyncIterator[ProbeUpdate | ConnectResult]:
    """``run_steps`` for decision entries: send each approved decision test call once.

    All calls share the plan's token budget and are not retried. Only an
    earlier accepted routing result may be reused; the typed decision and
    image tests always run again. Later calls are skipped after an
    authentication failure, if routing was not accepted, or (for the image
    test) if a typed decision test did not answer correctly.
    """
    from ._diagnostics import timeout_details

    entry = configure_entry(proposal.entry)
    records = entry["provenance"]["probes"]
    charged = 0
    accepted = []
    baseline_ok = False
    auth_failed = False
    for probe in proposal.probes:
        previous = records.get(probe.name, {})
        # Only an accepted routing result is reused; answer and media tests rerun.
        if (
            probe.name == "routing"
            and verification_is_current(previous, entry, probe.name)
            and previous.get("outcome") == "accepted"
            and previous.get("request") == probe.body
            and previous.get("client") == "decision"
        ):
            accepted.append(probe.name)
            baseline_ok = True
            yield ProbeUpdate(
                probe.name, {**deepcopy(previous), "reason": "previous result reused"}
            )
            continue
        record = {"outcome": "not_probed"}
        records[probe.name] = record
        reason = None
        if approved == "none" or approved == "minimal" and probe.name != "routing":
            reason = "not approved"
        elif probe.skip_reason:
            reason = probe.skip_reason
        elif auth_failed:
            # Do not send a rejected key again in the same run.
            reason = "Authentication failed; later checks not sent"
        elif charged + probe.token_estimate > proposal.budget_tokens:
            reason = "budget exhausted"
        elif probe.name != "routing" and not baseline_ok:
            reason = "Decision baseline not accepted; fresh checks not tested"
        elif probe.name == "media:image" and any(
            records.get(name, {}).get("answer_correct") is not True
            for name in ("decision:typed", "decision:positive", "decision:negative")
        ):
            reason = "Typed decision baseline did not pass; image not tested"
        if reason:
            record["reason"] = reason
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        key = api_key
        if key is None and entry.get("api_key_env"):
            key = os.environ.get(entry["api_key_env"])
            if not key:
                raise ValueError(f"Set {entry['api_key_env']} before probing, or approve none")
        yield ProbeUpdate(probe.name, {"outcome": "running"})
        charged += probe.token_estimate
        stamp_verification(record, entry, probe.name, timeout_seconds=probe.timeout_seconds)
        started = time.monotonic()
        deadline = asyncio.timeout(probe.timeout_seconds)
        record["request_shape"] = {
            "api_style": entry["api_style"],
            "timeout_seconds": probe.timeout_seconds,
            "question_count": len(probe.body["questions"]),
            "image_transport": "openai_input_image"
            if entry["api_style"] == "openai"
            else entry["decision_image_transport"],
        }
        try:
            async with deadline:
                response, status, facts = await _run_decision_probe(
                    proposal.alias, entry, probe, key
                )
            record.update(facts)
            usage = response.usage
            reported = 0
            for field in ("input_tokens", "output_tokens"):
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    record[field] = value
                    reported += value
            record.update(
                request_accepted=True,
                client="decision",
                transport="httpx",
                resolved_model=response.model,
                reported_tokens=reported,
                status_code=status,
                checked_at=datetime.now(UTC).isoformat(),
            )
            charged += max(0, reported - probe.token_estimate)
            record["tokens_charged_to_budget"] = max(probe.token_estimate, reported)
            if probe.name == "routing":
                record.update(outcome="accepted", request=deepcopy(probe.body))
                baseline_ok = True
            elif probe.name.startswith("decision:"):
                record["outcome"] = "accepted" if facts["answer_correct"] else "not_confirmed"
                if not facts["answer_correct"]:
                    record["reason"] = "Native typed response accepted but fixture answer incorrect"
            else:
                passed = (
                    facts["attachment_observed"]
                    and facts["access_reported"]
                    and facts["answer_correct"]
                )
                record["outcome"] = "passed" if passed else "not_confirmed"
                if not passed:
                    record["reason"] = (
                        "Image access, known color accuracy and actual wire attachment must all be confirmed"
                    )
            if record["request_accepted"]:
                accepted.append(probe.name)
        except Exception as exc:
            status = getattr(exc, "_connect_http_status", None)
            usage = getattr(exc, "_connect_usage", {})
            reported = sum(usage.values())
            record.update(usage)
            record.update(getattr(exc, "_connect_wire", {}))
            record["reported_tokens"] = reported
            charged += max(0, reported - probe.token_estimate)
            record.update(
                error=type(exc).__name__,
                outcome="not_probed",
                tokens_charged_to_budget=max(probe.token_estimate, reported),
            )
            if isinstance(status, int):
                record["status_code"] = status
                record["outcome"] = "rejected" if status in {400, 422} else "not_probed"
                auth_failed = auth_failed or status in {401, 403}
                if 200 <= status < 300:
                    record.update(
                        outcome="not_confirmed",
                        request_accepted=True,
                        reason="Reply not understood by the decision client",
                        typed_result_validated=False,
                    )
            record.update(timeout_details(exc, deadline_expired=deadline.expired()))
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        yield ProbeUpdate(probe.name, deepcopy(record))
    entry["provenance"].update(
        requests_accepted=accepted, tokens_charged_to_budget=charged, warnings=entry_warnings(entry)
    )
    yield ConnectResult(proposal.alias, invalidate_verification(entry))


async def check_interfaces(
    alias: str,
    model: str,
    api_base: str,
    api_key_env: str,
    *,
    budget_tokens: int = DEFAULT_CHECK_BUDGET,
    output_tokens: int = DISCOVERY_OUTPUT_TOKENS,
    api_key: str | None = None,
    styles: tuple[str, ...] = ("chat", "responses", "anthropic"),
    endpoints: Mapping[str, str] | None = None,
    allow_cross_origin_credentials: bool = False,
    timeout_seconds: float = 30,
    reasoning_template: str | None = None,
    reasoning_level: str = "medium",
    reasoning_output_tokens: int = DEFAULT_REASONING_OUTPUT_TOKENS,
) -> AsyncIterator[ProbeUpdate | InterfaceResult]:
    """Send one routing test call per API style in ``styles`` to see which ones work.

    These are paid calls: the frontend warns the user before calling this.
    All styles share one token budget and nothing is retried. Yields a
    ``ProbeUpdate`` per progress event (named after the style) and finally an
    ``InterfaceResult``; close the iterator when cancelling. A timeout,
    authentication error or rejection is recorded as such, never as "not
    supported". A failure on one style does not stop the others, since styles
    authenticate differently. The chosen style's entry can be passed to
    ``plan(existing_entry=...)`` to reuse its routing result.

    ``reasoning_template``, if given, also sends one reasoning test call per
    style (the same puzzle ``plan`` uses for its level checks), with settings
    built for each style by ``reasoning_settings``. Then
    ``InterfaceResult.reasoning_observed`` and ``recommended_style`` show
    which styles actually return reasoning for this model, not only which
    accept a call. An endpoint may accept every style while only one returns
    reasoning; for example, some NVIDIA Inference Hub models return
    ``reasoning_content`` on Chat without extra parameters, but on Responses
    only when ``reasoning.summary`` is requested. Only the "effort" template
    works for chat and responses alike; an anthropic-only comparison needs a
    matching template (adaptive, budget or thinking). Templates are ignored
    for decision styles.

    ``endpoints`` maps a style to its own endpoint instead of ``api_base``.
    Decision endpoints are full URLs and are never derived from a chat URL.
    When several styles are compared, every decision style needs an entry in
    ``endpoints``; a single decision style may use ``api_base``. All
    endpoints are validated before any request. If an API key is given or
    ``api_key_env`` is set, an endpoint on a different origin (scheme, host
    or port) than ``api_base`` requires ``allow_cross_origin_credentials=True``
    (``--share-interface-key`` in the CLI), so keys are never sent to another
    origin by default. Calls without a key, and different paths on the same
    origin, are always allowed.

    Raises ValueError for invalid styles, timeouts or endpoints.
    """
    if (
        not styles
        or any(style not in _PATHS for style in styles)
        or not 1 <= timeout_seconds <= 120
    ):
        raise ValueError("Choose supported interfaces and a timeout between 1 and 120 seconds")
    api_base = normalize_endpoint(api_base)
    if endpoints is not None and not isinstance(endpoints, Mapping):
        raise ValueError("endpoints must map API styles to endpoint URLs")
    selected_endpoints = {}
    for style, endpoint in (endpoints or {}).items():
        if style not in _PATHS or not isinstance(endpoint, str):
            raise ValueError("endpoints must map supported API styles to endpoint URLs")
        selected_endpoints[style] = normalize_endpoint(endpoint)
    unique_styles = tuple(dict.fromkeys(styles))
    if len(unique_styles) > 1:
        for style in unique_styles:
            if style in DECISION_API_STYLES and style not in selected_endpoints:
                raise ValueError(
                    f"Mixed interfaces require an explicit endpoints entry for {style}"
                )

    def origin(endpoint: str) -> tuple[str, str | None, int | None]:
        parsed = urlsplit(endpoint)
        scheme = parsed.scheme.lower()
        port = parsed.port if parsed.port is not None else {"https": 443, "http": 80}.get(scheme)
        return (scheme, parsed.hostname, port)

    if (api_key or api_key_env) and not allow_cross_origin_credentials:
        source_origin = origin(api_base)
        for style in unique_styles:
            if origin(selected_endpoints.get(style, api_base)) != source_origin:
                raise ValueError(
                    "Cross-origin interface credentials require "
                    "allow_cross_origin_credentials=True (--share-interface-key in the CLI)"
                )
    results = {}
    spent = 0
    proposals = {}
    for style in unique_styles:
        style_reasoning_levels = (
            {reasoning_level: reasoning_settings(reasoning_template, style, reasoning_level)}
            if reasoning_template and style not in DECISION_API_STYLES
            else None
        )
        proposal = plan(
            alias,
            model,
            style,
            selected_endpoints.get(style, api_base),
            api_key_env,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
            reasoning_levels=style_reasoning_levels,
            reasoning_output_tokens=reasoning_output_tokens,
            decision_checks=False,
        )
        if style in DECISION_API_STYLES:
            # Decision requests have no reply token field to cap.
            probes: tuple[Probe, ...] = (
                replace(proposal.probes[0], timeout_seconds=timeout_seconds),
            )
        else:
            routing = replace(
                proposal.probes[0],
                timeout_seconds=timeout_seconds,
                body={
                    **proposal.probes[0].body,
                    "max_output_tokens" if style == "responses" else "max_tokens": output_tokens,
                },
                token_estimate=output_tokens + 512,
                uses_configured_cap=False,
            )
            level_probes = tuple(
                replace(
                    p,
                    # Reasoning takes longer than a plain reply; a short
                    # timeout would hide reasoning on slower models.
                    timeout_seconds=max(timeout_seconds, REASONING_DISCOVERY_TIMEOUT),
                    body={
                        **p.body,
                        "max_output_tokens" if style == "responses" else "max_tokens": (
                            reasoning_output_tokens
                        ),
                    },
                    token_estimate=reasoning_output_tokens + 512,
                    uses_configured_cap=False,
                )
                for p in proposal.probes
                if p.name.startswith("level:")
            )
            probes = (routing, *level_probes)
        proposal.entry["provenance"]["interface_endpoint"] = {
            "api_style": style,
            "endpoint": proposal.entry.get("endpoint", proposal.entry.get("api_base")),
            "source": "endpoints" if style in selected_endpoints else "api_base",
        }
        proposals[style] = replace(proposal, probes=probes)
    for style, proposal in proposals.items():
        proposal = replace(proposal, budget_tokens=max(0, budget_tokens - spent))
        async with aclosing(run_steps(proposal, approved="all", api_key=api_key)) as steps:
            async for event in steps:
                if isinstance(event, ConnectResult):
                    results[style] = event
                    spent += event.entry["provenance"]["tokens_charged_to_budget"]
                else:
                    yield ProbeUpdate(style, event.outcome)
    yield InterfaceResult(results, spent)


ALIAS_MISSING = object()
ALIAS_UNCONDITIONAL = object()


def alias_target(models: dict, alias: str):
    """Return a copy of ``models[alias]`` as the value the user agreed to replace.

    Pass it to ``write(expected_alias=...)`` so the save fails if the file
    changed since. Returns ``ALIAS_MISSING`` if the alias does not exist,
    which is different from an alias set to null. Do not serialize or hash
    the value: older entries may contain secrets.
    """
    from copy import deepcopy

    return deepcopy(models[alias]) if alias in models else ALIAS_MISSING


@contextmanager
def registry_write_lock(path: Path):
    """Hold the exclusive write lock of the registry file ``path``; yield the resolved path.

    The lock is a separate ``.<name>.lock`` file next to the registry file.
    When a save changes both the secrets file and the model file, take the
    secrets lock first, then this one. Do not call ``write()`` while holding
    this lock: it takes the lock itself through its own file descriptor.
    Recovery code holds it while checking which aliases use a credential and
    while restoring the secrets file.
    """
    import fcntl

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never delete the lock file: a new file would let a writer skip processes
    # still waiting on the old one.
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield path


def write(entry: dict, path: Path, *, alias: str, expected_alias=ALIAS_UNCONDITIONAL) -> None:
    """Save ``entry`` as ``alias`` in the registry file ``path``, under its write lock.

    ``expected_alias`` is the value from ``alias_target`` that the user saw
    when agreeing to the change; the save fails if the file's current value
    differs (``ALIAS_MISSING`` means the alias must not exist yet). Every
    frontend must pass it. Library callers may leave it out for an
    unconditional update (kept for backward compatibility). After a
    conflict, reload and ask the user again; never retry automatically. The
    error does not reveal the file's current value.

    Only the YAML text of this alias is replaced, so other entries and
    comments are kept as they are, and the file is replaced atomically.
    Raises ValueError if the entry is invalid or if normalising it for the
    save would change the settings that were tested.
    """
    candidate = configure_entry(entry)
    from ._verification import effective_config_snapshot
    from ._verification import routing_allows_save as current_baseline

    if current_baseline(entry, entry) and effective_config_snapshot(
        entry
    ) != effective_config_snapshot(candidate):
        raise ValueError(
            "Save normalization changed active settings; review and retest the candidate"
        )
    entry = candidate
    # Save the settings only. Test results, request bodies, catalogue data and
    # warnings belong to the Connect session (shown by
    # ``nooa connect check --json``); saving them would write the whole history
    # of the Connect run into the registry.
    entry.pop("provenance", None)
    with registry_write_lock(path) as path:
        _write_entry(entry, path, alias=alias, expected_alias=expected_alias)


def _write_entry(
    entry: dict, path: Path, *, alias: str, expected_alias=ALIAS_UNCONDITIONAL
) -> None:
    """Replace ``alias`` in the registry file; the caller must hold the write lock.

    Reads the file, checks ``expected_alias`` (see ``write``), edits only the
    alias's YAML text, verifies that the result parses to the expected data
    and that the alias loads back exactly as ``entry``, then atomically
    replaces the file, keeping its permissions and line endings. Raises
    ValueError if the file is not a valid registry, uses YAML anchors, or
    was changed by someone else meanwhile.
    """
    original = path.read_bytes() if path.exists() else None
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    newline = "\r\n" if original and b"\r\n" in original else "\n"
    source = (original or b"").decode("utf-8").replace("\r\n", "\n")
    data = yaml.safe_load(source)
    if data is None:
        data = {}
    if isinstance(data, dict) and data.get("models") is None:
        data["models"] = {}
    if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
        raise ValueError("Registry must be a mapping with a models mapping")
    current = data.get("models", {})
    if expected_alias is not ALIAS_UNCONDITIONAL:
        matches = (
            alias not in current
            if expected_alias is ALIAS_MISSING
            else alias in current and current[alias] == expected_alias
        )
        if not matches:
            raise ValueError("Alias changed since review; reload and confirm replacement again")
    document = yaml.compose(source)
    if any(isinstance(token, (yaml.AnchorToken, yaml.AliasToken)) for token in yaml.scan(source)):
        raise ValueError(
            "Registry uses YAML anchors or merge keys; expand them before updating with Connect"
        )
    models_node = (
        next((value for key, value in document.value if key.value == "models"), None)
        if document
        else None
    )
    alias_indent = models_node.start_mark.column if models_node and models_node.value else 2
    dumped = yaml.safe_dump({alias: entry}, sort_keys=False, allow_unicode=True).rstrip()
    indented = "\n".join(" " * alias_indent + line for line in dumped.splitlines()) + "\n"

    def content_end(node):
        # A block mapping's end position includes the comments after it.
        # Use the end of its last value instead, so comments placed before
        # the next entry stay with that entry.
        if isinstance(node, yaml.MappingNode) and node.value and not node.flow_style:
            return content_end(node.value[-1][1])
        if isinstance(node, yaml.SequenceNode) and node.value and not node.flow_style:
            return content_end(node.value[-1])
        return node.end_mark

    if models_node is not None and isinstance(models_node, yaml.ScalarNode):
        # "models:" with no value is null in YAML; treat it as empty.
        text = (
            source[: models_node.start_mark.index]
            + "\n"
            + indented
            + source[models_node.end_mark.index :]
        )
    elif models_node is None:
        separator = "" if not source or source.endswith("\n") else "\n"
        text = source + separator + "models:\n" + indented
    elif models_node.flow_style:
        models = {**data["models"], alias: entry}
        # An inline ({...}) mapping cannot be edited line by line; replace it whole.
        replacement = yaml.safe_dump(models, default_flow_style=True, sort_keys=False).strip()
        text = (
            source[: models_node.start_mark.index]
            + replacement
            + source[models_node.end_mark.index :]
        )
    elif alias in data["models"]:
        key, value = next((key, value) for key, value in models_node.value if key.value == alias)
        lines = source.splitlines(keepends=True)
        end = content_end(value)
        end_line = end.line + bool(end.column)
        text = "".join(lines[: key.start_mark.line]) + indented + "".join(lines[end_line:])
    else:
        end = content_end(models_node)
        insertion = sum(
            len(line) for line in source.splitlines(keepends=True)[: end.line + bool(end.column)]
        )
        prefix = "" if insertion == 0 or source[insertion - 1] == "\n" else "\n"
        text = source[:insertion] + prefix + indented + source[insertion:]
    expected = {**data, "models": {**data.get("models", {}), alias: entry}}
    if yaml.safe_load(text) != expected:
        raise ValueError("Could not construct the registry update without changing other entries")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text.replace("\n", newline))
            os.chmod(temporary.name, mode)
            temporary.flush()
            os.fsync(temporary.fileno())
        from nooa.unifiedllm.registry import load_model_entries

        loaded = load_model_entries(temporary_path).get(alias)
        if loaded != entry:
            raise ValueError(
                "Saved alias did not resolve to the reviewed candidate; no changes made"
            )
        if (path.read_bytes() if path.exists() else None) != original:
            raise ValueError("Registry changed during write; reload before retrying")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
