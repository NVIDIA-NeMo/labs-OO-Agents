# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model onboarding shared by the CLI and TUI; no prompts or terminal output.

``plan`` prepares data for approval. ``run`` sends only approved, capped requests
without network-error retries. ``write`` updates one alias in a registry file. Catalogue limits
are estimates with sources; acceptance does not prove a reasoning setting was
obeyed. The runtime reads the result, never these onboarding templates.
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
from contextlib import aclosing
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
# OpenRouter defaults to text output, excluding native decision models.
OPENROUTER_MODEL_OUTPUTS = "text,decisions"
# Effectively unlimited: no real check plan approaches this, so an unset
# --budget-tokens never causes checks to be skipped. A finite sentinel (not
# float("inf")) keeps every arithmetic site, type annotation, and JSON
# encoding of a ConnectPlan unchanged.
DEFAULT_CHECK_BUDGET = 10**15
DEFAULT_REASONING_OUTPUT_TOKENS = 4096
# Reply cap for the interface discovery probes (one short routing request per
# API format). Reasoning models spend part of the cap thinking before the
# visible reply, so a small cap (it was 200) makes them fail with "ran out of
# reply tokens" before they answer. Must stay within plan()'s 1..4096 range.
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
    """Recognize a field rejection, never persist provider error text."""
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
        # A structured field rejection is sufficient even when the message is
        # generic. Rejected history/model fields must never disable include.
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


def _disable_encrypted_reasoning(entry: dict, status: int) -> dict:
    # Explicit empty include suppresses the runtime's native-endpoint default too.
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


# Omission preserves a saved declaration; explicit None clears it.
STRUCTURED_OUTPUT_UNSET = object()


def _validate_structured_output(entry: dict, override=STRUCTURED_OUTPUT_UNSET):
    """Validate route-local declarations before credentials or request construction."""
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
# Decision API styles: their endpoint is the complete decisions URL, used as given.
DECISION_API_STYLES = ("systemone", "openai")


@dataclass(frozen=True)
class ProviderPreset:
    """Public connection defaults shared by frontends; models are discovered live."""

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
    """One proposed POST, with no credentials in its body."""

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
    """Reviewable input to run; token and price estimates are not billing caps."""

    alias: str
    entry: dict[str, Any]
    probes: tuple[Probe, ...]
    budget_tokens: int
    token_estimate: int
    price_estimate: float | None
    session_checks: bool = False


@dataclass(frozen=True)
class ConnectResult:
    alias: str
    entry: dict[str, Any]


@dataclass(frozen=True)
class InterfaceResult:
    """Observed interface results, not a permanent provider-support catalogue."""

    results: dict[str, ConnectResult]
    tokens_charged_to_budget: int

    @property
    def accepted(self) -> tuple[str, ...]:
        return tuple(
            style
            for style, result in self.results.items()
            if result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"
        )

    @property
    def accepted_client_types(self) -> tuple[str, ...]:
        """Distinct accepted runtime client types, in accepted interface order."""
        return tuple(
            dict.fromkeys(self.results[style].entry["client_type"] for style in self.accepted)
        )

    @property
    def reasoning_observed(self) -> dict[str, bool]:
        """Which interfaces put a reasoning item on the wire, not merely accepted a call.

        Empty (all False) unless check_interfaces was given reasoning_levels;
        an interface can be accepted and still never surface reasoning.
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
        """Prefer accepted order when decision and chat-family clients both accept.

        For chat-family-only comparisons, prefer observed reasoning, then order.
        No interface is recommended if nothing was accepted.
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
    """A request starting or finishing, for frontend progress display."""

    name: str
    outcome: ProbeRecord


@dataclass(frozen=True)
class Verdict:
    """Shared result: unapproved checks are skipped; missing evidence needs attention."""

    passed: tuple[str, ...]
    needs_attention: tuple[str, ...]
    skipped: tuple[str, ...]
    unobserved_levels: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.passed) and not self.needs_attention and not self.skipped


def verdict(entry: dict) -> Verdict:
    """Classify saved check evidence without making calls or guessing support."""
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
    """Advertised models, not verified interface support.

    ``api_base`` remains the inference base (legacy root discovery may resolve /v1).
    ``models_url`` records the actual listing source, independent of inference.
    """

    api_base: str
    models: tuple[dict[str, Any], ...]
    models_url: str | None = None
    request_count: int = 0


class DiscoveryError(ValueError):
    """Safe discovery failure; frontends can offer a new key on 401 or 403."""

    def __init__(self, message: str, *, status_code: int | None = None, request_count: int = 0):
        super().__init__(message)
        self.status_code = status_code
        self.request_count = request_count


def normalize_endpoint(endpoint: str) -> str:
    """Accept an API base or /models URL without embedded credentials."""
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
    if style == "anthropic":
        return {"anthropic-version": "2023-06-01", **({"x-api-key": key} if key else {})}
    return {"Authorization": f"Bearer {key}"} if key else {}


def validate_models_url(models_url: str) -> str:
    """Validate an explicit full catalogue GET URL, without rewriting its path."""
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
    """List endpoint models without saving credentials or making generation calls.

    ``models_url`` is an explicit full GET URL, required for decision routes.
    It never changes the inference endpoint and gets no root-path fallback.
    ``listing_style`` chooses listing authentication/pagination, not inference.
    ``api_key`` authenticates listing only; callers must choose its source explicitly.
    Root URLs try /models, then /v1/models only on 404. Anthropic pagination
    uses the same fetcher. Limits apply to the whole discovery: 30 seconds,
    5 MiB and 5,000 entries. Redirects and automatic retries are disabled.
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
                    # Advertised hints only: do not enable interfaces or media from these.
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
                    # Published OpenRouter route ceilings can only tighten a listing.
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
    """Fetch public metadata without sending the endpoint's credentials."""
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
    """Shared normalization rule for match_models() and fuzzy_match_models().

    Kept in one place so the two matchers cannot silently disagree on what
    counts as "the same" model id.
    """
    return re.sub(r"[-_.]", "", value.lower())


def match_models(model: str, models: list[dict]) -> list[dict]:
    """Suggest up to three matches. The frontend must confirm a candidate."""
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
    """Best-effort "did you mean" suggestions once match_models finds nothing.

    A gateway-routed model ID (e.g. ``aws/anthropic/bedrock-claude-opus-5``)
    carries routing segments the catalogue never records, so exact/suffix
    matching in match_models() can find nothing even though the model is
    listed under its own name. This never auto-selects a candidate; the
    frontend must still confirm one.
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
    """Build an onboarding candidate, not a claim about a model's support.

    Responses always asks for ``reasoning.summary`` alongside effort. Some
    routes (OpenAI/Azure/DeepSeek observed) return summary text without being
    asked, but others (Kimi, Qwen observed on the Hub) omit the entire
    "reasoning" output item -- not just its text -- unless summary is
    explicitly requested, even though usage still reports billed reasoning
    tokens. Asking unconditionally is harmless where it's already the
    default and is required to observe reasoning at all on routes where it
    isn't.
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
    """List accepted reasoning-on checks that returned no reasoning information.

    Read the request settings, not label or provider names. This recognizes the
    onboarding templates; arbitrary custom settings with unknown meaning are
    left alone. Absence of evidence is not evidence that reasoning was disabled.
    Frontends can show one warning without changing the saved probe outcomes.
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
    """Apply safe persisted defaults to a detached entry, without doing any I/O.

    ``max_tokens`` is the runtime cap on every interface. Published ceilings
    remain provenance, not request allocations. ``include: []`` is an explicit
    opt-out from carrying encrypted reasoning on stateless Responses calls.
    ``structured_output_mode`` is a Completion-only declaration: omission
    preserves it, explicit None clears native mode, and 'auto' keeps the legacy
    carrier. Only explicit 'anthropic' selects output_config.format.
    ``cache_enabled=False`` disables explicit cache options/markers, not provider
    automatic caching. None preserves the entry; True selects the runtime auto policy.
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
        result.setdefault("provenance", {})["structured_output_profile"] = {"mode": effective_mode}
    if structured_output_mode is not STRUCTURED_OUTPUT_UNSET:
        result["structured_output_mode"] = structured_output_mode
    result["media_inputs"] = list(validate_media_inputs(result.get("media_inputs")))
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

        # Custom bodies/level patches are preserved, not silently rewritten.
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
            # Keep the CURRENT literal declaration, not an old endpoint ceiling
            # or its larger historical declaration after a route/scope edit.
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
            # Reconfiguration/import may lower the bound, never resurrect a
            # larger declaration after an intentional edit to the effective value.
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
        # An endpoint-derived output ceiling is historical on another route too.
        # A newly supplied literal ceiling still applies during initial planning.
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
    """Merge reported limits, preferring this endpoint; never infer context by addition.

    An input-only limit is a conservative context-management bound, not evidence
    of the model's total context size. Raw limits and their meanings stay visible.
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
    # Legacy callers may omit the ID; an explicit neighbour is never evidence.
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
    # Keep input advertisements separate from limits/settings; endpoint facts win
    # only for the exact selected model, never a neighbour on a shared hub.
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
    """Bounded descriptive/media facts only; a supplied catalogue is an explicit mapping.

    Never interpret descriptions as capabilities or copy chat settings. Attribution
    is constructed here, not accepted from provider records. Exact endpoint media
    facts take priority over the mapped OpenRouter route's advertisements.
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
    """Configuration warnings shared by the wizard, library and stage reports."""
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
    """Permissions belong to the normalized model/endpoint/interface, not credentials."""

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
    """The saved request cap, optionally replaced by a selected reasoning level."""
    if level is None:
        level = entry.get("reasoning_level")
    settings = (entry.get("reasoning_levels") or {}).get(level, {})
    return next((settings[k] for k in REPLY_CAP_KEYS if k in settings), entry["max_tokens"])


def refresh_plan(proposal: ConnectPlan) -> ConnectPlan:
    """Rebuild detached requests and reservations after a configuration edit.

    Interface discovery alone uses a small cap before model settings are chosen.
    Every configured probe sends the saved cap (or the selected level's cap).
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
            # refresh can start from a previously selected patch. Remove owned
            # fields before applying the new selection (e.g. adaptive -> Off
            # must not retain the old effort block or manual thinking budget).
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
    # Skips stay reviewable with full per-probe reservations/caps so removing a
    # skip later cannot bypass execution budget or approved-cap checks.
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
) -> ConnectPlan:
    """Prepare requests without reading credentials, files or network resources.

    ``structured_output_mode='anthropic'`` explicitly selects the native carrier
    for Completion chat/anthropic routes. Omission preserves a same-route saved
    declaration (including credential identity); explicit None clears it. None
    and 'auto' retain the existing carrier, never infer or discover native support.
    ``cache_enabled=False`` saves ``cache_breakpoint: null`` (no explicit cache
    options/markers). None preserves the default or same-route existing policy;
    True explicitly selects auto. This does not disable provider automatic caching
    or establish live compatibility.

    ``reasoning_level`` is the active saved selection, not ``reasoning_default``.
    Omission preserves a compatible same-route selection; None clears it.

    ``model`` is the exact endpoint model ID, without a LiteLLM prefix. The
    returned entry adds the prefix needed by today's runtime. Each request reserves
    its output cap plus 512 estimated input tokens (4,096 for selected generated
    media samples). ``endpoint_models_url`` is the verified listing source,
    independent of record media hints; omitted sources use the legacy /models URL.
    ``context_window`` is an explicit declaration clamped to positive literal
    context/input limits on the exact ID-bearing ``endpoint_model`` only. It
    never allocates a reply cap; declared/effective/source remain in provenance.
    Reported usage can increase that charge. Servers can ignore
    caps, so this is not a billing limit. Media facts never change declarations.
    Non-Chat video and audio remain transparently not tested. For decisions, a
    ``catalogue_settings=True`` is reserved for explicitly user-authored settings
    (e.g. a settings editor), not automatic provider metadata. Published foreign
    route/model profiles never supply reasoning defaults, allocations or prices.
    For decisions, supplied ``catalogue`` is an explicit OpenRouter ID mapping: only bounded
    descriptions, media facts and provenance are retained, never chat settings.
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
        # Published OpenRouter records fill facts, not unrelated route profiles.
        # ID-less dictionaries remain explicit hand-authored legacy settings.
        catalogue = deepcopy(catalogue)
        for field in ("reasoning", "default_parameters", "pricing", "top_provider"):
            catalogue.pop(field, None)
        # General model context remains a fallback; OpenRouter deployment
        # input/output ceilings belong to the OpenRouter route only.
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
        # Internal attribution keys in a record are never a listing contract.
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
        # Inherit permissions and controls only for the identical normalized route.
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
            # Preserve catalogue facts as advertised evidence, never chat configuration.
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
        # The Messages runtime adds /v1/messages; Chat/Responses expect an API base.
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
        # These are explicitly requested fields, not inferred model capabilities.
        # Legacy parameter filtering must not silently remove them for an alias
        # that is absent from its model catalogue.
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
    # Inherited declarations need the same filtering permissions as newly
    # authored ones; retaining an active selector alone must not drop controls.
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
            # Declarations survive supported saves, which strip provenance.
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
    entry = configure_entry(
        entry,
        reply_tokens=reply_tokens,
        cache_enabled=cache_enabled,
        structured_output_mode=structured_output_mode,
    )
    provenance = entry["provenance"]
    for probe in probes:
        if probe.name.startswith("level:"):
            probe.body.update(entry["reasoning_levels"][probe.name.removeprefix("level:")])
    if existing_entry:
        # Each probe is also compared to its exact request in run_steps. Adding
        # a level must not invalidate an unchanged routing or tool check.
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
            pass  # Missing catalogue prices are unknown, never zero.
    if session_checks:
        from nooa.unifiedllm.connect._session import reply_budget

        estimate += 3 * reply_budget(entry, max(0, budget_tokens - estimate))[2]
        price = None  # The longer replay depends on the actual seed response.
    return ConnectPlan(alias, entry, tuple(probes), budget_tokens, estimate, price, session_checks)


async def _run_probe(alias: str, entry: dict, probe: Probe, api_key: str | None):
    """Check the unsaved entry through the same client factory agents use.

    Configured probes retain the saved cap; timeouts and retries are bounded.
    Discovery and planning stay lightweight: runtime imports happen here only.
    Tool replies are inspected as data; Connect never executes model calls.
    """
    from nooa.unifiedllm import RetryConfig, Tool
    from nooa.unifiedllm.http_config import HttpConfig
    from nooa.unifiedllm.registry import client_from_config

    def probe_tool(value: str):
        raise RuntimeError("Setup checks never execute tools")

    params = deepcopy(probe.body)
    params.pop("model")  # The saved entry, not a second route, selects the model.
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
        # A level can set the cap itself; do not pass a conflicting override.
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

        # The actual Predict-generated contract, including strict fields/titles.
        expected_format = anthropic_output_format(output_model)

    async def capture_status(response):
        nonlocal schema_completed, schema_value
        response_status.append(response.status_code)
        if (typed_media_check or native) and response.is_success:
            # Keep envelope facts even when native schema validation fails.
            # Do not score access or retain reply text. Only correlate a schema
            # carrier with the exact requested output contract; native Predict
            # still has to validate and return that same boolean.
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

        # Typed calls use the constructor defaults; ordinary probes also carry
        # their planned overrides. Resolve controls exactly as runtime dispatch
        # does, including selected levels and flattened SDK extra_body values.
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
            # A faulty adapter dropping a selected control from resolution must
            # not erase the obligation to observe that declaration on the wire.
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
        # Responses SDKs omit an explicit empty include list. Absence is the
        # same opt-out, but a nonempty include must still fail observation.
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
            # Typed Predict does not request tools: any tools/tool_choice here
            # would be a schema surrogate. Explicit no-output_model tool probes
            # remain untouched. A wrapped extra_body is not the native wire shape.
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
            response.parsed = result  # Transient native method result, never persisted.
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
            # Exact requested schema output is a completed typed answer, not an
            # actionable tool call. Baseline/legacy normalization is unchanged.
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
            # LiteLLM can label a malformed HTTP-200 response as a 422 error.
            # Keep the observed status distinct from that local translation.
            exc._connect_http_status = response_status[-1]  # type: ignore[attr-defined]
        raise
    finally:
        hooks.remove(capture)
        response_hooks.remove(capture_status)
        await client.aclose()


async def _run_decision_probe(alias: str, entry: dict, probe: Probe, api_key: str | None):
    """One native typed call; only sanitized facts leave this function."""
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
        # Usage is independent of schema validity. Read locally, retain only counts.
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
    """Run an approved plan to completion without displaying progress."""
    async for event in run_steps(proposal, approved=approved, api_key=api_key):
        if isinstance(event, ConnectResult):
            return event
    raise RuntimeError("Probe run ended without a result")


async def check_stage(
    proposal: ConnectPlan, stage: str, *, api_key: str | None = None
) -> ConnectResult:
    """Run one selected check afresh, without prompts or registry writes.

    Stages are routing, decision, tools, reasoning, media, session, or all. Calling this function
    explicitly approves that stage within the plan's existing limits. Results
    retain other stages' provenance, but selected checks never reuse old evidence.
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
    # A selected level already present in the approved plan can supply the
    # routing/media cap. Never reapprove an edited cap merely by running a stage;
    # callers must explicitly refresh/review such edits first.
    level = proposal.entry.get("reasoning_level")
    planned_level = next((p for p in proposal.probes if p.name == f"level:{level}"), None)
    if planned_level and configured_reply_cap(proposal.entry) == (
        planned_level.token_estimate - planned_level.input_reservation
    ):
        refreshed = refresh_plan(proposal)
        proposal = replace(
            refreshed,
            # Existing level probes keep their independently approved caps.
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
    """Safe, copyable handoff for a person or agent; no credentials or raw bodies."""
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
    # Endpoints from arbitrary caller input may contain credentials or query data.
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
    """Execute approved probes without network retries or provider fallback.

    Minimal approval sends only the routing probe. Its failure stops all probes.
    HTTP 400 means rejected, not unsupported; auth and transient failures stay
    untested. Errors record status codes, not server bodies or credentials.
    Each probe yields progress before and after sending; the final event is
    the result to save. Frontends display these events; the library never prints.
    Consume the iterator completely, or close it with contextlib.aclosing when
    stopping early, so the HTTP client closes too.
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
            # response.reasoning joins only non-empty AssistantReasoning parts,
            # so a provider that returns a reasoning part with a signature but
            # deliberately empty/opaque text (Claude Sonnet 5/Opus 5 via Azure
            # or Bedrock) reads as no reasoning at all through that property.
            # Check for the part's presence directly, matching how session
            # checks in _session.py already detect it.
            reasoning_parts = [part for part in response.parts if part.kind == "reasoning"]
            # Confirmed live: Kimi/Qwen bill nonzero usage.reasoning_tokens on
            # routes that never put a reasoning item on the wire at all --
            # capture_parts has nothing to attach, so parts stays empty. Billed
            # tokens alone are not evidence; only a captured part is (visible
            # text or not, per the empty-text Claude case above).
            reasoning = bool(reasoning_parts)
            # Anthropic withholds the visible thinking text in (at least) two
            # distinct wire shapes, both preserved on .native by chat_parts.py:
            # a genuine redacted_thinking block (opaque "data" blob, no text
            # field at all), and a normal *signed* "thinking" block whose
            # "thinking" text happens to be empty (observed live for Claude
            # Sonnet 5/Opus 5 via this Bedrock/Azure route) — chat_parts.py
            # accepts empty text there as long as a real signature is present.
            # Either way there's no visible text left for litellm to estimate
            # a token count from — surface that instead of a bare, misleading 0.
            reasoning_encrypted = False
            reasoning_encrypted_bytes = None

            def _encrypted_blob_size(data: str) -> int:
                # The decoded byte count of an opaque encrypted blob is a
                # rough size signal only — a proxy for how much reasoning
                # state it carries, not a token count. Providers give no way
                # to convert this into an actual reasoning-token figure.
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
                        # chat_parts.py already popped "thinking" text out of
                        # this dict, so part.text (not the dict) is the only
                        # place left to check whether it was actually empty.
                        # A signature's length is fixed by the signing
                        # scheme, not by how much was thought — there is no
                        # size signal to report here, only that the text was
                        # withheld.
                        reasoning_encrypted = True
                    continue
                # openai/azure Chat Completions routes (chat_parts.py) store
                # their reasoning item under "reasoning_items", a third
                # shape distinct from both "thinking_blocks" above and the
                # unwrapped Responses-style native below — same
                # encrypted_content field, different wrapper key.
                #
                # An OpenAI-style reasoning item can carry a visible summary
                # (part.text) alongside encrypted_content at the same time —
                # the encrypted blob is opaque replay state, not proof the
                # readable text was withheld. Only flag "withheld" when there
                # is no visible text, matching the "thinking" branch above.
                block = part.native.get("reasoning_items")
                if isinstance(block, Mapping):
                    encrypted_content = block.get("encrypted_content")
                    if isinstance(encrypted_content, str) and encrypted_content and not part.text:
                        reasoning_encrypted = True
                        reasoning_encrypted_bytes = (reasoning_encrypted_bytes or 0) + (
                            _encrypted_blob_size(encrypted_content)
                        )
                    continue
                # Responses-style routes (response_parts.py) store the raw
                # output item on .native directly, not wrapped under
                # "thinking_blocks" or "reasoning_items" — e.g. {"type":
                # "reasoning", "encrypted_content": "..."}. Detect that shape
                # too, or Responses/OpenAI-style encrypted reasoning always
                # reads as "not encrypted" here regardless of what the
                # provider sent. Same visible-summary-plus-encrypted-blob
                # caveat as the reasoning_items branch above.
                encrypted_content = part.native.get("encrypted_content")
                if isinstance(encrypted_content, str) and encrypted_content and not part.text:
                    reasoning_encrypted = True
                    reasoning_encrypted_bytes = (reasoning_encrypted_bytes or 0) + (
                        _encrypted_blob_size(encrypted_content)
                    )
            # litellm's text-length reasoning_tokens estimate (see the
            # redacted_thinking/signed-empty-thinking comment above) only
            # exists in its Anthropic/Bedrock transformation code — for every
            # other provider it makes no estimate at all, even when real,
            # non-empty reasoning text came back (observed live for Qwen and
            # DeepSeek routes on this gateway). We already have that text
            # in hand; report its length ourselves rather than nothing. Never
            # the text itself — only its size, same privacy stance as the
            # encrypted-blob byte count above.
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
                    # Keep observed carrier/settings/envelope facts even when
                    # original-model validation or final-stop enforcement fails.
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
                # A different candidate, once, charged as another capped request.
                pending.appendleft(probe)
                yield ProbeUpdate(probe.name, deepcopy(record))
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
            # Score only the public final answer, never retain reasoning/reply text.
            # Correctness is separate from evidence that a setting was obeyed.
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
    """Share reservations across bounded stateless cases, with no retries."""
    from ._diagnostics import timeout_details

    entry = configure_entry(proposal.entry)
    records = entry["provenance"]["probes"]
    charged = 0
    accepted = []
    baseline_ok = False
    for probe in proposal.probes:
        previous = records.get(probe.name, {})
        # Only routing acceptance may be reused; all accuracy/media cases are fresh.
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
    """Try one routing request per interface, sharing one budget and no retries.

    The frontend warns before calling this paid operation. It can display each
    progress event and offer only the accepted interfaces. A timeout, auth error
    or rejection is recorded, never relabelled as unsupported. Different styles
    have different authentication conventions, so failure on one does not stop
    the other bounded attempts. Close the iterator when cancelling. The selected
    result can be passed as plan(existing_entry=...) to reuse its routing check.

    ``reasoning_template``, when given, also runs one reasoning-eliciting probe
    per interface (the same puzzle prompt ``plan`` uses for its own
    reasoning-level checks), built per style via ``reasoning_settings`` so each
    interface gets its own correct wire shape rather than one shape reused
    everywhere. This makes ``InterfaceResult.reasoning_observed`` and
    ``.recommended_style`` reflect which interfaces actually surface reasoning
    for this model, not just which accept an ordinary call. A route can accept
    calls on every interface while only one of them ever puts a "reasoning"
    item on the wire -- e.g. Responses needs ``reasoning.summary`` requested
    (see ``reasoning_settings``) for some Hub routes that Chat's plain
    ``reasoning_content`` surfaces without any extra parameter. Only the
    "effort" template is meaningful across chat and responses together; an
    anthropic-only comparison needs a matching template (adaptive/budget/
    thinking) passed explicitly.

    ``endpoints`` overrides ``api_base`` per style. Decision endpoints are full
    URLs, never inferred from a chat base or by appending a suffix. Mixed styles
    require an explicit mapping entry for every decision style; a single decision
    style can still use ``api_base``. All supplied endpoints are validated before
    any request. With a nonempty ``api_key`` or configured ``api_key_env``,
    mapped endpoints on a different origin (scheme, host, effective port) require
    explicit ``allow_cross_origin_credentials=True``. CLI callers opt in with
    ``--share-interface-key``. No credential is shared across origins by default;
    unauthenticated comparisons and same-origin path differences are permitted.
    Reasoning templates apply only to chat-family interfaces and
    are ignored in a decision-only comparison.
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
            # The decision probe has no reply-token field to cap.
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
                    timeout_seconds=timeout_seconds,
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


def write(entry: dict, path: Path, *, alias: str) -> None:
    """Replace one model; frontends warn before calling this for an existing alias.

    Splice the selected YAML entry instead of reformatting the whole file, so
    unrelated entries and comments remain intact. The final replace is atomic.
    """
    entry = configure_entry(entry)
    # Persist discovered settings only. Probe evidence, request bodies, catalogue
    # data and warnings belong to the connect session (surfaces via
    # ``nooa connect check --json``); embedding them here would write the whole
    # history of the connect call into the registry.
    entry.pop("provenance", None)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock a stable sidecar inode, not the registry inode replaced atomically.
    # Keep the lock file: unlinking it would let a third writer bypass waiters.
    import fcntl

    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _write_entry(entry, path, alias=alias)


def _write_entry(entry: dict, path: Path, *, alias: str) -> None:
    """Perform one locked read/modify/replace, retaining a concurrent-edit check."""
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
        # Block mapping end marks include following comments. Descend to the
        # last value so comments preceding a neighbor stay with that neighbor.
        if isinstance(node, yaml.MappingNode) and node.value and not node.flow_style:
            return content_end(node.value[-1][1])
        if isinstance(node, yaml.SequenceNode) and node.value and not node.flow_style:
            return content_end(node.value[-1])
        return node.end_mark

    if models_node is not None and isinstance(models_node, yaml.ScalarNode):
        # YAML's null mapping ("models:") is a valid empty registry.
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
        # An inline mapping must be replaced as a unit, not patched by lines.
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
        if (path.read_bytes() if path.exists() else None) != original:
            raise ValueError("Registry changed during write; reload before retrying")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
