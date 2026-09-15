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
import json
import math
import os
import re
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
TEMPLATES = ("effort", "adaptive", "budget", "toggle", "thinking")
DEFAULT_CHECK_BUDGET = 131072
DEFAULT_REASONING_OUTPUT_TOKENS = 4096
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
    if isinstance(param, str) and not (
        param == "include" or param.startswith("include[") or param == "reasoning.encrypted_content"
    ):
        return False  # Rejected history/model fields are not include-option rejection.
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


_RESERVED = {
    "model",
    "messages",
    "input",
    "api_base",
    "base_url",
    "api_key",
    "custom_llm_provider",
    "extra_body",
    "client",
}
_PATHS = {"chat": "chat/completions", "responses": "responses", "anthropic": "messages"}


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


@dataclass(frozen=True)
class ProbeUpdate:
    """A request starting or finishing, for frontend progress display."""

    name: str
    outcome: dict[str, Any]


@dataclass(frozen=True)
class Discovery:
    """Models advertised by the selected endpoint, not a public catalogue."""

    api_base: str
    models: tuple[dict[str, Any], ...]


class DiscoveryError(ValueError):
    """Safe discovery failure; frontends can offer a new key on 401 or 403."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


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


async def discover(
    endpoint: str, *, api_style: str = "chat", api_key: str | None = None
) -> Discovery:
    """List endpoint models without saving credentials or making generation calls.

    Root URLs try /models, then /v1/models only on 404. Anthropic pagination
    uses the same fetcher. Limits apply to the whole discovery: 30 seconds,
    5 MiB and 5,000 entries. Redirects and automatic retries are disabled.
    """
    if api_style not in _PATHS:
        raise ValueError("api_style must be chat, responses or anthropic")
    base = normalize_endpoint(endpoint)
    root = not urlsplit(base).path
    models: dict[str, dict] = {}
    cursors: set[str] = set()
    params = {"limit": "100"} if api_style == "anthropic" else {}
    size = 0
    count = 0
    try:
        async with (
            asyncio.timeout(30),
            httpx.AsyncClient(timeout=15, follow_redirects=False) as client,
        ):
            while True:
                async with client.stream(
                    "GET", base + "/models", headers=_headers(api_style, api_key), params=params
                ) as response:
                    if response.status_code == 404 and root:
                        base += "/v1"
                        root = False
                        continue
                    if not response.is_success:
                        raise DiscoveryError(
                            f"Model discovery returned HTTP {response.status_code}",
                            status_code=response.status_code,
                        )
                    chunks = []
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 5 * 1024 * 1024:
                            raise DiscoveryError("Model discovery exceeds 5 MiB")
                        chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list):
                    raise DiscoveryError("Model discovery must contain a data list")
                count += len(data)
                if count > 5000:
                    raise DiscoveryError("Model discovery exceeds 5,000 entries")
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
                                "max_input_tokens",
                            ),
                        ),
                        ("max_output_tokens", ("max_output_tokens", "max_completion_tokens")),
                    ):
                        for candidate in candidates:
                            value = item.get(candidate)
                            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                                model[field] = value
                                break
                    models[name] = model
                if api_style != "anthropic" or not payload.get("has_more"):
                    break
                cursor = payload.get("last_id")
                if (
                    not isinstance(cursor, str)
                    or not cursor
                    or cursor in cursors
                    or len(cursors) >= 50
                ):
                    raise DiscoveryError("Invalid or excessive model discovery pagination")
                cursors.add(cursor)
                params["after_id"] = cursor
    except (httpx.HTTPError, TimeoutError):
        raise DiscoveryError("Model discovery connection failed or timed out") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise DiscoveryError("Model discovery did not return valid JSON") from None
    if not models:
        raise DiscoveryError("Endpoint advertised no usable model IDs; supply a model explicitly")
    return Discovery(base, tuple(models[name] for name in sorted(models)))


async def catalogue() -> list[dict]:
    """Fetch public metadata without sending the endpoint's credentials."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        response = await client.get(CATALOGUE_URL)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ValueError("Catalogue response must contain a list of model objects")
        return data


def match_models(model: str, models: list[dict]) -> list[dict]:
    """Suggest up to three matches. The frontend must confirm a candidate."""

    def normalized(value):
        return re.sub(r"[-_.]", "", value.lower())

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


def reasoning_settings(template: str, style: str, level: str, *, budget: int = 4096) -> dict:
    """Build an onboarding candidate, not a claim about a model's support."""
    if template == "effort":
        return (
            {"reasoning": {"effort": level}}
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


def configure_entry(entry: dict, *, reply_tokens: int | None = None) -> dict:
    """Apply safe persisted defaults to a detached entry, without doing any I/O.

    ``max_tokens`` is the runtime cap on every interface. Published ceilings
    remain provenance, not request allocations. ``include: []`` is an explicit
    opt-out from carrying encrypted reasoning on stateless Responses calls.
    """
    result = deepcopy(entry)
    extra = result.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise ValueError("extra_body must be a mapping")
    responses = result.get("api_style") == "responses" or result.get("client_type") == "responses"
    managed = {"max_tokens", "max_output_tokens", "max_completion_tokens"}
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
    ceiling = result.pop("max_output_tokens", None)
    limits = provenance.setdefault("catalogue_limits", {})
    if not isinstance(limits, dict):
        raise ValueError("catalogue_limits must be a mapping")
    if isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0:
        limits["max_completion_tokens"] = ceiling
    ceiling = limits.get("max_completion_tokens")
    known_limits = [
        v
        for v in (ceiling, result.get("context_window"))
        if isinstance(v, int) and not isinstance(v, bool) and v > 0
    ]
    bound = min(known_limits) if known_limits else None
    cap = reply_tokens if reply_tokens is not None else result.get("max_tokens")
    source = "user" if reply_tokens is not None else "entry"
    if cap is None:
        cap = min(32768, bound) if bound else 32768
        source = "connect_default"
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError("Reply limit max_tokens must be a positive integer")
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
    return result


def plan(
    alias: str,
    model: str,
    api_style: str,
    api_base: str,
    api_key_env: str,
    *,
    catalogue: dict | None = None,
    reasoning_levels: dict | None = None,
    budget_tokens: int = DEFAULT_CHECK_BUDGET,
    output_tokens: int = 200,
    reasoning_output_tokens: int = DEFAULT_REASONING_OUTPUT_TOKENS,
    existing_entry: dict | None = None,
    session_checks: bool = False,
    reply_tokens: int | None = None,
) -> ConnectPlan:
    """Prepare requests without reading credentials, files or network resources.

    ``model`` is the exact endpoint model ID, without a LiteLLM prefix. The
    returned entry adds the prefix needed by today's runtime. Each request reserves
    its output cap plus 512 estimated input tokens. Reported usage can increase
    that charge. Servers can ignore caps, so this is not a billing limit.
    """
    if api_style not in _PATHS:
        raise ValueError("api_style must be chat, responses or anthropic")
    api_base = normalize_endpoint(api_base)
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
    vendor = "anthropic" if api_style == "anthropic" else "openai"
    entry: dict[str, Any] = {
        "model_name": f"{vendor}/{model}",
        "client_type": "responses" if api_style == "responses" else "completion",
        "api_style": api_style,
        # The Messages runtime adds /v1/messages; Chat/Responses expect an API base.
        "api_base": api_base.removesuffix("/v1") if api_style == "anthropic" else api_base,
        "api_key_env": api_key_env,
        "replay_vendor": vendor,
    }
    provenance: dict[str, Any] = {
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
    if catalogue is not None:
        entry["underlying_model"] = catalogue["id"]
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
    if api_style == "responses":
        entry["store"] = False
        entry["include"] = ["reasoning.encrypted_content"]
        provenance["encrypted_reasoning"] = {"source": "connect", "outcome": "not_probed"}
        route_keys = ("model_name", "api_base", "api_key_env", "api_style")
        if (
            existing_entry
            and all(existing_entry.get(key) == entry.get(key) for key in route_keys)
            and existing_entry.get("include") == []
            and existing_entry.get("provenance", {}).get("encrypted_reasoning", {}).get("outcome")
            == "rejected"
        ):
            entry["include"] = []
            provenance["encrypted_reasoning"] = deepcopy(
                existing_entry["provenance"]["encrypted_reasoning"]
            )
    token_key = "max_output_tokens" if api_style == "responses" else "max_tokens"
    body = {
        "model": model,
        token_key: output_tokens,
        "input" if api_style == "responses" else "messages": [
            {"role": "user", "content": "Compute 17 * 19. Reply with the number."}
        ],
    }
    if api_style == "responses":
        body["store"] = False
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
        probes.append(Probe(f"level:{label}", level_body, reasoning_output_tokens + 512))
    entry["provenance"] = provenance
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
    entry = configure_entry(entry, reply_tokens=reply_tokens)
    provenance = entry["provenance"]
    for probe in probes:
        if probe.name.startswith("level:"):
            probe.body.update(entry["reasoning_levels"][probe.name.removeprefix("level:")])
    if existing_entry:
        # Each probe is also compared to its exact request in run_steps. Adding
        # a level must not invalidate an unchanged routing or tool check.
        keys = ("model_name", "api_base", "api_key_env", "api_style")
        if all(existing_entry.get(key) == entry.get(key) for key in keys):
            previous = existing_entry.get("provenance", {}).get("probes", {})
            provenance["probes"] = {
                probe.name: deepcopy(previous[probe.name])
                for probe in probes
                if probe.name in previous
            }
    estimate = sum(probe.token_estimate for probe in probes)
    price = None
    if catalogue and catalogue.get("pricing"):
        try:
            rates = [float(catalogue["pricing"][key]) for key in ("prompt", "completion")]
            if all(math.isfinite(rate) and rate >= 0 for rate in rates):
                price = (
                    len(probes) * 512 * rates[0]
                    + sum(probe.token_estimate - 512 for probe in probes) * rates[1]
                )
        except (ValueError, TypeError, KeyError):
            pass  # Missing catalogue prices are unknown, never zero.
    if session_checks:
        from nooa._connect_session import reply_budget

        estimate += 3 * reply_budget(entry, max(0, budget_tokens - estimate))[2]
        price = None  # The longer replay depends on the actual seed response.
    return ConnectPlan(alias, entry, tuple(probes), budget_tokens, estimate, price, session_checks)


async def _run_probe(alias: str, entry: dict, probe: Probe, api_key: str | None):
    """Check the unsaved entry through the same client factory agents use.

    Only the reply cap, timeout and retries are changed for bounded checks.
    Discovery and planning stay lightweight: runtime imports happen here only.
    Tool replies are inspected as data; Connect never executes model calls.
    """
    from nooa.unifiedllm import RetryConfig, Tool
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
        if any(probe.body.get(key) != value for key, value in settings.items()):
            raise ValueError("Reasoning settings changed after planning; make a new plan")
        for key in settings:
            params.pop(key, None)
        if {"max_tokens", "max_completion_tokens", "max_output_tokens"} & settings.keys():
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                params.pop(key, None)
        # A level can set the cap itself; do not pass a conflicting override.
        params["reasoning_level"] = label
    client = client_from_config(
        alias,
        entry,
        api_key=api_key,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        num_retries=0,
    )
    settings_sent = []

    async def capture(request):
        from nooa._connect_session import settings_on_wire

        if probe.name.startswith("level:"):
            settings_sent.append(settings_on_wire(settings, json.loads(request.content)))

    hooks = client._http.httpx_async.event_hooks["request"]
    hooks.append(capture)
    try:
        response = await client.acall(messages=messages, **params)
        return response, (len(settings_sent) == 1 and all(settings_sent))
    finally:
        hooks.remove(capture)
        await client.aclose()


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

    Stages are routing, tools, reasoning, session, or all. Calling this function
    explicitly approves that stage within the plan's existing limits. Results
    retain other stages' provenance, but selected checks never reuse old evidence.
    """
    if stage not in {"routing", "tools", "reasoning", "session", "all"}:
        raise ValueError("Unknown check stage")
    probes = tuple(
        p
        for p in proposal.probes
        if stage == "all" or p.name == stage or stage == "reasoning" and p.name.startswith("level:")
    )
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
        proposal, entry=entry, probes=probes, session_checks=stage in {"session", "all"}
    )
    return await run(selected, approved="all", api_key=api_key)


def diagnostic_prompt(
    stage: str, entry: dict, checks: dict, *, run_context: dict | None = None
) -> str:
    """Safe, copyable handoff for a person or agent; no credentials or raw bodies."""
    from nooa._connect_diagnostics import installation_context

    installation = installation_context()
    context = {
        key: entry[key]
        for key in (
            "model_name",
            "api_style",
            "client_type",
            "api_base",
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
    if "api_base" in context:
        try:
            context["api_base"] = normalize_endpoint(context["api_base"])
        except (TypeError, ValueError):
            context["api_base"] = "[invalid endpoint omitted]"
    outcomes = {
        name: {
            key: record[key]
            for key in (
                "outcome",
                "error",
                "detail",
                "reason",
                "checked_at",
                "elapsed_seconds",
                "error_chain",
                "timeout_kind",
                "request_shape",
                "status_code",
                "reasoning_observed",
                "answer_correct",
                "state_retained",
                "settings_retained",
                "settings_sent",
                "configured_reply_tokens",
                "tested_reply_tokens",
                "input_tokens",
                "cached_input_tokens",
                "finish_reason",
            )
            if key in record
        }
        for name, record in checks.items()
    }
    return (
        f"Diagnose and fix NOOA Connect stage {stage!r}. "
        "First read the nooa-agent-authoring skill and companion docs located by the "
        "installation references below. "
        "Inspect the configuration, credential lookup, and actual request construction. "
        "Model listing alone does not validate credentials. Distinguish rejection, missing "
        "evidence, and unsupported features; do not infer support from an HTTP success alone. "
        "Use nooa.connect library functions or nooa connect --stage to isolate the failure, "
        "then rerun the affected checks within configured limits. Never print or copy key "
        "values, reasoning text, opaque state, or raw error bodies. Preserve unrelated aliases "
        "and inspect layer precedence before editing the target file. This handoff does not "
        "authorize additional paid calls; confirm the remaining allowance before rerunning. "
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
    entry = configure_entry(proposal.entry)
    provenance = entry["provenance"]
    records = provenance["probes"]
    spent = 0
    stopped = False
    key = api_key
    pending = deque(proposal.probes)
    while pending:
        probe = pending.popleft()
        if entry.get("include") == [] and "include" in probe.body:
            probe = replace(probe, body={**probe.body, "include": []})
        previous = records.get(probe.name, {})
        if (
            previous.get("outcome") == "accepted"
            and previous.get("request") == probe.body
            and previous.get("client") == "unifiedllm"
            and (not probe.name.startswith("level:") or previous.get("settings_sent") is True)
        ):
            if probe.name == "routing" and entry.get("include"):
                provenance["encrypted_reasoning"]["outcome"] = "accepted"
            yield ProbeUpdate(
                probe.name, {**deepcopy(previous), "reason": "previous result reused"}
            )
            continue
        record: dict[str, Any] = {"outcome": "not_probed"}
        records[probe.name] = record
        if approved == "none" or approved == "minimal" and probe.name != "routing":
            record["reason"] = "not approved"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        if stopped or spent + probe.token_estimate > proposal.budget_tokens:
            record["reason"] = "previous check failed" if stopped else "budget exhausted"
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
        if required_cap not in probe.body or any(
            not isinstance(cap, int)
            or isinstance(cap, bool)
            or cap < 1
            or cap > probe.token_estimate - 512
            for cap in caps
        ):
            record["reason"] = "declared output cap exceeds the approved probe cap"
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        if key is None and entry["api_key_env"]:
            key = os.environ.get(entry["api_key_env"])
            if not key:
                raise ValueError(f"Set {entry['api_key_env']} before probing, or approve none")
        spent += probe.token_estimate
        yield ProbeUpdate(probe.name, {"outcome": "running"})
        started = time.monotonic()
        deadline = asyncio.timeout(120 if probe.name.startswith("level:") else 30)
        record["request_shape"] = {
            "api_style": style,
            "output_tokens": probe.body[required_cap],
            **{k: deepcopy(probe.body[k]) for k in ("store", "include") if k in probe.body},
        }
        try:
            async with deadline:
                response, settings_sent = await _run_probe(proposal.alias, entry, probe, key)
            usage = response.usage
            reasoning = bool(response.reasoning or (usage and usage.reasoning_tokens))
            tool = any(call.name == "probe_tool" for call in response.tool_calls)
            tokens = usage.input_tokens + usage.output_tokens if usage else 0
        except Exception as exc:
            from nooa._connect_diagnostics import timeout_details

            status = getattr(exc, "status_code", None)
            record["error"] = type(exc).__name__
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if isinstance(status, int):
                record["status_code"] = status
                record["outcome"] = "rejected" if status == 400 else "not_probed"
            record.update(timeout_details(exc, deadline_expired=deadline.expired()))
            if (
                probe.name == "routing"
                and entry.get("include") == ["reasoning.encrypted_content"]
                and _include_rejected(exc)
            ):
                rejection = _disable_encrypted_reasoning(entry, status)
                yield ProbeUpdate("encrypted_reasoning", deepcopy(rejection))
                # A different candidate, once, charged as another capped request.
                pending.appendleft(probe)
                yield ProbeUpdate(probe.name, deepcopy(record))
                continue
            stopped = probe.name == "routing" or status in {401, 403}
            yield ProbeUpdate(probe.name, deepcopy(record))
            continue
        record.update(
            outcome="accepted",
            elapsed_seconds=round(time.monotonic() - started, 3),
            client="unifiedllm",
            request=deepcopy(probe.body),
            reasoning_observed=reasoning,
            tool_observed=tool,
            reported_tokens=tokens,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None,
            finish_reason=response.finish_reason,
            checked_at=datetime.now(UTC).isoformat(),
        )
        if probe.name.startswith("level:"):
            # Score only the public final answer, never retain reasoning/reply text.
            # Correctness is separate from evidence that a setting was obeyed.
            record["answer_correct"] = (
                re.sub(r"[\s,]+", "", response.content or "").upper() == "BGDACEFH"
            )
            record["settings_sent"] = settings_sent
            if not settings_sent:
                record["outcome"] = "not_confirmed"
                record["reason"] = (
                    "The client did not send the requested settings; check parameter filtering"
                )
        spent += max(0, tokens - probe.token_estimate)
        yield ProbeUpdate(probe.name, deepcopy(record))
        if probe.name == "routing" and entry.get("include"):
            provenance["encrypted_reasoning"]["outcome"] = "accepted"
    if records.get("tools", {}).get("tool_observed"):
        entry["tools"] = True  # No call is not evidence that tools are unsupported.
    provenance["requests_accepted"] = [
        name for name, item in records.items() if item["outcome"] == "accepted"
    ]
    provenance["reasoning_observed"] = [
        name for name, item in records.items() if item.get("reasoning_observed")
    ]
    provenance["tokens_charged_to_budget"] = spent
    if proposal.session_checks:
        from nooa._connect_session import session_steps

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
    yield ConnectResult(proposal.alias, entry)


async def check_interfaces(
    alias: str,
    model: str,
    api_base: str,
    api_key_env: str,
    *,
    budget_tokens: int = 4096,
    output_tokens: int = 200,
    api_key: str | None = None,
) -> AsyncIterator[ProbeUpdate | InterfaceResult]:
    """Try one routing request per interface, sharing one budget and no retries.

    The frontend warns before calling this paid operation. It can display each
    progress event and offer only the accepted interfaces. A timeout, auth error
    or rejection is recorded, never relabelled as unsupported. Different styles
    have different authentication conventions, so failure on one does not stop
    the other bounded attempts. Close the iterator when cancelling. The selected
    result can be passed as plan(existing_entry=...) to reuse its routing check.
    """
    results = {}
    spent = 0
    for style in _PATHS:
        proposal = plan(
            alias,
            model,
            style,
            api_base,
            api_key_env,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
        )
        proposal = replace(
            proposal, probes=proposal.probes[:1], budget_tokens=max(0, budget_tokens - spent)
        )
        async with aclosing(run_steps(proposal, approved="minimal", api_key=api_key)) as steps:
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
    original = path.read_text() if path.exists() else None
    source = original or ""
    data = yaml.safe_load(source)
    if data is None:
        data = {}
    if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
        raise ValueError("Registry must be a mapping with a models mapping")
    dumped = yaml.safe_dump({alias: entry}, sort_keys=False, allow_unicode=True).rstrip()
    indented = "\n".join("  " + line for line in dumped.splitlines()) + "\n"
    document = yaml.compose(source)
    models_node = (
        next((value for key, value in document.value if key.value == "models"), None)
        if document
        else None
    )
    if models_node is None:
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
        lines = dumped.splitlines()
        replacement = "\n".join([lines[0], *("  " + line for line in lines[1:])])
        suffix = source[value.end_mark.index :]
        if suffix and not suffix.startswith("\n"):
            replacement += "\n" + " " * value.end_mark.column
        text = source[: key.start_mark.index] + replacement + suffix
    else:
        insertion = models_node.end_mark.index
        prefix = "" if insertion == 0 or source[insertion - 1] == "\n" else "\n"
        text = source[:insertion] + prefix + indented + source[insertion:]
    expected = {**data, "models": {**data.get("models", {}), alias: entry}}
    if yaml.safe_load(text) != expected:
        raise ValueError("Could not construct the registry update without changing other entries")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as temporary:
        temporary.write(text)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if (path.read_text() if path.exists() else None) != original:
            raise ValueError("Registry changed during write; reload before retrying")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
