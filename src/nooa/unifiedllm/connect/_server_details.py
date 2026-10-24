# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded selected-model facts. No inference, SDK dispatch, or global/key cache."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True)
class ModelDetails:
    """Allowlisted facts and field-level evidence, separate from request settings.

    ``requests`` counts attempted GETs in this call (not a reused listing).
    Missing metadata is an ordinary result, never an inference-test failure.
    """

    model_id: str
    model: dict
    records: tuple[dict, ...] = ()
    notices: tuple[str, ...] = ()
    requests: int = 0
    catalogue_model: dict | None = None


_INHERIT_KEY = object()


def token_limit(value):
    """Published SDK schemas may encode token ceilings as integral floats."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    return int(value) if value > 0 else None


def positive_integer(value):
    return token_limit(value) is not None


def safe_string(value, maximum=500):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= maximum
        and not any(ord(c) < 32 for c in value)
    )


def string_list(value):
    return isinstance(value, list) and len(value) <= 100 and all(safe_string(v, 200) for v in value)


def listing_facts(item):
    """Shape only published listing identity, limits and descriptive hints."""
    from ._advertised_media import media_metadata

    result = {"id": item["id"]}
    for name in ("context_window", "context_length", "max_input_tokens", "max_output_tokens"):
        if positive_integer(item.get(name)):
            result[name] = token_limit(item[name])
    for name in ("name", "display_name", "description", "client_type", "model_type"):
        if safe_string(item.get(name), 4096):
            result[name] = item[name]
    for name in (
        "input_modalities",
        "output_modalities",
        "supported_interfaces",
        "supported_parameters",
    ):
        if string_list(item.get(name)):
            result[name] = list(item[name])
    architecture = item.get("architecture")
    if isinstance(architecture, dict):
        hints = {
            k: list(architecture[k])
            for k in ("input_modalities", "output_modalities")
            if string_list(architecture.get(k))
        }
        if hints:
            result["architecture"] = hints
    media = media_metadata(item)
    media_architecture = media.pop("architecture", {})
    result.update(media)
    if media_architecture:
        result.setdefault("architecture", {}).update(media_architecture)
    from ._reasoning import reasoning_metadata

    reasoning = reasoning_metadata(item)
    if reasoning:
        result.setdefault("capabilities", {}).update(reasoning["capabilities"])
    return result


def openrouter_facts(item):
    """Public catalogue facts only, not provider routing/configuration objects."""
    import math

    from ._advertised_media import media_metadata

    if not isinstance(item, dict) or not safe_string(item.get("id")):
        raise ValueError("Catalogue models must have a valid non-empty string id")
    result = {"id": item["id"]}
    for key in ("name", "description"):
        if safe_string(item.get(key), 4096):
            result[key] = item[key]
    if positive_integer(item.get("context_length")):
        result["context_length"] = token_limit(item["context_length"])
    top = item.get("top_provider")
    result["top_provider"] = (
        {
            k: token_limit(top[k])
            for k in ("context_length", "max_completion_tokens")
            if positive_integer(top.get(k))
        }
        if isinstance(top, dict)
        else {}
    )
    if string_list(item.get("supported_parameters")):
        result["supported_parameters"] = list(item["supported_parameters"])
    result.update(media_metadata(item))
    architecture = item.get("architecture")
    if isinstance(architecture, dict) and string_list(architecture.get("output_modalities")):
        result.setdefault("architecture", {})["output_modalities"] = list(
            architecture["output_modalities"]
        )
    result["reasoning"] = {"supported_efforts": []}
    reasoning = item.get("reasoning")
    if isinstance(reasoning, dict):
        if string_list(reasoning.get("supported_efforts")):
            result["reasoning"]["supported_efforts"] = list(reasoning["supported_efforts"])
        if safe_string(reasoning.get("default_effort"), 200):
            result["reasoning"]["default_effort"] = reasoning["default_effort"]
        if isinstance(reasoning.get("default_enabled"), bool):
            result["reasoning"]["default_enabled"] = reasoning["default_enabled"]
    result["default_parameters"] = {}
    defaults = item.get("default_parameters")
    if isinstance(defaults, dict) and positive_integer(defaults.get("max_tokens")):
        result["default_parameters"]["max_tokens"] = token_limit(defaults["max_tokens"])
    pricing = item.get("pricing")
    if isinstance(pricing, dict):
        prices = {}
        for key in ("prompt", "completion"):
            value = pricing.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                try:
                    number = float(value)
                except (ValueError, OverflowError):
                    continue
                if (
                    math.isfinite(number)
                    and number >= 0
                    and (not isinstance(value, str) or len(value) <= 100)
                ):
                    prices[key] = value
        if prices:
            result["pricing"] = prices
    return result


async def bounded_json(client, url, *, headers=None, params=None, byte_budget=None):
    """One GET, at most 5 MiB; no redirects, retries or provider body errors."""
    async with client.stream("GET", url, headers=headers or {}, params=params or {}) as response:
        if not response.is_success:
            raise ValueError(f"Model details unavailable (HTTP {response.status_code}).")
        chunks, size = [], 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if byte_budget is not None:
                byte_budget[0] -= len(chunk)
            if size > 5 * 1024 * 1024 or (byte_budget is not None and byte_budget[0] < 0):
                raise ValueError("Model details exceed 5 MiB.")
            chunks.append(chunk)
    try:
        data = json.loads(b"".join(chunks))
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Model details did not return valid JSON.") from None
    if not isinstance(data, dict):
        raise ValueError("Model details have an unexpected shape.")
    return data


def merge_group(model, group, source):
    """Group ceilings only tighten listing limits; false flags remain descriptive.

    LiteLLM group limits/flags aggregate backends. They are not route permissions
    or request parameter templates. ``evaluation`` is a class hint, not a URL.
    """
    facts, fields = {}, {}
    for name in ("max_input_tokens", "max_output_tokens"):
        value = group.get(name)
        if positive_integer(value):
            old = model.get(name)
            model[name] = (
                min(token_limit(old), token_limit(value))
                if positive_integer(old)
                else token_limit(value)
            )
            facts[name] = value
            fields[name] = "data[]." + name
    # A listing total/input bound must never be raised by aggregate group data.
    contexts = [
        token_limit(model[k])
        for k in ("context_window", "context_length")
        if positive_integer(model.get(k))
    ]
    if contexts and positive_integer(model.get("max_input_tokens")):
        model["max_input_tokens"] = min(token_limit(model["max_input_tokens"]), *contexts)
    for name in (
        "supports_vision",
        "supports_function_calling",
        "supports_reasoning",
        "supports_parallel_function_calling",
        "supports_system_messages",
        "supports_response_schema",
        "supports_prompt_caching",
        "supports_audio_input",
        "supports_audio_output",
        "supports_web_search",
    ):
        if isinstance(group.get(name), bool):
            facts[name] = group[name]
            fields[name] = "data[]." + name
    for name in ("supported_openai_params", "supported_reasoning_efforts"):
        if string_list(group.get(name)):
            facts[name] = list(group[name])
            fields[name] = "data[]." + name
    if safe_string(group.get("mode"), 100):
        facts["mode"] = group["mode"]
        fields["mode"] = "data[].mode"
        if group["mode"] == "evaluation":
            model.setdefault("model_type", "decision")
    return {
        "source": source,
        "schema": "litellm.model_group.info",
        "id": model["id"],
        "facts": facts,
        "fields": fields,
    }


async def server_details(
    endpoint: str,
    model_id: str,
    *,
    api_style="chat",
    api_key=None,
    listing=None,
    models_url=None,
    listing_style=None,
    listing_api_key=_INHERIT_KEY,
    metadata=True,
    litellm=False,
    catalogue_id=None,
) -> ModelDetails:
    """Fetch server facts for one exact model, optionally validate LiteLLM metadata.

    Reuse a ``Discovery`` from this exact API base and listing URL without another
    models GET. Native OpenAI/Anthropic use the existing listing/header contracts;
    OpenAI identity-only lists leave limits unknown. ``litellm=True`` authorizes
    a public OpenAPI product check and, only if it advertises the GET contract,
    one authenticated group GET. The configured base must be root or end in /v1;
    remove only that suffix, preserving a deployment prefix. No arbitrary URL
    guessing, schema-provided servers, admin endpoints or cross-origin keys.
    Decisions need an explicit listing URL. False flags/efforts stay descriptive.
    For the exact OpenRouter API base, public catalogue records are fetched with
    no key; ``catalogue_id`` may explicitly select an additional metadata mapping
    in that same GET, without changing the selected model facts or wire ID.
    Each component is bounded to 30 seconds (60 seconds overall), 5 MiB,
    and 5,000 records. Reused records must come from the caller's current key
    scope; the adapter maintains no shared visibility/key cache.
    ``metadata=False`` is a complete zero-GET bypass, including supplied records.
    There is no persistent cache; callers can reuse this invocation's result.
    """
    from . import (
        CATALOGUE_URL,
        DECISION_API_STYLES,
        DiscoveryError,
        catalogue,
        discover,
        normalize_endpoint,
        validate_models_url,
    )

    def details(*args, **kwargs):
        # Protect both current credentials at the public library boundary, even
        # when a metadata string reflects a differently scoped listing key.
        from dataclasses import asdict

        from ._diagnostics import scrub_report

        data = asdict(ModelDetails(*args, **kwargs))
        for key in (api_key, listing_api_key):
            if isinstance(key, str):
                data = scrub_report(data, api_key=key)
        data["records"] = tuple(data["records"])
        data["notices"] = tuple(data["notices"])
        return ModelDetails(**data)

    base = normalize_endpoint(endpoint)
    if not safe_string(model_id):
        raise ValueError("Supply one exact model ID (at most 500 characters).")
    if not metadata:
        return details(model_id, {"id": model_id})
    if catalogue_id is not None and not safe_string(catalogue_id):
        raise ValueError("Supply one exact catalogue ID (at most 500 characters).")
    selected, records, notices, requests = {"id": model_id}, [], [], 0
    wanted_url = validate_models_url(models_url) if models_url else None
    if listing_api_key is _INHERIT_KEY:
        if wanted_url:

            def origin(url):
                parts = urlsplit(url)
                return (
                    parts.scheme,
                    parts.hostname,
                    parts.port or (443 if parts.scheme == "https" else 80),
                )

            if origin(wanted_url) != origin(base):
                raise ValueError(
                    "A different catalogue origin requires an explicit listing key or None."
                )
        listing_api_key = api_key
    if base + "/models" == CATALOGUE_URL and wanted_url in {None, CATALOGUE_URL}:
        # Public data for this exact configured route; no inference credential.
        if listing is None:
            try:
                models = await catalogue()
            except (ValueError, httpx.HTTPError, TimeoutError):
                return details(
                    model_id,
                    selected,
                    (
                        {
                            "source": CATALOGUE_URL,
                            "schema": "openrouter.models",
                            "id": model_id,
                            "facts": {},
                            "fields": {},
                            "outcome": "unavailable",
                        },
                    ),
                    ("Public model details unavailable; continue with known facts or test.",),
                    1,
                )
            public_model = next((m for m in models if m["id"] == model_id), None)
            mapped_model = (
                next((m for m in models if m["id"] == catalogue_id), None)
                if catalogue_id
                else public_model
            )
            requests = 1
        else:
            if normalize_endpoint(listing.api_base) != base or listing.models_url != CATALOGUE_URL:
                raise ValueError("Model listing belongs to a different connection.")
            raw = next((m for m in listing.models if m.get("id") == model_id), None)
            public_model = openrouter_facts(raw.get("public_model", raw)) if raw else None
            mapping = (
                next((m for m in listing.models if m.get("id") == catalogue_id), None)
                if catalogue_id
                else raw
            )
            mapped_model = (
                openrouter_facts(mapping.get("public_model", mapping)) if mapping else None
            )
        if public_model:
            selected = listing_facts(public_model)
            contexts = [
                v
                for v in (
                    public_model.get("context_length"),
                    public_model.get("top_provider", {}).get("context_length"),
                )
                if positive_integer(v)
            ]
            if contexts:
                selected["context_window"] = min(contexts)
            maximum = public_model.get("top_provider", {}).get("max_completion_tokens")
            if maximum:
                selected["max_output_tokens"] = maximum
            from ._diagnostics import scrub_report

            public_model = scrub_report(public_model, api_key=api_key)
            selected = scrub_report(selected, api_key=api_key)
        records = (
            {
                "source": CATALOGUE_URL,
                "schema": "openrouter.models",
                "id": model_id,
                "facts": deepcopy(public_model or {}),
                "fields": {k: "data[]." + k for k in (public_model or {}) if k != "id"},
            },
        )
        from ._diagnostics import scrub_report

        mapped_model = scrub_report(mapped_model, api_key=api_key)
        mapped_model = scrub_report(mapped_model, api_key=listing_api_key)
        return details(model_id, selected, records, (), requests, mapped_model)
    if listing is not None:
        listed_base = normalize_endpoint(listing.api_base)
        compatible_base = listed_base == base or (
            not urlsplit(base).path and listed_base == base + "/v1"
        )
        if api_style in DECISION_API_STYLES and wanted_url is not None:
            left, right = urlsplit(listed_base), urlsplit(base)
            compatible_base = compatible_base or (
                left.scheme,
                left.hostname,
                left.port or (443 if left.scheme == "https" else 80),
            ) == (
                right.scheme,
                right.hostname,
                right.port or (443 if right.scheme == "https" else 80),
            )
        expected_source = wanted_url or listed_base + "/models"
        effective_source = (
            validate_models_url(listing.models_url)
            if listing.models_url
            else listed_base + "/models"
        )
        if not compatible_base or effective_source != expected_source:
            raise ValueError("Model listing belongs to a different connection or listing URL.")
    elif api_style not in DECISION_API_STYLES or models_url:
        try:
            listing = await discover(
                base,
                api_style=api_style,
                api_key=listing_api_key,
                models_url=models_url,
                listing_style=listing_style,
            )
        except DiscoveryError as exc:
            requests += exc.request_count
            notices.append(str(exc))
            # Auth failure never triggers a second request with the same key.
            return details(model_id, selected, (), tuple(notices), requests)
        requests += listing.request_count
    if listing is not None:
        from ._diagnostics import scrub_report

        raw = next((m for m in listing.models if m.get("id") == model_id), selected)
        selected = listing_facts(raw)
        selected = scrub_report(selected, api_key=api_key)
        selected = scrub_report(selected, api_key=listing_api_key)
        source = (
            validate_models_url(listing.models_url)
            if listing.models_url
            else normalize_endpoint(listing.api_base) + "/models"
        )
        from ._advertised_media import media_metadata

        if media_metadata(selected):
            selected["media_source"] = source
        fields = {k: "data[]." + k for k in selected if k not in {"id", "media_source"}}
        # Accept only locally generated allowlisted paths, never arbitrary source URLs.
        for key, value in (
            raw.get("metadata_fields", {}).items()
            if isinstance(raw.get("metadata_fields"), dict)
            else ()
        ):
            if key in selected and value in {
                "data[].context_window",
                "data[].context_length",
                "data[].max_model_len",
                "data[].max_input_tokens",
                "data[].max_output_tokens",
                "data[].max_completion_tokens",
                "data[].max_tokens",
                "data[].top_provider.context_length",
                "data[].top_provider.max_completion_tokens",
            }:
                fields[key] = value
        for kind in ("image_input", "pdf_input"):
            if kind in selected.get("capabilities", {}):
                fields["capabilities." + kind + ".supported"] = (
                    "data[].capabilities." + kind + ".supported"
                )
        from ._reasoning import metadata_fields

        fields.update(metadata_fields(selected))
        records.append(
            {
                "source": source,
                "schema": "models",
                "id": model_id,
                "facts": deepcopy(selected),
                "fields": fields,
            }
        )
    if not litellm or api_style in DECISION_API_STYLES:
        return details(model_id, selected, tuple(records), tuple(notices), requests)
    path = urlsplit(base).path
    if path and not path.endswith("/v1"):
        notices.append("Use the configured LiteLLM root or /v1 API base for server details.")
        return details(model_id, selected, tuple(records), tuple(notices), requests)
    root = base.removesuffix("/v1")
    schema_url, group_url = root + "/openapi.json", root + "/model_group/info"
    try:
        async with (
            asyncio.timeout(30),
            httpx.AsyncClient(timeout=15, follow_redirects=False) as client,
        ):
            byte_budget = [5 * 1024 * 1024]
            requests += 1
            schema = await bounded_json(
                client, schema_url, byte_budget=byte_budget
            )  # Public identification: no key.
            info, paths = schema.get("info"), schema.get("paths")
            route = paths.get("/model_group/info") if isinstance(paths, dict) else None
            operation = route.get("get") if isinstance(route, dict) else None
            parameters = operation.get("parameters") if isinstance(operation, dict) else None
            if not (
                isinstance(schema.get("openapi"), str)
                and schema["openapi"].startswith("3.")
                and isinstance(info, dict)
                and info.get("title") == "LiteLLM API"
                and isinstance(operation, dict)
                and isinstance(parameters, list)
                and len(parameters) <= 100
                and any(
                    isinstance(p, dict)
                    and p.get("name") == "model_group"
                    and p.get("in") == "query"
                    for p in parameters
                )
            ):
                raise ValueError(
                    "Server does not advertise LiteLLM model-group details; continuing with known facts."
                )
            requests += 1
            payload = await bounded_json(
                client,
                group_url,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                params={"model_group": model_id},
                byte_budget=byte_budget,
            )
            data = payload.get("data")
            if not isinstance(data, list) or len(data) > 5000:
                raise ValueError("Model-group details have an unexpected shape.")
            groups = [g for g in data if isinstance(g, dict) and g.get("model_group") == model_id]
            if len(groups) != 1:
                raise ValueError("Server details did not contain one exact selected model group.")
            records.append(merge_group(selected, groups[0], group_url))
            records[-1]["product"] = {"title": "LiteLLM API"}
            if safe_string(info.get("version"), 100):
                records[-1]["product"]["version"] = info["version"]
    except (httpx.HTTPError, TimeoutError):
        notices.append(
            "Model details connection failed or timed out; continue with known facts or test."
        )
    except ValueError as exc:
        notices.append(str(exc))
    from ._diagnostics import scrub_report

    return details(
        model_id,
        scrub_report(selected, api_key=api_key),
        tuple(scrub_report(records, api_key=api_key)),
        tuple(notices),
        requests,
    )
