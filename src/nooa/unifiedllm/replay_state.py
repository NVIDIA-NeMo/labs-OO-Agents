# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issuer-scoped capture and replay of opaque OpenAI reasoning state.

The event IR treats provider state as an opaque dictionary. This module is the
only code that opens its NOOA envelope or places the payload on provider wire
messages. Unknown issuers and compatibility mismatches fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import litellm

from nooa._llm_state import LLM_STATE_KEY, carried_state

logger = logging.getLogger(__name__)

_STATE_VERSION = 1
_CHAT_FORMAT = "litellm-chat"
_RESPONSES_FORMAT = "openai-responses"
_ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def response_item_type(item: Any) -> str | None:
    value = _field(item, "type")
    return value if isinstance(value, str) else None


def opaque_item(item: Any) -> Any:
    """Detach one provider-owned item for durable storage."""
    # Inspect the type: permissive mocks/proxies synthesize arbitrary instance
    # attributes and can otherwise recurse forever here.
    if callable(getattr(type(item), "model_dump", None)):
        return opaque_item(item.model_dump(exclude_none=True))
    if isinstance(item, dict):
        return {key: opaque_item(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [opaque_item(value) for value in item]
    return copy.deepcopy(item)


def _normalized_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "default"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _effective_endpoint(provider: str | None, configured: Any, resolved: Any) -> str:
    endpoint = configured or getattr(litellm, "api_base", None)
    if not endpoint and provider == "openai":
        endpoint = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
    elif not endpoint and provider == "azure":
        endpoint = os.getenv("AZURE_API_BASE")
    return _normalized_endpoint(endpoint or resolved)


def _credential_fingerprint(provider: str, configured: Any, resolved: Any) -> str | None:
    credential = configured or resolved
    if not credential:
        try:
            credential = litellm.get_api_key(provider, resolved)
        except Exception as exc:  # noqa: BLE001 - missing identity disables replay
            logger.debug("Could not resolve opaque-state credential for %s: %s", provider, exc)
            credential = None
    if not credential and provider == "azure":
        credential = (
            os.getenv("AZURE_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("AZURE_AD_TOKEN")
        )
    reveal = getattr(credential, "get_secret_value", None)
    if callable(reveal):
        credential = reveal()
    if not isinstance(credential, str) or not credential:
        return None
    return hashlib.sha256(credential.encode()).hexdigest()


def replay_scope(
    model: str,
    api_style: Literal["chat", "responses"],
    params: dict[str, Any],
    declared_scope: str | None = None,
) -> str | None:
    """Return a non-secret compatibility key for the effective issuer route.

    LiteLLM resolves routing identity. Model, endpoint, credential/account, and
    API style are exact by default. A declared scope may group verified model
    aliases, but never bypasses endpoint, account, provider, or API isolation.
    """
    configured_endpoint = params.get("api_base") or params.get("base_url")
    try:
        resolved_model, provider, resolved_key, resolved_endpoint = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=params.get("custom_llm_provider"),
            api_base=configured_endpoint,
        )
    except Exception as exc:  # noqa: BLE001 - unknown routes fail closed
        logger.debug("Could not resolve opaque-state issuer for %r: %s", model, exc)
        return None
    # This PR understands only OpenAI's encrypted reasoning wire formats.
    # Other providers may use similarly named fields with different replay
    # contracts; they remain fail-closed until their adapters opt in.
    if provider not in {"openai", "azure"}:
        return None

    credential = _credential_fingerprint(
        provider,
        params.get("api_key") or params.get("azure_ad_token"),
        resolved_key,
    )
    if credential is None:
        logger.debug("Opaque-state replay disabled for %r: unknown credential", model)
        return None

    organization = (
        params.get("organization")
        or params.get("openai_organization")
        or getattr(litellm, "organization", None)
        or os.getenv("OPENAI_ORGANIZATION")
    )
    project = (
        params.get("project")
        or params.get("openai_project")
        or getattr(litellm, "project", None)
        or os.getenv("OPENAI_PROJECT")
    )
    account = {
        key: value
        for key, value in (("organization", organization), ("project", project))
        if isinstance(value, str) and value
    }
    identity = {
        "route": declared_scope or resolved_model,
        "endpoint": _effective_endpoint(provider, configured_endpoint, resolved_endpoint),
        "credential": credential,
        "account": account,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{api_style}:{provider}:sha256:{digest}"


def _envelope(scope: str | None, state_format: str, payload: dict[str, Any]) -> dict | None:
    if scope is None or not payload:
        return None
    return {
        "version": _STATE_VERSION,
        "scope": scope,
        "format": state_format,
        "payload": copy.deepcopy(payload),
    }


def _matching_payload(state: Any, scope: str | None, state_format: str) -> dict | None:
    if (
        scope is None
        or not isinstance(state, dict)
        or state.get("version") != _STATE_VERSION
        or state.get("scope") != scope
        or state.get("format") != state_format
        or not isinstance(state.get("payload"), dict)
    ):
        return None
    return cast(dict[str, Any], copy.deepcopy(state["payload"]))


def _is_state_only(state: Any, state_format: str) -> bool:
    return (
        isinstance(state, dict)
        and state.get("version") == _STATE_VERSION
        and state.get("format") == state_format
        and isinstance(state.get("payload"), dict)
        and state["payload"].get("state_only") is True
    )


def capture_chat_state(message: Any, scope: str | None) -> dict | None:
    items = _field(message, "reasoning_items")
    if not isinstance(items, list) or not items:
        return None
    payload: dict[str, Any] = {"reasoning_items": [opaque_item(item) for item in items]}
    if not _field(message, "content") and not _field(message, "tool_calls"):
        payload["state_only"] = True
    return _envelope(scope, _CHAT_FORMAT, payload)


def capture_responses_state(output: list[Any], scope: str | None) -> dict | None:
    items: list[Any] = []
    order: list[dict[str, Any]] = []
    has_public_carrier = False
    for item in output:
        item_type = response_item_type(item)
        if item_type == "reasoning":
            order.append({"type": "reasoning", "index": len(items)})
            items.append(opaque_item(item))
        elif item_type == "function_call":
            call_id = _field(item, "call_id")
            if isinstance(call_id, str):
                order.append({"type": "function_call", "call_id": call_id})
                has_public_carrier = True
        elif item_type == "message":
            order.append({"type": "message"})
            has_public_carrier = True
    if not items:
        return None
    payload: dict[str, Any] = {"items": items, "order": order}
    if not has_public_carrier:
        payload["state_only"] = True
    return _envelope(scope, _RESPONSES_FORMAT, payload)


def prepare_chat_messages(messages: list[dict[str, Any]], scope: str | None) -> list[dict]:
    """Strip private/raw state and restore only a matching Chat payload."""
    prepared: list[dict[str, Any]] = []
    for original in messages:
        state = copy.deepcopy(carried_state(original))
        message = copy.deepcopy(dict(original))
        message.pop(LLM_STATE_KEY, None)
        message.pop("reasoning_items", None)
        payload = _matching_payload(state, scope, _CHAT_FORMAT)
        if (
            payload is None
            and _is_state_only(state, _CHAT_FORMAT)
            and message.get("role") == "assistant"
            and not message.get("content")
            and not message.get("tool_calls")
        ):
            continue
        if payload and isinstance(payload.get("reasoning_items"), list):
            message["reasoning_items"] = payload["reasoning_items"]
        prepared.append(message)
    return prepared


def _clean_responses_batch(batch: Any) -> list[dict[str, Any]]:
    if not isinstance(batch, list):
        return []
    clean: list[dict[str, Any]] = []
    for original in batch:
        if not isinstance(original, dict) or response_item_type(original) == "reasoning":
            continue
        item = copy.deepcopy(original)
        item.pop(LLM_STATE_KEY, None)
        item.pop("reasoning_items", None)
        clean.append(item)
    return clean


def prepare_responses_batch(
    batch: Any,
    state: Any,
    scope: str | None,
) -> list[dict[str, Any]]:
    """Restore a matching Responses payload among its public turn carriers."""
    clean = _clean_responses_batch(batch)
    payload = _matching_payload(state, scope, _RESPONSES_FORMAT)
    if payload is None:
        return [] if _is_state_only(state, _RESPONSES_FORMAT) else clean
    items = payload.get("items")
    order = payload.get("order")
    if not isinstance(items, list) or not isinstance(order, list):
        return clean
    if payload.get("state_only") is True:
        return [copy.deepcopy(item) for item in items if isinstance(item, dict)]

    calls = {
        item.get("call_id"): item
        for item in clean
        if item.get("type") == "function_call" and isinstance(item.get("call_id"), str)
    }
    message = next((item for item in clean if item.get("role") == "assistant"), None)
    emitted_calls: set[str] = set()
    emitted_message = False
    pending: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    last_carrier_emitted = False

    for slot in order:
        if not isinstance(slot, dict):
            continue
        if slot.get("type") == "reasoning":
            index = slot.get("index")
            if isinstance(index, int) and 0 <= index < len(items):
                item = items[index]
                if isinstance(item, dict):
                    pending.append(copy.deepcopy(item))
            continue
        if slot.get("type") == "function_call":
            call_id = slot.get("call_id")
            carrier = calls.get(call_id) if isinstance(call_id, str) else None
            if carrier is not None and isinstance(call_id, str):
                replay.extend(pending)
                replay.append(copy.deepcopy(carrier))
                emitted_calls.add(call_id)
                last_carrier_emitted = True
            else:
                last_carrier_emitted = False
            pending = []
            continue
        if slot.get("type") == "message":
            if message is not None and not emitted_message:
                replay.extend(pending)
                replay.append(copy.deepcopy(message))
                emitted_message = True
                last_carrier_emitted = True
            else:
                last_carrier_emitted = False
            pending = []

    if last_carrier_emitted:
        replay.extend(pending)
    replay.extend(
        copy.deepcopy(item)
        for item in clean
        if not (
            item is message
            and emitted_message
            or item.get("type") == "function_call"
            and item.get("call_id") in emitted_calls
        )
    )
    return replay


def add_encrypted_reasoning_include(api_params: dict[str, Any], scope: str | None) -> None:
    """Request OpenAI encrypted reasoning only on endpoints known to support it."""
    configured = api_params.get("include")
    include = list(configured) if isinstance(configured, (list, tuple, set)) else []
    if configured is not None and not isinstance(configured, (list, tuple, set)):
        include.append(configured)
    if _ENCRYPTED_REASONING_INCLUDE in include:
        api_params["include"] = include
        return
    if scope and scope.startswith("responses:azure:"):
        include.append(_ENCRYPTED_REASONING_INCLUDE)
    elif scope and scope.startswith("responses:openai:"):
        endpoint = _effective_endpoint(
            "openai", api_params.get("api_base") or api_params.get("base_url"), None
        )
        if endpoint == "https://api.openai.com/v1":
            include.append(_ENCRYPTED_REASONING_INCLUDE)
    if include:
        api_params["include"] = include
