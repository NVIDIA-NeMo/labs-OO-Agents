# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility-scoped capture and replay of opaque OpenAI reasoning state.

The event IR treats provider state as an opaque dictionary. This module is the
only code that opens its NOOA envelope or places the payload on provider wire
messages. Unknown providers and compatibility mismatches fail closed.
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

from nooa._llm_state import (
    LLM_STATE_KEY,
    carried_reasoning,
    carried_state,
    demote_reasoning_text,
    demote_responses_batch,
)

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


def _uses_native_openai_endpoint(api_params: dict[str, Any]) -> bool:
    endpoint = (
        api_params.get("api_base")
        or api_params.get("base_url")
        or getattr(litellm, "api_base", None)
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    return _normalized_endpoint(endpoint) == "https://api.openai.com/v1"


def replay_scope(
    model: str,
    api_style: Literal["chat", "responses"],
    params: dict[str, Any],
) -> str | None:
    """Return a non-secret compatibility key for an opaque provider payload.

    LiteLLM resolves provider and model identity. Provider, API style, and exact
    model are intentionally the whole key. Transport routes and authentication
    do not change the provider wire format, so gateway or credential changes
    must not silently disable capture or replay.
    """
    configured_endpoint = params.get("api_base") or params.get("base_url")
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=params.get("custom_llm_provider"),
            api_base=configured_endpoint,
        )
    except Exception as exc:  # noqa: BLE001 - unknown routes fail closed
        logger.debug("Could not resolve opaque-state provider for %r: %s", model, exc)
        return None
    # This PR understands only OpenAI's encrypted reasoning wire formats.
    # Other providers may use similarly named fields with different replay
    # contracts; they remain fail-closed until their adapters opt in.
    if provider not in {"openai", "azure"}:
        return None

    digest = hashlib.sha256(resolved_model.encode()).hexdigest()
    return f"{api_style}:{provider}:sha256:{digest}"


def _envelope(scope: str | None, state_format: str, payload: dict[str, Any]) -> dict | None:
    if scope is None or not payload:
        return None
    return {
        "version": _STATE_VERSION,
        "scope": scope,
        "format": state_format,
        "payload": payload,
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
    # Event history owns this payload. Provider adapters may read and serialize
    # it, but must not mutate caller input or require a per-request history copy.
    return cast(dict[str, Any], state["payload"])


def capture_chat_state(message: Any, scope: str | None) -> dict | None:
    items = _field(message, "reasoning_items")
    if not isinstance(items, list) or not items:
        return None
    payload: dict[str, Any] = {"reasoning_items": [opaque_item(item) for item in items]}
    if not _field(message, "content") and not _field(message, "tool_calls"):
        payload["state_only"] = True
    return _envelope(scope, _CHAT_FORMAT, payload)


def _responses_message_text(item: Any) -> str:
    content = _field(item, "content", [])
    if not isinstance(content, list):
        return ""
    texts = [
        text
        for block in content
        if isinstance((text := _field(block, "text")), str)
    ]
    return "\n".join(texts)


def _responses_call_slot(item: Any) -> dict[str, Any] | None:
    call_id = _field(item, "call_id")
    name = _field(item, "name") or ""
    arguments = _field(item, "arguments") or ""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    if not all(isinstance(value, str) for value in (call_id, name, arguments)):
        return None
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _valid_responses_payload(payload: dict[str, Any]) -> bool:
    """Validate NOOA-owned structure while leaving encrypted item contents opaque."""
    if set(payload) - {"items", "order", "state_only"}:
        return False
    items = payload.get("items")
    order = payload.get("order")
    if not isinstance(items, list) or not items or not isinstance(order, list):
        return False
    if any(
        not isinstance(item, dict)
        or response_item_type(item) != "reasoning"
        or not isinstance(item.get("encrypted_content"), str)
        or not item["encrypted_content"]
        for item in items
    ):
        return False

    indexes: list[int] = []
    carriers: list[dict[str, Any]] = []
    for slot in order:
        if not isinstance(slot, dict):
            return False
        slot_type = slot.get("type")
        if slot_type == "reasoning":
            if set(slot) != {"type", "index"}:
                return False
            index = slot.get("index")
            if not isinstance(index, int) or not 0 <= index < len(items):
                return False
            indexes.append(index)
        elif slot_type == "function_call":
            if set(slot) != {"type", "call_id", "name", "arguments"}:
                return False
            if not all(
                isinstance(slot.get(key), str) for key in ("call_id", "name", "arguments")
            ):
                return False
            carriers.append(slot)
        elif slot_type == "message":
            if set(slot) != {"type", "content"} or not isinstance(slot.get("content"), str):
                return False
            carriers.append(slot)
        else:
            return False

    state_only = payload.get("state_only")
    if state_only not in (None, True):
        return False
    return (
        sorted(indexes) == list(range(len(items)))
        and len({slot["call_id"] for slot in carriers if slot["type"] == "function_call"})
        == sum(slot["type"] == "function_call" for slot in carriers)
        and (state_only is True) == (not carriers)
    )


def _public_responses_carriers(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    carriers: list[dict[str, Any]] = []
    for item in items:
        if response_item_type(item) == "function_call":
            slot = _responses_call_slot(item)
            if slot is not None:
                carriers.append(slot)
        elif item.get("role") == "assistant":
            content = item.get("content", "")
            if isinstance(content, str):
                carriers.append({"type": "message", "content": content})
    return carriers


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
            slot = _responses_call_slot(item)
            if slot is not None:
                order.append(slot)
                has_public_carrier = True
        elif item_type == "message":
            order.append({"type": "message", "content": _responses_message_text(item)})
            has_public_carrier = True
    if not items:
        return None
    payload: dict[str, Any] = {"items": items, "order": order}
    if not has_public_carrier:
        payload["state_only"] = True
    if not _valid_responses_payload(payload):
        logger.warning("Discarding malformed OpenAI Responses reasoning state.")
        return None
    return _envelope(scope, _RESPONSES_FORMAT, payload)


def prepare_chat_messages(messages: list[dict[str, Any]], scope: str | None) -> list[dict]:
    """Strip private/raw state and restore only a matching Chat payload."""
    prepared: list[dict[str, Any]] = []
    for original in messages:
        state = carried_state(original)
        reasoning = carried_reasoning(original)
        message = copy.deepcopy(dict(original))
        message.pop(LLM_STATE_KEY, None)
        message.pop("reasoning_items", None)
        payload = _matching_payload(state, scope, _CHAT_FORMAT)
        if payload and isinstance(payload.get("reasoning_items"), list):
            message["reasoning_items"] = payload["reasoning_items"]
        else:
            demote_reasoning_text(message, reasoning)
        if (
            payload is None
            and state is not None
            and not reasoning
            and message.get("role") == "assistant"
            and not message.get("content")
            and not message.get("tool_calls")
        ):
            continue
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
    reasoning: str | None = None,
) -> list[dict[str, Any]]:
    """Restore a matching Responses payload among its public turn carriers."""
    clean = _clean_responses_batch(batch)
    payload = _matching_payload(state, scope, _RESPONSES_FORMAT)
    if payload is None:
        return demote_responses_batch(clean, state, reasoning)
    items = payload.get("items")
    order = payload.get("order")
    if not _valid_responses_payload(payload):
        logger.warning(
            "Opaque Responses reasoning state is malformed; portable reasoning text "
            "will be replayed instead."
        )
        return demote_responses_batch(clean, state, reasoning)
    assert isinstance(items, list)
    assert isinstance(order, list)
    if payload.get("state_only") is True:
        if clean != [{"role": "assistant", "content": ""}]:
            logger.warning(
                "Opaque Responses reasoning state was not replayed because its empty "
                "public carrier changed."
            )
            return demote_responses_batch(clean, state, reasoning)
        return cast(list[dict[str, Any]], items)

    expected_carriers = [slot for slot in order if slot.get("type") != "reasoning"]
    if _public_responses_carriers(clean) != expected_carriers:
        logger.warning(
            "Opaque Responses reasoning state was not replayed because its public "
            "carriers changed; portable reasoning text will be replayed instead."
        )
        return demote_responses_batch(clean, state, reasoning)

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
                    pending.append(item)
            continue
        if slot.get("type") == "function_call":
            call_id = slot.get("call_id")
            carrier = calls.get(call_id) if isinstance(call_id, str) else None
            if carrier is not None and isinstance(call_id, str):
                replay.extend(pending)
                replay.append(carrier)
                emitted_calls.add(call_id)
                last_carrier_emitted = True
            else:
                last_carrier_emitted = False
            pending = []
            continue
        if slot.get("type") == "message":
            if message is not None and not emitted_message:
                replay.extend(pending)
                replay.append(message)
                emitted_message = True
                last_carrier_emitted = True
            else:
                last_carrier_emitted = False
            pending = []

    if last_carrier_emitted:
        replay.extend(pending)
    replay.extend(
        item
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
    elif (
        scope and scope.startswith("responses:openai:") and _uses_native_openai_endpoint(api_params)
    ):
        include.append(_ENCRYPTED_REASONING_INCLUDE)
    if include:
        api_params["include"] = include
