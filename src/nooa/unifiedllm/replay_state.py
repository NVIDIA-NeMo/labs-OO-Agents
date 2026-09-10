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


def _chat_public_carrier(message: Any) -> dict[str, Any] | None:
    """Project the public assistant data to which opaque Chat state is bound."""
    if _field(message, "role") != "assistant":
        return None
    raw_calls = _field(message, "tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        return None

    calls: list[dict[str, str]] = []
    for call in raw_calls:
        function = _field(call, "function")
        call_id = _field(call, "id")
        name = _field(function, "name")
        arguments = _field(function, "arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        if not all(isinstance(value, str) for value in (call_id, name, arguments)):
            return None
        calls.append({"id": call_id, "name": name, "arguments": arguments})

    content = _field(message, "content")
    if content is not None and not isinstance(content, str):
        return None
    content = (content or None) if calls else (content or "")
    return {"content": content, "tool_calls": calls}


def _valid_chat_payload(payload: dict[str, Any]) -> bool:
    if set(payload) - {"reasoning_items", "carrier", "state_only"}:
        return False
    items = payload.get("reasoning_items")
    carrier = payload.get("carrier")
    if not isinstance(items, list) or not items or not isinstance(carrier, dict):
        return False
    if set(carrier) != {"content", "tool_calls"}:
        return False
    if carrier.get("content") is not None and not isinstance(carrier.get("content"), str):
        return False
    calls = carrier.get("tool_calls")
    if not isinstance(calls, list) or not all(
        isinstance(call, dict)
        and set(call) == {"id", "name", "arguments"}
        and all(isinstance(call.get(key), str) for key in ("id", "name", "arguments"))
        for call in calls
    ):
        return False
    if len({call["id"] for call in calls}) != len(calls):
        return False
    if "state_only" in payload and payload["state_only"] is not True:
        return False
    return (payload.get("state_only") is True) == (carrier == {"content": "", "tool_calls": []})


def capture_chat_state(message: Any, scope: str | None) -> dict | None:
    items = _field(message, "reasoning_items")
    if not isinstance(items, list) or not items:
        return None
    carrier = _chat_public_carrier(message)
    if carrier is None:
        logger.warning("Discarding OpenAI Chat reasoning state with a malformed carrier.")
        return None
    payload: dict[str, Any] = {
        "reasoning_items": [opaque_item(item) for item in items],
        "carrier": carrier,
    }
    if not _field(message, "content") and not _field(message, "tool_calls"):
        payload["state_only"] = True
    if not _valid_chat_payload(payload):
        logger.warning("Discarding malformed OpenAI Chat reasoning state.")
        return None
    return _envelope(scope, _CHAT_FORMAT, payload)


def responses_message_text(item: Any) -> str:
    content = _field(item, "content", [])
    if not isinstance(content, list):
        return ""
    texts = [
        text
        for block in content
        if _field(block, "type") == "output_text"
        and isinstance((text := _field(block, "text")), str)
    ]
    return "".join(texts)


def responses_output_text(output: list[Any]) -> str:
    """Match the SDK's separator-free aggregation of assistant text blocks."""
    return "".join(
        responses_message_text(item) for item in output if response_item_type(item) == "message"
    )


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
    if not isinstance(items, list) or not isinstance(order, list) or not order:
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
            if not all(isinstance(slot.get(key), str) for key in ("call_id", "name", "arguments")):
                return False
            carriers.append(slot)
        elif slot_type == "message":
            if (
                set(slot) - {"type", "content", "phase"}
                or not isinstance(slot.get("content"), str)
                or ("phase" in slot and slot["phase"] not in ("commentary", "final_answer"))
            ):
                return False
            carriers.append(slot)
        else:
            return False

    state_only = payload.get("state_only")
    if state_only not in (None, True):
        return False
    return (
        bool(items or carriers != _flatten_responses_carriers(carriers))
        and sorted(indexes) == list(range(len(items)))
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


def _flatten_responses_carriers(carriers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project provider message boundaries into the canonical LLMResponse view."""
    calls = [slot for slot in carriers if slot["type"] == "function_call"]
    messages = [slot for slot in carriers if slot["type"] == "message"]
    text = "".join(slot["content"] for slot in messages)
    return (
        [{"type": "message", "content": text}] if text or (messages and not calls) else []
    ) + calls


def capture_responses_state(output: list[Any], scope: str | None) -> dict | None:
    items: list[Any] = []
    order: list[dict[str, Any]] = []
    has_public_carrier = False
    reasoning_items = [item for item in output if response_item_type(item) == "reasoning"]
    summary_only = any(_field(item, "encrypted_content") is None for item in reasoning_items)
    if summary_only and any(
        _field(item, "encrypted_content") is not None for item in reasoning_items
    ):
        logger.warning(
            "Responses reasoning contains items without encrypted content; replaying "
            "portable summaries instead of an incomplete opaque reasoning sequence."
        )
    for item in output:
        item_type = response_item_type(item)
        if item_type == "reasoning":
            # A summary-only reasoning item has no opaque data to retain.
            if summary_only:
                continue
            order.append({"type": "reasoning", "index": len(items)})
            items.append(opaque_item(item))
        elif item_type == "function_call":
            slot = _responses_call_slot(item)
            if slot is not None:
                order.append(slot)
                has_public_carrier = True
        elif item_type == "message":
            slot = {"type": "message", "content": responses_message_text(item)}
            phase = _field(item, "phase")
            if phase is not None:
                slot["phase"] = phase
            order.append(slot)
            has_public_carrier = True
    carriers = [slot for slot in order if slot["type"] != "reasoning"]
    # Keep provider message boundaries/phase when the canonical flat text and
    # calls alone cannot reproduce them, even without encrypted reasoning.
    if not items and carriers == _flatten_responses_carriers(carriers):
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
        restored = False
        if payload is not None and not _valid_chat_payload(payload):
            logger.warning(
                "Opaque Chat reasoning state is malformed; portable reasoning text "
                "will be replayed instead."
            )
        elif payload is not None and _chat_public_carrier(message) != payload["carrier"]:
            logger.warning(
                "Opaque Chat reasoning state was not replayed because its public assistant "
                "carrier changed; portable reasoning text will be replayed instead."
            )
        elif payload is not None:
            message["reasoning_items"] = payload["reasoning_items"]
            restored = True
        if not restored:
            demote_reasoning_text(message, reasoning)
        if (
            not restored
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

    expected_carriers = [
        {key: value for key, value in slot.items() if key != "phase"}
        for slot in order
        if slot["type"] != "reasoning"
    ]
    public_carriers = _public_responses_carriers(clean)
    if public_carriers not in (expected_carriers, _flatten_responses_carriers(expected_carriers)):
        logger.warning(
            "Opaque Responses reasoning state was not replayed because its public "
            "carriers changed; portable reasoning text will be replayed instead."
        )
        return demote_responses_batch(clean, state, reasoning)

    calls = iter(item for item in clean if item.get("type") == "function_call")
    messages = iter(item for item in clean if item.get("role") == "assistant")
    exact_messages = public_carriers == expected_carriers
    replay: list[dict[str, Any]] = []
    for slot in order:
        if slot["type"] == "reasoning":
            replay.append(items[slot["index"]])
        elif slot["type"] == "function_call":
            replay.append(next(calls))
        else:
            message = (
                next(messages)
                if exact_messages
                else {"role": "assistant", "content": slot["content"]}
            )
            if "phase" in slot:
                message["phase"] = slot["phase"]
            replay.append(message)
    replay.extend(
        item
        for item in clean
        if item.get("role") != "assistant" and item.get("type") != "function_call"
    )
    # Message-phase metadata alone must not suppress portable reasoning text.
    return demote_responses_batch(replay, None, reasoning) if not items else replay


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
