# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility-scoped capture and replay of closed-provider reasoning state.

The event IR treats provider state as an opaque dictionary. This module is the
only code that opens its NOOA envelope or places the payload on provider wire
messages. Expected incompatibility warns and demotes portable text; malformed
current state raises instead of silently hiding a framework or provider change.
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
    ReplayCarryingMessage,
    carried_cache_boundary,
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
_INLINE_THOUGHT_SIGNATURE_SEPARATOR = "__thought__"
_SUPPORTED_PROVIDERS = {
    "openai",
    "azure",
    "anthropic",
    "gemini",
}


class ReasoningReplayError(RuntimeError):
    """Opaque reasoning state is present but violates NOOA's replay contract."""


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


def _warn_unknown_fields(fields: dict[str, Any], known: set[str], location: str) -> None:
    unknown = sorted(key for key, value in fields.items() if key not in known and value is not None)
    if unknown:
        logger.warning(
            "Ignoring unrecognized provider field(s) %s on %s; opaque reasoning retention "
            "may need updating.",
            ", ".join(unknown),
            location,
        )


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
    except Exception as exc:  # noqa: BLE001 - model routing is third-party input
        logger.warning(
            "Opaque reasoning replay is disabled because LiteLLM could not resolve model %r: %s",
            model,
            exc,
        )
        return None
    if provider not in _SUPPORTED_PROVIDERS or (
        api_style == "responses" and provider not in {"openai", "azure"}
    ):
        return None

    digest = hashlib.sha256(resolved_model.encode()).hexdigest()
    return f"{api_style}:{provider}:sha256:{digest}"


def _envelope(scope: str | None, state_format: str, payload: dict[str, Any]) -> dict | None:
    if not payload:
        return None
    if scope is None:
        raise ReasoningReplayError(
            "The provider returned opaque reasoning state, but NOOA could not establish "
            "a safe replay scope for it."
        )
    return {
        "version": _STATE_VERSION,
        "scope": scope,
        "format": state_format,
        "payload": payload,
    }


def _matching_payload(state: Any, scope: str | None, state_format: str) -> dict | None:
    if state is None:
        return None
    if not isinstance(state, dict) or state.get("version") != _STATE_VERSION:
        version = state.get("version") if isinstance(state, dict) else None
        logger.warning(
            "Ignoring opaque reasoning state with unsupported or legacy version %r; "
            "portable reasoning text will be replayed instead.",
            version,
        )
        return None

    source_scope = state.get("scope")
    source_format = state.get("format")
    payload = state.get("payload")
    if (
        set(state) != {"version", "scope", "format", "payload"}
        or not isinstance(source_scope, str)
        or not isinstance(source_format, str)
        or source_format not in {_CHAT_FORMAT, _RESPONSES_FORMAT}
        or not isinstance(payload, dict)
    ):
        raise ReasoningReplayError(
            "Malformed version-1 opaque reasoning envelope: expected a known format, "
            "a string scope, and a mapping payload."
        )
    _scope_provider(source_scope)
    if source_scope != scope or source_format != state_format:
        logger.warning(
            "Opaque reasoning state captured for %s (%s) is incompatible with %s (%s); "
            "portable reasoning text will be replayed instead.",
            source_scope,
            source_format,
            scope or "an unsupported target",
            state_format,
        )
        return None
    # Event history owns this payload. Provider adapters may read and serialize
    # it, but must not mutate caller input or require a per-request history copy.
    return cast(dict[str, Any], payload)


def _scope_provider(scope: str | None) -> str | None:
    if scope is None:
        return None
    if not isinstance(scope, str):
        raise ReasoningReplayError("Malformed opaque reasoning scope: expected a string.")
    parts = scope.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ReasoningReplayError(
            f"Malformed opaque reasoning scope {scope!r}: expected api:provider:identity."
        )
    return parts[1]


def _chat_public_carrier(message: Any, scope: str | None) -> dict[str, Any] | None:
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
        calls.append(
            {
                "id": public_tool_call_id(call, scope),
                "name": name,
                "arguments": arguments,
            }
        )

    content = _field(message, "content")
    if content is not None and not isinstance(content, str):
        return None
    # The canonical formatter uses None for an empty assistant tool-call turn
    # and an empty string for an assistant turn with no public carrier.
    content = (content or None) if calls else (content or "")
    return {"content": content, "tool_calls": calls}


def _valid_chat_carrier(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"content", "tool_calls"}:
        return False
    if value.get("content") is not None and not isinstance(value.get("content"), str):
        return False
    calls = value.get("tool_calls")
    return isinstance(calls, list) and all(
        isinstance(call, dict)
        and set(call) == {"id", "name", "arguments"}
        and all(isinstance(call.get(key), str) for key in ("id", "name", "arguments"))
        for call in calls
    )


def _sanitize_chat_payload(payload: dict[str, Any], scope: str | None) -> dict[str, Any]:
    """Validate the small provider-specific envelope owned by NOOA."""
    provider = _scope_provider(scope)
    fields_by_provider = {
        "openai": ("reasoning_items", "thinking_blocks"),
        "azure": ("reasoning_items", "thinking_blocks"),
        "anthropic": ("thinking_blocks",),
        "gemini": ("thinking_blocks",),
    }
    clean: dict[str, Any] = {}
    for key in fields_by_provider.get(provider or "", ()):
        value = payload.get(key)
        if isinstance(value, list) and value:
            clean[key] = value

    provider_fields = (
        payload.get("provider_specific_fields")
        if provider in {"openai", "azure", "gemini"}
        else None
    )
    signatures = (
        provider_fields.get("thought_signatures") if isinstance(provider_fields, dict) else None
    )
    if (
        isinstance(signatures, list)
        and signatures
        and all(isinstance(signature, str) and signature for signature in signatures)
    ):
        clean["provider_specific_fields"] = {"thought_signatures": signatures}

    tool_state = payload.get("tool_calls") if provider in {"openai", "azure", "gemini"} else None
    if isinstance(tool_state, list):
        calls: list[dict[str, Any] | None] = []
        for item in tool_state:
            fields = item.get("provider_specific_fields") if isinstance(item, dict) else None
            signature = fields.get("thought_signature") if isinstance(fields, dict) else None
            inline_signature = (
                item.get("inline_thought_signature") if isinstance(item, dict) else None
            )
            call: dict[str, Any] = {}
            if isinstance(signature, str) and signature:
                call["provider_specific_fields"] = {"thought_signature": signature}
            if isinstance(inline_signature, str) and inline_signature:
                call["inline_thought_signature"] = inline_signature
            if signature and inline_signature and signature != inline_signature:
                raise ReasoningReplayError("Conflicting stored tool-call thought signatures.")
            calls.append(call or None)
        if any(item is not None for item in calls):
            clean["tool_calls"] = calls

    has_state = bool(clean)
    carrier = payload.get("carrier")
    if has_state and _valid_chat_carrier(carrier):
        clean["carrier"] = carrier
    if has_state and payload.get("state_only") is True:
        clean["state_only"] = True
    if (
        not has_state
        or clean != payload
        or (
            "tool_calls" in clean
            and len(clean["tool_calls"]) != len(clean.get("carrier", {}).get("tool_calls", ()))
        )
        or (
            clean.get("state_only") is True
            and ("tool_calls" in clean or clean.get("carrier") != {"content": "", "tool_calls": []})
        )
    ):
        raise ReasoningReplayError(
            f"Opaque chat reasoning state is malformed or unsupported for provider {provider!r}."
        )
    return clean


def _tool_call_state(tool_call: Any, scope: str | None) -> dict[str, Any] | None:
    dumped = opaque_item(tool_call)
    if not isinstance(dumped, dict):
        raise ReasoningReplayError("Malformed provider tool call: expected a mapping.")
    fields = dumped.get("provider_specific_fields")
    if fields is not None and not isinstance(fields, dict):
        raise ReasoningReplayError("Malformed tool-call provider_specific_fields.")
    if fields:
        _warn_unknown_fields(fields, {"thought_signature"}, "a provider tool call")
    signature = fields.get("thought_signature") if fields else None
    if (
        fields
        and "thought_signature" in fields
        and (not isinstance(signature, str) or not signature)
    ):
        raise ReasoningReplayError("Malformed tool-call thought_signature.")
    call_id = dumped.get("id")
    inline_candidate = None
    if isinstance(call_id, str) and _INLINE_THOUGHT_SIGNATURE_SEPARATOR in call_id:
        inline_candidate = call_id.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[1]
        if not inline_candidate:
            raise ReasoningReplayError("Malformed inline tool-call thought signature.")
    if signature and inline_candidate and signature != inline_candidate:
        raise ReasoningReplayError("Conflicting thought signatures on one provider tool call.")
    inline_signature = (
        inline_candidate
        if _scope_provider(scope) == "gemini" or signature == inline_candidate
        else None
    )
    if signature is None and inline_signature is None:
        return None
    state: dict[str, Any] = {}
    if signature is not None:
        state["provider_specific_fields"] = {"thought_signature": signature}
    if inline_signature is not None:
        state["inline_thought_signature"] = inline_signature
    return state


def public_tool_call_id(value: Any, scope: str | None) -> str:
    """Return an application call id without LiteLLM's inline Gemini state."""
    call_id = _field(value, "id", "")
    if not isinstance(call_id, str) or not call_id:
        raise ReasoningReplayError("Malformed provider tool call: expected a non-empty id.")
    state = _tool_call_state(value, scope)
    if state is None or "inline_thought_signature" not in state:
        return call_id
    return call_id.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[0]


def capture_chat_state(message: Any, scope: str | None) -> dict | None:
    payload: dict[str, Any] = {}
    for key in ("reasoning_items", "thinking_blocks"):
        value = _field(message, key)
        if value is None or value == []:
            continue
        if not isinstance(value, list):
            raise ReasoningReplayError(f"Malformed provider response field {key!r}.")
        payload[key] = opaque_item(value)

    provider_fields = _field(message, "provider_specific_fields")
    if provider_fields is not None and not isinstance(provider_fields, dict):
        raise ReasoningReplayError("Malformed provider_specific_fields in provider response.")
    if provider_fields:
        _warn_unknown_fields(provider_fields, {"thought_signatures"}, "a provider message")
    # LiteLLM also puts benign fields such as `refusal` here. Only a known
    # thought-signature field belongs in the opaque replay envelope.
    if isinstance(provider_fields, dict) and "thought_signatures" in provider_fields:
        payload["provider_specific_fields"] = opaque_item(
            {"thought_signatures": provider_fields["thought_signatures"]}
        )

    raw_tool_calls_value = _field(message, "tool_calls")
    if raw_tool_calls_value is not None and not isinstance(raw_tool_calls_value, list):
        raise ReasoningReplayError("Malformed tool_calls in provider response.")
    raw_tool_calls = raw_tool_calls_value or []
    tool_state = [_tool_call_state(call, scope) for call in raw_tool_calls]
    if any(item is not None for item in tool_state):
        payload["tool_calls"] = tool_state

    if not payload:
        return None
    carrier = _chat_public_carrier(message, scope)
    if carrier is None or len({call["id"] for call in carrier["tool_calls"]}) != len(
        carrier["tool_calls"]
    ):
        raise ReasoningReplayError(
            "Cannot retain opaque reasoning state for a malformed public assistant carrier."
        )
    payload["carrier"] = carrier
    if not _field(message, "content") and not _field(message, "tool_calls"):
        payload["state_only"] = True
    payload = _sanitize_chat_payload(payload, scope)
    return _envelope(scope, _CHAT_FORMAT, payload)


def _responses_message_text(item: Any) -> str:
    content = _field(item, "content", [])
    if not isinstance(content, list):
        return ""
    texts = [text for block in content if isinstance((text := _field(block, "text")), str)]
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
            if not all(isinstance(slot.get(key), str) for key in ("call_id", "name", "arguments")):
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


def _strip_inline_signature(value: Any) -> Any:
    if isinstance(value, str) and _INLINE_THOUGHT_SIGNATURE_SEPARATOR in value:
        return value.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[0]
    return value


def _strip_chat_state(
    message: dict[str, Any], source_scope: str | None, target_scope: str | None
) -> dict[str, str]:
    public_call_ids: dict[str, str] = {}
    removed = False
    gemini_wire = "gemini" in {
        _scope_provider(source_scope),
        _scope_provider(target_scope),
    }
    for key in ("reasoning_items", "thinking_blocks", "provider_specific_fields"):
        removed = key in message or removed
        message.pop(key, None)
    content = message.get("content")
    if isinstance(content, list):
        public_blocks = [
            block
            for block in content
            if not (
                isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"}
            )
        ]
        removed = len(public_blocks) != len(content) or removed
        message["content"] = public_blocks
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            fields = call.get("provider_specific_fields")
            explicit_signature = (
                fields.get("thought_signature") if isinstance(fields, dict) else None
            )
            inline_candidate = (
                call_id.split(_INLINE_THOUGHT_SIGNATURE_SEPARATOR, 1)[1]
                if isinstance(call_id, str) and _INLINE_THOUGHT_SIGNATURE_SEPARATOR in call_id
                else None
            )
            if gemini_wire or (inline_candidate and inline_candidate == explicit_signature):
                public_id = _strip_inline_signature(call_id)
                removed = public_id != call_id or removed
                if isinstance(call_id, str) and public_id != call_id:
                    public_call_ids[call_id] = public_id
                call["id"] = public_id
            removed = "provider_specific_fields" in call or removed
            call.pop("provider_specific_fields", None)
            function = call.get("function")
            if isinstance(function, dict):
                removed = "provider_specific_fields" in function or removed
                function.pop("provider_specific_fields", None)
    if gemini_wire and "tool_call_id" in message:
        public_id = _strip_inline_signature(message["tool_call_id"])
        removed = public_id != message["tool_call_id"] or removed
        message["tool_call_id"] = public_id
    if removed:
        logger.warning(
            "Removed untrusted provider reasoning fields from a public chat message; "
            "replay opaque state through a persisted LLMResponse instead."
        )
    return public_call_ids


def _restore_chat_state(message: dict[str, Any], payload: dict[str, Any]) -> bool:
    if _chat_public_carrier(message, None) != payload.get("carrier"):
        logger.warning(
            "Opaque reasoning state was not replayed because its public assistant carrier "
            "changed; portable reasoning text will be replayed instead."
        )
        return False

    for key in ("reasoning_items", "thinking_blocks", "provider_specific_fields"):
        if key in payload:
            message[key] = payload[key]
    tool_calls = message.get("tool_calls")
    tool_state = payload.get("tool_calls")
    if tool_state is None:
        return True
    # The payload validator and identity check above guarantee equal lists.
    for call, state in zip(cast(list[Any], tool_calls), cast(list[Any], tool_state), strict=True):
        if state is not None:
            call = cast(dict[str, Any], call)
            state = cast(dict[str, Any], state)
            if "provider_specific_fields" in state:
                call["provider_specific_fields"] = state["provider_specific_fields"]
            if inline_signature := state.get("inline_thought_signature"):
                call["id"] = f"{call['id']}{_INLINE_THOUGHT_SIGNATURE_SEPARATOR}{inline_signature}"
    return True


def capture_responses_state(output: list[Any], scope: str | None) -> dict | None:
    # NOOA only knows the OpenAI/Azure Responses item contract. Other providers
    # may expose a similarly shaped API through a gateway, but that is not
    # evidence that their opaque state is wire-compatible.
    if _scope_provider(scope) not in {"openai", "azure"}:
        if any(response_item_type(item) == "reasoning" for item in output):
            raise ReasoningReplayError(
                "The provider returned Responses reasoning state, but NOOA only supports "
                "opaque Responses replay for OpenAI and Azure routes."
            )
        return None
    items: list[Any] = []
    order: list[dict[str, Any]] = []
    call_ids: list[str] = []
    message_count = 0
    malformed_call_id = False
    unknown_output_types: set[str] = set()
    for item in output:
        item_type = response_item_type(item)
        if item_type == "reasoning":
            order.append({"type": "reasoning", "index": len(items)})
            items.append(opaque_item(item))
        elif item_type == "function_call":
            slot = _responses_call_slot(item)
            if slot is None:
                malformed_call_id = True
                continue
            order.append(slot)
            call_ids.append(cast(str, slot["call_id"]))
        elif item_type == "message":
            order.append({"type": "message", "content": _responses_message_text(item)})
            message_count += 1
        else:
            unknown_output_types.add(item_type or "<missing>")
    if not items:
        return None
    if unknown_output_types:
        raise ReasoningReplayError(
            "Cannot retain Responses reasoning state beside unsupported output type(s): "
            + ", ".join(sorted(unknown_output_types))
        )
    if malformed_call_id:
        raise ReasoningReplayError(
            "Cannot retain Responses reasoning state beside a malformed function call id."
        )
    if len(set(call_ids)) != len(call_ids) or message_count > 1:
        raise ReasoningReplayError(
            "Cannot retain Responses reasoning state with duplicate call ids or multiple messages."
        )
    payload: dict[str, Any] = {"items": items, "order": order}
    if not call_ids and not message_count:
        payload["state_only"] = True
    if not _valid_responses_payload(payload):
        raise ReasoningReplayError("Malformed OpenAI Responses reasoning state.")
    return _envelope(scope, _RESPONSES_FORMAT, payload)


def responses_reasoning_text(output: list[Any]) -> str | None:
    """Return provider-visible Responses reasoning summaries as plain text."""
    texts: list[str] = []
    for item in output:
        if response_item_type(item) != "reasoning":
            continue
        summary_items = _field(item, "summary")
        if summary_items is None:
            summary_items = []
        elif not isinstance(summary_items, list):
            raise ReasoningReplayError("Malformed Responses reasoning summary.")
        for summary in summary_items:
            text = _field(summary, "text")
            if not isinstance(text, str):
                raise ReasoningReplayError("Malformed Responses reasoning summary text.")
            if text:
                texts.append(text)
    return "\n".join(texts) or None


def prepare_chat_messages(messages: list[dict[str, Any]], scope: str | None) -> list[dict]:
    """Strip private/raw state and restore only a matching Chat payload."""
    prepared: list[dict[str, Any]] = []
    public_call_ids: dict[str, str] = {}
    private_call_ids: dict[str, str] = {}
    for original in messages:
        cache_boundary = carried_cache_boundary(original)
        state = carried_state(original)
        reasoning = carried_reasoning(original)
        message = copy.deepcopy(dict(original))
        message.pop(LLM_STATE_KEY, None)
        source_scope = (
            state.get("scope")
            if isinstance(state, dict) and state.get("version") == _STATE_VERSION
            else None
        )
        public_call_ids.update(
            _strip_chat_state(
                message,
                source_scope if isinstance(source_scope, str) else None,
                scope,
            )
        )
        payload = _matching_payload(state, scope, _CHAT_FORMAT)
        if payload is not None:
            payload = _sanitize_chat_payload(payload, scope)
        restored = payload is not None and _restore_chat_state(message, payload)
        if restored:
            public_calls = cast(dict[str, Any], payload)["carrier"]["tool_calls"]
            private_calls = message.get("tool_calls") or []
            for public_call, private_call in zip(public_calls, private_calls, strict=True):
                public_id = public_call["id"]
                private_id = private_call["id"]
                if public_id != private_id:
                    private_call_ids[public_id] = private_id
        tool_call_id = message.get("tool_call_id")
        # Structural signature evidence on a raw assistant call also applies to
        # its matching result, even when neither route identifies Gemini.
        if isinstance(tool_call_id, str) and tool_call_id in public_call_ids:
            tool_call_id = public_call_ids.pop(tool_call_id)
            message["tool_call_id"] = tool_call_id
        if isinstance(tool_call_id, str) and tool_call_id in private_call_ids:
            message["tool_call_id"] = private_call_ids.pop(tool_call_id)
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
            if cache_boundary:
                prepared.append(ReplayCarryingMessage({}, cache_boundary_before=True))
            continue
        prepared.append(
            ReplayCarryingMessage(message, cache_boundary_before=True)
            if cache_boundary
            else message
        )
    return prepared


def _clean_responses_batch(batch: Any) -> list[dict[str, Any]]:
    if not isinstance(batch, list):
        raise ReasoningReplayError("Malformed Responses replay batch: expected a list.")
    clean: list[dict[str, Any]] = []
    for original in batch:
        if not isinstance(original, dict):
            raise ReasoningReplayError("Malformed Responses replay item: expected a mapping.")
        if response_item_type(original) == "reasoning":
            logger.warning(
                "Removed an untrusted reasoning item from public Responses input; replay "
                "opaque state through a persisted LLMResponse instead."
            )
            continue
        item = copy.deepcopy(original)
        item.pop(LLM_STATE_KEY, None)
        if "reasoning_items" in item:
            logger.warning(
                "Removed untrusted reasoning_items from public Responses input; replay "
                "opaque state through a persisted LLMResponse instead."
            )
            item.pop("reasoning_items")
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
    if _scope_provider(scope) not in {"openai", "azure"}:
        raise ReasoningReplayError(
            "Opaque Responses replay is only supported for OpenAI and Azure scopes."
        )
    unknown = sorted(set(payload) - {"items", "order", "state_only"})
    if unknown:
        raise ReasoningReplayError(
            f"Malformed Responses reasoning state: unknown field(s) {', '.join(unknown)}."
        )
    items = payload.get("items")
    order = payload.get("order")
    if not _valid_responses_payload(payload):
        raise ReasoningReplayError("Malformed OpenAI Responses reasoning state.")
    assert isinstance(items, list)
    assert isinstance(order, list)
    if payload.get("state_only") is True:
        if clean != [{"role": "assistant", "content": ""}]:
            logger.warning(
                "Opaque Responses reasoning state was not replayed because its empty public "
                "carrier changed; portable reasoning text will be replayed instead."
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

    calls = [item for item in clean if item.get("type") == "function_call"]
    messages = [item for item in clean if item.get("role") == "assistant"]
    calls_by_id = {cast(str, item["call_id"]): item for item in calls}
    message = messages[0] if messages else None
    emitted_calls: set[str] = set()
    emitted_message = False
    pending: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    last_carrier_emitted = False

    for slot in order:
        if slot.get("type") == "reasoning":
            pending.append(cast(dict[str, Any], items[cast(int, slot["index"])]))
            continue
        if slot.get("type") == "function_call":
            call_id = cast(str, slot["call_id"])
            replay.extend(pending)
            replay.append(calls_by_id[call_id])
            emitted_calls.add(call_id)
            last_carrier_emitted = True
            pending = []
            continue
        if slot.get("type") == "message":
            replay.extend(pending)
            replay.append(cast(dict[str, Any], message))
            emitted_message = True
            last_carrier_emitted = True
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
