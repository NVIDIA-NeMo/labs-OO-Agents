# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve provider scope, validate wire input, and prepare Chat messages.

Native part adapters own capture and projection. Incompatible turns replay
portable text; malformed current state raises rather than silently hiding errors.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
from typing import Any, Literal
from urllib.parse import urlsplit

import litellm

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm.cache_policy import reject_boundary_dict, wrap_anthropic_text
from nooa.unifiedllm.errors import ReasoningReplayError

logger = logging.getLogger(__name__)
LLM_STATE_KEY = "_nooa_llm_state"

_ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
_INLINE_THOUGHT_SIGNATURE_SEPARATOR = "__thought__"
_SUPPORTED_PROVIDERS = {
    "openai",
    "azure",
    "anthropic",
    "gemini",
    "deepseek",  # Plain reasoning_content is required on thinking-mode tool turns.
}


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def response_item_type(item: Any) -> str | None:
    value = _field(item, "type")
    return value if isinstance(value, str) else None


def unsupported_responses_parts(output: list[Any]) -> list[str]:
    """Identify turn parts the canonical response cannot currently project."""
    unsupported: list[str] = []
    for item in output:
        item_type = response_item_type(item)
        if item_type not in {"reasoning", "function_call", "message"}:
            unsupported.append(str(item_type))
        elif item_type == "message":
            content = _field(item, "content")
            if not isinstance(content, list):
                raise ReasoningReplayError("Responses message content must be a list of blocks.")
            for block in content:
                if not isinstance(block, dict) and not callable(
                    getattr(type(block), "model_dump", None)
                ):
                    raise ReasoningReplayError("Responses text blocks must be mappings.")
                block_type = response_item_type(block)
                if block_type != "output_text":
                    unsupported.append(f"message.{block_type}")
    return unsupported


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


def _configured_endpoint(params: dict[str, Any]) -> str | None:
    """The endpoint this call's own configuration points at, if any.

    Dispatch always lets a per-call ``base_url`` win over an inherited
    ``api_base`` (it pops ``base_url`` into ``api_base`` right before the
    request); every reader of these two keys must agree with that precedence
    or it can disagree with where the call actually goes for a client whose
    ``api_base`` and a call's ``base_url`` differ.
    """
    return params.get("base_url") or params.get("api_base")


def _uses_native_openai_endpoint(api_params: dict[str, Any], *, direct: bool = False) -> bool:
    endpoint = (
        _configured_endpoint(api_params)
        or (None if direct else getattr(litellm, "api_base", None))
        or os.getenv("OPENAI_BASE_URL")
        or (None if direct else os.getenv("OPENAI_API_BASE"))
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
    configured_endpoint = _configured_endpoint(params)
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


def _tool_call_state(tool_call: Any) -> dict[str, Any] | None:
    """Extract signatures before deciding whether their route permits replay.

    OpenAI-compatible gateways may encode the signature only in the call id.
    That encoding is private state even without a corroborating field or a
    recognized provider. Capture removes it from the public id; scope gating
    later decides whether to retain and replay it.
    """
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
    inline_signature = inline_candidate
    if signature is None and inline_signature is None:
        return None
    state: dict[str, Any] = {}
    if signature is not None:
        state["provider_specific_fields"] = {"thought_signature": signature}
    if inline_signature is not None:
        state["inline_thought_signature"] = inline_signature
    return state


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


def reject_native_message(
    message: dict[str, Any], scope: str | None, *, reject_google_signature: bool = False
) -> None:
    """Raw dictionaries are portable input, not a provider-state replay API."""
    private_keys = {LLM_STATE_KEY, "reasoning_items", "thinking_blocks", "provider_specific_fields"}
    nodes = [message]
    content = message.get("content")
    if isinstance(content, list):
        nodes.extend(block for block in content if isinstance(block, dict))
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ReasoningReplayError("Malformed tool_calls: expected a list.")
    for call in calls:
        if not isinstance(call, dict):
            raise ReasoningReplayError("Malformed tool_calls entry: expected a mapping.")
        nodes.append(call)
        function = call.get("function")
        if not isinstance(function, dict):
            raise ReasoningReplayError("Malformed tool call function: expected a mapping.")
        nodes.append(function)
    for node in nodes:
        extension = node.get("extra_content")
        google = extension.get("google") if isinstance(extension, dict) else None
        if reject_google_signature and isinstance(google, dict) and google.get("thought_signature"):
            raise ReasoningReplayError(
                "Google thought signatures require a canonical LLMResponse, not a wire dict."
            )
        # SDK dumps include optional provider fields with null/empty values;
        # those carry no native state and are valid portable input.
        if any(node.get(key) for key in private_keys) or node.get("type") in {
            "reasoning",
            "thinking",
            "redacted_thinking",
        }:
            raise ReasoningReplayError(
                "Opaque provider fields require a canonical LLMResponse, not a wire dict."
            )
    ids = [message.get("tool_call_id")]
    ids.extend(call.get("id") for call in calls)
    if any(
        isinstance(value, str) and _INLINE_THOUGHT_SIGNATURE_SEPARATOR in value for value in ids
    ):
        raise ReasoningReplayError(
            "Inline thought signatures require an LLMResponse, not a wire dict."
        )


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


def prepare_chat_messages(
    messages: list[LLMResponse | dict[str, Any] | CacheBoundary],
    scope: str | None,
    *,
    anthropic_cache_marking: bool = False,
    direct: bool = False,
) -> list[dict | CacheBoundary]:
    """Project stored turns; retain explicit fields in caller-written dictionaries.

    Portable reasoning demotion belongs to LLMResponse projection. A raw
    reasoning_content field is a caller's explicit wire setting, not a request
    to fold that text into content. Raw dictionaries still cannot carry opaque
    state, and request containers are detached before the SDK can mutate them.

    ``anthropic_cache_marking`` reflects whether this call's cache boundary will
    actually be marked with Anthropic's ``cache_control`` block form -- the
    same check ``CompletionClient._prepare_cache_boundary`` uses -- not
    ``scope``'s resolved provider. litellm's own provider resolution and
    NOOA's Anthropic-route detection can disagree for gateway-routed models
    (e.g. ``openai/azure/anthropic/...``), where ``scope`` resolves to
    ``"openai"`` even though Anthropic-style marking is what's actually
    applied. Content must be pre-wrapped in whichever shape marking will use,
    or a message's wire shape flips the turn it stops being the one marked.
    """
    from .chat_parts import project_chat_turn

    prepared: list[dict[str, Any] | CacheBoundary] = []
    private_call_ids: dict[str, str] = {}
    for original in messages:
        if isinstance(original, LLMResponse):
            message, ids = project_chat_turn(
                original, scope, anthropic_cache_marking=anthropic_cache_marking
            )
            private_call_ids.update(ids)
            if (
                message.get("content")
                or message.get("reasoning_content")
                or message.get("tool_calls")
                or any(
                    key in message
                    for key in ("thinking_blocks", "reasoning_items", "provider_specific_fields")
                )
            ):
                prepared.append(message)
            continue
        if isinstance(original, CacheBoundary):
            prepared.append(original)
            continue
        # Accept any Mapping; validation reads a plain dict before the one deep
        # copy that detaches caller-owned containers for the SDK.
        message = dict(original)
        reject_boundary_dict(message)
        reject_native_message(message, scope, reject_google_signature=direct)
        message = copy.deepcopy(message)
        if anthropic_cache_marking and isinstance(message.get("content"), str):
            # Same stability rationale as project_chat_turn's assistant-content
            # wrapping -- apply_cache_policy's Anthropic marking wraps whichever
            # message it marks this turn into a content block; unmarked
            # plain-dict messages (user turns, tool results) must already be in
            # that same shape or they'll flip once this message stops being the
            # one marked. Anthropic rejects an empty text block, so leave a
            # genuinely empty string alone.
            content = message["content"]
            if content:
                message["content"] = [wrap_anthropic_text(content)]
        call_id = message.get("tool_call_id")
        if isinstance(call_id, str):
            message["tool_call_id"] = private_call_ids.get(call_id, call_id)
        prepared.append(message)
    return prepared


def native_encrypted_reasoning_expected(
    api_params: dict[str, Any], scope: str | None, *, direct: bool = False
) -> bool:
    """Whether this call's route can return real ``reasoning.encrypted_content``.

    LiteLLM's provider resolution collapses every OpenAI-compatible gateway
    route (Hub-proxied open-weight models included) to the same provider name
    ("openai") that a genuine OpenAI/Azure call resolves to, so provider name
    alone cannot tell them apart -- both read as ``responses:openai:...`` or
    ``responses:azure:...`` in ``scope``. The actual endpoint can: only calls
    that really reach ``api.openai.com`` or an Azure OpenAI resource ever get
    an encrypted envelope back. Everything else that exposes reasoning does so
    as summary text only, as its native (not truncated) wire shape.
    """
    if scope and scope.startswith("responses:azure:"):
        return True
    return bool(
        scope
        and scope.startswith("responses:openai:")
        and _uses_native_openai_endpoint(api_params, direct=direct)
    )


def add_encrypted_reasoning_include(
    api_params: dict[str, Any], scope: str | None, *, native_encrypted_reasoning: bool
) -> None:
    """Request OpenAI encrypted reasoning only on endpoints known to support it."""
    configured = api_params.get("include")
    if isinstance(configured, (list, tuple, set)) and not configured:
        api_params.pop("include", None)
        return  # Explicit opt-out, including on native OpenAI/Azure endpoints.
    include = list(configured) if isinstance(configured, (list, tuple, set)) else []
    if configured is not None and not isinstance(configured, (list, tuple, set)):
        include.append(configured)
    if _ENCRYPTED_REASONING_INCLUDE in include:
        api_params["include"] = include
        return
    if native_encrypted_reasoning:
        include.append(_ENCRYPTED_REASONING_INCLUDE)
    if include:
        api_params["include"] = include
