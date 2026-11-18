# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which request interface a client sends through.

A client sends through one of four interfaces: ``chat`` (OpenAI-compatible Chat
Completions), ``anthropic`` (native Anthropic Messages), ``responses`` (the
OpenAI Responses API) and ``decision`` (native decision endpoints). The
interface is resolved from the actual LiteLLM dispatch (model, endpoint and
``custom_llm_provider``), never from the model name or ``api_style``.
``nooa.media_policy`` says which media kinds each interface can carry.

This module sits outside ``nooa.unifiedllm`` so that importing it (and
``nooa.media_policy``) stays light: LiteLLM is imported only when a provider is
actually resolved.
"""

from collections.abc import Mapping
from typing import Any


def interface_for_provider(provider: str | None) -> str:
    """Return the wire format a ``CompletionClient`` request uses for ``provider``.

    ``provider`` is the LiteLLM provider name for the request. LiteLLM sends
    ``anthropic`` requests as native Anthropic Messages; every other provider is
    given an OpenAI-compatible Chat request. ``ResponsesClient`` does not use this
    function: its requests always use the ``responses`` interface.
    """
    return "anthropic" if provider == "anthropic" else "chat"


def resolved_input_interface(
    model: str, config: Mapping[str, Any], *, client_type: str = "completion"
) -> str:
    """Return the interface a client built from ``model`` and ``config`` would use.

    ``client_type`` ``responses`` and ``decision`` fix the interface directly. For
    completion clients, LiteLLM resolves the provider from the model, the endpoint
    (``api_base`` or ``base_url``) and ``custom_llm_provider``, exactly as it does
    when sending. No network call, key lookup or metadata is used. If the provider
    cannot be resolved (for example an incomplete profile), the result is ``chat``,
    which does not restrict any media kind.
    """
    if client_type in {"responses", "decision"}:
        return client_type
    provider = resolved_provider(model, config)
    # LiteLLM cannot route some model strings; treat them as Chat, which is
    # also how CompletionClient would try to send them.
    return "chat" if provider is None else interface_for_provider(provider)


def resolved_provider(model: str, config: Mapping[str, Any]) -> str | None:
    """Return the LiteLLM provider a completion client for ``model`` sends to.

    Resolved by ``litellm.get_llm_provider`` from the model, the endpoint
    (``api_base`` or ``base_url``) and ``custom_llm_provider``, as when
    sending; no network call or key lookup. Returns None when LiteLLM cannot
    resolve the model.
    """
    if not model:
        return None
    import litellm

    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model=model,
            api_base=config.get("api_base") or config.get("base_url"),
            custom_llm_provider=config.get("custom_llm_provider"),
        )
    except Exception:
        return None
    return provider


def entry_input_interface(entry: Mapping[str, Any]) -> str:
    """Return the interface for a registry entry (``model_name`` plus settings)."""
    return resolved_input_interface(
        entry.get("model_name") or "",
        entry,
        client_type=entry.get("client_type") or "completion",
    )
