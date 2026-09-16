# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the legacy dependency only when a legacy client needs it."""

import importlib
from copy import deepcopy

_initialized = False


def _module():
    global _initialized
    module = importlib.import_module("litellm")
    if not _initialized:
        module.modify_params = True
        module.disable_aiohttp_transport = True
        _initialized = True
    return module


class _LazyLiteLLM:
    def __getattr__(self, name):
        return getattr(_module(), name)

    def __setattr__(self, name, value):
        setattr(_module(), name, value)

    def __delattr__(self, name):
        delattr(_module(), name)


litellm = _LazyLiteLLM()


def preserve_readable_reasoning(params):
    """Keep readable history when a legacy adapter strips Chat extensions.

    This is a transport serialization fallback, not opaque-state replay. Native
    adapters for signed state keep their own handling; OpenAI-compatible adapters
    receive reasoning_content unchanged. Never mutate the caller's history.
    """
    messages = params.get("messages")
    if not isinstance(messages, list) or not any(
        isinstance(m, dict) and m.get("role") == "assistant" and m.get("reasoning_content")
        for m in messages
    ):
        return params
    _, provider, _, _ = litellm.get_llm_provider(
        model=params["model"],
        custom_llm_provider=params.get("custom_llm_provider"),
        api_base=params.get("api_base") or params.get("base_url"),
        api_key=params.get("api_key"),
    )
    if provider in set(litellm.openai_compatible_providers) | {
        "openai",
        "azure",
        "openrouter",
        "anthropic",
        "gemini",
        "vertex_ai",
        "bedrock",
    }:
        return params
    result = dict(params)
    result["messages"] = deepcopy(messages)
    for message in result["messages"]:
        if message.get("role") != "assistant":
            continue
        reasoning = message.pop("reasoning_content", None)
        if not isinstance(reasoning, str) or not reasoning:
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [{"type": "text", "text": reasoning}, *content]
        else:
            message["content"] = reasoning + ("\n\n" + content if content else "")
    return result
