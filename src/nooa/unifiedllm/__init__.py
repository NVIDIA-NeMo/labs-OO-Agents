# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified LLM client facade.

The concrete client implementation imports LiteLLM and its provider stack.
This package keeps lightweight data types importable without that cost and
loads provider-backed functionality only when the corresponding export is used.
"""

from __future__ import annotations

from typing import Any

_TYPE_EXPORTS = {
    "LLMResponse",
    "Tool",
    "ToolCall",
    "create_tool_from_callable",
}

_CLIENT_EXPORTS = {
    "UnifiedLLM",
    "CompletionClient",
    "ReasoningCompletionClient",
    "ResponsesClient",
    "extract_and_parse_json",
}

_REGISTRY_EXPORTS = {
    "MODELS",
    "ensure_loaded",
    "get_llm_client",
    "get_registry_config",
    "reload_registry",
    "resolve_api_key_from_config",
}

_RETRY_EXPORTS = {
    "EmptyContentError",
    "RetryingWrapper",
    "sync_retry",
    "with_retry",
}

_MODULE_BY_NAME = {
    **dict.fromkeys(_TYPE_EXPORTS, "nooa.unifiedllm.types"),
    **dict.fromkeys(_CLIENT_EXPORTS, "nooa.unifiedllm.unifiedllm"),
    **dict.fromkeys(_REGISTRY_EXPORTS, "nooa.unifiedllm.registry"),
    **dict.fromkeys(_RETRY_EXPORTS, "nooa.unifiedllm.retry"),
    "HttpConfig": "nooa.unifiedllm.http_config",
    "RetryConfig": "nooa.unifiedllm.retry_config",
    "FakeLLMClient": "nooa.unifiedllm.fake",
}

__all__ = [
    # Core classes
    "UnifiedLLM",
    "CompletionClient",
    "ReasoningCompletionClient",
    "ResponsesClient",
    # Model registry
    "get_llm_client",
    "get_registry_config",
    "reload_registry",
    "ensure_loaded",
    "resolve_api_key_from_config",
    "MODELS",
    # Tools
    "Tool",
    "ToolCall",
    "create_tool_from_callable",
    # Response types
    "LLMResponse",
    # HTTP config
    "HttpConfig",
    # Retry utilities
    "EmptyContentError",
    "RetryConfig",
    "RetryingWrapper",
    "with_retry",
    "sync_retry",
    # Testing
    "FakeLLMClient",
    # Utilities
    "extract_and_parse_json",
]


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
