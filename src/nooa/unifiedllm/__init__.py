# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified LLM client package.

Keep this package import light: strategies and agent classes import protocol
types during class definition, while LiteLLM-backed clients are only needed
when a real model client is constructed or called.
"""

from nooa.unifiedllm.registry import (
    MODELS,
    ensure_loaded,
    get_llm_client,
    get_registry_config,
    reload_registry,
    resolve_api_key_from_config,
)
from nooa.unifiedllm.types import (
    LLMResponse,
    Tool,
    ToolCall,
    create_tool_from_callable,
)

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

_CLIENT_EXPORTS = {
    "CompletionClient",
    "ReasoningCompletionClient",
    "ResponsesClient",
    "UnifiedLLM",
    "extract_and_parse_json",
}
_RETRY_EXPORTS = {"EmptyContentError", "RetryingWrapper", "sync_retry", "with_retry"}


def __getattr__(name: str):
    if name in _CLIENT_EXPORTS:
        from nooa.unifiedllm import unifiedllm as _clients

        value = getattr(_clients, name)
    elif name == "FakeLLMClient":
        from nooa.unifiedllm.fake import FakeLLMClient as value
    elif name == "HttpConfig":
        from nooa.unifiedllm.http_config import HttpConfig as value
    elif name == "RetryConfig":
        from nooa.unifiedllm.retry_config import RetryConfig as value
    elif name in _RETRY_EXPORTS:
        from nooa.unifiedllm import retry as _retry

        value = getattr(_retry, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
