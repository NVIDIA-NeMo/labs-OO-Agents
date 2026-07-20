# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nooa.unifiedllm.fake import FakeLLMClient
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.registry import (
    MODELS,
    ensure_loaded,
    get_llm_client,
    get_registry_config,
    reload_registry,
    resolve_api_key_from_config,
)
from nooa.unifiedllm.retry import (
    EmptyContentError,
    RetryingWrapper,
    sync_retry,
    with_retry,
)
from nooa.unifiedllm.retry_config import RetryConfig
from nooa.unifiedllm.unifiedllm import (
    CompletionClient,
    LLMResponse,
    ReasoningCompletionClient,
    ResponsesClient,
    Tool,
    ToolCall,
    UnifiedLLM,
    create_tool_from_callable,
    extract_and_parse_json,
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
