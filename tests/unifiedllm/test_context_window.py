# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient, get_llm_client


def _completion_response() -> litellm.ModelResponse:
    message = litellm.Message(content="ok", role="assistant")
    choice = litellm.Choices(message=message, index=0, finish_reason="stop")
    return litellm.ModelResponse(choices=[choice], model="test-model")


def _responses_response() -> SimpleNamespace:
    return SimpleNamespace(output=[], usage=None, status="completed", incomplete_details=None)


def test_completion_client_context_window_config_is_honored():
    client = CompletionClient(model="unknown/context-window-model", context_window=262144)

    assert client.context_window == 262144
    assert "context_window" not in client.config


def test_responses_client_context_window_config_is_honored():
    client = ResponsesClient(model="unknown/context-window-model", context_window=262144)

    assert client.context_window == 262144
    assert "context_window" not in client.config


def test_context_window_config_overrides_registry_config():
    client = CompletionClient(model="unknown/context-window-model", context_window=262144)
    client._registry_config = {"context_window": 131072}

    assert client.context_window == 262144


def test_zero_context_window_remains_an_explicit_override():
    client = CompletionClient(model="unknown/context-window-model", context_window=0)
    client._registry_config = {"context_window": 131072}

    assert client.context_window == 0
    assert "context_window" not in client.config


def test_none_context_window_falls_through_to_registry_config():
    client = CompletionClient(model="unknown/context-window-model", context_window=None)
    client._registry_config = {"context_window": 131072}

    assert client.context_window == 131072
    assert "context_window" not in client.config


def test_registry_only_context_window_still_resolves():
    client = CompletionClient(model="unknown/context-window-model")
    client._registry_config = {"context_window": 131072}

    assert client.context_window == 131072


def test_get_llm_client_override_is_metadata_not_request_config(monkeypatch):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setitem(
        registry.MODELS,
        "context-window-alias",
        {"model_name": "unknown/context-window-model", "context_window": 131072},
    )

    client = get_llm_client("context-window-alias", context_window=262144)

    assert client.context_window == 262144
    assert "context_window" not in client.config


def test_completion_sync_request_omits_context_window():
    client = CompletionClient(model="test-model", context_window=262144)

    with patch("litellm.completion", return_value=_completion_response()) as completion:
        client.call([{"role": "user", "content": "hello"}])

    assert "context_window" not in completion.call_args.kwargs


@pytest.mark.asyncio
async def test_completion_async_request_omits_context_window():
    client = CompletionClient(model="test-model", context_window=262144)

    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=_completion_response()
    ) as completion:
        await client.acall([{"role": "user", "content": "hello"}])

    assert "context_window" not in completion.call_args.kwargs


def test_responses_sync_request_omits_context_window():
    client = ResponsesClient(model="test-model", context_window=262144)

    with patch("litellm.responses", return_value=_responses_response()) as responses:
        client.call([{"role": "user", "content": "hello"}])

    assert "context_window" not in responses.call_args.kwargs


@pytest.mark.asyncio
async def test_responses_async_request_omits_context_window():
    client = ResponsesClient(model="test-model", context_window=262144)

    with patch(
        "litellm.aresponses", new_callable=AsyncMock, return_value=_responses_response()
    ) as responses:
        await client.acall([{"role": "user", "content": "hello"}])

    assert "context_window" not in responses.call_args.kwargs
