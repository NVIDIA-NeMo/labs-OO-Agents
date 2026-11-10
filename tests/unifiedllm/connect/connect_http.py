# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mock HTTP below the SDKs without replacing their required client classes."""

import json

import httpx


def mock_http(monkeypatch, handle):
    # LiteLLM otherwise fetches Anthropic beta-header metadata synchronously,
    # outside the async model transport. Use its bundled metadata in HTTP tests.
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    original = httpx.AsyncClient.__init__
    original_send = httpx.AsyncClient.send

    def init(self, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        kwargs["trust_env"] = False  # Proxy mounts must not bypass the mock transport.
        original(self, **kwargs)

    async def send(self, request, **kwargs):
        # A credential-less SDK can fall back to a cached, unmocked client.
        # Fail before I/O rather than allowing an accidental provider call.
        if not isinstance(self._transport, httpx.MockTransport):
            raise AssertionError("Connect HTTP tests require a mocked SDK transport")
        return await original_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)


def mock_post(monkeypatch, post):
    async def handle(request):
        return await post(
            None, str(request.url), json=json.loads(request.content), headers=request.headers
        )

    mock_http(monkeypatch, handle)


def response_body(style, text='{"value":323}'):
    common = {"id": "test-response", "model": "model"}
    if style == "responses":
        return {
            **common,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg-1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    if style == "anthropic":
        return {
            **common,
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 20, "output_tokens": 2},
        }
    return {
        **common,
        "object": "chat.completion",
        "created": 1,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    }
