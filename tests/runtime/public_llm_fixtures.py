# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core SDK HTTP fixtures: never a public acall monkeypatch or a live endpoint."""

import json

import anthropic
import httpx
import openai
import pytest

from nooa.unifiedllm import load_llm, registry
from tests.unifiedllm.test_native_load_llm import zero_delay
from tests.unifiedllm.test_native_projection import payload


class NativeModels:
    def __init__(self):
        self.sdks = []
        self.http = []
        self.views = []
        self.seen = []

    def make(self, family="openai-chat", *, handler=None, **settings):
        def default_handler(request):
            self.seen.append((family, json.loads(request.content)))
            return httpx.Response(200, json=payload(family))

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler or default_handler))
        cls = anthropic.AsyncAnthropic if family == "anthropic-messages" else openai.AsyncOpenAI
        sdk = cls(
            api_key="fixture-key",
            base_url="https://fixture.invalid"
            if family == "anthropic-messages"
            else "https://fixture.invalid/v1",
            http_client=http,
            max_retries=9,
        )
        self.sdks.append(sdk)
        self.http.append(http)
        options = {
            "adapter": family,
            "model": "fixture-model",
            "base_url": "https://fixture.invalid/v1",
            "api_key": "fixture-key",
            "client": sdk,
            "context_window": 12800,
            "max_tokens": 100,
            "retry_config": zero_delay(),
        }
        options.update(settings)
        view = load_llm("fixture", **options)
        self.views.append(view)
        return view

    async def close(self):
        for view in self.views:
            await view.aclose()
        # Public views borrow the originals, including their retry configuration.
        assert all(sdk.max_retries == 9 for sdk in self.sdks)
        assert all(not http.is_closed for http in self.http)
        for sdk in self.sdks:
            await sdk.close()


@pytest.fixture
async def native_models(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})
    fixture = NativeModels()
    yield fixture
    await fixture.close()
