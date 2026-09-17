# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nooa.unifiedllm import unifiedllm


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [False, True])
async def test_cancellation_drains_response_collection(monkeypatch, override):
    """The shield covers response collection as well as provider dispatch."""
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    reply = unifiedllm.litellm.ModelResponse()

    async def completion(**kwargs):
        return reply

    async def collect(raw):
        assert raw is reply
        entered.set()
        await release.wait()
        finished.set()
        return raw

    monkeypatch.setattr(unifiedllm.litellm, "acompletion", completion)
    monkeypatch.setattr(unifiedllm, "_collect_async", collect)
    async with unifiedllm.CompletionClient("openai/test", api_key="test") as client:
        params = {"model": "openai/test", "api_key": "test"}
        if not override:
            params["client"] = client._http.async_client
        task = asyncio.create_task(client._asend(params))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.wait_for(finished.wait(), 2)


@pytest.mark.asyncio
async def test_litellm_acompletion_continues_after_caller_cancellation(monkeypatch):
    """Caller cancellation must not drop LiteLLM's nested provider coroutine."""
    provider_started = asyncio.Event()
    provider_finished = asyncio.Event()

    async def provider_coroutine():
        provider_started.set()
        await asyncio.sleep(0)
        provider_finished.set()
        return unifiedllm.litellm.ModelResponse()

    async def fake_acompletion(**_kwargs):
        provider = provider_coroutine()
        provider_started.set()
        await asyncio.sleep(0.05)
        return await provider

    monkeypatch.setattr(unifiedllm.litellm, "acompletion", fake_acompletion)

    async with unifiedllm.CompletionClient("openai/test", api_key="test") as client:
        task = asyncio.create_task(
            client._asend({"model": client.model, "client": client._http.async_client})
        )
        await asyncio.wait_for(provider_started.wait(), timeout=1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.wait_for(provider_finished.wait(), timeout=1)
