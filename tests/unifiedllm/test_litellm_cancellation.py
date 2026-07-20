# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nooa.unifiedllm import unifiedllm


@pytest.mark.asyncio
async def test_litellm_acompletion_continues_after_caller_cancellation(monkeypatch):
    """Caller cancellation must not drop LiteLLM's nested provider coroutine."""
    provider_started = asyncio.Event()
    provider_finished = asyncio.Event()

    async def provider_coroutine():
        provider_started.set()
        await asyncio.sleep(0)
        provider_finished.set()
        return {"ok": True}

    async def fake_acompletion(**_kwargs):
        provider = provider_coroutine()
        provider_started.set()
        await asyncio.sleep(0.05)
        return await provider

    monkeypatch.setattr(unifiedllm.litellm, "acompletion", fake_acompletion)

    task = asyncio.create_task(unifiedllm._litellm_acompletion({}))
    await asyncio.wait_for(provider_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(provider_finished.wait(), timeout=1)
