# SPDX-License-Identifier: Apache-2.0
"""Cancellation behavior for the private AnyLLM transport boundary."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nooa.unifiedllm import CompletionClient


@pytest.mark.asyncio
async def test_acompletion_propagates_caller_cancellation():
    client = CompletionClient("test-model")
    started = asyncio.Event()

    async def blocked(**kwargs):
        started.set()
        await asyncio.Event().wait()

    client._transport.acompletion = AsyncMock(side_effect=blocked)
    task = asyncio.create_task(client.acall([{"role": "user", "content": "hi"}]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
