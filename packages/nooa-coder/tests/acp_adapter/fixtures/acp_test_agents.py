# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agents for the ACP adapter tests, loaded by file spec (``<this file>:Class``)."""

from typing import Any

from nooa.errors import GenerationError
from nooa.interactive import InteractiveAgent
from nooa.unifiedllm import FakeLLMClient

MESSAGES = {
    # The wording CodeActStrategy uses (nooa/strategies/codeact.py).
    "tokens": (
        "The model used all available output tokens before completing a tool call. "
        "Increase `max_tokens` (16384 or more is often needed for reasoning models)."
    ),
    "iterations": (
        "Generation failed after 3 iterations (max_iterations=3). Unable to complete `handle`."
    ),
    "retries": (
        "Generation failed after 3 errors (max_retries=3). "
        "Unable to generate valid code for `handle`."
    ),
    "other": "Generation failed: the provider rejected the request",
}


class LimitAgent(InteractiveAgent, llm=FakeLLMClient()):
    """Every turn fails with the GenerationError named by the prompt text."""

    async def handle(self, notification: dict[str, list[Any]]) -> Any:
        text = str(notification["user_messages"][-1])
        raise GenerationError(MESSAGES[text])


class BrokenAgent(InteractiveAgent, llm=FakeLLMClient()):
    """Every turn fails with an ordinary exception."""

    async def handle(self, notification: dict[str, list[Any]]) -> Any:
        raise RuntimeError("the turn broke")
