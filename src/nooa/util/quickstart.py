# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Provides some quickstart settings to get you started with some reasonable defaults and to make the quickstart examples in the README.md more concise.
"""

import asyncio
import os
from collections.abc import Callable, Coroutine
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from nooa import Agent, hidden, strategy
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm.registry import get_llm_client

# Load environment variables
load_dotenv(override=True)

# The examples run against any litellm-supported provider. By default they pick
# whichever credential you have set (see the README's "API Keys"):
#   * NVIDIA_API_KEY           -> NVIDIA build.nvidia.com NIM (public), served at
#                                 integrate.api.nvidia.com (litellm `nvidia_nim/`)
#   * OPENAI_API_KEY           -> OpenAI (public)
#   * NVIDIA_INFERENCE_API_KEY -> NVIDIA internal inference gateway
#                                 (inference-api.nvidia.com; NVIDIA employees)
# To use a specific model, set MODEL to any litellm name and provide its key,
# e.g. MODEL = "claude-haiku-4-5" with ANTHROPIC_API_KEY.
_internal_key = os.getenv("NVIDIA_INFERENCE_API_KEY") or os.getenv("NVIDIA_INTERNAL_API_KEY")
if os.getenv("NVIDIA_API_KEY"):
    # build.nvidia.com NIM. litellm routes `nvidia_nim/*` to
    # integrate.api.nvidia.com; it reads the key from NVIDIA_NIM_API_KEY, so
    # pass NVIDIA_API_KEY (the build.nvidia.com convention) explicitly.
    MODEL = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    llm = get_llm_client(MODEL, api_key=os.environ["NVIDIA_API_KEY"])
elif os.getenv("OPENAI_API_KEY"):
    MODEL = "gpt-5-mini"
    llm = get_llm_client(MODEL)
elif _internal_key:
    # NVIDIA-internal inference gateway (OpenAI-compatible).
    MODEL = "openai/azure/openai/gpt-5-mini"
    llm = get_llm_client(
        MODEL, api_base="https://inference-api.nvidia.com/v1", api_key=_internal_key
    )
else:
    # No key set — default to OpenAI so the examples raise a clear
    # missing-OPENAI_API_KEY error rather than a confusing one.
    MODEL = "gpt-5-mini"
    llm = get_llm_client(MODEL)


# Decorator for running example entry points
def autorun(func: Callable[[], Coroutine[Any, Any, Any]]) -> Callable[[], Coroutine[Any, Any, Any]]:
    # Mark the entry point hidden BEFORE running so it never leaks into any
    # agent's execution context as a callable tool (module-level functions are
    # visible-by-default; the example `main()` is harness glue, not a tool).
    hidden(func)
    print("\n\nEXAMPLE OUTPUT:")
    asyncio.run(func())
    return func


class Artwork:
    """A piece of art with an appraisal."""

    def __init__(self, title: str, artist: str, appraised_value: float):
        self.title = title
        self.artist = artist
        self._appraised_value = appraised_value

    def get_appraisal(self) -> dict[str, Any]:
        """Get the appraisal details including current market value."""
        return {
            "title": self.title,
            "artist": self.artist,
            "value": self._appraised_value,
            "currency": "USD",
        }


class StockHolding:
    """A stock position with shares and price."""

    def __init__(self, symbol: str, shares: int, price_per_share: float):
        self.symbol = symbol
        self._shares = shares
        self._price_per_share = price_per_share

    def get_total_value(self) -> float:
        """Calculate total value of the holding."""
        return self._shares * self._price_per_share


class Jewelry:
    """A piece of jewelry valued by carats."""

    def __init__(self, description: str, carats: float, rate_per_carat: float):
        self.description = description
        self._carats = carats
        self._rate_per_carat = rate_per_carat

    def compute_value(self) -> float:
        """Compute total value based on carats and rate."""
        return self._carats * self._rate_per_carat


class Collectible:
    """A collectible item whose value depends on condition."""

    def __init__(self, name: str, base_value: float, condition: str):
        self.name = name
        self._base_value = base_value
        self.condition = condition

    def estimate_value(self) -> float:
        """Estimate value based on condition. Returns adjusted value."""
        multipliers = {"mint": 1.0, "excellent": 0.85, "good": 0.7, "fair": 0.5}
        return self._base_value * multipliers.get(self.condition, 0.5)


# Export everything needed for examples
__all__ = [
    # LLM
    "llm",
    "MODEL",
    # Core
    "Agent",
    "strategy",
    # Strategies
    "CodeActStrategy",
    "PredictStrategy",
    # Pydantic
    "BaseModel",
    "Field",
    # Async
    "asyncio",
    # Example runner
    "autorun",
    # Types
    "Artwork",
    "StockHolding",
    "Jewelry",
    "Collectible",
]
