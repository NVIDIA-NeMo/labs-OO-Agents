# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM + embedding builders for the ARC-AGI-3 solver example.

Configured purely from the environment (see repo-root ``.env``)::

    ARC_LLM_MODEL=openai/openai/gpt-5.4
    ARC_LLM_BASE_URL=https://inference-api.nvidia.com/v1
    ARC_LLM_API_KEY=...
    MEM_EMBED_MODEL=openai/azure/openai/text-embedding-3-large
    MEM_EMBED_BASE_URL=https://inference-api.nvidia.com/v1
    MEM_EMBED_API_KEY=...
    MEM_EMBED_DIMS=1024

Follows ``examples/memory_bench/llm.py``; falls back to ``FakeLLMClient`` /
hashing embeddings when credentials are absent so smoke tests run offline.
"""

from __future__ import annotations

import os
from pathlib import Path

from nooa_memory.config import EmbeddingConfig

from nooa.unifiedllm import CompletionClient, FakeLLMClient, RetryConfig
from nooa.unifiedllm.unifiedllm import ResponsesClient


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency): real environment wins via setdefault."""
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return


_load_dotenv()


def has_llm_creds() -> bool:
    return all(os.environ.get(k) for k in ("ARC_LLM_MODEL", "ARC_LLM_BASE_URL", "ARC_LLM_API_KEY"))


def has_embedding_creds() -> bool:
    return bool(os.environ.get("MEM_EMBED_API_KEY"))


def build_llm(
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    request_timeout: float | None = 180.0,
):
    """Real gateway client when ARC_LLM_* is set, else a FakeLLMClient."""
    if not has_llm_creds():
        return FakeLLMClient()
    extra: dict[str, object] = {}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    if request_timeout is not None:
        # Hard per-request HTTP timeout: a gateway request that hangs would otherwise
        # block the solving loop forever (litellm calls aren't cancellation-responsive).
        extra["timeout"] = request_timeout
    model = "openai/" + os.environ["ARC_LLM_MODEL"]  # litellm needs the provider prefix
    reasoning_model = "nemotron" in model.lower()
    # Reasoning-effort routing (verified live against the gateway 2026-07-16):
    #  * Anthropic Opus 4.8+ uses the *adaptive* thinking API — thinking.type=adaptive
    #    + output_config.effort. litellm's reasoning_effort maps to the OLD
    #    thinking.type=enabled, which Opus 4.8 REJECTS ("not supported for this model;
    #    use thinking.type.adaptive and output_config.effort"), and drop_params then
    #    silently drops it -> reasoning_tokens=0, NO thinking. So send the adaptive
    #    params raw via extra_body (CompletionClient forwards **config to litellm).
    #  * gpt-5.x (Responses API) and everything else keep plain reasoning_effort.
    if reasoning_effort:
        if "anthropic" in model.lower():
            extra["extra_body"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": reasoning_effort},
            }
        else:
            extra["reasoning_effort"] = reasoning_effort
    # gpt-5.x on this gateway must use the Responses API (like the reference profiles'
    # use_responses_api profiles): chat-completions with tools + reasoning_effort
    # gets rerouted gateway-side with a mangled model id → 403 key_model_access_denied.
    # (Observed live when this matched only "gpt-5.5": a gpt-5.6-sol fleet 403'd on
    # every call. gpt-5.4 / 5.5 / 5.6-sol are all verified working via Responses.)
    use_responses = os.environ.get("ARC_LLM_USE_RESPONSES", "1" if "gpt-5" in model else "0") == "1"
    client_cls = ResponsesClient if use_responses else CompletionClient
    kwargs = dict(
        model=model,
        api_base=os.environ["ARC_LLM_BASE_URL"],
        api_key=os.environ["ARC_LLM_API_KEY"],
        retry_config=RetryConfig(retry_on_empty_content=not reasoning_model),
        drop_params=True,  # the gateway rejects some params on the reasoning route
        **extra,
    )
    return client_cls(**kwargs)


def llm_uri() -> str:
    """Model identifier recorded in run outputs."""
    return os.environ.get("ARC_LLM_MODEL", "fake-llm")


def build_embedding_config(force: str = "auto") -> EmbeddingConfig:
    """Gateway embeddings when keys are set, else the deterministic hashing embedder.

    ``force`` ∈ {"auto", "litellm", "hashing"}.
    """
    want_real = force == "litellm" or (force == "auto" and has_embedding_creds())
    if want_real:
        dims = os.environ.get("MEM_EMBED_DIMS")
        return EmbeddingConfig(
            backend="litellm",
            model=os.environ.get("MEM_EMBED_MODEL", "openai/azure/openai/text-embedding-3-large"),
            endpoint=os.environ.get("MEM_EMBED_BASE_URL", "https://inference-api.nvidia.com/v1"),
            api_key=os.environ.get("MEM_EMBED_API_KEY"),
            dimensions=int(dims) if dims else 1024,
        )
    return EmbeddingConfig(backend="hashing", dim=256)
