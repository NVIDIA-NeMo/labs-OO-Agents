# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live provider compatibility smoke tests.

One parametrized test per capability, one row per provider. Each row is guarded
by its own ``provider_compat_<id>`` marker so a specific provider can be
selected with ``-m provider_compat_ollama`` etc. All rows are also marked
``integration``, so nothing here runs in the default suite.

A row that cannot reach its server, or whose configured model isn't loaded,
skips with a clear reason rather than failing.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import get_llm_client


class ArithmeticReport(BaseModel):
    total: int
    label: str


class Sentiment(BaseModel):
    label: Literal["positive", "negative"]


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    base_url_env: str
    model_env: str
    api_key_env: str | None
    default_base_url: str
    default_model: str
    model_prefixes: tuple[str, ...]
    health_url: Callable[[str], str]
    parse_available: Callable[[dict], set[str]]


def _ollama_health(base_url: str) -> str:
    server = base_url.rstrip("/")
    if server.endswith("/v1"):
        server = server[: -len("/v1")]
    return f"{server}/api/tags"


def _openai_compat_health(base_url: str) -> str:
    server = base_url.rstrip("/")
    if server.endswith("/v1"):
        return f"{server}/models"
    return f"{server}/v1/models"


OLLAMA = ProviderSpec(
    id="ollama",
    base_url_env="NOOA_TEST_OLLAMA_BASE_URL",
    model_env="NOOA_TEST_OLLAMA_MODEL",
    api_key_env=None,
    default_base_url="http://localhost:11434",
    default_model="ollama_chat/qwen3:1.7b",
    model_prefixes=("ollama_chat/",),
    health_url=_ollama_health,
    parse_available=lambda payload: {e.get("name") for e in payload.get("models", [])},
)

VLLM = ProviderSpec(
    id="vllm",
    base_url_env="NOOA_TEST_VLLM_BASE_URL",
    model_env="NOOA_TEST_VLLM_MODEL",
    api_key_env="NOOA_TEST_VLLM_API_KEY",
    default_base_url="http://localhost:8000/v1",
    default_model="hosted_vllm/Qwen/Qwen3-1.7B",
    model_prefixes=("hosted_vllm/", "openai/"),
    health_url=_openai_compat_health,
    parse_available=lambda payload: {e.get("id") for e in payload.get("data", [])},
)


PROVIDERS = [
    pytest.param(OLLAMA, marks=pytest.mark.provider_compat_ollama, id="ollama"),
    pytest.param(VLLM, marks=pytest.mark.provider_compat_vllm, id="vllm"),
]


def _served_model_name(model: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _require_model(spec: ProviderSpec, base_url: str, model: str, api_key: str | None) -> None:
    url = spec.health_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError) as exc:
        pytest.skip(f"{spec.id} server is not reachable at {url}: {exc}")

    served = _served_model_name(model, spec.model_prefixes)
    available = spec.parse_available(payload)
    if served not in available:
        pytest.skip(
            f"{spec.id} model {served!r} is not available at {url}; "
            f"available: {sorted(a for a in available if a)!r}"
        )


def _build_llm(spec: ProviderSpec) -> tuple[object, float]:
    model = os.getenv(spec.model_env, spec.default_model)
    base_url = os.getenv(spec.base_url_env, spec.default_base_url)
    api_key = os.getenv(spec.api_key_env) if spec.api_key_env else None
    max_tokens = int(os.getenv(f"NOOA_TEST_{spec.id.upper()}_MAX_TOKENS", "4096"))
    timeout_seconds = float(os.getenv(f"NOOA_TEST_{spec.id.upper()}_TIMEOUT_SECONDS", "180"))

    _require_model(spec, base_url, model, api_key)

    extra: dict = {"api_key": api_key} if api_key else {}
    llm = get_llm_client(
        model,
        api_base=base_url,
        temperature=0.0,
        max_tokens=max_tokens,
        **extra,
    )
    return llm, timeout_seconds


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("spec", PROVIDERS)
async def test_codeact_tool_round_trip(spec: ProviderSpec) -> None:
    """CodeAct can call a tool and return a typed Pydantic value."""
    llm, timeout_seconds = _build_llm(spec)

    class CodeActSmokeAgent(Agent, llm=llm):
        def add_numbers(self, left: int, right: int) -> int:
            """Add two integers."""
            return left + right

        def build_report(self, left: int, right: int) -> ArithmeticReport:
            """Build an arithmetic report from two integers."""
            return ArithmeticReport(total=self.add_numbers(left, right), label="computed")

        async def report(self) -> ArithmeticReport:
            """Use Python execution to call self.build_report(19, 23), then return the report object."""
            ...

    agent = CodeActSmokeAgent()
    report = await asyncio.wait_for(agent.report(), timeout=timeout_seconds)

    assert report == ArithmeticReport(total=42, label="computed")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("spec", PROVIDERS)
async def test_predict_structured_output(spec: ProviderSpec) -> None:
    """PredictStrategy returns a valid Pydantic model via a single LLM call."""
    llm, timeout_seconds = _build_llm(spec)

    class PredictSmokeAgent(Agent, llm=llm):
        @strategy(PredictStrategy())
        async def classify(self, text: str) -> Sentiment:
            """Classify the sentiment of the following text as either 'positive' or 'negative'.

            Text: {text}
            """
            ...

    agent = PredictSmokeAgent()
    result = await asyncio.wait_for(
        agent.classify("I absolutely love this product, it works perfectly!"),
        timeout=timeout_seconds,
    )

    assert result == Sentiment(label="positive")
