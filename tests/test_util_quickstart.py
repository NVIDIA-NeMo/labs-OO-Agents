# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quickstart credential precedence and provider routing, without live clients."""

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "credentials,model,kwargs",
    [
        (
            {
                "NVIDIA_API_KEY": "public",
                "OPENAI_API_KEY": "openai",
                "NVIDIA_INFERENCE_API_KEY": "internal",
            },
            "nvidia_nim/nvidia/nemotron-3.5-super-vl-120b-a12b",
            {"api_key": "public"},
        ),
        (
            {"OPENAI_API_KEY": "openai", "NVIDIA_INFERENCE_API_KEY": "internal"},
            "gpt-5-mini",
            {},
        ),
        (
            {"NVIDIA_INFERENCE_API_KEY": "internal", "NVIDIA_INTERNAL_API_KEY": "legacy"},
            "openai/azure/openai/gpt-5-mini",
            {"api_base": "https://inference-api.nvidia.com/v1", "api_key": "internal"},
        ),
        (
            {"NVIDIA_INTERNAL_API_KEY": "legacy"},
            "openai/azure/openai/gpt-5-mini",
            {"api_base": "https://inference-api.nvidia.com/v1", "api_key": "legacy"},
        ),
        (
            {"NVIDIA_INFERENCE_API_KEY": "", "NVIDIA_INTERNAL_API_KEY": "legacy"},
            "openai/azure/openai/gpt-5-mini",
            {"api_base": "https://inference-api.nvidia.com/v1", "api_key": "legacy"},
        ),
        ({"NVIDIA_API_KEY": "", "OPENAI_API_KEY": "openai"}, "gpt-5-mini", {}),
        ({}, "gpt-5-mini", {}),
    ],
)
def test_quickstart_provider_selection(monkeypatch, credentials, model, kwargs):
    """Public keys stay public, precedence is stable, and no-key errors use OpenAI."""
    for key in (
        "NVIDIA_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "OPENAI_API_KEY",
        "NVIDIA_INFERENCE_API_KEY",
        "NVIDIA_INTERNAL_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in credentials.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    factory = Mock()
    monkeypatch.setattr("nooa.unifiedllm.registry.get_llm_client", factory)

    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "src/nooa/util/quickstart.py")
    )

    assert namespace["MODEL"] == model
    assert namespace["llm"] is factory.return_value
    factory.assert_called_once_with(model, **kwargs)
