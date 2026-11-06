# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model selection in nooa.util.quickstart (no LLM calls)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

KEYS = (
    "NOOA_QUICKSTART_MODEL",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "NVIDIA_INFERENCE_API_KEY",
    "NVIDIA_INTERNAL_API_KEY",
)


def _load(monkeypatch, **env):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("nooa.util.quickstart", None)
    with (
        patch("dotenv.load_dotenv"),
        patch("nooa.unifiedllm.registry.get_llm_client") as client,
    ):
        module = importlib.import_module("nooa.util.quickstart")
    sys.modules.pop("nooa.util.quickstart", None)
    return module, client


def test_named_model_wins(monkeypatch):
    module, client = _load(
        monkeypatch, NOOA_QUICKSTART_MODEL="my-connected-model", OPENAI_API_KEY="k"
    )
    assert module.MODEL == "my-connected-model"
    client.assert_called_once_with("my-connected-model")


def test_openai_key_uses_openai_without_a_hint(monkeypatch, capsys):
    module, client = _load(monkeypatch, OPENAI_API_KEY="k")
    assert module.MODEL == "gpt-5-mini"
    client.assert_called_once_with("gpt-5-mini")
    assert "nooa connect" not in capsys.readouterr().err


@pytest.mark.parametrize("env", [{}, {"NVIDIA_INFERENCE_API_KEY": "k"}])
def test_unconfigured_model_points_to_nooa_connect(monkeypatch, capsys, env):
    module, _client = _load(monkeypatch, **env)
    assert module.MODEL == "gpt-5-mini"
    err = capsys.readouterr().err
    assert "nooa connect" in err
    assert "NOOA_QUICKSTART_MODEL" in err
