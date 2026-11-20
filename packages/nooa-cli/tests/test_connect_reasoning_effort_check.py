# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A configured reasoning effort the model's metadata does not list gets a clear warning."""

import json
import logging

import httpx
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect, registry
from tests.unifiedllm.connect.connect_http import mock_http

# The OpenRouter catalogue record shape (moonshotai/kimi-k2-thinking style):
# supported efforts without "medium", default "max".
RECORD = {
    "id": "vendor/model",
    "context_length": 262144,
    "reasoning": {"supported_efforts": ["max", "high", "low", "none"], "default_effort": "max"},
}


def _catalogue(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        assert str(request.url).split("?")[0] == connect.CATALOGUE_URL
        return httpx.Response(200, json={"data": [RECORD]})

    mock_http(monkeypatch, handle)
    return calls


def _model_info(*extra):
    return CliRunner().invoke(
        command,
        [
            "vendor/model",
            "--stage",
            "model-info",
            "--endpoint",
            "https://openrouter.ai/api/v1",
            "--api-key-env",
            "",
            *extra,
        ],
    )


def test_model_info_warns_for_an_unlisted_effort(monkeypatch):
    calls = _catalogue(monkeypatch)
    result = _model_info("--reasoning-level", "medium")

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    warning = report["data"]["reasoning_warning"]
    assert "'medium' is not supported" in warning
    assert "max, high, low, none" in warning and "(default max)" in warning
    assert "medium" in result.stderr and "not supported" in result.stderr
    assert len(calls) == 1  # the same metadata read as before, nothing extra


def test_model_info_is_quiet_for_a_listed_effort_or_none(monkeypatch):
    _catalogue(monkeypatch)
    for extra in (("--reasoning-level", "high"), ()):
        result = _model_info(*extra)
        assert result.exit_code == 0, result.output
        assert "reasoning_warning" not in json.loads(result.stdout)["data"]
        assert "not supported" not in result.stderr


def _entry(**extra):
    levels = {
        label: {"reasoning_effort": label} for label in RECORD["reasoning"]["supported_efforts"]
    }
    return {
        "model": "openai/vendor/model",
        "api_base": "https://openrouter.ai/api/v1",
        "reasoning_levels": levels,
        "reasoning_default": "max",
        "media_inputs": [],
        "provenance": {"reasoning_levels": {"source": "catalogue", "template": "effort"}},
        **extra,
    }


def test_config_load_warns_once_for_an_unlisted_effort(monkeypatch, caplog):
    monkeypatch.setattr(registry, "_warned_effort", set())
    monkeypatch.setattr(
        httpx.Client, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
    )
    with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.registry"):
        registry.client_from_config("kimi", _entry(reasoning_effort="medium")).close()
        registry.client_from_config("kimi", _entry(reasoning_effort="medium")).close()
    lines = [r.getMessage() for r in caplog.records if "reasoning effort" in r.getMessage()]
    assert len(lines) == 1
    assert "Model 'kimi'" in lines[0] and "'medium' is not supported" in lines[0]
    assert "from the catalogue" in lines[0]


def test_config_load_is_quiet_when_supported_or_unknown(monkeypatch, caplog):
    monkeypatch.setattr(registry, "_warned_effort", set())
    with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.registry"):
        registry.client_from_config("a", _entry(reasoning_effort="high")).close()
        registry.client_from_config(
            "b", _entry(extra_body={"reasoning": {"effort": "low"}})
        ).close()
        no_levels = _entry(reasoning_effort="medium")
        del no_levels["reasoning_levels"], no_levels["reasoning_default"]
        registry.client_from_config("c", no_levels).close()
    assert not [r for r in caplog.records if "reasoning effort" in r.getMessage()]


def test_reasoning_level_selection_is_not_a_raw_effort(monkeypatch, caplog):
    monkeypatch.setattr(registry, "_warned_effort", set())
    with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.registry"):
        registry.client_from_config("d", _entry(reasoning_level="low")).close()
    assert not [r for r in caplog.records if "reasoning effort" in r.getMessage()]
