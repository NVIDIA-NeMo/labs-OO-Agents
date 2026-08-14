# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model selection for the ACP entry point."""

import click.testing
import pytest
from nooa_acp.cli import command


@pytest.fixture
def stubbed_serve(monkeypatch):
    """Capture the llm_factory the command builds instead of serving."""
    captured = {}

    def fake_serve(llm_factory):
        captured["llm_factory"] = llm_factory
        return "coroutine-placeholder"

    monkeypatch.setattr("nooa_acp.server.serve", fake_serve)
    monkeypatch.setattr("nooa_acp.cli.asyncio.run", lambda coro: coro)
    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda *a, **k: None)
    return captured


def test_model_is_required(monkeypatch):
    monkeypatch.delenv("NOOA_MODEL", raising=False)

    result = click.testing.CliRunner().invoke(command, [])

    # No default model: the caller has to choose one.
    assert result.exit_code == 2
    assert "--model" in result.output


def test_model_is_read_from_the_environment(monkeypatch, stubbed_serve):
    monkeypatch.setenv("NOOA_MODEL", "openai/gpt-4o-mini")
    requested = {}
    monkeypatch.setattr(
        "nooa.unifiedllm.get_llm_client",
        lambda name, **kwargs: requested.setdefault("name", name),
    )

    result = click.testing.CliRunner().invoke(command, [])

    assert result.exit_code == 0, result.output
    stubbed_serve["llm_factory"]()
    assert requested["name"] == "openai/gpt-4o-mini"


def test_explicit_flag_overrides_the_environment(monkeypatch, stubbed_serve):
    monkeypatch.setenv("NOOA_MODEL", "openai/gpt-4o-mini")
    requested = {}
    monkeypatch.setattr(
        "nooa.unifiedllm.get_llm_client",
        lambda name, **kwargs: requested.setdefault("name", name),
    )

    result = click.testing.CliRunner().invoke(command, ["--model", "anthropic/claude-sonnet-4-5"])

    assert result.exit_code == 0, result.output
    stubbed_serve["llm_factory"]()
    assert requested["name"] == "anthropic/claude-sonnet-4-5"
