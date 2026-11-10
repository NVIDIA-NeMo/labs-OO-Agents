# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model selection for the ACP entry point."""

import os
import shutil
import subprocess

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


def _console_script() -> str:
    """Locate the installed ``nooa-acp`` console script.

    Deliberately fails rather than skips. Every other test in this package
    imports ``nooa_acp``, so the package is always installed when these run; a
    missing script means the ``[project.scripts]`` entry is broken, which is
    exactly the breakage this test exists to catch.
    """
    path = shutil.which("nooa-acp")
    assert path is not None, "nooa-acp console script is not installed — check [project.scripts]"
    return path


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("NOOA_MODEL", None)
    return env


def test_console_script_is_installed_and_runnable():
    # Covers the [project.scripts] -> nooa_acp.cli:main binding, which the
    # in-process CliRunner tests and the fake_agent fixture both bypass.
    result = subprocess.run(
        [_console_script(), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )

    assert result.returncode == 0, result.stderr
    assert "Serve the NOOA coding agent over ACP" in result.stdout


def test_console_script_requires_a_model():
    # Also proves main() reaches the click command: without the wiring this
    # would not produce a usage error.
    result = subprocess.run(
        [_console_script()],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )

    assert result.returncode == 2
    assert "--model" in result.stderr
