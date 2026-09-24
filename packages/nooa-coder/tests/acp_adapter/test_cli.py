# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The nooa-coder command line: model, agent, sessions directory, tee."""

import shutil
import subprocess

import click.testing
import pytest
from nooa_coder.acp.cli import command


@pytest.fixture
def served(monkeypatch):
    """Capture what the command would serve with, instead of serving."""
    captured: dict = {}
    monkeypatch.setattr("nooa_coder.acp.cli.run", lambda **kwargs: captured.update(kwargs))
    # The real one repoints this process's stdout at stderr.
    monkeypatch.setattr("nooa_coder.acp.cli.reserve_stdout_for_acp", lambda: 1)
    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda *a, **k: None)
    return captured


@pytest.fixture
def requested(monkeypatch):
    calls: list = []

    def fake_client(name, **kwargs):
        calls.append((name, kwargs))
        return name

    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", fake_client)
    return calls


def _invoke(args, **kwargs):
    return click.testing.CliRunner().invoke(command, args, **kwargs)


def test_a_model_is_required(monkeypatch):
    monkeypatch.delenv("NOOA_MODEL", raising=False)
    result = _invoke([])
    assert result.exit_code == 2
    assert "--model" in result.output


def test_the_model_comes_from_the_environment_or_the_flag(monkeypatch, served, requested, tmp_path):
    monkeypatch.setenv("NOOA_MODEL", "env-model")
    assert _invoke([]).exit_code == 0
    assert served["model"] == "env-model"
    assert _invoke(["--model", "flag-model"]).exit_code == 0
    assert served["model"] == "flag-model"
    # The factory builds the alias a session asks for, else the default.
    served["llm_factory"](None, tmp_path)
    served["llm_factory"]("other", tmp_path)
    assert requested == [("flag-model", {"client_type": None}), ("other", {"client_type": None})]


def test_client_type_and_the_nvidia_key_reach_the_client(monkeypatch, served, requested, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = _invoke(["--model", "nvidia_nim/some/model", "--client-type", "responses"])
    assert result.exit_code == 0, result.output
    served["llm_factory"](None, tmp_path)
    served["llm_factory"]("openai/gpt", tmp_path)
    assert requested == [
        ("nvidia_nim/some/model", {"client_type": "responses", "api_key": "nvapi-test"}),
        ("openai/gpt", {"client_type": "responses"}),
    ]


def test_a_relative_agent_file_is_resolved_where_the_command_runs(served, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _invoke(["--model", "m", "--agent", "agents/my_agent.py:MyAgent"]).exit_code == 0
    assert served["agent_spec"] == f"{tmp_path / 'agents/my_agent.py'}:MyAgent"
    assert _invoke(["--model", "m", "--agent", "pkg.module:Agent"]).exit_code == 0
    assert served["agent_spec"] == "pkg.module:Agent"


def test_agent_and_legacy_agent_exclude_each_other(served):
    result = _invoke(["--model", "m", "--agent", "pkg:A", "--legacy-agent"])
    assert result.exit_code == 2
    assert "--legacy-agent" in result.output


def test_legacy_agent_selects_the_coding_agent(served):
    from nooa_coder.coding.identity import CODING_AGENT

    assert _invoke(["--model", "m", "--legacy-agent"]).exit_code == 0
    assert served["agent_spec"] == CODING_AGENT
    assert _invoke(["--model", "m"]).exit_code == 0
    assert served["agent_spec"] is None  # the workspace setting, else the coding agent


def test_sessions_dir_and_tee_are_passed_on(served, tmp_path):
    result = _invoke(
        ["--model", "m", "--sessions-dir", str(tmp_path / "s"), "--tee", str(tmp_path / "t.jsonl")]
    )
    assert result.exit_code == 0, result.output
    assert served["sessions_dir"] == tmp_path / "s"
    assert served["tee"] == tmp_path / "t.jsonl"


def test_worker_mode_is_not_available_yet(served):
    result = _invoke(["--model", "m", "--worker", "/tmp/socket"])
    assert result.exit_code == 2
    assert "--worker" in result.output


def test_the_command_is_the_nooa_coder_plugin():
    from nooa_cli.commands import discover_commands

    assert dict(discover_commands())["coder"] is command


@pytest.mark.parametrize("argv", [["nooa-coder", "--help"], ["nooa", "coder", "--help"]])
def test_the_console_scripts_run(argv):
    path = shutil.which(argv[0])
    assert path is not None, f"{argv[0]} is not installed; check [project.scripts]"
    result = subprocess.run([path, *argv[1:]], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Serve the NOOA coding agent over ACP" in result.stdout


def test_the_console_script_requires_a_model(monkeypatch):
    monkeypatch.delenv("NOOA_MODEL", raising=False)
    path = shutil.which("nooa-coder")
    assert path is not None
    result = subprocess.run([path], capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "--model" in result.stderr


def test_help_mentions_the_tee():
    assert "--tee" in _invoke(["--help"]).output


@pytest.mark.parametrize("module", ["nooa_coder.acp.cli", "nooa_coder.acp.tee"])
def test_the_entry_points_import_without_the_framework(module):
    """`nooa` loads every plugin command at startup; this one must stay light."""
    import sys

    result = subprocess.run(
        [sys.executable, "-c", f"import sys; import {module}; assert 'nooa' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
