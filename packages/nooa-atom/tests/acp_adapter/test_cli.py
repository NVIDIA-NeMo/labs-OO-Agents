# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The nooa-atom command line: model, agent, sessions directory, tee."""

import shutil
import subprocess

import click.testing
import pytest
from nooa_atom.acp.cli import command


@pytest.fixture
def served(monkeypatch):
    """Capture what the command would serve with, instead of serving."""
    monkeypatch.setenv("NOOA_MODEL", "m")
    captured: dict = {}
    monkeypatch.setattr("nooa_atom.acp.cli.run", lambda **kwargs: captured.update(kwargs))
    # The real one repoints this process's stdin and stdout.
    monkeypatch.setattr("nooa_atom.acp.cli.reserve_stdio_for_acp", lambda: (0, 1))
    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda *a, **k: None)
    return captured


@pytest.fixture
def requested(monkeypatch):
    calls: list = []

    def fake_client(name, workspace, **kwargs):
        calls.append((name, kwargs))
        return name

    monkeypatch.setattr("nooa_atom.workspace.models.workspace_llm_client", fake_client)
    return calls


def _invoke(args, **kwargs):
    return click.testing.CliRunner().invoke(command, args, **kwargs)


def test_without_a_model_the_first_configured_one_is_used(
    monkeypatch, served, requested, tmp_path
):
    monkeypatch.delenv("NOOA_MODEL", raising=False)
    assert _invoke([]).exit_code == 0
    assert served["model"] is None
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa" / "llm_config.yaml").write_text(
        "models:\n  first:\n    model_name: openai/a\n  second:\n    model_name: openai/b\n"
    )
    served["llm_factory"](None, tmp_path)
    assert requested[-1][0] == "first"


def test_the_model_comes_from_the_environment(monkeypatch, served, requested, tmp_path):
    monkeypatch.setenv("NOOA_MODEL", "env-model")
    assert _invoke([]).exit_code == 0
    assert served["model"] == "env-model"
    # The factory builds the alias a session asks for, else the default.
    served["llm_factory"](None, tmp_path)
    served["llm_factory"]("other", tmp_path)
    assert requested == [("env-model", {"client_type": None}), ("other", {"client_type": None})]


def test_client_type_and_the_nvidia_key_reach_the_client(monkeypatch, served, requested, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NOOA_MODEL", "nvidia_nim/some/model")
    result = _invoke(["--client-type", "responses"])
    assert result.exit_code == 0, result.output
    served["llm_factory"](None, tmp_path)
    served["llm_factory"]("openai/gpt", tmp_path)
    assert requested == [
        ("nvidia_nim/some/model", {"client_type": "responses", "api_key": "nvapi-test"}),
        ("openai/gpt", {"client_type": "responses"}),
    ]


def test_the_factory_reads_the_session_workspace_configuration(served, tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa" / "llm_config.yaml").write_text(
        "models:\n  mine:\n    model_name: openai/mine-model\n"
    )
    assert _invoke([]).exit_code == 0
    assert served["llm_factory"]("mine", tmp_path).model == "openai/mine-model"


def test_a_relative_agent_file_is_resolved_where_the_command_runs(served, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _invoke(["--agent", "agents/my_agent.py:MyAgent"]).exit_code == 0
    assert served["agent_spec"] == f"{tmp_path / 'agents/my_agent.py'}:MyAgent"
    assert _invoke(["--agent", "pkg.module:Agent"]).exit_code == 0
    assert served["agent_spec"] == "pkg.module:Agent"


def test_without_agent_the_workspace_setting_decides(served):
    assert _invoke([]).exit_code == 0
    assert served["agent_spec"] is None  # the workspace setting, else the Atom agent


def test_sessions_dir_and_tee_are_passed_on(served, tmp_path):
    result = _invoke(
        ["--sessions-dir", str(tmp_path / "s"), "--tee", str(tmp_path / "t.jsonl")]
    )
    assert result.exit_code == 0, result.output
    assert served["sessions_dir"] == tmp_path / "s"
    assert served["tee"] == tmp_path / "t.jsonl"


def test_the_command_is_the_nooa_atom_plugin():
    from nooa_cli.commands import discover_commands

    assert dict(discover_commands())["atom"] is command


def test_nooa_atom_runs():
    path = shutil.which("nooa")
    assert path is not None, "nooa is not installed; nooa-atom depends on nooa-cli"
    result = subprocess.run([path, "atom", "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Serve NOOA Atom over ACP" in result.stdout
