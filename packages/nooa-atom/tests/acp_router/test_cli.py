# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-atom's roles: router by default, --http, --single-process, and the hidden worker options."""

import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from nooa_atom.acp import cli


@pytest.fixture
def served(monkeypatch):
    """What the command passes to run(), instead of serving."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "reserve_stdio_for_acp", lambda: (0, 1))
    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda *a, **k: None)
    return captured


@pytest.fixture
def roles(monkeypatch):
    """Which role run() chose, instead of serving."""
    calls: dict[str, Any] = {}

    def record(role: str):
        def fake(**kwargs: Any) -> None:
            calls[role] = kwargs

        return fake

    monkeypatch.setattr(cli, "reserve_stdio_for_acp", lambda: (0, 1))
    monkeypatch.setattr(cli, "_run_router", record("router"))
    monkeypatch.setattr(cli, "_run_worker", record("worker"))
    monkeypatch.setattr(cli, "_run_http", lambda options, **kwargs: calls.update(http=options))

    async def fake_serve(**kwargs: Any) -> None:
        calls["single"] = kwargs

    monkeypatch.setattr(cli, "_serve", fake_serve)
    return calls


def _factory(alias: str | None, workspace: Path) -> Any:
    return None


def test_help_describes_the_four_roles():
    result = CliRunner().invoke(cli.command, ["--help"])
    assert result.exit_code == 0
    assert "router (default)" in result.output
    assert "ws://HOST:PORT/acp" in result.output
    assert "--single-process" in result.output
    assert "worker" in result.output
    assert "--worker-fd" not in result.output.split("Options:")[1]  # hidden


def test_the_command_passes_the_role_options_to_run(served):
    assert CliRunner().invoke(cli.command, ["--model", "m"]).exit_code == 0
    assert (served["single_process"], served["worker_fd"], served["id_base"]) == (False, None, None)
    args = ["--model", "m", "--worker-fd", "7", "--id-base", str(3 << 32)]
    assert CliRunner().invoke(cli.command, args).exit_code == 0
    assert (served["worker_fd"], served["id_base"]) == (7, 3 << 32)
    assert CliRunner().invoke(cli.command, ["--model", "m", "--single-process"]).exit_code == 0
    assert served["single_process"] is True


@pytest.mark.parametrize(
    "args",
    [
        ["--worker-fd", "7"],
        ["--id-base", "1"],
        ["--single-process", "--worker-fd", "7", "--id-base", "1"],
    ],
)
def test_inconsistent_role_options_are_usage_errors(served, args):
    result = CliRunner().invoke(cli.command, ["--model", "m", *args])
    assert result.exit_code == 2


def test_http_needs_a_token_or_no_auth(served, monkeypatch):
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    result = CliRunner().invoke(cli.command, ["--model", "m", "--http"])
    assert result.exit_code == 2
    assert cli.TOKEN_ENV in result.output
    assert not served


def test_http_takes_the_token_out_of_the_environment(served, monkeypatch):
    """Workers and the cells they run inherit the environment; the token stays in the server."""
    monkeypatch.setenv(cli.TOKEN_ENV, "s3cret")
    args = ["--model", "m", "--http", "--port", "9001", "--allowed-origin", "https://a.example"]
    result = CliRunner().invoke(cli.command, args)
    assert result.exit_code == 0, result.output
    assert served["http"] == cli.HttpOptions(
        host="127.0.0.1", port=9001, token="s3cret", allowed_origins=("https://a.example",)
    )
    import os

    assert cli.TOKEN_ENV not in os.environ


def test_no_auth_is_only_for_loopback(served, monkeypatch):
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    result = CliRunner().invoke(cli.command, ["--model", "m", "--http", "--no-auth"])
    assert result.exit_code == 0, result.output
    assert served["http"].token is None
    served.clear()
    args = ["--model", "m", "--http", "--no-auth", "--host", "0.0.0.0"]
    result = CliRunner().invoke(cli.command, args)
    assert result.exit_code == 2
    assert "loopback" in result.output
    assert not served


@pytest.mark.parametrize("extra", [["--single-process"], ["--tee", "frames.jsonl"]])
def test_http_rejects_the_stdio_only_options(served, monkeypatch, extra):
    monkeypatch.setenv(cli.TOKEN_ENV, "t")
    result = CliRunner().invoke(cli.command, ["--model", "m", "--http", *extra])
    assert result.exit_code == 2


def test_a_worker_of_an_http_server_needs_no_token(served, monkeypatch):
    """Workers re-run the server's command line, --http included."""
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    args = ["--model", "m", "--http", "--worker-fd", "7", "--id-base", "1"]
    result = CliRunner().invoke(cli.command, args)
    assert result.exit_code == 0, result.output
    assert served["worker_fd"] == 7


def test_run_serves_http_when_asked(roles):
    options = cli.HttpOptions(host="127.0.0.1", port=1, token="t")
    cli.run(llm_factory=_factory, single_process=False, http=options)
    assert roles == {"http": options}


def test_a_factory_in_the_context_object_replaces_the_default(served):
    result = CliRunner().invoke(cli.command, ["--model", "m"], obj={"llm_factory": _factory})
    assert result.exit_code == 0, result.output
    assert served["llm_factory"] is _factory


def test_run_is_the_router_by_default_with_workers_from_the_same_command(roles, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fake_agent.py", "--blocking"])
    cli.run(llm_factory=_factory, model="fake")
    assert list(roles) == ["router"]


def test_run_reads_the_worker_options_from_argv(roles, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fake_agent.py", "--worker-fd", "9", "--id-base", "8"])
    cli.run(llm_factory=_factory, model="fake")
    assert roles["worker"]["fd"] == 9 and roles["worker"]["id_base"] == 8
    assert roles["worker"]["llm_factory"] is _factory


def test_run_reads_single_process_from_argv(roles, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fake_agent.py", "--single-process"])
    cli.run(llm_factory=_factory, model="fake")
    assert list(roles) == ["single"]


def test_explicit_role_arguments_win_over_argv(roles, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fake_agent.py", "--single-process"])
    cli.run(llm_factory=_factory, single_process=False)
    assert list(roles) == ["router"]


def test_the_cli_module_imports_nothing_heavy_at_load_time():
    """Importing the command must not import nooa: every nooa command would pay for it.

    Only module-level imports count; imports under ``if TYPE_CHECKING:``
    and inside functions run later or never.
    """
    import ast

    tree = ast.parse(Path(cli.__file__).read_text())
    loaded: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            loaded.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            loaded.append(node.module)
    heavy = [name for name in loaded if name.split(".")[0] in ("nooa", "nooa_atom", "acp")]
    assert heavy == []


def test_llm_config_summary_names_the_files_and_the_env_var(tmp_path, monkeypatch):
    """The server logs where a workspace's model configuration comes from."""
    from nooa_atom.acp.cli import llm_config_summary

    workspace = tmp_path / "workspace"
    (workspace / ".nooa").mkdir(parents=True)
    own = workspace / ".nooa" / "llm_config.yaml"
    own.write_text("models: {}\n")
    config = tmp_path / "llm_config.yaml"
    config.write_text("models: {}\n")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(config))
    summary = llm_config_summary(workspace)
    assert f"{own.resolve()}, {config.resolve()}" in summary
    assert f"NEMO_OO_LLM_CONFIG={config}" in summary

    monkeypatch.delenv("NEMO_OO_LLM_CONFIG")
    assert "NEMO_OO_LLM_CONFIG not set" in llm_config_summary(workspace)
