# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual setup banner at Connect's lazy runtime boundary."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import click
import httpx
import pytest
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

import nooa.agent as agent_module
import nooa.tracing as tracing
from nooa import llm_config
from nooa.unifiedllm import connect
from tests.tracing.test_announcements import trace_setup as _trace_setup
from tests.unifiedllm.connect.connect_http import mock_http, response_body

trace_setup = _trace_setup
_default_exporters = tracing._default_exporters


def unreachable_endpoint(monkeypatch):
    monkeypatch.setattr(tracing, "_default_exporters", _default_exporters)
    monkeypatch.setenv("OTLP_ENDPOINT", "https://unreachable.test/v1/traces")


@pytest.fixture(autouse=True)
def local_runtime(monkeypatch):
    monkeypatch.setattr(agent_module, "_auto_tracing_attempted", False)
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(tracing, "resolve_otlp_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.mark.parametrize("unreachable", [False, True])
@pytest.mark.parametrize("stage", ["routing", "interfaces"])
def test_stage_json_contains_no_setup_banner(trace_setup, monkeypatch, stage, unreachable):
    if unreachable:
        unreachable_endpoint(monkeypatch)
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    if stage == "interfaces":
        # Interface detection currently uses untyped probes (no Agent). Exercise
        # setup at the real runtime boundary too, not a mocked quiet manager.
        original_probe = connect._run_probe

        async def probe(*args, **kwargs):
            agent_module._try_auto_enable_tracing()
            return await original_probe(*args, **kwargs)

        monkeypatch.setattr(connect, "_run_probe", probe)
    result = CliRunner().invoke(
        command,
        [
            "gpt-5.1",
            "--as",
            "candidate",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--stage",
            stage,
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["checks"]["routing" if stage == "routing" else "chat"]["outcome"] == "accepted"
    assert "OTel tracing enabled" not in result.output
    assert "internal-traces.test" not in result.output
    assert agent_module._auto_tracing_attempted is True
    assert tracing._enabled is not unreachable
    if unreachable:
        assert tracing._probe_failed is True
        assert result.stderr == ""
        assert "unreachable.test" not in result.output
        trace_setup[2].assert_not_called()
    else:
        trace_setup[2].assert_called_once()
        trace_setup[3].assert_called_once_with(trace_setup[0])
    tracing.enable_tracing(exporters=[trace_setup[1]])


@pytest.mark.parametrize("unreachable", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_wizard_checking_row_stays_readable(trace_setup, monkeypatch, inline, unreachable):
    if unreachable:
        unreachable_endpoint(monkeypatch)
    original = view.CheckProgress.__init__

    def init(self):
        original(self)
        self.inline = inline

    monkeypatch.setattr(view.CheckProgress, "__init__", init)
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    proposal = connect.plan("candidate", "gpt-5.1", "chat", "https://api.test/v1", "")

    @click.command()
    def display():
        result = asyncio.run(wizard.show_checks(connect.run_steps(proposal, approved="all")))
        assert result.entry["provenance"]["probes"]["routing"]["outcome"] == "accepted"

    result = CliRunner().invoke(display)
    assert result.exit_code == 0, result.output
    assert "Connection — checking" in result.output
    assert "Connection: Passed" in result.output
    assert "checkingOTel" not in result.output
    assert "OTel tracing enabled" not in result.output
    assert "internal-traces.test" not in result.output
    assert tracing._enabled is not unreachable
    assert agent_module._auto_tracing_attempted
    if unreachable:
        assert tracing._probe_failed is True
        assert result.stderr == ""
        assert "unreachable.test" not in result.output
        trace_setup[2].assert_not_called()
    else:
        trace_setup[2].assert_called_once()
    assert tracing._announce_trace_target.get() is True


@pytest.mark.parametrize("fail", [False, True])
def test_connect_scope_restores_provider_policy_and_preserves_errors(trace_setup, fail):
    import litellm

    previous = litellm.suppress_debug_info

    @click.command()
    def display():
        with view.quiet_provider_messages():
            tracing.enable_tracing()
            click.echo("genuine progress")
            if fail:
                raise RuntimeError("genuine failure")

    result = CliRunner().invoke(display)
    assert result.output == "genuine progress\n"
    assert result.exit_code == (1 if fail else 0)
    if fail:
        assert isinstance(result.exception, RuntimeError)
        assert str(result.exception) == "genuine failure"
    assert litellm.suppress_debug_info == previous
    assert tracing._announce_trace_target.get() is True
    tracing.enable_tracing(exporters=[trace_setup[1]])


def test_first_initialization_in_fresh_process():
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            str(Path(__file__).resolve()),
            "-k",
            "stage_json or wizard_checking",
            "-q",
            "-o",
            "addopts=",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "8 passed" in result.stdout


@pytest.mark.parametrize("diagnostics", [False, True])
def test_command_diagnostics_do_not_reveal_tracing_target(trace_setup, monkeypatch, diagnostics):
    from nooa_cli.commands import _connect_prompts as prompts

    monkeypatch.setattr(prompts, "choose_reply_limit", lambda suggested, ceiling, **kw: suggested)
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(): list(current))
    monkeypatch.setattr(prompts, "choose_media_mode", lambda **kw: "manual")
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    result = CliRunner().invoke(
        command,
        [
            "gpt-5.1",
            "--as",
            "candidate",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--probe",
            "minimal",
            *(["--show-diagnostics"] if diagnostics else []),
        ],
        input="y\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "Connection — checking" in result.output
    assert "Connection: Passed" in result.output
    assert "OTel tracing enabled" not in result.output
    assert "internal-traces.test" not in result.output
    assert tracing._enabled and agent_module._auto_tracing_attempted
    trace_setup[2].assert_called_once()
