# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Connect CLI stays lazy and preserves its scripted interface."""

import json
import subprocess
import sys

from click.testing import CliRunner
from nooa_cli.commands.connect import command


def test_connect_entrypoint_does_not_import_framework():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import nooa_cli.commands.connect; assert 'nooa' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_connect_help_lists_options():
    result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--edit-model" in result.output
    assert "--stage" in result.output


def test_connect_plan_uses_library_without_http():
    result = CliRunner().invoke(
        command,
        [
            "example",
            "--stage",
            "plan",
            "--api-style",
            "chat",
            "--endpoint",
            "https://models.example/v1",
            "--api-key-env",
            "",
            "--as",
            "local",
        ],
    )
    assert result.exit_code == 0, result.output
    # Stage JSON is the stdout contract; result.output also holds stderr notices.
    report = json.loads(result.stdout)
    assert report["stage"] == "plan"
    assert report["ok"] is True


def test_connect_usage_error_preserves_exit_code():
    result = CliRunner().invoke(command, ["--stage", "not-a-stage"])
    assert result.exit_code == 2


ROUTING_STAGE_ARGS = [
    "example",
    "--stage",
    "routing",
    "--api-style",
    "chat",
    "--endpoint",
    "https://models.example/v1",
    "--api-key-env",
    "",
    "--as",
    "local",
]


def _deny_http(monkeypatch):
    import httpx

    from tests.unifiedllm.connect.connect_http import mock_http

    mock_http(
        monkeypatch, lambda request: httpx.Response(401, json={"error": {"message": "denied"}})
    )


def test_connect_failed_stage_preserves_exit_code(monkeypatch):
    _deny_http(monkeypatch)
    result = CliRunner().invoke(command, ROUTING_STAGE_ARGS)
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["ok"] is False


def test_connect_stage_json_survives_stderr_tracing_notice(monkeypatch):
    """An unreachable OTLP_ENDPOINT prints a stderr notice mid-stage; stdout stays JSON.

    Model checks auto-enable tracing, which probes OTLP_ENDPOINT (2 s timeout) and
    prints "... is not reachable — tracing disabled." to stderr when the probe
    fails. CliRunner's ``output`` mixes that line into the stage JSON, so a slow
    probe under load made the test above flaky when OTLP_ENDPOINT was set (e.g.
    loaded from a parent ``.env``). Force the probe to fail here.
    """
    import nooa.agent
    import nooa.tracing as tracing

    _deny_http(monkeypatch)
    monkeypatch.setenv("OTLP_ENDPOINT", "http://127.0.0.1:9/v1/traces")
    monkeypatch.setattr(nooa.agent, "_auto_tracing_attempted", False)
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "_probe_failed", False)
    monkeypatch.setattr(tracing, "resolve_otlp_endpoint", lambda *args, **kwargs: None)

    result = CliRunner().invoke(command, ROUTING_STAGE_ARGS)

    assert result.exit_code == 1, result.output
    assert "is not reachable" in result.stderr
    assert json.loads(result.stdout)["ok"] is False
