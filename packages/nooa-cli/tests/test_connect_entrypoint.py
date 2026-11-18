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


def _deny_http(monkeypatch, stderr_notice=None):
    import httpx

    from tests.unifiedllm.connect.connect_http import mock_http

    # The SDK validates credentials before sending even a mocked request.
    monkeypatch.setenv("OPENAI_API_KEY", "test-connect-key")

    def deny(request):
        if stderr_notice is not None:
            print(stderr_notice, file=sys.stderr)
        return httpx.Response(401, json={"error": {"message": "denied"}})

    mock_http(monkeypatch, deny)


def test_connect_failed_stage_preserves_exit_code(monkeypatch):
    _deny_http(monkeypatch)
    result = CliRunner().invoke(command, ROUTING_STAGE_ARGS)
    assert result.exit_code == 1, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["data"]["entry"]["provenance"]["probes"]["routing"]["status_code"] == 401


def test_connect_stage_json_survives_stderr_tracing_notice(monkeypatch):
    """A tracing-style stderr notice must not corrupt the stage's JSON output.

    Agent construction intentionally suppresses tracing setup messages. Emit the
    notice during the mocked request to test CLI stream separation independently
    of tracing startup and its process-global state.
    """
    notice = "OTLP_ENDPOINT (http://127.0.0.1:9/v1/traces) is not reachable — tracing disabled."
    _deny_http(monkeypatch, stderr_notice=notice)

    result = CliRunner().invoke(command, ROUTING_STAGE_ARGS)

    assert result.exit_code == 1, result.output
    assert notice in result.stderr, result.output
    assert json.loads(result.stdout)["ok"] is False
