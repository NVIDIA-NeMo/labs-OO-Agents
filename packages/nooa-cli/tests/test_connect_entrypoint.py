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
    report = json.loads(result.output)
    assert report["stage"] == "plan"
    assert report["ok"] is True


def test_connect_usage_error_preserves_exit_code():
    result = CliRunner().invoke(command, ["--stage", "not-a-stage"])
    assert result.exit_code == 2


def test_connect_failed_stage_preserves_exit_code(monkeypatch):
    import httpx

    from tests.unifiedllm.connect.connect_http import mock_http

    mock_http(
        monkeypatch, lambda request: httpx.Response(401, json={"error": {"message": "denied"}})
    )
    result = CliRunner().invoke(
        command,
        [
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
        ],
    )
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["ok"] is False
