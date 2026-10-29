# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa connect on native Windows: saving works; POSIX-only failures stay readable."""

import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

import nooa._platform as platform
from nooa._platform import PosixOnlyFeatureError

ARGS = [
    "fake-model",
    "--provider",
    "custom",
    "--endpoint",
    "http://127.0.0.1:9/v1",
    "--api-style",
    "chat",
    "--api-key-env",
    "NOOA_TEST_FAKE_KEY",
    "--as",
    "fake",
    "--no-probe",
    "--no-catalogue",
    "--yes",
]


def test_connect_saves_on_native_windows(monkeypatch, tmp_path):
    # Native Windows takes the filelock branch of connect.write (no fcntl there).
    monkeypatch.setattr(platform, "is_native_windows", lambda: True)
    output = tmp_path / "llm_config.yaml"
    result = CliRunner().invoke(command, [*ARGS, "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "fake" in yaml.safe_load(output.read_text())["models"]


def test_wizard_reports_posix_only_save_cleanly(monkeypatch, tmp_path):
    def save_fails(state):
        raise PosixOnlyFeatureError("A POSIX-only step")

    for name in (
        "select_connection",
        "configure_media_inputs",
        "select_model",
        "check_interfaces",
        "configure_metadata",
        "configure_checks",
        "run_checks",
    ):
        monkeypatch.setattr(wizard, name, lambda state: True)
    monkeypatch.setattr(wizard, "save_model", save_fails)
    result = CliRunner().invoke(command, [*ARGS, "--output", str(tmp_path / "llm_config.yaml")])

    assert result.exit_code != 0
    assert "WSL 2" in result.output and "Traceback" not in result.output


def test_stage_json_keeps_the_wsl_hint():
    detail = view.local_failure(PosixOnlyFeatureError("A POSIX-only step"))
    assert "WSL 2" in detail and "credentials" not in detail
