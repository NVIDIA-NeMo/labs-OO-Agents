# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media declarations are explicit configuration, never capability probes."""

import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts, _connect_wizard
from nooa_cli.commands.connect import command


@pytest.mark.parametrize(
    "value,expected", [(None, []), ("image,pdf", ["image", "pdf"]), ("", []), ("-", [])]
)
def test_no_probe_new_entry_saves_exact_declaration(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    path = tmp_path / "models.yaml"
    args = [
        "test",
        "--as",
        "local",
        "--endpoint",
        "https://example.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "",
        "--no-probe",
        "--no-catalogue",
        "--yes",
        "--output",
        str(path),
    ]
    if value is not None:
        args += ["--media-inputs", value]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == expected


@pytest.mark.parametrize("value", [None, "", "video"])
def test_edit_preserves_or_clears_explicitly(tmp_path, monkeypatch, value):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "local": {
                        "model_name": "openai/test",
                        "api_base": "https://example.test/v1",
                        "media_inputs": ["image"],
                        "max_tokens": 1000,
                        "custom_knob": 42,
                    }
                }
            }
        )
    )
    args = ["--edit-model", "local", "--no-probe", "--yes"]
    if value is not None:
        args += ["--media-inputs", value]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["media_inputs"] == (["image"] if value is None else ["video"] if value else [])
    assert entry["custom_knob"] == 42


def test_ui_reprompts_invalid_and_preserves_empty_choice(monkeypatch):
    answers = iter(["file", "image,pdf"])
    monkeypatch.setattr(_connect_prompts, "prompt", lambda *a, **k: next(answers))
    state = _connect_wizard.WizardState(editing={"media_inputs": ["audio"]}, yes=False)
    assert _connect_wizard.configure_media_inputs(state)
    assert state.media_inputs == ["image", "pdf"]


def test_stage_plan_declaration_and_invalid_no_write(tmp_path):
    args = [
        "test",
        "--stage",
        "plan",
        "--api-style",
        "chat",
        "--endpoint",
        "https://example.test/v1",
        "--media-inputs",
        "image,pdf",
    ]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["data"]["entry"]["media_inputs"] == ["image", "pdf"]
    result = CliRunner().invoke(command, args[:-1] + ["file"])
    assert result.exit_code != 0
    assert "media_inputs" in result.output
    assert list(tmp_path.iterdir()) == []


def test_explicit_edit_can_repair_invalid_old_declaration():
    state = _connect_wizard.WizardState(
        editing={"media_inputs": "image"}, media_inputs="", yes=True
    )
    assert _connect_wizard.configure_media_inputs(state)
    assert state.media_inputs == []


def test_interactive_edit_enter_retains_previous_declaration(monkeypatch):
    def prompt(*args, **kwargs):
        assert kwargs["default"] == "image,pdf"
        return kwargs["default"]

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    state = _connect_wizard.WizardState(editing={"media_inputs": ["image", "pdf"]}, yes=False)
    assert _connect_wizard.configure_media_inputs(state)
    assert state.media_inputs == ["image", "pdf"]
