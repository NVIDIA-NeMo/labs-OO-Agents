# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--save-checks`` keeps check results in the saved entry only when given (offline)."""

import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm.registry import load_model_entries
from tests.unifiedllm.connect.connect_http import mock_http

SECRET = "sk-save-checks-private-value"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("SAVE_CHECKS_KEY", SECRET)
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(prompts, "choose_media_mode", lambda **kw: "manual")
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(): list(current))
    mock_http(monkeypatch, lambda request: pytest.fail("offline setup made an HTTP request"))


def wizard(path, *extra, answers):
    return CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "SAVE_CHECKS_KEY",
            "--max-tokens",
            "1000",
            "--no-catalogue",
            "--no-probe",
            "--media-inputs",
            "",
            "--output",
            str(path),
            *extra,
        ],
        input=answers,
    )


def saved(path):
    return yaml.safe_load(path.read_text())["models"]["local"]


@pytest.mark.parametrize("flag", [[], ["--save-checks"]])
@pytest.mark.parametrize("yes", [[], ["--yes"]])
def test_wizard_saves_check_results_only_with_the_flag(tmp_path, flag, yes):
    path = tmp_path / "models.yaml"
    answers = "262144\n262144\n" if yes else "262144\n262144\nsave\ny\n"
    result = wizard(path, *yes, *flag, answers=answers)
    assert result.exit_code == 0, result.output
    # The wizard never asks about check results; only the option saves them.
    assert "check results" not in result.output
    entry = saved(path)
    assert ("checks" in entry) is bool(flag)
    assert "provenance" not in entry
    assert SECRET not in path.read_text()
    if flag:
        assert entry["checks"]["probes"]["routing"]["outcome"] == "not_probed"
        # The registry loads the alias with its checks; the settings are unchanged.
        loaded = load_model_entries(path)["local"]
        assert {k: v for k, v in loaded.items() if k != "checks"} == {
            k: v for k, v in entry.items() if k != "checks"
        }


def stage_input(tmp_path):
    # The shape of a plan/check report entry: settings plus their provenance.
    entry = {
        "model_name": "openai/model",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "api_key_env": "SAVE_CHECKS_KEY",
        "max_tokens": 1000,
        "provenance": {
            "probes": {
                "routing": {
                    "outcome": "not_probed",
                    "request": {"messages": [{"role": "user", "content": "Reply with ok."}]},
                }
            }
        },
    }
    document = tmp_path / "plan.json"
    document.write_text(json.dumps({"alias": "local", "entry": entry}))
    return document


def save_stage(document, path, *extra):
    return CliRunner().invoke(
        command,
        ["--stage", "save", "--input", str(document), "--output", str(path), *extra],
    )


def test_stage_save_keeps_checks_only_with_the_flag(tmp_path):
    document = stage_input(tmp_path)
    path = tmp_path / "models.yaml"
    result = save_stage(document, path)
    assert result.exit_code == 0, result.output
    assert "checks" not in saved(path)
    assert "provenance" not in saved(path)

    result = save_stage(document, path, "--yes", "--save-checks")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    entry = saved(path)
    # The checks section is the provenance the report shows for the same entry.
    assert entry["checks"] == report["data"]["entry"]["provenance"]
    assert {k: v for k, v in entry.items() if k != "checks"} == {
        k: v for k, v in report["data"]["entry"].items() if k != "provenance"
    }
    assert SECRET not in path.read_text()

    # Saving again without the flag replaces the alias and drops the section.
    result = save_stage(document, path, "--yes")
    assert result.exit_code == 0, result.output
    assert "checks" not in saved(path)


def test_save_checks_needs_a_saving_stage(tmp_path):
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "local",
            "--stage",
            "plan",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--save-checks",
        ],
    )
    assert result.exit_code == 2
    assert "--save-checks requires --stage save" in result.output
