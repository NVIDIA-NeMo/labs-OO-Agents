# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reviewed configuration boundaries stay consistent across frontends."""

import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_registry, _connect_stages
from nooa_cli.commands.connect import command


def test_tombstones_do_not_reappear_in_editor_or_credential_suggestions(tmp_path, monkeypatch):
    from nooa import llm_config

    low, high = tmp_path / "low.yaml", tmp_path / "high.yaml"
    low.write_text(
        "models: {ghost: {model_name: openai/model, api_base: 'https://api.test/v1', api_key_env: TEST_KEY}}\n"
    )
    high.write_text("models: {ghost: null}\n")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [low, high])
    assert "ghost" not in _connect_registry.entries()
    assert (
        _connect_registry.credential_names(_connect_registry.entries(), "https://api.test/v1") == []
    )


def test_discovery_replay_accepts_its_own_v1_fallback(tmp_path):
    path = tmp_path / "discovery.json"
    models = [{"id": "model"}]
    path.write_text(json.dumps({"data": {"api_base": "https://api.test/v1", "models": models}}))
    assert _connect_stages.read_discovery(path, "https://api.test") == models
    with pytest.raises(ValueError, match="different endpoint"):
        _connect_stages.read_discovery(path, "https://other.test")


@pytest.mark.parametrize("nested", [False, True])
def test_stage_save_refuses_literal_credentials_without_echo(tmp_path, nested):
    source, target = tmp_path / "input.json", tmp_path / "registry.yaml"
    credential = {"api_key": "literal-private-test"}
    entry = {"model_name": "openai/model", **({"extra_body": credential} if nested else credential)}
    source.write_text(json.dumps({"alias": "model", "entry": entry}))
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert result.exit_code == 2, result.output
    assert "literal-private-test" not in result.output
    assert not target.exists()


def test_stage_save_can_populate_null_models(tmp_path):
    source, target = tmp_path / "input.json", tmp_path / "registry.yaml"
    target.write_text("models:\n")
    source.write_text(json.dumps({"alias": "model", "entry": {"model_name": "openai/model"}}))
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert "model" in yaml.safe_load(target.read_text())["models"]


def test_editor_can_add_a_level_supplied_by_levels_file(tmp_path, monkeypatch):
    from nooa_cli.commands._connect_wizard import WizardState, configure_checks

    levels = tmp_path / "levels.yaml"
    levels.write_text("ultra: {reasoning_effort: high}\n")
    state = WizardState(
        editing={"model_name": "openai/model", "api_key_env": "", "max_tokens": 8192},
        candidate={"id": "model", "reasoning": {"supported_efforts": ["ultra"]}},
        levels_file=str(levels),
        api_style="chat",
        model="model",
        endpoint="https://api.test/v1",
        api_key_env="",
        budget_tokens=131072,
        output_tokens=200,
        reasoning_output_tokens=4096,
        approval="none",
        yes=True,
    )
    assert configure_checks(state)
    assert state.proposal.entry["reasoning_levels"] == {"ultra": {"reasoning_effort": "high"}}
