# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human, unattended wizard and imported stage paths share exact context bounds."""

import json
from copy import deepcopy

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect


@pytest.mark.parametrize("declared,effective", [(128000, 64000), (48000, 48000)])
@pytest.mark.parametrize("yes", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_wizard_explicit_context_bound_before_review(declared, effective, yes, exact):
    server = {
        "id": "model" if exact else "neighbour",
        "max_input_tokens": 64000,
        "max_output_tokens": 2000,
    }
    state = wizard.WizardState(
        model="model",
        endpoint="https://api.test/v1",
        api_style="chat",
        api_key_env="",
        selected_endpoint_model=server,
        context_window=declared,
        metadata_prepared=True,
        yes=yes,
        guided=False,
        output_tokens=200,
        reasoning_output_tokens=200,
        budget_tokens=100000,
        reply_tokens=1000,
        media_inputs=[],
    )
    before = deepcopy(server)
    assert wizard.configure_checks(state)
    expected = effective if exact else declared
    assert state.configured["context_window"] == expected
    assert state.configured["max_tokens"] == 1000
    evidence = state.configured["provenance"]["context_window"]
    assert evidence["declared"] == declared and evidence["effective"] == expected
    assert evidence["source"] == "user"
    assert server == before


@pytest.mark.parametrize("declared,effective", [(128000, 64000), (48000, 48000)])
@pytest.mark.parametrize("exact", [False, True])
def test_stage_plan_save_import_explicit_bound(tmp_path, declared, effective, exact):
    endpoint = "https://api.test/v1"
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps(
            {
                "api_base": endpoint,
                "models": [
                    {
                        "id": "model" if exact else "neighbour",
                        "max_input_tokens": 64000,
                        "max_output_tokens": 2000,
                    }
                ],
            }
        )
    )
    flags = [
        "model",
        "--as",
        "local",
        "--endpoint",
        endpoint,
        "--api-style",
        "chat",
        "--api-key-env",
        "",
        "--reply-tokens",
        "1000",
    ]
    runner = CliRunner()
    planned = runner.invoke(
        command,
        flags
        + ["--stage", "plan", "--discovery-file", str(listing), "--context-window", str(declared)],
    )
    assert planned.exit_code == 0, planned.output
    document = json.loads(planned.stdout)
    entry = document["data"]["entry"]
    expected = effective if exact else declared
    assert entry["context_window"] == expected and entry["max_tokens"] == 1000
    assert entry["provenance"]["context_window"]["declared"] == declared
    # An edited imported plan must not bypass its selected endpoint evidence.
    entry["context_window"] = declared
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(document))
    output = tmp_path / "models.yaml"
    saved = runner.invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(output), "--yes"]
    )
    assert saved.exit_code == 0, saved.output
    assert yaml.safe_load(output.read_text())["models"]["local"]["context_window"] == expected
    # Explicit save-time flags with imported exact listing use the same bound.
    override = runner.invoke(
        command,
        [
            "--stage",
            "save",
            "--input",
            str(source),
            "--output",
            str(output),
            "--yes",
            "--context-window",
            str(declared),
            "--discovery-file",
            str(listing),
        ],
    )
    assert override.exit_code == 0, override.output
    assert yaml.safe_load(output.read_text())["models"]["local"]["context_window"] == expected
    entry.pop("provenance", None)
    source.write_text(json.dumps(document))
    imported = runner.invoke(
        command,
        [
            "--stage",
            "save",
            "--input",
            str(source),
            "--output",
            str(output),
            "--yes",
            "--discovery-file",
            str(listing),
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert yaml.safe_load(output.read_text())["models"]["local"]["context_window"] == expected


def test_configure_import_never_restores_larger_old_declaration():
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        context_window=128000,
        reply_tokens=1000,
        endpoint_model={"id": "model", "max_input_tokens": 64000},
    )
    entry = deepcopy(proposal.entry)
    entry["context_window"] = 48000
    configured = connect.configure_entry(entry)
    assert configured["context_window"] == 48000
    assert configured["provenance"]["context_window"]["effective"] == 48000
    assert configured["provenance"]["context_window"]["declared"] == 48000


@pytest.mark.parametrize("declared,effective", [(128000, 64000), (48000, 48000)])
@pytest.mark.parametrize("yes", [False, True])
def test_actual_wizard_flag_save_uses_selected_listing(
    monkeypatch, tmp_path, declared, effective, yes
):
    from nooa import llm_config
    from tests.unifiedllm.connect.connect_http import mock_http

    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    mock_http(monkeypatch, lambda request: pytest.fail("Offline save must not use HTTP"))
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps(
            {
                "api_base": "https://api.test/v1",
                "models": [{"id": "model", "max_input_tokens": 64000, "max_output_tokens": 2000}],
            }
        )
    )
    output = tmp_path / "models.yaml"
    args = [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "",
        "--media-inputs",
        "",
        "--no-catalogue",
        "--no-probe",
        "--reply-tokens",
        "1000",
        "--context-window",
        str(declared),
        "--discovery-file",
        str(listing),
        "--output",
        str(output),
    ]
    if yes:
        args += ["--yes"]
    result = CliRunner().invoke(command, args, input="save\ny\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(output.read_text())["models"]["local"]
    assert entry["context_window"] == effective and entry["max_tokens"] == 1000
