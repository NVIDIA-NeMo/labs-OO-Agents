# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero-network plan/save/edit plus native typed CLI wire evidence."""

import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands._connect_wizard import structured_output_rows
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.config import ModelConfig
from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._records import text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))


def flags(style="chat"):
    return [
        "azure/anthropic/claude-opus-5-5",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        style,
        "--api-key-env",
        "",
        "--reply-tokens",
        "8192",
    ]


def plan(**kwargs):
    return connect.plan(
        "local",
        "azure/anthropic/claude-opus-5-5",
        "chat",
        "https://api.test/v1",
        "",
        reply_tokens=8192,
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["none", "auto", "anthropic"])
def test_zero_network_stage_plan_profile_and_no_cache(monkeypatch, mode):
    def forbidden(request):
        raise AssertionError("plan must not send HTTP")

    mock_http(monkeypatch, forbidden)
    result = CliRunner().invoke(
        command, [*flags(), "--stage", "plan", "--structured-output-mode", mode, "--no-cache"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    entry = report["data"]["entry"]
    assert entry["structured_output_mode"] == (None if mode == "none" else mode)
    assert entry["max_tokens"] == 8192 and entry["cache_breakpoint"] is None
    assert entry["api_style"] == "chat"
    assert not report["checks"]
    for probe in report["data"]["probes"]:
        assert "structured_output_mode" not in probe["body"]


@pytest.mark.parametrize("mode", [None, "none", "auto", "anthropic"])
def test_save_yaml_reload_preserves_cap_and_explicit_profile(tmp_path, mode):
    proposal = plan(structured_output_mode="anthropic", cache_enabled=False)
    original = deepcopy(proposal.entry)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps({"data": {"alias": "local", "entry": proposal.entry}}))
    target = tmp_path / "models.yaml"
    args = ["--stage", "save", "--input", str(source), "--output", str(target)]
    if mode is not None:
        args.extend(["--structured-output-mode", mode])
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(target.read_text())["models"]["local"]
    expected = None if mode == "none" else mode or "anthropic"
    assert entry["structured_output_mode"] == expected
    assert entry["max_tokens"] == 8192 and entry["cache_breakpoint"] is None
    assert "provenance" not in entry
    typed = ModelConfig.from_registry("local", entry)
    assert typed.structured_output_mode == expected
    registry.reload_registry(target)
    c = registry.get_llm_client("local", api_key="fake")
    try:
        assert c.structured_output_mode == expected
        assert c.get_context_limits().reserved_output_tokens == 8192
        assert "structured_output_mode" not in c.config
    finally:
        c.close()
        registry.reload_registry()
    assert proposal.entry == original


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("mode", [None, "none", "anthropic"])
def test_guided_no_probe_create_edit_preserves_or_clears_profile(
    tmp_path, monkeypatch, editing, mode
):
    def forbidden(request):
        raise AssertionError("offline wizard must not send HTTP")

    mock_http(monkeypatch, forbidden)
    target = tmp_path / "models.yaml"
    args = [
        *flags(),
        "--no-probe",
        "--no-metadata",
        "--media-inputs",
        "",
        "--no-cache",
        "--yes",
        "--output",
        str(target),
    ]
    if editing:
        entry = plan(structured_output_mode="anthropic").entry
        entry.update(extra_body={"custom": 123}, temperature=0.2)
        target.write_text(yaml.safe_dump({"models": {"local": entry}}))
        monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [target])
        args = [
            "--edit-model",
            "local",
            "--no-probe",
            "--no-metadata",
            "--no-cache",
            "--yes",
            "--output",
            str(target),
        ]
    if mode is not None:
        args.extend(["--structured-output-mode", mode])
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(target.read_text())["models"]["local"]
    expected = "anthropic" if editing and mode is None else None if mode == "none" else mode
    if expected is None and mode is None:
        assert "structured_output_mode" not in saved
    else:
        assert saved["structured_output_mode"] == expected
        assert "Structured output" in result.output
        assert f"{expected or 'none'} (declared)" in result.output
    assert saved["max_tokens"] == 8192 and saved["cache_breakpoint"] is None
    if editing:
        assert saved["extra_body"]["custom"] == 123 and saved["temperature"] == 0.2


@pytest.mark.parametrize("style", ["responses", "systemone", "openai"])
def test_native_cli_rejects_foreign_carriers_before_any_io(monkeypatch, style):
    def forbidden(request):
        raise AssertionError("invalid route must not send HTTP")

    mock_http(monkeypatch, forbidden)
    result = CliRunner().invoke(
        command, [*flags(style), "--stage", "plan", "--structured-output-mode", "anthropic"]
    )
    assert result.exit_code != 0
    assert "requires Completion" in result.output


@pytest.mark.parametrize("stage", ["discover", "catalogue", "model-info", "interfaces", "decision"])
def test_native_profile_not_applied_to_discovery(stage):
    result = CliRunner().invoke(
        command, [*flags(), "--stage", stage, "--structured-output-mode", "anthropic"]
    )
    assert result.exit_code != 0 and "profile control" in result.output


def test_same_route_inheritance_and_profile_change_invalidates_all_evidence():
    proposal = plan(structured_output_mode="anthropic")
    entry = proposal.entry
    entry["provenance"].update(
        probes={"routing": {"outcome": "accepted", "structured_output_mode": "anthropic"}},
        session_checks={"outcome": "accepted"},
        requests_accepted=["routing"],
        reasoning_observed=["high"],
    )
    original = deepcopy(entry)
    assert plan(existing_entry=entry).entry["structured_output_mode"] == "anthropic"
    changed = connect.configure_entry(entry, structured_output_mode=None)
    assert changed["structured_output_mode"] is None
    assert changed["provenance"]["probes"] == {}
    assert (
        not {"session_checks", "requests_accepted", "reasoning_observed"}
        & changed["provenance"].keys()
    )
    assert entry == original
    # Direct edits/removal and refresh cannot recycle an old request body.
    edited = deepcopy(entry)
    edited.pop("structured_output_mode")
    refreshed = connect.refresh_plan(replace(proposal, entry=edited))
    assert refreshed.entry["provenance"]["probes"] == {}


@pytest.mark.parametrize(
    "replacement",
    [
        {"model": "other"},
        {"api_base": "https://other.test/v1"},
        {"api_style": "anthropic"},
        {"api_key_env": "OTHER_KEY"},
    ],
)
def test_connect_route_replacement_does_not_inherit_native(replacement):
    original = plan(structured_output_mode="anthropic").entry
    params = {
        "alias": "local",
        "model": "azure/anthropic/claude-opus-5-5",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "api_key_env": "",
        "reply_tokens": 8192,
        "existing_entry": original,
    }
    params.update(replacement)
    proposal = connect.plan(**params)
    assert "structured_output_mode" not in proposal.entry


@pytest.mark.parametrize("location", ["extra_body", "reasoning_levels"])
def test_framework_mode_never_in_provider_settings(location):
    entry = plan().entry
    entry[location] = (
        {"structured_output_mode": "anthropic"}
        if location == "extra_body"
        else {"high": {"structured_output_mode": "anthropic"}}
    )
    with pytest.raises(ValueError, match="connection"):
        connect.configure_entry(entry)


def test_cli_actual_native_predict_wire_and_declared_summary(monkeypatch):
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [*flags(), "--stage", "routing", "--structured-output-mode", "anthropic", "--no-cache"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert text_baseline_passed(report["checks"]["routing"])
    assert len(sent) == 1
    assert sent[0]["model"] == "azure/anthropic/claude-opus-5-5"
    assert sent[0]["max_tokens"] == 8192
    assert sent[0]["output_config"]["format"]["schema"]["properties"]["value"]["type"] == "integer"
    assert (
        not {"response_format", "tools", "tool_choice", "structured_output_mode"} & sent[0].keys()
    )
    assert report["checks"]["routing"]["structured_output_schema_observed"]
    rows = dict(structured_output_rows(report["data"]["entry"]))
    assert rows["Structured output"] == "anthropic (declared)"
    assert rows["Native carrier"] == "output_config.format observed"
