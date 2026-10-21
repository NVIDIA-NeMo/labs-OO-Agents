# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache-off is an explicit profile control in headless, guided and edit paths."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))


def flags():
    return [
        "claude-sonnet-4-6",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "responses",
        "--api-key-env",
        "",
        "--reply-tokens",
        "1000",
    ]


def assert_entry(entry):
    assert entry["cache_breakpoint"] is None
    assert entry["max_tokens"] == 1000
    assert entry["store"] is False
    assert entry["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("stage", ["plan", "routing", "all"])
def test_stage_cache_off_json_and_no_plan_probe(monkeypatch, stage):
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("responses"))
        ),
    )
    result = CliRunner().invoke(command, [*flags(), "--stage", stage, "--no-cache"])
    report = json.loads(result.stdout)
    assert report["stage"] == stage
    assert set(report) == {
        "version",
        "stage",
        "ok",
        "data",
        "checks",
        "error",
        "run_context",
        "warnings",
        "diagnostic_prompt",
    }
    assert_entry(report["data"]["entry"])
    assert len(sent) == 0 if stage == "plan" else len(sent) >= 1
    if stage in {"plan", "routing"}:
        assert result.exit_code == 0, result.output
    for body in sent:
        assert "prompt_cache_options" not in body
        assert "cache_control" not in json.dumps(body)
    if sent:
        assert sent[0]["text"]["format"]["strict"] is True
        assert report["checks"]["routing"]["typed_result_validated"]
        assert report["checks"]["routing"]["answer_correct"]


@pytest.mark.parametrize("override", [False, True])
def test_stage_save_cache_control_and_default_preservation(tmp_path, override):
    entry = connect.plan(
        "local", "model", "responses", "https://api.test/v1", "", reply_tokens=1000
    ).entry
    if not override:
        entry["cache_breakpoint"] = None
    source = tmp_path / "plan.json"
    source.write_text(json.dumps({"data": {"alias": "local", "entry": entry}}))
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        ["--stage", "save", "--input", str(source), "--output", str(target)]
        + (["--no-cache"] if override else []),
    )
    assert result.exit_code == 0, result.output
    assert_entry(yaml.safe_load(target.read_text())["models"]["local"])
    assert_entry(json.loads(result.stdout)["data"]["entry"])


@pytest.mark.parametrize("yes", [False, True])
@pytest.mark.parametrize("editing", [False, True])
def test_guided_yes_edit_cache_off_no_metadata_or_probe(tmp_path, monkeypatch, yes, editing):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline setup must not request HTTP")

    mock_http(monkeypatch, forbidden)
    target = tmp_path / "models.yaml"
    args = [
        *flags(),
        "--no-cache",
        "--no-metadata",
        "--no-probe",
        "--media-inputs",
        "",
        "--output",
        str(target),
    ]
    if editing:
        entry = connect.plan(
            "local", "claude-sonnet-4-6", "responses", "https://api.test/v1", "", reply_tokens=1000
        ).entry
        entry.update(cache_breakpoint="openai", extra_body={"custom": 123}, temperature=0.2)
        target.write_text(yaml.safe_dump({"models": {"local": entry}}))
        monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [target])
        args = [
            "--edit-model",
            "local",
            "--no-cache",
            "--no-metadata",
            "--no-probe",
            "--output",
            str(target),
            "--media-inputs",
            "",
        ]
    result = CliRunner().invoke(command, args + (["--yes"] if yes else []), input="save\ny\ny\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(target.read_text())["models"]["local"]
    assert_entry(entry)
    assert "Caching: Off" in " ".join(result.output.split())
    if editing:
        assert entry["extra_body"] == {"custom": 123} and entry["temperature"] == 0.2


@pytest.mark.parametrize("mode", ["plan", "wizard", "save"])
def test_decision_cache_option_is_usage_error(tmp_path, mode):
    target = tmp_path / "models.yaml"
    args = [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/decisions",
        "--api-style",
        "openai",
        "--no-cache",
    ]
    if mode == "plan":
        args += ["--stage", "plan"]
    elif mode == "wizard":
        args += ["--yes", "--no-probe", "--no-metadata", "--output", str(target)]
    else:
        source = tmp_path / "plan.json"
        source.write_text(
            json.dumps(
                {
                    "alias": "local",
                    "entry": connect.plan(
                        "local", "model", "openai", "https://api.test/decisions", ""
                    ).entry,
                }
            )
        )
        args = ["--stage", "save", "--input", str(source), "--output", str(target), "--no-cache"]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 2, result.output
    assert "caching" in result.output.lower() and "decision" in result.output.lower()
    assert not target.exists()


@pytest.mark.parametrize("bad", [None, [], "bad"])
def test_bad_save_provenance_is_usage_error(tmp_path, bad):
    source = tmp_path / "plan.json"
    entry = connect.plan("local", "model", "responses", "https://api.test/v1", "").entry
    entry["provenance"] = bad
    source.write_text(json.dumps({"alias": "local", "entry": entry}))
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target), "--no-cache"]
    )
    assert result.exit_code == 2, result.output
    assert "provenance must be a mapping" in result.output
    assert not target.exists()


@pytest.mark.parametrize("stage", ["discover", "model-info", "catalogue", "interfaces"])
def test_metadata_interface_stages_reject_profile_control(stage):
    result = CliRunner().invoke(command, [*flags(), "--stage", stage, "--no-cache"])
    assert result.exit_code == 2, result.output
    assert "profile control" in result.output


@pytest.mark.parametrize("yes", [False, True])
def test_default_edit_preserves_disabled_cache_and_retention_with_native_probe(
    tmp_path, monkeypatch, yes
):
    target = tmp_path / "models.yaml"
    entry = connect.plan(
        "local",
        "claude-sonnet-4-6",
        "responses",
        "https://api.test/v1",
        "",
        reply_tokens=1000,
        cache_enabled=False,
    ).entry
    entry.update(store=True, include=[], temperature=0.2, extra_body={"custom": 123})
    target.write_text(yaml.safe_dump({"models": {"local": entry}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [target])
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("responses"))
        ),
    )
    args = [
        "--edit-model",
        "local",
        "--no-metadata",
        "--probe",
        "minimal",
        "--media-inputs",
        "",
        "--output",
        str(target),
    ]
    result = CliRunner().invoke(command, args + (["--yes"] if yes else []), input="test\ny\ny\n")
    assert result.exit_code == 0, result.output
    actual = yaml.safe_load(target.read_text())["models"]["local"]
    for key in ("cache_breakpoint", "max_tokens", "store", "include", "temperature", "extra_body"):
        assert actual[key] == entry[key]
    assert len(sent) == 1
    body = sent[0]
    assert "prompt_cache_options" not in body and "prompt_cache_breakpoint" not in json.dumps(body)
    # LiteLLM omits an empty include list on the wire; the saved explicit opt-out remains [].
    assert body["store"] is True and not body.get("include") and body["max_output_tokens"] == 1000
    assert body["text"]["format"]["strict"] is True


def test_stage_save_rejects_conflicting_cache_options_without_write(tmp_path):
    entry = connect.plan("local", "model", "responses", "https://api.test/v1", "").entry
    entry["extra_body"] = {"prompt_cache_options": {"mode": "explicit"}, "custom": 123}
    source = tmp_path / "plan.json"
    source.write_text(json.dumps({"alias": "local", "entry": entry}))
    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target), "--no-cache"]
    )
    assert result.exit_code == 2, result.output
    assert "conflicts with explicit cache options/markers" in result.output
    assert not target.exists()
