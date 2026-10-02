# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport selection is explicit and evidence cannot cross the SDK boundary."""

import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa.unifiedllm import CompletionClient, ResponsesClient, connect, registry
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    from nooa import llm_config

    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)


def options(style):
    return [
        "vendor/model",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        style,
        "--api-key-env",
        "",
        "--as",
        "local",
        "--max-tokens",
        "1024",
    ]


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("litellm", [False, True])
def test_cli_staged_check_save_load_real_sdk_http(tmp_path, monkeypatch, style, litellm):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    flag = ["--litellm"] if litellm else []
    result = CliRunner().invoke(command, [*options(style), "--stage", "routing", *flag])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    entry = report["data"]["entry"]
    expected = "litellm" if litellm else "direct"
    assert entry["direct"] is (not litellm)
    assert entry["transport"] == expected
    record = report["checks"]["routing"]
    assert record["transport"] == expected
    assert record["settings_sent"] is True
    assert record["wire_evidence"]["request_count"] == 1
    assert record["wire_evidence"]["model"] == "vendor/model"
    assert record["wire_evidence"]["api_style"] == style
    assert (
        "LiteLLM callbacks unavailable" in record["wire_evidence"]["telemetry"]
        if not litellm
        else True
    )
    assert len(sent) == 1
    body = json.loads(sent[0].content)
    assert body["model"] == "vendor/model"
    assert "direct" not in body and "transport" not in body
    if not litellm:
        assert sent[0].headers["x-stainless-lang"] == "python"
    source, target = tmp_path / "result.json", tmp_path / "models.yaml"
    source.write_text(result.stdout)
    # No flag here: preserve the transport actually checked.
    saved = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert saved.exit_code == 0, saved.output
    persisted = yaml.safe_load(target.read_text())["models"]["local"]
    assert persisted["direct"] is (not litellm)
    assert persisted["transport"] == expected
    assert "provenance" not in persisted
    registry.reload_registry(target)
    client = registry.get_llm_client("local", api_key="test-key")
    assert client.direct is (not litellm)
    client.close()


@pytest.mark.parametrize("stage", ["plan", "interfaces"])
@pytest.mark.parametrize("litellm", [False, True])
def test_plan_and_interface_selection_propagates(monkeypatch, stage, litellm):
    def handle(request):
        style = (
            "chat"
            if request.url.path.endswith("chat/completions")
            else ("responses" if request.url.path.endswith("responses") else "anthropic")
        )
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command, [*options("chat"), "--stage", stage, *(["--litellm"] if litellm else [])]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    entries = [data["entry"]] if stage == "plan" else [r["entry"] for r in data["results"].values()]
    assert len(entries) == (1 if stage == "plan" else 3)
    assert all(e["direct"] is (not litellm) for e in entries)
    if stage == "interfaces":
        assert all(
            e["provenance"]["probes"]["routing"]["transport"]
            == ("litellm" if litellm else "direct")
            for e in entries
        )


@pytest.mark.parametrize("tested", [False, True])
def test_stage_save_transport_change_requires_new_evidence(tmp_path, tested):
    entry = connect.plan("local", "model", "chat", "https://api.test/v1", "").entry
    if tested:
        entry["provenance"]["probes"]["routing"] = {"outcome": "accepted", "transport": "direct"}
    source, target = tmp_path / "input.json", tmp_path / "models.yaml"
    source.write_text(json.dumps({"alias": "local", "entry": entry}))
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target), "--litellm"]
    )
    assert result.exit_code == (2 if tested else 0), result.output
    if tested:
        assert "Transport differs from tested input" in result.stdout
        assert not target.exists()
    else:
        assert yaml.safe_load(target.read_text())["models"]["local"]["direct"] is False


def test_old_stage_input_preserves_actual_litellm_transport(tmp_path):
    source, target = tmp_path / "input.json", tmp_path / "models.yaml"
    # Old transport metadata was only a preference: actual old dispatch was LiteLLM.
    source.write_text(
        json.dumps(
            {
                "alias": "local",
                "entry": {
                    "model_name": "openai/model",
                    "transport": "direct",
                    "provenance": {
                        "probes": {"routing": {"outcome": "accepted", "transport": "litellm"}}
                    },
                },
            }
        )
    )
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(target.read_text())["models"]["local"]
    assert saved["direct"] is False
    assert saved["transport"] == "litellm"


@pytest.mark.asyncio
async def test_transport_switch_invalidates_reused_checks(monkeypatch):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    plan = connect.plan("local", "model", "chat", "https://api.test/v1", "", direct=False)
    checked = await connect.run(plan, approved="minimal", api_key="key")
    preserved = connect.plan(
        "local", "model", "chat", "https://api.test/v1", "", existing_entry=checked.entry
    )
    assert preserved.entry["direct"] is False
    await connect.run(preserved, approved="minimal", api_key="key")
    assert len(sent) == 1
    changed = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        existing_entry=checked.entry,
        direct=True,
    )
    assert changed.entry["provenance"]["probes"] == {}
    await connect.run(changed, approved="minimal", api_key="key")
    assert len(sent) == 2
    # Even manually carrying the old record cannot bypass the runtime identity check.
    stale = replace(
        changed, entry=changed.entry | {"provenance": deepcopy(checked.entry["provenance"])}
    )
    await connect.run(stale, approved="minimal", api_key="key")
    assert len(sent) == 3
    converted = connect.configure_entry(checked.entry, direct=True)
    assert converted["direct"] is True
    assert not converted["provenance"].get("probes")
    assert checked.entry["provenance"]["probes"]["routing"]["transport"] == "litellm"


@pytest.mark.parametrize("stage", ["discover", "catalogue"])
def test_metadata_stages_do_not_leak_inference_selection(monkeypatch, stage):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "model"}]}))
    results = []
    for flag in ([], ["--litellm"]):
        result = CliRunner().invoke(command, [*options("chat"), "--stage", stage, *flag])
        assert result.exit_code == 0, result.output
        results.append(json.loads(result.stdout))
    assert results[0] == results[1]
    assert "direct" not in results[0]["data"]


def test_connect_does_not_change_ordinary_defaults():
    assert connect.configure_entry({"model_name": "openai/model"})["direct"] is True
    for cls in (CompletionClient, ResponsesClient):
        client = cls("openai/model")
        assert client.direct is False
        client.close()
    client = registry.client_from_config(
        "old", {"model_name": "openai/model", "transport": "direct"}
    )
    assert client.direct is False
    client.close()


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("status", [401, 200])
@pytest.mark.asyncio
async def test_failure_reports_observed_transport_without_paid_calls(monkeypatch, direct, status):
    mock_http(monkeypatch, lambda request: httpx.Response(status, json={"malformed": True}))
    plan = connect.plan("local", "model", "chat", "https://api.test/v1", "", direct=direct)
    result = await connect.run(plan, approved="minimal", api_key="key")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["outcome"] != "accepted"
    assert record["transport"] == ("direct" if direct else "litellm")
    assert record["wire_evidence"]["request_count"] == 1
    assert record["status_code"] == status
    assert "wire_evidence" in connect.public_record(record)


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("litellm", [False, True])
@pytest.mark.parametrize("edit", [False, True])
def test_full_wizard_configured_probe_and_edit_override(
    tmp_path, monkeypatch, style, litellm, edit
):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    target = tmp_path / "models.yaml"
    if edit:
        entry = connect.plan(
            "local", "vendor/model", style, "https://api.test/v1", "", direct=litellm
        ).entry
        entry["provenance"]["session_checks"] = {"session": {"outcome": "accepted"}}
        target.write_text(yaml.safe_dump({"models": {"local": entry}}))
        argv = ["--edit-model", "local"]
    else:
        argv = [*options(style), "--no-catalogue"]
    result = CliRunner().invoke(
        command,
        [
            *argv,
            "--probe",
            "minimal",
            "--yes",
            "--output",
            str(target),
            *(["--litellm"] if litellm else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 1
    saved = yaml.safe_load(target.read_text())["models"]["local"]
    assert saved["direct"] is (not litellm)
    assert saved["transport"] == ("litellm" if litellm else "direct")
    assert "provenance" not in saved
    assert json.loads(sent[0].content)["model"] == "vendor/model"
    if not litellm:
        assert sent[0].headers["x-stainless-lang"] == "python"


def test_diagnostic_rerun_preserves_litellm_and_ignores_environment_selector(monkeypatch):
    import shlex

    from nooa_cli.commands._connect_registry import diagnostic_context

    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "direct")
    context = diagnostic_context(
        model="model",
        endpoint="https://api.test/v1",
        stage="interfaces",
        remaining=2136,
        direct=False,
    )
    assert "--litellm" in shlex.split(context["rerun_command"])
    assert context["requested_transport"] == "litellm"
    assert context["transport_override"] is None


@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_invalid_stored_direct_boolean_is_rejected(value):
    with pytest.raises(ValueError, match="direct must be a boolean"):
        connect.configure_entry({"model_name": "openai/model", "direct": value})


def test_direct_is_constructor_only_in_connect_levels():
    with pytest.raises(ValueError, match="cannot set routing"):
        connect.plan(
            "local",
            "model",
            "chat",
            "https://api.test/v1",
            "",
            reasoning_levels={"high": {"direct": True}},
        )


@pytest.mark.parametrize(
    "wire_model", ["claude-sonnet-4-6", "vendor/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"]
)
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.asyncio
async def test_connect_model_prefix_maps_to_selected_sdk(monkeypatch, wire_model, style):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    proposal = connect.plan("local", wire_model, style, "https://api.test/v1", "")
    client = registry.client_from_config("local", proposal.entry, api_key="test-key")
    assert client.direct is True
    assert client._http.api_style == style
    try:
        await client.acall([{"role": "user", "content": "Hello"}])
    finally:
        await client.aclose()
    assert len(sent) == 1
    assert json.loads(sent[0].content)["model"] == wire_model
    assert (
        sent[0].url.path
        == {
            "chat": "/v1/chat/completions",
            "responses": "/v1/responses",
            "anthropic": "/v1/messages",
        }[style]
    )
    assert sent[0].headers["x-stainless-lang"] == "python"


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("effort", ["high", "none"])
@pytest.mark.asyncio
async def test_live_reasoning_progress_keeps_observed_transport(monkeypatch, direct, effort):
    from nooa_cli.commands import _connect_wizard as wizard

    updates = []

    class Progress:
        def update(self, name, outcome, *, missing_reasoning):
            updates.append((name, missing_reasoning))

        def finish(self, *, summary):
            pass

    monkeypatch.setattr(wizard.view, "CheckProgress", Progress)
    proposal = connect.plan("local", "model", "chat", "https://api.test/v1", "", direct=direct)
    result = connect.ConnectResult(proposal.alias, proposal.entry)

    async def events():
        yield connect.ProbeUpdate(
            "level:custom",
            {
                "outcome": "accepted",
                "reasoning_observed": False,
                "transport": "direct" if direct else "litellm",
            },
        )
        yield result

    assert (
        await wizard.display_checks(
            events(), reasoning_levels={"custom": {"reasoning_effort": effort}}
        )
        is result
    )
    assert updates == [("level:custom", effort == "high")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("requests_accepted", ["routing"]),
        ("reasoning_observed", ["level:high"]),
        ("encrypted_reasoning", {"source": "connect", "outcome": "rejected"}),
    ],
)
def test_stage_save_transport_guard_covers_aggregate_evidence(tmp_path, field, value):
    entry = connect.plan("local", "model", "chat", "https://api.test/v1", "").entry
    entry["provenance"] = {field: value}
    source, target = tmp_path / "input.json", tmp_path / "models.yaml"
    source.write_text(json.dumps({"alias": "local", "entry": entry}))
    result = CliRunner().invoke(
        command,
        [
            "--stage",
            "save",
            "--input",
            str(source),
            "--output",
            str(target),
            "--litellm",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "Transport differs from tested input" in result.output
    assert not target.exists()
