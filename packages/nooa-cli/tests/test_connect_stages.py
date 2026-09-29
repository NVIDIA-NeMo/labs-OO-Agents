# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent stage commands never prompt and share the real library/HTTP path."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body

BASE = [
    "gpt-5.1",
    "--endpoint",
    "https://api.test/v1",
    "--api-style",
    "chat",
    "--as",
    "local",
    "--api-key-env",
    "STAGE_TEST_KEY",
    "--max-tokens",
    "2048",
]


@pytest.mark.parametrize("stage", ["interfaces", "routing"])
@pytest.mark.parametrize("pasted", [False, True])
def test_full_stage_report_scrubs_key_accidentally_pasted_as_model(monkeypatch, stage, pasted):
    secret = "private-active-test-key"
    monkeypatch.setenv("STAGE_TEST_KEY", secret)
    mock_http(monkeypatch, lambda request: httpx.Response(401))
    options = [secret, *BASE[1:], "--stage", stage]
    result = CliRunner().invoke(
        command,
        options + (["--prompt-key"] if pasted else []),
        input=secret + "\n" if pasted else None,
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert secret not in result.output
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
    assert all("request" not in check for check in report["checks"].values())


def test_explicit_prompt_key_keeps_stage_json_clean(monkeypatch):
    monkeypatch.delenv("STAGE_TEST_KEY", raising=False)
    sent = []

    def handle(request):
        sent.append(request)
        assert request.headers["authorization"] == "Bearer pasted-test-key"
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command, [*BASE, "--stage", "routing", "--prompt-key"], input="pasted-test-key\n"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    assert len(sent) == 1
    assert "pasted-test-key" not in result.output


@pytest.mark.parametrize(
    "stage,count", [("routing", 1), ("tools", 1), ("reasoning", 2), ("session", 3), ("all", 7)]
)
def test_each_check_stage_is_independent_json(monkeypatch, tmp_path, stage, count):
    monkeypatch.setenv("STAGE_TEST_KEY", "test-secret")
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        data = response_body("chat")
        if body["messages"][0]["content"] == connect.REASONING_CHECK_PROMPT:
            data["choices"][0]["message"]["content"] = "B G D A C E F H"
        data["choices"][0]["message"]["reasoning_content"] = "private reasoning"
        data["usage"]["prompt_tokens_details"] = {"cached_tokens": 15}
        if any(t["function"]["name"] == "probe_tool" for t in body.get("tools", [])):
            data["choices"][0]["finish_reason"] = "tool_calls"
            data["choices"][0]["message"]["tool_calls"] = [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "probe_tool", "arguments": '{"value":"ok"}'},
                }
            ]
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    path = tmp_path / "levels.yaml"
    path.write_text("high: {reasoning_effort: high}\nlow: {reasoning_effort: low}\n")
    args = [*BASE, "--stage", stage, "--levels-file", str(path)]
    result = CliRunner().invoke(command, args, input="")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["version"] == 1
    assert report["diagnostic_prompt"] is None
    assert len(requests) == count
    assert "private reasoning" not in result.output
    assert "Approve" not in result.output
    assert all(
        body.get("max_tokens", body.get("max_completion_tokens")) == 2048
        for i, body in enumerate(requests)
    )
    if stage == "reasoning":
        assert [body["reasoning_effort"] for body in requests] == ["high", "low"]
        assert all(
            body["messages"][0]["content"] == connect.REASONING_CHECK_PROMPT for body in requests
        )
        assert all(record["answer_correct"] is True for record in report["checks"].values())


def test_failure_is_json_with_safe_agent_handoff(monkeypatch):
    monkeypatch.setenv("STAGE_KEY", "secret-value")
    mock_http(
        monkeypatch,
        lambda request: httpx.Response(401, json={"error": {"message": "secret-value rejected"}}),
    )
    result = CliRunner().invoke(
        command, [*BASE, "--stage", "routing", "--api-key-env", "STAGE_KEY"]
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["checks"]["routing"]["status_code"] == 401
    assert "Diagnose and fix" in report["diagnostic_prompt"]
    assert "STAGE_KEY" in report["diagnostic_prompt"]
    assert "secret-value" not in result.output
    assert "Give Feedback" not in result.output


def test_plan_then_explicit_save_has_no_http(monkeypatch, tmp_path):
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP in plan or save"))
    planned = CliRunner().invoke(command, [*BASE, "--stage", "plan"])
    assert planned.exit_code == 0, planned.output
    source = tmp_path / "plan.json"
    source.write_text(planned.stdout)
    target = tmp_path / "models.yaml"
    saved = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert saved.exit_code == 0, saved.output
    assert yaml.safe_load(target.read_text())["models"]["local"]["model_name"] == "openai/gpt-5.1"
    blocked = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert blocked.exit_code == 2
    assert "Alias exists" in json.loads(blocked.stdout)["error"]["message"]


@pytest.mark.parametrize(
    "args", [["--stage", "routing"], [*BASE, "--stage", "routing", "--no-probe"]]
)
def test_invalid_stage_options_fail_before_calls(monkeypatch, args):
    mock_http(monkeypatch, lambda request: pytest.fail("Invalid options cannot send HTTP"))
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 2
    report = json.loads(result.stdout)
    assert report["error"]["type"] == "UsageError"
    assert report["diagnostic_prompt"]


@pytest.mark.asyncio
async def test_library_stage_rechecks_selected_probe_without_mutating_plan(monkeypatch):
    from copy import deepcopy

    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    plan = connect.plan("local", "model", "chat", "https://api.test/v1", "")
    prior = await connect.run(plan, approved="minimal", api_key="key")
    from dataclasses import replace

    plan = replace(plan, entry=prior.entry)
    before = deepcopy(plan.entry)
    await connect.check_stage(plan, "routing", api_key="key")
    assert len(sent) == 2
    assert plan.entry == before


@pytest.mark.parametrize("stage", ["discover", "catalogue", "interfaces"])
def test_discovery_stages_report_structured_results(monkeypatch, stage):
    monkeypatch.setenv("STAGE_TEST_KEY", "test-secret")

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(command, [*BASE, "--stage", stage])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["data"]


def test_stage_catalogue_offers_fuzzy_suggestions_when_nothing_matches(monkeypatch):
    """--stage catalogue used to call only match_models(), never
    fuzzy_match_models() — the wizard's own "did you mean" fallback for a
    gateway-routed model ID never reached this non-interactive JSON front
    door used by scripts/agents.
    """

    async def catalogue():
        return [{"id": "anthropic/claude-opus-5"}]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    result = CliRunner().invoke(
        command,
        [
            "aws/anthropic/bedrock-claude-opus-5",
            "--stage",
            "catalogue",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["data"]["models"] == []
    assert report["data"]["fuzzy_models"] == [{"id": "anthropic/claude-opus-5"}]


@pytest.mark.parametrize("stage", ["tools", "reasoning", "session"])
def test_acceptance_without_feature_evidence_is_inconclusive(monkeypatch, tmp_path, stage):
    monkeypatch.setenv("STAGE_TEST_KEY", "test-secret")
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    levels = tmp_path / "levels.yaml"
    levels.write_text("high: {reasoning_effort: high}\n")
    result = CliRunner().invoke(command, [*BASE, "--stage", stage, "--levels-file", str(levels)])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert not report["ok"]
    assert report["diagnostic_prompt"]


def test_stage_budget_exhaustion_and_missing_levels_are_actionable(monkeypatch):
    monkeypatch.setenv("STAGE_TEST_KEY", "test-secret")
    mock_http(monkeypatch, lambda request: pytest.fail("Neither case may send HTTP"))
    result = CliRunner().invoke(command, [*BASE, "--stage", "routing", "--budget-tokens", "1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["checks"]["routing"]["reason"] == "budget exhausted"
    result = CliRunner().invoke(command, [*BASE, "--stage", "reasoning"])
    assert result.exit_code == 2
    assert "--levels-file" in json.loads(result.stdout)["error"]["message"]


def test_agent_handoff_excludes_payloads_and_credential_url():
    prompt = connect.diagnostic_prompt(
        "routing",
        {"api_key": "secret-value", "api_base": "https://user:secret-value@api.test/v1"},
        {"routing": {"outcome": "rejected", "request": {"input": "OPAQUE-secret"}}},
    )
    assert "secret-value" not in prompt
    assert "OPAQUE-secret" not in prompt
    assert "invalid endpoint omitted" in prompt


@pytest.mark.parametrize("entry", [None, [], {"api_key": "secret-value"}])
def test_invalid_save_document_still_returns_safe_json(monkeypatch, tmp_path, entry):
    mock_http(monkeypatch, lambda request: pytest.fail("Save must not send HTTP"))
    source = tmp_path / "invalid.json"
    target = tmp_path / "models.yaml"
    source.write_text(json.dumps({"alias": "test", "entry": entry}))
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["diagnostic_prompt"]
    assert "secret-value" not in result.output
    assert not target.exists()
