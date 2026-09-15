# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encrypted reasoning defaults and bounded compatibility fallback on real HTTP bodies."""

import json
from copy import deepcopy

import httpx
import pytest

from nooa import connect
from tests.connect_http import mock_http, response_body


def proposal(*, api_base="https://api.openai.com/v1", **kwargs):
    return connect.plan("model", "gpt-5.1", "responses", api_base, "", **kwargs)


@pytest.mark.asyncio
async def test_default_saved_and_sent_without_probe_side_effects(monkeypatch):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        return httpx.Response(200, json=response_body("responses"))

    mock_http(monkeypatch, handle)
    plan = proposal()
    before = deepcopy(plan.entry)
    skipped = await connect.run(plan, approved="none")
    assert not bodies
    assert skipped.entry["include"] == ["reasoning.encrypted_content"]
    result = await connect.run(plan, approved="minimal", api_key="secret")
    assert result.entry["provenance"]["encrypted_reasoning"] == {
        "source": "connect",
        "outcome": "accepted",
    }
    assert plan.entry == before


@pytest.mark.asyncio
@pytest.mark.parametrize("budget, count, outcome", [(4096, 2, "accepted"), (712, 1, "not_probed")])
@pytest.mark.parametrize("api_base", ["https://api.openai.com/v1", "https://api.test/v1"])
async def test_rejection_removes_wire_field_even_on_native_endpoint(
    monkeypatch, budget, count, outcome, api_base
):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["store"] is False
        assert body["max_output_tokens"] == 200
        if len(bodies) == 1:
            assert body["include"] == ["reasoning.encrypted_content"]
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter include secret-never-persist",
                        "param": "include",
                    }
                },
            )
        assert "include" not in body
        return httpx.Response(200, json=response_body("responses"))

    mock_http(monkeypatch, handle)
    plan = proposal(budget_tokens=budget, api_base=api_base)
    result = await connect.run(plan, approved="minimal", api_key="key")
    assert len(bodies) == count
    assert result.entry["include"] == []
    provenance = result.entry["provenance"]
    assert provenance["probes"]["routing"]["outcome"] == outcome
    assert provenance["encrypted_reasoning"]["outcome"] == "rejected"
    assert provenance["tokens_charged_to_budget"] == count * 712
    assert "secret-never-persist" not in repr(provenance)
    again = proposal(existing_entry=result.entry, api_base=api_base)
    assert again.entry["include"] == []
    assert proposal().entry["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,message,param",
    [
        (401, "Unsupported include", "include"),
        (429, "Unsupported include", "include"),
        (500, "Unsupported include", "include"),
        (400, "Invalid model", "model"),
        (400, "Unknown model include", "model"),
        (400, "Invalid encrypted_content", "input[0]"),
    ],
)
async def test_unrelated_failures_do_not_disable_reasoning(monkeypatch, status, message, param):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(status, json={"error": {"message": message, "param": param}})

    mock_http(monkeypatch, handle)
    result = await connect.run(proposal(), approved="minimal", api_key="key")
    assert len(sent) == 1
    assert result.entry["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_session_field_rejection_is_recorded_without_rerunning_session(monkeypatch):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["include"] == ["reasoning.encrypted_content"]
        if body["max_output_tokens"] == 2048:
            return httpx.Response(
                422,
                json={
                    "error": {
                        "message": "reasoning.encrypted_content is not supported",
                        "param": "include",
                    }
                },
            )
        return httpx.Response(200, json=response_body("responses"))

    mock_http(monkeypatch, handle)
    result = await connect.run(
        proposal(session_checks=True, budget_tokens=65536), approved="all", api_key="key"
    )
    assert len(bodies) == 3  # Routing, tools, then session seed: no fourth call.
    assert result.entry["include"] == []
    assert result.entry["provenance"]["session_checks"]["session"]["outcome"] == "not_confirmed"


def test_wizard_explains_and_saves_default_without_extra_question(tmp_path):
    import yaml
    from click.testing import CliRunner
    from nooa_cli.commands.connect import command

    target = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "gpt-5.1",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "responses",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--no-probe",
            "--yes",
            "--output",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "encrypted reasoning" in result.output
    assert "not to store replies" in result.output
    entry = yaml.safe_load(target.read_text())["models"]["local"]
    assert entry["include"] == ["reasoning.encrypted_content"]
    assert entry["provenance"]["encrypted_reasoning"]["source"] == "connect"
