# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approved wizard checks fail closed before registry/secret writes (offline HTTP)."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa import llm_config, secrets
from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._records import TYPED_TEXT_BASELINE
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.delenv("BASELINE_KEY", raising=False)
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(prompts, "choose_media_mode", lambda **kw: "manual")
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(): list(current))


def options(path, editing):
    common = ["--output", str(path), "--max-tokens", "1000", "--no-catalogue", "--prompt-key"]
    if editing:
        entry = {
            "model_name": "openai/model",
            "api_style": "chat",
            "api_base": "https://api.test/v1",
            "api_key_env": "BASELINE_KEY",
            "max_tokens": 1000,
            "media_inputs": ["pdf"],
            "temperature": 0.2,
        }
        path.write_text(yaml.safe_dump({"models": {"local": entry}}))
        return ["--edit-model", "local", *common]
    return [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "BASELINE_KEY",
        *common,
    ]


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("yes", [False, True])
@pytest.mark.parametrize("failure", ["typed", "wrong", "incomplete", "missing_settings", "budget"])
def test_failed_configured_baseline_stops_before_save(monkeypatch, tmp_path, editing, yes, failure):
    path = tmp_path / "models.yaml"
    args = options(path, editing)
    before = path.read_bytes() if editing else None
    calls, writes = [], []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        text = '{"value":323}'
        if failure == "typed":
            text = '{"value":"PRIVATE-ANSWER"}'
        elif failure == "wrong":
            text = '{"value":322}'
        reply = response_body("chat", text)
        if failure == "incomplete":
            reply["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=reply)

    mock_http(monkeypatch, handle)
    if failure == "missing_settings":
        original = connect._run_probe

        async def unobserved(*args):
            response, _, transport = await original(*args)
            return response, None, transport

        monkeypatch.setattr(connect, "_run_probe", unobserved)
    monkeypatch.setattr(connect, "write", lambda *a, **kw: writes.append("registry"))
    monkeypatch.setattr(secrets, "write_secret_env", lambda *a, **kw: writes.append("key"))
    if failure == "budget":
        args += ["--budget-tokens", "1"]
    if yes:
        args += ["--yes"]
    # Extra default-yes answers must not be consumed by ordinary save/key prompts.
    result = CliRunner().invoke(
        command,
        args,
        input="PRIVATE-KEY\n"
        + ("" if yes else ("" if editing else "262144\n262144\n") + "test\n")
        + "\n" * 12,
    )
    assert result.exit_code == 0, result.output
    assert "Nothing was saved" in result.output
    assert "--no-probe" in result.output and "save without testing" in " ".join(
        result.output.split()
    )
    assert "Write model entry" not in result.output
    assert "Save this key" not in result.output and "Save model" not in result.output
    assert "PRIVATE-ANSWER" not in result.output and "PRIVATE-KEY" not in result.output
    assert not writes
    assert path.read_bytes() == before if editing else not path.exists()
    assert len(calls) == (0 if failure == "budget" else 1)
    if calls:
        assert "response_format" in calls[0]
    if failure == "missing_settings":
        assert "not observed" in result.output


GOOD = {
    "baseline_kind": TYPED_TEXT_BASELINE,
    "outcome": "accepted",
    "typed_result_validated": True,
    "answer_correct": True,
    "final_answer_observed": True,
    "settings_sent": True,
    "finish_reason": "stop",
}


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("yes", [False, True])
@pytest.mark.parametrize(
    "evidence", ["missing", "legacy", "missing_answer", "missing_final", "interface"]
)
def test_missing_baseline_evidence_cannot_reach_wizard_save(
    monkeypatch, tmp_path, editing, yes, evidence
):
    path = tmp_path / "models.yaml"
    args = options(path, editing)
    before = path.read_bytes() if editing else None
    record = deepcopy(GOOD)
    if evidence == "missing":
        record = None
    elif evidence == "legacy":
        record.pop("baseline_kind")
        record.pop("typed_result_validated")
    elif evidence == "interface":
        record["baseline_kind"] = "interface_acceptance"
    else:
        record.pop("answer_correct" if evidence == "missing_answer" else "final_answer_observed")

    async def steps(plan, **kw):
        entry = deepcopy(plan.entry)
        entry["provenance"]["probes"] = {} if record is None else {"routing": record}
        yield connect.ConnectResult(plan.alias, entry)

    monkeypatch.setattr(connect, "run_steps", steps)
    monkeypatch.setattr(connect, "write", lambda *a, **kw: pytest.fail("registry write"))
    monkeypatch.setattr(secrets, "write_secret_env", lambda *a, **kw: pytest.fail("key write"))
    result = CliRunner().invoke(
        command,
        args + (["--yes"] if yes else []),
        input="PRIVATE-KEY\n"
        + ("" if yes else ("" if editing else "262144\n262144\n") + "test\n")
        + "\n" * 12,
    )
    assert result.exit_code == 0, result.output
    assert "Nothing was saved" in result.output and "--no-probe" in result.output
    assert "Save model" not in result.output and "Write model entry" not in result.output
    assert path.read_bytes() == before if editing else not path.exists()


@pytest.mark.parametrize("record", [None, {}, {**GOOD, "answer_correct": False}, GOOD])
def test_defensive_save_guard_precedes_prompts_and_writes(monkeypatch, tmp_path, record):
    entry = {
        "provenance": {"probes": {} if record is None else {"routing": record}},
        "media_inputs": [],
    }
    entry = connect.configure_entry(entry)
    if record == GOOD:
        # This case simulates a fresh baseline execution, not imported legacy proof.
        connect.stamp_verification(entry["provenance"]["probes"]["routing"], entry, "routing")
    state = wizard.WizardState(
        approval="all",
        yes=True,
        path=tmp_path / "models.yaml",
        result=connect.ConnectResult("local", entry),
    )
    if record == GOOD:
        assert wizard.baseline_allows_save(state)
    else:
        monkeypatch.setattr(prompts, "confirm", lambda *a, **kw: pytest.fail("save prompt"))
        monkeypatch.setattr(connect, "write", lambda *a, **kw: pytest.fail("registry write"))
        assert not wizard.save_model(state)
        assert not state.path.exists()


@pytest.mark.parametrize("flag", [["--no-probe"], ["--probe", "none"]])
@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("yes", [False, True])
def test_explicit_manual_offline_save_remains_available(monkeypatch, tmp_path, flag, editing, yes):
    path = tmp_path / "models.yaml"
    args = options(path, editing)
    args.remove("--prompt-key")
    monkeypatch.setenv("BASELINE_KEY", "existing-key")
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(): ["image", "pdf"])
    mock_http(monkeypatch, lambda request: pytest.fail("manual setup made HTTP request"))
    result = CliRunner().invoke(
        command,
        args + flag + ["--media-inputs", "image,pdf"] + (["--yes"] if yes else []),
        input=("save\ny\ny\n" if editing else "262144\n262144\nsave\ny\n"),
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["media_inputs"] == ["image", "pdf"]
    assert "provenance" not in entry
    assert (
        "Alias: local" in " ".join(result.output.split())
        and "not tested (manual setup)" in result.output
    )
