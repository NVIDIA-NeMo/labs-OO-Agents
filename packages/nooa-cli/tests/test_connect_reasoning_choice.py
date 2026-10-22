# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guided human reasoning selection: GET-only proposal then active saved default."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import registry
from tests.unifiedllm.connect.connect_http import mock_http, response_body
from tests.unifiedllm.connect.reasoning_fixture import MODEL_ID, NATIVE_MODEL


@pytest.fixture
def native(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("OFFLINE_KEY", "fake")
    calls = []
    choices = []
    original = prompts.labelled_choice

    def choose(label, options, **kwargs):
        choices.append((label, tuple(options)))
        return original(label, options, **kwargs)

    monkeypatch.setattr(prompts, "labelled_choice", choose)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [deepcopy(NATIVE_MODEL)]})
        data = response_body("anthropic", '{"value":323}')
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    flags = [
        MODEL_ID,
        "--as",
        "local",
        "--provider",
        "anthropic",
        "--api-style",
        "anthropic",
        "--api-key-env",
        "OFFLINE_KEY",
        "--media-inputs",
        "",
        "--reply-tokens",
        "4096",
        "--probe",
        "minimal",
        "--output",
        str(path),
    ]
    return flags, path, calls, choices


@pytest.mark.parametrize(
    "selection, active", [("high", "high"), ("xhigh", "xhigh"), ("off", "off"), ("default", None)]
)
def test_guided_selection_save_reload_default_wire(native, monkeypatch, selection, active):
    flags, path, calls, choices = native
    result = CliRunner().invoke(command, flags, input=selection + "\nsave\ny\n")
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    assert result.output.index("Model details") < result.output.index("Choose how much reasoning")
    assert result.output.index("Choose how much reasoning") < result.output.index(
        "Proposed configuration"
    )
    options = next(opts for label, opts in choices if label == "Reasoning")
    assert [v for v, _ in options] == [
        "default",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "off",
        "cancel",
    ]
    assert "not tested (manual setup)" in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry.get("reasoning_level") == active
    assert entry["max_tokens"] == 4096
    assert set(entry["reasoning_levels"]) == {"low", "medium", "high", "xhigh", "max", "off"}
    assert "thinking" not in entry and "output_config" not in entry
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    registry.reload_registry(path)
    c = registry.get_llm_client("local")
    try:
        asyncio.run(c.acall([{"role": "user", "content": "hello"}]))
        wire = json.loads(calls[-1].content)
        if active == "off":
            assert wire["thinking"] == {"type": "disabled"}
            assert "output_config" not in wire
        elif active:
            assert wire["thinking"] == {"type": "adaptive"}
            assert wire["output_config"]["effort"] == active
        else:
            assert "thinking" not in wire and "output_config" not in wire
        assert wire["max_tokens"] == 4096
    finally:
        asyncio.run(c.aclose())


def test_cancel_reasoning_writes_nothing(native):
    flags, path, calls, choices = native
    result = CliRunner().invoke(command, flags, input="cancel\n")
    assert result.exit_code == 0, result.output
    assert not path.exists()
    assert all(r.method == "GET" for r in calls)
    assert "Nothing saved" in result.output


def test_guided_change_reasoning_and_test_same_selected_setting(native):
    flags, path, calls, choices = native
    result = CliRunner().invoke(
        command, flags, input="low\nchange\n1000000\n128000\ncustom\n4096\nhigh\ntest\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert "Reasoning levels (comma-separated)" not in result.output
    posts = [json.loads(r.content) for r in calls if r.method == "POST"]
    assert posts and all(
        b["thinking"] == {"type": "adaptive"}
        and b["output_config"]["effort"] == "high"
        and b["max_tokens"] == 4096
        for b in posts
    )
    assert yaml.safe_load(path.read_text())["models"]["local"]["reasoning_level"] == "high"
    assert result.output.count("Reasoning:") == 3  # two proposals and the selected save summary


def test_edit_keep_current_raw_manual_profile_and_active(native):
    flags, path, calls, choices = native
    # An existing manual block is preserved, never recompiled as adaptive.
    original = {
        "model_name": "anthropic/" + MODEL_ID,
        "api_style": "anthropic",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env": "OFFLINE_KEY",
        "context_window": 1000000,
        "max_tokens": 4096,
        "media_inputs": [],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "reasoning_levels": {"on": {"thinking": {"type": "enabled", "budget_tokens": 1024}}},
        "reasoning_level": "on",
        "cache_breakpoint": None,
    }
    path.write_text(yaml.safe_dump({"models": {"local": original}}))
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "local",
            "--output",
            str(path),
            "--media-inputs",
            "",
            "--probe",
            "minimal",
        ],
        input="keep\nsave\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["reasoning_level"] == "on" and entry["thinking"] == original["thinking"]
    assert entry["reasoning_levels"] == original["reasoning_levels"]
    assert entry["max_tokens"] == 4096
    assert all(r.method == "GET" for r in calls)


def test_unknown_false_foreign_and_decision_do_not_offer_guessed_off(monkeypatch):
    state = wizard.WizardState(
        model=MODEL_ID,
        api_style="chat",
        endpoint="https://custom.test/v1",
        selected_endpoint_model=NATIVE_MODEL,
        endpoint_models_url="https://custom.test/v1/models",
    )
    monkeypatch.setattr(
        prompts, "labelled_choice", lambda *a, **kw: pytest.fail("no known controls")
    )
    assert wizard.choose_reasoning(state)
    assert state.guided_reasoning_choice is None
    state.api_style = "systemone"
    assert wizard.choose_reasoning(state)
    assert state.guided_reasoning_levels is None


def test_tty_reasoning_uses_labelled_menu(monkeypatch):
    state = wizard.WizardState(
        model=MODEL_ID,
        api_style="anthropic",
        endpoint="https://api.anthropic.com/v1",
        selected_endpoint_model=NATIVE_MODEL,
        endpoint_models_url="https://api.anthropic.com/v1/models",
    )
    seen = []
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)

    def choose(label, options, **kwargs):
        seen.append((label, options, kwargs))
        return "medium"

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    assert wizard.choose_reasoning(state)
    assert seen[0][0] == "Reasoning" and seen[0][2]["default"] == "default"
    assert state.guided_reasoning_active == "medium"


def test_saved_alias_controls_never_cross_routes(monkeypatch):
    saved = {
        "model_name": "openai/old",
        "api_style": "chat",
        "api_base": "https://old.test/v1",
        "client_type": "completion",
        "reasoning_level": "high",
        "reasoning_levels": {"high": {"reasoning_effort": "high"}},
    }
    state = wizard.WizardState(
        model="new", api_style="anthropic", endpoint="https://new.test/v1", existing=saved
    )
    monkeypatch.setattr(
        prompts, "labelled_choice", lambda *a, **kw: pytest.fail("foreign controls")
    )
    assert wizard.choose_reasoning(state)
    assert state.guided_reasoning_active is None and state.guided_reasoning_levels is None


def test_same_route_keep_raw_controls_on_alias_reonboarding(monkeypatch):
    raw = {
        "model_name": "anthropic/" + MODEL_ID,
        "api_style": "anthropic",
        "api_base": "https://api.anthropic.com/v1",
        "client_type": "completion",
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    state = wizard.WizardState(
        model=MODEL_ID,
        api_style="anthropic",
        endpoint=raw["api_base"],
        alias="local",
        api_key_env="",
        existing=raw,
        guided=True,
        media_inputs=[],
        budget_tokens=200000,
        output_tokens=2048,
        reasoning_output_tokens=4096,
        reply_tokens=4096,
        yes=True,
    )
    monkeypatch.setattr(prompts, "labelled_choice", lambda *a, **kw: "keep")
    assert wizard.choose_reasoning(state)
    assert wizard.configure_checks(state)
    assert state.proposal.entry["thinking"] == raw["thinking"]
    assert "reasoning_level" not in state.proposal.entry


def test_existing_level_named_default_is_not_the_model_default_action(monkeypatch):
    state = wizard.WizardState(
        model="model",
        api_style="chat",
        endpoint="https://api.test/v1",
        levels="default",
        reasoning_template="toggle",
    )

    def choose(label, options, **kwargs):
        assert ("default", "Model default (no new override)") in options
        assert ("level:default", "Default") in options
        return "level:default"

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    assert wizard.choose_reasoning(state)
    assert state.guided_reasoning_active == "default"
