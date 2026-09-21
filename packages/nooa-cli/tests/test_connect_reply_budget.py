# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recommendation or custom value sets the same actual reply limit as scripts."""

import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command


@pytest.mark.parametrize(
    "choice,expected",
    [
        ("", 32768),
        ("recommended", 32768),
        ("high", 65536),
        ("extended", 131072),
        ("smaller", 8192),
        ("short", 2048),
        ("custom\n512", 512),
    ],
)
def test_wizard_reply_budget_is_a_runtime_cap(tmp_path, monkeypatch, choice, expected):
    from nooa import llm_config

    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "model",
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
            "--output",
            str(path),
        ],
        input=choice + "\ny\n",
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["max_tokens"] == expected
    assert entry["include"] == ["reasoning.encrypted_content"]
    assert entry["store"] is False
    assert "Recommended — 32,768 tokens (NOOA default)" in result.output
    assert "Short replies and simple questions" not in result.output
    assert "Coding agents and tool use" not in result.output
    assert "Long documents and deeper reasoning" not in result.output


def test_reply_dialog_offers_recommendation_smaller_and_custom(monkeypatch, capsys):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    calls = []
    answers = iter(["custom", "99999", "16384"])

    def prompt(text, **kwargs):
        calls.append((text, kwargs))
        return next(answers)

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    assert (
        _connect_prompts.choose_reply_limit(32768, 65536, source="catalogue_recommendation")
        == 16384
    )
    menu = calls[0][1]
    assert menu["choices"] == ("recommended", "high", "smaller", "short", "custom")
    assert menu["labels"]["high"] == "High reasoning budget — 65,536 tokens"
    assert menu["default"] == "recommended"
    assert menu["open_menu"] is True
    assert menu["labels"]["recommended"] == "Recommended — 32,768 tokens (catalogue recommendation)"
    assert calls[1][1]["default"] == "32768"
    output = capsys.readouterr()
    assert "Known upper limit: 65,536 tokens" in output.out
    assert "at most 65,536" in output.err


@pytest.mark.parametrize("ceiling", [32768, 64000])
def test_high_reasoning_options_never_exceed_known_limit(monkeypatch, ceiling):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    def prompt(text, **kwargs):
        assert "high" not in kwargs["choices"]
        assert "extended" not in kwargs["choices"]
        return "recommended"

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    assert _connect_prompts.choose_reply_limit(32768, ceiling) == 32768


def test_model_maximum_choice_is_offered_and_selectable(monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    def prompt(text, **kwargs):
        assert kwargs["choices"] == ("recommended", "high", "max", "smaller", "short", "custom")
        assert kwargs["labels"]["max"] == "Model maximum — 100,000 tokens"
        return "max"

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    assert _connect_prompts.choose_reply_limit(32768, 100000, output_ceiling=100000) == 100000


def test_model_maximum_is_omitted_when_unknown(monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    def prompt(text, **kwargs):
        assert "max" not in kwargs["choices"]
        return "recommended"

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    # ceiling is known (context window) but output_ceiling (true max output) isn't.
    assert _connect_prompts.choose_reply_limit(32768, 200000, output_ceiling=None) == 32768


def test_boolean_catalogue_output_limit_is_not_treated_as_one_token(tmp_path, monkeypatch):
    from nooa_cli.commands import _connect_prompts as prompts

    from nooa.unifiedllm import connect

    real_configure_entry = connect.configure_entry

    def configure_with_boolean_catalogue_limit(*args, **kwargs):
        configured = real_configure_entry(*args, **kwargs)
        configured["provenance"]["catalogue_limits"] = {"max_completion_tokens": True}
        return configured

    chosen = {}

    def choose_reply_limit(suggested, ceiling, *, output_ceiling=None, **kwargs):
        chosen["ceiling"] = ceiling
        chosen["output_ceiling"] = output_ceiling
        return suggested

    monkeypatch.setattr(connect, "configure_entry", configure_with_boolean_catalogue_limit)
    monkeypatch.setattr(prompts, "choose_reply_limit", choose_reply_limit)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "model",
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
            "--output",
            str(path),
        ],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert chosen["output_ceiling"] is None
    assert chosen["ceiling"] != 1


def test_model_maximum_is_omitted_when_it_duplicates_extended(monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    def prompt(text, **kwargs):
        assert "max" not in kwargs["choices"]
        return "extended"

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    assert _connect_prompts.choose_reply_limit(32768, 131072, output_ceiling=131072) == 131072


def test_model_maximum_never_exceeds_the_stricter_context_ceiling(monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    def prompt(text, **kwargs):
        assert "max" not in kwargs["choices"]
        return "recommended"

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    # output_ceiling (200000) exceeds the stricter overall ceiling (40000, e.g.
    # from context window), so it must not be offered as an achievable choice.
    assert _connect_prompts.choose_reply_limit(32768, 40000, output_ceiling=200000) == 32768


def test_stage_save_fills_defaults_and_reports_shadow(tmp_path, monkeypatch):
    from nooa import llm_config

    source = tmp_path / "override.yaml"
    source.write_text("models: {local: {model_name: openai/old}}\n")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(source))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [source])
    document = tmp_path / "entry.json"
    document.write_text(
        json.dumps(
            {"alias": "local", "entry": {"model_name": "openai/model", "client_type": "responses"}}
        )
    )
    destination = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(document), "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["data"]["shadowed_by"] == str(source)
    entry = report["data"]["entry"]
    assert entry["max_tokens"] == 32768
    assert entry["include"] == ["reasoning.encrypted_content"]
    assert yaml.safe_load(destination.read_text())["models"]["local"] == entry


def test_scripted_plan_honours_explicit_reply_cap():
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "responses",
            "--max-tokens",
            "1234",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)["data"]
    assert plan["entry"]["max_tokens"] == 1234
    assert plan["probes"][0]["body"]["max_output_tokens"] == 1234


def test_stage_reasoning_budget_override_reaches_wire(monkeypatch, tmp_path):
    import httpx

    from tests.unifiedllm.connect.connect_http import mock_http, response_body

    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["max_output_tokens"] == 8192
        return httpx.Response(200, json=response_body("responses", "B G D A C E F H"))

    mock_http(monkeypatch, handle)
    monkeypatch.setenv("TEST_REASONING_KEY", "test-key")
    levels = tmp_path / "levels.yaml"
    levels.write_text("high: {reasoning: {effort: high}}\n")
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "reasoning",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "responses",
            "--api-key-env",
            "TEST_REASONING_KEY",
            "--levels-file",
            str(levels),
            "--max-tokens",
            "8192",
            "--budget-tokens",
            "9000",
            "--yes",
        ],
    )
    assert result.exit_code == 1  # Correct answer, but mock returns no reasoning evidence.
    assert len(seen) == 1
    report = json.loads(result.stdout)
    assert report["checks"]["level:high"]["answer_correct"] is True


def test_shadowing_source_extra_priority_treats_the_save_target_as_highest(tmp_path, monkeypatch):
    """entries()'s extra_path is always appended last (highest priority) --
    a --working-dir save is never actually shadowed by anything as long as
    the caller keeps pairing -w with the same directory. Without
    extra_priority=True, shadowing_source used to report a shadow warning
    on every --working-dir save with a same-named alias defined elsewhere,
    even though that is --working-dir's whole intended, correct use.
    """
    from nooa_cli.commands._connect_registry import shadowing_source

    from nooa import llm_config

    elsewhere = tmp_path / "elsewhere.yaml"
    elsewhere.write_text("models: {local: {model_name: openai/old}}\n")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [elsewhere])
    # shadowing_source's own priority list is built from conventional
    # locations (bundled/user/project/NEMO_OO_LLM_CONFIG), independent of
    # the mocked llm_config_chain() used only to resolve `found` above --
    # register `elsewhere` as one so it participates in that comparison.
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(elsewhere))

    target = tmp_path / "target.yaml"
    assert shadowing_source("local", target) == str(elsewhere)
    assert shadowing_source("local", target, extra_priority=True) is None
