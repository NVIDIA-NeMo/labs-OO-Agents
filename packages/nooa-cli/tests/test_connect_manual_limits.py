# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal manual declarations, supported persistence and real guided consent."""

import asyncio
import json
from copy import deepcopy

import click
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


@pytest.fixture
def manual_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    calls, proposals, decisions = [], [], []
    show = wizard.show_guided_proposal
    choose = prompts.labelled_choice

    def show_proposal(state, **kwargs):
        proposals.append(deepcopy(state.proposal.entry))
        return show(state, **kwargs)

    def consent(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            assert all(request.method == "GET" for request in calls)
            decisions.append(tuple(options))
        return choose(label, options, **kwargs)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            public = request.url.host == "openrouter.ai"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "gateway/wired-model" if public else "wire/model",
                            **(
                                {
                                    "context_length": 128000,
                                    "top_provider": {"max_completion_tokens": 4096},
                                    "reasoning": {"supported_efforts": ["guessed"]},
                                }
                                if public
                                else {}
                            ),
                        }
                    ]
                },
            )
        body = json.loads(request.content)
        assert "max_input_tokens" not in body
        assert body["max_tokens"] == 1000
        return httpx.Response(200, json=response_body("chat"))

    monkeypatch.setattr(wizard, "show_guided_proposal", show_proposal)
    monkeypatch.setattr(prompts, "labelled_choice", consent)
    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    flags = [
        "wire/model",
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
        "--probe",
        "minimal",
        "--reply-tokens",
        "1000",
        "--output",
        str(path),
    ]
    return flags, path, calls, proposals, decisions


@pytest.mark.parametrize("context,input_limit", [(128000, 64000), (32000, 96000)])
@pytest.mark.parametrize("manual", ["blank-picker", "no-catalogue", "no-metadata"])
@pytest.mark.parametrize("action", ["save", "test"])
def test_actual_guided_manual_limits_are_literal_and_conservative(
    manual_setup, context, input_limit, manual, action
):
    flags, path, calls, proposals, decisions = manual_setup
    if manual != "blank-picker":
        flags += ["--" + manual]
    answers = "\n" if manual == "blank-picker" else ""
    answers += f"{context}\n{input_limit}\n{action}\ny\n"
    result = CliRunner().invoke(command, flags, input=answers)
    assert result.exit_code == 0, result.output
    assert len(decisions) == 1
    assert len([r for r in calls if r.method == "POST"]) == (action == "test")
    assert "Context window (tokens)" in result.output
    assert "Input token limit (tokens)" in result.output
    assert "Model details: Your settings" in " ".join(result.output.split())
    entry = proposals[0]
    assert entry["context_window"] == min(context, input_limit)
    provenance = entry["provenance"]
    assert provenance["manual_limits"] == {
        "source": "user",
        "context_window": context,
        "max_input_tokens": input_limit,
    }
    assert provenance["context_window"] == {
        "source": "user",
        "value": min(context, input_limit),
        "declared": context,
        "effective": min(context, input_limit),
        "effective_source": "user",
    }
    assert "catalogue" not in provenance
    assert not provenance["probes"] and not entry.get("reasoning_levels")
    assert "max_input_tokens" not in entry and "max_output_tokens" not in entry
    assert entry["max_tokens"] == 1000
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert saved["context_window"] == min(context, input_limit)
    assert saved["max_tokens"] == 1000
    assert "provenance" not in saved and "max_input_tokens" not in saved


@pytest.mark.parametrize("invalid", ["", "-", "0", "-4", "1.5", "True", "unknown", "１２"])
@pytest.mark.parametrize("field", ["context", "input"])
def test_invalid_manual_numbers_reprompt_before_consent(manual_setup, invalid, field):
    flags, path, calls, _, decisions = manual_setup
    answers = (
        invalid + "\n128000\n64000\n" if field == "context" else "128000\n" + invalid + "\n64000\n"
    )
    result = CliRunner().invoke(command, flags + ["--no-catalogue"], input=answers + "save\ny\n")
    assert result.exit_code == 0, result.output
    assert "Enter a positive whole number." in result.output
    assert len(decisions) == 1 and not calls
    assert yaml.safe_load(path.read_text())["models"]["local"]["context_window"] == 64000


@pytest.mark.parametrize("answers", ["", "128000\n", "128000\n64000\nchange\n48000\n"])
def test_cancellation_at_either_number_or_reedit_writes_nothing(manual_setup, answers):
    flags, path, calls, proposals, _ = manual_setup
    result = CliRunner().invoke(command, flags + ["--no-catalogue"], input=answers)
    assert result.exit_code == 1, result.output
    assert not path.exists() and not calls
    if proposals:
        assert proposals[0]["provenance"]["manual_limits"]["context_window"] == 128000


def test_change_revises_both_raw_limits_and_keeps_reply_separate(manual_setup):
    flags, path, calls, proposals, decisions = manual_setup
    result = CliRunner().invoke(
        command,
        flags + ["--no-catalogue"],
        input="128000\n64000\nchange\n48000\n24000\nrecommended\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(decisions) == 2 and not calls
    assert proposals[0]["context_window"] == 64000
    assert proposals[1]["context_window"] == 24000
    assert proposals[1]["provenance"]["manual_limits"] == {
        "source": "user",
        "context_window": 48000,
        "max_input_tokens": 24000,
    }
    assert proposals[1]["max_tokens"] == 1000
    assert yaml.safe_load(path.read_text())["models"]["local"]["context_window"] == 24000


def test_explicit_context_flag_is_not_an_input_or_reply_limit(manual_setup):
    flags, path, _, proposals, _ = manual_setup
    result = CliRunner().invoke(
        command, flags + ["--no-catalogue", "--context-window", "48000"], input="24000\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert "Context window (tokens):" not in result.output
    assert "Input token limit (tokens)" in result.output
    assert proposals[0]["provenance"]["manual_limits"]["context_window"] == 48000
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert saved["context_window"] == 24000 and saved["max_tokens"] == 1000


@pytest.mark.parametrize("input_limit", [1000, 999])
def test_explicit_reply_cannot_consume_or_exceed_effective_bound(manual_setup, input_limit):
    flags, path, calls, _, decisions = manual_setup
    result = CliRunner().invoke(
        command, flags + ["--no-catalogue"], input=f"128000\n{input_limit}\nsave\ny\n"
    )
    assert result.exit_code != 0
    assert not calls and not decisions and not path.exists()
    assert "Reply limit leaves no room for input" in result.output


@pytest.mark.asyncio
async def test_save_reload_resolves_supported_effective_bound_only(manual_setup, monkeypatch):
    flags, path, calls, _, _ = manual_setup
    # Invoke the synchronous wizard before entering the runtime client loop.
    result = await asyncio.to_thread(
        CliRunner().invoke, command, flags + ["--no-catalogue"], input="128000\n64000\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)
    registry.reload_registry(path)
    client = registry.get_llm_client("local")
    try:
        limits = client.get_context_limits()
        assert limits.context_window == 64000
        assert limits.reserved_output_tokens == 1000
        assert limits.usable_input_tokens == 63000
        await client.acall(messages=[{"role": "user", "content": "Hello"}])
    finally:
        await client.aclose()
    assert [r.method for r in calls] == ["POST"]
    body = json.loads(calls[0].content)
    assert body["max_tokens"] == 1000 and "max_input_tokens" not in body


@pytest.mark.parametrize(
    "server_limits",
    [
        {"context_window": 128000, "max_input_tokens": 64000},
        {"max_input_tokens": 64000},
        {"context_window": 64000},
    ],
)
def test_skip_retains_exact_server_limits_and_raw_provenance(monkeypatch, server_limits):
    endpoint_model = {"id": "model", **server_limits}
    state = wizard.WizardState(
        model="model",
        endpoint="https://api.test/v1",
        api_style="chat",
        api_key_env="",
        selected_endpoint_model=endpoint_model,
        candidate={
            "id": "public/model",
            "context_length": 999999,
            "reasoning": {"supported_efforts": ["guess"]},
        },
        metadata_prepared=True,
        yes=False,
        guided=False,
        output_tokens=200,
        reasoning_output_tokens=200,
        budget_tokens=100000,
        reply_tokens=1000,
        media_inputs=[],
    )
    seen = []

    def prompt(label, **kwargs):
        seen.append(label)
        assert label == "Model settings"
        return "skip"

    monkeypatch.setattr(prompts, "prompt", prompt)
    assert wizard.configure_metadata(state)
    assert wizard.configure_checks(state)
    assert seen == ["Model settings"]
    entry = state.proposal.entry
    assert entry["context_window"] == 64000
    assert entry["provenance"]["endpoint_limits"] == {
        **server_limits,
        "url": "https://api.test/v1/models",
    }
    assert not entry.get("reasoning_levels") and "catalogue" not in entry["provenance"]
    assert state.manual_limits is None


def test_actual_skip_requires_both_numbers_without_published_identity(monkeypatch):
    state = wizard.WizardState(
        model="model",
        endpoint="https://api.test/v1",
        api_style="chat",
        api_key_env="",
        candidate={"id": "public/model", "context_length": 999999},
        metadata_prepared=True,
        yes=False,
        guided=False,
        media_inputs=[],
        output_tokens=200,
        reasoning_output_tokens=200,
        budget_tokens=100000,
        reply_tokens=1000,
    )

    @click.command()
    def review():
        assert wizard.configure_metadata(state)
        assert wizard.configure_checks(state)

    result = CliRunner().invoke(review, input="skip\n128000\n64000\n")
    assert result.exit_code == 0, result.output
    assert "Source: Your settings" in " ".join(result.output.split())
    assert state.candidate is None
    assert state.proposal.entry["context_window"] == 64000
    assert "catalogue" not in state.proposal.entry["provenance"]


@pytest.mark.parametrize("yes,style", [(True, "chat"), (False, "openai"), (False, "systemone")])
def test_yes_and_native_never_prompt_for_chat_limits(monkeypatch, yes, style):
    state = wizard.WizardState(
        model="model", yes=yes, api_style=style, no_catalogue=True, manual_limits_required=True
    )
    monkeypatch.setattr(
        prompts, "manual_model_limits", lambda *a, **kw: pytest.fail("Must not prompt")
    )
    wizard.require_manual_limits(state)
    assert state.manual_limits is None and state.context_window is None


def test_guided_images_test_choice_then_save_clears_pending_permissions(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    path = tmp_path / "models.yaml"
    calls, proposals, decisions = [], [], []
    flags = [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "",
        "--probe",
        "all",
        "--reply-tokens",
        "1000",
        "--output",
        str(path),
    ]
    show = wizard.show_guided_proposal
    choose = prompts.labelled_choice

    def show_proposal(state, **kwargs):
        proposals.append(deepcopy(state.proposal.entry))
        return show(state, **kwargs)

    def consent(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            assert all(r.method == "GET" for r in calls)
            decisions.append(tuple(options))
        return choose(label, options, **kwargs)

    monkeypatch.setattr(wizard, "show_guided_proposal", show_proposal)
    monkeypatch.setattr(prompts, "labelled_choice", consent)

    def handle(request):
        calls.append(request)
        assert request.method == "GET"  # No Test now: not even a baseline POST.
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "wire/model",
                        "context_window": 128000,
                        "max_input_tokens": 64000,
                        "input_modalities": ["image"],
                    }
                ]
            },
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(command, flags, input="test\nsave\ny\n")
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Suggested inputs: Images" in output and "Current inputs: Off" in output
    assert "1 (test): Test and enable Images" in output
    assert "Images: Test before enabling" in output
    assert len(decisions) == 1 and calls
    assert proposals[0]["media_inputs"] == []
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == []
    assert all(r.method == "GET" for r in calls)


def test_cancelled_reedit_does_not_apply_partial_raw_values(monkeypatch):
    state = wizard.WizardState(
        model="model",
        api_style="chat",
        guided=True,
        manual_limits={"context_window": 128000, "max_input_tokens": 64000},
        context_window=64000,
    )
    original = deepcopy(state.manual_limits)
    count = 0

    def prompt(label, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return "48000"
        raise click.Abort()

    monkeypatch.setattr(prompts, "prompt", prompt)
    with pytest.raises(click.Abort):
        wizard.require_manual_limits(state, edit=True)
    assert state.manual_limits == original and state.context_window == 64000


def test_actual_guided_skip_discards_public_guesses_and_requires_manual_numbers(manual_setup):
    flags, path, calls, proposals, decisions = manual_setup
    result = CliRunner().invoke(
        command, flags, input="gateway/wired-model\nskip\n128000\n64000\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert len(decisions) == 2 and all(r.method == "GET" for r in calls)
    assert dict(decisions[0])["skip"] == "Continue without catalogue settings"
    assert "skip" not in dict(decisions[1])
    assert len(proposals) == 2
    final = proposals[1]
    assert final["context_window"] == 64000 and final["max_tokens"] == 1000
    assert final["provenance"]["manual_limits"]["context_window"] == 128000
    assert "catalogue" not in final["provenance"] and not final.get("reasoning_levels")
    assert "Model details: Your settings" in " ".join(result.output.split())
    assert yaml.safe_load(path.read_text())["models"]["local"]["context_window"] == 64000


@pytest.mark.parametrize("field", ["context", "input"])
@pytest.mark.parametrize("edit", [False, True])
def test_one_token_bound_reprompts_in_initial_setup_and_reedit(manual_setup, field, edit):
    flags, path, calls, proposals, decisions = manual_setup
    numbers = "1\n128000\n64000\n" if field == "context" else "128000\n1\n64000\n"
    answers = "128000\n64000\nchange\n" if edit else ""
    answers += numbers + ("recommended\n" if edit else "") + "save\ny\n"
    result = CliRunner().invoke(command, flags + ["--no-catalogue"], input=answers)
    assert result.exit_code == 0, result.output
    assert "enter at least 2 tokens" in " ".join(result.output.split())
    assert len(decisions) == (2 if edit else 1) and not calls
    assert proposals[-1]["context_window"] == 64000
    assert yaml.safe_load(path.read_text())["models"]["local"]["context_window"] == 64000


@pytest.mark.parametrize("action", ["test", "save"])
def test_deferred_completion_reviews_raw_limits_and_preserves_edited_reply(
    monkeypatch, tmp_path, action
):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    path = tmp_path / "models.yaml"
    calls, proposals, decisions = [], [], []
    show, choose = wizard.show_guided_proposal, prompts.labelled_choice

    def show_proposal(state, **kwargs):
        proposals.append(deepcopy(state.proposal.entry))
        return show(state, **kwargs)

    def consent(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            # There may be approved discovery requests before the final review,
            # but no configured 1234-token baseline can precede that review.
            assert all(json.loads(r.content).get("max_tokens") != 1234 for r in calls)
            decisions.append(tuple(options))
        return choose(label, options, **kwargs)

    def handle(request):
        calls.append(request)
        assert request.method == "POST"
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(404)

    monkeypatch.setattr(wizard, "show_guided_proposal", show_proposal)
    monkeypatch.setattr(prompts, "labelled_choice", consent)
    mock_http(monkeypatch, handle)
    flags = [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-key-env",
        "",
        "--no-catalogue",
        "--media-inputs",
        "",
        "--probe",
        "minimal",
        "--reply-tokens",
        "2000",
        "--output-tokens",
        "200",
        "--output",
        str(path),
    ]
    # Literal limits precede even an undecided interface; intentional edits still
    # have their own current review, without a second automatic limit collection.
    answers = (
        "128000\n64000\ndiscover\n"
        "change\n128000\n64000\ncustom\n1234\n"
        "change\n128000\n64000\nrecommended\n"
        "change\n48000\n24000\nrecommended\n" + action + "\n"
    )
    if action == "save":
        answers += "chat\n"
    answers += ("test\n" if action == "test" else "") + "y\n"
    result = CliRunner().invoke(command, flags, input=answers)
    assert result.exit_code == 0, result.output
    assert len(decisions) == (5 if action == "test" else 4)
    assert proposals[-3 if action == "test" else -2]["context_window"] == 64000
    assert proposals[-1]["context_window"] == 24000
    assert proposals[-1]["max_tokens"] == 1234  # Not the original 2000 or a default.
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert saved["context_window"] == 24000 and saved["max_tokens"] == 1234
    assert "Model details: Your settings" in " ".join(result.output.split())
    assert "Context window (declared): 48,000 tokens" in " ".join(result.output.split())
    assert "Input token limit (declared): 24,000 tokens" in " ".join(result.output.split())
    assert len([r for r in calls if json.loads(r.content).get("max_tokens") == 1234]) == (
        action == "test"
    )
    if action == "save":
        assert not calls


def test_guided_skip_after_edit_discards_generated_reasoning_and_context(manual_setup):
    flags, path, calls, proposals, _ = manual_setup
    result = CliRunner().invoke(
        command,
        flags,
        input="gateway/wired-model\nchange\n\n\nrecommended\nskip\n128000\n64000\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(proposals) == 3 and all(r.method == "GET" for r in calls)
    assert not proposals[1].get("reasoning_levels")  # Foreign metadata never invents controls.
    assert not proposals[2].get("reasoning_levels")
    assert "catalogue" not in proposals[2]["provenance"]
    assert proposals[2]["provenance"]["manual_limits"] == {
        "source": "user",
        "context_window": 128000,
        "max_input_tokens": 64000,
    }
    assert yaml.safe_load(path.read_text())["models"]["local"]["context_window"] == 64000


def test_guided_skip_after_edit_restores_explicit_context_flag_prefill(manual_setup):
    flags, _, calls, proposals, _ = manual_setup
    result = CliRunner().invoke(
        command,
        flags + ["--context-window", "48000"],
        input="gateway/wired-model\nchange\n96000\n\nrecommended\nskip\n24000\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    assert proposals[-1]["provenance"]["manual_limits"]["context_window"] == 48000
    assert proposals[-1]["context_window"] == 24000
    assert not proposals[-1].get("reasoning_levels")


def test_final_manual_reedit_can_increase_a_provisional_context_and_reply(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    mock_http(monkeypatch, lambda request: pytest.fail("Save without testing is offline"))
    path = tmp_path / "models.yaml"
    flags = [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-key-env",
        "",
        "--no-catalogue",
        "--media-inputs",
        "",
        "--reply-tokens",
        "1000",
        "--output",
        str(path),
    ]
    result = CliRunner().invoke(
        command,
        flags,
        input="8000\n64000\ndiscover\nchange\n128000\n64000\ncustom\n10000\nsave\nchat\ny\n",
    )
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert saved["context_window"] == 64000  # Not the stale provisional 8000.
    assert saved["max_tokens"] == 10000
    output = " ".join(result.output.split())
    assert "Context window (declared): 128,000 tokens" in output
    assert "Input token limit (declared): 64,000 tokens" in output
    assert "Effective context bound: 64,000 tokens" in output
