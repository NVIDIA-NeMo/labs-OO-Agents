# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual guided CLI ordering, provisional chat limits and exact native contracts."""

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
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    return tmp_path / "models.yaml"


def flags(path):
    return [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-key-env",
        "",
        "--probe",
        "minimal",
        "--reply-tokens",
        "1000",
        "--output",
        str(path),
    ]


def http_setup(monkeypatch, *, public=None, endpoint=None):
    calls = []
    if public is None:
        public = [{"id": "gateway/wired-model", "context_length": 128000}]
    if endpoint is None:
        endpoint = [{"id": "wire/model"}]

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": public if request.url.host == "openrouter.ai" else endpoint,
                },
            )
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json=response_body("chat"))
        if request.url.path == "/exact-decision":
            return httpx.Response(200, json={"answers": {"supported": {"noul": 0.8}}})
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    return calls


@pytest.mark.parametrize("route", ["fuzzy-blank", "ambiguous-blank", "no-catalogue", "no-metadata"])
def test_initial_manual_pair_precedes_one_details_interface_media_and_consent(
    monkeypatch, setup, route
):
    public = [{"id": "wire/model"}, {"id": "wire/model"}] if route == "ambiguous-blank" else None
    calls = http_setup(monkeypatch, public=public)
    if route == "ambiguous-blank":

        async def catalogue():
            return public

        monkeypatch.setattr(connect, "catalogue", catalogue)
    consent = prompts.labelled_choice
    reviews = []

    def choose(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            assert all(r.method == "GET" for r in calls)
            reviews.append(label)
        return consent(label, options, **kwargs)

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    options = flags(setup) + (["--" + route] if route.startswith("no-") else [])
    result = CliRunner().invoke(
        command,
        options,
        input=(("\n" if route.endswith("blank") else "") + "128000\n64000\nchat\n-\ntest\ny\n"),
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    labels = [
        "Context window (tokens):",
        "Input token limit (tokens):",
        "Model details Model:",
        "Request interface [",
        "Media setup",
        "Would you like to test these settings?",
    ]
    positions = [text.index(label) for label in labels]
    assert positions == sorted(positions)
    assert text.count("Model details Model:") == 1
    assert text.count("Context window (tokens):") == text.count("Input token limit (tokens):") == 1
    initial = text[positions[2] : positions[3]]
    assert "Source: Your settings" in initial
    assert initial.count("Context window (declared):") == 1
    assert initial.count("Input token limit (declared):") == 1
    assert "Effective context bound: 64,000 tokens" in initial
    assert len(reviews) == 1 and sum(r.method == "POST" for r in calls) == 1
    saved = yaml.safe_load(setup.read_text())["models"]["local"]
    assert saved["context_window"] == 64000 and saved["max_tokens"] == 1000
    assert "max_input_tokens" not in saved


@pytest.mark.parametrize("answers", ["", "128000\n"])
def test_initial_cancellation_has_no_details_interface_media_post_or_save(
    monkeypatch, setup, answers
):
    calls = http_setup(monkeypatch)
    result = CliRunner().invoke(command, flags(setup), input="\n" + answers)
    assert result.exit_code == 1, result.output
    assert not setup.exists() and all(r.method == "GET" for r in calls)
    assert "Request interface" not in result.output and "Media setup" not in result.output
    assert "Model details" not in result.output


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_later_decision_discards_provisional_pair_and_context_provenance(
    monkeypatch, setup, deferred, prefill
):
    calls = http_setup(monkeypatch, public=[])
    states = []
    save = wizard.save_model

    def capture(state):
        states.append(state)
        return save(state)

    monkeypatch.setattr(wizard, "save_model", capture)
    options = flags(setup)
    at = options.index("--reply-tokens")
    del options[at : at + 2]
    options += ["--no-catalogue", "--media-inputs", ""]
    if deferred:
        options += ["--systemone-endpoint", "https://api.test/exact-decision"]
        answers = "128000\n64000\ndiscover\nsystemone\ntest\ny\n"
    else:
        options[options.index("--endpoint") + 1] = "https://api.test/exact-decision"
        answers = "128000\n64000\nsystemone\ntest\ny\n"
    if prefill:
        options += ["--context-window", "128000"]
        answers = answers.removeprefix("128000\n")
    result = CliRunner().invoke(command, options, input=answers)
    assert result.exit_code == 0, result.output
    state = states[-1]
    assert state.manual_limits is None and state.context_window is None
    assert not state.manual_limits_required and not state.manual_limits_deferred
    entry = state.proposal.entry
    assert entry["client_type"] == "decision"
    for key in ("context_window", "manual_limits", "max_tokens", "max_input_tokens"):
        assert key not in entry and key not in entry["provenance"]
    saved = yaml.safe_load(setup.read_text())["models"]["local"]
    assert not any(key in saved for key in ("context_window", "max_tokens", "max_input_tokens"))
    decision = [r for r in calls if r.url.path == "/exact-decision"]
    assert len(decision) == 1
    assert set(json.loads(decision[0].content)) == {"model", "state", "questions"}
    assert result.output.count("Context window (tokens)") == (0 if prefill else 1)
    # Discover now runs interface checks before the single proposal review.
    assert result.output.count("Would you like to test these settings?") == 1


@pytest.mark.parametrize("type_fact", [{"model_type": "decision"}, {"mode": "evaluation"}])
def test_exact_native_type_bypasses_manual_pair(monkeypatch, setup, type_fact):
    http_setup(monkeypatch, public=[], endpoint=[{"id": "wire/model", **type_fact}])
    if "mode" in type_fact:

        async def details(endpoint, model, **kwargs):
            return connect.ModelDetails(
                model,
                {"id": model, **type_fact},
                records=(
                    {
                        "id": model,
                        "schema": "models",
                        "source": endpoint + "/models",
                        "facts": type_fact,
                    },
                ),
            )

        monkeypatch.setattr(connect, "server_details", details)
    options = flags(setup)
    at = options.index("--reply-tokens")
    del options[at : at + 2]
    options += ["--media-inputs", "", "--no-probe"]
    result = CliRunner().invoke(command, options, input="systemone\nsave\ny\n")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "Model type: Decision" in text and "Source: This server" in text
    assert "Context window (tokens)" not in result.output
    assert "Input token limit (tokens)" not in result.output
    assert "context_window" not in yaml.safe_load(setup.read_text())["models"]["local"]


def test_exact_server_bound_bypasses_manual_pair_before_unknown_interface(monkeypatch, setup):
    calls = http_setup(
        monkeypatch,
        public=[],
        endpoint=[
            {
                "id": "wire/model",
                "context_window": 128000,
                "max_input_tokens": 64000,
            }
        ],
    )
    result = CliRunner().invoke(command, flags(setup) + ["--no-probe"], input="chat\n-\nsave\ny\n")
    assert result.exit_code == 0, result.output
    assert "Context window (tokens)" not in result.output
    assert "Input token limit (tokens)" not in result.output
    assert all(r.method == "GET" for r in calls)
    assert yaml.safe_load(setup.read_text())["models"]["local"]["context_window"] == 64000


@pytest.mark.parametrize("native", ["provider", "endpoint"])
@pytest.mark.parametrize("override", [None, "chat"])
def test_native_anthropic_defaults_without_interface_offer_or_post(
    monkeypatch, setup, native, override
):
    calls = http_setup(
        monkeypatch, public=[], endpoint=[{"id": "wire/model", "max_input_tokens": 64000}]
    )
    options = flags(setup) + ["--no-probe"]
    if native == "provider":
        options += ["--provider", "anthropic"]
        at = options.index("--endpoint")
        del options[at : at + 2]
    else:
        options[options.index("--endpoint") + 1] = connect.PROVIDERS["anthropic"].api_base
    if override:
        options += ["--api-style", override]
    result = CliRunner().invoke(command, options, input="-\nsave\ny\n")
    assert result.exit_code == 0, result.output
    assert "Request interface" not in result.output and "Find what works" not in result.output
    assert all(r.method == "GET" for r in calls)
    assert result.output.index("Model details") < result.output.index("Media setup")
    entry = yaml.safe_load(setup.read_text())["models"]["local"]
    assert entry["api_style"] == (override or "anthropic")


@pytest.mark.parametrize("route", ["provider", "endpoint", "proxy", "other-provider"])
def test_exact_openai_namespace_is_supplementary_not_wire_or_server_metadata(
    monkeypatch, setup, route
):
    calls = http_setup(
        monkeypatch,
        public=[{"id": "openai/gpt-6-luna", "context_length": 64000}],
        endpoint=[{"id": "gpt-6-luna"}],
    )
    options = flags(setup)
    options[0] = "gpt-6-luna"
    options += ["--no-probe", "--media-inputs", ""]
    if route == "provider":
        options += ["--provider", "openai"]
    elif route == "endpoint":
        options[options.index("--endpoint") + 1] = connect.PROVIDERS["openai"].api_base
    elif route == "other-provider":
        options += ["--provider", "nvidia"]
    native = route in {"provider", "endpoint"}
    result = CliRunner().invoke(
        command, options, input=("" if native else "\n128000\n64000\n") + "chat\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert ("did you mean" not in result.output) == native
    assert all(r.method == "GET" for r in calls)
    saved = yaml.safe_load(setup.read_text())["models"]["local"]
    assert saved["model_name"] == "openai/gpt-6-luna"
    assert saved["context_window"] == 64000
    if native:
        assert "Source: OpenRouter" in " ".join(result.output.split())
        assert "Source: This server" not in " ".join(result.output.split())


@pytest.mark.parametrize("mapping", [None, "vendor/explicit"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_openai_namespace_respects_explicit_mapping_direct_collision_and_duplicates(
    monkeypatch, setup, mapping, duplicate
):
    public = [{"id": "openai/gpt-6-luna", "context_length": 64000}]
    if mapping:
        public.append({"id": mapping, "context_length": 32000})
    elif not duplicate:
        public.append({"id": "gpt-6-luna", "context_length": 48000})
    if duplicate:
        public += deepcopy(public)
    calls = http_setup(monkeypatch, public=public, endpoint=[{"id": "gpt-6-luna"}])
    if duplicate:

        async def catalogue():
            return public

        monkeypatch.setattr(connect, "catalogue", catalogue)
    options = flags(setup) + ["--provider", "openai", "--no-probe", "--media-inputs", ""]
    options[0] = "gpt-6-luna"
    if mapping:
        options += ["--catalogue-model", mapping]
    result = CliRunner().invoke(
        command,
        options,
        input=(((mapping or "openai/gpt-6-luna") + "\n") if duplicate else "") + "chat\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "did you mean" not in result.output
    assert ("Possible catalogue models" in result.output) == duplicate
    assert all(r.method == "GET" for r in calls)
    entry = yaml.safe_load(setup.read_text())["models"]["local"]
    assert entry["context_window"] == (32000 if mapping else 64000 if duplicate else 48000)
    assert entry["model_name"] == "openai/gpt-6-luna"


@pytest.mark.parametrize("fact", [{"model_type": "decision"}, {"mode": "evaluation"}])
def test_completion_override_of_known_type_allows_initial_limits_and_reedit(
    monkeypatch, setup, fact
):
    calls = http_setup(monkeypatch, public=[], endpoint=[{"id": "wire/model", **fact}])
    state = wizard.WizardState(
        model="wire/model",
        endpoint="https://api.test/v1",
        api_style="chat",
        no_catalogue=True,
        yes=False,
        context_window=None,
    )
    state.selected_endpoint_model = {"id": "wire/model", **fact}
    answers = iter(["128000", "64000", "48000", "24000"])
    monkeypatch.setattr(prompts, "prompt", lambda *a, **k: next(answers))
    wizard.require_manual_limits(state)
    assert state.context_window == 64000
    wizard.require_manual_limits(state, edit=True)
    assert state.manual_limits == {"context_window": 48000, "max_input_tokens": 24000}
    assert state.context_window == 24000 and not calls


@pytest.mark.parametrize("native", ["anthropic", "openai"])
def test_native_contracts_preserve_headless_no_mandatory_manual_flags(monkeypatch, setup, native):
    calls = http_setup(
        monkeypatch,
        public=[{"id": "openai/gpt-6-luna", "context_length": 64000}],
        endpoint=[{"id": "gpt-6-luna"}],
    )
    options = flags(setup)
    options[0] = "gpt-6-luna"
    options += ["--provider", native, "--no-probe", "--media-inputs", "", "--yes"]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 0, result.output
    assert (
        "Context window (tokens)" not in result.output and "Request interface" not in result.output
    )
    assert all(r.method == "GET" for r in calls)
    entry = yaml.safe_load(setup.read_text())["models"]["local"]
    assert entry["api_style"] == connect.PROVIDERS[native].api_style
    if native == "openai":
        assert entry["context_window"] == 64000


def test_real_public_duplicate_catalogue_is_rejected_not_auto_selected(monkeypatch, setup):
    calls = http_setup(
        monkeypatch,
        public=[
            {"id": "openai/gpt-6-luna", "context_length": 64000},
            {"id": "openai/gpt-6-luna", "context_length": 32000},
        ],
        endpoint=[{"id": "gpt-6-luna"}],
    )
    options = flags(setup)
    options[0] = "gpt-6-luna"
    options += ["--provider", "openai", "--no-probe", "--media-inputs", ""]
    result = CliRunner().invoke(command, options, input="chat\nsave\ny\n")
    assert result.exit_code == 0, result.output
    assert "OpenRouter: Unavailable" in " ".join(result.output.split())
    assert "did you mean" not in result.output and "Possible catalogue models" not in result.output
    assert all(r.method == "GET" for r in calls)
    entry = yaml.safe_load(setup.read_text())["models"]["local"]
    assert entry.get("context_window") not in {64000, 32000}


def test_openai_provider_discovery_also_checks_openai_decisions(monkeypatch, setup):
    """Find what works on OpenAI includes its /v1/decisions route for decision models."""
    http_setup(
        monkeypatch,
        public=[{"id": "openai/gpt-6-luna", "context_length": 64000}],
        endpoint=[{"id": "gpt-6-luna"}],
    )
    seen = {}

    def capture(*args, styles=None, endpoints=None, **kwargs):
        seen.update(styles=tuple(styles), endpoints=dict(endpoints))
        raise KeyboardInterrupt

    monkeypatch.setattr(connect, "check_interfaces", capture)
    options = flags(setup)
    options[0] = "gpt-6-luna"
    options += ["--provider", "openai", "--media-inputs", ""]
    CliRunner().invoke(command, options, input="discover\n")
    assert "openai" in seen["styles"]
    assert seen["endpoints"]["openai"] == "https://api.openai.com/v1/decisions"


def test_media_the_interface_cannot_send_needs_no_diagnosis():
    assert wizard.media_cannot_apply({"implementation_supported": False})
    assert wizard.media_cannot_apply({"reason": "Unsupported media input for responses: audio"})
    assert wizard.media_cannot_apply({"reason": "Video testing not implemented for this route"})
    assert not wizard.media_cannot_apply({"reason": "not approved"})
    assert not wizard.media_cannot_apply({"outcome": "failed", "reason": "HTTP 400"})
