# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model selection precedes inference choice; discovery never guesses custom decision routes."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body

BASE = "https://mixed.test/v1"
DECISION = "https://mixed.test/custom/exact-decision"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(wizard, "entries", lambda *a, **kw: {})
    monkeypatch.setattr(wizard, "shadowing_source", lambda *a, **kw: None)
    monkeypatch.setattr(prompts, "choose_reply_limit", lambda suggested, ceiling, **kw: suggested)
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(), **kw: list(current))
    monkeypatch.setattr(prompts, "choose_media_mode", lambda **kw: "manual")


def options(path):
    return [
        "exact/model",
        "--as",
        "local",
        "--endpoint",
        BASE,
        "--api-key-env",
        "",
        "--no-catalogue",
        "--output",
        str(path),
    ]


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic", "systemone", "openai"])
def test_known_interface_after_model_skips_comparison(monkeypatch, tmp_path, style):
    path = tmp_path / "models.yaml"
    requests = []
    menu_seen = []
    real_prompt = prompts.prompt

    def prompt(text, **kwargs):
        menu_seen.append(text)
        if text == "Request interface":
            assert kwargs["choices"] == (
                "discover",
                "chat",
                "responses",
                "anthropic",
                "systemone",
                "openai",
            )
            assert kwargs["default"] == "discover"
            assert kwargs["open_menu"] is True
        return real_prompt(text, **kwargs)

    def handle(request):
        requests.append(request)
        assert request.method == "POST"
        if style in connect.DECISION_API_STYLES:
            assert str(request.url) == BASE
            return httpx.Response(
                200,
                json=(
                    {"answers": [{"type": "predicate", "probability": 0.8}]}
                    if style == "openai"
                    else {"answers": {"supported": {"noul": 0.8}}}
                ),
            )
        suffix = {"chat": "/chat/completions", "responses": "/responses", "anthropic": "/messages"}
        assert request.url.path.endswith(suffix[style])
        return httpx.Response(200, json=response_body(style))

    monkeypatch.setattr(prompts, "prompt", prompt)
    if style in connect.DECISION_API_STYLES:
        monkeypatch.setattr(
            prompts, "choose_media_mode", lambda **kw: pytest.fail("No decision media menu")
        )
        monkeypatch.setattr(
            prompts, "choose_media_inputs", lambda *a, **kw: pytest.fail("No decision media menu")
        )
    monkeypatch.setattr(connect, "check_interfaces", lambda *a, **kw: pytest.fail("No comparison"))
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [*options(path), "--probe", "minimal"],
        input=f"262144\n262144\n{style}\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert menu_seen == [
        "Context window (tokens)",
        "Input token limit (tokens)",
        "Request interface",
    ]
    assert len(requests) == 1
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["api_style"] == style
    assert "Working API formats" not in result.output
    assert "API format [" not in result.output


@pytest.mark.parametrize("mode", ["explicit", "edit", "yes"])
def test_explicit_edit_and_yes_bypass_interface_menu(monkeypatch, tmp_path, mode):
    path = tmp_path / "models.yaml"
    real_prompt = prompts.prompt

    def prompt(label, **kwargs):
        assert label not in {"Request interface", "API format"}
        return real_prompt(label, **kwargs)

    monkeypatch.setattr(prompts, "prompt", prompt)
    mock_http(monkeypatch, lambda request: pytest.fail("Manual exact-ID setup is offline"))
    flags = [*options(path), "--no-probe"]
    if mode == "explicit":
        flags += ["--api-style", "chat"]
    elif mode == "yes":
        # Provider fills the known default without asking for an inference interface.
        flags += ["--provider", "nvidia", "--yes"]
    else:
        entry = {"model_name": "openai/exact/model", "api_base": BASE, "api_style": "chat"}
        path.write_text(yaml.safe_dump({"models": {"local": entry}}))
        monkeypatch.setattr(wizard, "entries", lambda *a, **kw: {"local": (entry, path)})
        monkeypatch.setattr(prompts, "edit_model_details", lambda current: current)
        flags = ["--edit-model", "local", "--no-probe", "--api-key-env", "", "--output", str(path)]
    result = CliRunner().invoke(
        command, flags, input="save\ny\ny\n" if mode == "edit" else "262144\n262144\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert "Request interface" not in result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["api_style"] == "chat"


def test_menu_order_provider_url_key_model_interface(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    sequence = []
    real_prompt = prompts.prompt

    def prompt(text, **kwargs):
        sequence.append(text)
        return real_prompt(text, **kwargs)

    monkeypatch.setattr(prompts, "prompt", prompt)
    mock_http(
        monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "exact/model"}]})
    )
    result = CliRunner().invoke(
        command,
        ["--as", "local", "--no-probe", "--no-catalogue", "--output", str(path)],
        input=f"custom\n{BASE}\n-\nexact/model\n262144\n262144\nchat\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert sequence[:7] == [
        "Choose a provider",
        "Model server URL",
        "Key environment variable (new to enter a key; - for no authentication)",
        "Search available models (type to search; Tab to select)",
        "Context window (tokens)",
        "Input token limit (tokens)",
        "Request interface",
    ]
    assert "API format" not in sequence


@pytest.mark.parametrize("choice", ["systemone", "discover"])
def test_hub_shared_listing_and_default_full_systemone_url(monkeypatch, tmp_path, choice):
    path = tmp_path / "models.yaml"
    requests = []
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "hub-secret")

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer hub-secret"
        if request.method == "GET":
            assert str(request.url) == "https://inference-api.nvidia.com/v1/models"
            return httpx.Response(200, json={"data": [{"id": "exact/model"}]})
        if request.url.path == "/v1/systemone":
            return httpx.Response(
                200,
                json={
                    "answers": {"supported": {"noul": 0.8}},
                    "choices": [{"message": {"content": '{"supported": true}'}}],
                },
            )
        assert request.url.path in {"/v1/chat/completions", "/v1/responses", "/v1/messages"}
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "--provider",
            "hub",
            "--as",
            "local",
            "--no-catalogue",
            "--probe",
            "minimal",
            "--output",
            str(path),
        ],
        input=f"exact/model\n262144\n262144\n{choice}\ntest\n"
        + ("test\n" if choice == "discover" else "")
        + "y\n",
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["endpoint"] == "https://inference-api.nvidia.com/v1/systemone"
    assert entry["api_style"] == "systemone"
    assert sum(r.method == "GET" for r in requests) == 1
    assert sum(r.url.path == "/v1/systemone" for r in requests) == 1
    assert "hub-secret" not in result.output + path.read_text()


@pytest.mark.parametrize("candidate", [None, DECISION])
def test_custom_discovery_never_guesses_decision_url_and_reuses_selected_request(
    monkeypatch, tmp_path, candidate
):
    path = tmp_path / "models.yaml"
    requests = []
    reviewed = []
    original_write = connect.write

    def write(entry, target, *, alias, **kwargs):
        reviewed.append(entry)
        original_write(entry, target, alias=alias, **kwargs)

    def handle(request):
        requests.append(request)
        if str(request.url) == DECISION:
            body = json.loads(request.content)
            assert set(body) == {"model", "state", "questions"}
            assert body["model"] == "exact/model"
            return httpx.Response(
                200,
                json={
                    "answers": {"supported": {"noul": 0.8}},
                    "usage": {"input_tokens": 14, "output_tokens": 0},
                },
            )
        assert request.url.host == "mixed.test"
        assert request.url.path in {"/v1/chat/completions", "/v1/responses", "/v1/messages"}
        if candidate is None and request.url.path.endswith("chat/completions"):
            return httpx.Response(
                200,
                json=response_body(
                    "chat",
                    '{"value":323}' if "response_format" in json.loads(request.content) else "323",
                ),
            )
        return httpx.Response(404)

    monkeypatch.setattr(connect, "write", write)
    mock_http(monkeypatch, handle)
    flags = [*options(path), "--probe", "minimal", "--budget-tokens", "9000"]
    if candidate is None:
        flags += ["--reply-tokens", "200"]
    if candidate:
        flags += ["--systemone-endpoint", candidate]
    result = CliRunner().invoke(
        command,
        flags,
        input="262144\n262144\ndiscover\ntest\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert path.exists(), result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    if candidate:
        assert entry["endpoint"] == DECISION
        assert entry["api_style"] == "systemone"
        assert len([r for r in requests if str(r.url) == DECISION]) == 1
        routing = reviewed[0]["provenance"]["probes"]["routing"]
        assert routing["outcome"] == "accepted"
        assert reviewed[0]["provenance"]["tokens_charged_to_budget"] <= 9000
    else:
        assert entry["api_base"] == BASE
        assert entry["api_style"] == "chat"
        assert all("systemone" not in str(r.url) for r in requests)


@pytest.mark.parametrize("flag", ["--media-inputs", "--test-media-inputs"])
@pytest.mark.parametrize("style", ["systemone", "openai"])
def test_decision_media_has_clear_no_transport_error(monkeypatch, tmp_path, flag, style):
    path = tmp_path / "models.yaml"
    mock_http(monkeypatch, lambda request: pytest.fail("No decision media transport"))
    result = CliRunner().invoke(
        command,
        [
            *options(path),
            "--api-style",
            style,
            *(["--no-probe"] if flag == "--media-inputs" else []),
            flag,
            "audio",
        ],
        input="test\n",
    )
    assert result.exit_code == 2, result.output
    assert "currently supports image attachments only. Use media_inputs: [image]" in " ".join(
        result.output.split()
    )
    assert not path.exists()


@pytest.mark.parametrize("budget", [1, 511])
def test_known_decision_cannot_save_without_budgeted_baseline(monkeypatch, tmp_path, budget):
    path = tmp_path / "models.yaml"
    mock_http(monkeypatch, lambda request: pytest.fail("Budget must prevent inference"))
    result = CliRunner().invoke(
        command,
        [*options(path), "--api-style", "systemone", "--budget-tokens", str(budget)],
        input="test\n",
    )
    assert result.exit_code != 0, result.output
    assert "Nothing was saved" in result.output
    assert "check budget used up" in result.output
    assert not path.exists()


@pytest.mark.parametrize("style", ["chat", "responses", "systemone", "openai"])
def test_interfaces_stage_explicit_style_checks_only_that_style(monkeypatch, style):
    requests = []

    def handle(request):
        requests.append(request)
        if style in connect.DECISION_API_STYLES:
            assert str(request.url) == DECISION
            return httpx.Response(
                200,
                json=(
                    {"answers": [{"type": "predicate", "probability": 0.8}]}
                    if style == "openai"
                    else {"answers": {"supported": {"noul": 0.8}}}
                ),
            )
        assert request.url.path.endswith("/chat/completions" if style == "chat" else "/responses")
        return httpx.Response(200, json=response_body(style))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "exact/model",
            "--stage",
            "interfaces",
            "--endpoint",
            DECISION if style in connect.DECISION_API_STYLES else BASE,
            "--api-style",
            style,
            "--api-key-env",
            "",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert data["accepted"] == [style]
    assert set(data["results"]) == {style}
    assert len(requests) == 1


@pytest.mark.parametrize("style", ["systemone", "openai"])
def test_rejected_decision_key_does_not_save_or_repeat_request(monkeypatch, tmp_path, style):
    path = tmp_path / "models.yaml"
    monkeypatch.setenv("INTERFACE_KEY", "decision-secret-canary")
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer decision-secret-canary"
        return httpx.Response(401, json={"error": "decision-secret-canary rejected"})

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            *options(path),
            "--api-style",
            style,
            "--api-key-env",
            "INTERFACE_KEY",
            "--probe",
            "minimal",
            "--show-diagnostics",
        ],
        input="test\n",
    )
    assert result.exit_code != 0, result.output
    assert len(requests) == 1
    assert "Nothing was saved" in result.output
    assert "decision-secret-canary" not in result.output
    assert not path.exists()


def test_user_selects_decision_among_multiple_successes_and_reuses_full_route(
    monkeypatch, tmp_path
):
    path = tmp_path / "models.yaml"
    requests = []
    reviewed = []
    original_write = connect.write

    def write(entry, target, *, alias, **kwargs):
        reviewed.append(entry)
        original_write(entry, target, alias=alias, **kwargs)

    def handle(request):
        requests.append(request)
        if str(request.url) == DECISION:
            return httpx.Response(200, json={"answers": {"supported": {"noul": 0.8}}})
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json=response_body("chat", "323"))
        assert request.url.path in {"/v1/responses", "/v1/messages"}
        return httpx.Response(404)

    monkeypatch.setattr(connect, "write", write)
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            *options(path),
            "--systemone-endpoint",
            DECISION,
            "--probe",
            "minimal",
            "--budget-tokens",
            "20000",
        ],
        input="262144\n262144\ndiscover\ntest\nsystemone\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "Working API formats: OpenAI-compatible Chat, System One Decisions" in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["api_style"] == "systemone"
    assert entry["endpoint"] == DECISION
    assert "api_base" not in entry
    assert len([request for request in requests if str(request.url) == DECISION]) == 1
    provenance = reviewed[0]["provenance"]
    assert provenance["tokens_charged_to_budget"] == (
        3 * (connect.DISCOVERY_OUTPUT_TOKENS + 512) + 4608 + 512
    )
    assert provenance["probes"]["routing"]["outcome"] == "accepted"
    assert set(provenance["interfaces"]) == {"chat", "responses", "anthropic", "systemone"}


@pytest.mark.parametrize("opt_in", [False, True])
def test_custom_cross_origin_interface_key_requires_explicit_flag(monkeypatch, tmp_path, opt_in):
    path = tmp_path / "models.yaml"
    requests = []
    decision = "https://other.test/exact-decision"
    monkeypatch.setenv("INTERFACE_KEY", "interface-secret-canary")

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer interface-secret-canary"
        if str(request.url) == decision:
            return httpx.Response(200, json={"answers": {"supported": {"noul": 0.8}}})
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    flags = [
        *options(path),
        "--api-key-env",
        "INTERFACE_KEY",
        "--systemone-endpoint",
        decision,
        "--probe",
        "minimal",
    ]
    if opt_in:
        flags += ["--share-interface-key"]
    result = CliRunner().invoke(command, flags, input="262144\n262144\ndiscover\ntest\ntest\ny\n")
    assert "interface-secret-canary" not in result.output
    if opt_in:
        assert result.exit_code == 0, result.output
        assert len(requests) == 4
        assert yaml.safe_load(path.read_text())["models"]["local"]["endpoint"] == decision
    else:
        assert result.exit_code != 0
        assert "--share-interface-key" in result.output
        assert requests == []
        assert not path.exists()


def test_hub_overridden_chat_base_refuses_auto_mapped_key(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "hub-secret-canary")
    mock_http(monkeypatch, lambda request: pytest.fail("No HTTP before cross-origin opt-in"))
    result = CliRunner().invoke(
        command,
        [
            "exact/model",
            "--as",
            "local",
            "--provider",
            "hub",
            "--endpoint",
            BASE,
            "--no-catalogue",
            "--probe",
            "minimal",
            "--output",
            str(path),
        ],
        input="262144\n262144\ndiscover\ntest\n",
    )
    assert result.exit_code != 0
    assert "--share-interface-key" in result.output
    assert "hub-secret-canary" not in result.output
    assert not path.exists()


@pytest.mark.parametrize("flag", ["--media-inputs", "--test-media-inputs"])
@pytest.mark.parametrize("style", ["systemone", "openai"])
def test_known_decision_missing_model_media_rejected_before_listing(
    monkeypatch, tmp_path, flag, style
):
    mock_http(monkeypatch, lambda request: pytest.fail("Must reject before listing"))
    result = CliRunner().invoke(
        command,
        [
            "--as",
            "local",
            "--endpoint",
            DECISION,
            "--models-url",
            BASE + "/models",
            "--api-style",
            style,
            "--api-key-env",
            "",
            flag,
            "audio",
            "--output",
            str(tmp_path / "models.yaml"),
        ],
        input="test\n",
    )
    assert result.exit_code == 2, result.output
    assert "currently supports image attachments only. Use media_inputs: [image]" in " ".join(
        result.output.split()
    )


@pytest.mark.parametrize("flag", ["--media-inputs", "--test-media-inputs"])
def test_discover_selected_decision_media_rejected_after_routing_only(monkeypatch, tmp_path, flag):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "exact/model"}]})
        body = json.loads(request.content)
        if str(request.url) == DECISION:
            assert set(body) == {"model", "state", "questions"}
            return httpx.Response(200, json={"answers": {"supported": {"noul": 0.8}}})
        assert not any(
            kind in request.content.decode() for kind in ["image_url", "input_audio", "data:image"]
        )
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "--as",
            "local",
            "--endpoint",
            BASE,
            "--systemone-endpoint",
            DECISION,
            "--api-key-env",
            "",
            "--no-catalogue",
            flag,
            "audio",
            "--output",
            str(path),
        ],
        input="exact/model\n262144\n262144\ndiscover\ntest\ncancel\n",
    )
    assert result.exit_code == 2, result.output
    assert "currently supports image attachments only. Use media_inputs: [image]" in " ".join(
        result.output.split()
    )
    assert len(requests) == 5
    assert requests[0].method == "GET"
    assert not path.exists()


@pytest.mark.parametrize("opt_in", [False, True])
def test_interfaces_stage_cross_origin_key_flag(monkeypatch, opt_in):
    requests = []
    monkeypatch.setenv("INTERFACE_KEY", "stage-secret-canary")

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer stage-secret-canary"
        if request.url.host == "other.test":
            return httpx.Response(200, json={"answers": {"supported": {"noul": 0.8}}})
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    flags = [
        "exact/model",
        "--stage",
        "interfaces",
        "--endpoint",
        BASE,
        "--systemone-endpoint",
        "https://other.test/full",
        "--api-key-env",
        "INTERFACE_KEY",
    ]
    if opt_in:
        flags += ["--share-interface-key"]
    result = CliRunner().invoke(command, flags)
    report = json.loads(result.stdout)
    assert "stage-secret-canary" not in result.output
    if opt_in:
        assert result.exit_code == 0, result.output
        assert report["data"]["accepted"] == ["systemone"]
        assert len(requests) == 4
    else:
        assert result.exit_code != 0
        assert "--share-interface-key" in report["error"]["message"]
        assert requests == []


@pytest.mark.parametrize("listing_url", [None, "https://catalogue.test/shared/models"])
@pytest.mark.parametrize("no_catalogue", [False, True])
def test_picker_discover_anthropic_retains_endpoint_limits(
    monkeypatch, tmp_path, listing_url, no_catalogue
):
    path = tmp_path / "models.yaml"
    requests = []
    reviewed = []
    original_write = connect.write

    def write(entry, target, *, alias, **kwargs):
        reviewed.append(entry)
        original_write(entry, target, alias=alias, **kwargs)

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            assert "authorization" not in request.headers
            assert "x-api-key" not in request.headers
            assert str(request.url) == (listing_url or BASE + "/models")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "exact/model",
                            "context_length": 4096,
                            "max_output_tokens": 128,
                        }
                    ]
                },
            )
        if request.url.path == "/v1/messages":
            return httpx.Response(200, json=response_body("anthropic"))
        assert request.url.path in {"/v1/chat/completions", "/v1/responses"}
        return httpx.Response(404)

    monkeypatch.setattr(connect, "write", write)
    mock_http(monkeypatch, handle)

    async def catalogue():
        return [
            {
                "id": "exact/model",
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 32768},
            }
        ]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    flags = [
        "--as",
        "local",
        "--endpoint",
        BASE,
        "--api-key-env",
        "",
        "--probe",
        "minimal",
        "--budget-tokens",
        "9000",
        "--output",
        str(path),
    ]
    if no_catalogue:
        flags += ["--no-catalogue"]
    if listing_url:
        flags += ["--models-url", listing_url, "--discovery-api-key-env", "-"]
    result = CliRunner().invoke(command, flags, input="exact/model\ndiscover\ntest\ntest\ny\n")
    assert result.exit_code == 0, result.output
    assert "Found 1 models" in result.output
    assert path.exists(), result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["api_style"] == "anthropic"
    assert entry["api_base"] == "https://mixed.test"
    assert entry["context_window"] == 4096
    assert entry["max_tokens"] == 128
    assert reviewed[0]["provenance"]["endpoint_limits"]["context_window"] == 4096
    assert reviewed[0]["provenance"]["endpoint_limits"]["max_output_tokens"] == 128
    baseline = reviewed[0]["provenance"]["probes"]["routing"]
    assert baseline["configured_reply_tokens"] == 128
    assert baseline["verification_config"]["params"]["max_tokens"] == 128
    assert any(
        json.loads(r.content).get("max_tokens") == 128
        for r in requests
        if r.method == "POST" and r.url.path == "/v1/messages"
    )
    assert sum(r.method == "GET" for r in requests) == 1


@pytest.mark.parametrize("listing_url", [None, "https://catalogue.test/shared/models"])
@pytest.mark.parametrize(
    "style,selected_endpoint,retained",
    [
        ("anthropic", "https://mixed.test", True),
        ("anthropic", "https://mixed.test/other", False),
        ("chat", "https://mixed.test", False),
        ("responses", "https://mixed.test/other/v1", False),
        ("anthropic", "https://other.test", False),
    ],
)
def test_listing_association_only_follows_same_anthropic_route(
    monkeypatch, listing_url, style, selected_endpoint, retained
):
    selected = connect.plan("local", "exact/model", style, selected_endpoint, "").entry
    selected["provenance"]["probes"]["routing"] = {"outcome": "accepted"}
    interfaces = connect.InterfaceResult(
        results={style: connect.ConnectResult("local", selected)},
        tokens_charged_to_budget=0,
    )

    async def checks(*args, **kwargs):
        return interfaces

    monkeypatch.setattr(wizard, "show_checks", checks)
    monkeypatch.setattr(connect, "check_interfaces", lambda *a, **kw: None)
    state = wizard.WizardState(
        endpoint=BASE,
        discovery_endpoint=BASE,
        models_url=listing_url,
        endpoint_models=[{"id": "exact/model", "context_window": 4096, "max_output_tokens": 128}],
        approval="minimal",
        interface_endpoints={},
        output_tokens=200,
        model="exact/model",
        data={},
        no_catalogue=True,
        yes=True,
        budget_tokens=9000,
        test_media_inputs=None,
    )
    # Interface selection must never perform new catalogue/credential I/O.
    mock_http(monkeypatch, lambda request: pytest.fail("Cached metadata must stay offline"))
    assert wizard.check_interfaces(state)
    assert state.endpoint == selected_endpoint
    assert wizard.configure_metadata(state)
    assert (state.candidate is not None) is retained
    if retained:
        assert state.candidate["endpoint_limits"]["max_output_tokens"] == 128
