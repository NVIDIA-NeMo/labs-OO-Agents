# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real guided CLI consent ordering, with HTTP counters below the SDK."""

import json

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture
def guided(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    calls = []
    decisions = []
    original = prompts.labelled_choice

    def choose(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            if not decisions:
                assert calls and all(r.method == "GET" for r in calls)
            decisions.append(tuple(options))
        return original(label, options, **kwargs)

    monkeypatch.setattr(prompts, "labelled_choice", choose)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            public = request.url.host == "openrouter.ai"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "model",
                            "context_length": 128000 if public else 64000,
                            "top_provider": {"max_completion_tokens": 65536},
                            "architecture": {"input_modalities": ["text", "image"]},
                        }
                    ]
                },
            )
        assert decisions  # Model-call consent preceded every inference request.
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json=response_body("chat", '{"value":323}' if "response_format" in body else "323"),
            )
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    flags = [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-key-env",
        "",
        "--media-inputs",
        "",
        "--probe",
        "minimal",
        "--reply-tokens",
        "2000",
        "--output",
        str(path),
    ]
    return flags, path, calls, decisions


@pytest.mark.parametrize("action", ["test", "save", "2"])
def test_metadata_precedes_consent_and_save_without_testing_has_zero_post(guided, action):
    flags, path, calls, decisions = guided
    result = CliRunner().invoke(command, [*flags, "--api-style", "chat"], input=action + "\ny\n")
    assert result.exit_code == 0, result.output
    assert decisions == [
        (
            ("test", "Test now"),
            ("save", "Save without testing"),
            ("change", "Change settings"),
            ("skip", "Continue without catalogue settings"),
            ("structured", "Change answer format"),
            ("cancel", "Cancel without saving"),
        )
    ]
    assert [r.method for r in calls] == ["GET", "GET"] + (["POST"] if action == "test" else [])
    assert result.output.count("Proposed configuration") == 1
    assert "Approve setup" not in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["max_tokens"] == 2000
    assert entry["context_window"] == 64000  # Exact server facts outrank public metadata.
    assert entry["media_inputs"] == []  # Advertised image support is not permission.
    if action != "test":
        assert "not tested (manual setup)" in result.output


def test_change_then_test_uses_actual_configured_controls(guided):
    flags, path, calls, decisions = guided
    result = CliRunner().invoke(
        command,
        [*flags, "--api-style", "chat"],
        input="change\n48000\n8192\ncustom\n3000\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(decisions) == 2
    assert [r.method for r in calls] == ["GET", "GET", "POST"]
    body = json.loads(calls[-1].content)
    assert body["max_tokens"] == 3000
    assert "response_format" in body
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["max_tokens"] == 3000
    assert entry["max_tokens"] <= 8192
    assert entry["context_window"] == 48000
    assert entry["media_inputs"] == []
    assert "Model settings" not in result.output

    updated = result.output.index("Proposed configuration (updated)")
    consent = result.output.index("Would you like to test these settings?", updated)
    assert "48,000" in result.output[updated:consent] and "3,000" in result.output[updated:consent]


def test_deferred_style_tests_bounded_interfaces_after_metadata(guided):
    flags, path, calls, decisions = guided
    result = CliRunner().invoke(command, flags, input="discover\ntest\ntest\ny\n")
    assert result.exit_code == 0, result.output
    assert len(decisions) == 2
    posts = [r for r in calls if r.method == "POST"]
    assert [r.method for r in calls[:2]] == ["GET", "GET"]
    assert {r.url.path for r in posts} == {"/v1/chat/completions", "/v1/responses", "/v1/messages"}
    assert len(posts) == 5  # Three bounded routing probes, chat reasoning, and typed baseline.
    for request in posts[:-1]:
        cap = json.loads(request.content).get(
            "max_tokens", json.loads(request.content).get("max_output_tokens")
        )
        assert cap <= connect.DEFAULT_REASONING_OUTPUT_TOKENS
    assert json.loads(posts[-1].content)["max_tokens"] == 2000
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["api_style"] == "chat"
    assert entry["max_tokens"] == 2000


def test_deferred_style_manual_save_selects_style_without_invented_evidence(guided):
    flags, path, calls, decisions = guided
    result = CliRunner().invoke(command, flags, input="discover\nsave\nresponses\nsave\ny\n")
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    assert "Working API formats" not in result.output
    assert "not tested (manual setup)" in result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["api_style"] == "responses"


def test_no_probe_allows_metadata_but_never_offers_model_calls(guided):
    flags, path, calls, decisions = guided
    result = CliRunner().invoke(
        command, [*flags, "--api-style", "chat", "--no-probe"], input="save\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    assert decisions == [
        (
            ("save", "Save without testing"),
            ("change", "Change settings"),
            ("skip", "Continue without catalogue settings"),
            ("structured", "Change answer format"),
            ("cancel", "Cancel without saving"),
        )
    ]
    assert path.exists()


@pytest.mark.parametrize("yes", [False, True])
def test_exact_offline_edit_preserves_caps_media_and_custom_params(monkeypatch, tmp_path, yes):
    path = tmp_path / "models.yaml"
    original = {
        "model_name": "openai/model",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "api_key_env": "",
        "max_tokens": 9000,
        "context_window": 64000,
        "media_inputs": ["pdf"],
        "temperature": 0.2,
        "reasoning_levels": {"high": {"chat_template_kwargs": {"enable_thinking": True}}},
    }
    path.write_text(yaml.safe_dump({"models": {"local": original}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    mock_http(monkeypatch, lambda r: pytest.fail("Offline exact edit must not request anything"))
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "local",
            "--no-probe",
            "--media-inputs",
            "pdf",
            *(["--yes"] if yes else []),
        ],
        input="default\nsave\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    for key, value in original.items():
        assert entry[key] == value


@pytest.mark.parametrize("style", ["systemone", "openai"])
@pytest.mark.parametrize("action", ["save", "test"])
def test_exact_decision_metadata_is_get_only_before_consent_and_never_chat_params(
    monkeypatch, tmp_path, style, action
):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    calls, decisions, reviewed = [], [], []
    original_choice = prompts.labelled_choice
    original_write = connect.write

    def choose(label, options, **kwargs):
        if label == "Would you like to test these settings?":
            assert [r.method for r in calls] == ["GET", "GET"]
            decisions.append(options)
        return original_choice(label, options, **kwargs)

    def write(entry, target, **kwargs):
        reviewed.append(entry)
        original_write(entry, target, **kwargs)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            public = request.url.host == "openrouter.ai"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "decision-model",
                            "description": "public" if public else "server",
                            "context_length": 1,
                            "max_tokens": 1,
                            "supported_parameters": ["reasoning", "temperature"],
                            "architecture": {
                                "input_modalities": ["text", "image"] if public else ["text"]
                            },
                            "reasoning": {"supported_efforts": ["high"]},
                        }
                    ]
                },
            )
        assert decisions
        body = json.loads(request.content)
        assert body["model"] == "decision-model"
        assert not (
            {"max_tokens", "max_output_tokens", "reasoning", "reasoning_effort", "temperature"}
            & body.keys()
        )
        return httpx.Response(
            200,
            json=(
                {"answers": [{"type": "predicate", "probability": 0.8}]}
                if style == "openai"
                else {"answers": {"supported": {"noul": 0.8}}}
            ),
        )

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    monkeypatch.setattr(connect, "write", write)
    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "decision-model",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/exact-decision",
            "--api-style",
            style,
            "--models-url",
            "https://api.test/models",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--output",
            str(path),
        ],
        input=action + "\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert [r.method for r in calls] == ["GET", "GET"] + (["POST"] if action == "test" else [])
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["media_inputs"] == []
    assert not ({"max_tokens", "context_window", "reasoning_levels", "temperature"} & entry.keys())
    facts = reviewed[0]["provenance"]["advertised_model"]
    assert facts["description"] == "server"
    assert facts["media_source"] == "https://api.test/models"
    assert reviewed[0]["provenance"]["models_url"] == "https://api.test/models"


@pytest.mark.parametrize("no_probe", [False, True])
def test_yes_retains_explicit_approval_and_no_probe_overrides_it(monkeypatch, tmp_path, no_probe):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "model", "context_length": 64000}]})
        assert json.loads(request.content)["max_tokens"] == 3000
        return httpx.Response(200, json=response_body("chat", '{"value":323}'))

    mock_http(monkeypatch, handle)
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
            "chat",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--probe",
            "minimal",
            "--reply-tokens",
            "3000",
            "--yes",
            "--output",
            str(path),
            *(["--no-probe"] if no_probe else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Would you like to test these settings?" not in result.output
    assert [r.method for r in calls] == ["GET", "GET"] + ([] if no_probe else ["POST"])
    assert yaml.safe_load(path.read_text())["models"]["local"]["max_tokens"] == 3000


def test_change_preserves_explicit_level_request_blocks_caps_and_media(guided):
    flags, path, calls, decisions = guided
    levels = path.with_name("levels.yaml")
    levels.write_text(
        yaml.safe_dump(
            {
                "custom": {
                    "max_tokens": 7000,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            }
        )
    )
    result = CliRunner().invoke(
        command,
        [*flags, "--api-style", "chat", "--levels-file", str(levels), "--media-inputs", "pdf"],
        input="default\nchange\n\n\nrecommended\ndefault\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    assert len(decisions) == 2
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["max_tokens"] == 2000
    assert entry["media_inputs"] == ["pdf"]
    assert entry["reasoning_levels"]["custom"]["max_tokens"] == 7000
    assert entry["reasoning_levels"]["custom"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert result.output.count("Proposed configuration") == 2


def test_save_without_testing_discards_selected_media_tests_but_not_declarations(guided):
    flags, path, calls, decisions = guided
    # Existing permissions are declarations, not permission to run the selected tests.
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "local": {
                        "model_name": "openai/model",
                        "api_style": "chat",
                        "api_base": "https://api.test/v1",
                        "api_key_env": "",
                        "media_inputs": ["pdf"],
                        "max_tokens": 2000,
                    }
                }
            }
        )
    )
    index = flags.index("--media-inputs")
    flags = flags[:index] + flags[index + 2 :]
    result = CliRunner().invoke(
        command,
        [*flags, "--api-style", "chat", "--test-media-inputs", "image,pdf", "--probe", "all"],
        input="save\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert all(r.method == "GET" for r in calls)
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["pdf"]
    assert "not tested (manual setup)" in result.output


@pytest.mark.parametrize(
    "revision,finish,count,enabled",
    [
        ("off", "save", 4, []),
        ("image", "test", 9, ["image"]),
        ("cancel", "", 4, None),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_deferred_decision_revises_default_all_media_with_explicit_review(
    monkeypatch, tmp_path, revision, finish, count, enabled, explicit
):
    from tests.unifiedllm.connect.test_connect_decision_smoke import fixture_response

    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    calls, reviews = [], []
    original = prompts.labelled_choice

    def choose(label, options, **kwargs):
        if label == "Review media for the selected decision interface":
            assert len(calls) == 4  # Only bounded routing discovery has run.
            reviews.append(options)
        return original(label, options, **kwargs)

    def handle(request):
        calls.append(request)
        assert request.method == "POST"  # Exact --no-catalogue path is pure testing.
        if request.url.path == "/exact-decision":
            return fixture_response(request)
        assert b"input_audio" not in request.content and b"video_url" not in request.content
        return httpx.Response(404)

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    tail = "y\ny\n" if revision == "image" else ""
    if finish:
        tail += finish + "\n" + ("keep\n" if revision == "image" else "") + "y\n"
    result = CliRunner().invoke(
        command,
        [
            "fixture",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--systemone-endpoint",
            "https://api.test/exact-decision",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--budget-tokens",
            "100000",
            "--output",
            str(path),
            *(["--test-media-inputs", "image,audio,video,pdf"] if explicit else []),
        ],
        input="262144\n262144\ndiscover\n"
        + ("" if explicit else "test\n")
        + "test\n"
        + revision
        + "\n"
        + tail,
    )
    assert result.exit_code == (2 if explicit and revision == "cancel" else 0), result.output
    assert reviews and len(calls) == count
    assert path.exists() is (enabled is not None)
    if enabled is not None:
        entry = yaml.safe_load(path.read_text())["models"]["local"]
        assert entry["media_inputs"] == enabled
        updated = result.output.index("Proposed configuration (updated)")
        consent = result.output.index("Would you like to test these settings?", updated)
        proposal = result.output[updated:consent]
        assert "System One Decisions" in proposal
        assert "audio" not in proposal and "video" not in proposal and "pdf" not in proposal
        if revision == "image":
            assert "TypeSafe multipart" in proposal and "experimental images on" in proposal
            assert "Media samples: Images" in " ".join(proposal.split())
            assert "Images: Test before enabling" in " ".join(proposal.split())


def test_selected_typesafe_missing_optin_warning_is_not_json(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    mock_http(monkeypatch, lambda request: pytest.fail("Offline manual image declaration"))
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "fixture",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/exact",
            "--api-style",
            "systemone",
            "--api-key-env",
            "",
            "--media-inputs",
            "image",
            "--decision-image-transport",
            "typesafe_multipart",
            "--no-catalogue",
            "--no-probe",
            "--output",
            str(path),
        ],
        input="n\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "experimental image opt-in is missing" in result.output
    assert "Image testing isn't available with System One JSON" not in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["decision_image_transport"] == "typesafe_multipart"
    assert entry["typesafe_danger_zone_images"] is False


@pytest.mark.parametrize("previous", [[], ["image"]])
@pytest.mark.parametrize("retesting", [False, True])
def test_guided_decision_permission_change_supersedes_inheritance(
    monkeypatch, tmp_path, previous, retesting
):
    path = tmp_path / "models.yaml"
    original = {
        "model_name": "fixture",
        "client_type": "decision",
        "api_style": "openai",
        "endpoint": "https://api.test/decisions",
        "api_key_env": "",
        "media_inputs": list(previous),
        "decision_image_transport": "json",
        "typesafe_danger_zone_images": False,
    }
    path.write_text(yaml.safe_dump({"models": {"local": original}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    mock_http(
        monkeypatch, lambda request: pytest.fail("Offline permission edit must not send HTTP")
    )
    mode = "text" if previous else "manual"
    expected = [] if previous else ["image"]
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "local",
            "--no-catalogue",
            *(["--test-media-inputs", "image"] if retesting else ["--no-probe"]),
        ],
        input=f"change\n{mode}\nsave\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["media_inputs"] == expected
    assert entry["decision_image_transport"] == original["decision_image_transport"]
    assert entry["typesafe_danger_zone_images"] is False
    updated = result.output.index("Proposed configuration (updated)")
    consent = result.output.index("Would you like to test these settings?", updated)
    assert ("Images: Off" if previous else "Images: Enabled (not tested)") in " ".join(
        result.output[updated:consent].split()
    )
    assert "Media samples:" not in result.output[updated:consent]
    assert not original.get("provenance")


@pytest.mark.parametrize("public_details", [False, True])
def test_guided_source_uses_effective_facts_not_id_only_server(
    monkeypatch, tmp_path, public_details
):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        model = {"id": "model"}
        if request.url.host == "openrouter.ai" and public_details:
            model.update(context_length=64000, top_provider={"max_completion_tokens": 8192})
        return httpx.Response(200, json={"data": [model]})

    mock_http(monkeypatch, handle)
    reviewed = []
    original_write = connect.write

    def write(entry, target, **kwargs):
        reviewed.append(entry)
        original_write(entry, target, **kwargs)

    monkeypatch.setattr(connect, "write", write)
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
            "chat",
            "--api-key-env",
            "",
            "--media-inputs",
            "",
            "--output",
            str(path),
        ],
        input="save\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2 and all(request.method == "GET" for request in calls)
    start = result.output.index("Proposed configuration")
    end = result.output.index("Would you like to test these settings?", start)
    summary = result.output[start:end]
    assert "Model details from your server" not in summary
    assert ("Model details: OpenRouter" in " ".join(summary.split())) is public_details
    assert "Images: Off" in " ".join(summary.split())
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["media_inputs"] == []
    if public_details:
        assert entry["context_window"] == 64000
        assert reviewed[0]["provenance"]["limit_sources"]["context_length"] == "catalogue"
    else:
        assert "context_window" not in entry
