# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit suggestion choice and offline metadata-first headless setup."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

BASE = "https://api.test/v1"
META = {
    "id": "model",
    "architecture": {"input_modalities": ["image", "audio", "video", "file"]},
    "media_source": connect.CATALOGUE_URL,
}


def state(**kwargs):
    return wizard.WizardState(
        model="model",
        endpoint=BASE,
        api_style="chat",
        data={},
        approval="all",
        yes=False,
        media_hints=deepcopy(META),
        **kwargs,
    )


@pytest.mark.parametrize(
    "action,expected",
    [
        ("test", ["pdf"]),
        ("suggested", ["image", "audio", "video"]),
        ("manual", ["audio"]),
        ("none", []),
    ],
)
def test_explicit_suggestion_review_preserves_or_changes_only_on_choice(
    monkeypatch, action, expected
):
    previous = connect.plan("local", "model", "chat", BASE, "", media_inputs=["pdf"]).entry
    current = state(editing=previous)
    seen = []

    def choose(label, options, *, default):
        seen.append(options)
        assert default == "test"
        assert list(dict(options)) == ["test", "suggested", "manual", "none"]
        assert dict(options)["suggested"] == "Enable listed inputs without testing"
        return action

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda previous: ["audio"])
    assert wizard.configure_media_inputs(current)
    assert current.media_inputs == expected
    assert previous["media_inputs"] == ["pdf"]
    assert current.test_media_inputs == (["image", "audio", "video"] if action == "test" else None)
    assert len(seen) == 1


@pytest.mark.parametrize("choice", ["-", "pdf"])
def test_explicit_declaration_is_never_overwritten_or_prompted(monkeypatch, choice):
    current = state(media_inputs=choice)
    monkeypatch.setattr(
        prompts,
        "labelled_choice",
        lambda *a, **kw: pytest.fail("Explicit declaration must not prompt"),
    )
    assert wizard.configure_media_inputs(current)
    assert current.media_inputs == ([] if choice == "-" else ["pdf"])
    assert current.test_media_inputs is None


def test_yes_shows_suggestions_without_adopting_or_testing(monkeypatch, capsys):
    current = state()
    current.yes = True
    monkeypatch.setattr(
        prompts, "labelled_choice", lambda *a, **kw: pytest.fail("--yes must not adopt suggestions")
    )
    assert wizard.configure_media_inputs(current)
    assert current.media_inputs == [] and current.test_media_inputs is None
    output = capsys.readouterr().out
    assert "Source: OpenRouter" in " ".join(output.split())
    assert "Advertisements are unverified" not in output
    assert "PDF support isn't listed. Test a generated PDF or leave PDFs off." in output


@pytest.mark.parametrize("approval,no_probe", [("none", True), ("minimal", False), ("all", True)])
def test_optional_tests_require_wizard_approval(monkeypatch, approval, no_probe):
    current = state(no_probe=no_probe)
    current.approval = approval

    def choose(label, options, **kw):
        assert "test" not in dict(options)
        assert kw["default"] == "manual"
        return "none"

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    assert wizard.configure_media_inputs(current)
    assert current.test_media_inputs is None and current.media_inputs == []


def test_suggested_checks_are_selected_explicitly_and_unavailable_not_claimed(monkeypatch):
    current = state()
    current.api_style = "responses"

    def choose(label, options, **kw):
        assert dict(options)["test"] == "Enable listed inputs with test"
        return "test"

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    assert wizard.configure_media_inputs(current)
    assert current.test_media_inputs == ["image"]
    assert current.media_inputs == []  # No result yet; no permission yet.


def test_systemone_suggestions_do_not_guess_transport(monkeypatch, capsys):
    current = state(budget_tokens=9000)
    current.api_style = "systemone"

    def choose(label, options, **kw):
        assert "test" not in dict(options)
        assert dict(options)["suggested"] == "Enable listed inputs without testing"
        return "suggested"

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    monkeypatch.setattr(
        prompts, "confirm", lambda *a, **kw: pytest.fail("Metadata must not choose transport")
    )
    assert wizard.configure_media_inputs(current)
    assert current.media_inputs == ["image"]
    assert current.decision_image_transport == "json"
    assert current.typesafe_danger_zone_images is False
    assert current.test_media_inputs is None
    assert "Image testing isn't available for this connection" in " ".join(
        capsys.readouterr().out.split()
    )


def test_cached_metadata_only_model_does_not_add_settings_prompt(monkeypatch):
    current = state(no_catalogue=True)
    current.media_hints = None
    current.endpoint_models = [
        {**META, "media_source": BASE + "/models"},
        {"id": "neighbour", "input_modalities": ["pdf"]},
    ]
    current.discovery_endpoint = BASE
    mock_http(monkeypatch, lambda request: pytest.fail("Cached setup is offline"))
    assert wizard.prepare_metadata(current)
    assert current.candidate is None
    assert current.media_hints["architecture"] == META["architecture"]
    monkeypatch.setattr(prompts, "labelled_choice", lambda *a, **kw: "none")
    assert wizard.configure_media_inputs(current)
    monkeypatch.setattr(
        prompts, "prompt", lambda *a, **kw: pytest.fail("No new model settings prompt")
    )
    assert wizard.configure_metadata(current)


def test_changed_interface_does_not_use_other_endpoint_hints(monkeypatch):
    current = state(no_catalogue=True)
    current.endpoint_models = [META]
    current.discovery_endpoint = "https://other.test/v1"
    mock_http(monkeypatch, lambda request: pytest.fail("Unknown offline setup"))
    assert wizard.prepare_metadata(current)
    assert current.media_hints is None and current.selected_endpoint_model is None


def catalogue(path, *, full_report=True):
    model = {
        "id": "openai/model",
        "context_length": 16384,
        "top_provider": {"max_completion_tokens": 4096},
        "default_parameters": {"max_tokens": 9999, "temperature": 0.7, "unknown": 4},
        "supported_parameters": ["fake_param"],
        "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "low"},
        "architecture": {"input_modalities": ["image", "file"]},
        "pricing": {"prompt": "0.00001", "completion": "0.00002"},
    }
    data = {"source": connect.CATALOGUE_URL, "models": [model]}
    document = (
        {"version": 1, "stage": "catalogue", "ok": True, "data": data} if full_report else data
    )
    path.write_text(json.dumps(document))
    return path


def plan_args(path):
    return [
        "native-model",
        "--stage",
        "plan",
        "--as",
        "local",
        "--endpoint",
        BASE,
        "--api-style",
        "responses",
        "--api-key-env",
        "",
        "--catalogue-file",
        str(path),
        "--catalogue-model",
        "openai/model",
    ]


@pytest.mark.parametrize("full_report", [True, False])
def test_offline_catalogue_plan_and_save_are_zero_http(monkeypatch, tmp_path, full_report):
    path = catalogue(tmp_path / "catalogue.json", full_report=full_report)
    mock_http(monkeypatch, lambda request: pytest.fail("Metadata plan/save must not call HTTP"))
    runner = CliRunner()
    result = runner.invoke(command, plan_args(path))
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    entry = report["data"]["entry"]
    assert entry["model_name"] == "openai/native-model"
    assert entry["context_window"] == 16384
    assert "max_completion_tokens" not in entry["provenance"]["catalogue_limits"]
    assert "max_output_tokens" not in entry
    assert entry["max_tokens"] == 8192  # Conservative default within the general model context.
    assert not entry.get("reasoning_levels") and "reasoning_default" not in entry
    assert entry["media_inputs"] == []
    assert entry["provenance"]["media_suggestions"]["suggested"] == ["image"]
    assert not any(probe["name"].startswith("media:") for probe in report["data"]["probes"])
    assert not any(
        key in entry
        for key in (
            "pricing",
            "supported_parameters",
            "temperature",
            "unknown",
            "default_parameters",
        )
    )
    report_path = tmp_path / "plan.json"
    report_path.write_text(json.dumps(report))
    registry = tmp_path / "models.yaml"
    registry.write_text("# prior comment\nmodels:\n  other:\n    model_name: openai/other\n")
    # The caller has approved the exact proposed settings and explicit media choice.
    saved = runner.invoke(
        command,
        [
            "--stage",
            "save",
            "--input",
            str(report_path),
            "--output",
            str(registry),
            "--media-inputs",
            "image",
        ],
    )
    assert saved.exit_code == 0, saved.output
    data = yaml.safe_load(registry.read_text())
    assert data["models"]["local"]["media_inputs"] == ["image"]
    assert data["models"]["other"]["model_name"] == "openai/other"
    assert "# prior comment" in registry.read_text()


def test_exact_catalogue_id_needs_no_mapping_but_no_fuzzy_guess(monkeypatch, tmp_path):
    path = catalogue(tmp_path / "catalogue.json")
    mock_http(monkeypatch, lambda request: pytest.fail("Offline only"))
    args = plan_args(path)[:-2]
    missing = CliRunner().invoke(command, args)
    assert missing.exit_code == 2 and "match exactly once" in missing.output
    args[0] = "openai/model"
    exact = CliRunner().invoke(command, args)
    assert exact.exit_code == 0, exact.output


@pytest.mark.parametrize(
    "document",
    [
        {"source": "https://other.test/models", "models": []},
        {
            "version": 1,
            "stage": "discover",
            "ok": True,
            "data": {"source": connect.CATALOGUE_URL, "models": []},
        },
        {
            "source": connect.CATALOGUE_URL,
            "models": [{"id": "openai/model"}, {"id": "openai/model"}],
        },
        {"source": connect.CATALOGUE_URL, "models": [{"id": 3}]},
    ],
)
def test_catalogue_file_source_and_shape_validation(monkeypatch, tmp_path, document):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(document))
    mock_http(monkeypatch, lambda request: pytest.fail("Invalid local metadata must not call HTTP"))
    result = CliRunner().invoke(command, plan_args(path))
    assert result.exit_code == 2, result.output


def test_native_limits_win_over_openrouter_route_facts(monkeypatch, tmp_path):
    path = catalogue(tmp_path / "catalogue.json")
    discovery = tmp_path / "discovery.json"
    discovery.write_text(
        json.dumps(
            {
                "api_base": BASE,
                "models_url": BASE + "/models",
                "models": [
                    {
                        "id": "native-model",
                        "context_window": 8192,
                        "max_output_tokens": 2048,
                        "capabilities": {
                            "image_input": {"supported": False},
                            "pdf_input": {"supported": True},
                        },
                        "media_source": BASE + "/models",
                    }
                ],
            }
        )
    )
    mock_http(monkeypatch, lambda request: pytest.fail("Offline plan"))
    result = CliRunner().invoke(
        command, [*plan_args(path), "--discovery-file", str(discovery), "--media-inputs", "-"]
    )
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)["data"]["entry"]
    assert entry["context_window"] == 8192 and entry["max_tokens"] == 2048
    assert entry["provenance"]["media_suggestions"]["suggested"] == ["pdf"]
    assert entry["media_inputs"] == []


def test_save_existing_alias_requires_explicit_replacement(monkeypatch, tmp_path):
    path = catalogue(tmp_path / "catalogue.json")
    mock_http(monkeypatch, lambda request: pytest.fail("No inference or lookups"))
    planned = CliRunner().invoke(command, plan_args(path))
    proposal = tmp_path / "proposal.json"
    proposal.write_text(planned.output)
    registry = tmp_path / "models.yaml"
    original = "models:\n  local:\n    model_name: openai/prior\n"
    registry.write_text(original)
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(proposal), "--output", str(registry)]
    )
    assert result.exit_code == 2 and registry.read_text() == original


def test_catalogue_file_cannot_forge_internal_attribution(monkeypatch, tmp_path):
    path = catalogue(tmp_path / "catalogue.json")
    document = json.loads(path.read_text())
    document["data"]["models"][0].update(
        {
            "media_model_id": "neighbour",
            "limit_sources": {"context_length": "endpoint", "max_completion_tokens": "endpoint"},
            "endpoint_limits": {"context_window": 999999},
            "media_source": "https://other.test/models",
        }
    )
    path.write_text(json.dumps(document))
    mock_http(monkeypatch, lambda request: pytest.fail("Offline catalogue"))
    result = CliRunner().invoke(command, plan_args(path))
    assert result.exit_code == 0, result.output
    provenance = json.loads(result.output)["data"]["entry"]["provenance"]
    assert provenance["limit_sources"] == {"context_length": "catalogue"}
    assert "endpoint_limits" not in provenance
    assert provenance["media_suggestions"]["model_id"] == "openai/model"
    assert provenance["media_suggestions"]["source"] == connect.CATALOGUE_URL


@pytest.mark.parametrize(
    "defaults",
    [
        None,
        [],
        {"max_tokens": None, "temperature": 0.7},
        {"max_tokens": True},
        {"max_tokens": "9999"},
    ],
)
def test_misc_default_parameters_do_not_become_requests(monkeypatch, tmp_path, defaults):
    path = catalogue(tmp_path / "catalogue.json")
    document = json.loads(path.read_text())
    document["data"]["models"][0]["default_parameters"] = defaults
    path.write_text(json.dumps(document))
    mock_http(monkeypatch, lambda request: pytest.fail("Plan is local"))
    result = CliRunner().invoke(command, [*plan_args(path), "--max-tokens", "1024", "--yes"])
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)["data"]["entry"]
    assert entry["max_tokens"] == 1024
    assert entry["media_inputs"] == []
    assert "temperature" not in entry


def test_missing_decision_metadata_keeps_existing_images_by_default(monkeypatch):
    existing = connect.plan("local", "model", "openai", BASE, "", media_inputs=["image"]).entry
    current = state(editing=existing, budget_tokens=9000)
    current.media_hints = None
    current.api_style = "openai"

    def choose(previous):
        assert previous == ["image"]
        return "keep"

    monkeypatch.setattr(prompts, "choose_decision_images", choose)
    assert wizard.configure_media_inputs(current)
    assert current.media_inputs == ["image"]
    assert current.test_media_inputs is None


def test_optional_routing_from_catalogue_file_makes_only_one_authorized_post(monkeypatch, tmp_path):
    from tests.unifiedllm.connect.connect_http import response_body

    path = catalogue(tmp_path / "catalogue.json")
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "POST" and request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body["max_output_tokens"] == 1024
        assert not any(
            key in body for key in ("temperature", "unknown", "default_parameters", "pricing")
        )
        return httpx.Response(200, json=response_body("responses"))

    mock_http(monkeypatch, handle)
    args = plan_args(path)
    args[args.index("plan")] = "routing"
    result = CliRunner().invoke(command, [*args, "--max-tokens", "1024"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    entry = json.loads(result.output)["data"]["entry"]
    assert entry["media_inputs"] == []
    assert entry["provenance"]["media_suggestions"]["suggested"] == ["image"]


def test_discovery_file_attribution_comes_from_verified_selection(monkeypatch, tmp_path):
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "api_base": BASE,
                "models_url": BASE + "/models",
                "models": [
                    {
                        "id": "model",
                        "architecture": {"input_modalities": ["image"]},
                        "media_source": "https://other.test/models",
                        "media_model_id": "neighbour",
                    }
                ],
            }
        )
    )
    mock_http(monkeypatch, lambda request: pytest.fail("Cached metadata only"))
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--endpoint",
            BASE,
            "--api-style",
            "chat",
            "--discovery-file",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)["data"]["entry"]["provenance"]["media_suggestions"]
    assert report["source"] == BASE + "/models"
    assert report["model_id"] == "model"
    assert report["suggested"] == ["image"]


@pytest.mark.parametrize("style", ["chat", "openai"])
@pytest.mark.parametrize(
    "hints",
    [
        None,
        {},
        {"id": "model"},
        {"id": "model", "input_modalities": []},
        {"id": "model", "architecture": {"input_modalities": []}},
    ],
)
def test_missing_or_empty_media_lists_omit_source_advertisements(monkeypatch, capsys, style, hints):
    from nooa_cli.commands._connect_advertised_media import offer_suggestions

    mock_http(monkeypatch, lambda request: pytest.fail("Rendering missing metadata is local"))
    monkeypatch.setattr(
        prompts, "labelled_choice", lambda *a, **k: pytest.fail("No suggestions to choose")
    )
    current = state()
    current.api_style = style
    current.media_hints = deepcopy(hints)
    if current.media_hints is not None:
        current.media_hints["media_source"] = connect.CATALOGUE_URL
    before = deepcopy(current.media_hints)
    previous = ["image"]
    assert offer_suggestions(current, previous) is False
    output = capsys.readouterr().out
    assert "No media is listed." in output and "Current inputs: Images" in " ".join(output.split())
    for absent in (
        "OpenRouter",
        connect.CATALOGUE_URL,
        "Listed media inputs",
        "Model details from",
        "advertisement",
        "unverified",
        "unmapped",
        "native",
        "wire",
    ):
        assert absent not in output
    assert current.media_inputs is None and current.test_media_inputs is None
    assert previous == ["image"] and current.media_hints == before


@pytest.mark.parametrize("source", [connect.CATALOGUE_URL, "https://server.test/models"])
def test_media_suggestions_render_actual_source_without_changing_permissions(
    monkeypatch, capsys, source
):
    from nooa_cli.commands._connect_advertised_media import offer_suggestions

    mock_http(monkeypatch, lambda request: pytest.fail("Suggestion rendering must not send HTTP"))
    current = state()
    current.yes = True
    current.media_hints = {"id": "model", "input_modalities": ["image"], "media_source": source}
    before = deepcopy(current.media_hints)
    assert offer_suggestions(current, []) is False
    output = capsys.readouterr().out
    assert "Media listed: Images" in " ".join(output.split())
    assert "Media source: " + (
        "OpenRouter" if source == connect.CATALOGUE_URL else source
    ) in " ".join(output.split())
    assert current.media_inputs is None and current.test_media_inputs is None
    assert current.media_hints == before
