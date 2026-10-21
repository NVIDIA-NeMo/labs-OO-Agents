"""Explicit decision metadata lookup and offline file mappings."""

import json

import httpx
import pytest
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands._connect_stages import read_catalogue
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http

MODEL = "nvidia/alexwortega/openjev-4b"
META = {
    "id": "typesafe/jev-router",
    "description": "Image-aware router",
    "input_modalities": ["image", "audio"],
    "context_length": 99999,
    "default_parameters": {"max_tokens": 12345, "store": True},
    "reasoning": {"supported_efforts": ["high"]},
    "include": ["x"],
}


def state(**kwargs):
    return wizard.WizardState(
        model=MODEL,
        endpoint="https://api.test/decisions",
        api_style="systemone",
        api_key_env="",
        data={},
        yes=True,
        **kwargs,
    )


def write_catalogue(tmp_path, models=None):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"source": connect.CATALOGUE_URL, "models": models or [META]}))
    return path


@pytest.mark.parametrize("style", connect.DECISION_API_STYLES)
def test_file_plan_is_offline_and_explicit(monkeypatch, tmp_path, style):
    path = write_catalogue(tmp_path)
    mock_http(monkeypatch, lambda request: pytest.fail("Offline plan must not send HTTP"))
    options = [
        MODEL,
        "--stage",
        "plan",
        "--api-style",
        style,
        "--endpoint",
        "https://api.test/decisions",
        "--api-key-env",
        "MISSING_KEY",
        "--catalogue-file",
        str(path),
    ]
    monkeypatch.delenv("MISSING_KEY", raising=False)
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 2 and "exactly once" in result.output
    result = CliRunner().invoke(command, options + ["--catalogue-model", META["id"]])
    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)["data"]["entry"]
    assert entry["model_name"] == MODEL and entry["media_inputs"] == []
    assert entry["provenance"]["catalogue"]["id"] == META["id"]
    assert entry["provenance"]["catalogue_metadata"]["description"] == META["description"]
    assert (
        not {"max_tokens", "context_window", "reasoning_levels", "tools", "store", "include"}
        & entry.keys()
    )
    assert (
        entry["decision_image_transport"] == "json"
        and entry["typesafe_danger_zone_images"] is False
    )


@pytest.mark.parametrize("stage", ["routing", "decision", "media", "all"])
def test_file_mapping_is_accepted_by_decision_checks_without_extra_inference(
    monkeypatch, tmp_path, stage
):
    path = write_catalogue(tmp_path)
    seen = []

    async def check(proposal, selected_stage, *, api_key):
        assert selected_stage == stage
        plain = connect.plan("candidate", MODEL, "systemone", "https://api.test/decisions", "")
        assert proposal.probes == plain.probes
        assert proposal.entry["provenance"]["catalogue"]["id"] == META["id"]
        seen.append(proposal)
        return connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "check_stage", check)
    mock_http(monkeypatch, lambda request: pytest.fail("No extra metadata HTTP"))
    result = CliRunner().invoke(
        command,
        [
            MODEL,
            "--stage",
            stage,
            "--api-style",
            "systemone",
            "--endpoint",
            "https://api.test/decisions",
            "--api-key-env",
            "",
            "--catalogue-file",
            str(path),
            "--catalogue-model",
            META["id"],
        ],
    )
    assert seen, result.output  # Unchecked verdict is intentionally not a passing check.
    assert json.loads(result.stdout)["data"]["entry"]["media_inputs"] == []


@pytest.mark.parametrize("models", [[META, META], [{"id": "typesafe/jev_router"}]])
def test_file_validator_still_rejects_duplicate_or_nonexact_mapping(tmp_path, models):
    with pytest.raises(ValueError, match="exactly once"):
        read_catalogue(write_catalogue(tmp_path, models), MODEL, catalogue_model=META["id"])


def test_wizard_unknown_id_never_assumes_hub_mapping(monkeypatch, capsys):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [META]}))
    current = state()
    assert wizard.prepare_metadata(current)
    assert current.candidate is None
    assert "No OpenRouter details found for this model" in capsys.readouterr().out
    assert current.media_hints == {}
    assert not current.no_catalogue


@pytest.mark.parametrize("explicit", [False, True])
def test_wizard_exact_or_explicit_mapping_fetches_only_public_get(monkeypatch, explicit):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET" and str(request.url) == connect.CATALOGUE_URL
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [META]})

    mock_http(monkeypatch, handle)
    current = state(catalogue_model=META["id"] if explicit else None)
    if not explicit:
        current.model = META["id"]
    assert wizard.prepare_metadata(current)
    assert len(requests) == 1
    assert current.candidate["id"] == META["id"]
    assert (
        not {"context_length", "reasoning", "default_parameters", "include"}
        & current.candidate.keys()
    )
    assert current.media_hints["media_source"] == connect.CATALOGUE_URL
    assert current.media_inputs is None and current.test_media_inputs is None


def test_wizard_unknown_offers_explicit_choice_or_continue(monkeypatch, capsys):
    async def catalogue():
        return [META]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    for choice in ("", META["id"]):
        current = state()
        current.yes = False

        def choose(label, choice=choice, **kwargs):
            assert "Explicit OpenRouter match" in label
            assert "" in kwargs["choices"] and META["id"] in kwargs["choices"]
            return choice

        monkeypatch.setattr(prompts, "prompt", choose)
        assert wizard.prepare_metadata(current)
        assert (current.candidate is not None) == bool(choice)
    out = capsys.readouterr().out
    assert "No OpenRouter details found for this model" in out
    assert "--catalogue-model" in out and "--no-catalogue" in out


@pytest.mark.parametrize("mode", ["skip", "edit"])
@pytest.mark.parametrize("permissions", [[], ["image"]])
def test_wizard_skip_and_edit_never_fetch(monkeypatch, mode, permissions):
    from copy import deepcopy

    mock_http(monkeypatch, lambda request: pytest.fail("Offline wizard must not send HTTP"))
    existing = {"model_name": MODEL, "media_inputs": permissions.copy()}
    current = state(no_catalogue=True) if mode == "skip" else state(editing=existing)
    current.media_inputs = permissions.copy()
    current.decision_image_transport = "typesafe_multipart"
    current.typesafe_danger_zone_images = True
    before = deepcopy(existing)
    assert wizard.prepare_metadata(current)
    assert current.candidate is None
    assert current.media_inputs == permissions and current.test_media_inputs is None
    assert current.decision_image_transport == "typesafe_multipart"
    assert current.typesafe_danger_zone_images is True
    assert existing == before


def test_explicit_missing_match_and_no_catalogue_conflict(monkeypatch):
    async def catalogue():
        return [META]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    missing = state(catalogue_model="absent")
    assert wizard.prepare_metadata(missing)
    assert missing.candidate is None  # Missing optional details never block tests.
    with pytest.raises(Exception, match="cannot be used"):
        wizard.prepare_metadata(state(catalogue_model=META["id"], no_catalogue=True))


@pytest.mark.parametrize("credential", ["UNSET_KEY", "bad-name", None])
def test_decision_lookup_validates_explicit_listing_credentials_before_http(
    monkeypatch, credential
):
    monkeypatch.delenv("UNSET_KEY", raising=False)
    mock_http(monkeypatch, lambda request: pytest.fail("Invalid config must not fetch"))
    current = state(models_url="https://other.test/models", discovery_api_key_env=credential)
    with pytest.raises(ValueError):
        wizard.prepare_metadata(current)
