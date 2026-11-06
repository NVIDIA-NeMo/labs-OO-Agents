# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect implementation ceilings, explicit repair and no preconsent side effects."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands._connect_input_policy import repair_native_media
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._records import check_status, public_record
from nooa.unifiedllm.connect._verification import stamp_verification
from tests.unifiedllm.connect.connect_http import mock_http, response_body

ALL = ["image", "audio", "video", "pdf"]


def entry(style, media=ALL):
    return {
        "model_name": ("anthropic/" if style == "anthropic" else "openai/") + "model",
        "api_style": style,
        "client_type": "responses" if style == "responses" else "completion",
        "api_base": "https://policy.test/v1",
        "api_key_env": "",
        "media_inputs": list(media),
        "context_window": 64000,
        "max_tokens": 128,
        "cache_breakpoint": None,
        "provenance": {"probes": {"media:audio": {"outcome": "passed", "historic": "keep"}}},
    }


@pytest.fixture
def offline(monkeypatch):
    calls = []

    def forbidden(request):
        calls.append(request)
        raise AssertionError("Invalid native declarations/repair must not send HTTP")

    mock_http(monkeypatch, forbidden)
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    return calls


def state(style, **kwargs):
    return wizard.WizardState(
        model="model",
        alias="local",
        endpoint="https://policy.test/v1",
        api_style=style,
        api_key_env="",
        approval="none",
        no_probe=True,
        data={},
        context_window=64000,
        reply_tokens=128,
        output_tokens=128,
        reasoning_output_tokens=128,
        budget_tokens=16384,
        **kwargs,
    )


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize("stage", [None, "plan", "routing", "media", "all"])
def test_flags_headless_stages_reject_before_http_or_write(offline, tmp_path, style, kind, stage):
    target = tmp_path / "new-parent" / "models.yaml"
    args = [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://policy.test/v1",
        "--api-style",
        style,
        "--api-key-env",
        "",
        "--media-inputs",
        kind,
        "--output",
        str(target),
    ]
    if stage:
        args = args[:-2] + ["--stage", stage]  # check stages emit JSON, not a registry file
    else:
        args += ["--no-probe", "--no-catalogue", "--yes"]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 2, result.output
    assert "Unsupported media input" in result.output and kind in result.output
    assert not offline and not target.parent.exists()
    if stage:
        assert json.loads(result.stdout)["error"]["type"] == "UsageError"


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("replacement", [None, "image,pdf", "-"])
def test_stage_save_legacy_requires_explicit_repair(offline, tmp_path, style, replacement):
    original = entry(style)
    before = deepcopy(original)
    source = tmp_path / "legacy.json"
    target = tmp_path / "new-parent" / "models.yaml"
    source.write_text(json.dumps({"alias": "legacy", "entry": original}))
    source_bytes = source.read_bytes()
    args = ["--stage", "save", "--input", str(source), "--output", str(target)]
    if replacement is not None:
        args += ["--media-inputs", replacement]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == (2 if replacement is None else 0), result.output
    if replacement is None:
        assert "Unsupported media input" in result.output
        assert not target.parent.exists()
    else:
        saved = yaml.safe_load(target.read_text())["models"]["legacy"]
        assert saved["media_inputs"] == (["image", "pdf"] if replacement != "-" else [])
        assert "provenance" not in saved
    assert source.read_bytes() == source_bytes and original == before and not offline


@pytest.mark.parametrize("style", ["anthropic", "responses"])
def test_native_suggestions_manual_picker_cannot_enable_audio_video(monkeypatch, offline, style):
    s = state(style, media_hints={"id": "model", "input_modalities": ALL})
    s.metadata_displayed = True
    report = connect.advertised_media_suggestions(s.media_hints, api_style=style)
    assert report["advertised"] == ALL and report["suggested"] == ["image", "pdf"]
    assert report["checkable"] == ["image", "pdf"]
    options_seen = []

    def choose(label, options, **kwargs):
        options_seen.append(dict(options))
        assert not any("Audio" in v or "Videos" in v for k, v in options if k == "suggested")
        return "manual"

    def picker(current, *, available):
        assert available == ("image", "pdf") and list(current) == []
        return ["pdf"]

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    monkeypatch.setattr(prompts, "choose_media_inputs", picker)
    assert wizard.configure_media_inputs(s)
    assert s.media_inputs == ["pdf"] and s.test_media_inputs is None
    assert options_seen and not offline


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("action", ["cancel", "supported", "off"])
def test_explicit_repair_never_erases_source_or_defaults_to_revocation(monkeypatch, style, action):
    original = entry(style)
    before = deepcopy(original)
    s = state(style)

    def choose(label, options, *, default):
        assert "Repair unsupported" in label and default == "cancel"
        assert list(dict(options)) == ["supported", "off", "cancel"]
        return action

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    result = repair_native_media(s, original["media_inputs"], entry=original)
    assert original == before
    if action == "cancel":
        assert result is None and s.media_inputs is None and not s.guided_media_revised
    else:
        assert result == (["image", "pdf"] if action == "supported" else [])
        assert s.media_inputs == result and not s.inherit_media_inputs and s.guided_media_revised


@pytest.mark.parametrize("style", ["anthropic", "responses"])
def test_actual_edit_cancel_offers_repair_without_credentials_calls_or_writes(
    monkeypatch, offline, tmp_path, style
):
    target = tmp_path / "models.yaml"
    raw = entry(style)
    target.write_text(yaml.safe_dump({"models": {"legacy": raw}}))
    before = target.read_bytes()
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [target])
    result = CliRunner().invoke(
        command, ["--edit-model", "legacy", "--output", str(target), "--no-probe"], input="cancel\n"
    )
    assert result.exit_code == 0, result.output
    assert "Repair unsupported media permissions" in result.output
    assert target.read_bytes() == before and not offline
    assert not target.with_name("secrets.yaml").exists()


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("replacement", [None, "image,pdf", "-"])
def test_actual_legacy_headless_edit_invalid_or_explicit_repair(
    monkeypatch, offline, tmp_path, style, replacement
):
    target = tmp_path / "models.yaml"
    raw = entry(style)
    target.write_text(yaml.safe_dump({"models": {"legacy": raw}}))
    before = target.read_bytes()
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [target])
    args = ["--edit-model", "legacy", "--output", str(target), "--no-probe", "--yes"]
    if replacement is not None:
        args += ["--media-inputs", replacement]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == (2 if replacement is None else 0), result.output
    if replacement is None:
        assert target.read_bytes() == before and "Unsupported media input" in result.output
    else:
        saved = yaml.safe_load(target.read_text())["models"]["legacy"]
        assert saved["media_inputs"] == (["image", "pdf"] if replacement != "-" else [])
    assert not offline


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("action", ["cancel", "supported"])
def test_route_change_requires_explicit_selection_before_planning(
    monkeypatch, offline, style, action
):
    s = state(style, media_inputs=ALL.copy())  # Selected manually on previous Chat route.
    original = s.media_inputs.copy()
    monkeypatch.setattr(prompts, "labelled_choice", lambda *a, **kw: action)
    monkeypatch.setattr(wizard, "require_manual_limits", lambda state: None)
    result = wizard.configure_checks(s)
    assert result is (action == "supported")
    if action == "cancel":
        assert s.media_inputs == original and s.proposal is None
    else:
        assert s.proposal.entry["media_inputs"] == ["image", "pdf"]
        assert not s.inherit_media_inputs
    assert not offline


@pytest.mark.parametrize("style", ["anthropic", "responses"])
def test_posttest_manual_invalid_cannot_mutate_or_be_claimed_correct(monkeypatch, offline, style):
    s = state(style, test_media_inputs=ALL.copy())
    s.result = connect.ConnectResult("local", entry(style, ["image"]))
    before = deepcopy(s.result.entry)
    monkeypatch.setattr(wizard, "baseline_allows_save", lambda state: True)
    monkeypatch.setattr(prompts, "review_media_inputs", lambda current: "manual")

    def picker(current, *, available):
        assert available == ("image", "pdf")
        return ["audio"]  # Explicit invalid selection cannot bypass backend gate.

    monkeypatch.setattr(prompts, "choose_media_inputs", picker)
    with pytest.raises(ValueError, match="Unsupported media input.*audio"):
        wizard.save_model(s)
    assert s.result.entry == before and not offline


@pytest.mark.parametrize("style", ["anthropic", "responses"])
def test_unsupported_historical_pass_is_not_derived_permission_or_current_pass(style):
    raw = entry(style, [])
    record = {"outcome": "passed", "client": "unifiedllm"}
    stamp_verification(record, raw, "media:audio")
    before = deepcopy(record)
    assert check_status("media:audio", record, entry=raw) == "attention"
    assert record == before and record["outcome"] == "passed"
    assert public_record({"outcome": "not_probed", "implementation_supported": False}) == {
        "outcome": "not_probed",
        "implementation_supported": False,
    }


@pytest.mark.parametrize("style", ["anthropic", "responses"])
def test_all_native_rows_report_unsupported_and_send_only_image_pdf(monkeypatch, style):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        value = 323 if len(calls) == 1 else True
        return httpx.Response(200, json=response_body(style, json.dumps({"value": value})))

    mock_http(monkeypatch, respond)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--endpoint",
            "https://policy.test/v1",
            "--api-style",
            style,
            "--api-key-env",
            "",
            "--stage",
            "media",
            "--test-media-inputs",
            ",".join(ALL),
            "--reply-tokens",
            "1000",
        ],
    )
    assert result.exit_code == 1, result.output  # Unsupported rows are not a fully-passed report.
    report = json.loads(result.stdout)
    assert len(calls) == 3 and report["data"]["entry"]["media_inputs"] == ["image", "pdf"]
    for kind in ["audio", "video"]:
        assert report["checks"]["media:" + kind]["outcome"] == "not_probed"
        assert report["checks"]["media:" + kind]["implementation_supported"] is False
        assert "Unsupported media input" in report["checks"]["media:" + kind]["reason"]
    assert not {"input_audio", "video_url"}.intersection(
        b.get("type")
        for body in calls
        for msg in body["input" if style == "responses" else "messages"]
        if isinstance(msg.get("content"), list)
        for b in msg["content"]
    )


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("action", ["cancel", "supported"])
def test_existing_chat_profile_route_change_does_not_silently_erase(
    monkeypatch, offline, style, action
):
    old = entry("chat")
    s = state(style, editing=None)
    s.data = {"models": {"local": old}}
    before = deepcopy(old)
    seen = []

    def choose(label, options, **kw):
        seen.append(label)
        assert "Repair unsupported" in label
        return action

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    assert wizard.configure_media_inputs(s) is (action == "supported")
    assert seen and old == before and not offline
    assert s.media_inputs == (["image", "pdf"] if action == "supported" else None)


def test_custom_chat_edit_native_label_preserves_valid_untested_declaration(
    monkeypatch, offline, tmp_path
):
    raw = {**entry("anthropic", ["audio", "video"]), "custom_llm_provider": "openai"}
    target = tmp_path / "models.yaml"
    target.write_text(yaml.safe_dump({"models": {"custom": raw}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [target])
    result = CliRunner().invoke(
        command, ["--edit-model", "custom", "--output", str(target), "--no-probe", "--yes"]
    )
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(target.read_text())["models"]["custom"]
    assert saved["media_inputs"] == ["audio", "video"]
    assert saved["custom_llm_provider"] == "openai" and not offline
