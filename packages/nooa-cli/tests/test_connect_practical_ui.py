# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queued practical UI contracts: native facts, explicit plans and permissions."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body

BASE = "https://server.test/v1"
WIRE = "bedrock-claude-opus-4-7"
KINDS = ["image", "audio", "video", "pdf"]


def state(**kwargs):
    return wizard.WizardState(
        model=WIRE,
        endpoint=BASE,
        api_style="chat",
        data={},
        approval="all",
        output_tokens=1000,
        **kwargs,
    )


@pytest.mark.parametrize(
    "native",
    [
        {"max_input_tokens": 1000000, "max_output_tokens": 64000},
        {"description": "Exact server description"},
        {"model_type": "chat"},
        {"supported_interfaces": ["chat"]},
        {"input_modalities": ["text", "image"]},
        {"capabilities": {"image_input": {"supported": False}}},
    ],
)
def test_useful_exact_native_facts_skip_fuzzy_prompt(monkeypatch, capsys, native):
    current = state()
    facts = {"id": WIRE, **native}

    async def details(*args, **kwargs):
        return connect.ModelDetails(
            WIRE,
            facts,
            (
                {
                    "schema": "openai.models",
                    "source": BASE + "/models",
                    "id": WIRE,
                    "facts": deepcopy(facts),
                },
            ),
        )

    async def catalogue():
        return [{"id": "anthropic/claude-opus-4.6", "context_length": 200000}]

    monkeypatch.setattr(connect, "server_details", details)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    monkeypatch.setattr(
        connect, "fuzzy_match_models", lambda *a: pytest.fail("No automatic version mapping")
    )
    monkeypatch.setattr(
        prompts,
        "prompt",
        lambda *a, **k: pytest.fail("Native details must not be hidden by a prompt"),
    )
    mock_http(monkeypatch, lambda r: pytest.fail("Metadata projection is offline"))
    assert wizard.prepare_metadata(current)
    assert current.selected_endpoint_model == facts
    assert current.media_hints["id"] == WIRE
    assert current.candidate is None or current.candidate["id"] == WIRE
    output = capsys.readouterr().out
    assert output.count("Model details") == 1
    assert "Source: This server" in " ".join(output.split())
    if "description" in native:
        assert native["description"] in output
    if "max_input_tokens" in native:
        assert current.candidate["context_length"] == 1000000
        assert current.candidate["top_provider"]["max_completion_tokens"] == 64000


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"id": WIRE}, False),
        ({"id": WIRE, "architecture": {"input_modalities": []}}, False),
        ({"id": WIRE, "input_modalities": []}, False),
        ({"id": WIRE, "max_input_tokens": True, "description": ""}, False),
        ({"id": "neighbour", "max_output_tokens": 1000}, False),
        ({"id": WIRE, "max_output_tokens": 1000}, True),
    ],
)
def test_usable_route_predicate_rejects_empty_and_neighbour(value, expected):
    current = state(selected_endpoint_model=value)
    assert wizard.usable_server_metadata(current) is expected


@pytest.mark.parametrize("flag", [True, False])
def test_native_group_capabilities_are_descriptive_not_settings(monkeypatch, capsys, flag):
    current = state(selected_endpoint_model={"id": WIRE})
    record = {
        "id": WIRE,
        "schema": "litellm.model_group.info",
        "source": BASE + "/model_group/info",
        "facts": {
            "supports_vision": flag,
            "supports_function_calling": flag,
            "supports_reasoning": flag,
            "supported_reasoning_efforts": ["low", "max"],
        },
    }
    current.server_details = connect.ModelDetails(WIRE, {"id": WIRE}, (record,))
    assert wizard.usable_server_metadata(current)
    wizard.show_model_metadata(current)
    text = " ".join(capsys.readouterr().out.split())
    assert "Source: This server" in text
    assert ("Media types: Images" in text) is flag
    assert "Vision listed" not in text and "Images advertised" not in text
    assert text.count("Media types:") == int(flag)
    assert "Reasoning efforts listed: low, max" in text
    assert current.candidate is None and current.media_inputs is None
    record["id"] = "neighbour"
    assert not wizard.usable_server_metadata(current)


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_native_metadata_retains_explicit_optional_public_mapping(monkeypatch, explicit):
    mapped = {"id": "anthropic/claude-opus-4.6", "context_length": 200000}
    current = state(catalogue_model=mapped["id"] if explicit else None)

    async def details(*a, **k):
        return connect.ModelDetails(WIRE, {"id": WIRE})

    async def catalogue():
        return [mapped]

    monkeypatch.setattr(connect, "server_details", details)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    seen = []

    def match(model, models):
        seen.append(model)
        return models

    monkeypatch.setattr(connect, "fuzzy_match_models", match)
    monkeypatch.setattr(prompts, "prompt", lambda *a, **k: mapped["id"])
    assert wizard.prepare_metadata(current)
    assert current.candidate["id"] == mapped["id"]
    assert seen == ([] if explicit else [WIRE])
    assert current.model == WIRE  # Mapping never rewrites the wire identity.


@pytest.mark.parametrize("mapping", [None, "public-model"])
def test_exact_or_explicit_public_fills_gaps_without_overwriting_native(monkeypatch, mapping):
    native = {"id": WIRE, "description": "Native", "max_output_tokens": 64000}
    current = state(catalogue_model=mapping)

    async def details(*a, **k):
        return connect.ModelDetails(
            WIRE,
            native,
            (
                {
                    "schema": "openai.models",
                    "source": BASE + "/models",
                    "id": WIRE,
                    "facts": native,
                },
            ),
        )

    async def catalogue():
        return [
            {
                "id": mapping or WIRE,
                "description": "Public",
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 8000},
                "reasoning": {"supported_efforts": ["high"], "default_effort": "high"},
            }
        ]

    monkeypatch.setattr(connect, "server_details", details)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    monkeypatch.setattr(
        prompts, "prompt", lambda *a, **k: pytest.fail("Exact metadata needs no prompt")
    )
    assert wizard.prepare_metadata(current)
    assert current.candidate["context_length"] == 200000
    assert current.candidate["top_provider"]["max_completion_tokens"] == 64000
    assert current.candidate["limit_sources"] == {
        "context_length": "catalogue",
        "max_completion_tokens": "endpoint",
    }
    assert "reasoning" not in current.candidate
    assert current.model == WIRE
    assert current.server_details.records[-1]["schema"] == "openrouter.fallback"
    assert current.server_details.records[-1]["selected_id"] == WIRE


@pytest.mark.parametrize("selected", [[], ["image"], ["audio", "video", "pdf"], KINDS])
@pytest.mark.parametrize("action", ["test", "enable"])
def test_manual_selection_plans_only_and_off_has_no_extra_prompt(monkeypatch, selected, action):
    current = state()
    previous = ["pdf"]
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda values: list(selected))
    choices = []

    def choose(label, options, *, default):
        choices.append(label)
        assert default == "test"
        names = ", ".join(view.media_label(kind) for kind in selected)
        assert label == f"Would you like to test these media types: {names}?"
        assert dict(options) == {
            "test": f"Test {names}",
            "enable": f"Enable {names} without testing",
        }
        return action

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    mock_http(monkeypatch, lambda r: pytest.fail("Selection is not network consent"))
    wizard.select_manual_media(current, previous)
    assert len(choices) == bool(selected)
    assert current.media_inputs == (previous if selected and action == "test" else selected)
    assert current.test_media_inputs == (selected if selected and action == "test" else None)
    assert previous == ["pdf"]


@pytest.mark.parametrize(
    "style,selected,checkable",
    [
        ("chat", KINDS, True),
        ("responses", ["audio", "video"], False),
        ("responses", KINDS, True),
        ("systemone", ["image"], False),
    ],
)
def test_manual_unavailable_checks_never_imply_permissions(
    monkeypatch, capsys, style, selected, checkable
):
    current = state()
    current.api_style = style
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda values, **kw: selected)
    monkeypatch.setattr(
        prompts,
        "labelled_choice",
        lambda *a, **k: "test" if checkable else pytest.fail("No tests available"),
    )
    if style == "responses" and {"audio", "video"}.intersection(selected):
        import click

        with pytest.raises(click.UsageError, match="Unsupported media input"):
            wizard.select_manual_media(current, [])
        assert current.media_inputs is None and current.test_media_inputs is None
        return
    wizard.select_manual_media(current, [])
    assert current.test_media_inputs == (selected if checkable else None)
    assert current.media_inputs == (
        [] if checkable else selected
    )  # Explicit untested declaration only.
    if style != "chat":
        assert (
            "is not supported" if style == "responses" else "isn't available"
        ) in capsys.readouterr().out


@pytest.mark.parametrize(
    "style,listed", [("chat", ["image"]), ("responses", ["audio"]), ("openai", ["image"])]
)
def test_listed_test_all_selects_known_types_even_unlisted_or_unavailable(
    monkeypatch, style, listed
):
    current = state(media_hints={"id": WIRE, "input_modalities": listed})
    current.api_style = style

    def choose(label, options, *, default):
        if style == "responses":
            # Audio is advertised, not implementable/suggested. The fallback
            # media menu can still explicitly test all and report skipped rows.
            assert dict(options)["test"] == "Test all media inputs (generated samples)"
            return "test"
        assert dict(options)["test-all"] == "Test all media inputs"
        assert default == "test"
        return "test-all"

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    assert wizard.configure_media_inputs(current)
    assert current.test_media_inputs == (["image"] if style == "openai" else KINDS)
    assert current.media_inputs == []


@pytest.mark.parametrize(
    "media_action,selection,use_action,final_action,expected",
    [
        # Saving without testing keeps formats that were waiting for a test, as not tested.
        ("manual", "image,audio,video,pdf", "test", "save", KINDS),
        ("manual", "image,audio,video,pdf", "enable", "save", KINDS),
        ("manual", "-", None, "save", []),
        ("test-all", None, None, "save", KINDS),
        ("suggested", None, None, "save", ["image"]),
        ("none", None, None, "save", []),
        ("manual", "image,audio,video,pdf", "test", "test", ["image", "audio", "pdf"]),
        ("test-all", None, None, "test", ["image", "audio", "pdf"]),
        ("test", None, None, "test", ["image"]),
    ],
)
def test_actual_guided_manual_and_listed_consent_save_and_partial_results(
    monkeypatch, tmp_path, media_action, selection, use_action, final_action, expected
):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("PRACTICAL_UI_TEST_KEY", "fake-test-key")
    calls, consent, selected_prompts = [], [], []
    original = prompts.labelled_choice

    def choose(label, options, **kwargs):
        if label.startswith("Would you like to test these media types:"):
            assert all(r.method == "GET" for r in calls)
            selected_prompts.append(options)
        if label == "Would you like to test these settings?":
            assert all(r.method == "GET" for r in calls)
            consent.append(options)
        return original(label, options, **kwargs)

    monkeypatch.setattr(prompts, "labelled_choice", choose)

    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": WIRE,
                            "context_length": 1000000,
                            "max_output_tokens": 64000,
                            "description": "Exact native server model",
                            "input_modalities": ["text", "image"],
                        }
                    ]
                },
            )
        assert consent
        body = json.loads(request.content)
        assert body["model"] == WIRE
        # Video cannot be enabled by a rejected recognition check; other samples pass.
        serialized = json.dumps(body)
        if '"video_url"' in serialized:
            return httpx.Response(422)
        answer = """{"value":323}"""
        if '"image_url"' in serialized:
            answer = '{"value":true}'
        elif '"input_audio"' in serialized:
            answer = '{"value":true}'
        elif '"file"' in serialized:
            answer = '{"value":true}'
        return httpx.Response(200, json=response_body("chat", answer))

    mock_http(monkeypatch, handle)
    written = []
    original_write = connect.write

    def write(entry, *args, **kwargs):
        written.append(deepcopy(entry))
        return original_write(entry, *args, **kwargs)

    monkeypatch.setattr(connect, "write", write)
    target = tmp_path / "models.yaml"
    answers = [
        media_action,
        *([selection] if selection is not None else []),
        *([use_action] if use_action else []),
        final_action,
    ]
    if final_action == "test":
        answers.append("keep")
    answers.append("y")
    result = CliRunner().invoke(
        command,
        [
            WIRE,
            "--as",
            "local",
            "--endpoint",
            BASE,
            "--api-style",
            "chat",
            "--api-key-env",
            "PRACTICAL_UI_TEST_KEY",
            "--reply-tokens",
            "2000",
            "--budget-tokens",
            "50000",
            "--probe",
            "all",
            "--output",
            str(target),
        ],
        input="\n".join(answers) + "\n",
    )
    assert result.exit_code == 0, result.output
    assert len(consent) == 1
    assert len(selected_prompts) == int(media_action == "manual" and selection != "-")
    posts = [r for r in calls if r.method == "POST"]
    assert len(posts) == (0 if final_action == "save" else 3 if media_action == "test" else 6), (
        result.output
    )
    entry = yaml.safe_load(target.read_text())["models"]["local"]
    assert entry["media_inputs"] == expected
    assert entry["context_window"] == 1000000
    assert result.output.count("\n  Model details\n") == 1
    assert "Which of these is your model?" not in result.output
    if final_action == "save":
        assert not any(
            key.startswith("media:") for key in written[0]["provenance"].get("probes", {})
        )


@pytest.mark.parametrize("style", ["openai", "systemone"])
@pytest.mark.parametrize("selected", [[], ["image"]])
def test_post_result_decision_manual_picker_is_image_only_and_never_retests(
    monkeypatch, tmp_path, style, selected
):
    entry = connect.plan("local", WIRE, style, BASE, "", media_inputs=["image"]).entry
    current = wizard.WizardState(
        alias="local",
        model=WIRE,
        api_style=style,
        path=tmp_path / "models.yaml",
        no_probe=True,
        test_media_inputs=["image"],
        result=connect.ConnectResult(alias="local", entry=entry),
    )
    seen = []

    def choose(previous, *, available):
        seen.append((previous, available))
        assert available == ("image",)
        return selected

    monkeypatch.setattr(prompts, "review_media_inputs", lambda previous: "manual")
    monkeypatch.setattr(prompts, "choose_media_inputs", choose)
    monkeypatch.setattr(
        prompts,
        "labelled_choice",
        lambda *a, **k: pytest.fail("Post-result manual declaration must not schedule new tests"),
    )
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    mock_http(monkeypatch, lambda r: pytest.fail("Result review must not make requests"))
    assert wizard.save_model(current)
    assert seen == [(["image"], ("image",))]
    saved = yaml.safe_load(current.path.read_text())["models"]["local"]
    assert saved["media_inputs"] == selected


@pytest.mark.parametrize("style", ["responses", "anthropic"])
def test_audio_video_unsupported_native_format_is_scoped_and_zero_sample_send(monkeypatch, style):
    from nooa_cli.commands import _connect_view as view

    report = connect.advertised_media_suggestions(
        {"id": WIRE, "input_modalities": ["audio", "video"]},
        api_style=style,
    )
    assert report["checkable"] == []
    for kind, reason in report["unavailable_checks"].items():
        assert "Unsupported media input" in reason and kind in reason
        detail = view.media_test_detail(kind, {"outcome": "not_probed", "reason": reason})
        assert "Unsupported by this request interface" in detail and "cannot be enabled" in detail
    mock_http(monkeypatch, lambda r: pytest.fail("Unavailable media fixture must not be sent"))
    from nooa.unifiedllm.connect._media import media_content

    for kind in ("audio", "video"):
        content, _, reason = media_content(kind, style)
        assert content is None and "Unsupported media input" in reason


@pytest.mark.parametrize("style", ["responses", "anthropic"])
@pytest.mark.parametrize("previous", [[], ["audio"]])
def test_guided_test_all_proposal_marks_unimplemented_checks_before_consent(
    monkeypatch, capsys, style, previous
):
    current = state(guided=True, test_media_inputs=KINDS)
    current.api_style = style
    if previous:
        with pytest.raises(ValueError, match="Unsupported media input.*audio"):
            connect.plan(
                "local", WIRE, style, BASE, "", media_inputs=previous, test_media_inputs=KINDS
            )
        assert current.media_inputs is None  # No permission is silently revoked to plan.
        return
    current.proposal = connect.plan(
        "local", WIRE, style, BASE, "", media_inputs=previous, test_media_inputs=KINDS
    )
    mock_http(monkeypatch, lambda r: pytest.fail("A proposal cannot send samples"))
    wizard.show_guided_proposal(current)
    text = " ".join(capsys.readouterr().out.split())
    assert (
        "Audio: "
        + (
            "Enabled; interface doesn’t support this format"
            if previous
            else "Off; interface doesn’t support this format"
        )
        in text
    )
    assert "Video: Off; interface doesn’t support this format" in text
    # The row already explains unsupported formats; no separate test reason.
    assert "Audio test:" not in text and "Video test:" not in text
    assert "Use Chat" not in text
    assert current.proposal.entry["media_inputs"] == previous
