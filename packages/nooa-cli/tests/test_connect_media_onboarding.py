# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approval, review and persistence of generated media checks (fake HTTP only)."""

import json
import re

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


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("NOOA_CONNECT_TEST_API_KEY", "offline-dummy")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])


def options(path):
    return [
        "model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "NOOA_CONNECT_TEST_API_KEY",
        "--no-catalogue",
        "--max-tokens",
        "1000",
        "--budget-tokens",
        "12000",
        "--output",
        str(path),
    ]


def fake_http(monkeypatch, media_answer='{"value":true}'):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        content = next(
            (
                m["content"]
                for m in calls[-1]["messages"]
                if isinstance(m.get("content"), list)
                and any(b.get("type") != "text" for b in m["content"])
            ),
            calls[-1]["messages"][0]["content"],
        )
        return httpx.Response(
            200,
            json=response_body(
                "chat",
                '{"value":323}'
                if "response_format" in calls[-1]
                and not any(
                    isinstance(m.get("content"), list)
                    and any(b.get("type") != "text" for b in m["content"])
                    for m in calls[-1]["messages"]
                )
                else "323"
                if isinstance(content, str)
                else media_answer,
            ),
        )

    mock_http(monkeypatch, handle)
    return calls


@pytest.mark.parametrize("answer,enabled", [('{"value":true}', ["image"]), ("wrong", []), ("", [])])
def test_explicit_noninteractive_test_saves_only_passed_samples(
    monkeypatch, tmp_path, answer, enabled
):
    calls = fake_http(monkeypatch, answer)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, options(path) + ["--test-media-inputs", "image,video", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 3  # Routing, tools, image; video and conversation do not fit/are skipped.
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["media_inputs"] == enabled
    assert "provenance" not in entry
    assert (
        "Images: enabled (test passed)" if "image" in enabled else "Images: off"
    ) in result.output
    assert "check budget used up" in result.output
    text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    assert f"Image input: {'Passed' if enabled else 'Failed'}" in text
    assert "Video input: Skipped" in text
    assert "not accuracy or permanent model support" not in result.output


@pytest.mark.parametrize(
    "flags",
    [
        ["--test-media-inputs", "image", "--no-probe"],
        ["--test-media-inputs", "image", "--probe", "minimal"],
        ["--test-media-inputs", "image", "--media-inputs", "pdf"],
        ["--test-media-inputs", "file"],
        ["--test-media-inputs", ""],
        ["--media-inputs", "image,image"],
    ],
)
def test_invalid_unattended_flags_validate_before_calls_or_writes(monkeypatch, tmp_path, flags):
    calls = fake_http(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, options(path) + flags + ["--yes"])
    assert result.exit_code == 2, result.output
    assert not calls and not path.exists()


def test_manual_no_probe_makes_no_calls_and_labels_declarations_unverified(monkeypatch, tmp_path):
    calls = fake_http(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, options(path) + ["--media-inputs", "audio,pdf", "--no-probe", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert not calls
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["audio", "pdf"]
    assert "not tested (manual setup)" in result.output
    assert "Connection: not tested (manual setup)" in " ".join(result.output.split())


def test_initial_decline_prevents_all_probes_and_writes(monkeypatch, tmp_path):
    calls = fake_http(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, options(path) + ["--test-media-inputs", "image"], input="save\nn\n"
    )
    assert result.exit_code == 0
    assert not calls and not path.exists()
    assert "Media samples: Images" in " ".join(result.output.split())
    assert "approved token budget" not in result.output


def test_interactive_test_choice_after_single_approval_review_can_manually_enable_unverified(
    monkeypatch, tmp_path
):
    calls = fake_http(monkeypatch, "wrong")
    path = tmp_path / "models.yaml"
    # Approval, test/manual choice, final review/manual choice, declaration, save.
    result = CliRunner().invoke(command, options(path), input="test\ntest\nmanual\nimage,pdf\ny\n")
    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert result.output.count("Would you like to test these settings?") == 1
    assert "Approve setup" not in result.output
    assert "Test all media inputs (generated samples)" in result.output
    assert "Choose media inputs manually" in result.output
    assert "Formats to TEST" not in result.output
    assert "Media samples: Images, Audio, Video, PDFs" in " ".join(result.output.split())
    for kind in ["image", "audio", "video", "pdf"]:
        assert f"Generated {kind} sample:" not in result.output
    assert "check budget used up" in result.output
    text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    assert "Image input: Failed · Could not read the answer" in text
    for kind in ["Audio", "Video", "PDF"]:
        assert f"{kind} input: Skipped" in text
    assert "Images: enabled" in text
    assert "Images: enabled (not confirmed by this test)" in text
    assert "PDFs: enabled (not tested)" in text
    assert "Inputs enabled manually will be saved as not tested." in text
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["image", "pdf"]


def test_cancel_at_review_does_not_save_or_send_extra_requests(monkeypatch, tmp_path):
    calls = fake_http(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, options(path), input="test\ntest\n")
    assert result.exit_code == 1
    assert len(calls) == 3 and not path.exists()


def test_ui_edit_mode_defaults_manual_and_no_probe_never_offers_tests(monkeypatch):
    seen = []

    def prompt(label, *args, **kw):
        seen.append((label, kw))
        return kw["default"]

    monkeypatch.setattr(prompts, "prompt", prompt)
    monkeypatch.setattr(prompts, "labelled_choice", prompt)
    state = wizard.WizardState(editing={"media_inputs": ["audio"]}, approval="all", yes=False)
    assert wizard.configure_media_inputs(state)
    assert seen[0][1]["default"] == "manual" and state.media_inputs == ["audio"]
    seen.clear()
    state = wizard.WizardState(editing={"media_inputs": ["pdf"]}, approval="none", yes=False)
    assert wizard.configure_media_inputs(state)
    assert len(seen) == 1 and "Enable media inputs manually" in seen[0][0]
    assert state.media_inputs == ["pdf"]
    assert state.test_media_inputs is None


@pytest.mark.parametrize("previous", [[], ["audio", "video"]])
def test_guided_test_plans_all_formats_without_follow_on_prompt(monkeypatch, previous):
    seen = []

    def choose(*, editing):
        seen.append(editing)
        return "test"

    def forbidden(*args, **kwargs):
        pytest.fail("Test all must not ask a follow-on format-entry question")

    monkeypatch.setattr(prompts, "choose_media_mode", choose)
    monkeypatch.setattr(prompts, "prompt", forbidden)
    editing = {"media_inputs": list(previous)} if previous else None
    state = wizard.WizardState(editing=editing, approval="all", yes=False)
    assert wizard.configure_media_inputs(state)
    assert seen == [bool(editing)]
    assert state.test_media_inputs == ["image", "audio", "video", "pdf"]
    assert state.media_inputs == previous
    if editing:
        assert editing["media_inputs"] == previous
        assert state.media_inputs is not editing["media_inputs"]
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        media_inputs=state.media_inputs,
        test_media_inputs=state.test_media_inputs,
        reply_tokens=1000,
        session_checks=False,
    )
    assert [probe.name for probe in proposal.probes] == [
        "routing",
        "tools",
        "media:image",
        "media:audio",
        "media:video",
        "media:pdf",
    ]
    assert proposal.token_estimate == 2 * 1512 + 4 * 5096
    assert all(probe.token_estimate == 5096 for probe in proposal.probes[2:])
    assert proposal.probes[4].skip_reason is None


def test_stage_plan_preview_media_and_check_save_roundtrip(monkeypatch, tmp_path):
    calls = fake_http(monkeypatch)
    base = [
        "model",
        "--api-key-env",
        "NOOA_CONNECT_TEST_API_KEY",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--max-tokens",
        "1000",
        "--test-media-inputs",
        "image,pdf",
    ]
    result = CliRunner().invoke(command, base + ["--stage", "plan"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert [p["name"] for p in data["probes"]] == ["routing", "tools", "media:image", "media:pdf"]
    assert data["entry"]["media_inputs"] == [] and not calls
    assert data["token_estimate"] == 2 * 1512 + 2 * 5096
    result = CliRunner().invoke(command, base[:-1] + ["image", "--stage", "media"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["data"]["entry"]["media_inputs"] == ["image"]
    assert document["checks"]["media:image"]["outcome"] == "passed"
    assert "base64" not in result.stdout
    source = tmp_path / "result.json"
    source.write_text(result.stdout)
    path = tmp_path / "saved.yaml"
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["candidate"]["media_inputs"] == ["image"]
    assert len(calls) == 2


def test_library_existing_entry_declarations_preserved_when_omitted():
    original = {
        "model_name": "openai/model",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "media_inputs": ["audio", "video"],
    }
    assert connect.plan(
        "local", "model", "chat", "https://api.test/v1", "", existing_entry=original
    ).entry["media_inputs"] == ["audio", "video"]


@pytest.mark.parametrize(
    "answer,formats,budget",
    [
        ('{"value":false}', "image", 12000),
        ("", "image", 12000),
        ('{"value":false}', "video", 12000),
        ('{"value":true}', "image", 1),
    ],
)
def test_edit_retest_preserves_prior_permissions_even_if_not_passed(
    monkeypatch, tmp_path, answer, formats, budget
):
    calls = fake_http(monkeypatch, answer)
    path = tmp_path / "models.yaml"
    entry = {
        "model_name": "openai/model",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
        "media_inputs": ["image", "audio", "video"],
        "max_tokens": 1000,
    }
    path.write_text(yaml.safe_dump({"models": {"local": entry}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "local",
            "--output",
            str(path),
            "--test-media-inputs",
            formats,
            "--budget-tokens",
            str(budget),
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == [
        "image",
        "audio",
        "video",
    ]
    assert len(calls) == (0 if budget == 1 else 3)
    if budget != 1:
        text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
        assert f"{'Images' if formats == 'image' else 'Video'}: enabled" in text
        assert f"{'Images' if formats == 'image' else 'Video'}: off" not in text


@pytest.mark.parametrize("answer,budget,count", [("wrong", 12000, 3), ('{"value":true}', 1, 0)])
def test_guided_edit_all_formats_retains_prior_declarations(
    monkeypatch, tmp_path, answer, budget, count
):
    calls = fake_http(monkeypatch, answer)
    path = tmp_path / "models.yaml"
    entry = {
        "model_name": "openai/model",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
        "media_inputs": ["audio", "video"],
        "max_tokens": 1000,
    }
    path.write_text(yaml.safe_dump({"models": {"local": entry}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "local",
            "--output",
            str(path),
            "--budget-tokens",
            str(budget),
            "--max-tokens",
            "1000",
        ],
        # Approval, TEST, keep context/ceiling/levels, declarations, replace, write.
        input="test\ntest\nkeep\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == count
    assert "Formats to TEST" not in result.output
    assert "Media samples: Images, Audio, Video, PDFs" in " ".join(result.output.split())
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["audio", "video"]


@pytest.mark.parametrize("previous", [[], ["image"]])
def test_http_failure_displays_effective_setting_without_changing_evidence(
    monkeypatch, tmp_path, previous
):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        content = next(
            (
                m["content"]
                for m in calls[-1]["messages"]
                if isinstance(m.get("content"), list)
                and any(b.get("type") != "text" for b in m["content"])
            ),
            calls[-1]["messages"][0]["content"],
        )
        return (
            httpx.Response(
                200,
                json=response_body(
                    "chat", '{"value":323}' if "response_format" in calls[-1] else "323"
                ),
            )
            if isinstance(content, str)
            else httpx.Response(
                400, json={"error": {"message": "invalid model", "type": "invalid_request_error"}}
            )
        )

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    if previous:
        path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "local": {
                            "model_name": "openai/model",
                            "api_style": "chat",
                            "api_base": "https://api.test/v1",
                            "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
                            "max_tokens": 1000,
                            "media_inputs": previous,
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
        args = ["--edit-model", "local", "--output", str(path), "--budget-tokens", "12000"]
    else:
        args = options(path)
    captured = []
    original = connect.verdict

    def verdict(entry):
        captured.append(entry["provenance"]["probes"].get("media:image", {}).copy())
        return original(entry)

    monkeypatch.setattr(connect, "verdict", verdict)
    result = CliRunner().invoke(command, args + ["--test-media-inputs", "image", "--yes"])
    assert result.exit_code == 0, result.output
    text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    setting = "Images: enabled (not confirmed by this test)" if previous else "Images: off"
    assert f"Image input: Failed · Service returned HTTP 400; {setting}." in text
    assert (
        "Image input: Failed · Service returned HTTP 400; image support was not confirmed"
        not in text
    )
    assert len(calls) == 3
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == previous
    record = next(record for record in captured if record.get("outcome") == "inconclusive")
    assert record["reason"] == "Service returned HTTP 400; image support was not confirmed"


@pytest.mark.parametrize("yes", [False, True])
def test_test_all_media_includes_real_video_and_saves_passed_temporal_sample(
    monkeypatch, tmp_path, yes
):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        content = next(
            (m["content"] for m in body["messages"] if isinstance(m.get("content"), list)),
            body["messages"][0]["content"],
        )
        answer = '{"value":323}' if "response_format" in body else "323"
        if isinstance(content, list):
            block = content[-1]
            answer = {
                "image_url": '{"value":true}',
                "input_audio": '{"value":true}',
                "video_url": '{"value":true}',
                "file": '{"value":true}',
            }[block["type"]]
            if block["type"] == "video_url":
                import base64

                from nooa.unifiedllm.connect._media import video_fixture

                assert (
                    base64.b64decode(block["video_url"]["url"].split(",", 1)[1]) == video_fixture()
                )
        return httpx.Response(200, json=response_body("chat", answer))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    args = options(path)
    args[args.index("12000")] = "100000"
    captured = []
    original = connect.verdict

    def verdict(entry):
        captured.append(entry["provenance"]["probes"].copy())
        return original(entry)

    monkeypatch.setattr(connect, "verdict", verdict)
    result = CliRunner().invoke(
        command,
        args + (["--test-media-inputs", "image,audio,video,pdf", "--yes"] if yes else []),
        input="test\ntest\nkeep\ny\n",
    )
    assert result.exit_code == 0, result.output
    text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    assert result.output.count("Would you like to test these settings?") == (0 if yes else 1)
    assert "Approve setup" not in result.output
    from nooa.unifiedllm.connect._records import text_baseline_passed

    assert captured and all(text_baseline_passed(facts["routing"]) for facts in captured)
    assert all(
        captured[-1][f"media:{kind}"]["outcome"] == "passed"
        for kind in ["image", "audio", "video", "pdf"]
    )
    assert "response_format" in calls[0]
    for kind in ["Image", "Audio", "Video", "PDF"]:
        assert f"{kind} input: Passed" in text
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == [
        "image",
        "audio",
        "video",
        "pdf",
    ]
    video_requests = [
        body
        for body in calls
        if any(
            isinstance(m.get("content"), list) and m["content"][-1]["type"] == "video_url"
            for m in body["messages"]
        )
    ]
    assert len(video_requests) == 1


def test_final_review_can_disable_retained_enabled_failed_input(monkeypatch, tmp_path):
    calls = fake_http(monkeypatch, "wrong")
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "local": {
                        "model_name": "openai/model",
                        "api_style": "chat",
                        "api_base": "https://api.test/v1",
                        "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
                        "max_tokens": 1000,
                        "media_inputs": ["image"],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    captured = []
    original = connect.verdict

    def verdict(entry):
        captured.append(entry["provenance"]["probes"].get("media:image", {}).copy())
        return original(entry)

    monkeypatch.setattr(connect, "verdict", verdict)
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "local",
            "--output",
            str(path),
            "--test-media-inputs",
            "image",
            "--budget-tokens",
            "12000",
        ],
        input="test\nmanual\n-\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    assert "Images: enabled" in text
    assert "Images: off" in text
    assert "Images: off." in text[text.index("Images: off") :]
    assert len(calls) == 3
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == []
    assert any(
        record.get("outcome") == "inconclusive"
        and record.get("reason") == "Could not read the answer; image support was not confirmed"
        for record in captured
    )


def test_approved_checks_do_not_repeat_charge_or_price_messages(monkeypatch, tmp_path):
    """After initial consent, show progress and token limits, not spending reminders."""
    fake_http(monkeypatch)
    result = CliRunner().invoke(
        command, options(tmp_path / "models.yaml") + ["--yes", "--test-media-inputs", "image"]
    )
    assert result.exit_code == 0, result.output
    assert "Checks may incur charges." not in result.output
    assert "Uses the approved token budget" not in result.output
    progress = result.output.split("Checking connection", 1)[1]
    for wording in ("charges", "spending cap", "billing", "estimated price", "catalogue prices"):
        assert wording not in progress
    for boilerplate in (
        "budget remaining:",
        "Estimated tokens:",
        "reasoning settings",
        "Generated image sample:",
        "Check plan",
        "Checks use your saved reply limits",
        "Then 3 conversation",
    ):
        assert boilerplate not in progress
    assert "Image input" in progress


@pytest.mark.parametrize(
    "video_answer,cause",
    [
        ('{"value":false}', "Model cannot access this input"),
        ("private-invalid-schema-answer", "Could not read the answer"),
    ],
)
@pytest.mark.parametrize(
    "previous,manual,enabled",
    [
        ([], False, ["image", "audio", "pdf"]),
        (["video"], False, ["video", "image", "audio", "pdf"]),
        ([], True, ["video"]),
    ],
)
def test_typed_test_all_failed_video_permission_is_accurate(
    monkeypatch, tmp_path, video_answer, cause, previous, manual, enabled
):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        blocks = [
            b
            for m in body["messages"]
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if b.get("type") != "text"
        ]
        answer = '{"value":323}' if "response_format" in body else "323"
        if blocks:
            assert len(blocks) == 1
            answer = {
                "image_url": '{"value":true}',
                "input_audio": '{"value":true}',
                "video_url": video_answer,
                "file": '{"value":true}',
            }[blocks[0]["type"]]
        return httpx.Response(200, json=response_body("chat", answer))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    if previous:
        path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "local": {
                            "model_name": "openai/model",
                            "api_style": "chat",
                            "client_type": "completion",
                            "api_base": "https://api.test/v1",
                            "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
                            "max_tokens": 1000,
                            "media_inputs": previous,
                        }
                    }
                }
            )
        )
    args = options(path)
    args[args.index("12000")] = "100000"
    if previous:
        monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
        args = [
            "--edit-model",
            "local",
            "--output",
            str(path),
            "--budget-tokens",
            "100000",
            "--max-tokens",
            "1000",
        ]
    if manual:
        monkeypatch.setattr(prompts, "choose_media_inputs", lambda current=(): ["video"])
    result = CliRunner().invoke(
        command,
        args + ["--test-media-inputs", "image,audio,video,pdf"] + ([] if manual else ["--yes"]),
        input="test\nmanual\ny\n" if manual else None,
    )
    assert result.exit_code == 0, result.output
    text = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    assert (
        f"Video input: {'Not available · The model reported no access to this sample' if cause == 'Model cannot access this input' else 'Failed · ' + cause}"
        in text
    )
    assert (
        "Video: enabled"
        if previous or manual
        else "Video: off"
        if cause == "Model cannot access this input"
        else "Video: off"
    ) in text
    assert "Request accepted;" not in text and "understanding not confirmed" not in text
    assert "private-invalid-schema-answer" not in text
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert saved["media_inputs"] == enabled
    assert "provenance" not in saved
    video = [
        b
        for body in calls
        for m in body["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "video_url"
    ]
    assert len(video) == 1


@pytest.mark.parametrize(
    "answer,passed",
    [('{"value":true}', True), ('{"value":false}', False), ("bad-schema-private", False)],
)
@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
def test_typed_media_stage_json_acceptance_and_native_validation(
    monkeypatch, tmp_path, answer, passed, kind
):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json=response_body("chat", '{"value":323}' if len(calls) == 1 else answer)
        )

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "NOOA_CONNECT_TEST_API_KEY",
            "--max-tokens",
            "1000",
            "--budget-tokens",
            "10000",
            "--test-media-inputs",
            kind,
            "--stage",
            "media",
        ],
    )
    assert result.exit_code == (0 if passed else 1), result.output
    document = json.loads(result.stdout)
    record = document["checks"][f"media:{kind}"]
    assert len(calls) == 2
    assert record["request_accepted"] is True
    assert record["attachment_observed"] is True
    assert record["output_settings_observed"] is True
    assert record["typed_result_validated"] is (answer != "bad-schema-private")
    assert record.get("access_reported") is (None if answer == "bad-schema-private" else passed)
    assert "answer_correct" not in record
    assert document["data"]["entry"]["media_inputs"] == ([kind] if passed else [])
    assert "base64" not in result.stdout and "bad-schema-private" not in result.stdout
    source = tmp_path / "result.json"
    source.write_text(result.stdout)
    path = tmp_path / "saved.yaml"
    save = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(path)]
    )
    assert save.exit_code == 0, save.output
    assert yaml.safe_load(path.read_text())["models"]["candidate"]["media_inputs"] == (
        [kind] if passed else []
    )


@pytest.mark.parametrize("previous", [[], ["audio", "video"]])
def test_test_all_boolean_access_summary_and_retained_permissions(monkeypatch, tmp_path, previous):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        blocks = [
            b
            for m in body["messages"]
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if b.get("type") != "text"
        ]
        if blocks:
            assert len(blocks) == 1
            answer = json.dumps({"value": blocks[0]["type"] in {"image_url", "file"}})
        else:
            answer = '{"value":323}' if "response_format" in body else "323"
        return httpx.Response(200, json=response_body("chat", answer))

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    args = options(path)
    args[args.index("12000")] = "100000"
    if previous:
        path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "local": {
                            "model_name": "openai/model",
                            "api_style": "chat",
                            "api_base": "https://api.test/v1",
                            "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
                            "max_tokens": 1000,
                            "media_inputs": previous,
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
        args = [
            "--edit-model",
            "local",
            "--output",
            str(path),
            "--budget-tokens",
            "100000",
            "--max-tokens",
            "1000",
        ]
    result = CliRunner().invoke(
        command, args + ["--test-media-inputs", "image,audio,video,pdf", "--yes"]
    )
    assert result.exit_code == 0, result.output
    output = " ".join(re.sub(r" · \d+\.\ds", "", result.output).split())
    for kind in ["Audio", "Video"]:
        assert (
            f"{kind} input: Not available · The model reported no access to this sample" in output
        )
        assert f"✓ {kind}" not in output and f"{kind} input: Failed" not in output
    assert "Not available in this test: audio, video" in output
    assert "Keep these settings" in output and "choose another model or API" in output
    assert "Review media inputs" not in output
    assert "Wrong answer" not in output and "understanding not confirmed" not in output
    expected = previous + ["image", "pdf"]
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == expected
    if previous:
        assert "Audio: enabled (not tested)" in output
    else:
        assert "Audio: off" in output
    assert (
        sum(
            any(
                isinstance(m.get("content"), list)
                and any(b.get("type") != "text" for b in m["content"])
                for m in body["messages"]
            )
            for body in calls
        )
        == 4
    )


@pytest.mark.parametrize(
    "change,flag,expected",
    [
        (None, None, ["image", "pdf"]),
        (None, "", []),
        ("model", None, []),
        ("endpoint", None, []),
        ("api-style", None, []),
    ],
)
def test_ordinary_yes_reconnect_route_local_omission(monkeypatch, tmp_path, change, flag, expected):
    path = tmp_path / "models.yaml"
    entry = {
        "model_name": "openai/model",
        "api_style": "chat",
        "api_base": "https://api.test/v1",
        "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
        "media_inputs": ["image", "pdf"],
        "max_tokens": 1000,
    }
    path.write_text(yaml.safe_dump({"models": {"local": entry}}))
    args = options(path) + ["--yes", "--no-probe"]
    if change == "model":
        args[0] = "other"
    elif change:
        key = "--" + change
        args[args.index(key) + 1] = "https://other.test/v1" if change == "endpoint" else "responses"
    if flag is not None:
        args += ["--media-inputs", flag]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == expected


@pytest.mark.parametrize("answer", ['{"value":false}', "invalid"])
def test_ordinary_yes_reconnect_failed_retest_retains_permissions(monkeypatch, tmp_path, answer):
    fake_http(monkeypatch, answer)
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "local": {
                        "model_name": "openai/model",
                        "api_style": "chat",
                        "api_base": "https://api.test/v1",
                        "api_key_env": "NOOA_CONNECT_TEST_API_KEY",
                        "media_inputs": ["image", "pdf"],
                        "max_tokens": 1000,
                    }
                }
            }
        )
    )
    result = CliRunner().invoke(command, options(path) + ["--yes", "--test-media-inputs", "image"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["image", "pdf"]
