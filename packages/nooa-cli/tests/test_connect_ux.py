# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole normal setup and actual choice renders; HTTP is mocked below NOOA/SDK."""

import json
import os
import re

import click
import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands.connect import command
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts.choice_input import ChoiceInput

from nooa import llm_config
from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.mark.parametrize("columns", [80, 120])
@pytest.mark.parametrize("save", [False, True])
def test_normal_wizard_transcript(monkeypatch, tmp_path, columns, save):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((columns, 40))
    )

    # Fix display time only, not runtime deadlines or request evidence.
    class Clock:
        @staticmethod
        def monotonic():
            return 7.0

    monkeypatch.setattr(view, "time", Clock)
    requests = []
    metadata = {
        "id": "model",
        "context_length": 128000,
        "top_provider": {"max_completion_tokens": 65536},
        "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "low"},
    }

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            assert (
                "temporary-secret" not in request.headers.get("authorization", "")
                if request.url.host == "openrouter.ai"
                else True
            )
            return httpx.Response(200, json={"data": [metadata]})
        assert request.url.host == "api.test"
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        body = json.loads(request.content)
        media = next(
            (
                b
                for m in body["messages"]
                if isinstance(m["content"], list)
                for b in m["content"]
                if b["type"] != "text"
            ),
            None,
        )
        text = '{"value":323}' if "response_format" in body else "323"
        if media:
            text = '{"value":false}' if media["type"] == "input_audio" else '{"value":true}'
        result = response_body("chat", text)
        message = result["choices"][0]["message"]
        message["reasoning_content"] = "private-reasoning"
        if body.get("tools"):
            message["tool_calls"] = [
                {
                    "id": "call-test",
                    "type": "function",
                    "function": {"name": "probe_tool", "arguments": '{"value":"ok"}'},
                }
            ]
            result["choices"][0]["finish_reason"] = "tool_calls"
        result["usage"]["prompt_tokens_details"] = {"cached_tokens": 18}
        return httpx.Response(200, json=result)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        ["--output", str(path)],
        input=(
            "y\ncustom\nhttps://api.test/v1\nnew\nUX_KEY\ntemporary-secret\n"
            "test\nmodel\nuse\nrecommended\nkeep\nlocal\n" + ("y\nn\n" if save else "n\n")
        ),
    )
    assert result.exit_code == 0, result.output
    output = result.output.replace(str(path), "models.yaml").replace(
        str(tmp_path / "user" / "secrets.yaml"), "secrets.yaml"
    )
    plain = " ".join(output.split())
    (tmp_path / "transcript.txt").write_text(output)
    (tmp_path / "transcript-normalized.txt").write_text(plain + "\n")
    checkpoints = [
        "NOOA / CONNECT",
        "Approve setup",
        "Choose a provider",
        "Model server URL",
        "Key environment variable",
        "API key",
        "configure media inputs",
        "Search advertised model IDs",
        "OpenAI-compatible Chat: Passed",
        "Model details",
        "Model settings",
        "Reply limit",
        "Connection: Passed",
        "Tool use: Passed",
        "Reasoning (low): Passed",
        "Reasoning (high): Passed",
        "Image input: Passed",
        "Audio input: Not available",
        "Video input: Passed",
        "PDF input: Passed",
        "Conversation (1/3): Passed",
        "Conversation (2/3): Passed",
        "Conversation (3/3): Passed",
        "Cache reuse: Passed",
        "Reasoning retention: Passed",
        "Results",
        "Save model",
        "Enabled media inputs",
        "How would you like to continue?",
        "Save this model as",
        "Save summary",
        "Write model entry",
    ]
    positions = [plain.index(item) for item in checkpoints]
    assert positions == sorted(positions), output
    assert "No tool call returned" not in plain
    assert "temporary-secret" not in output and "private-reasoning" not in output
    assert "reasoning chars" not in output and "Reasoning tokens ·" not in output
    assert plain.count("Checks may incur charges.") == 1
    rows = [row for row in output.splitlines() if re.match(r"  [✓!–] ", row)]
    assert rows and all(
        re.search(r": (Passed|Not available|Failed|Skipped)( · 0.0s)?$", row) for row in rows
    )
    assert all(len(row) <= columns for row in rows)
    assert path.exists() is save
    assert not (tmp_path / "user" / "secrets.yaml").exists()
    if save:
        entry = yaml.safe_load(path.read_text())["models"]["local"]
        assert entry["media_inputs"] == ["image", "video", "pdf"]
        assert entry["max_tokens"] == 32768
    assert len([r for r in requests if r.method == "GET"]) == 2


@pytest.mark.parametrize("columns", [80, 120])
@pytest.mark.parametrize(
    "case",
    [
        "provider",
        "saved_model",
        "saved_key",
        "api",
        "retry",
        "metadata",
        "catalogue",
        "reasoning",
        "reply",
        "media",
        "review",
        "manual",
    ],
)
def test_choices_reach_actual_renderer(monkeypatch, columns, case):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    original = ChoiceInput._create_application
    screens = []
    options = {
        "provider": (
            "Choose a provider",
            tuple(connect.PROVIDERS) + ("custom",),
            None,
            {**{k: v.label for k, v in connect.PROVIDERS.items()}, "custom": "Custom endpoint"},
        ),
        "saved_model": ("Model to edit", ("work", "other"), None, None),
        "saved_key": ("Saved key variable", ("KEY", "-"), None, {"-": "No authentication"}),
        "api": ("API format", ("chat", "responses", "anthropic"), "chat", None),
        "retry": (
            "Next step",
            ("key", "server", "retry", "longer", "cancel"),
            "key",
            {
                "key": "Change key",
                "server": "Edit server and model",
                "retry": "Try again unchanged",
                "longer": "Retry one API format",
                "cancel": "Exit without saving",
            },
        ),
        "metadata": (
            "Model settings",
            ("use", "edit", "skip", "cancel"),
            "use",
            {
                "use": "Use these settings",
                "edit": "Edit settings",
                "skip": "Continue without these settings",
                "cancel": "Cancel setup",
            },
        ),
        "catalogue": ("Catalogue model", ("", "vendor/model"), "", None),
        "reasoning": ("Default reasoning level", ("low", "high", "-"), "low", None),
    }

    class Output(DummyOutput):
        def get_size(self):
            return Size(rows=40, columns=columns)

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=Output()):

        def create(chooser):
            app = original(chooser)
            sent = False

            def rendered(_):
                nonlocal sent
                screen = app.renderer.last_rendered_screen
                if screen is not None:
                    screens.append(
                        "\n".join(
                            "".join(row[x].char for x in sorted(row))
                            for _, row in sorted(screen.data_buffer.items())
                        )
                    )
                if not sent:
                    sent = True
                    pipe.send_text("\x1b[B\r")

            app.after_render += rendered
            return app

        monkeypatch.setattr(ChoiceInput, "_create_application", create)
        if case in options:
            text, values, default, labels = options[case]
            selected = prompts.prompt(text, choices=values, default=default, labels=labels)
            assert selected == values[0 if default is None else values.index(default) + 1]
        elif case == "reply":
            assert prompts.choose_reply_limit(32768) == 65536
        elif case == "media":
            assert prompts.choose_media_mode() == "manual"
        elif case == "review":
            assert prompts.review_media_inputs(["image"]) == "manual"
        else:
            assert prompts.choose_media_inputs() == ["image"]
    assert screens and "Enter Continue" in screens[0]
    assert "→ to edit" not in screens[0] and "Enter to use" not in screens[0]
    assert all(len(line) <= columns for screen in screens for line in screen.splitlines())
    for token in ("recommended:", "highflag", "keep:", "test:"):
        assert token not in screens[0]
    if case in options:
        _, values, _, labels = options[case]
        for value in values:
            assert (labels or {}).get(value, prompts.choice_label(value)) in " ".join(
                screens[0].split()
            )
    elif case == "reply":
        for label in (
            "Recommended — 32,768 tokens",
            "High reasoning budget — 65,536 tokens",
            "Custom…",
        ):
            assert label in " ".join(screens[0].split())
    elif case == "manual":
        assert "Text only" in screens[0] and "Image, Audio, Video, PDF" in screens[0]


def test_normal_mode_keeps_technical_reasoning_details_optional():
    @click.command()
    def display():
        progress = view.CheckProgress()
        progress.update(
            "level:high",
            {
                "outcome": "accepted",
                "reasoning_observed": True,
                "answer_correct": True,
                "reasoning_encrypted": True,
                "reasoning_encrypted_bytes": 400,
                "output_tokens": 500,
            },
        )
        progress.finish()

    result = CliRunner().invoke(display)
    assert result.exit_code == 0
    assert "Reasoning (high): Passed" in result.output
    assert "bytes" not in result.output and "Reasoning tokens" not in result.output


@pytest.mark.parametrize("keys,expected", [("\r", ["pdf", "image"]), ("\x03", None)])
def test_manual_media_default_retains_order_or_aborts(monkeypatch, keys, expected):
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        if expected is None:
            with pytest.raises(click.Abort):
                prompts.choose_media_inputs(["pdf", "image"])
        else:
            assert prompts.choose_media_inputs(["pdf", "image"]) == expected


def test_no_default_requires_selection_and_custom_budget_is_editable(monkeypatch):
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    original = ChoiceInput._create_application
    screens = []
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):

        def create(chooser):
            app = original(chooser)

            def rendered(_):
                screen = app.renderer.last_rendered_screen
                if screen is None:
                    return
                if app not in screens:
                    screens.append(app)
                    if len(screens) == 1:
                        pipe.send_text("\r")  # Neither absence nor Enter chooses a provider.
                    else:
                        pipe.send_text("\x1b[B\r")

            app.after_render += rendered
            return app

        monkeypatch.setattr(ChoiceInput, "_create_application", create)
        assert (
            prompts.prompt("Provider", choices=("custom",), labels={"custom": "Custom endpoint"})
            == "custom"
        )
        assert len(screens) == 2
    # A real labelled preset menu followed by the real editable numeric input.
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        monkeypatch.setattr(ChoiceInput, "_create_application", original)
        pipe.send_text("\x1b[B" * 5 + "\r12345\r")
        assert prompts.choose_reply_limit(32768) == 12345


@pytest.mark.parametrize(
    "name,record,expected",
    [
        ("routing", {"outcome": "accepted"}, "Failed"),
        (
            "tools",
            {"outcome": "accepted", "tool_observed": True, "finish_reason": "tool_calls"},
            "Passed",
        ),
        ("tools", {"outcome": "accepted", "tool_observed": False}, "Failed"),
        ("session:seed", {"outcome": "accepted", "finish_reason": "tool_calls"}, "Passed"),
        ("cache", {"outcome": "warning", "reason": "No cache reuse reported"}, "Not available"),
        ("cache", {"outcome": "not_confirmed", "reason": "Earlier messages changed"}, "Failed"),
        (
            "reasoning_retention",
            {"outcome": "not_confirmed", "reason": "No replayable reasoning returned"},
            "Not available",
        ),
        (
            "reasoning_retention",
            {"outcome": "not_confirmed", "reason": "Reasoning was returned but not preserved"},
            "Failed",
        ),
        ("media:image", {"outcome": "unavailable"}, "Not available"),
        ("media:audio", {"outcome": "not_probed", "reason": "budget exhausted"}, "Skipped"),
        ("media:pdf", {"outcome": "not_probed", "error": "AuthenticationError"}, "Failed"),
    ],
)
def test_display_outcomes_do_not_upgrade_unconfirmed_evidence(name, record, expected):
    from nooa.unifiedllm.connect._records import check_status

    original = record.copy()
    assert view.check_outcome(name, record, check_status(name, record)) == expected
    assert record == original
