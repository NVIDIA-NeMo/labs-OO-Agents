# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The guided proposal's single "Change settings" submenu."""

import httpx
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands.connect import command

from nooa import llm_config
from nooa.unifiedllm import connect, registry
from tests.unifiedllm.connect.connect_http import mock_http


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("UX_KEY", "PRIVATE-KEY")
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", False)

    async def catalogue():
        return []

    monkeypatch.setattr(connect, "catalogue", catalogue)


@pytest.fixture
def menus(monkeypatch):
    seen = []
    original = prompts.labelled_choice

    def choose(message, options, **kw):
        seen.append((message, tuple(options), kw.get("default")))
        return original(message, options, **kw)

    monkeypatch.setattr(prompts, "labelled_choice", choose)
    return seen


def submenus(seen):
    return [
        (options, default)
        for message, options, default in seen
        if message == "What would you like to change?"
    ]


def chat_args(path, *extra):
    return [
        "wire-model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "UX_KEY",
        "--no-catalogue",
        "--media-inputs",
        "",
        "--reply-tokens",
        "4096",
        "--output",
        str(path),
        *extra,
    ]


def no_posts(monkeypatch):
    def handle(request):
        assert request.method == "GET", "Editing settings must not call the model"
        return httpx.Response(200, json={"data": []})

    mock_http(monkeypatch, handle)


def native_model():
    return {
        "id": "claude-opus-4-7",
        "max_input_tokens": 32768,
        "max_tokens": 8192,
        "capabilities": {
            "structured_outputs": {"supported": True},
            "thinking": {
                "supported": True,
                "types": {
                    "enabled": {"supported": True},
                    "adaptive": {"supported": False},
                    "disabled": {"supported": True},
                },
            },
            "effort": {"supported": False},
        },
    }


def native_args(path):
    return [
        "claude-opus-4-7",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "anthropic",
        "--api-key-env",
        "UX_KEY",
        "--media-inputs",
        "",
        "--reply-tokens",
        "4096",
        "--probe",
        "minimal",
        "--output",
        str(path),
    ]


def test_chat_submenu_lists_plain_options_in_order(monkeypatch, tmp_path, menus):
    no_posts(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, chat_args(path), input="32768\n32768\nchange\nback\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert submenus(menus) == [
        (
            (
                ("limits", "Reply limit and context"),
                ("reasoning", "Reasoning"),
                ("media", "Media"),
                ("structured", "Answer format"),
                ("back", "Back"),
            ),
            "limits",
        )
    ]


def test_catalogue_route_offers_continue_without_catalogue_before_back(
    monkeypatch, tmp_path, menus
):
    async def catalogue():
        return [
            {
                "id": "wire-model",
                "context_length": 64000,
                "top_provider": {"max_completion_tokens": 8192},
            }
        ]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    no_posts(monkeypatch)
    path = tmp_path / "models.yaml"
    options = [arg for arg in chat_args(path) if arg != "--no-catalogue"]
    result = CliRunner().invoke(command, options, input="change\nback\nsave\ny\n")
    assert result.exit_code == 0, result.output
    ((choices, default),) = submenus(menus)
    assert [value for value, _ in choices] == [
        "limits",
        "reasoning",
        "media",
        "structured",
        "skip",
        "back",
    ]
    assert default == "limits"


def test_decision_submenu_offers_media_then_back(monkeypatch, tmp_path, menus):
    path = tmp_path / "models.yaml"
    original = {
        "model_name": "fixture",
        "client_type": "decision",
        "api_style": "openai",
        "endpoint": "https://api.test/decisions",
        "api_key_env": "",
        "media_inputs": [],
        "decision_image_transport": "json",
        "typesafe_danger_zone_images": False,
    }
    path.write_text(yaml.safe_dump({"models": {"local": original}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    mock_http(monkeypatch, lambda request: pytest.fail("Offline edit must not send HTTP"))
    result = CliRunner().invoke(
        command,
        ["--edit-model", "local", "--no-catalogue", "--no-probe"],
        input="change\nmedia\nmanual\nsave\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert submenus(menus) == [((("media", "Media"), ("back", "Back")), "media")]
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["image"]


def test_back_returns_to_the_proposal_with_nothing_changed(monkeypatch, tmp_path, menus):
    no_posts(monkeypatch)
    plain, back = tmp_path / "plain.yaml", tmp_path / "back.yaml"
    first = CliRunner().invoke(command, chat_args(plain), input="32768\n32768\nsave\ny\n")
    second = CliRunner().invoke(
        command, chat_args(back), input="32768\n32768\nchange\nback\nsave\ny\n"
    )
    assert first.exit_code == second.exit_code == 0, second.output
    assert yaml.safe_load(back.read_text()) == yaml.safe_load(plain.read_text())
    assert "Proposed configuration (updated)" not in second.output
    tops = [m for m, _, _ in menus if m == "Would you like to test these settings?"]
    assert len(tops) == 3  # first run once; second run before and after Back


def test_media_edit_enables_a_newly_picked_format(monkeypatch, tmp_path):
    no_posts(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        chat_args(path),
        input="32768\n32768\nchange\nmedia\nimage\nenable\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    updated = result.output.index("Proposed configuration (updated)")
    assert "Images: Enabled (not tested)" in " ".join(result.output[updated:].split())
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["image"]


def test_media_edit_test_choice_plans_a_media_check(monkeypatch, tmp_path):
    no_posts(monkeypatch)
    path = tmp_path / "models.yaml"
    # End of input at the next decision cancels; nothing is tested or saved.
    result = CliRunner().invoke(
        command, chat_args(path), input="32768\n32768\nchange\nmedia\nimage\ntest\n"
    )
    assert result.exit_code != 0
    assert not path.exists()
    updated = " ".join(
        result.output[result.output.index("Proposed configuration (updated)") :].split()
    )
    assert "Images: Test before enabling" in updated
    assert "Media samples: Images" in updated


def test_reasoning_only_change_does_not_reask_limits(monkeypatch, tmp_path):
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [native_model()]})

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, native_args(path), input="default\nchange\nreasoning\non\n2048\nsave\ny\n"
    )
    assert result.exit_code == 0, result.output
    after = result.output[result.output.index("What would you like to change?") :]
    assert "Context window (tokens)" not in after
    assert "Reply limit [" not in after
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["reasoning_level"] == "on"
    assert entry["max_tokens"] == 4096


def test_limits_change_does_not_reask_reasoning(monkeypatch, tmp_path):
    no_posts(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        chat_args(path),
        input="32768\n32768\nchange\nlimits\n16384\n16384\ncustom\n2048\nsave\ny\n",
    )
    assert result.exit_code == 0, result.output
    after = result.output[result.output.index("What would you like to change?") :]
    assert "Reasoning: Model default (controls unknown)." not in after
    assert yaml.safe_load(path.read_text())["models"]["local"]["max_tokens"] == 2048


HEADERS = (
    "1 / 5 · Connection",
    "2 / 5 · Model",
    "3 / 5 · Settings",
    "4 / 5 · Test",
    "5 / 5 · Save model",
)


def headers(output):
    plain = " ".join(output.split())
    return [h for h in HEADERS if h in plain], [plain.count(h) for h in HEADERS], plain


def test_step_headers_once_and_in_order_for_test_and_save(monkeypatch, tmp_path):
    from tests.unifiedllm.connect.connect_http import response_body

    mock_http(monkeypatch, lambda request: httpx.Response(200, json=response_body("chat")))
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        chat_args(path, "--probe", "minimal"),
        input="32768\n32768\ntest\ny\n",
    )
    assert result.exit_code == 0, result.output
    shown, counts, plain = headers(result.output)
    assert shown == list(HEADERS) and counts == [1] * 5, result.output
    positions = [plain.index(h) for h in HEADERS]
    assert positions == sorted(positions)
    # Settings covers interface, media, reasoning and the proposal; Test covers the checks.
    assert positions[1] < plain.index("Model details") < positions[2]
    assert positions[2] < plain.index("Proposed configuration") < positions[3]
    assert positions[3] < plain.index("Checking connection") < positions[4]


def test_step_headers_skip_test_but_keep_the_total_when_saving_untested(monkeypatch, tmp_path):
    no_posts(monkeypatch)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, chat_args(path), input="32768\n32768\nsave\ny\n")
    assert result.exit_code == 0, result.output
    shown, counts, plain = headers(result.output)
    assert shown == [h for h in HEADERS if "Test" not in h], result.output
    assert counts == [1, 1, 1, 0, 1]
    assert "/ 4 ·" not in plain
