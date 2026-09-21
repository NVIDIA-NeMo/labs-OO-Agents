# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry-based editing and credential selection never need discovery."""

import os

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    from nooa import llm_config

    monkeypatch.setattr(
        _connect_prompts, "choose_reply_limit", lambda suggested, ceiling, **kw: suggested
    )

    path = tmp_path / "llm_config.yaml"
    entry = {
        "model_name": "openai/vendor/model",
        "api_style": "chat",
        "transport": "litellm",
        "api_base": "https://api.test/v1",
        "api_key_env": "EDIT_TEST_KEY",
        "context_window": 50000,
        "max_tokens": 8192,
        "reasoning_levels": {"high": {"chat_template_kwargs": {"enable_thinking": True}}},
        "reasoning_default": "high",
        "extra_body": {"custom": 123},
        "custom_metadata": "keep me",
        "cache_breakpoint": None,
    }
    path.write_text(yaml.safe_dump({"models": {"saved": entry}}))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    monkeypatch.setenv("EDIT_TEST_KEY", "test-key")
    return path, entry


@pytest.mark.parametrize("selector", [[], ["saved"]])
def test_edit_jumps_to_settings_preserving_custom_fields(registry, monkeypatch, selector):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    from nooa.unifiedllm import connect

    path, original = registry

    async def forbidden(*args, **kwargs):
        raise AssertionError("Editing does not discover models or fetch metadata")

    monkeypatch.setattr(connect, "discover", forbidden)
    monkeypatch.setattr(connect, "catalogue", forbidden)

    def edit(data):
        assert data["context_length"] == 50000
        data["context_length"] = 64000
        return data

    monkeypatch.setattr(_connect_prompts, "edit_model_details", edit)
    result = CliRunner().invoke(
        command,
        ["--edit-model", *selector, "--no-probe"],
        input=("saved\n" if not selector else "") + "use\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    actual = yaml.safe_load(path.read_text())["models"]["saved"]
    assert actual["context_window"] == 64000
    for key in (
        "model_name",
        "transport",
        "max_tokens",
        "reasoning_levels",
        "extra_body",
        "custom_metadata",
        "cache_breakpoint",
    ):
        assert actual[key] == original[key]
    assert "Choose a provider" not in result.output
    assert "Model server URL" not in result.output


def test_edit_cancel_preserves_file(registry, monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    path, _ = registry
    before = path.read_bytes()
    monkeypatch.setattr(_connect_prompts, "edit_model_details", lambda data: data)
    result = CliRunner().invoke(command, ["--edit-model", "saved", "--no-probe"], input="cancel\n")
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before
    assert not path.with_name("secrets.yaml").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_endpoint_reuses_saved_variable_unless_explicit(registry, explicit):
    path, _ = registry
    argv = [
        "new-model",
        "--as",
        "new-alias",
        "--endpoint",
        "https://api.test/v1/",
        "--api-style",
        "chat",
        "--no-probe",
        "--no-catalogue",
        "--yes",
    ]
    if explicit:
        argv += ["--api-key-env", "EXPLICIT_KEY"]
    result = CliRunner().invoke(command, argv)
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["new-alias"]
    assert entry["api_key_env"] == ("EXPLICIT_KEY" if explicit else "EDIT_TEST_KEY")
    assert "test-key" not in result.output


@pytest.mark.parametrize("choice", ["y", "n"])
def test_pasted_key_is_saved_only_with_separate_consent(registry, monkeypatch, choice):
    path, _ = registry
    monkeypatch.delenv("NEW_TEST_KEY", raising=False)
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "new",
            "--endpoint",
            "https://other.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "NEW_TEST_KEY",
            "--prompt-key",
            "--no-probe",
            "--no-catalogue",
        ],
        input=f"private-value\ny\n{choice}\n",
    )
    assert result.exit_code == 0, result.output
    secret = path.with_name("secrets.yaml")
    assert secret.exists() == (choice == "y")
    if choice == "y":
        assert yaml.safe_load(secret.read_text()) == {"env": {"NEW_TEST_KEY": "private-value"}}
        assert secret.stat().st_mode & 0o777 == 0o600
    assert "private-value" not in result.output + path.read_text()
    assert "NEW_TEST_KEY" not in os.environ


def test_registry_matching_does_not_cross_server_paths(registry):
    from nooa_cli.commands._connect_registry import credential_names

    path, entry = registry
    assert credential_names({"a": (entry, path)}, "https://api.test") == ["EDIT_TEST_KEY"]
    assert credential_names({"a": (entry, path)}, "https://api.test/other/v1") == []
    assert credential_names({"a": (entry, path)}, "https://different.test/v1") == []


def test_selected_endpoint_key_reaches_runtime_http(registry, monkeypatch):
    import httpx

    from tests.unifiedllm.connect.connect_http import mock_http, response_body

    sent = []

    def handle(request):
        sent.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=response_body("chat"))

    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [
            "new-model",
            "--as",
            "new",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--no-catalogue",
            "--probe",
            "minimal",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 1


def test_final_save_cancel_does_not_persist_pasted_key(registry):
    path, _ = registry
    before = path.read_bytes()
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "new",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "NEW_TEST_KEY",
            "--prompt-key",
            "--no-probe",
            "--no-catalogue",
        ],
        input="private-value\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before
    assert not path.with_name("secrets.yaml").exists()


def test_edit_model_with_working_dir_warns_when_source_is_not_the_target(
    registry, tmp_path, monkeypatch
):
    """--edit-model finds an alias via entries(), which only redirects to
    the --working-dir target's own file when that file already exists on
    disk (it cannot read a file that isn't there yet) -- otherwise the
    alias comes from the general chain instead. Saving that into the
    working-dir target without saying so looks like an in-place edit of the
    target's own prior settings, when it is actually a copy of a different
    file's data into a project that never defined this alias.
    """
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    from nooa.unifiedllm import connect

    path, _ = registry

    async def forbidden(*args, **kwargs):
        raise AssertionError("Editing does not discover models or fetch metadata")

    monkeypatch.setattr(connect, "discover", forbidden)
    monkeypatch.setattr(connect, "catalogue", forbidden)
    monkeypatch.setattr(
        _connect_prompts, "choose_reply_limit", lambda suggested, ceiling, **kw: suggested
    )

    target = tmp_path / "target_project"
    target.mkdir()
    result = CliRunner().invoke(
        command,
        [
            "--edit-model",
            "saved",
            "--working-dir",
            str(target),
            "--no-probe",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Editing saved from" in result.output
    assert "does not define" in result.output and "yet" in result.output
    assert "copying it in from" in result.output
    saved = yaml.safe_load((target / ".nooa" / "llm_config.yaml").read_text())
    assert saved["models"]["saved"]["model_name"] == "openai/vendor/model"
