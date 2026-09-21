# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from tests.unifiedllm.connect.connect_http import mock_http, response_body


@pytest.fixture(autouse=True)
def sdk_credentials(monkeypatch):
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    from nooa import llm_config

    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    # Reply-budget interaction has its own contract tests.
    monkeypatch.setattr(
        _connect_prompts, "choose_reply_limit", lambda suggested, ceiling, **kw: suggested
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def args(path):
    return [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "CONNECT_TEST_KEY",
        "--no-catalogue",
        "--no-probe",
        "--output",
        str(path),
    ]


def test_working_dir_saves_under_its_nooa_directory_like_the_tui(tmp_path):
    workspace = tmp_path / "myproject"
    workspace.mkdir()
    options = [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "CONNECT_TEST_KEY",
        "--no-catalogue",
        "--no-probe",
        "--working-dir",
        str(workspace),
        "--yes",
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 0, result.output
    target = workspace / ".nooa" / "llm_config.yaml"
    assert target.exists()
    assert (
        yaml.safe_load(target.read_text())["models"]["local"]["api_base"] == "https://api.test/v1"
    )


def test_working_dir_and_output_are_mutually_exclusive(tmp_path):
    workspace = tmp_path / "myproject"
    workspace.mkdir()
    options = [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--working-dir",
        str(workspace),
        "--output",
        str(tmp_path / "explicit.yaml"),
        "--yes",
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_working_dir_with_a_non_save_stage_is_a_clear_usage_error(tmp_path):
    """--working-dir computes an --output path, but every stage other than
    save rejects any --output at all (those stages just emit JSON and never
    write a registry file). Without this check that combination crashed with
    a generic, confusing "--input and --output are for stage save" error.
    """
    workspace = tmp_path / "myproject"
    workspace.mkdir()
    options = [
        "wire/model",
        "--stage",
        "routing",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--working-dir",
        str(workspace),
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 2
    assert "--working-dir only applies to" in result.output


def test_working_dir_with_stage_save_still_works(tmp_path):
    workspace = tmp_path / "myproject"
    workspace.mkdir()
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "alias": "local",
                "entry": {
                    "model_name": "wire/model",
                    "api_base": "https://api.test/v1",
                    "api_style": "chat",
                    "api_key_env": "CONNECT_TEST_KEY",
                    "client_type": "completion",
                },
            }
        )
    )
    options = [
        "--stage",
        "save",
        "--input",
        str(plan_file),
        "--working-dir",
        str(workspace),
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 0, result.output
    assert (workspace / ".nooa" / "llm_config.yaml").exists()


def test_working_dir_expands_a_literal_tilde(tmp_path, monkeypatch):
    """click's exists=True check on --working-dir used to run on the raw,
    unexpanded argument, before the command's own expanduser() call — so a
    literal "~" was rejected as nonexistent even though it's a real
    directory. --working-dir no longer declares exists=True; existence is
    checked explicitly after expansion instead.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    options = [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "CONNECT_TEST_KEY",
        "--no-catalogue",
        "--no-probe",
        "--working-dir",
        "~",
        "--yes",
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".nooa" / "llm_config.yaml").exists()


def test_working_dir_reports_a_missing_directory_clearly():
    options = [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--working-dir",
        "/no/such/directory/at/all",
        "--yes",
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_working_dir_reports_a_file_as_not_a_directory(tmp_path, monkeypatch):
    """click's own file_okay=False check only runs when its raw, unexpanded
    argument happens to already exist as given — a "~"-relative path skips
    it entirely (click never expands "~" itself), so a real file reached
    only through "~" used to be misreported as "does not exist" instead of
    "is not a directory". A plain absolute path to the same file is already
    caught correctly by click's own check, so this needs the "~" form to
    actually exercise the gap.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notadir.txt").write_text("hello")
    options = [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--working-dir",
        "~/notadir.txt",
        "--yes",
    ]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 2
    assert "is not a directory" in result.output
    assert "does not exist" not in result.output


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("mode", ["missing", "observed", "usage", "rejected", "unprobed"])
def test_enabled_reasoning_without_evidence_warns_once_before_save(
    tmp_path, monkeypatch, style, mode
):
    if style == "anthropic" and mode == "usage":
        # LiteLLM discards this non-native usage extension; direct can retain it.
        monkeypatch.setenv("NOOA_LLM_TRANSPORT", "litellm")
    import json

    import httpx

    sent = []

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        enabled = (
            body.get("reasoning_effort") == "high"
            or body.get("reasoning", {}).get("effort") == "high"
            or body.get("thinking", {}).get("type") == "adaptive"
        )
        if enabled and mode == "rejected":
            return httpx.Response(400)
        data = response_body(style)
        if enabled and mode == "observed":
            if style == "chat":
                data["choices"][0]["message"]["reasoning_content"] = "private reasoning"
            else:
                data["output" if style == "responses" else "content"].insert(
                    0,
                    {
                        "type": "reasoning",
                        "id": "r1",
                        "summary": [{"type": "summary_text", "text": "private reasoning"}],
                    }
                    if style == "responses"
                    else {"type": "thinking", "thinking": "private reasoning", "signature": "sig"},
                )
        if enabled and mode == "usage":
            data["usage"][
                "completion_tokens_details" if style == "chat" else "output_tokens_details"
            ] = {"reasoning_tokens": 8}
        return httpx.Response(200, json=data)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "claude-sonnet-4-6" if style == "anthropic" else "gpt-5.1",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            style,
            "--api-key-env",
            "",
            "--no-catalogue",
            "--reasoning-template",
            "adaptive" if style == "anthropic" else "effort",
            "--levels",
            "high,none",
            "--max-tokens",
            "2048",
            "--output",
            str(path),
            *(["--no-probe"] if mode == "unprobed" else []),
        ],
        input="y\ny\n",
    )
    assert result.exit_code == 0, result.output
    warning = "Warning: no reasoning information was returned for: high."
    assert result.output.count(warning) == (
        1 if mode == "missing" or (mode == "usage" and style == "anthropic") else 0
    )
    assert "returned for: none" not in result.output
    assert "private reasoning" not in result.output
    assert len(sent) == (0 if mode == "unprobed" else 5 if mode == "rejected" else 7)
    if mode == "missing":
        assert result.output.index(warning) < result.output.index("Write model entry")
        assert "Try another API format or review the server's reasoning settings" in result.output
        probes = yaml.safe_load(path.read_text())["models"]["local"]["provenance"]["probes"]
        assert probes["level:high"]["outcome"] == "accepted"
        assert probes["level:high"]["reasoning_observed"] is False


def test_offline_cli_needs_no_key_and_writes_generated_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    path = tmp_path / "connected.yaml"
    result = CliRunner().invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["model_name"] == "openai/wire/model"
    assert "skipped" in result.output
    assert "not approved" in result.output


def test_recovery_edit_server_preserves_budget_and_saves_only_new_route(tmp_path, monkeypatch):
    import httpx

    sent = []

    def handle(request):
        sent.append(request)
        if request.url.host == "new.example" and request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "old-model",
            "--as",
            "local",
            "--endpoint",
            "https://old.example/v1",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--probe",
            "minimal",
            "--max-tokens",
            "200",
            "--budget-tokens",
            "6000",
            "--output",
            str(path),
        ],
        input="y\nserver\nhttps://new.example/v1\nnew-model\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert [r.url.host for r in sent] == ["old.example"] * 3 + ["new.example"] * 3
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["api_base"] == "https://new.example/v1"
    assert entry["model_name"] == "openai/new-model"
    assert entry["provenance"]["tokens_charged_to_budget"] == 6 * 712


@pytest.mark.parametrize("yes", [False, True])
def test_multiple_saved_key_variables_require_an_explicit_choice(tmp_path, monkeypatch, yes):
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    name: {
                        "model_name": "openai/model",
                        "api_base": "https://api.test/v1",
                        "api_key_env": name,
                    }
                    for name in ("KEY_ONE", "KEY_TWO")
                }
            }
        )
    )
    options = [
        "model",
        "--as",
        "new",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--no-catalogue",
        "--no-probe",
        "--output",
        str(path),
    ]
    result = CliRunner().invoke(command, options + (["--yes"] if yes else []), input="KEY_TWO\ny\n")
    assert result.exit_code == (2 if yes else 0), result.output
    if yes:
        assert "supply --api-key-env" in result.output
        assert "new" not in yaml.safe_load(path.read_text())["models"]
    else:
        assert "Saved key variable" in result.output
        assert yaml.safe_load(path.read_text())["models"]["new"]["api_key_env"] == "KEY_TWO"


def _single_saved_key_options(path):
    return [
        "model",
        "--as",
        "new",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--no-catalogue",
        "--no-probe",
        "--output",
        str(path),
    ]


def _write_single_saved_key_entry(path, name="CONNECT_SAVED_KEY"):
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "existing": {
                        "model_name": "openai/model",
                        "api_base": "https://api.test/v1",
                        "api_key_env": name,
                    }
                }
            }
        )
    )


def test_saved_key_message_says_using_only_when_a_value_is_actually_set(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    _write_single_saved_key_entry(path)
    monkeypatch.setenv("CONNECT_SAVED_KEY", "sk-already-set")
    result = CliRunner().invoke(command, _single_saved_key_options(path), input="y\n")
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Using saved key variable CONNECT_SAVED_KEY for this endpoint." in normalized_output
    assert "API key (used only for this setup)" not in normalized_output
    assert yaml.safe_load(path.read_text())["models"]["new"]["api_key_env"] == "CONNECT_SAVED_KEY"


def test_saved_key_message_is_honest_and_still_prompts_when_no_value_is_set(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    _write_single_saved_key_entry(path)
    monkeypatch.delenv("CONNECT_SAVED_KEY", raising=False)
    result = CliRunner().invoke(
        command, _single_saved_key_options(path), input="sk-freshly-entered\ny\n"
    )
    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert (
        "This endpoint previously used key variable CONNECT_SAVED_KEY, "
        "but it has no value set." in normalized_output
    )
    assert "Using saved key variable CONNECT_SAVED_KEY for this endpoint." not in normalized_output
    assert yaml.safe_load(path.read_text())["models"]["new"]["api_key_env"] == "CONNECT_SAVED_KEY"


@pytest.mark.parametrize("save_key", [False, True])
def test_new_key_path_persists_only_after_separate_confirmation(tmp_path, monkeypatch, save_key):
    from nooa import paths

    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    monkeypatch.delenv("NOOA_MODEL_API_KEY", raising=False)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [*args(path), "--api-key-env", "new"],
        input="\nnew-private-test-key\ny\n" + ("y\n" if save_key else "n\n"),
    )
    assert result.exit_code == 0, result.output
    assert (
        yaml.safe_load(path.read_text())["models"]["local"]["api_key_env"] == "NOOA_MODEL_API_KEY"
    )
    assert "new-private-test-key" not in result.output + path.read_text()
    secrets = tmp_path / "secrets.yaml"
    assert secrets.exists() is save_key
    if save_key:
        assert (
            yaml.safe_load(secrets.read_text())["env"]["NOOA_MODEL_API_KEY"]
            == "new-private-test-key"
        )


def test_wizard_discovery_auth_failure_stops_before_generation_or_save(tmp_path, monkeypatch):
    import httpx

    sent = []

    def handle(request):
        sent.append(request.method)
        return httpx.Response(401)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://api.test/v1",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--output",
            str(path),
        ],
        input="y\n",
    )
    assert result.exit_code == 1
    assert "Authentication failed" in result.output
    assert sent == ["GET"]
    assert not path.exists()


@pytest.mark.parametrize("show_config", [False, True])
def test_full_yaml_preview_is_explicit_and_does_not_change_saved_entry(
    tmp_path, monkeypatch, show_config
):
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, [*args(path), *(["--show-config"] if show_config else [])], input="y\n"
    )
    assert result.exit_code == 0, result.output
    assert ("provenance:" in result.output) is show_config
    assert "Results ·" in result.output
    if show_config:
        assert result.output.index("Save model") < result.output.index("provenance:")
        assert result.output.index("provenance:") < result.output.index("Write model entry")
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["model_name"] == "openai/wire/model"
    assert all(p["outcome"] == "not_probed" for p in entry["provenance"]["probes"].values())


def test_declining_final_write_leaves_no_file(tmp_path):
    path = tmp_path / "connected.yaml"
    result = CliRunner().invoke(command, args(path), input="n\n")
    assert result.exit_code == 0, result.output
    assert not path.exists()


def test_yaml_failure_names_file_and_line_without_echoing_contents(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("models: [\n  secret-value: }\n")
    result = CliRunner().invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 1
    assert str(path) in result.output
    assert "line 2" in result.output
    assert "secret-value" not in result.output
    assert "Agent diagnostic prompt" in result.output


def test_write_failure_keeps_path_and_actionable_reason(tmp_path, monkeypatch):
    from nooa.unifiedllm import connect

    path = tmp_path / "models.yaml"

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(connect, "write", denied)
    result = CliRunner().invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 1
    assert str(path) in result.output
    assert "Permission denied" in result.output
    assert "Agent diagnostic prompt" in result.output
    assert not path.exists()


def test_explicit_levels_are_written_as_request_blocks(tmp_path):
    path = tmp_path / "connected.yaml"
    result = CliRunner().invoke(
        command, [*args(path), "--yes", "--reasoning-template", "effort", "--levels", "low,high"]
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["reasoning_levels"] == {
        "low": {"reasoning_effort": "low"},
        "high": {"reasoning_effort": "high"},
    }


def test_existing_hand_written_alias_is_replaced_with_warning(tmp_path):
    path = tmp_path / "connected.yaml"
    runner = CliRunner()
    path.write_text("# My models\nmodels:\n  local: {model_name: openai/old}\n")
    result = runner.invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Warning:" in result.output
    assert "local" in result.output
    assert str(path) in result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["model_name"] == "openai/wire/model"
    assert path.read_text().startswith("# My models\n")


def test_endpoint_first_flow_uses_shared_discovery(tmp_path, monkeypatch):
    from nooa.unifiedllm import connect

    calls = []

    async def discover(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return connect.Discovery("https://api.test/v1", ({"id": "wire/model"},))

    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.setenv("CONNECT_TEST_KEY", "discovery-key")
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path)[1:], input="wire/model\ny\n")
    assert result.exit_code == 0, result.output
    assert calls == [("https://api.test/v1", {"api_style": "chat", "api_key": "discovery-key"})]
    assert yaml.safe_load(path.read_text())["models"]["local"]["model_name"] == "openai/wire/model"


def test_masked_key_is_transient_and_cancel_does_not_write(tmp_path, monkeypatch):
    import os

    from nooa.unifiedllm import connect

    calls = []

    async def discover(endpoint, **kwargs):
        calls.append(endpoint)
        assert kwargs["api_key"] == "temporary-secret"
        return connect.Discovery(endpoint, ({"id": "wire/model"},))

    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, [*args(path)[1:], "--prompt-key"], input="temporary-secret\n"
    )
    assert result.exit_code != 0  # EOF cancels model selection.
    assert calls == ["https://api.test/v1"]
    assert "temporary-secret" not in result.output
    assert not path.exists()
    assert "CONNECT_TEST_KEY" not in os.environ


def test_declining_replace_after_checks_keeps_original_entry(tmp_path, monkeypatch):
    from nooa.unifiedllm import connect

    async def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled replacement must not call the endpoint")

    monkeypatch.setattr(connect, "catalogue", forbidden)
    path = tmp_path / "models.yaml"
    original = "models: {local: {model_name: openai/old}}\n"
    path.write_text(original)
    result = CliRunner().invoke(command, args(path), input="n\n")
    assert result.exit_code == 0, result.output
    assert result.output.index("Results ·") < result.output.index("Replace this model?")
    assert path.read_text() == original


def test_prompted_key_is_passed_to_probe_but_never_saved(tmp_path, monkeypatch):
    from nooa.unifiedllm import connect

    calls = []

    async def run(proposal, *, approved, api_key=None):
        calls.append(approved)
        assert api_key == "temporary-secret"
        yield connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "run_steps", run)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, [*args(path), "--prompt-key", "--yes"], input="temporary-secret\n"
    )
    assert result.exit_code == 0, result.output
    assert calls == ["none"]
    assert "temporary-secret" not in result.output + path.read_text()


def test_bare_command_walks_through_setup_and_checks_inline(tmp_path, monkeypatch):
    import click
    import httpx

    from nooa import paths
    from nooa.unifiedllm import connect

    output, requests = [], []
    real_echo = click.echo
    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    monkeypatch.delenv("CONNECT_WIZARD_KEY", raising=False)

    def echo(message=None, **kwargs):
        output.append(str(message))
        return real_echo(message, **kwargs)

    def handle(request):
        requests.append(request)
        assert request.headers.get("authorization", request.headers.get("x-api-key")) in {
            "Bearer temporary-secret",
            "temporary-secret",
        }
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "example-model"}]})
        # Feedback appears before the HTTP operation, not only at the end.
        assert any("— checking" in line for line in output)
        assert any("may incur charges" in line for line in output)
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "323", "reasoning_content": "private test reasoning"}}
                ]
            },
        )

    async def catalogue():
        return []

    monkeypatch.setattr(click, "echo", echo)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    mock_http(monkeypatch, handle)
    result = CliRunner().invoke(
        command,
        [],
        input=(
            "y\ncustom\nhttps://api.test/v1\nCONNECT_WIZARD_KEY\ntemporary-secret\n"
            "example-model\nmy-model\ny\nn\n"
        ),
    )
    assert result.exit_code == 0, result.output
    # An unset --budget-tokens is now unlimited, so every check that would
    # previously have been skipped by the old 131072-token default budget
    # now actually runs (3 more than before).
    assert [r.method for r in requests] == ["GET"] + ["POST"] * 8
    entry = yaml.safe_load((tmp_path / "llm_config.yaml").read_text())["models"]["my-model"]
    assert entry["model_name"] == "openai/example-model"
    assert "temporary-secret" not in result.output + yaml.safe_dump(entry)
    assert "reasoning returned" in result.output
    assert "acceptance alone" not in result.output
    assert "private test reasoning" not in result.output
    assert result.output.index("Chat interface — checking") < result.output.index(
        "Chat interface: Connected"
    )
    assert result.output.index("Chat interface: Connected") < result.output.index(
        "Tool use — checking"
    )
    assert result.output.index("Tool use — checking") < result.output.index("Save this model as")
    assert result.output.index("Save model") < result.output.index("Save this model as")
    assert "API format [" not in result.output  # One success is selected automatically.
    assert "Run these paid probes?" not in result.output


@pytest.mark.parametrize("responses_ok", [True, False])
def test_interface_menu_only_offers_successes_or_explicit_manual_escape(
    tmp_path, monkeypatch, responses_ok
):
    import httpx
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    choices = []
    real_prompt = _connect_prompts.prompt

    def prompt(text, **kwargs):
        if text == "API format":
            choices.append(tuple(kwargs["choices"]))
        return real_prompt(text, **kwargs)

    def handle(request):
        if responses_ok and request.url.path.endswith("responses"):
            return httpx.Response(200, json=response_body("responses"))
        if responses_ok and request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})
        return httpx.Response(401)

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
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
            "--api-key-env",
            "",
            "--no-catalogue",
            "--output",
            str(path),
        ],
        input="y\nresponses\ny\n" if responses_ok else "y\ncancel\n",
    )
    if responses_ok:
        assert result.exit_code == 0, result.output
        assert choices == [("chat", "responses")]
        assert yaml.safe_load(path.read_text())["models"]["local"]["api_style"] == "responses"
    else:
        assert result.exit_code == 0
        assert not choices
        assert not path.exists()
        assert "Nothing was saved" in result.output
        assert "could not confirm" in result.output.lower()


@pytest.mark.parametrize("recover", [True, False])
def test_authentication_recovery_keeps_budget_and_secrets(tmp_path, monkeypatch, recover):
    import httpx
    import litellm

    from nooa.unifiedllm import connect

    monkeypatch.setenv("CONNECT_BAD", "wrong-test-secret")
    monkeypatch.setenv("CONNECT_GOOD", "right-test-secret")
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    sent = []

    def handle(request):
        sent.append(request)
        if request.headers.get("authorization") != "Bearer right-test-secret":
            return httpx.Response(401, json={"error": {"message": "wrong-test-secret rejected"}})
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(404)

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
            "--api-key-env",
            "CONNECT_BAD",
            "--no-catalogue",
            "--probe",
            "minimal",
            "--output",
            str(path),
        ],
        input="y\nkey\nCONNECT_GOOD\ny\n" if recover else "y\ncancel\n",
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("Approve API checks") == 1
    assert "Key rejected by this server" in result.output
    assert "Give Feedback" not in result.output
    assert "Provider List" not in result.output
    assert "wrong-test-secret" not in result.output
    assert "right-test-secret" not in result.output
    handoff = result.output.split("Agent diagnostic prompt:\n", 1)[1]
    details, _ = json.JSONDecoder().raw_decode(handoff[handoff.index("{") :])
    context = details["run_context"]
    assert context["target_file"] == str(path.resolve())
    assert context["alias"] == "local"
    assert context["remaining_budget_tokens"] == connect.DEFAULT_CHECK_BUDGET - 3 * 712
    assert context["interface_timeout_seconds"] == 30
    assert "--stage interfaces" in context["rerun_command"]
    assert "skills/nooa-model-configuration/SKILL.md" in handoff
    assert "git clone" not in handoff
    assert litellm.suppress_debug_info is False
    assert len(sent) == (7 if recover else 3)
    if recover:
        entry = yaml.safe_load(path.read_text())["models"]["local"]
        assert entry["api_key_env"] == "CONNECT_GOOD"
        assert entry["provenance"]["tokens_charged_to_budget"] == 6 * 712 + 32768 + 512
        assert "test-secret" not in path.read_text()
    else:
        assert not path.exists()


def test_retry_stops_when_original_budget_is_exhausted(tmp_path, monkeypatch):
    import httpx

    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(401)

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
            "--api-key-env",
            "",
            "--no-catalogue",
            "--budget-tokens",
            "2848",
            "--output",
            str(path),
        ],
        input="y\nretry\n",
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 4
    assert "budget is exhausted" in result.output
    assert result.output.count("Approve API checks") == 1
    assert not path.exists()


def test_no_probe_points_to_manual_skill_and_does_no_http(tmp_path, monkeypatch):
    import httpx

    def forbidden(*args, **kwargs):
        raise AssertionError("Manual setup must not use HTTP")

    monkeypatch.setattr(httpx.AsyncClient, "post", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "model-configuration.md" in result.output
    assert "nooa-model-configuration" in result.output


def test_interface_and_later_checks_share_the_cli_budget(tmp_path, monkeypatch):
    import httpx

    sent = []

    def handle(request):
        sent.append(request)
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})
        return httpx.Response(404)

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
            "--api-key-env",
            "",
            "--no-catalogue",
            "--budget-tokens",
            "2136",
            "--output",
            str(path),
        ],
        input="y\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 3  # All of the budget was spent testing interfaces.
    provenance = yaml.safe_load(path.read_text())["models"]["local"]["provenance"]
    assert provenance["tokens_charged_to_budget"] == 2136
    assert provenance["probes"]["routing"]["outcome"] == "not_probed"
    assert provenance["probes"]["tools"]["outcome"] == "not_probed"
    assert "approved budget is too small" in result.output
    assert "setup is incomplete; budget exhausted before routing" in result.output
    assert result.output.index("approved budget is too small") < result.output.index(
        "Connection: budget exhausted"
    )


def test_script_mode_requires_missing_options_without_prompting():
    result = CliRunner().invoke(command, ["--yes"])
    assert result.exit_code == 2
    assert "--endpoint" in result.output


@pytest.mark.parametrize("ambiguous", [False, True])
def test_model_details_appear_before_accepting_published_settings(tmp_path, monkeypatch, ambiguous):
    from nooa.unifiedllm import connect

    model_info = {
        "id": "wire/model",
        "context_length": 128000,
        "top_provider": {"max_completion_tokens": 8192},
        "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "low"},
    }

    async def catalogue():
        return [model_info, {"id": "other/model"}] if ambiguous else [model_info]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    path = tmp_path / "models.yaml"
    options = [arg for arg in args(path) if arg != "--no-catalogue"]
    result = CliRunner().invoke(
        command, options, input=("wire/model\n" if ambiguous else "") + "use\ny\n"
    )
    assert result.exit_code == 0, result.output
    for text in ("128,000", "8,192", "low, high", "Source: OpenRouter"):
        assert result.output.index(text) < result.output.index("Model settings")
    assert "not proof" not in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["context_window"] == 128000
    assert entry["reasoning_default"] == "low"


@pytest.mark.parametrize("select", [True, False])
def test_no_exact_catalogue_match_offers_a_fuzzy_suggestion(tmp_path, monkeypatch, select):
    from nooa.unifiedllm import connect

    # "gateway/wired-model" is close to "wire/model" but not an exact or
    # suffix match, the way a gateway-routed model ID (aws/..., bedrock-...)
    # commonly isn't an exact match for the catalogue's own model ID.
    suggestion = {
        "id": "gateway/wired-model",
        "context_length": 50000,
        "top_provider": {"max_completion_tokens": 4096},
    }

    async def catalogue():
        return [suggestion]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    path = tmp_path / "models.yaml"
    options = [arg for arg in args(path) if arg != "--no-catalogue"]
    result = CliRunner().invoke(
        command,
        options,
        input=("gateway/wired-model\n" if select else "\n") + "use\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "Model not found; did you mean one of these?" in result.output
    assert "gateway/wired-model" in result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    if select:
        assert entry["context_window"] == 50000
        assert "No catalogue match" not in result.output
    else:
        assert "context_window" not in entry
        assert "No catalogue match; model limits and reasoning levels remain unknown." in (
            result.output
        )


def test_no_exact_catalogue_match_is_not_guessed_under_yes(tmp_path, monkeypatch):
    from nooa.unifiedllm import connect

    async def catalogue():
        return [{"id": "gateway/wired-model"}]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    path = tmp_path / "models.yaml"
    options = [arg for arg in args(path) if arg != "--no-catalogue"] + ["--yes"]
    result = CliRunner().invoke(command, options)
    assert result.exit_code == 0, result.output
    assert "Model not found; did you mean" not in result.output
    assert "No catalogue match; model limits and reasoning levels remain unknown." in result.output


@pytest.mark.parametrize("action", ["edit", "keep_context", "skip", "cancel"])
def test_model_settings_can_be_edited_skipped_or_cancelled(tmp_path, monkeypatch, action):
    from nooa.unifiedllm import connect

    published = {
        "id": "wire/model",
        "context_length": 128000,
        "top_provider": {"max_completion_tokens": 8192},
        "reasoning": {"supported_efforts": ["low", "high"], "default_effort": "low"},
    }

    async def catalogue():
        return [published]

    monkeypatch.setattr(connect, "catalogue", catalogue)
    path = tmp_path / "models.yaml"
    options = [arg for arg in args(path) if arg != "--no-catalogue"]
    answers = {
        "edit": "edit\n0\n64000\n2048\nlow,medium\nmedium\nuse\ny\n",
        "keep_context": "edit\n\n2048\nlow,medium\nmedium\nuse\ny\n",
        "skip": "skip\ny\n",
        "cancel": "cancel\n",
    }
    result = CliRunner().invoke(command, options, input=answers[action])
    assert result.exit_code == 0, result.output
    assert published["context_length"] == 128000
    if action == "cancel":
        assert not path.exists()
        assert "Plan:" not in result.output
        return
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    if action == "skip":
        assert "context_window" not in entry
        assert "reasoning_levels" not in entry
        assert "Continuing without the published model settings" in result.output
    else:
        if action == "edit":
            assert "Enter a positive whole number" in result.output
        assert entry["context_window"] == (128000 if action == "keep_context" else 64000)
        assert entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == 2048
        assert entry["reasoning_levels"] == {
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
        }
        assert entry["reasoning_default"] == "medium"
        for field in (
            "context_window",
            "max_output_tokens",
            "reasoning_levels",
            "reasoning_default",
        ):
            assert entry["provenance"][field]["source"] == "user"
        probes = entry["provenance"]["probes"]
        assert set(probes) == {"routing", "tools", "level:low", "level:medium"}


def test_default_budget_covers_explicit_small_cap_and_every_level(tmp_path, monkeypatch):
    # The exact count below includes LiteLLM's pre-HTTP interface rejection.
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "litellm")
    import json

    import httpx

    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})
        return httpx.Response(404)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    levels = "max,xhigh,high,medium,low,none"
    result = CliRunner().invoke(
        command,
        [
            "gpt-5.6-sol",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--reasoning-template",
            "effort",
            "--levels",
            levels,
            "--max-tokens",
            "2048",
            "--output",
            str(path),
        ],
        input="y\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert [
        body["reasoning_effort"] for body in bodies if "reasoning_effort" in body
    ] == levels.split(",")
    # Two interfaces reach HTTP; the runtime rejects the third before sending.
    # Routing is rechecked at the configured cap; tools and all six levels run.
    # The session seed is also attempted; this minimal mock lacks a finish reason.
    assert len(bodies) == 11
    probes = yaml.safe_load(path.read_text())["models"]["local"]["provenance"]["probes"]
    assert all(record["outcome"] == "accepted" for record in probes.values())


def test_unset_budget_tokens_is_unlimited_and_never_warns(tmp_path, monkeypatch):
    import httpx

    def handle(request):
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})
        return httpx.Response(404)

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
            "--api-key-env",
            "",
            "--no-catalogue",
            "--output",
            str(path),
        ],
        input="y\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "budget remaining: unlimited" in result.output
    assert "too small for all checks" not in result.output
    assert "The approved check budget is exhausted" not in result.output


def test_bare_command_cancel_before_endpoint_does_nothing(tmp_path, monkeypatch):
    from nooa import paths

    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    result = CliRunner().invoke(command, [], input="")
    assert result.exit_code == 1
    assert not list(tmp_path.iterdir())


def test_declining_initial_approval_makes_no_calls_or_writes(tmp_path, monkeypatch):
    def forbidden(request):
        raise AssertionError("Approval was declined")

    mock_http(monkeypatch, forbidden)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, ["--output", str(path)], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Approve API checks" in result.output
    assert not path.exists()


def test_initial_approval_defaults_to_yes(tmp_path, monkeypatch):
    import httpx

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response_body("chat"))

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
            "--no-catalogue",
            "--probe",
            "minimal",
            "--output",
            str(path),
        ],
        input="\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "Approve API checks within this budget? [Y/n]" in result.output
    assert len(requests) == 1
    assert not path.exists()  # Saving remains a separate choice.


@pytest.mark.parametrize(
    "provider,base,style,key_env",
    [
        ("nvidia", "https://integrate.api.nvidia.com/v1", "chat", "NVIDIA_API_KEY"),
        ("openai", "https://api.openai.com/v1", "responses", "OPENAI_API_KEY"),
        ("anthropic", "https://api.anthropic.com/v1", "anthropic", "ANTHROPIC_API_KEY"),
        (
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "chat",
            "GEMINI_API_KEY",
        ),
        ("openrouter", "https://openrouter.ai/api/v1", "chat", "OPENROUTER_API_KEY"),
    ],
)
def test_provider_menu_fills_connection_defaults(
    tmp_path, monkeypatch, provider, base, style, key_env
):
    from nooa import paths
    from nooa.unifiedllm import connect

    seen = []

    async def discover(endpoint, **kwargs):
        seen.append((endpoint, kwargs))
        return connect.Discovery(endpoint, ({"id": "test-model"},))

    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.setenv(key_env, "preset-test-key")
    result = CliRunner().invoke(
        command, ["--no-catalogue", "--no-probe"], input=f"{provider}\ntest-model\n\nmy-model\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert seen == [(base, {"api_style": style, "api_key": "preset-test-key"})]
    entry = yaml.safe_load((tmp_path / "llm_config.yaml").read_text())["models"]["my-model"]
    saved_base = base.removesuffix("/v1") if style == "anthropic" else base
    assert (entry["api_base"], entry["api_style"], entry["api_key_env"]) == (
        saved_base,
        style,
        key_env,
    )
    assert "preset-test-key" not in result.output
    assert "Model server URL:" not in result.output
    assert "Key environment variable" not in result.output
    assert result.output.index("Model:") < result.output.index("API format [")


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
def test_mixed_endpoint_selects_model_before_request_interface(tmp_path, monkeypatch, style):
    import httpx

    seen = []

    def handle(request):
        seen.append(request)
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200, json={"data": [{"id": "vendor-a/model"}, {"id": "vendor-b/model"}]}
        )

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://api.test/v1",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--no-probe",
            "--output",
            str(path),
        ],
        input=f"vendor-b/model\n{style}\nselected\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert result.output.index("Model:") < result.output.index("API format [")
    entry = yaml.safe_load(path.read_text())["models"]["selected"]
    assert entry["api_style"] == style
    assert entry["model_name"].endswith("/vendor-b/model")


def test_provider_flag_supports_scripted_setup(tmp_path):
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "test-model",
            "--provider",
            "nvidia",
            "--as",
            "local",
            "--no-catalogue",
            "--no-probe",
            "--yes",
            "--output",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["api_key_env"] == "NVIDIA_API_KEY"


def test_large_model_list_and_invalid_choice_do_not_flood_terminal(tmp_path, monkeypatch):
    from nooa.unifiedllm import connect

    monkeypatch.setenv("CONNECT_TEST_KEY", "discovery-key")

    async def discover(endpoint, **kwargs):
        return connect.Discovery(endpoint, tuple({"id": f"vendor/model-{i}"} for i in range(1000)))

    monkeypatch.setattr(connect, "discover", discover)
    result = CliRunner().invoke(
        command, args(tmp_path / "models.yaml")[1:], input="wrong\nvendor/model-42\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert "1000 model(s)" in result.output
    assert "vendor/model-999" not in result.output
    assert "vendor/model-998" not in result.output
    assert len(result.output) < 6000


@pytest.mark.parametrize("custom_path", [False, True])
def test_server_url_suggestions_include_existing_file_without_credentials(
    tmp_path, monkeypatch, custom_path
):
    import click
    from nooa_cli.commands import _connect_prompts as _connect_prompts

    from nooa import paths

    path = tmp_path / "models.yaml"
    original = yaml.safe_dump(
        {
            "models": {
                "one": {"api_base": "https://first.example/v1", "api_key": "do-not-complete"},
                "two": {"api_base": "https://second.example/v1"},
                "duplicate": {"api_base": "https://first.example/v1/"},
                "preset": {"api_base": "https://api.openai.com/v1"},
                "no-url": {"model_name": "some-model"},
                "bad": {"api_base": None},
                "secret-url": {"api_base": "https://user:secret@private.example/v1"},
            }
        }
    )
    path.write_text(original)
    monkeypatch.setattr(paths, "get_user_dir", lambda name: path)
    seen = []

    def prompt(text, **kwargs):
        assert text == "Model server URL"
        assert kwargs.get("open_menu") is True
        seen.extend(kwargs["suggestions"])
        raise click.Abort()

    monkeypatch.setattr(_connect_prompts, "prompt", prompt)
    result = CliRunner().invoke(
        command,
        [
            "--provider",
            "custom",
            "--no-probe",
            "--no-catalogue",
            *(["--output", str(path)] if custom_path else []),
        ],
    )
    assert result.exit_code == 1
    assert "https://first.example/v1" in seen
    assert "https://second.example/v1" in seen
    assert "https://api.openai.com/v1" in seen
    assert len(seen) == len(set(seen))
    assert "https://first.example/v1/" not in seen
    assert "secret" not in str(seen) and "do-not-complete" not in str(seen)
    assert path.read_text() == original
