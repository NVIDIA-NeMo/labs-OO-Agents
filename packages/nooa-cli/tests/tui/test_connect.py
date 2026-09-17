# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Connect follows the console wizard, including its prompt completions."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from nooa_cli.tui.commands import ConnectCommand
from nooa_cli.tui.config import TUIConfig

from nooa.unifiedllm import connect


@pytest.fixture
def wizard(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / ".nooa"))
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COMPLETION_TEST_NAME", "not-a-completion")
    timeline = []
    prompts = []
    answers = {
        "Approve API checks within this budget?": "yes",
        "Choose a provider": "custom",
        "Model server URL": "https://gateway.example/v1",
        "Key environment variable (new to enter a key; - for no authentication)": "-",
        "Model": "vendor/model",
        "API format": "responses",
        "Model settings": "use",
        "Reply budget": "recommended",
        "Save this model as": "work",
    }

    async def prompt(text, **options):
        timeline.append(text)
        prompts.append((text, options))
        if text in answers:
            return answers[text]
        if text.startswith("Write model entry"):
            return "yes"
        raise AssertionError(f"Unexpected prompt: {text}")

    frontend = SimpleNamespace(prompt_connect=AsyncMock(side_effect=prompt), render=AsyncMock())
    command = ConnectCommand(frontend, TUIConfig(), SimpleNamespace(cwd=tmp_path))
    command._reload_model_registry = Mock()

    async def discover(endpoint, **kwargs):
        timeline.append("discover")
        assert kwargs["api_style"] == "chat"  # Listing convention does not choose generation API.
        return connect.Discovery(endpoint, ({"id": "vendor/model"},))

    async def interfaces(alias, model, endpoint, key_env, **kwargs):
        timeline.append("interface checks")
        assert model == "vendor/model"
        results = {}
        for style in ("chat", "responses"):
            plan = connect.plan(alias, model, style, endpoint, key_env)
            plan.entry["provenance"]["probes"]["routing"] = {"outcome": "accepted"}
            results[style] = connect.ConnectResult(alias, plan.entry)
        yield connect.InterfaceResult(results, 1200)

    async def catalogue():
        timeline.append("catalogue")
        return [
            {
                "id": "vendor/model",
                "context_length": 32000,
                "top_provider": {"max_completion_tokens": 4096},
            }
        ]

    async def checks(proposal, **kwargs):
        timeline.append("configured checks")
        assert proposal.entry["api_style"] == "responses"
        assert proposal.entry["max_tokens"] <= 4096
        assert proposal.budget_tokens == connect.DEFAULT_CHECK_BUDGET - 1200
        assert kwargs["approved"] == "all"
        yield connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.setattr(connect, "check_interfaces", interfaces)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    monkeypatch.setattr(connect, "run_steps", checks)
    return SimpleNamespace(
        command=command, timeline=timeline, prompts=prompts, answers=answers, root=tmp_path
    )


async def test_order_and_completions_match_console_wizard(wizard):
    path = wizard.root / ".nooa/llm_config.yaml"
    path.parent.mkdir()
    path.write_text(
        "models:\n  existing:\n    model_name: openai/other\n    api_base: https://saved.example/v1\n"
    )
    result = await wizard.command.execute([])
    assert result.success, result
    order = wizard.timeline
    assert order.index("Model") < order.index("interface checks") < order.index("API format")
    assert order.index("API format") < order.index("catalogue") < order.index("Reply budget")
    assert (
        order.index("Reply budget")
        < order.index("configured checks")
        < order.index("Save this model as")
    )
    options = dict(wizard.prompts)
    assert options["API format"]["choices"] == ("chat", "responses")  # Only successful interfaces.
    assert "https://saved.example/v1" in options["Model server URL"]["suggestions"]
    key_prompt = options["Key environment variable (new to enter a key; - for no authentication)"]
    assert "COMPLETION_TEST_NAME" in key_prompt["suggestions"]
    assert "not-a-completion" not in str(options)
    assert options["Model"]["choices"] == ("vendor/model",)
    assert "existing" in options["Save this model as"]["suggestions"]
    assert options["Save this model as"]["existing"] == ("existing",)
    assert options["Model settings"]["choices"] == ("use", "edit", "skip", "cancel")
    assert "recommended" in options["Reply budget"]["choices"]
    entry = yaml.safe_load(path.read_text())["models"]["work"]
    assert entry["api_style"] == "responses" and entry["transport"] == "direct"
    assert entry["context_window"] == 32000
    wizard.command._reload_model_registry.assert_called_once()


async def test_disallowed_checks_stop_before_discovery(wizard):
    wizard.answers["Approve API checks within this budget?"] = "no"
    assert (await wizard.command.execute([])).success
    assert "discover" not in wizard.timeline
    assert not (wizard.root / ".nooa/llm_config.yaml").exists()


async def test_cancel_prompt_leaves_no_file(wizard):
    wizard.answers["Model"] = None
    assert (await wizard.command.execute([])).success
    assert "interface checks" not in wizard.timeline
    assert not (wizard.root / ".nooa/llm_config.yaml").exists()


async def test_cancel_during_checks_joins_worker_and_sdk_cleanup(wizard, monkeypatch):
    started, closed = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
            yield  # Make this a check iterator.
        finally:
            await asyncio.sleep(0.01)
            closed.set()

    monkeypatch.setattr(connect, "check_interfaces", blocked)
    task = asyncio.create_task(wizard.command.execute([]))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert closed.is_set()
    assert "Save this model as" not in wizard.timeline
    assert not (wizard.root / ".nooa/llm_config.yaml").exists()


async def test_masked_key_is_confirmed_at_save_and_empty_export_is_replaced(wizard, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    wizard.answers["Key environment variable (new to enter a key; - for no authentication)"] = (
        "OPENAI_API_KEY"
    )
    wizard.answers["API key (used only for this setup)"] = "private-key"
    wizard.answers["Save this key for future NOOA runs?"] = "yes"
    assert (await wizard.command.execute([])).success
    options = dict(wizard.prompts)
    assert options["API key (used only for this setup)"]["hide_input"]
    assert "private-key" not in str(options)
    assert "private-key" not in str(wizard.command.frontend.render.call_args_list)
    assert os.environ["OPENAI_API_KEY"] == "private-key"
    assert (
        yaml.safe_load((wizard.root / ".nooa/secrets.yaml").read_text())["env"]["OPENAI_API_KEY"]
        == "private-key"
    )


async def test_invalid_key_option_never_echoes_pasted_value(wizard, caplog):
    result = await wizard.command.execute(["--api-key", "PASTEDKEY"])
    assert not result.success
    assert "PASTEDKEY" not in str(result) + caplog.text
    assert not wizard.prompts


async def test_manual_mode_never_tests_interfaces(wizard, monkeypatch):
    async def unchecked(proposal, **kwargs):
        assert kwargs["approved"] == "none"
        yield connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "run_steps", unchecked)
    result = await wizard.command.execute(["--no-probe"])
    assert result.success
    assert "interface checks" not in wizard.timeline
    assert "Approve API checks within this budget?" not in wizard.timeline
    assert dict(wizard.prompts)["API format"]["choices"] == ("chat", "responses", "anthropic")


async def test_renderer_failure_stops_worker_without_hanging(wizard):
    wizard.command.frontend.render.side_effect = RuntimeError("render failed")
    result = await asyncio.wait_for(wizard.command.execute([]), 5)
    assert not result.success
    assert not (wizard.root / ".nooa/llm_config.yaml").exists()


async def test_cancel_native_prompt_joins_worker(wizard):
    started = asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    wizard.command.frontend.prompt_connect.side_effect = blocked
    task = asyncio.create_task(wizard.command.execute([]))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert not (wizard.root / ".nooa/llm_config.yaml").exists()


async def test_only_working_interface_is_selected_without_format_prompt(wizard, monkeypatch):
    async def interfaces(alias, model, endpoint, key_env, **kwargs):
        plan = connect.plan(alias, model, "responses", endpoint, key_env)
        plan.entry["provenance"]["probes"]["routing"] = {"outcome": "accepted"}
        yield connect.InterfaceResult({"responses": connect.ConnectResult(alias, plan.entry)}, 1200)

    monkeypatch.setattr(connect, "check_interfaces", interfaces)
    assert (await wizard.command.execute([])).success
    assert "API format" not in wizard.timeline
    entry = yaml.safe_load((wizard.root / ".nooa/llm_config.yaml").read_text())["models"]["work"]
    assert entry["api_style"] == "responses"


async def test_help_renders_in_host_and_stage_mode_does_not_escape_to_stdout(wizard, capsys):
    assert (await wizard.command.execute(["--help"])).success
    output = str(wizard.command.frontend.render.call_args_list)
    assert "--no-probe" in output and "--edit-model" in output
    assert not (await wizard.command.execute(["--stage", "discover"])).success
    assert not wizard.prompts
    assert capsys.readouterr().out == ""


async def test_concurrent_wizard_hosts_keep_prompts_and_async_output_separate(
    tmp_path, monkeypatch
):
    from nooa_cli.commands import _connect_io as io
    from nooa_cli.commands import _connect_prompts as prompts
    from nooa_cli.commands.connect import command
    from nooa_cli.tui.connect_wizard import run_native_wizard

    entered = 0
    ready = asyncio.Event()

    async def prompt(text, **options):
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        await ready.wait()
        return text

    def invoke(**options):
        name = options["model"]
        assert prompts.prompt(name) == name

        async def emit():
            io.echo("result:" + name)

        io.run_async(emit())
        return name

    monkeypatch.setattr(command, "callback", invoke)
    hosts = [SimpleNamespace(prompt_connect=prompt, render=AsyncMock()) for _ in range(2)]
    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                run_native_wizard(host, [name], tmp_path / name)
                for host, name in zip(hosts, ("first", "second"), strict=True)
            )
        ),
        5,
    )
    assert results == ["first", "second"]
    for host, name in zip(hosts, results, strict=True):
        assert [call.args[0].content for call in host.render.call_args_list] == ["result:" + name]
    assert io.current_host() is None
