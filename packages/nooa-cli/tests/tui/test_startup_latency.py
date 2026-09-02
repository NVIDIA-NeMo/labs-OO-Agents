# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup latency regressions for the native TUI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest


def _configure_project(monkeypatch, tmp_path):
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    import nooa_cli.tui.session_manager as session_manager

    monkeypatch.setattr(session_manager, "SESSIONS_DIR", project_dir / "sessions")
    return project_dir


def test_tui_package_import_does_not_import_agent_stack() -> None:
    code = (
        "import sys\n"
        "import nooa_cli.tui\n"
        "print('nooa_cli.tui.agent' in sys.modules)\n"
        "print('nooa_cli.coding.agent' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False"]


@pytest.mark.asyncio
async def test_empty_registry_startup_prompts_connect_without_default_model_probe(
    tmp_path, monkeypatch
):
    from nooa_cli.tui.bootstrap import bootstrap, build_startup_info
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.session import Session
    from nooa_cli.tui.toolbar import ToolbarRegistry

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.agent.summarization.policy = "none"

    monkeypatch.setattr("nooa.llm_config.llm_config_chain", lambda: [])
    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda: None)
    monkeypatch.setattr(
        "nooa_cli.tui.config.get_llm",
        lambda _config: (_ for _ in ()).throw(AssertionError("get_llm should not run")),
    )

    result = await bootstrap(cfg)
    try:
        assert result.blocking_llm_health is not None
        assert result.blocking_llm_health.blocking is True
        assert result.blocking_llm_health.pending is False
        assert result.blocking_llm_health.error_message == (
            "No LLM connected. Run `/connect` to configure one."
        )
        assert result.blocking_llm_health.fix_hint is None
        contents = [getattr(output, "content", "") for output in result.messages]
        assert contents == []
        assert all("ANTHROPIC_API_KEY" not in content for content in contents)
        assert all("claude-opus" not in content for content in contents)

        startup_info = build_startup_info(result)
        assert startup_info.model == "run /connect to configure one"
        assert startup_info.short_model == "No LLM connected"
        assert startup_info.llm_ready is False
        assert startup_info.llm_status == "not_connected"

        session = Session.__new__(Session)
        session.config = cfg
        session.registry = SimpleNamespace(blocking_llm_health=result.blocking_llm_health)
        session._toolbar = ToolbarRegistry(load_plugins=False)
        session._session_manager = SimpleNamespace(session_id="12345678-abcd", name=None)
        session.agent = SimpleNamespace(shell=SimpleNamespace(cwd=str(tmp_path)))
        session._context_usage_label = lambda: ""
        assert "No LLM" in session._session_label()
        assert "claude-opus" not in session._session_label()
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_unconfigured_builtin_default_prompts_connect_without_claude_probe(
    tmp_path, monkeypatch
):
    from nooa_cli.tui.bootstrap import bootstrap, build_startup_info
    from nooa_cli.tui.config import Config

    project_dir = _configure_project(monkeypatch, tmp_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "llm_config.yaml").write_text(
        "models:\n"
        "  qwen3-1.7b:\n"
        "    model_name: ollama_chat/qwen3:1.7b\n"
        "    api_base: http://localhost:11434\n"
    )
    cfg = Config()
    cfg.agent.summarization.policy = "none"

    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    monkeypatch.setattr("nooa.secrets.load_secrets_into_env", lambda: None)
    monkeypatch.setattr(
        "nooa_cli.tui.config.get_llm",
        lambda _config: (_ for _ in ()).throw(AssertionError("get_llm should not run")),
    )

    result = await bootstrap(cfg)
    try:
        assert result.blocking_llm_health is not None
        assert result.blocking_llm_health.error_message == (
            "No LLM connected. Run `/connect` to configure one."
        )
        assert result.messages == []

        startup_info = build_startup_info(result)
        assert startup_info.short_model == "No LLM connected"
        assert startup_info.model == "run /connect to configure one"
        assert startup_info.llm_status == "not_connected"
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_pending_llm_health_does_not_emit_durable_startup_status(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap
    from nooa_cli.tui.config import Config

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "provider/model-a"
    cfg.agent.summarization.policy = "none"
    monkeypatch.setattr(
        "nooa_cli.tui.config.get_llm",
        lambda config: SimpleNamespace(model=config.tui.default_model),
    )

    result = await bootstrap(cfg)
    try:
        assert result.blocking_llm_health is not None
        assert result.blocking_llm_health.pending is True
        assert all(
            "Checking LLM endpoint in the background" not in str(getattr(output, "content", ""))
            for output in result.messages
        )
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_pending_llm_health_queues_prompts_until_background_probe_succeeds(monkeypatch):
    from nooa_cli.tui.health_check import HealthCheckResult
    from nooa_cli.tui.session import Session

    rendered = []
    session = Session.__new__(Session)
    session.agent = SimpleNamespace(llm=SimpleNamespace(model="model-a"))
    session.registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking LLM endpoint for model 'model-a'.",
            blocking=True,
            pending=True,
        ),
        startup_info=SimpleNamespace(llm_ready=False, llm_status="checking"),
    )
    session.frontend = SimpleNamespace(
        render=AsyncMock(side_effect=lambda output: rendered.append(output))
    )
    session._app = SimpleNamespace(
        invalidate=Mock(),
        set_llm_probe_status=Mock(),
        release_deferred_messages=Mock(),
        reject_deferred_messages=Mock(),
    )
    session._background_tasks = set()

    monkeypatch.setattr(
        "nooa_cli.tui.health_check.probe_llm",
        AsyncMock(return_value=HealthCheckResult(ok=True)),
    )

    assert session._llm_submission_pending() is True
    assert session._llm_submission_error() is None

    session._start_llm_health_check()
    await asyncio_wait_for_background_tasks(session)

    assert session.registry.blocking_llm_health is None
    assert session.registry.startup_info.llm_ready is True
    assert session.registry.startup_info.llm_status == "ready"
    assert session._app.set_llm_probe_status.mock_calls == [
        call("probing LLM endpoint..."),
        call(""),
    ]
    session._app.release_deferred_messages.assert_called_once_with()
    session._app.reject_deferred_messages.assert_not_called()
    session._app.invalidate.assert_called_once()
    assert rendered == []
    assert session._llm_submission_error() is None


@pytest.mark.asyncio
async def test_pending_llm_health_keeps_blocking_on_auth_failure(monkeypatch):
    from nooa_cli.tui.health_check import HealthCheckResult
    from nooa_cli.tui.session import Session

    rendered = []
    session = Session.__new__(Session)
    session.agent = SimpleNamespace(llm=SimpleNamespace(model="model-a"))
    session.registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking LLM endpoint for model 'model-a'.",
            blocking=True,
            pending=True,
        ),
        startup_info=SimpleNamespace(llm_ready=False, llm_status="checking"),
    )
    session.frontend = SimpleNamespace(
        render=AsyncMock(side_effect=lambda output: rendered.append(output))
    )
    session._app = SimpleNamespace(
        invalidate=Mock(),
        set_llm_probe_status=Mock(),
        release_deferred_messages=Mock(),
        reject_deferred_messages=Mock(),
    )
    session._background_tasks = set()

    failure = HealthCheckResult(
        ok=False,
        error_message="Authentication failed for model 'model-a'.",
        fix_hint="export API_KEY",
        blocking=True,
    )
    monkeypatch.setattr("nooa_cli.tui.health_check.probe_llm", AsyncMock(return_value=failure))

    session._start_llm_health_check()
    await asyncio_wait_for_background_tasks(session)

    assert session.registry.blocking_llm_health is failure
    assert session.registry.startup_info.llm_ready is False
    assert session.registry.startup_info.llm_status == "unavailable"
    assert session._app.set_llm_probe_status.mock_calls == [
        call("probing LLM endpoint..."),
        call(""),
    ]
    session._app.release_deferred_messages.assert_not_called()
    session._app.reject_deferred_messages.assert_called_once()
    assert "unavailable" in session._app.reject_deferred_messages.call_args.args[0]
    session._app.invalidate.assert_called_once()
    assert "unavailable" in session._llm_submission_error()
    assert any("Authentication failed" in output.content for output in rendered)
    assert any("export API_KEY" in output.content for output in rendered)


@pytest.mark.asyncio
async def test_pending_llm_health_rejects_queued_prompts_on_transient_failure(monkeypatch):
    from nooa_cli.tui.health_check import HealthCheckResult
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session.agent = SimpleNamespace(llm=SimpleNamespace(model="model-a"))
    session.registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking LLM endpoint for model 'model-a'.",
            blocking=True,
            pending=True,
        ),
        startup_info=SimpleNamespace(llm_ready=False, llm_status="checking"),
    )
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._app = SimpleNamespace(
        invalidate=Mock(),
        set_llm_probe_status=Mock(),
        release_deferred_messages=Mock(),
        reject_deferred_messages=Mock(),
    )
    session._background_tasks = set()
    failure = HealthCheckResult(
        ok=False,
        error_message="Temporary endpoint failure.",
        blocking=False,
    )
    monkeypatch.setattr("nooa_cli.tui.health_check.probe_llm", AsyncMock(return_value=failure))

    session._start_llm_health_check()
    await asyncio_wait_for_background_tasks(session)

    assert session.registry.blocking_llm_health is None
    assert session.registry.startup_info.llm_ready is True
    session._app.release_deferred_messages.assert_not_called()
    session._app.reject_deferred_messages.assert_called_once_with("Temporary endpoint failure.")
    assert session._llm_submission_pending() is False
    assert session._llm_submission_error() is None


@pytest.mark.asyncio
async def test_model_switch_clears_pending_startup_info(monkeypatch):
    from nooa_cli.tui.commands import ModelCommand
    from nooa_cli.tui.health_check import HealthCheckResult

    candidate = SimpleNamespace(model="new/model")
    agent = SimpleNamespace(
        llm=SimpleNamespace(model="old/model"),
        set_llm=lambda llm: setattr(agent, "llm", llm),
    )
    startup_info = SimpleNamespace(
        model="old/model",
        short_model="model",
        llm_ready=False,
        llm_status="checking",
    )
    registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking LLM endpoint for model 'old/model'.",
            blocking=True,
            pending=True,
        ),
        startup_info=startup_info,
    )
    config = SimpleNamespace(default_model="old/model")
    command = ModelCommand(AsyncMock(), config, agent, registry=registry)
    command.frontend._app = SimpleNamespace(
        release_deferred_messages=Mock(),
        refresh_transcript_blocks=Mock(return_value=True),
        invalidate=Mock(),
    )
    command._persist_tui_setting = lambda _key, _value: Path("settings.yaml")

    async def _agent_run_async(fn):
        return fn()

    command._agent_run_async = _agent_run_async
    monkeypatch.setattr("nooa_cli.tui.config.get_llm_for_model", lambda _model: candidate)
    monkeypatch.setattr(
        "nooa_cli.tui.health_check.probe_llm",
        AsyncMock(return_value=HealthCheckResult(ok=True)),
    )
    monkeypatch.setattr("nooa.interactive.apply_model_limits", lambda _agent: None)

    result = await command.execute(["new/model"])

    assert result.success is True
    assert registry.blocking_llm_health is None
    assert startup_info.model == "new/model"
    assert startup_info.short_model == "model"
    assert startup_info.llm_ready is True
    assert startup_info.llm_status == "ready"
    command.frontend._app.release_deferred_messages.assert_called_once_with()
    command.frontend._app.refresh_transcript_blocks.assert_called_once_with("startup-info")
    command.frontend._app.invalidate.assert_not_called()


@pytest.mark.asyncio
async def test_model_validation_exception_rejects_transferred_prompts(monkeypatch):
    from nooa_cli.tui.commands import ModelCommand
    from nooa_cli.tui.health_check import HealthCheckResult

    candidate = SimpleNamespace(model="new/model")
    agent = SimpleNamespace(llm=SimpleNamespace(model="old/model"))
    registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking old model.",
            blocking=True,
            pending=True,
        ),
        startup_info=SimpleNamespace(llm_ready=False, llm_status="checking"),
        llm_health_generation=0,
    )
    command = ModelCommand(
        AsyncMock(),
        SimpleNamespace(default_model="old/model"),
        agent,
        registry=registry,
    )
    command.frontend._app = SimpleNamespace(
        set_llm_probe_status=Mock(),
        reject_deferred_messages=Mock(),
    )
    monkeypatch.setattr("nooa_cli.tui.config.get_llm_for_model", lambda _model: candidate)
    monkeypatch.setattr(
        "nooa_cli.tui.health_check.probe_llm",
        AsyncMock(side_effect=RuntimeError("validation crashed")),
    )

    result = await command.execute(["new/model"])

    assert result.success is False
    assert registry.llm_health_generation == 1
    assert registry.blocking_llm_health.blocking is True
    command.frontend._app.reject_deferred_messages.assert_called_once_with(
        "Failed to switch model: validation crashed"
    )


@pytest.mark.asyncio
async def test_model_switch_owns_deferred_prompts_before_old_probe_finishes(monkeypatch):
    import asyncio

    from nooa_cli.tui.commands import ModelCommand
    from nooa_cli.tui.health_check import HealthCheckResult
    from nooa_cli.tui.session import Session

    old_llm = SimpleNamespace(model="old/model")
    candidate = SimpleNamespace(model="new/model")
    old_result = asyncio.Future()
    candidate_result = asyncio.Future()
    agent = SimpleNamespace(
        llm=old_llm,
        set_llm=lambda llm: setattr(agent, "llm", llm),
    )
    registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking old model.",
            blocking=True,
            pending=True,
        ),
        startup_info=SimpleNamespace(llm_ready=False, llm_status="checking"),
    )
    app = SimpleNamespace(
        invalidate=Mock(),
        set_llm_probe_status=Mock(),
        release_deferred_messages=Mock(),
        reject_deferred_messages=Mock(),
        refresh_transcript_blocks=Mock(return_value=True),
    )
    session = Session.__new__(Session)
    session.agent = agent
    session.registry = registry
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._app = app
    session._background_tasks = set()

    async def probe(llm):
        return await (old_result if llm is old_llm else candidate_result)

    monkeypatch.setattr("nooa_cli.tui.health_check.probe_llm", probe)
    monkeypatch.setattr("nooa_cli.tui.config.get_llm_for_model", lambda _model: candidate)
    monkeypatch.setattr("nooa.interactive.apply_model_limits", lambda _agent: None)

    session._start_llm_health_check()
    await asyncio.sleep(0)

    command = ModelCommand(
        AsyncMock(),
        SimpleNamespace(default_model="old/model"),
        agent,
        registry=registry,
    )
    command.frontend._app = app
    command._persist_tui_setting = lambda _key, _value: Path("settings.yaml")
    command._agent_run_async = lambda fn: asyncio.sleep(0, result=fn())
    switch_task = asyncio.create_task(command.execute(["new/model"]))
    await asyncio.sleep(0)

    old_result.set_result(
        HealthCheckResult(ok=False, error_message="Old model failed.", blocking=True)
    )
    await asyncio_wait_for_background_tasks(session)

    app.release_deferred_messages.assert_not_called()
    app.reject_deferred_messages.assert_not_called()

    candidate_result.set_result(HealthCheckResult(ok=True))
    result = await switch_task

    assert result.success is True
    assert agent.llm is candidate
    app.release_deferred_messages.assert_called_once_with()
    app.reject_deferred_messages.assert_not_called()


async def test_cancelled_model_probe_still_resolves_deferred_prompt_ownership(monkeypatch):
    """A cancelled /model probe must not strand deferred startup prompts.

    Cancellation is a BaseException in Python 3.13; the old ``except
    Exception`` let a cancelled probe skip _mark_model_check_failed after
    _begin_model_validation had taken prompt ownership, so deferred prompts
    waited forever.
    """
    import asyncio

    from nooa_cli.tui.commands import ModelCommand
    from nooa_cli.tui.health_check import HealthCheckResult
    from nooa_cli.tui.session import Session

    old_llm = SimpleNamespace(model="old/model")
    candidate = SimpleNamespace(model="new/model")
    agent = SimpleNamespace(
        llm=old_llm,
        set_llm=lambda llm: setattr(agent, "llm", llm),
    )
    registry = SimpleNamespace(
        blocking_llm_health=HealthCheckResult(
            ok=False,
            error_message="Checking old model.",
            blocking=True,
            pending=True,
        ),
        llm_health_generation=0,
        startup_info=SimpleNamespace(llm_ready=False, llm_status="checking"),
    )
    app = SimpleNamespace(
        invalidate=Mock(),
        set_llm_probe_status=Mock(),
        release_deferred_messages=Mock(),
        reject_deferred_messages=Mock(),
        refresh_transcript_blocks=Mock(return_value=True),
    )
    session = Session.__new__(Session)
    session.agent = agent
    session.registry = registry
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._app = app
    session._background_tasks = set()

    probe_gate = asyncio.Event()

    async def hanging_probe(llm):
        await probe_gate.wait()
        return HealthCheckResult(ok=True)

    monkeypatch.setattr("nooa_cli.tui.health_check.probe_llm", hanging_probe)
    monkeypatch.setattr("nooa_cli.tui.config.get_llm_for_model", lambda _model: candidate)
    monkeypatch.setattr("nooa.interactive.apply_model_limits", lambda _agent: None)

    command = ModelCommand(
        AsyncMock(),
        SimpleNamespace(default_model="old/model"),
        agent,
        registry=registry,
    )
    command.frontend._app = app
    command._persist_tui_setting = lambda _key, _value: Path("settings.yaml")
    command._agent_run_async = lambda fn: asyncio.sleep(0, result=fn())

    switch_task = asyncio.create_task(command.execute(["new/model"]))
    await asyncio.sleep(0)
    assert registry.llm_health_generation == 1  # ownership transferred

    switch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await switch_task

    # Ownership resolved: deferred prompts were rejected, never stranded.
    app.reject_deferred_messages.assert_called_once()
    app.release_deferred_messages.assert_not_called()


async def asyncio_wait_for_background_tasks(session) -> None:
    import asyncio

    pending = list(session._background_tasks)
    if pending:
        await asyncio.gather(*pending)
