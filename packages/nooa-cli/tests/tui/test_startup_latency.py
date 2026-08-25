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


def test_interactive_import_defers_optional_datascience_stack() -> None:
    code = (
        "import sys\n"
        "import nooa.interactive\n"
        "print('numpy' in sys.modules)\n"
        "print('pandas' in sys.modules)\n"
        "print('plotly.express' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False", "False"]


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
            "Checking LLM endpoint in the background"
            not in str(getattr(output, "content", ""))
            for output in result.messages
        )
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_pending_llm_health_blocks_prompts_until_background_probe_succeeds(monkeypatch):
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
    session._app = SimpleNamespace(invalidate=Mock(), set_llm_probe_status=Mock())
    session._background_tasks = set()

    monkeypatch.setattr(
        "nooa_cli.tui.health_check.probe_llm",
        AsyncMock(return_value=HealthCheckResult(ok=True)),
    )

    assert "still being checked" in session._llm_submission_error()

    session._start_llm_health_check()
    await asyncio_wait_for_background_tasks(session)

    assert session.registry.blocking_llm_health is None
    assert session.registry.startup_info.llm_ready is True
    assert session.registry.startup_info.llm_status == "ready"
    assert session._app.set_llm_probe_status.mock_calls == [
        call("probing LLM endpoint..."),
        call(""),
    ]
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
    session._app = SimpleNamespace(invalidate=Mock(), set_llm_probe_status=Mock())
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
    session._app.invalidate.assert_called_once()
    assert "unavailable" in session._llm_submission_error()
    assert any("Authentication failed" in output.content for output in rendered)
    assert any("export API_KEY" in output.content for output in rendered)


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


async def asyncio_wait_for_background_tasks(session) -> None:
    import asyncio

    pending = list(session._background_tasks)
    if pending:
        await asyncio.gather(*pending)
