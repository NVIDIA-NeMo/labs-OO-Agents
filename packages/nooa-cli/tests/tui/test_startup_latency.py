# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup latency regressions for the native TUI."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
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


def test_interactive_state_exports_do_not_import_core_agent_stack() -> None:
    code = (
        "import sys\n"
        "from nooa_cli.interactive import AgentLifecycle, AgentState\n"
        "print(AgentLifecycle.IDLE.value)\n"
        "print(AgentState.__name__)\n"
        "print('nooa' in sys.modules)\n"
        "print(any(name.startswith('nooa.') for name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["idle", "AgentState", "False", "False"]


def test_config_load_does_not_import_core_agent_stack(tmp_path) -> None:
    code = (
        "import sys\n"
        "from nooa_cli.tui.config import Config\n"
        "Config.load(no_splash=True)\n"
        "print('nooa' in sys.modules)\n"
        "print(any(name.startswith('nooa.') for name in sys.modules))\n"
    )
    env = os.environ | {
        "NEMO_OO_PROJECT_DIR": str(tmp_path / "project" / ".nooa"),
        "NEMO_OO_USER_DIR": str(tmp_path / "user"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False"]


def test_config_load_uses_cwd_project_settings_without_core_import(tmp_path) -> None:
    project_dir = tmp_path / ".nooa"
    project_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (project_dir / "settings.yaml").write_text("tui:\n  default_model: cwd-model\n")
    code = (
        "import sys\n"
        "from nooa_cli.tui.config import Config\n"
        "cfg = Config.load(no_splash=True)\n"
        "print(cfg.tui.default_model)\n"
        "print('nooa' in sys.modules)\n"
    )
    env = os.environ.copy()
    for name in ("NEMO_OO_PROJECT_DIR", "NEMO_OO_USER_DIR", "NEMO_OO_SETTINGS"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )
    assert result.stdout.splitlines() == ["cwd-model", "False"]


def test_frontend_import_does_not_import_core_agent_stack() -> None:
    code = (
        "import sys\n"
        "import nooa_cli.tui.frontend\n"
        "print('nooa' in sys.modules)\n"
        "print(any(name.startswith('nooa.') for name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False"]


def test_tui_agent_import_does_not_import_litellm() -> None:
    code = (
        "import sys\n"
        "from nooa_cli.tui.agent import TUIAgent\n"
        "print(TUIAgent.__name__)\n"
        "print('litellm' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines()[-2:] == ["TUIAgent", "False"]


def test_bootstrap_registry_alias_defers_litellm_import(tmp_path) -> None:
    project_dir = tmp_path / ".nooa"
    project_dir.mkdir()
    (project_dir / "settings.yaml").write_text("tui:\n  default_model: local-model\n")
    (project_dir / "llm_config.yaml").write_text(
        "models:\n"
        "  local-model:\n"
        "    model_name: openai/local-model\n"
        "    api_base: http://localhost:9999/v1\n"
        "    context_window: 1000\n"
    )
    code = (
        "import asyncio, sys\n"
        "from nooa_cli.tui.config import Config\n"
        "from nooa_cli.tui.bootstrap import bootstrap\n"
        "async def main():\n"
        "    cfg = Config.load(no_splash=True, no_trace=True)\n"
        "    result = await bootstrap(cfg)\n"
        "    try:\n"
        "        print(type(result.agent.llm).__name__)\n"
        "        print(result.agent.llm.model)\n"
        "        print(result.agent.llm.context_window)\n"
        "        print('litellm' in sys.modules)\n"
        "    finally:\n"
        "        result.session_manager.close()\n"
        "asyncio.run(main())\n"
    )
    env = os.environ | {
        "NEMO_OO_PROJECT_DIR": str(project_dir),
        "NEMO_OO_USER_DIR": str(tmp_path / "user"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.stdout.splitlines()[-4:] == [
        "LazyRegistryLLMClient",
        "openai/local-model",
        "1000",
        "False",
    ]


@pytest.mark.asyncio
async def test_deferred_bootstrap_starts_app_before_agent_construction(monkeypatch):
    import nooa_cli.interactive as interactive_module
    import nooa_cli.tui.bootstrap as bootstrap_module
    import nooa_cli.tui.tui_application as app_module
    from nooa_cli.tui.config import Config, DisplayMode
    from nooa_cli.tui.session import Session

    events: list[str] = []
    app_started = asyncio.Event()
    agent_attached = asyncio.Event()
    loop_progressed_during_bootstrap = asyncio.Event()

    class FakeFrontend:
        console = None

        async def render(self, output) -> None:
            events.append(f"render:{getattr(output, 'content', type(output).__name__)}")

        def close(self) -> None:
            events.append("frontend.close")

    class FakeApp:
        display_mode = DisplayMode.FULLSCREEN

        def __init__(self, **_kwargs) -> None:
            self._running = False
            events.append("app.init")

        @property
        def is_running(self) -> bool:
            return self._running

        async def run_async(self) -> None:
            self._running = True
            events.append("app.run_async")
            app_started.set()
            await agent_attached.wait()
            self._running = False

        def set_llm_probe_status(self, text: str) -> None:
            events.append(f"status:{text}")

        def set_completer(self, _completer) -> None:
            events.append("completer.set")

        def invalidate(self) -> None:
            pass

        def close_agent_observation(self) -> None:
            events.append("app.close_agent_observation")

        def runtime_state_changed(self) -> None:
            pass

        def runtime_notification_received(self) -> None:
            pass

        def runtime_cancelled(self) -> None:
            pass

        @property
        def agent(self):
            return None

        @agent.setter
        def agent(self, _agent) -> None:
            events.append("agent.attached")
            agent_attached.set()

    class FakeRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("runner.init")

        def set_dispatch_hooks(self, **_kwargs) -> None:
            pass

        def set_user_message_accepted_callback(self, _callback) -> None:
            pass

        def bind(self) -> None:
            events.append("runner.bind")

        def activate(self, _loop) -> None:
            events.append("runner.activate")

        async def shutdown(self) -> None:
            events.append("runner.shutdown")

        def run(self, fn):
            return fn()

        async def run_async(self, fn):
            result = fn()
            if hasattr(result, "__await__"):
                return await result
            return result

    class FakePolicy:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def before_handle(self, _agent) -> None:
            pass

        async def after_handle(self, _agent, _result) -> None:
            pass

        def on_notification(self, _notification) -> None:
            pass

        def invalidate_keep_going(self) -> None:
            pass

        async def shutdown(self) -> None:
            events.append("policy.shutdown")

    class FakeRenderer:
        def __init__(self, **_kwargs) -> None:
            pass

        def attach(self) -> None:
            events.append("renderer.attach")

        def detach(self) -> None:
            events.append("renderer.detach")

    class FakeRegistry:
        blocking_llm_health = None
        startup_info = SimpleNamespace(llm_ready=True, llm_status="ready")

        def commands(self):
            return []

    cfg = Config()
    result = SimpleNamespace(
        agent=SimpleNamespace(event_manager=None),
        config=cfg,
        session_manager=None,
        resumed=False,
        restored=False,
        session_id=None,
        messages=[],
        tracing_enabled=False,
        blocking_llm_health=None,
        startup_info=None,
    )

    async def deferred_bootstrap():
        assert app_started.is_set()
        time.sleep(0.05)
        assert loop_progressed_during_bootstrap.is_set()
        events.append("bootstrap")
        return result

    async def mark_loop_progress() -> None:
        await asyncio.sleep(0.01)
        events.append("loop.tick")
        loop_progressed_during_bootstrap.set()

    monkeypatch.setattr(app_module, "TUIApplication", FakeApp)
    monkeypatch.setattr(interactive_module, "LocalAgentRunner", FakeRunner)
    monkeypatch.setattr("nooa_cli.tui.local_turn_policy.LocalTurnPolicy", FakePolicy)
    monkeypatch.setattr("nooa_cli.tui.agent_event_renderer.AgentEventRenderer", FakeRenderer)
    monkeypatch.setattr(bootstrap_module, "build_initial_outputs", lambda *_a, **_k: [])
    monkeypatch.setattr(bootstrap_module, "build_registry", lambda *_a, **_k: FakeRegistry())

    session = Session(
        frontend=FakeFrontend(),
        agent=None,
        config=cfg,
        registry=None,
        initial_outputs=[],
    )
    session._deferred_bootstrap = deferred_bootstrap
    session._dump_exit_diagnostics = lambda: None
    session._restore_terminal = lambda: None
    session._print_exit_message = lambda: None

    asyncio.create_task(mark_loop_progress())
    await session.run()

    assert events.index("app.run_async") < events.index("bootstrap")
    assert events.index("app.run_async") < events.index("status:booting the NOOA runtime...")
    assert events.index("status:booting the NOOA runtime...") < events.index("bootstrap")
    assert events.index("app.run_async") < events.index("loop.tick") < events.index("bootstrap")
    assert events.index("bootstrap") < events.index("runner.init")
    assert events.index("runner.bind") < events.index("agent.attached")


@pytest.mark.asyncio
async def test_deferred_bootstrap_exit_closes_bootstrapped_session(monkeypatch):
    import nooa_cli.tui.tui_application as app_module
    from nooa_cli.tui.config import Config, DisplayMode
    from nooa_cli.tui.session import Session

    class FakeFrontend:
        console = None

        def close(self) -> None:
            pass

        async def render(self, _output) -> None:
            pass

    class FakeApp:
        display_mode = DisplayMode.FULLSCREEN

        def __init__(self, **_kwargs) -> None:
            self._running = False

        @property
        def is_running(self) -> bool:
            return self._running

        async def run_async(self) -> None:
            self._running = True
            await asyncio.sleep(0)
            self._running = False

        def set_llm_probe_status(self, _text: str) -> None:
            pass

        def close_agent_observation(self) -> None:
            pass

        def invalidate(self) -> None:
            pass

    session_manager = Mock()
    result = SimpleNamespace(
        agent=SimpleNamespace(),
        config=Config(),
        session_manager=session_manager,
    )

    async def deferred_bootstrap():
        await asyncio.sleep(0.01)
        return result

    monkeypatch.setattr(app_module, "TUIApplication", FakeApp)

    session = Session(
        frontend=FakeFrontend(),
        agent=None,
        config=Config(),
        registry=None,
        initial_outputs=[],
    )
    session._deferred_bootstrap = deferred_bootstrap
    session._dump_exit_diagnostics = lambda: None
    session._restore_terminal = lambda: None
    session._print_exit_message = lambda: None

    await session.run()

    assert session.agent is result.agent
    session_manager.close.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_startup_status_animates_while_active():
    from .tui_app_harness import TUIHarness

    async with TUIHarness() as h:
        h.app.set_llm_probe_status("booting the NOOA runtime...")
        await h.wait_for(lambda: "booting the NOOA runtime..." in h.app.status_text())
        first = h.app.status_text()

        await h.wait_for(
            lambda: "booting the NOOA runtime..." in h.app.status_text()
            and h.app.status_text() != first
        )


def test_plain_message_before_agent_ready_stays_in_composer():
    from nooa_cli.tui.tui_application import TUIApplication

    app = TUIApplication.__new__(TUIApplication)
    app._agent_controller = SimpleNamespace(state=None)
    app._resolve_composer_submission = lambda text, *, mention_base=None: text
    app._jump_fullscreen_to_tail = Mock()
    app._history = []
    app._history_cursor = None
    app._commands_dispatched = []
    emitted: list[str] = []
    app.emit_block = emitted.append

    keep_text = TUIApplication._accept_handler(app, SimpleNamespace(text="hello"))

    assert keep_text is True
    assert app._history == []
    assert "NOOA is still booting" in emitted[0]


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
    command.frontend._app = SimpleNamespace(
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
    command.frontend._app.refresh_transcript_blocks.assert_called_once_with("startup-info")
    command.frontend._app.invalidate.assert_not_called()


async def asyncio_wait_for_background_tasks(session) -> None:
    import asyncio

    pending = list(session._background_tasks)
    if pending:
        await asyncio.gather(*pending)
