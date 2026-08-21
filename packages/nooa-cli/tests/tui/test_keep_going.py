# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the retained opt-in keep-going mode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nooa_cli.tui.commands import KeepGoingCommand
from nooa_cli.tui.config import TUIConfig
from nooa_cli.tui.keep_going import KeepGoingDecision
from nooa_cli.tui.output import TextOutput
from nooa_cli.tui.tui_application import DispatcherExit

from nooa.runtime.channels import QueueManager

from .tui_app_harness import make_local_tui_app


class _Agent:
    def __init__(self) -> None:
        self.vars: dict[str, object] = {}
        self.queue_manager = QueueManager()
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.user_messages = self._user_messages_in.reader
        self._system_messages_in = self.queue_manager.queue("system_messages")
        self.system_messages = self._system_messages_in.reader


class _ReflectionRunner:
    def __init__(self) -> None:
        self.scheduled = 0
        self.invalidate = None

    async def interrupt(self) -> None:
        return None

    def on_response_done(self) -> None:
        self.scheduled += 1

    def indicator_frame(self) -> str:
        return ""


def _config(*, enabled: bool, model: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        tui=SimpleNamespace(
            keep_going=enabled,
            keep_going_model=model,
        )
    )


@pytest.mark.asyncio
async def test_keep_going_command_configures_and_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    config = TUIConfig()
    agent = SimpleNamespace(vars={})
    command = KeepGoingCommand(AsyncMock(), config, agent)

    result = await command.execute(["model", "audit-model"])
    assert result.success
    assert config.keep_going_model == "audit-model"
    assert agent.vars["tui_keep_going_model"] == "audit-model"

    result = await command.execute(["on"])
    assert result.success
    assert config.keep_going is True
    assert agent.vars["tui_keep_going"] is True

    import yaml

    settings = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    assert settings["tui"]["keep_going_model"] == "audit-model"
    assert settings["tui"]["keep_going"] is True


@pytest.mark.asyncio
async def test_keep_going_command_requires_a_judge_model():
    config = TUIConfig()
    command = KeepGoingCommand(AsyncMock(), config, SimpleNamespace(vars={}))

    result = await command.execute(["on"])

    assert not result.success
    assert any(
        "model is not configured" in output.content
        for output in result.outputs
        if isinstance(output, TextOutput)
    )


def test_keep_going_model_completion_uses_model_registry(monkeypatch):
    from nooa_cli.tui.completer import Completer

    import nooa.unifiedllm as unifiedllm

    monkeypatch.setattr(
        unifiedllm,
        "MODELS",
        {"audit-alpha": object(), "audit-beta": object(), "other": object()},
    )
    registry = SimpleNamespace(_user_skills={}, get_active_help=lambda: {})

    items = Completer(registry).complete("/keep-going model audit-")

    assert [item.text for item in items] == [
        "/keep-going model audit-alpha",
        "/keep-going model audit-beta",
    ]

    assert [item.text for item in Completer(registry).complete("/keep-going ")] == [
        "/keep-going on",
        "/keep-going off",
        "/keep-going model",
    ]


def test_keep_going_judge_output_type_is_minimal():
    assert set(KeepGoingDecision.model_fields) == {"should_reprompt", "reason", "next_action"}


@pytest.mark.asyncio
async def test_build_keep_going_prompt_shapes_internal_continuation(monkeypatch):
    from nooa_cli.tui import keep_going

    async def fake_judge(**kwargs):
        assert kwargs["model"] == "audit-model"
        return KeepGoingDecision(
            should_reprompt=True,
            reason="open todos remain",
            next_action="Finish the open todos.",
        )

    monkeypatch.setattr(keep_going, "judge_keep_going", fake_judge)

    prompt = await keep_going.build_keep_going_prompt(
        _Agent(),
        SimpleNamespace(kind="DONE"),
        model="audit-model",
    )

    assert prompt is not None
    assert prompt.display_reason == "open todos remain"
    assert "Reason: open todos remain" in prompt.prompt
    assert "Next action: Finish the open todos." in prompt.prompt


@pytest.mark.asyncio
async def test_dispatcher_queues_keep_going_as_system_message(monkeypatch):
    from nooa_cli.tui import keep_going

    agent = _Agent()
    calls: list[dict] = []

    async def fake_build(agent_arg, result, *, model):
        assert agent_arg is agent
        assert result.kind == "DONE"
        assert model == "audit-model"
        return SimpleNamespace(
            prompt="[keep-going] continue",
            display_reason="open todo remains",
        )

    async def handle(notification):
        calls.append(notification)
        if len(calls) == 1:
            return SimpleNamespace(kind="DONE", explanation="")
        raise DispatcherExit()

    monkeypatch.setattr(keep_going, "build_keep_going_prompt", fake_build)
    agent.handle = handle
    app = make_local_tui_app(agent, config=_config(enabled=True, model="audit-model"))

    app.submit_message("first")
    await asyncio.wait_for(app._test_agent_runner.task, timeout=1)

    assert calls == [
        {"user_messages": ["first"]},
        {"system_messages": ["[keep-going] continue"]},
    ]


@pytest.mark.asyncio
async def test_new_user_input_cancels_stale_keep_going_audit(monkeypatch):
    from nooa_cli.tui import keep_going

    agent = _Agent()
    calls: list[dict] = []
    audit_started = asyncio.Event()
    release_audit = asyncio.Event()

    async def fake_build(agent_arg, result, *, model):
        audit_started.set()
        await release_audit.wait()
        return SimpleNamespace(prompt="[keep-going] stale", display_reason="stale audit")

    async def handle(notification):
        calls.append(notification)
        if len(calls) == 1:
            return SimpleNamespace(kind="DONE", explanation="")
        if len(calls) == 2:
            return SimpleNamespace(kind="WAIT", explanation="")
        raise DispatcherExit()

    monkeypatch.setattr(keep_going, "build_keep_going_prompt", fake_build)
    agent.handle = handle
    app = make_local_tui_app(agent, config=_config(enabled=True, model="audit-model"))

    app.submit_message("first")
    await asyncio.wait_for(audit_started.wait(), timeout=1)
    app.submit_message("second")
    for _ in range(100):
        if len(calls) == 2:
            break
        await asyncio.sleep(0)
    release_audit.set()
    await asyncio.sleep(0.01)

    assert calls == [
        {"user_messages": ["first"]},
        {"user_messages": ["second"]},
    ]
    assert await app._test_agent_runner.cancel_for_transition()


@pytest.mark.asyncio
async def test_reflection_waits_for_keep_going_audit_then_runs_when_no_continuation(monkeypatch):
    from nooa_cli.tui import keep_going

    agent = _Agent()
    runner = _ReflectionRunner()
    agent._tui_reflection_runner = runner
    audit_finished = asyncio.Event()

    async def fake_build(agent_arg, result, *, model):
        audit_finished.set()
        return None

    async def handle(notification):
        return SimpleNamespace(kind="DONE", explanation="")

    monkeypatch.setattr(keep_going, "build_keep_going_prompt", fake_build)
    agent.handle = handle
    app = make_local_tui_app(agent, config=_config(enabled=True, model="audit-model"))

    app.submit_message("first")
    await asyncio.wait_for(audit_finished.wait(), timeout=1)
    for _ in range(100):
        if runner.scheduled:
            break
        await asyncio.sleep(0)

    assert runner.scheduled == 1
    assert await app._test_agent_runner.cancel_for_transition()


@pytest.mark.asyncio
async def test_after_handle_racing_shutdown_cannot_start_another_audit(monkeypatch):
    """Shutdown closes and snapshots producers before awaiting the old audit."""
    from nooa_cli.tui import keep_going
    from nooa_cli.tui.local_turn_policy import LocalTurnPolicy

    first_audit_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_cancelled_audit = asyncio.Event()
    build_calls = 0

    async def fake_build(_agent, _result, *, model):
        nonlocal build_calls
        assert model == "audit-model"
        build_calls += 1
        first_audit_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_cancelled_audit.wait()
        return None

    class Runtime:
        def cancel_tasks(self, tasks):
            for task in tasks:
                task.cancel()

        async def run_async(self, fn):
            return await fn()

        def queue_continuation(self, prompt):
            raise AssertionError(f"unexpected continuation: {prompt}")

    agent = _Agent()
    agent.vars.update(tui_keep_going=True, tui_keep_going_model="audit-model")
    monkeypatch.setattr(keep_going, "build_keep_going_prompt", fake_build)
    policy = LocalTurnPolicy(
        agent,
        Runtime(),
        _config(enabled=True, model="audit-model"),
        emit_output=AsyncMock(),
        invalidate=lambda: None,
    )
    result = SimpleNamespace(kind="DONE", explanation="")
    await policy.after_handle(agent, result)
    await first_audit_started.wait()

    shutdown = asyncio.create_task(policy.shutdown())
    await cancellation_seen.wait()
    assert shutdown.done() is False

    # This call overlaps shutdown, after shutdown has atomically closed the policy
    # and captured the tasks it must await.
    await policy.after_handle(agent, result)
    assert build_calls == 1
    assert len(policy._tasks) == 1

    release_cancelled_audit.set()
    await shutdown
    assert policy._tasks == set()


@pytest.mark.asyncio
async def test_shutdown_blocks_cancel_suppressing_emit_from_publishing(monkeypatch):
    """A cancelled audit stuck in output cannot continue after shutdown."""
    from nooa_cli.tui import keep_going
    from nooa_cli.tui.local_turn_policy import LocalTurnPolicy

    emit_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_cancelled_emit = asyncio.Event()
    continuations = []

    async def fake_build(_agent, _result, *, model):
        assert model == "audit-model"
        return SimpleNamespace(prompt="stale continuation", display_reason="stale reason")

    async def emit(_output):
        emit_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_cancelled_emit.wait()

    class Runtime:
        def cancel_tasks(self, tasks):
            for task in tasks:
                task.cancel()

        async def run_async(self, fn):
            return await fn()

        def queue_continuation(self, prompt):
            continuations.append(prompt)
            return None

    class Reflection(_ReflectionRunner):
        def teardown(self):
            return None

    agent = _Agent()
    agent.vars.update(tui_keep_going=True, tui_keep_going_model="audit-model")
    reflection = Reflection()
    agent._tui_reflection_runner = reflection
    monkeypatch.setattr(keep_going, "build_keep_going_prompt", fake_build)
    policy = LocalTurnPolicy(
        agent,
        Runtime(),
        _config(enabled=True, model="audit-model"),
        emit_output=emit,
        invalidate=lambda: None,
    )
    await policy.after_handle(agent, SimpleNamespace(kind="DONE", explanation=""))
    await emit_started.wait()

    shutdown = asyncio.create_task(policy.shutdown())
    await cancellation_seen.wait()
    assert shutdown.done() is False
    release_cancelled_emit.set()
    await shutdown

    assert continuations == []
    assert reflection.scheduled == 0
    assert policy._tasks == set()


@pytest.mark.asyncio
async def test_after_handle_after_shutdown_has_no_output_or_reflection(monkeypatch):
    from nooa_cli.tui import keep_going
    from nooa_cli.tui.local_turn_policy import LocalTurnPolicy

    outputs = []
    audit_started = asyncio.Event()

    async def fake_build(_agent, _result, *, model):
        audit_started.set()
        return SimpleNamespace(prompt="unexpected", display_reason="unexpected")

    async def emit(output):
        outputs.append(output)

    class Runtime:
        def cancel_tasks(self, tasks):
            for task in tasks:
                task.cancel()

        async def run_async(self, fn):
            return await fn()

        def queue_continuation(self, prompt):
            raise AssertionError(f"unexpected continuation: {prompt}")

    class Reflection(_ReflectionRunner):
        def teardown(self):
            return None

    agent = _Agent()
    agent.vars.update(tui_keep_going=True, tui_keep_going_model="audit-model")
    reflection = Reflection()
    agent._tui_reflection_runner = reflection
    monkeypatch.setattr(keep_going, "build_keep_going_prompt", fake_build)
    policy = LocalTurnPolicy(
        agent,
        Runtime(),
        _config(enabled=True, model="audit-model"),
        emit_output=emit,
        invalidate=lambda: None,
    )

    await policy.shutdown()
    await policy.after_handle(
        agent, SimpleNamespace(kind="DONE", explanation="should not be emitted")
    )

    assert outputs == []
    assert reflection.scheduled == 0
    assert audit_started.is_set() is False
    assert policy._tasks == set()
