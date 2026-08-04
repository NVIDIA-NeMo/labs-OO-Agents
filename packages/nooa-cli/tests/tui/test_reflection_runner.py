# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle-reflection runner + /reflection command
(design: docs/design/memory-system/design-idle-reflection.md §3/§9 "Runner")."""

import asyncio
import time

import pytest
from nooa_cli.tui.config import Config
from nooa_memory.reflection import ReflectionReport


def _configure_project(monkeypatch, tmp_path):
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    import nooa_cli.tui.session_manager as session_manager

    monkeypatch.setattr(session_manager, "SESSIONS_DIR", project_dir / "sessions")
    return project_dir


async def _bootstrap_memory_agent(
    tmp_path, monkeypatch, *, reflection=True, debounce_s=0.05, grace_s=0.2
):
    """A bootstrapped TUIAgent with session memory + short reflection knobs.

    Generative hooks are pinned OFF: mechanics tests must never construct
    real LLM-backed callables — the episode-phase tests below inject scripted
    ones explicitly.
    """
    from nooa_cli.tui.bootstrap import bootstrap

    project_dir = _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.agent.summarization.policy = "none"
    cfg.tui.memory = "session"
    cfg.tui.reflection = reflection
    cfg.tui.reflection_generative = False
    cfg.tui.reflection_debounce_s = debounce_s
    cfg.tui.reflection_grace_s = grace_s
    result = await bootstrap(cfg)
    return cfg, result, project_dir


async def _wait_for(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within timeout")


def _slow_reflect(item_sleep_s: float, items: int):
    """A reflect_interruptible stand-in: per-item sleeps honoring should_stop."""

    def reflect(should_stop, *, trigger="idle"):
        t0 = time.monotonic()
        for _ in range(items):
            if should_stop():
                return ReflectionReport(
                    interrupted=True,
                    stopped_in="merge_duplicates",
                    duration_ms=(time.monotonic() - t0) * 1000,
                )
            time.sleep(item_sleep_s)
        return ReflectionReport(merged=1, duration_ms=(time.monotonic() - t0) * 1000)

    return reflect


# ---------------------------------------------------------------------------
# config + bootstrap
# ---------------------------------------------------------------------------


def test_reflection_config_defaults():
    cfg = Config.load()
    assert cfg.tui.reflection is False
    assert cfg.tui.reflection_agents == {}
    assert cfg.tui.reflection_debounce_s == 10.0
    assert cfg.tui.reflection_grace_s == 0.5


@pytest.mark.asyncio
async def test_bootstrap_reflection_off_keeps_post_task_policy(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, reflection=False)
    try:
        runner = result.agent._tui_reflection_runner
        assert runner.enabled is False
        assert result.agent.memory._mgr.config.reflection.trigger == "post_task"
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_reflection_on_remaps_trigger_to_manual(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, reflection=True)
    try:
        runner = result.agent._tui_reflection_runner
        assert runner.enabled is True
        assert result.agent.memory._mgr.config.reflection.trigger == "manual"
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_memory_written_feeds_dirty_and_run_resets_it(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    try:
        runner = result.agent._tui_reflection_runner
        mgr = result.agent.memory._mgr
        assert runner.dirty == 0
        mgr.remember("the deploy command is make ship")
        mgr.remember("rollback runbook: run make undeploy twice")
        assert runner.dirty == 2

        runner.on_response_done()
        await _wait_for(lambda: runner.last_report is not None)
        # ReflectionCompleted is not dirt: a run never re-triggers itself.
        assert runner.dirty == 0
        assert runner.state == "idle"
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


# ---------------------------------------------------------------------------
# start gate truth table: {enabled, dirty, agent-idle}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "dirty", "idle", "should_run"),
    [
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
async def test_start_gate_truth_table(tmp_path, monkeypatch, enabled, dirty, idle, should_run):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    try:
        agent = result.agent
        runner = agent._tui_reflection_runner
        mgr = agent.memory._mgr
        calls: list[str] = []

        def fake_reflect(should_stop, *, trigger="idle"):
            calls.append(trigger)
            return ReflectionReport()

        monkeypatch.setattr(mgr, "reflect_interruptible", fake_reflect)

        runner.enabled = enabled
        runner.dirty = 3 if dirty else 0
        if not idle:
            agent._user_messages_in.put("queued follow-up")

        runner.on_response_done()
        await asyncio.sleep(cfg.tui.reflection_debounce_s * 4)

        assert (len(calls) > 0) is should_run
        if should_run:
            assert calls == ["idle"]
        else:
            assert runner.state == "idle"
        if not idle:
            agent._user_messages_in.flush()
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_pending_non_user_queue_work_counts_as_not_idle(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    try:
        agent = result.agent
        runner = agent._tui_reflection_runner
        runner.dirty = 1
        agent._system_messages_in.put("pending system work")

        runner.on_response_done()
        assert runner.state == "idle"  # gate refused before any task was made
        await asyncio.sleep(cfg.tui.reflection_debounce_s * 4)
        assert runner.last_report is None
        agent._system_messages_in.flush()
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


# ---------------------------------------------------------------------------
# debounce + interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debounce_cancellation_never_shows_indicator(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, debounce_s=1.0)
    try:
        runner = result.agent._tui_reflection_runner
        mgr = result.agent.memory._mgr
        called = []
        monkeypatch.setattr(
            mgr, "reflect_interruptible", lambda s, *, trigger="idle": called.append(1)
        )

        runner.dirty = 1
        runner.on_response_done()
        assert runner.state == "debounce"
        assert runner.indicator_frame() == ""  # never shown during DEBOUNCE

        await runner.interrupt()  # input arrived during debounce
        assert runner.state == "idle"
        assert runner.indicator_frame() == ""  # silent: no 'interrupted' flash
        await asyncio.sleep(0.05)
        assert called == []
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_interrupt_stops_slow_run_within_grace(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, grace_s=0.5)
    try:
        runner = result.agent._tui_reflection_runner
        mgr = result.agent.memory._mgr
        # 100 items x 20 ms: a full pass takes 2 s, one item stops in 20 ms.
        monkeypatch.setattr(mgr, "reflect_interruptible", _slow_reflect(0.02, 100))

        runner.dirty = 1
        runner.on_response_done()
        await _wait_for(lambda: runner.state == "running")

        t0 = time.monotonic()
        await runner.interrupt()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5  # within grace, not the 2 s full pass

        await _wait_for(lambda: runner.last_report is not None)
        assert runner.last_report.interrupted is True
        assert runner.last_report.stopped_in == "merge_duplicates"
        assert runner.state == "idle"
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_prompt_never_blocked_by_stuck_run(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, grace_s=0.1)
    try:
        runner = result.agent._tui_reflection_runner
        mgr = result.agent.memory._mgr

        def stuck_reflect(should_stop, *, trigger="idle"):
            time.sleep(0.8)  # one stuck item that refuses to observe the stop
            return ReflectionReport(interrupted=True, stopped_in="merge_duplicates")

        monkeypatch.setattr(mgr, "reflect_interruptible", stuck_reflect)

        runner.dirty = 1
        runner.on_response_done()
        await _wait_for(lambda: runner.state == "running")

        t0 = time.monotonic()
        await runner.interrupt()
        elapsed = time.monotonic() - t0
        # Timestamp-asserted: interrupt() returned after grace, long before
        # the stuck item finished — the prompt would dispatch here.
        assert elapsed < 0.5
        assert runner.state == "running"  # executor still winding down

        await _wait_for(lambda: runner.state == "idle", timeout=2.0)
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


# ---------------------------------------------------------------------------
# indicator frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indicator_frame_states(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, grace_s=0.5)
    try:
        runner = result.agent._tui_reflection_runner
        mgr = result.agent.memory._mgr
        monkeypatch.setattr(mgr, "reflect_interruptible", _slow_reflect(0.02, 100))

        assert runner.indicator_frame() == ""  # IDLE -> zero width

        runner.dirty = 1
        runner.on_response_done()
        await _wait_for(lambda: runner.state == "running")

        frames = set()
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline and runner.state == "running":
            frames.add(runner.indicator_frame())
            await asyncio.sleep(0.05)
        assert all(f.startswith("✦ reflecting ") for f in frames)
        assert len(frames) >= 2  # the wave glides between repaint ticks

        await runner.interrupt()
        assert runner.indicator_frame() == "✦ reflection interrupted"  # one flash
        await asyncio.sleep(0.4)
        assert runner.indicator_frame() == ""  # flash does not linger
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


# ---------------------------------------------------------------------------
# /reflection command
# ---------------------------------------------------------------------------


def _registry_for(cfg, result):
    from nooa_cli.tui.commands import CommandRegistry

    return CommandRegistry(
        config=cfg.tui,
        agent=result.agent,
        frontend=object(),
        skills_dirs=[],
        session_manager=result.session_manager,
        root_config=cfg,
    )


@pytest.mark.asyncio
async def test_reflection_command_round_trip(tmp_path, monkeypatch):
    cfg, result, project_dir = await _bootstrap_memory_agent(
        tmp_path, monkeypatch, reflection=False
    )
    try:
        registry = _registry_for(cfg, result)
        cmd = registry._commands["reflection"]
        agent_key = "nooa_cli.tui.agent:TUIAgent"

        status = await cmd.execute([])
        assert status.success
        assert "Idle reflection: off" in status.outputs[0].content

        on_result = await cmd.execute(["on"])
        assert on_result.success
        assert cfg.tui.reflection_agents[agent_key] is True
        mgr = result.agent.memory._mgr
        assert mgr.config.reflection.trigger == "manual"
        runner = result.agent._tui_reflection_runner  # rebuilt by the command
        assert runner.enabled is True
        # Persisted as a real bool in settings.yaml — the file Config.load reads.
        reloaded = Config.load()
        assert reloaded.tui.reflection_agents.get(agent_key) is True

        runner.dirty = 2
        runner.last_report = ReflectionReport(merged=3, edges_added=5, pruned=1, duration_ms=1400.0)
        status = await cmd.execute(["status"])
        assert status.success
        text = status.outputs[0].content
        assert "Idle reflection: on" in text
        assert "dirty: 2" in text
        assert "merged 3, +5 edges" in text
        assert "1.4s" in text

        off_result = await cmd.execute(["off"])
        assert off_result.success
        assert cfg.tui.reflection_agents[agent_key] is False
        assert result.agent.memory._mgr.config.reflection.trigger == "post_task"
        assert result.agent._tui_reflection_runner.enabled is False
        reloaded = Config.load()
        assert reloaded.tui.reflection_agents.get(agent_key) is False
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_reflection_status_reports_interrupted_run(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    try:
        registry = _registry_for(cfg, result)
        cmd = registry._commands["reflection"]
        runner = result.agent._tui_reflection_runner
        runner.last_report = ReflectionReport(
            interrupted=True, stopped_in="form_edges", duration_ms=300.0
        )
        status = await cmd.execute([])
        assert "interrupted @ form_edges" in status.outputs[0].content
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_reflection_off_mid_run_cancels_and_rewires(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    try:
        registry = _registry_for(cfg, result)
        cmd = registry._commands["reflection"]
        old_runner = result.agent._tui_reflection_runner
        mgr = result.agent.memory._mgr
        monkeypatch.setattr(mgr, "reflect_interruptible", _slow_reflect(0.02, 100))

        old_runner.dirty = 1
        old_runner.on_response_done()
        await _wait_for(lambda: old_runner.state == "running")

        off_result = await cmd.execute(["off"])
        assert off_result.success
        # The mid-run pass was stopped (within one item) and the old runner
        # torn down: unsubscribed and replaced by a fresh, disabled one.
        await _wait_for(lambda: old_runner.state == "idle", timeout=2.0)
        assert old_runner._unsubscribe is None
        new_runner = result.agent._tui_reflection_runner
        assert new_runner is not old_runner
        assert new_runner.enabled is False
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_memory_off_tears_down_runner(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    try:
        registry = _registry_for(cfg, result)
        memory_cmd = registry._commands["memory"]
        runner = result.agent._tui_reflection_runner

        off_result = await memory_cmd.execute(["off"])
        assert off_result.success
        assert not hasattr(result.agent, "_tui_reflection_runner")
        assert runner._unsubscribe is None  # MemoryWritten handler removed
        assert not hasattr(result.agent, "memory")
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


# ---------------------------------------------------------------------------
# generative episode phase: LLM writes the episode, then consolidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_run_writes_episode_then_reasoner_consumes(tmp_path, monkeypatch):
    """The user-specified pipeline: phase 1 writes an episode about the recent
    session window; consolidation runs next, so the reasoner abstracts that
    fresh episode IN THE SAME PASS — and the runner's own write is not dirt."""
    from nooa_memory import Memory, MemoryType

    from nooa.events import Task

    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    agent = result.agent
    runner = agent._tui_reflection_runner
    mgr = agent.memory._mgr
    try:
        writer_calls: list[str] = []

        def scripted_writer(events_text: str) -> str:
            writer_calls.append(events_text)
            return "Episode: worked through the reflection harness tests end to end."

        def scripted_reasoner(episodes):
            assert any("reflection harness tests" in e.content for e in episodes)
            return [
                Memory(
                    content="Insight: idle reflection needs episode records to abstract from.",
                    type=MemoryType.REFLECTION,
                )
            ]

        runner._episode_writer = scripted_writer
        mgr._reasoner = scripted_reasoner

        agent.event_manager.add(Task(prompt="please run the reflection harness tests"))
        mgr.remember("some fact to make the store dirty")  # dirt -> run eligible
        runner.on_response_done()
        await _wait_for(lambda: runner.last_report is not None and runner.state == "idle")

        assert len(writer_calls) == 1
        assert "reflection harness tests" in writer_calls[0]  # transcript reached the LLM

        episodes = [m for m in mgr.store.all_memories() if m.type is MemoryType.EPISODE]
        assert len(episodes) == 1
        assert episodes[0].title == "session episode"
        assert episodes[0].source_task_ref == "idle_reflection"

        insights = [m for m in mgr.store.all_memories() if m.type is MemoryType.REFLECTION]
        assert len(insights) == 1  # reasoner consumed the fresh episode, same pass
        derived = {e.target_id for e in insights[0].edges}
        assert episodes[0].id in derived

        assert runner.last_report.created == 1
        assert runner.dirty == 0  # the runner's own episode write is not dirt
    finally:
        runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_unnoteworthy_episode_writes_nothing(tmp_path, monkeypatch):
    from nooa_memory import MemoryType

    from nooa.events import Task

    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    agent = result.agent
    runner = agent._tui_reflection_runner
    mgr = agent.memory._mgr
    try:
        runner._episode_writer = lambda events_text: None  # "not noteworthy"
        agent.event_manager.add(Task(prompt="small talk"))
        mgr.remember("dirt")
        runner.on_response_done()
        await _wait_for(lambda: runner.last_report is not None and runner.state == "idle")

        assert [m for m in mgr.store.all_memories() if m.type is MemoryType.EPISODE] == []
    finally:
        runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_interrupt_during_debounce_never_calls_episode_writer(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, debounce_s=5.0)
    agent = result.agent
    runner = agent._tui_reflection_runner
    mgr = agent.memory._mgr
    try:
        calls: list[str] = []
        runner._episode_writer = lambda events_text: calls.append(events_text) or None
        mgr.remember("dirt")
        runner.on_response_done()
        assert runner.state == "debounce"
        await runner.interrupt()
        assert calls == []  # stopped before phase 1 ever started
        assert runner.state == "idle"
    finally:
        runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()


# ---------------------------------------------------------------------------
# /reflection now — eager runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_now_skips_debounce_and_dirty_gate(tmp_path, monkeypatch):
    """Eager runs need neither dirt nor a debounce — and work even while the
    idle toggle is OFF (an explicit request runs whatever is configured)."""
    cfg, result, _ = await _bootstrap_memory_agent(
        tmp_path, monkeypatch, reflection=False, debounce_s=5.0
    )
    runner = result.agent._tui_reflection_runner
    try:
        assert runner.enabled is False and runner.dirty == 0
        assert runner.run_now() is True
        await _wait_for(lambda: runner.last_report is not None and runner.state == "idle")
        assert runner.last_report.interrupted is False  # a full deterministic pass
    finally:
        runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_run_now_refuses_while_a_run_is_pending(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch)
    runner = result.agent._tui_reflection_runner
    mgr = result.agent.memory._mgr
    try:
        monkeypatch.setattr(mgr, "reflect_interruptible", _slow_reflect(0.05, 100))
        assert runner.run_now() is True
        await _wait_for(lambda: runner.state == "running")
        assert runner.run_now() is False  # one run at a time
        await runner.interrupt()
    finally:
        runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_reflection_now_command(tmp_path, monkeypatch):
    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, reflection=False)
    runner = result.agent._tui_reflection_runner
    try:
        registry = _registry_for(cfg, result)
        cmd = registry._commands["reflection"]

        assert cmd.validate_args(["now"])[0] is True
        assert cmd.validate_args(["eagerly"])[0] is False

        out = await cmd.execute(["now"])
        assert out.success
        assert "Reflection started" in out.outputs[0].content
        await _wait_for(lambda: runner.last_report is not None and runner.state == "idle")

        # while a (slow) run is active, a second `now` reports the pending run
        mgr = result.agent.memory._mgr
        monkeypatch.setattr(mgr, "reflect_interruptible", _slow_reflect(0.05, 100))
        assert runner.run_now() is True
        await _wait_for(lambda: runner.state == "running")
        out = await cmd.execute(["now"])
        assert out.success
        assert "already pending" in out.outputs[0].content
        await runner.interrupt()
    finally:
        runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_rebuilt_runner_inherits_preexisting_dirt(tmp_path, monkeypatch):
    """The user-reported bug: memories written BEFORE /reflection on must
    still count as dirt after the toggle rebuilds the runner — otherwise the
    idle gate never opens for exactly the memories the user wants consolidated."""
    from nooa_cli.tui.reflection_runner import ReflectionRunner

    cfg, result, _ = await _bootstrap_memory_agent(tmp_path, monkeypatch, reflection=False)
    agent = result.agent
    mgr = agent.memory._mgr
    old_runner = agent._tui_reflection_runner
    try:
        mgr.remember("written before the reflection toggle, take one")
        mgr.remember("another pre-toggle memory, quite different content")

        # what `/reflection on` does: tear down + rebuild the runner
        old_runner.teardown()
        rebuilt = ReflectionRunner(agent, mgr, cfg.tui, enabled=True)
        assert rebuilt.dirty >= 2  # pre-toggle writes still count
        rebuilt.teardown()

        # after a consolidation pass, a rebuilt runner starts clean
        mgr.reflect_interruptible(lambda: False, trigger="idle")
        fresh = ReflectionRunner(agent, mgr, cfg.tui, enabled=True)
        assert fresh.dirty == 0
        fresh.teardown()
    finally:
        old_runner.teardown()
        if result.session_manager is not None:
            result.session_manager.close()
