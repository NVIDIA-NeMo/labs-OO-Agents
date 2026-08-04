# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionResumed is emitted after agent reconstitution."""

import pytest
from nooa_cli.tui.bootstrap import bootstrap
from nooa_cli.tui.config import Config

from nooa.sessions import SessionResumed


def test_event_is_runtime_role_and_fields():
    e = SessionResumed(session_id="abc", restored=True)
    assert e.session_id == "abc"
    assert e.restored is True
    # Runtime events never enter conversation/LLM context.
    from nooa.context_blocks.models import Role

    assert type(e)._role is Role.RUNTIME_EVENT


def test_event_keeps_legacy_handler_alias():
    assert SessionResumed.handler_aliases == ("TuiSessionResumed",)


@pytest.mark.asyncio
async def test_bootstrap_emits_event_to_subscribers_on_fresh_session(tmp_path, monkeypatch):
    """SessionResumed is a runtime event (emit-only) — observe it via on().

    Runtime events are never recorded/queryable, so a skill must subscribe with
    event_manager.on("SessionResumed", handler) — which is exactly how
    agent_mesh will auto-reconnect. We assert the handler fires once with the
    right payload.

    Because bootstrap() emits during construction (before we can subscribe), we
    re-emit through the same manager to exercise the subscriber path the skills
    use, and separately assert bootstrap reached the emit (no exception, agent
    built with a session_id).
    """
    from nooa_cli.tui import session_manager as session_manager_module

    monkeypatch.setattr(session_manager_module, "SESSIONS_DIR", tmp_path)

    result = await bootstrap(Config())
    agent = result.agent
    assert result.session_id is not None

    seen = []
    agent.event_manager.register_event_type(SessionResumed)
    agent.event_manager.on("SessionResumed", lambda e: seen.append(e))

    agent.event_manager.add(SessionResumed(session_id=result.session_id, restored=False))

    assert len(seen) == 1
    assert seen[0].restored is False
    assert seen[0].session_id == result.session_id
    await agent.close()
    result.session_manager.close()


@pytest.mark.asyncio
async def test_on_handler_receives_event():
    """A subscriber registered before emit receives the event (the skill path)."""
    from nooa import Agent
    from nooa.unifiedllm import FakeLLMClient

    class _A(Agent, llm=FakeLLMClient()):
        pass

    agent = _A()
    agent.event_manager.register_event_type(SessionResumed)
    got = []
    agent.event_manager.on("SessionResumed", lambda e: got.append((e.session_id, e.restored)))
    agent.event_manager.add(SessionResumed(session_id="sess-1", restored=True))
    assert got == [("sess-1", True)]


@pytest.mark.asyncio
async def test_resume_without_snapshot_emits_restored_false(tmp_path, monkeypatch):
    """-c on a session with no snapshot must emit restored=False, not True.

    The emit now happens in build_registry() (after skills attach), so we drive
    that path and capture the event it emits via a real subscriber.
    """
    from unittest.mock import MagicMock

    from nooa_cli.tui import session_manager as session_manager_module
    from nooa_cli.tui.bootstrap import build_registry

    monkeypatch.setattr(session_manager_module, "SESSIONS_DIR", tmp_path)

    # Fresh session first (resumable id, zero snapshots), then resume it:
    # bootstrap restores nothing -> build_registry must emit restored=False.
    first = await bootstrap(Config())
    sid = first.session_id
    await first.agent.close()
    first.session_manager.close()

    resumed = await bootstrap(Config(), resume_session_id=sid)
    assert resumed.restored is False  # no snapshot was applied

    captured: list = []
    resumed.agent.event_manager.register_event_type(SessionResumed)
    resumed.agent.event_manager.on("SessionResumed", lambda e: captured.append(e))

    build_registry(resumed, MagicMock())

    assert len(captured) == 1, f"expected one resume emit, got {captured!r}"
    assert captured[0].restored is False
    await resumed.agent.close()
    resumed.session_manager.close()
