# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionResumed must be emitted AFTER skills attach, so subscribers receive it.

Regression: bootstrap() emitted SessionResumed before library skills (e.g.
agent_mesh) were attached in build_registry(). The event fired into the void —
no subscriber existed yet — so a skill's resume handler never ran. Moving the
emit into build_registry() (after skill activation) fixes the ordering.
"""

import pytest
from nooa_cli.sessions import SessionResumed


@pytest.mark.asyncio
async def test_emit_happens_after_skill_can_subscribe(tmp_path, monkeypatch):
    """A handler subscribed during build_registry() receives the resume event."""
    from nooa_cli.tui import session_manager as session_manager_module
    from nooa_cli.tui.bootstrap import bootstrap, build_registry
    from nooa_cli.tui.config import Config

    monkeypatch.setattr(session_manager_module, "SESSIONS_DIR", tmp_path)

    result = await bootstrap(Config())

    # Subscribe BEFORE build_registry runs the emit — emulating a skill that
    # attaches during build_registry's library-skill activation.
    seen = []
    result.agent.event_manager.register_event_type(SessionResumed)
    result.agent.event_manager.on("SessionResumed", lambda e: seen.append(e))

    # build_registry needs a frontend; a minimal stub is enough for the emit path.
    from unittest.mock import MagicMock

    build_registry(result, MagicMock())

    assert len(seen) == 1
    assert seen[0].session_id == result.session_id
    await result.agent.close()
    result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_does_not_emit_on_its_own(tmp_path, monkeypatch):
    """bootstrap() must NOT emit — otherwise it fires before skills attach."""
    from nooa_cli.tui import session_manager as session_manager_module
    from nooa_cli.tui.bootstrap import bootstrap
    from nooa_cli.tui.config import Config

    monkeypatch.setattr(session_manager_module, "SESSIONS_DIR", tmp_path)

    from nooa.runtime.event_manager import EventManager

    captured = []
    real_add = EventManager.add

    def _tee(self, event, **kw):
        if isinstance(event, SessionResumed):
            captured.append(event)
        return real_add(self, event, **kw)

    import pytest as _pytest  # noqa: F401

    EventManager.add = _tee
    try:
        result = await bootstrap(Config())
    finally:
        EventManager.add = real_add

    assert captured == []  # bootstrap itself emits nothing now
    await result.agent.close()
    result.session_manager.close()


def test_bootstrap_result_carries_restored_flag():
    from nooa_cli.tui.bootstrap import BootstrapResult

    assert "restored" in BootstrapResult.__dataclass_fields__
