# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration-level tests proving ``/clear`` and ``/session new`` don't
destroy the previous session's SQLite data.

**Why this exists.** An earlier bug had ``ClearCommand`` call
``agent.event_manager.clear()``, which wiped the *current* SQLite
storage's events before ``Session._swap_session_manager`` could close
and preserve them. Mock-based tests elsewhere (e.g.
``test_commands.py::test_clear_command_output``) assert
``event_manager.clear`` wasn't called — a negative mock check, not
proof of persistence. These tests use a real ``SQLiteStorageManager``
+ real ``SessionManager`` to prove the positive invariant: old-session
rows stay on disk and reload correctly after the command runs.

Rewritten from the original ``test_clear_bug.py`` after that file was
over-broadly removed during a typeahead-loop cleanup. The integration
path through ``Session.run`` + ``TestFrontend`` isn't restored here —
the unit tests below are what pin the data-preservation invariant;
the full-REPL path is covered indirectly by the other TUI behaviour
tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nooa_cli.tui.commands import (
    ClearCommand,
    CommandHandler,
    CommandRegistry,
)
from nooa_cli.tui.output import ClearScreen
from nooa_cli.tui.session_manager import SessionManager

from nooa.storage import SQLiteStorageManager

# ── helpers ────────────────────────────────────────────────────────────


def _make_sm(tmp_path, *, model="m", agent_cls="A", working_dir="", session_id=None):
    """Create a real SQLite-backed ``SessionManager`` in ``tmp_path``."""
    return SessionManager.create(
        session_id=session_id,
        model=model,
        agent_cls=agent_cls,
        working_dir=working_dir,
    )


def _make_mock_agent(storage: SQLiteStorageManager) -> MagicMock:
    """Mock agent wired to a real ``SQLiteStorageManager``.

    The agent owns a real ``EventManager`` bound to the storage's
    backend so ``clear()`` would actually destroy data (proves the bug
    was real) and so survival is verifiable (proves the fix works).
    """
    from nooa.runtime.event_manager import EventManager

    agent = MagicMock()
    agent._storage = storage
    agent.event_manager = EventManager(backend=storage.event_backend)
    agent.handle = AsyncMock()
    # on() must return a callable (unsubscribe fn) for any renderer wiring.
    agent.event_manager.on = MagicMock(return_value=lambda: None)
    return agent


async def _apply_post_session_swap(result):
    assert result.post_session_swap is not None
    value = result.post_session_swap()
    if hasattr(value, "__await__"):
        await value


# ── /clear preserves old session's SQLite rows ─────────────────────────


@pytest.mark.asyncio
async def test_clear_does_not_destroy_old_session_data(tmp_path):
    """``ClearCommand`` must leave the old session's events on disk.

    Positive assertion: after ``/clear`` runs and the old session
    manager closes, ``SessionManager.load_turns(old_sid)`` returns the
    turn recorded before the command fired.
    """
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        old_sm.record_user("remember me")
        old_sid = old_sm.session_id

        agent = _make_mock_agent(old_sm._storage)

        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )

        result = await cmd.execute([])

        assert result.success is True
        assert any(isinstance(o, ClearScreen) for o in result.outputs)

        # Close the newly-created session manager so we can reload the
        # old one's data cleanly.
        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

        turns = SessionManager.load_turns(old_sid)

    assert len(turns) == 1, (
        f"/clear destroyed the old session's data — expected 1 turn, got {len(turns)}"
    )
    assert turns[0].content == "remember me"


@pytest.mark.asyncio
async def test_clear_resets_in_memory_todos(tmp_path):
    """``/clear`` must wipe the agent's in-memory todo state.

    Swapping storage alone only clears conversation history (events).
    ``TodoManager._todos`` lives on the agent instance, so a naive
    storage swap lets old todos bleed into the new "fresh" session.
    """
    from nooa.tools.todo import TodoManager

    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        agent.todo = TodoManager()
        agent.todo.add("task from previous session")
        assert len(agent.todo._todos) == 1

        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )
        result = await cmd.execute([])
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    assert agent.todo._todos == {}, "/clear left stale todos in memory"
    assert agent.todo._order == []


@pytest.mark.asyncio
async def test_clear_without_todo_skill_is_safe(tmp_path):
    """Agents without a ``todo`` attribute still work — the reset helper
    is guarded."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        if hasattr(agent, "todo"):
            del agent.todo

        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )
        result = await cmd.execute([])
        assert result.success is True

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()


@pytest.mark.asyncio
async def test_clear_creates_new_session_with_different_id(tmp_path):
    """``/clear`` result carries a fresh ``SessionManager`` distinct
    from the old one — i.e. the caller has something to swap to."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        old_sid = old_sm.session_id

        agent = _make_mock_agent(old_sm._storage)
        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )

        result = await cmd.execute([])

        assert result.new_session_manager is not None
        assert result.new_session_manager.session_id != old_sid

        result.new_session_manager.close()
        old_sm.close()


# ── /session new preserves old session's SQLite rows ───────────────────


@pytest.mark.asyncio
async def test_session_new_does_not_destroy_old_session_data(tmp_path):
    """``/session new`` has the same preservation contract as ``/clear``
    — dispatched via the full ``CommandHandler`` path so any registry
    wiring regressions are caught here too."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        old_sm.record_user("keep this")
        old_sid = old_sm.session_id

        agent = _make_mock_agent(old_sm._storage)

        registry = CommandRegistry(
            frontend=AsyncMock(),
            config=MagicMock(default_model="test"),
            agent=agent,
            session_manager=old_sm,
            skills_dirs=None,
            mcp_file=None,
        )
        handler = CommandHandler(registry=registry, frontend=AsyncMock())

        result = await handler.handle("/session new")

        assert result.success is True

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

        turns = SessionManager.load_turns(old_sid)

    assert len(turns) == 1, (
        f"/session new destroyed the old session's data — expected 1 turn, got {len(turns)}"
    )
    assert turns[0].content == "keep this"


# ── subscriber survives storage swap (the original silent-preview bug) ─


@pytest.mark.asyncio
async def test_subscriber_survives_clear_swap(tmp_path):
    """Pins the original silent-preview bug: a handler subscribed once
    on ``agent.event_manager`` keeps firing after ``/clear`` swaps the
    storage underneath. Before the structural fix this test would fail
    because ``agent.event_manager`` was a property and the swap would
    orphan the subscription on the abandoned manager."""
    from nooa.events import Task

    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        # Replace the renderer-style mocked subscription with a real
        # handler so we can prove it fires across the swap.
        from nooa.runtime.event_manager import EventManager

        agent.event_manager = EventManager(backend=old_sm._storage.event_backend)
        received: list[str] = []
        agent.event_manager.on("Task", lambda e: received.append(e.prompt))

        # Pre-swap: handler fires.
        agent.event_manager.add(Task(prompt="before"))
        assert received == ["before"]

        # Run /clear and apply the swap that Session._swap_session_manager would do.
        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )
        result = await cmd.execute([])
        new_sm = result.new_session_manager
        assert new_sm is not None
        agent._storage = new_sm._storage
        agent.event_manager.set_backend(new_sm._storage.event_backend)

        # Post-swap: handler still fires; event lands in the new backend.
        agent.event_manager.add(Task(prompt="after"))
        assert received == ["before", "after"]
        new_tags = [e.tag for e in new_sm._storage.event_backend.all_events()]
        assert any(
            isinstance(e, Task) and e.prompt == "after"
            for e in new_sm._storage.event_backend.all_events()
        ), f"'after' event missing from new backend; tags present: {new_tags}"

        new_sm.close()
        old_sm.close()


@pytest.mark.asyncio
async def test_clear_resets_agent_vars(tmp_path):
    """``/clear`` must wipe ``agent.vars`` so ``self.v`` starts fresh."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        agent.vars = {"spec": "old plan", "cursor": 42}

        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )
        result = await cmd.execute([])
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    assert agent.vars == {}, f"/clear left stale vars: {agent.vars}"


@pytest.mark.asyncio
async def test_clear_resets_user_context_blocks(tmp_path):
    """``/clear`` must remove user-set context blocks but keep protected ones."""
    from nooa.runtime.context_manager import ContextManager

    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)

        cm = ContextManager()
        cm.set_static_protected("system_prompt", "you are an agent")
        cm["user_note"] = "remember this"
        cm["plan"] = "step 1, step 2"
        agent.context_manager = cm

        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )
        result = await cmd.execute([])
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    assert "system_prompt" in cm, "protected block was removed"
    assert "user_note" not in cm, "/clear left user context block 'user_note'"
    assert "plan" not in cm, "/clear left user context block 'plan'"


@pytest.mark.asyncio
async def test_clear_resets_shell(tmp_path):
    """``/clear`` must reset the shell session."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        agent.shell = MagicMock()
        agent.shell.reset = AsyncMock()

        cmd = ClearCommand(
            agent=agent,
            config=MagicMock(default_model="test"),
            frontend=AsyncMock(),
            session_manager=old_sm,
        )
        result = await cmd.execute([])
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    agent.shell.reset.assert_awaited_once()


# ── /session new resets agent state ────────────────────────────────────


@pytest.mark.asyncio
async def test_session_new_resets_agent_vars(tmp_path):
    """``/session new`` must wipe ``agent.vars`` so ``self.v`` starts fresh."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        agent.vars = {"spec": "old plan", "cursor": 42}

        registry = CommandRegistry(
            frontend=AsyncMock(),
            config=MagicMock(default_model="test"),
            agent=agent,
            session_manager=old_sm,
            skills_dirs=None,
            mcp_file=None,
        )
        handler = CommandHandler(registry=registry, frontend=AsyncMock())

        result = await handler.handle("/session new")
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    assert agent.vars == {}, f"/session new left stale vars: {agent.vars}"


@pytest.mark.asyncio
async def test_session_new_resets_user_context_blocks(tmp_path):
    """``/session new`` must remove user-set context blocks but keep protected ones."""
    from nooa.runtime.context_manager import ContextManager

    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)

        cm = ContextManager()
        cm.set_static_protected("system_prompt", "you are an agent")
        cm["user_note"] = "remember this"
        agent.context_manager = cm

        registry = CommandRegistry(
            frontend=AsyncMock(),
            config=MagicMock(default_model="test"),
            agent=agent,
            session_manager=old_sm,
            skills_dirs=None,
            mcp_file=None,
        )
        handler = CommandHandler(registry=registry, frontend=AsyncMock())

        result = await handler.handle("/session new")
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    assert "system_prompt" in cm, "protected block was removed"
    assert "user_note" not in cm, "/session new left user context block"


@pytest.mark.asyncio
async def test_session_new_resets_shell(tmp_path):
    """``/session new`` must reset the shell session."""
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        old_sm = _make_sm(tmp_path)
        agent = _make_mock_agent(old_sm._storage)
        agent.shell = MagicMock()
        agent.shell.reset = AsyncMock()

        registry = CommandRegistry(
            frontend=AsyncMock(),
            config=MagicMock(default_model="test"),
            agent=agent,
            session_manager=old_sm,
            skills_dirs=None,
            mcp_file=None,
        )
        handler = CommandHandler(registry=registry, frontend=AsyncMock())

        result = await handler.handle("/session new")
        await _apply_post_session_swap(result)

        if result.new_session_manager is not None:
            result.new_session_manager.close()
        old_sm.close()

    agent.shell.reset.assert_awaited_once()
