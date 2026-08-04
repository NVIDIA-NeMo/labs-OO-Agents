# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gl-212 asyncio.Lock loop-mismatch diagnostic instrumentation.

Covers the diagnostic code path in ``Session._loud_handler`` that fires
when an exception contains "bound to a different event loop".
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

from nooa_cli.tui.session import Session


def _make_session_stub(*, emit_block_side_effect=None):
    """Build a minimal Session stub with only the fields _loud_handler touches."""
    session = Session.__new__(Session)
    session._loud_handler_reentrant = False
    session._app = MagicMock()
    if emit_block_side_effect:
        session._app.emit_block.side_effect = emit_block_side_effect
    # Use a MagicMock as _startup_loop to avoid leaking real event loops.
    # The diagnostic code only calls id() and `is` on it.
    session._startup_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    session.agent = MagicMock()
    # Default: agent has no shell or _actor
    session.agent.shell = None
    session.agent._actor = None
    return session


class TestDiagnosticTrigger:
    """Diagnostics fire only for loop-mismatch exceptions."""

    def test_triggers_on_bound_to_different_loop(self):
        """Diagnostic dump emits when exception message matches."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("Task <foo> cb=[...] bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        # emit_block should have been called at least once (diag + normal)
        calls = session._app.emit_block.call_args_list
        assert len(calls) >= 1
        diag_text = calls[0][0][0]
        assert "[gl-212]" in diag_text
        assert "diagnostic dump" in diag_text

    def test_does_not_trigger_on_unrelated_exception(self):
        """No diagnostic dump for exceptions without the loop-mismatch message."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = ValueError("something else entirely")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        # emit_block called once for normal handler, NOT for diagnostics
        calls = session._app.emit_block.call_args_list
        assert len(calls) == 1
        text = calls[0][0][0]
        assert "[gl-212]" not in text

    def test_does_not_trigger_without_exception(self):
        """No diagnostic dump when context has no exception."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        context = {"message": "Task was destroyed but it is pending!"}

        session._loud_handler(loop, context)

        calls = session._app.emit_block.call_args_list
        # May be 0 or 1 depending on message filtering, but never has [gl-212]
        for call in calls:
            assert "[gl-212]" not in call[0][0]


class TestDiagnosticContent:
    """The diagnostic dump contains expected loop/lock information."""

    def test_includes_handler_loop_id(self):
        """Dump shows the handler loop's id."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert f"id={id(loop):#x}" in diag_text

    def test_includes_startup_loop_id(self):
        """Dump shows the startup loop's id."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert f"id={id(session._startup_loop):#x}" in diag_text

    def test_includes_same_flag_when_loops_match(self):
        """When handler loop IS the startup loop, same=True."""
        session = _make_session_stub()
        loop = session._startup_loop  # same loop
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "same=True" in diag_text

    def test_includes_same_flag_when_loops_differ(self):
        """When handler loop is NOT the startup loop, same=False."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "same=False" in diag_text

    def test_includes_exception_repr(self):
        """Dump shows the repr of the exception."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "RuntimeError" in diag_text
        assert "bound to a different event loop" in diag_text

    def test_includes_restart_suggestion(self):
        """Dump includes the user-facing suggestion to restart."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "restart the TUI" in diag_text
        assert "gl#212" in diag_text


class TestLockInspection:
    """Lock inspection paths in the diagnostic dump."""

    def test_bash_session_lock_reported(self):
        """When BashSession._lock exists, it is included in the dump."""
        session = _make_session_stub()
        mock_lock = asyncio.Lock()
        session.agent.shell = MagicMock()
        session.agent.shell._session = MagicMock()
        session.agent.shell._session._lock = mock_lock

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "BashSession._lock:" in diag_text

    def test_bash_session_lock_with_loop_attr(self):
        """When lock has _loop attribute (pre-3.10 or custom lock), its id is reported.

        Note: real asyncio.Lock on Python >=3.12 has no _loop attribute. This test
        simulates a non-standard/legacy lock that does have _loop to exercise the
        hasattr(lock, "_loop") code path.
        """
        session = _make_session_stub()
        mock_lock = MagicMock()
        mock_lock._loop = MagicMock(spec=asyncio.AbstractEventLoop)
        # hasattr check must pass
        session.agent.shell = MagicMock()
        session.agent.shell._session = MagicMock()
        session.agent.shell._session._lock = mock_lock

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "lock._loop: id=" in diag_text

    def test_bash_session_lock_without_loop_attr(self):
        """On Python 3.12+, Lock has no _loop attr — no crash, no lock._loop line."""
        session = _make_session_stub()
        # Real asyncio.Lock on 3.12+ has no _loop attribute
        real_lock = asyncio.Lock()
        session.agent.shell = MagicMock()
        session.agent.shell._session = MagicMock()
        session.agent.shell._session._lock = real_lock

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "BashSession._lock:" in diag_text
        # On Python 3.12+, _loop doesn't exist so this line should NOT appear
        if not hasattr(real_lock, "_loop"):
            assert "lock._loop: id=" not in diag_text

    def test_actor_generation_lock_reported(self):
        """When Actor._generation_lock exists, it is included in the dump."""
        session = _make_session_stub()
        mock_lock = asyncio.Lock()
        session.agent._actor = MagicMock()
        session.agent._actor._generation_lock = mock_lock

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "Actor._generation_lock:" in diag_text

    def test_bash_inspection_failure_handled(self):
        """If shell inspection raises, it's caught and noted."""
        session = _make_session_stub()

        # Use a dedicated class to avoid leaking property descriptors onto MagicMock
        class _Agent:
            @property
            def shell(self):
                raise RuntimeError("boom")

            _actor = None

        session.agent = _Agent()

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "BashSession inspection failed" in diag_text

    def test_actor_inspection_failure_handled(self):
        """If actor inspection raises, it's caught and noted."""
        session = _make_session_stub()

        # Use a dedicated class to avoid leaking property descriptors onto MagicMock
        class _Agent:
            shell = None

            @property
            def _actor(self):
                raise RuntimeError("kaboom")

        session.agent = _Agent()

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        assert "Actor inspection failed" in diag_text

    def test_no_shell_no_crash(self):
        """When agent has no shell attribute, no crash."""
        session = _make_session_stub()
        session.agent.shell = None

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        # Should not raise
        session._loud_handler(loop, context)

    def test_no_actor_no_crash(self):
        """When agent has no _actor attribute, no crash."""
        session = _make_session_stub()
        session.agent._actor = None

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        # Should not raise
        session._loud_handler(loop, context)


class TestReentrancyGuard:
    """Diagnostic emit uses reentrancy guard correctly."""

    def test_reentrant_falls_back_to_stderr(self):
        """When _loud_handler_reentrant is True, diag goes to stderr."""
        session = _make_session_stub()
        session._loud_handler_reentrant = True

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        fake_stderr = StringIO()
        with patch.object(sys, "__stderr__", fake_stderr):
            session._loud_handler(loop, context)

        output = fake_stderr.getvalue()
        assert "[gl-212]" in output

    def test_non_reentrant_uses_emit_block(self):
        """When not reentrant, diagnostic goes through emit_block."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        calls = session._app.emit_block.call_args_list
        assert any("[gl-212]" in call[0][0] for call in calls)

    def test_emit_block_failure_in_diag_falls_back_to_stderr(self):
        """If emit_block raises during diagnostic emit, fall back to stderr.

        The diagnostic try/except catches the failure and writes to stderr
        instead, then continues to the normal handler path so the original
        traceback is not lost.
        """
        session = _make_session_stub()
        call_count = [0]

        def emit_block_side_effect(text):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("emit_block failed")

        session._app.emit_block.side_effect = emit_block_side_effect

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        fake_stderr = StringIO()
        with patch.object(sys, "__stderr__", fake_stderr):
            session._loud_handler(loop, context)

        # Diagnostic was written to stderr as fallback
        assert "[gl-212]" in fake_stderr.getvalue()
        # Normal handler still ran (second emit_block call)
        assert call_count[0] >= 2

    def test_reentrant_flag_reset_after_emit(self):
        """_loud_handler_reentrant is reset to False after emit_block."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        assert session._loud_handler_reentrant is False


class TestFallThrough:
    """After diagnostics, the normal handler path still executes."""

    def test_normal_handler_formats_exception_after_diag(self):
        """The full traceback is still emitted via normal handler path."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "error in task", "exception": exc}

        session._loud_handler(loop, context)

        # emit_block called at least twice: once for diag, once for normal
        calls = session._app.emit_block.call_args_list
        assert len(calls) >= 2
        # Second call is the normal handler output
        normal_text = calls[1][0][0]
        assert "RuntimeError" in normal_text

    def test_unclosed_session_still_filtered_after_diag_path(self):
        """'Unclosed client session' messages are still dropped even if diag
        section was not triggered."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        context = {"message": "Unclosed client session"}

        session._loud_handler(loop, context)

        # Should not have emitted anything
        session._app.emit_block.assert_not_called()

    def test_unclosed_connector_still_filtered(self):
        """'Unclosed connector' messages are still dropped."""
        session = _make_session_stub()
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        context = {"message": "Unclosed connector"}

        session._loud_handler(loop, context)

        session._app.emit_block.assert_not_called()


class TestStartupLoop:
    """_startup_loop is captured and used correctly."""

    def test_startup_loop_is_none_when_not_set(self):
        """If _startup_loop was never set, diagnostic still works (shows None)."""
        session = _make_session_stub()
        del session._startup_loop  # simulate never having run()

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        exc = RuntimeError("bound to a different event loop")
        context = {"message": "", "exception": exc}

        session._loud_handler(loop, context)

        diag_text = session._app.emit_block.call_args_list[0][0][0]
        # getattr with default None should produce id(None)
        assert "startup loop:" in diag_text


class TestBangShell:
    """Tests for the TUI-owned shell used by bang (!) commands."""

    async def test_get_bang_shell_creates_shell_tools(self):
        """_get_bang_shell lazily creates a ShellTools instance."""
        from nooa.tools.shell_tools import ShellTools

        session = Session.__new__(Session)
        session._bang_shell = None
        session.agent = MagicMock()
        session.agent.shell = MagicMock()
        session.agent.shell.cwd = tempfile.gettempdir()

        shell = await session._get_bang_shell()
        assert isinstance(shell, ShellTools)
        assert session._bang_shell is shell

    async def test_get_bang_shell_returns_same_instance(self):
        """Repeated calls return the same ShellTools instance."""
        session = Session.__new__(Session)
        session._bang_shell = None
        session.agent = MagicMock()
        session.agent.shell = MagicMock()
        session.agent.shell.cwd = tempfile.gettempdir()

        shell1 = await session._get_bang_shell()
        shell2 = await session._get_bang_shell()
        assert shell1 is shell2

    async def test_get_bang_shell_is_not_agent_shell(self):
        """The bang shell is a distinct instance from the agent's shell."""
        from nooa.tools.shell_tools import ShellTools

        session = Session.__new__(Session)
        session._bang_shell = None
        session.agent = MagicMock()
        session.agent.shell = ShellTools(cwd=tempfile.gettempdir())

        bang_shell = await session._get_bang_shell()
        assert bang_shell is not session.agent.shell

    async def test_get_bang_shell_uses_agent_cwd(self):
        """The bang shell inherits cwd from the agent's shell."""
        from pathlib import Path

        session = Session.__new__(Session)
        session._bang_shell = None
        session.agent = MagicMock()
        session.agent.shell = MagicMock()
        session.agent.shell.cwd = Path("/some/dir")

        shell = await session._get_bang_shell()
        assert str(shell.cwd) == "/some/dir"

    async def test_get_bang_shell_syncs_cwd(self):
        """When agent shell cwd changes, bang shell syncs on next call."""
        from pathlib import Path

        session = Session.__new__(Session)
        session._bang_shell = None
        session.agent = MagicMock()
        session.agent.shell = MagicMock()
        session.agent.shell.cwd = Path(tempfile.gettempdir())

        shell = await session._get_bang_shell()
        assert str(shell.cwd) == tempfile.gettempdir()

        # Simulate agent changing directory to a real path
        session.agent.shell.cwd = Path("/")
        await session._get_bang_shell()
        # The bang shell should have cd'd to /
        assert str(shell.cwd) == "/"
