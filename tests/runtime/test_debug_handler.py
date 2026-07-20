# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.runtime.debug_handler.

Covers:
- register_llm_call / unregister_llm_call
- llm_call_context context manager
- _dump_pending_llm_calls
- _detect_llm_in_stack
- _get_debug_dump_path
- _dump_cell_code
- _debug_signal_handler
- install_debug_handler
- dump_debug_info
"""

import io
import linecache
import os
import signal
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_module_state():
    """Reset module-level globals to clean state between tests."""
    from pathlib import Path

    import nooa.runtime.debug_handler as dh

    dh._pending_llm_calls.clear()
    dh._llm_call_counter = 0
    dh._pending_code_execs.clear()
    dh._code_exec_counter = 0
    dh._handler_installed = False
    dh._dump_dir = Path(".")


# ---------------------------------------------------------------------------
# register_llm_call / unregister_llm_call
# ---------------------------------------------------------------------------


class TestRegisterUnregisterLlmCall:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_register_returns_call_id(self):
        from nooa.runtime.debug_handler import register_llm_call

        call_id = register_llm_call("gpt-4")
        assert call_id.startswith("llm_")

    def test_register_increments_counter(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, register_llm_call

        id1 = register_llm_call("gpt-4")
        id2 = register_llm_call("claude-3")
        assert id1 != id2
        assert len(_pending_llm_calls) == 2

    def test_register_stores_metadata(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, register_llm_call

        call_id = register_llm_call("gpt-4", prompt_tokens=1500, endpoint="https://api.openai.com")
        info = _pending_llm_calls[call_id]
        assert info["model"] == "gpt-4"
        assert info["prompt_tokens"] == 1500
        assert info["endpoint"] == "https://api.openai.com"
        assert "start_time" in info
        assert "start_timestamp" in info
        assert "thread" in info

    def test_register_stores_extra_metadata(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, register_llm_call

        call_id = register_llm_call("gpt-4", custom_key="custom_value")
        info = _pending_llm_calls[call_id]
        assert info["custom_key"] == "custom_value"

    def test_unregister_removes_call(self):
        from nooa.runtime.debug_handler import (
            _pending_llm_calls,
            register_llm_call,
            unregister_llm_call,
        )

        call_id = register_llm_call("gpt-4")
        assert call_id in _pending_llm_calls
        unregister_llm_call(call_id)
        assert call_id not in _pending_llm_calls

    def test_unregister_nonexistent_no_error(self):
        from nooa.runtime.debug_handler import unregister_llm_call

        # Should not raise
        unregister_llm_call("nonexistent_id")

    def test_register_optional_fields_none(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, register_llm_call

        call_id = register_llm_call("gpt-4")
        info = _pending_llm_calls[call_id]
        assert info["prompt_tokens"] is None
        assert info["endpoint"] is None

    def test_thread_name_recorded(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, register_llm_call

        call_id = register_llm_call("gpt-4")
        info = _pending_llm_calls[call_id]
        assert info["thread"] == threading.current_thread().name

    def test_concurrent_register_unique_ids(self):
        from nooa.runtime.debug_handler import register_llm_call

        ids = []
        errors = []

        def worker():
            try:
                cid = register_llm_call("gpt-4")
                ids.append(cid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(ids)) == 10  # All IDs must be unique


# ---------------------------------------------------------------------------
# llm_call_context
# ---------------------------------------------------------------------------


class TestLlmCallContext:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_registers_and_unregisters(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, llm_call_context

        with llm_call_context(model="gpt-4") as call_id:
            assert call_id in _pending_llm_calls
        assert call_id not in _pending_llm_calls

    def test_yields_call_id(self):
        from nooa.runtime.debug_handler import llm_call_context

        with llm_call_context(model="gpt-4") as call_id:
            assert call_id.startswith("llm_")

    def test_unregisters_on_exception(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, llm_call_context

        call_id_holder = []
        with pytest.raises(ValueError):
            with llm_call_context(model="gpt-4") as call_id:
                call_id_holder.append(call_id)
                raise ValueError("test error")

        assert call_id_holder[0] not in _pending_llm_calls

    def test_passes_metadata(self):
        from nooa.runtime.debug_handler import _pending_llm_calls, llm_call_context

        with llm_call_context(
            model="claude-3", prompt_tokens=500, endpoint="https://api.anthropic.com"
        ) as call_id:
            info = _pending_llm_calls[call_id]
            assert info["model"] == "claude-3"
            assert info["prompt_tokens"] == 500
            assert info["endpoint"] == "https://api.anthropic.com"


# ---------------------------------------------------------------------------
# _dump_pending_llm_calls
# ---------------------------------------------------------------------------


class TestDumpPendingLlmCalls:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_no_calls_message(self):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls

        out = io.StringIO()
        _dump_pending_llm_calls(out)
        assert "No pending LLM calls" in out.getvalue()

    def test_uses_stderr_by_default(self, capsys):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls

        _dump_pending_llm_calls()
        captured = capsys.readouterr()
        assert "PENDING LLM CALLS" in captured.err

    def test_shows_call_info(self):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls, register_llm_call

        call_id = register_llm_call("gpt-4", prompt_tokens=1000, endpoint="https://api.openai.com")
        out = io.StringIO()
        _dump_pending_llm_calls(out)
        content = out.getvalue()
        assert "gpt-4" in content
        assert "1000" in content
        assert "https://api.openai.com" in content
        assert call_id in content

    def test_shows_elapsed_time(self):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls, register_llm_call

        register_llm_call("gpt-4")
        out = io.StringIO()
        _dump_pending_llm_calls(out)
        assert "Waiting:" in out.getvalue()

    def test_no_tokens_line_when_none(self):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls, register_llm_call

        register_llm_call("gpt-4")  # No prompt_tokens
        out = io.StringIO()
        _dump_pending_llm_calls(out)
        assert "Prompt tokens" not in out.getvalue()

    def test_no_endpoint_line_when_none(self):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls, register_llm_call

        register_llm_call("gpt-4")  # No endpoint
        out = io.StringIO()
        _dump_pending_llm_calls(out)
        assert "Endpoint:" not in out.getvalue()

    def test_multiple_calls(self):
        from nooa.runtime.debug_handler import _dump_pending_llm_calls, register_llm_call

        register_llm_call("gpt-4")
        register_llm_call("claude-3")
        out = io.StringIO()
        _dump_pending_llm_calls(out)
        content = out.getvalue()
        assert "gpt-4" in content
        assert "claude-3" in content


# ---------------------------------------------------------------------------
# _detect_llm_in_stack
# ---------------------------------------------------------------------------


class TestDetectLlmInStack:
    def test_returns_empty_for_normal_frame(self):
        from nooa.runtime.debug_handler import _detect_llm_in_stack

        frame = sys._getframe()
        result = _detect_llm_in_stack(frame)
        # Current test frame won't match any LLM pattern
        assert isinstance(result, list)

    def test_detects_pattern_in_fake_frame(self):
        from nooa.runtime.debug_handler import _detect_llm_in_stack

        # Build a fake frame chain where the filename contains 'httpx'
        fake_frame = MagicMock()
        fake_frame.f_code.co_filename = "/site-packages/httpx/_client.py"
        fake_frame.f_code.co_name = "send"
        fake_frame.f_lineno = 100
        fake_frame.f_back = None

        result = _detect_llm_in_stack(fake_frame)
        assert any("httpx" in item.lower() for item in result)

    def test_no_duplicates_for_same_pattern(self):
        from nooa.runtime.debug_handler import _detect_llm_in_stack

        frame2 = MagicMock()
        frame2.f_code.co_filename = "/site-packages/httpx/_transport.py"
        frame2.f_code.co_name = "handle"
        frame2.f_lineno = 200
        frame2.f_back = None

        frame1 = MagicMock()
        frame1.f_code.co_filename = "/site-packages/httpx/_client.py"
        frame1.f_code.co_name = "send"
        frame1.f_lineno = 100
        frame1.f_back = frame2

        result = _detect_llm_in_stack(frame1)
        # Should only be listed once despite appearing in two frames
        descriptions = [item.split(" at ")[0] for item in result]
        assert descriptions.count("HTTP client (httpx)") == 1

    def test_handles_none_frame(self):
        from nooa.runtime.debug_handler import _detect_llm_in_stack

        result = _detect_llm_in_stack(None)
        assert result == []

    def test_handles_exception_gracefully(self):
        from nooa.runtime.debug_handler import _detect_llm_in_stack

        bad_frame = MagicMock()
        bad_frame.f_code.co_filename = "/site-packages/litellm/main.py"
        bad_frame.f_code.co_name = "completion"
        bad_frame.f_lineno = 42
        # Make f_back raise
        type(bad_frame).f_back = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))

        # Should not raise
        result = _detect_llm_in_stack(bad_frame)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _get_debug_dump_path
# ---------------------------------------------------------------------------


class TestGetDebugDumpPath:
    def test_returns_path_with_pid(self):
        from nooa.runtime.debug_handler import _get_debug_dump_path

        path = _get_debug_dump_path()
        assert str(os.getpid()) in path.name
        assert path.name.startswith("debug_dump_")
        assert path.suffix == ".txt"

    def test_uses_dump_dir(self, tmp_path):
        import nooa.runtime.debug_handler as dh

        original = dh._dump_dir
        try:
            dh._dump_dir = tmp_path
            path = dh._get_debug_dump_path()
            assert path.parent == tmp_path
        finally:
            dh._dump_dir = original


# ---------------------------------------------------------------------------
# _dump_cell_code
# ---------------------------------------------------------------------------


class TestDumpCellCode:
    def setup_method(self):
        # Remove any pre-existing Cell entries added by our tests
        for k in list(linecache.cache.keys()):
            if k.startswith("Cell "):
                del linecache.cache[k]

    def teardown_method(self):
        for k in list(linecache.cache.keys()):
            if k.startswith("Cell "):
                del linecache.cache[k]

    def test_no_cells_message(self):
        from nooa.runtime.debug_handler import _dump_cell_code

        out = io.StringIO()
        _dump_cell_code(out)
        assert "No Cell code" in out.getvalue()

    def test_uses_stderr_by_default(self, capsys):
        from nooa.runtime.debug_handler import _dump_cell_code

        _dump_cell_code()
        captured = capsys.readouterr()
        assert "REGISTERED CELL CODE" in captured.err

    def test_dumps_cell_entries(self):
        from nooa.runtime.debug_handler import _dump_cell_code

        lines = ["x = 1\n", "y = 2\n"]
        linecache.cache["Cell exec_1[0]"] = (len("".join(lines)), None, lines, "Cell exec_1[0]")

        out = io.StringIO()
        _dump_cell_code(out)
        content = out.getvalue()
        assert "Cell exec_1[0]" in content
        assert "x = 1" in content
        assert "y = 2" in content

    def test_shows_line_count(self):
        from nooa.runtime.debug_handler import _dump_cell_code

        lines = ["a = 1\n", "b = 2\n", "c = 3\n"]
        linecache.cache["Cell exec_2[0]"] = (len("".join(lines)), None, lines, "Cell exec_2[0]")

        out = io.StringIO()
        _dump_cell_code(out)
        assert "3 lines" in out.getvalue()


# ---------------------------------------------------------------------------
# _debug_signal_handler
# ---------------------------------------------------------------------------


class TestDebugSignalHandler:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_writes_dump_file(self, tmp_path):
        from nooa.runtime.debug_handler import _debug_signal_handler

        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        with patch(
            "nooa.runtime.debug_handler._get_debug_dump_path",
            return_value=dump_path,
        ):
            _debug_signal_handler(signal.SIGUSR2, frame)

        assert dump_path.exists()
        content = dump_path.read_text()
        assert "DEBUG DUMP" in content

    def test_writes_signal_number(self, tmp_path):
        from nooa.runtime.debug_handler import _debug_signal_handler

        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        with patch(
            "nooa.runtime.debug_handler._get_debug_dump_path",
            return_value=dump_path,
        ):
            _debug_signal_handler(12, frame)

        content = dump_path.read_text()
        assert "12" in content

    def test_writes_to_stderr(self, tmp_path, capsys):
        from nooa.runtime.debug_handler import _debug_signal_handler

        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        with patch(
            "nooa.runtime.debug_handler._get_debug_dump_path",
            return_value=dump_path,
        ):
            _debug_signal_handler(signal.SIGUSR2, frame)

        captured = capsys.readouterr()
        assert "DEBUG DUMP" in captured.err

    def test_includes_pending_llm_calls(self, tmp_path):
        from nooa.runtime.debug_handler import (
            _debug_signal_handler,
            register_llm_call,
        )

        register_llm_call("gpt-4")
        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        with patch(
            "nooa.runtime.debug_handler._get_debug_dump_path",
            return_value=dump_path,
        ):
            _debug_signal_handler(signal.SIGUSR2, frame)

        content = dump_path.read_text()
        assert "PENDING LLM CALLS" in content
        assert "gpt-4" in content

    def test_handles_file_write_error(self, tmp_path, capsys):
        from nooa.runtime.debug_handler import _debug_signal_handler

        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        # The error path is triggered when open() raises inside the try block
        with (
            patch(
                "nooa.runtime.debug_handler._get_debug_dump_path",
                return_value=dump_path,
            ),
            patch("builtins.open", side_effect=OSError("permission denied")),
        ):
            # Should not raise — error is caught and written to stderr
            _debug_signal_handler(signal.SIGUSR2, frame)

        captured = capsys.readouterr()
        assert "Debug dump error" in captured.err

    def test_shows_llm_stuck_warning_when_pending_calls(self, tmp_path):
        from nooa.runtime.debug_handler import (
            _debug_signal_handler,
            register_llm_call,
        )

        register_llm_call("gpt-4")
        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        with patch(
            "nooa.runtime.debug_handler._get_debug_dump_path",
            return_value=dump_path,
        ):
            _debug_signal_handler(signal.SIGUSR2, frame)

        content = dump_path.read_text()
        assert "STUCK IN LLM CALL" in content

    def test_shows_llm_stuck_when_detected_in_stack(self, tmp_path):
        from nooa.runtime.debug_handler import _debug_signal_handler

        dump_path = tmp_path / "debug_dump_test.txt"
        frame = sys._getframe()

        detected_items = ["HTTP client (httpx) at /site-packages/httpx/_client.py:100 in send()"]

        with (
            patch(
                "nooa.runtime.debug_handler._get_debug_dump_path",
                return_value=dump_path,
            ),
            patch(
                "nooa.runtime.debug_handler._detect_llm_in_stack",
                return_value=detected_items,
            ),
        ):
            _debug_signal_handler(signal.SIGUSR2, frame)

        content = dump_path.read_text()
        assert "STUCK IN LLM CALL" in content
        assert "httpx" in content


# ---------------------------------------------------------------------------
# install_debug_handler
# ---------------------------------------------------------------------------


class TestInstallDebugHandler:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()
        # Restore signal to default if possible
        try:
            signal.signal(signal.SIGUSR2, signal.SIG_DFL)
        except (ValueError, OSError):
            pass

    def test_install_sets_handler_installed(self):
        import nooa.runtime.debug_handler as dh

        dh.install_debug_handler()
        assert dh._handler_installed is True

    def test_install_idempotent(self):
        import nooa.runtime.debug_handler as dh

        dh.install_debug_handler()
        # Call again — should not re-install
        with patch("signal.signal") as mock_signal:
            dh.install_debug_handler()
            mock_signal.assert_not_called()

    def test_install_enables_faulthandler(self):

        import nooa.runtime.debug_handler as dh

        with patch("faulthandler.enable") as mock_fh:
            dh.install_debug_handler()
            mock_fh.assert_called_once()

    def test_install_handles_signal_error_gracefully(self):
        import nooa.runtime.debug_handler as dh

        with (
            patch("signal.signal", side_effect=ValueError("not main thread")),
            patch("faulthandler.enable"),
        ):
            # Should not raise
            dh.install_debug_handler()
        # _handler_installed remains False when signal.signal fails
        assert dh._handler_installed is False

    def test_install_handles_os_error(self):
        import nooa.runtime.debug_handler as dh

        with (
            patch("signal.signal", side_effect=OSError("platform error")),
            patch("faulthandler.enable"),
        ):
            dh.install_debug_handler()
        assert dh._handler_installed is False

    def test_dump_dir_arg_sets_module_state(self, tmp_path):
        import nooa.runtime.debug_handler as dh

        dh.install_debug_handler(dump_dir=tmp_path)
        assert dh._dump_dir == tmp_path


# ---------------------------------------------------------------------------
# dump_debug_info
# ---------------------------------------------------------------------------


class TestDumpDebugInfo:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_writes_to_provided_file(self):
        from nooa.runtime.debug_handler import dump_debug_info

        out = io.StringIO()
        dump_debug_info(out)
        content = out.getvalue()
        assert "Manual debug dump" in content

    def test_writes_to_stderr_by_default(self, capsys):
        from nooa.runtime.debug_handler import dump_debug_info

        dump_debug_info()
        captured = capsys.readouterr()
        assert "Manual debug dump" in captured.err

    def test_includes_pending_calls(self):
        from nooa.runtime.debug_handler import dump_debug_info, register_llm_call

        register_llm_call("gpt-4")
        out = io.StringIO()
        dump_debug_info(out)
        assert "gpt-4" in out.getvalue()

    def test_includes_cell_code_section(self):
        from nooa.runtime.debug_handler import dump_debug_info

        out = io.StringIO()
        dump_debug_info(out)
        assert "REGISTERED CELL CODE" in out.getvalue()


# ---------------------------------------------------------------------------
# Code-exec phase tracking + get_activity (/activity slash command)
# ---------------------------------------------------------------------------


class TestCodeExecTracking:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_register_unregister_code_exec(self):
        """register/unregister add and remove a code-exec entry with a preview."""
        from nooa.runtime.debug_handler import (
            get_pending_code_execs,
            register_code_exec,
            unregister_code_exec,
        )

        assert get_pending_code_execs() == []
        exec_id = register_code_exec("x = 1\nprint(x)")
        pending = get_pending_code_execs()
        assert len(pending) == 1
        assert pending[0]["exec_id"] == exec_id
        assert pending[0]["preview"] == "x = 1"
        assert pending[0]["elapsed"] >= 0
        unregister_code_exec(exec_id)
        assert get_pending_code_execs() == []

    def test_code_exec_context_clears_on_exit(self):
        """code_exec_context registers on enter and clears on normal exit."""
        from nooa.runtime.debug_handler import (
            code_exec_context,
            get_pending_code_execs,
        )

        with code_exec_context("a = 2"):
            assert len(get_pending_code_execs()) == 1
        assert get_pending_code_execs() == []

    def test_code_exec_context_clears_on_exception(self):
        """code_exec_context clears the entry even if the body raises."""
        from nooa.runtime.debug_handler import (
            code_exec_context,
            get_pending_code_execs,
        )

        with pytest.raises(ValueError):
            with code_exec_context("boom"):
                assert len(get_pending_code_execs()) == 1
                raise ValueError("boom")
        assert get_pending_code_execs() == []

    def test_preview_blank_for_empty_code(self):
        """Empty or None code yields a blank preview, never an index error."""
        from nooa.runtime.debug_handler import (
            get_pending_code_execs,
            register_code_exec,
        )

        register_code_exec("")
        assert get_pending_code_execs()[0]["preview"] == ""
        register_code_exec(None)
        assert get_pending_code_execs()[1]["preview"] == ""


class TestGetActivity:
    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_idle_when_nothing_in_flight(self):
        """get_activity reports idle with empty lists when nothing runs."""
        from nooa.runtime.debug_handler import get_activity

        activity = get_activity()
        assert activity["phase"] == "idle"
        assert activity["code_execs"] == []
        assert activity["llm_calls"] == []

    def test_executing_python_phase(self):
        """An open code-exec context makes get_activity report executing_python."""
        from nooa.runtime.debug_handler import code_exec_context, get_activity

        with code_exec_context("y = 3"):
            activity = get_activity()
            assert activity["phase"] == "executing_python"
            assert len(activity["code_execs"]) == 1

    def test_waiting_llm_phase(self):
        """An in-flight LLM call makes get_activity report waiting_llm."""
        from nooa.runtime.debug_handler import get_activity, llm_call_context

        with llm_call_context(model="gpt-test"):
            activity = get_activity()
            assert activity["phase"] == "waiting_llm"
            assert activity["llm_calls"][0]["model"] == "gpt-test"

    def test_python_wins_when_llm_call_made_from_cell(self):
        """A cell that is itself blocked on an LLM call still reports executing_python."""
        from nooa.runtime.debug_handler import (
            code_exec_context,
            get_activity,
            llm_call_context,
        )

        with code_exec_context("z = 4"):
            with llm_call_context(model="gpt-test"):
                activity = get_activity()
                # In a code cell that is itself blocked on the model — the cell
                # is what the agent is "running", so executing_python wins.
                assert activity["phase"] == "executing_python"
                assert len(activity["code_execs"]) == 1
                assert len(activity["llm_calls"]) == 1


# --- Event-driven activity tracking via LLMCallStart/LLMCallEnd "on" hooks ---


def test_attach_activity_tracking_reflects_llm_calls():
    from nooa.events import LLMCallEnd, LLMCallStart
    from nooa.runtime.debug_handler import (
        attach_activity_tracking,
        get_activity,
    )
    from nooa.runtime.event_manager import EventManager

    em = EventManager()
    unsub = attach_activity_tracking(em)
    try:
        assert get_activity()["phase"] == "idle"

        em.add(
            LLMCallStart(
                method_name="handle", strategy="CodeAct", generation_id="g", turn_number=1
            ),
            record=False,
        )
        act = get_activity()
        assert act["phase"] == "waiting_llm"
        assert len(act["llm_calls"]) == 1

        em.add(
            LLMCallEnd(method_name="handle", strategy="CodeAct", generation_id="g", turn_number=1),
            record=False,
        )
        assert get_activity()["phase"] == "idle"
    finally:
        unsub()


def test_attach_activity_tracking_clears_on_unsubscribe():
    from nooa.events import LLMCallStart
    from nooa.runtime.debug_handler import (
        attach_activity_tracking,
        get_activity,
    )
    from nooa.runtime.event_manager import EventManager

    em = EventManager()
    unsub = attach_activity_tracking(em)
    em.add(
        LLMCallStart(method_name="m", strategy="s", generation_id="g2", turn_number=1),
        record=False,
    )
    assert get_activity()["phase"] == "waiting_llm"
    # Unsubscribing must clear this tracker's still-pending registrations.
    unsub()
    assert get_activity()["phase"] == "idle"
