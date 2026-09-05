# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

try:
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None

import json
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nooa_cli.tui.runtime_registration import (
    TUIRuntimeRegistration,
    explicit_resume_argv,
    reexec_tui,
)

requires_fcntl = pytest.mark.skipif(fcntl is None, reason="requires POSIX file locking")


def test_explicit_resume_argv_replaces_short_and_long_forms():
    """Canonicalize existing short and long continue options."""
    assert explicit_resume_argv(
        ["/venv/bin/python", "/venv/bin/nooa", "tui", "-w", "/work", "-c", "old"], "new"
    ) == ["/venv/bin/python", "/venv/bin/nooa", "tui", "-w", "/work", "--continue", "new"]
    assert explicit_resume_argv(
        ["/venv/bin/python", "-m", "nooa_cli", "tui", "--continue=old", "--python"], "new"
    ) == ["/venv/bin/python", "-m", "nooa_cli", "tui", "--python", "--continue", "new"]


def test_explicit_resume_argv_inserts_continue_before_option_terminator():
    """Keep the resume option parseable when argv contains an option terminator."""
    argv = ["python", "-m", "nooa_cli", "tui", "--python", "--", "payload", "-c"]
    assert explicit_resume_argv(argv, "new") == [
        "python",
        "-m",
        "nooa_cli",
        "tui",
        "--python",
        "--continue",
        "new",
        "--",
        "payload",
        "-c",
    ]


def test_explicit_resume_argv_preserves_interpreter_c_before_tui():
    """Do not confuse Python interpreter flags with TUI options."""
    argv = ["python", "-c", "launch_tui()", "tui", "--continue", "old"]
    assert explicit_resume_argv(argv, "new") == [
        "python",
        "-c",
        "launch_tui()",
        "tui",
        "--continue",
        "new",
    ]


def test_explicit_resume_argv_requires_tui_command():
    """Reject invocations whose TUI command boundary cannot be identified."""
    with pytest.raises(ValueError, match="tui command"):
        explicit_resume_argv(["python", "-c", "launch_tui()"], "new")


@requires_fcntl
def test_publish_rejects_symlinked_lock_without_touching_target(tmp_path: Path):
    """Reject a planted lock symlink without modifying its target."""
    target = tmp_path / "target"
    target.write_text("keep me")
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
        patch("sys.orig_argv", ["python", "-m", "nooa_cli", "tui"]),
    ):
        registration = TUIRuntimeRegistration(
            session_id="session-1",
            working_dir=str(tmp_path),
            original_argv=["python", "-m", "nooa_cli", "tui"],
        )
        registration.lock_path.symlink_to(target)
        with pytest.raises(OSError):
            registration.publish()
        registration.close()
    assert target.read_text() == "keep me"


def test_registration_disables_restart_without_fcntl(tmp_path: Path):
    """Disable registration cleanly when POSIX file locking is unavailable."""
    loop = MagicMock()
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
        patch("sys.orig_argv", ["python", "-m", "nooa_cli", "tui"]),
        patch("nooa_cli.tui.runtime_registration.fcntl", None),
    ):
        registration = TUIRuntimeRegistration(
            session_id="session-1",
            working_dir=str(tmp_path),
            original_argv=["python", "-m", "nooa_cli", "tui"],
        )
        assert registration.install_restart_signal(loop, MagicMock()) is False
        with pytest.raises(OSError, match="POSIX file locking"):
            registration.publish()


@requires_fcntl
def test_registration_publishes_private_record_and_removes_only_its_own(tmp_path: Path):
    """Publish private metadata and avoid deleting a successor record."""
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=Path("/source")),
        patch("nooa_cli.tui.runtime_registration._source_revision", return_value="abc123"),
        patch("sys.orig_argv", ["/venv/bin/python", "/venv/bin/nooa", "tui", "-w", "/work"]),
    ):
        registration = TUIRuntimeRegistration(session_id="session-1", working_dir="/work")
        registration.publish()

    payload = json.loads(registration.path.read_text())
    assert registration.path.parent == tmp_path
    assert payload["pid"] == registration.pid
    assert payload["process_identity"]
    assert payload["restart_signal"] == "SIGUSR1"
    assert payload["session_id"] == "session-1"
    assert payload["source_root"] == "/source"
    assert payload["source_revision"] == "abc123"
    assert "argv" not in payload
    assert registration.path.stat().st_mode & 0o777 == 0o600
    assert registration.path.parent.stat().st_mode & 0o777 == 0o700

    payload["token"] = "successor"
    registration.path.write_text(json.dumps(payload))
    registration.close()
    assert registration.path.exists()

    payload["token"] = registration.token
    registration.path.write_text(json.dumps(payload))
    registration.close()
    assert not registration.path.exists()


@requires_fcntl
def test_install_restart_signal_routes_callback_and_restores_handler(tmp_path: Path):
    """Install and restore the graceful-restart signal handler."""
    if not hasattr(signal, "SIGUSR1"):
        return
    loop = MagicMock()
    callback = MagicMock()
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
        patch("nooa_cli.tui.runtime_registration.signal.getsignal", return_value=signal.SIG_DFL),
        patch("nooa_cli.tui.runtime_registration.signal.signal") as set_signal,
    ):
        registration = TUIRuntimeRegistration(
            session_id="session-1",
            working_dir=str(tmp_path),
            original_argv=["python", "-m", "nooa_cli", "tui"],
        )
        assert registration.install_restart_signal(loop, callback)
        registration.close()

    loop.add_signal_handler.assert_called_once_with(signal.SIGUSR1, callback)
    loop.remove_signal_handler.assert_called_once_with(signal.SIGUSR1)
    set_signal.assert_called_once_with(signal.SIGUSR1, signal.SIG_DFL)


@requires_fcntl
def test_update_session_republishes_resume_target(tmp_path: Path):
    """Track an in-process session change in metadata and restart argv."""
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
        patch("nooa_cli.tui.runtime_registration._source_revision", return_value="abc123"),
        patch("sys.orig_argv", ["python", "-m", "nooa_cli", "tui", "-c", "old"]),
    ):
        registration = TUIRuntimeRegistration(session_id="old", working_dir=str(tmp_path))
        registration.publish()
        registration.update_session("new")

    payload = json.loads(registration.path.read_text())
    assert payload["session_id"] == "new"
    assert "argv" not in payload
    assert registration.restart_argv[-2:] == ["--continue", "new"]
    registration.close()


def test_reexec_uses_path_search():
    """Re-execute through PATH so console-script launches remain valid."""
    with patch("nooa_cli.tui.runtime_registration.os.execvp") as execvp:
        reexec_tui(["python", "-m", "nooa_cli", "tui"])
    execvp.assert_called_once_with("python", ["python", "-m", "nooa_cli", "tui"])


@requires_fcntl
def test_publish_fails_closed_without_process_identity(tmp_path: Path):
    """Do not advertise restart without a stable process identity."""
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
        patch("nooa_cli.tui.runtime_registration.process_identity", return_value=None),
    ):
        registration = TUIRuntimeRegistration(
            session_id="session-1",
            working_dir=str(tmp_path),
            original_argv=["python", "-m", "nooa_cli", "tui"],
        )
        with pytest.raises(OSError, match="stable process identity"):
            registration.publish()
    assert not registration.path.exists()


@requires_fcntl
def test_close_releases_lock_when_record_disappeared(tmp_path: Path):
    """Release ownership even when the runtime record was removed."""
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
    ):
        registration = TUIRuntimeRegistration(
            session_id="session-1",
            working_dir=str(tmp_path),
            original_argv=["python", "-m", "nooa_cli", "tui"],
        )
        registration.publish()
        registration.path.unlink()
        registration.close()
    assert registration._lock_fd is None
    assert not registration.lock_path.exists()


@requires_fcntl
def test_registration_holds_runtime_ownership_lock(tmp_path: Path):
    """Hold the ownership lock until registration closes."""
    with (
        patch.dict("os.environ", {"NOOA_TUI_RUNTIME_DIR": str(tmp_path)}),
        patch("nooa_cli.tui.runtime_registration._source_root", return_value=tmp_path),
    ):
        registration = TUIRuntimeRegistration(
            session_id="session-1",
            working_dir=str(tmp_path),
            original_argv=["python", "-m", "nooa_cli", "tui"],
        )
        registration.publish()
        contender = os.open(registration.lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
        registration.close()
    assert not registration.lock_path.exists()
