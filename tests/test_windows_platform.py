# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-Windows policy: pure-Python core works, POSIX-only features name WSL 2.

The subprocess test runs ``scripts/windows_smoke.py`` under ``scripts/winsim.py``
(POSIX-only modules, signals and os functions hidden; ``sys.platform="win32"``)
so Linux CI catches a new import-time POSIX dependency. It is a simulation:
it does not exercise the Proactor loop, Windows paths or real file semantics.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from nooa._platform import PosixOnlyFeatureError, require_posix
from nooa.errors import NemoOOAgentsRuntimeError

REPO = Path(__file__).resolve().parents[1]


def test_require_posix_is_noop_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    require_posix("anything")


def test_require_posix_names_feature_and_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(PosixOnlyFeatureError) as info:
        require_posix("The widget")
    assert isinstance(info.value, NemoOOAgentsRuntimeError)
    assert isinstance(info.value, NotImplementedError)
    assert info.value.feature == "The widget"
    assert "The widget requires Linux or macOS" in str(info.value)
    assert "WSL 2" in str(info.value)


def test_bash_session_start_raises_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    from nooa.tools._bash_session import BashSession

    monkeypatch.setattr(sys, "platform", "win32")

    def no_pipe() -> None:
        raise AssertionError("os.pipe must not run on Windows")

    monkeypatch.setattr(os, "pipe", no_pipe)
    with pytest.raises(PosixOnlyFeatureError, match="bash session"):
        asyncio.run(BashSession().start())


def test_file_sqlite_raises_but_memory_works(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nooa.storage import SQLiteStorageManager

    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(PosixOnlyFeatureError, match="SQLite"):
        SQLiteStorageManager(str(tmp_path / "s.db"))
    assert not (tmp_path / "s.lock").exists()
    with SQLiteStorageManager(":memory:"):
        pass


def test_monitor_raises_on_first_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    from nooa.runtime.producers import monitor

    monkeypatch.setattr(sys, "platform", "win32")
    gen = monitor("echo hi")  # creating the generator is fine; iterating is the use

    async def first() -> None:
        await gen.__anext__()

    with pytest.raises(PosixOnlyFeatureError, match="monitor"):
        asyncio.run(first())


def test_debug_handler_skips_without_sigusr2(monkeypatch: pytest.MonkeyPatch) -> None:
    import nooa.runtime.debug_handler as dh

    monkeypatch.delattr(signal, "SIGUSR2")
    monkeypatch.setattr(dh, "_handler_installed", False)
    dh.install_debug_handler()
    assert dh._handler_installed is False


def test_core_runs_under_windows_simulation() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "winsim.py"),
            str(REPO / "scripts" / "windows_smoke.py"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "platform: win32" in result.stdout
    assert "predict: hello" in result.stdout
    assert "codeact: 5" in result.stdout
    for feature in ("sqlite", "bash", "monitor", "sandbox"):
        assert f"clear error [{feature}]" in result.stdout
    assert "SMOKE OK" in result.stdout
