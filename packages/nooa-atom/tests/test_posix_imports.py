# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every nooa_atom module imports with the POSIX-only stdlib hidden (review #3).

AGENTS.md: POSIX-only modules (``fcntl``...) and names (``os.killpg``...) are
used inside the functions that need them, after ``require_posix``; importing
a module never touches them. ``scripts/winsim.py`` hides them as on Windows.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nooa._platform import PosixOnlyFeatureError

REPO = Path(__file__).resolve().parents[3]

_IMPORT_ALL = r"""
import importlib, pkgutil, sys
import nooa_atom
POSIX = {"fcntl", "resource", "termios", "tty", "pty", "pwd", "grp", "readline", "curses", "_curses", "syslog"}
bad = []
for info in pkgutil.walk_packages(nooa_atom.__path__, "nooa_atom."):
    if info.name.endswith("__main__"):
        continue
    try:
        importlib.import_module(info.name)
    except ModuleNotFoundError as exc:
        if exc.name in POSIX or exc.name is None:
            bad.append(f"{info.name}: {exc!r}")
    except Exception as exc:
        bad.append(f"{info.name}: {exc!r}")
print("\n".join(bad) or "ALL IMPORTS OK")
sys.exit(1 if bad else 0)
"""


def test_every_nooa_atom_module_imports_without_posix_only_names():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "winsim.py"), "-c", _IMPORT_ALL],
        cwd=REPO,
        env={**os.environ, "NOOA_SKIP_PLATFORM_CHECK": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL IMPORTS OK" in result.stdout


@pytest.fixture
def native_windows(monkeypatch):
    monkeypatch.setattr("nooa._platform.is_native_windows", lambda: True)


def test_posix_features_name_themselves_on_windows(native_windows, tmp_path):
    from nooa_atom.acp.router import process_spawn
    from nooa_atom.acp.worker import start_parent_watchdog
    from nooa_atom.session.store import _lock_is_held
    from nooa_atom.skills.mcp_auth import FileTokenStorage

    with pytest.raises(PosixOnlyFeatureError, match="router"):
        process_spawn(["worker"])
    with pytest.raises(PosixOnlyFeatureError, match="watchdog"):
        start_parent_watchdog()
    with pytest.raises(PosixOnlyFeatureError, match="Session locks"):
        _lock_is_held(tmp_path / "lock")
    storage = FileTokenStorage("https://mcp.example/", tmp_path / "tokens.json")
    with pytest.raises(PosixOnlyFeatureError, match="token cache"):
        storage._update(access="x")
