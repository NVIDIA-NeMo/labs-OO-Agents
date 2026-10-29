# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-Windows policy: refused at import unless NOOA_SKIP_PLATFORM_CHECK=1;
with the override the pure-Python core works and POSIX-only features name WSL 2.

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

from nooa._platform import (
    PosixOnlyFeatureError,
    UnsupportedPlatformError,
    check_platform,
    require_posix,
)
from nooa.errors import NemoOOAgentsRuntimeError

REPO = Path(__file__).resolve().parents[1]


def test_check_platform_refuses_native_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("NOOA_SKIP_PLATFORM_CHECK", raising=False)
    with pytest.raises(UnsupportedPlatformError) as info:
        check_platform()
    assert isinstance(info.value, ImportError)
    assert "WSL 2" in str(info.value)
    assert "NOOA_SKIP_PLATFORM_CHECK" in str(info.value)


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_check_platform_override(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("NOOA_SKIP_PLATFORM_CHECK", value)
    check_platform()


def test_check_platform_is_noop_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("NOOA_SKIP_PLATFORM_CHECK", raising=False)
    check_platform()


def test_import_and_cli_refused_under_windows_simulation() -> None:
    env = {**os.environ, "NOOA_SKIP_PLATFORM_CHECK": "0"}
    winsim = str(REPO / "scripts" / "winsim.py")

    def run(code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, winsim, "-c", code],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    for code in ("import nooa", "import nooa.trace_explorer"):
        lib = run(code)
        assert lib.returncode == 1, (code, lib.stderr)
        assert lib.stderr.startswith("Error: nooa does not support native Windows"), lib.stderr
        assert "Traceback" not in lib.stderr
        assert len(lib.stderr.strip().splitlines()) == 1
    cli = run("import nooa_cli; nooa_cli.main()")
    assert cli.returncode == 1
    assert cli.stderr.startswith("Error: nooa does not support native Windows"), cli.stderr
    assert "Traceback" not in cli.stderr


def test_uncaught_refusal_prints_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("NOOA_SKIP_PLATFORM_CHECK", raising=False)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    with pytest.raises(UnsupportedPlatformError) as info:
        check_platform()
    sys.excepthook(info.type, info.value, info.tb)
    assert capsys.readouterr().err == f"Error: {info.value}\n"


def test_custom_excepthook_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    def custom(*args: object) -> None:
        pass

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("NOOA_SKIP_PLATFORM_CHECK", raising=False)
    monkeypatch.setattr(sys, "excepthook", custom)
    with pytest.raises(UnsupportedPlatformError):
        check_platform()
    assert sys.excepthook is custom


@pytest.mark.parametrize("value", ["0", "false", "", "on"])
def test_other_override_values_still_refuse(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("NOOA_SKIP_PLATFORM_CHECK", value)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    with pytest.raises(UnsupportedPlatformError):
        check_platform()


# Every importable nooa module must stay importable on native Windows with the
# override: POSIX-only modules are imported inside the functions that use them.
_IMPORT_ALL = r"""
import importlib, pkgutil, sys
import nooa, nooa_cli
POSIX = {"fcntl", "resource", "termios", "tty", "pty", "pwd", "grp", "readline", "curses", "_curses", "syslog"}
bad = []
for pkg in (nooa, nooa_cli):
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            if exc.name in POSIX:
                bad.append(f"{info.name}: {exc!r}")
        except AttributeError as exc:
            if "module 'os'" in str(exc) or "module 'signal'" in str(exc):
                bad.append(f"{info.name}: {exc!r}")
        except Exception:
            pass  # optional extras that are not installed, and so on
print("\n".join(bad) or "ALL IMPORTS OK")
sys.exit(1 if bad else 0)
"""


def test_every_module_imports_under_windows_simulation() -> None:
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
