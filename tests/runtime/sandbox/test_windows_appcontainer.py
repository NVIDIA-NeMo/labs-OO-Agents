# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests: prove the AppContainer backend *refuses*, not just runs.

Every test here is written so that its **failure mode is a containment leak**,
not a broken feature. A test that merely asserts "the cell returned 42" proves
the happy path; these assert that the same worker cannot reach what it must not
reach, and that the refusal comes from the OS rather than from Python code the
cell could bypass.

They need a private runtime root, which is expensive to build (~4.5 min,
~420 MB), so they are marked ``sandbox`` and skipped unless one is present:

    set NOOA_SANDBOX_RUNTIME=C:\\path\\to\\runtime-root
    pytest -m sandbox tests/runtime/sandbox/test_windows_appcontainer.py

The runtime root is built by ``experiments/windows-sandbox-probe/_mkroot.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(os.name != "nt", reason="AppContainer backend is Windows-only"),
]

_HERE = Path(__file__).resolve().parent
# The agent module must be importable by name in the container, which means it
# has to be a real module on a granted root, not a class in this test file.
_AGENT_MODULE = _HERE / "win_agent_module.py"


def _runtime_root() -> Path | None:
    explicit = os.environ.get("NOOA_SANDBOX_RUNTIME")
    if explicit:
        root = Path(explicit)
        return root if (root / "python.exe").exists() else None
    try:
        from nooa.runtime.sandbox.win_runtime import default_runtime_root

        root = default_runtime_root()
    except Exception:  # noqa: BLE001 - absence is the normal case off this host
        return None
    return root if (root / "python.exe").exists() else None


ROOT = _runtime_root()
pytestmark.append(pytest.mark.skipif(ROOT is None, reason="no sandbox runtime root available"))


def _config(workspace: Path, **overrides):
    from nooa.runtime.sandbox.config import SandboxConfig

    values = {
        "filesystem": True,
        "network": False,
        "workspace": str(workspace),
        "start_method": "spawn",
        "backend": "appcontainer",
        "require": True,
        "runtime_root": str(ROOT) if ROOT else None,
    }
    values.update(overrides)
    return SandboxConfig(**values)


class _Session:
    """One executor plus the temp workspace it was granted."""

    def __init__(self, workspace: Path, **overrides):
        from nooa.runtime.sandbox.executor import SandboxedExecutor

        self.workspace = workspace
        self.executor = SandboxedExecutor(
            _agent_instance(),
            _config(workspace, **overrides),
            cell_timeout=90,
            framework_builtins={},
        )

    def run(self, code: str, count: int = 1):
        return asyncio.run(self.executor.run_cell(code, execution_count=count))

    def close(self) -> None:
        asyncio.run(self.executor.aclose())


def _agent_instance():
    import sys

    sys.path.insert(0, str(_HERE))
    from win_agent_module import WinProbeAgent

    return WinProbeAgent()


@pytest.fixture
def session(tmp_path):
    # The agent module's directory is granted read-only, so the module must sit
    # on a path the process can find. It already does (tests/runtime/sandbox).
    workspace = tmp_path / "ws"
    workspace.mkdir()
    s = _Session(workspace)
    try:
        yield s
    finally:
        s.close()


# --- the happy path, as the baseline for every denial below ------------------
def test_cell_executes_inside_container(session):
    result = session.run("value = self.answer()\nvalue")
    assert result.error is None, result.error
    assert result.returned_value == 42


def test_broker_round_trip(session):
    result = session.run("self.echo('hi')")
    assert result.error is None, result.error
    assert result.returned_value == "probe:hi"


# --- filesystem denials ------------------------------------------------------
def test_cannot_read_outside_grants(session, tmp_path):
    """A file next to the workspace — not granted — must be unreadable.

    Using a sibling in the same temp directory keeps the test deterministic on
    any machine: the path exists, so a failure is EACCES and not ENOENT. A
    missing path would make this test pass for the wrong reason.
    """
    secret = tmp_path / "outside.txt"
    secret.write_text("classified", encoding="utf-8")
    result = session.run(f"open(r'{secret}').read()")
    assert result.error is not None, "the container read a file outside its grants"
    assert "PermissionError" in type(result.error).__name__ or "Permission denied" in str(
        result.error
    )


def test_cannot_read_the_project_checkout(session):
    """The source tree is never granted: the package comes from the runtime root."""
    target = Path(__file__).resolve().parents[3] / "src" / "nooa" / "__init__.py"
    result = session.run(f"open(r'{target}','rb').read(8)")
    assert result.error is not None, "the container read the project checkout"
    assert "Permission" in str(result.error) or "PermissionError" in type(result.error).__name__


def test_can_write_inside_workspace(session):
    result = session.run("open('inside.txt','w').write('ok')")
    assert result.error is None, result.error
    assert (session.workspace / "inside.txt").read_text(encoding="utf-8") == "ok"


def test_cannot_write_outside_workspace(session, tmp_path):
    target = tmp_path / "forbidden.txt"
    result = session.run(f"open(r'{target}','w').write('nope')")
    assert result.error is not None, "the container wrote outside its workspace"
    assert not target.exists(), "the file was created despite the reported error"


# --- network -----------------------------------------------------------------
def test_outbound_tcp_is_blocked(session):
    """No capability SIDs: socket() itself fails, so no request can escape.

    The exact exception differs by Python version (PermissionError from
    WSAEACCES, or OSError); what matters is that the connection was refused,
    not that it timed out.
    """
    code = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "s.connect(('1.1.1.1', 443))\n"
    )
    result = session.run(code)
    assert result.error is not None, "the container opened an outbound TCP connection"


# --- resource limits ---------------------------------------------------------
def test_cpu_limit_kills_and_the_session_recovers(session):
    """Exhausting the job's CPU budget terminates the worker, and the next cell
    still runs. A kill that left the session unusable would be a different bug
    (and a much more annoying one)."""
    session.close()
    session = _Session(session.workspace, max_cpu_seconds=1)
    try:
        result = session.run("while True:\n    pass")
        assert result.error is not None, "the CPU limit did not stop a runaway cell"
        # The process was killed by the job object; the exact type is
        # CellTimeoutError, but any reported error is evidence of containment.
        followup = session.run("1 + 1")
        assert followup.error is None, followup.error
        assert followup.returned_value == 2
    finally:
        session.close()


# --- teardown ----------------------------------------------------------------
def test_teardown_revokes_container_grants(tmp_path):
    """After close, the workspace DACL must no longer name the container SID.

    A leftover ACE is persistent — it survives the process — and would silently
    grant the next container more than it asked for.
    """
    from nooa.runtime.sandbox import win_api
    from nooa.runtime.sandbox.win_runtime import DEFAULT_CONTAINER_NAME

    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace)
    sid_text = win_api.sid_string(win_api.container_sid(DEFAULT_CONTAINER_NAME))
    try:
        assert session.run("1").error is None
        assert sid_text in win_api.read_dacl(str(workspace)), (
            "the grant was never written, so this test proves nothing"
        )
    finally:
        session.close()
    assert sid_text not in win_api.read_dacl(str(workspace)), "the container ACE survived teardown"
