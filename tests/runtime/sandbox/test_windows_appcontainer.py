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
import sys
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


def _require_root() -> bool:
    """CI sets this: a missing root must fail, not skip.

    Without it, a broken runtime-root build in CI would skip every test here
    and report a green job — the same silent gap these tests exist to close.
    """
    return os.environ.get("NOOA_SANDBOX_REQUIRE_ROOT") == "1"


if ROOT is None and _require_root():
    raise RuntimeError(
        "NOOA_SANDBOX_REQUIRE_ROOT=1 but no runtime root was found at "
        f"{os.environ.get('NOOA_SANDBOX_RUNTIME')!r}. Build it before running "
        "the acceptance tests; skipping here would hide a containment gap."
    )

pytestmark.append(
    pytest.mark.skipif(
        ROOT is None and not _require_root(), reason="no sandbox runtime root available"
    )
)


@pytest.fixture(autouse=True)
def _root_only_from_config(monkeypatch):
    """The environment variable only locates the root for this module.

    Every session names it through ``SandboxConfig.runtime_root``. Leaving the
    variable set as well would let a backend that ignores the config field pass
    here anyway, which is how that defect went unnoticed.
    """
    monkeypatch.delenv("NOOA_SANDBOX_RUNTIME", raising=False)


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

    def __init__(self, workspace: Path, *, agent=None, cell_timeout: float = 90, **overrides):
        from nooa.runtime.sandbox.executor import SandboxedExecutor

        self.workspace = workspace
        self.executor = SandboxedExecutor(
            agent if agent is not None else _agent_instance(),
            _config(workspace, **overrides),
            cell_timeout=cell_timeout,
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


def test_agent_built_on_the_quickstart_helpers_runs(tmp_path):
    """An agent written like the examples must start in the container.

    Importing ``nooa.util.quickstart`` runs nooa's config discovery, which the
    agent above never does. With the runtime root inside a checkout (CI's is)
    discovery used to reach the checkout's unreadable ``pyproject.toml`` and
    fail the import with PermissionError.
    """
    sys.path.insert(0, str(_HERE))
    from win_quickstart_agent import QuickstartAgent

    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace, agent=QuickstartAgent())
    try:
        result = session.run("self.answer()")
        assert result.error is None, result.error
        assert result.returned_value == 42
    finally:
        session.close()


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


def test_outbound_udp_round_trip_is_blocked(session):
    """A DNS query must get no reply: a bare ``sendto`` succeeds either way.

    A reply would prove the datagram left. The positive control (the same
    query answered with ``internetClient``) and the wire capture showing the
    datagram dropped before it leaves are in
    ``experiments/windows-sandbox-probe/udp_dns_probe.py``.
    """
    code = (
        "import socket\n"
        "query = bytes.fromhex('abcd01000001000000000000076578616d706c6503636f6d0000010001')\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "s.settimeout(4)\n"
        "s.sendto(query, ('1.1.1.1', 53))\n"
        "s.recvfrom(512)\n"
    )
    result = session.run(code)
    assert result.error is not None, "a UDP query from the container got a reply"


def test_network_true_reaches_the_cell(tmp_path):
    """The control for the two denials above: network=True must connect.

    Without it a launcher that never grants a capability would pass every
    network test while making ``network=True`` silently useless.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace, network=True)
    try:
        code = (
            "import socket\n"
            "socket.create_connection(('1.1.1.1', 443), timeout=4).close()\n"
            "'connected'"
        )
        result = session.run(code)
        assert result.error is None, result.error
        assert result.returned_value == "connected"
    finally:
        session.close()


# --- environment -------------------------------------------------------------
_SENTINEL = "NOOA_TEST_PARENT_ONLY"


def test_parent_environment_does_not_reach_the_cell(tmp_path, monkeypatch):
    """A variable only the parent has must be invisible to cell code.

    This is how a credential leaks: the parent holds an API token in its
    environment, and the container inherits it. The cell reports a boolean, so
    a failure never prints the value.
    """
    monkeypatch.setenv(_SENTINEL, "parent-only")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace)
    try:
        result = session.run(f"import os\n{_SENTINEL!r} in os.environ")
        assert result.error is None, result.error
        assert result.returned_value is False, "a parent-only variable reached the cell"
    finally:
        session.close()


def test_env_passthrough_lets_a_named_variable_in(tmp_path, monkeypatch):
    """The control: the same variable, named in ``env_passthrough``, arrives.

    Without it the test above could pass for the wrong reason — a sentinel set
    too late to be in the environment block at all.
    """
    monkeypatch.setenv(_SENTINEL, "parent-only")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace, env_passthrough=(_SENTINEL,))
    try:
        result = session.run(f"import os\nos.environ.get({_SENTINEL!r})")
        assert result.error is None, result.error
        assert result.returned_value == "parent-only"
    finally:
        session.close()


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


def test_memory_limit_stops_a_runaway_allocation(tmp_path):
    """A cell that allocates past the ceiling must die, not swap or OOM the host.

    ``max_memory_mb`` is headroom above the worker's own baseline, so the
    launcher can only set the job's absolute ``ProcessMemoryLimit`` after the
    worker reports that baseline. Setting it too early kills a worker that has
    not imported the agent yet; setting it too high, or not at all, makes the
    guardrail a no-op that no other test here would notice.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace, max_memory_mb=256)
    try:
        # Touch every page: an untouched bytearray is virtual, and the job's
        # limit is on committed memory, so allocating alone may not trip it.
        code = (
            "blocks = []\n"
            "for _ in range(40):\n"
            "    block = bytearray(32 * 1024 * 1024)\n"
            "    for offset in range(0, len(block), 4096):\n"
            "        block[offset] = 1\n"
            "    blocks.append(block)\n"
            "len(blocks)\n"
        )
        result = session.run(code)
        assert result.error is not None, "the memory limit did not stop the allocation"
        # Measured: the job's ProcessMemoryLimit makes the allocation fail and
        # the cell reports MemoryError; it does not kill the worker. Asserting
        # the error's kind is what distinguishes "the limit bit" from "the cell
        # happened to fail for some other reason".
        assert "memory" in str(result.error).lower(), result.error
        # The worker survives that error; what matters is that the session can
        # still be used, which is what distinguishes a limit from a wedge.
        followup = session.run("1 + 1", count=2)
        assert followup.error is None, followup.error
        assert followup.returned_value == 2
    finally:
        session.close()


def test_memory_limit_leaves_ordinary_work_alone(tmp_path):
    """The control for the test above: the same session runs normal cells.

    Without it, a launcher that sets the limit to something tiny (or to the
    baseline itself) would pass the denial test while making every real cell
    fail. A few MiB is well inside a 256 MiB headroom.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace, max_memory_mb=256)
    try:
        result = session.run("data = bytearray(8 * 1024 * 1024)\nlen(data)")
        assert result.error is None, result.error
        assert result.returned_value == 8 * 1024 * 1024
    finally:
        session.close()


def test_wallclock_timeout_kills_a_cpu_bound_cell(tmp_path):
    """The parent's deadline must kill a cell that ignores every other guard.

    This is the guardrail that does not depend on the child cooperating: no
    capability, ACL or job limit is consulted, the parent simply stops waiting
    and terminates. The follow-up cell is the part that matters — a kill that
    leaves the session unusable is a different, worse failure than the one this
    test exists to catch.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace, cell_timeout=1.0, timeout_grace_s=1.0)
    try:
        result = session.run("while True:\n    pass")
        assert result.error is not None, "the deadline did not stop a runaway cell"
        assert "deadline" in str(result.error).lower() or "kill" in str(result.error).lower()
        followup = session.run("40 + 2", count=2)
        assert followup.error is None, followup.error
        assert followup.returned_value == 42
    finally:
        session.close()


# --- session identity --------------------------------------------------------
def test_recovery_leaves_a_live_sessions_journal_alone(tmp_path):
    """Liveness is the journal's lock, not its existence.

    This is the mechanism the whole fix rests on, so it is tested without a
    container: a locked journal must survive recovery untouched, and the same
    journal must be reclaimed once its owner lets go.
    """
    from nooa.runtime.sandbox.win_runtime import (
        AclJournal,
        RuntimeRoot,
        recover_stale_grants,
    )

    root = RuntimeRoot(tmp_path)
    journal = AclJournal.open(root.journal_for("live-session"), "S-1-15-2-1")
    path = journal.path
    try:
        assert path.exists()
        assert recover_stale_grants(root) == []
        assert path.exists(), "recovery deleted a live session's journal"
        # A second session must not be able to take the same identity either.
        with pytest.raises(BlockingIOError):
            AclJournal.open(root.journal_for("live-session"), "S-1-15-2-1")
    finally:
        journal.clear()
    assert not path.exists()
    recover_stale_grants(root)
    assert not path.exists()


def test_concurrent_sessions_do_not_share_an_identity(tmp_path):
    """Starting B must leave A running and A's workspace writable.

    Both sessions share one runtime root, which is what the old backend keyed
    everything on: one profile name and one journal. B's startup then recovered
    "stale" grants — A's — and revoked them, and closing A deleted the journal B
    would need after a crash. The measured failure was a PermissionError in A,
    not an exception anywhere in B.
    """
    from nooa.runtime.sandbox import win_api

    ws_a = tmp_path / "ws-a"
    ws_b = tmp_path / "ws-b"
    ws_a.mkdir()
    ws_b.mkdir()
    a = _Session(ws_a)
    b = None
    try:
        assert a.run("1").error is None
        name_a = a.executor._proc.container_name
        sid_a = win_api.sid_string(win_api.container_sid(name_a))
        assert sid_a in win_api.read_dacl(str(ws_a))

        b = _Session(ws_b)
        assert b.run("1").error is None, "the second session could not start"
        name_b = b.executor._proc.container_name
        assert name_b != name_a, "two live sessions share one container identity"
        assert win_api.sid_string(win_api.container_sid(name_b)) != sid_a

        # The measurement: B's startup must not have touched A's grant.
        assert sid_a in win_api.read_dacl(str(ws_a)), (
            "starting a second session revoked the first session's grant"
        )
        result = a.run("open('after-b.txt','w').write('ok')")
        assert result.error is None, result.error
        assert (ws_a / "after-b.txt").read_text(encoding="utf-8") == "ok"
    finally:
        a.close()
        if b is not None:
            b.close()


def test_closing_a_session_leaves_another_session_running(tmp_path):
    """A's teardown must not delete B's recovery record or revoke B's grants."""
    from nooa.runtime.sandbox import win_api
    from nooa.runtime.sandbox.win_runtime import resolve_root

    ws_a = tmp_path / "ws-a"
    ws_b = tmp_path / "ws-b"
    ws_a.mkdir()
    ws_b.mkdir()
    a = _Session(ws_a)
    b = None
    try:
        assert a.run("1").error is None
        b = _Session(ws_b)
        assert b.run("1").error is None
        name_b = b.executor._proc.container_name
        sid_b = win_api.sid_string(win_api.container_sid(name_b))
        journal_b = resolve_root(str(ROOT)).journal_for(name_b)
        assert journal_b.exists(), "B has no recovery record, so this test proves nothing"
    finally:
        a.close()
    try:
        assert journal_b.exists(), "closing A deleted B's recovery record"
        assert sid_b in win_api.read_dacl(str(ws_b)), "closing A revoked B's grant"
        result = b.run("2 + 2")
        assert result.error is None, result.error
        assert result.returned_value == 4
    finally:
        b.close()


def test_teardown_removes_the_session_profile(tmp_path):
    """A per-session profile must not outlive its session.

    The name is unique per start, so a leaked profile is a permanent entry in
    the user's package registry that nothing will ever reuse or clean up.
    ``DeriveAppContainerSidFromAppContainerName`` keeps resolving a deleted
    name (the SID is derived, not looked up), so the profile's folder under
    ``%LOCALAPPDATA%\\Packages`` is what tells the two apart.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        pytest.skip("LOCALAPPDATA is not set, so profiles cannot be located")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace)
    assert session.run("1").error is None
    name = session.executor._proc.container_name
    profile = Path(local_app_data) / "Packages" / name
    assert profile.exists(), "the profile was never created, so this test proves nothing"
    session.close()
    assert not profile.exists(), "the session's AppContainer profile survived teardown"


# --- teardown ----------------------------------------------------------------
def test_teardown_revokes_container_grants(tmp_path):
    """After close, the workspace DACL must no longer name the container SID.

    A leftover ACE is persistent — it survives the process — and would silently
    grant the next container more than it asked for.
    """
    from nooa.runtime.sandbox import win_api

    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = _Session(workspace)
    try:
        assert session.run("1").error is None
        # The session's identity is minted per start, so it is only knowable
        # from the process that was created under it.
        name = session.executor._proc.container_name
        sid_text = win_api.sid_string(win_api.container_sid(name))
        assert sid_text in win_api.read_dacl(str(workspace)), (
            "the grant was never written, so this test proves nothing"
        )
    finally:
        session.close()
    assert sid_text not in win_api.read_dacl(str(workspace)), "the container ACE survived teardown"
