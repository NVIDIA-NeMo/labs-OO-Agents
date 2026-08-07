# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the sandboxed process executor.

These fork a real worker and exercise the full pipeline: persistent namespace,
``self.*`` tool brokering (sync + async), ``return_result`` signal marshaling,
the serialization boundary, and each guardrail enforced *inside a real cell*
(the paired leak is shown at the guard level in ``test_guards.py``).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from nooa import Agent
from nooa.runtime.sandbox.config import FileRule, SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.guards import probe_capabilities
from nooa.unifiedllm.fake import FakeLLMClient

pytestmark = pytest.mark.sandbox

CAPS = probe_capabilities()

_side_effects: list[str] = []


class _LiveSkill:
    """A deliberately non-picklable helper exposed as ``self.skill``."""

    def __init__(self) -> None:
        self._lock = __import__("threading").Lock()  # makes the instance unpicklable
        self.log: list[int] = []

    def remember(self, n: int) -> str:
        self.log.append(n)
        return f"stored:{n}"


class _LiveBag(dict):
    """A non-picklable dict-like exposed as ``self.bag`` (container protocol)."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = __import__("threading").Lock()  # unpicklable -> becomes a nested proxy


class _ToolAgent(Agent, llm=FakeLLMClient()):
    """Agent whose methods a sandboxed cell brokers back to the parent."""

    value: int = 41

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.skill = _LiveSkill()
        self.bag = _LiveBag()

    def add_one(self, n: int) -> int:
        return n + 1

    def record(self, tag: str) -> str:
        _side_effects.append(tag)
        return f"recorded:{tag}"

    async def async_double(self, n: int) -> int:
        return n * 2

    def finish(self, payload: int) -> None:
        # Mirrors ARC submit_actions: a tool that ends the turn by raising a signal.
        from nooa.strategies.codeact import _ReturnResultSignal

        raise _ReturnResultSignal(result={"result": payload})


def _return_result_builtins() -> dict:
    from nooa.strategies.codeact import _ReturnResultSignal

    def return_result(*args, **kwargs):
        if args:
            raise _ReturnResultSignal(result={"result": args[0]})
        raise _ReturnResultSignal(result=kwargs)

    return {"return_result": return_result}


def _executor(config: SandboxConfig | None = None, *, cell_timeout: float | None = 10.0):
    agent = _ToolAgent()
    return SandboxedExecutor(
        agent,
        config or SandboxConfig(require=False),
        cell_timeout=cell_timeout,
        framework_builtins=_return_result_builtins(),
    )


async def _run(ex: SandboxedExecutor, code: str, n: int = 1):
    return await ex.run_cell(code, execution_count=n)


# --- semantic parity --------------------------------------------------------
async def test_trivial_cell_returns_value():
    ex = _executor()
    try:
        res = await _run(ex, "1 + 1")
        assert res.success
        assert res.returned_value == 2
    finally:
        await ex.aclose()


async def test_persistent_namespace_across_cells():
    ex = _executor()
    try:
        r1 = await _run(ex, "def square(x):\n    return x * x", 1)
        assert r1.success
        r2 = await _run(ex, "square(7)", 2)
        assert r2.returned_value == 49
    finally:
        await ex.aclose()


async def test_persistent_namespace_keeps_assignments_from_failed_cell():
    ex = _executor()
    try:
        failed = await _run(ex, "answer = 41\nraise ValueError('boom')", 1)
        assert not failed.success
        assert isinstance(failed.error, ValueError)

        resumed = await _run(ex, "answer + 1", 2)
        assert resumed.success, resumed.error
        assert resumed.returned_value == 42
    finally:
        await ex.aclose()


async def test_persistent_namespace_keeps_assignments_before_return_result():
    ex = _executor()
    try:
        signaled = await _run(ex, "answer = 41\nreturn_result(answer)", 1)
        assert signaled.error is None
        assert signaled.signal is not None

        resumed = await _run(ex, "answer + 1", 2)
        assert resumed.success, resumed.error
        assert resumed.returned_value == 42
    finally:
        await ex.aclose()


async def test_stdout_is_captured():
    ex = _executor()
    try:
        res = await _run(ex, "print('hello from cell')")
        assert "hello from cell" in res.stdout
    finally:
        await ex.aclose()


async def test_sync_tool_brokering_hits_live_agent():
    ex = _executor()
    try:
        res = await _run(ex, "self.add_one(41)")
        assert res.returned_value == 42
    finally:
        await ex.aclose()


async def test_async_tool_brokering():
    ex = _executor()
    try:
        res = await _run(ex, "await self.async_double(21)")
        assert res.returned_value == 42
    finally:
        await ex.aclose()


async def test_tool_side_effect_lands_on_parent_agent():
    ex = _executor()
    try:
        _side_effects.clear()
        res = await _run(ex, "self.record('from-cell')")
        assert res.returned_value == "recorded:from-cell"
        # The side effect ran on the PARENT's live agent, not the forked copy.
        assert "from-cell" in _side_effects
    finally:
        await ex.aclose()


async def test_attribute_brokering_reads_live_agent():
    ex = _executor()
    try:
        res = await _run(ex, "self.value")
        assert res.returned_value == 41
    finally:
        await ex.aclose()


async def test_attribute_assignment_brokers_to_live_agent():
    """self.attr = value in a cell must land on the PARENT agent, not the proxy."""
    agent = _ToolAgent()
    ex = SandboxedExecutor(
        agent,
        SandboxConfig(require=False),
        cell_timeout=10.0,
        framework_builtins=_return_result_builtins(),
    )
    try:
        r1 = await _run(ex, "self.value = 100", 1)
        assert r1.success, r1.error
        assert agent.value == 100  # landed on the live parent agent
        # And a later cell reads the updated value back (no stale proxy dict).
        r2 = await _run(ex, "self.value", 2)
        assert r2.returned_value == 100
    finally:
        await ex.aclose()


async def test_unpicklable_return_result_payload_is_clean_error():
    """A non-picklable return_result(...) yields a clear error, not a worker crash."""
    ex = _executor()
    try:
        res = await _run(ex, "return_result(lambda: 1)")
        assert not res.success
        assert "picklable" in str(res.error).lower()
        # Worker survived (namespace intact): next cell still runs.
        ok = await _run(ex, "1 + 1", 2)
        assert ok.returned_value == 2
    finally:
        await ex.aclose()


async def test_nested_proxy_brokers_method_on_live_attribute():
    """self.skill is non-picklable, but self.skill.remember(...) still brokers to
    the parent's live object (nested proxying)."""
    ex = _executor()
    try:
        res = await _run(ex, "self.skill.remember(7)")
        assert res.success, res.error
        assert res.returned_value == "stored:7"
    finally:
        await ex.aclose()


async def test_return_result_signal_crosses_boundary():
    ex = _executor()
    try:
        res = await _run(ex, "return_result(123)")
        assert res.error is None
        assert res.signal is not None
        assert res.signal.result == {"result": 123}
    finally:
        await ex.aclose()


async def test_brokered_tool_raising_signal_flows_as_signal():
    """A brokered tool that raises return_result (ARC submit_actions pattern) must
    surface as ExecutionResult.signal, not crash the executor."""
    from nooa.strategies.codeact import _ReturnResultSignal

    ex = _executor()
    try:
        res = await _run(ex, "self.finish(99)")
        assert res.error is None
        assert isinstance(res.signal, _ReturnResultSignal)
        assert res.signal.result == {"result": 99}
    finally:
        await ex.aclose()


async def test_unpicklable_return_is_clear_error():
    ex = _executor()
    try:
        res = await _run(ex, "lambda x: x")
        assert not res.success
        assert "picklable" in str(res.error).lower()
    finally:
        await ex.aclose()


async def test_cell_cannot_reach_broker_pipe():
    """The broker/pipe must be unreachable from cell code (no self._broker escape)."""
    ex = _executor()
    try:
        res = await _run(ex, "self._broker")
        assert not res.success
        assert "AttributeError" in str(res.error) or "_broker" in str(res.error)
        # A follow-up normal cell still works (proxy is otherwise healthy).
        ok = await _run(ex, "self.add_one(1)", 2)
        assert ok.returned_value == 2
    finally:
        await ex.aclose()


async def test_slow_broker_call_respects_broker_timeout():
    """A stuck brokered self.* call is bounded by broker_timeout_s, not run open-ended.

    (Brokered parent-side time no longer consumes the CELL deadline — killing the
    worker for parent latency wiped REPL state in the ARC fleet — but a stuck
    ``self.*`` still must not hang the run: it gets its own generous bound.)
    """

    class _SlowAgent(Agent, llm=FakeLLMClient()):
        async def slow(self) -> int:
            import asyncio as _a

            await _a.sleep(30)
            return 1

    ex = SandboxedExecutor(
        _SlowAgent(),
        SandboxConfig(
            require=False,
            network=True,
            filesystem=False,
            timeout_grace_s=1.0,
            broker_timeout_s=1.0,
        ),
        cell_timeout=1.0,
        framework_builtins=_return_result_builtins(),
    )
    try:
        res = await _run(ex, "await self.slow()")
        assert not res.success
        assert "broker_timeout" in str(res.error).lower() or "kill" in str(res.error).lower()
        # Worker recovered.
        ok = await _run(ex, "1 + 1", 2)
        assert ok.returned_value == 2
    finally:
        await ex.aclose()


async def test_cell_error_is_reported_not_raised():
    ex = _executor()
    try:
        res = await _run(ex, "raise ValueError('boom')")
        assert not res.success
        assert "boom" in str(res.error)
        # The real exception type is reconstructed (not a generic surrogate).
        assert isinstance(res.error, ValueError)
    finally:
        await ex.aclose()


async def test_nested_proxy_container_protocol():
    """self.bag["k"] = v / self.bag["k"] on a non-picklable dict-like brokers to parent."""
    agent = _ToolAgent()
    ex = SandboxedExecutor(
        agent,
        SandboxConfig(require=False),
        cell_timeout=10.0,
        framework_builtins=_return_result_builtins(),
    )
    try:
        r1 = await _run(ex, "self.bag['plan'] = 'do-x'", 1)
        assert r1.success, r1.error
        assert agent.bag["plan"] == "do-x"  # landed on the live parent object
        r2 = await _run(ex, "self.bag['plan']", 2)
        assert r2.returned_value == "do-x"
        r3 = await _run(ex, "'plan' in self.bag", 3)
        assert r3.returned_value is True
    finally:
        await ex.aclose()


async def test_recovery_disabled_stops_after_kill():
    ex = _executor(
        SandboxConfig(
            require=False, network=True, filesystem=False, timeout_grace_s=1.0, recovery="disabled"
        ),
        cell_timeout=1.0,
    )
    try:
        killed = await _run(ex, "while True:\n    pass", 1)
        assert not killed.success
        # recovery='disabled' -> the worker is not resurrected; the next cell fails.
        after = await _run(ex, "1 + 1", 2)
        assert not after.success
        assert "disabled" in str(after.error).lower()
    finally:
        await ex.aclose()


async def test_workspace_is_created_if_missing():
    with tempfile.TemporaryDirectory() as base:
        ws = os.path.join(base, "created", "here")
        assert not os.path.exists(ws)
        ex = SandboxedExecutor(
            _ToolAgent(),
            SandboxConfig(require=False, network=True, workspace=ws),
            cell_timeout=10.0,
            framework_builtins=_return_result_builtins(),
        )
        try:
            assert os.path.isdir(ws)  # created at executor construction
            res = await _run(ex, f"open({os.path.join(ws, 'f.txt')!r}, 'w').write('ok')")
            assert res.success, res.error
        finally:
            await ex.aclose()


# --- guardrails enforced inside a real cell ---------------------------------
@pytest.mark.skipif(not CAPS.rlimit, reason="RLIMIT unavailable")
async def test_memory_guard_inside_cell():
    ex = _executor(SandboxConfig(require=False, network=True, filesystem=False, max_memory_mb=128))
    try:
        res = await _run(ex, "x = bytearray(1024 * 1024 * 1024)")
        assert not res.success
        assert "memory" in str(res.error).lower()
    finally:
        await ex.aclose()


@pytest.mark.skipif(not CAPS.seccomp, reason="seccomp unavailable")
async def test_network_guard_inside_cell():
    ex = _executor(SandboxConfig(require=False, network=False, filesystem=False))
    try:
        res = await _run(
            ex,
            "import socket\nsocket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        )
        assert not res.success
        assert "PermissionError" in str(res.error) or "permission" in str(res.error).lower()
    finally:
        await ex.aclose()


@pytest.mark.skipif(CAPS.landlock_abi < 1, reason="Landlock unavailable")
async def test_filesystem_guard_inside_cell():
    with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as secret:
        secret_path = os.path.join(secret, "s.txt")
        with open(secret_path, "w") as fh:
            fh.write("TOPSECRET")
        ex = _executor(
            SandboxConfig(
                require=False,
                network=True,
                filesystem=True,
                workspace=ws,
                allow=(FileRule(path=secret, access="read"),),
            )
        )
        try:
            # Workspace write allowed.
            ok = await _run(ex, f"open({os.path.join(ws, 'a.txt')!r}, 'w').write('x')", 1)
            assert ok.success, ok.error
        finally:
            await ex.aclose()

        ex2 = _executor(SandboxConfig(require=False, network=True, workspace=ws))
        try:
            res = await _run(ex2, f"open({secret_path!r}).read()")
            assert not res.success
            assert "PermissionError" in str(res.error) or "permission" in str(res.error).lower()
        finally:
            await ex2.aclose()


@pytest.mark.skipif(not CAPS.rlimit, reason="RLIMIT unavailable")
async def test_cpu_guard_kills_spin_loop():
    ex = _executor(
        SandboxConfig(require=False, network=True, filesystem=False, max_cpu_seconds=1),
        cell_timeout=30.0,
    )
    try:
        res = await _run(ex, "while True:\n    pass")
        assert not res.success
        assert "cpu" in str(res.error).lower() or "kill" in str(res.error).lower()
    finally:
        await ex.aclose()


async def test_wallclock_timeout_kills_cpu_bound_cell():
    ex = _executor(
        SandboxConfig(require=False, network=True, filesystem=False, timeout_grace_s=1.0),
        cell_timeout=1.0,
    )
    try:
        res = await _run(ex, "while True:\n    pass")
        assert not res.success
        assert "deadline" in str(res.error).lower() or "kill" in str(res.error).lower()
        # Worker recovered: the next cell runs.
        res2 = await _run(ex, "40 + 2", 2)
        assert res2.returned_value == 42
    finally:
        await ex.aclose()


async def test_require_true_raises_when_unavailable(monkeypatch):
    from nooa.runtime.sandbox import executor as ex_mod
    from nooa.runtime.sandbox.guards import Capabilities

    fake = Capabilities(linux=True, landlock_abi=0, seccomp=False, rlimit=True)
    monkeypatch.setattr(ex_mod, "_CAPS_CACHE", fake)
    from nooa.runtime.sandbox.errors import SandboxUnavailable

    with pytest.raises(SandboxUnavailable):
        SandboxedExecutor(
            _ToolAgent(),
            SandboxConfig(require=True, filesystem=True, network=False),
            cell_timeout=5.0,
        )
    ex_mod._CAPS_CACHE = None


# ---------------------------------------------------------------------------
# Regression tests: Windows signal guard in _classify_worker_death (issue #92)
# ---------------------------------------------------------------------------


def test_classify_worker_death_no_sigxcpu_no_sigkill(monkeypatch):
    """Regression test for #92.

    On platforms where signal.SIGXCPU and signal.SIGKILL don't exist (e.g.
    Windows), _classify_worker_death() must not raise AttributeError.  It
    should fall through and return the original WorkerDiedError unchanged.
    """
    import signal as signal_mod

    from nooa.runtime.sandbox import executor as ex_mod
    from nooa.runtime.sandbox.errors import WorkerDiedError
    from nooa.runtime.sandbox.guards import Capabilities

    # Inject a degraded-mode Windows-like Capabilities so __init__ doesn't crash
    fake_caps = Capabilities(linux=False, landlock_abi=0, seccomp=False, rlimit=False)
    monkeypatch.setattr(ex_mod, "_CAPS_CACHE", fake_caps)

    # Mock get_context so instantiating SandboxedExecutor doesn't fail on Windows due to 'fork'
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_context", lambda method: None)

    # Simulate Windows: remove POSIX-only signal constants
    monkeypatch.delattr(signal_mod, "SIGXCPU", raising=False)
    monkeypatch.delattr(signal_mod, "SIGKILL", raising=False)

    executor = SandboxedExecutor(
        _ToolAgent(),
        SandboxConfig(require=False),
        cell_timeout=None,
    )

    # Simulate a dead worker with a generic exit code (e.g. 1)
    class _FakeProc:
        exitcode = 1

    executor._proc = _FakeProc()
    exc = WorkerDiedError("worker died unexpectedly")

    # Before fix: raised AttributeError: module 'signal' has no attribute 'SIGXCPU'
    # After fix:  falls through and returns the original exception unchanged
    result = executor._classify_worker_death(exc)
    assert result is exc

    ex_mod._CAPS_CACHE = None


def test_classify_worker_death_sigxcpu_present(monkeypatch):
    """On Linux where signal.SIGXCPU exists, a SIGXCPU exit should map to CellTimeoutError."""
    import signal as signal_mod

    from nooa.runtime.sandbox import executor as ex_mod
    from nooa.runtime.sandbox.errors import CellTimeoutError, WorkerDiedError
    from nooa.runtime.sandbox.guards import Capabilities

    if not hasattr(signal_mod, "SIGXCPU"):
        pytest.skip("SIGXCPU not available on this platform")

    fake_caps = Capabilities(linux=False, landlock_abi=0, seccomp=False, rlimit=False)
    monkeypatch.setattr(ex_mod, "_CAPS_CACHE", fake_caps)

    executor = SandboxedExecutor(
        _ToolAgent(),
        SandboxConfig(require=False),
        cell_timeout=None,
    )

    class _FakeProc:
        exitcode = -signal_mod.SIGXCPU

    executor._proc = _FakeProc()
    result = executor._classify_worker_death(WorkerDiedError("cpu killed"))
    assert isinstance(result, CellTimeoutError)

    ex_mod._CAPS_CACHE = None
