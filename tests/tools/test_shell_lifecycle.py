# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped shell ownership leaves services alive until the harness cleans them up."""

import asyncio
import gc
import os
import shlex
import signal
import sys
import urllib.request
import weakref

import pytest

from nooa.tools._bash_session import BashSession
from nooa.tools.shell_lifecycle import preserve_background_services
from nooa.tools.shell_tools import ShellTools


@pytest.mark.asyncio
async def test_fresh_shell_server_survives_discard_and_scope_close(tmp_path):
    pid = None
    shell_ref = None

    async def start():
        nonlocal pid, shell_ref
        shell = ShellTools(cwd=str(tmp_path))
        shell_ref = weakref.ref(shell.session)
        assert shell.session._keep_background_on_close
        code = (
            "import http.server,os; "
            "s=http.server.HTTPServer(('127.0.0.1',0),http.server.SimpleHTTPRequestHandler); "
            "open('server.info','w').write(str(os.getpid())+' '+str(s.server_port)); "
            "s.serve_forever()"
        )
        _, _, status = await shell.session.run(
            f"{shlex.quote(sys.executable)} -u -c {shlex.quote(code)} >server.log 2>&1 &"
        )
        assert status == 0
        for _ in range(100):
            if (tmp_path / "server.info").exists():
                pid, port = map(int, (tmp_path / "server.info").read_text().split())
                return port
            await asyncio.sleep(0.02)
        raise AssertionError("Server did not start")

    try:
        async with preserve_background_services():
            port = await start()
            gc.collect()
            assert shell_ref() is not None  # Scope owns a discarded shell session.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            assert response.status == 200  # Grading happens after agent completion.
    finally:
        if pid is not None:
            os.kill(pid, signal.SIGKILL)  # Harness teardown after grading.
            for _ in range(100):
                try:
                    state = open(f"/proc/{pid}/stat").read().split()[2]
                    if state == "Z":
                        break
                except FileNotFoundError:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("Server survived harness teardown")


@pytest.mark.asyncio
async def test_scope_isolation_adoption_and_failure_cleanup(tmp_path):
    outside = BashSession(cwd=tmp_path)
    adopted = ShellTools(cwd=str(tmp_path))
    entered, release = asyncio.Event(), asyncio.Event()
    scoped = None

    async def invocation():
        nonlocal scoped
        with pytest.raises(ValueError, match="task failed"):
            async with preserve_background_services() as scope:
                scope.adopt(adopted.session)
                scoped = ShellTools(cwd=str(tmp_path), keep_background_on_close=False)
                await scoped.session.run("true")
                entered.set()
                await release.wait()
                raise ValueError("task failed")
        assert scoped.session._process is None

    running = asyncio.create_task(invocation())
    await entered.wait()
    assert not outside._keep_background_on_close
    assert not BashSession(cwd=tmp_path)._keep_background_on_close
    assert adopted.session._keep_background_on_close
    release.set()
    await running
    assert not BashSession(cwd=tmp_path)._keep_background_on_close


def test_preservation_destructor_does_not_kill_process_group(monkeypatch):
    from unittest.mock import Mock

    import nooa.tools._bash_session as module

    session = BashSession(keep_background_on_close=True)
    process = Mock(returncode=None)
    session._process = process
    killpg = Mock()
    drain = Mock()
    monkeypatch.setattr(module.os, "killpg", killpg)
    monkeypatch.setattr(module, "_spawn_output_drainer", drain)
    session.__del__()
    drain.assert_called_once_with(process)
    process.kill.assert_called_once()
    killpg.assert_not_called()
    session._process = None
