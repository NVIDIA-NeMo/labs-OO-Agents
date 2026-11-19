# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell's capture with several writers, a forked child and no temporary directory.

Each test replays a reviewer's reproducer (#580 review r11, correctness 1-3)
through ``Agent.runtime.execute_code``: a cell's file-backed capture outside
a session, in ``OverflowDir.default()``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import pytest

import nooa.agent
from nooa import Agent
from nooa.config.truncation_config import CaptureConfig, TruncationConfig
from nooa.overflow import OverflowDir
from nooa.unifiedllm import FakeLLMClient

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


@pytest.fixture(autouse=True)
def _no_auto_tracing(monkeypatch):
    monkeypatch.setattr(nooa.agent, "_auto_tracing_attempted", True)


@pytest.fixture
def fresh_default(monkeypatch):
    """No process-default overflow folder yet; one made in the test is closed after it."""
    from nooa.overflow import _overflow

    monkeypatch.setattr(_overflow, "_default", None, raising=False)
    yield
    made = getattr(_overflow, "_default", None)
    if made is not None:
        made.close()


def _run(code: str, tmp_path: Path) -> list[str]:
    """Run ``code`` in a fresh interpreter with ``TMPDIR=tmp_path``; its stdout lines."""
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def _agent(**capture: object) -> Agent:
    return Agent(
        llm=FakeLLMClient([]),
        truncation=TruncationConfig(capture=CaptureConfig(**capture)),
    )


async def test_a_write_while_the_file_opens_still_reaches_the_file(
    tmp_path, monkeypatch, fresh_default
):
    # capture_threads_runtime.py: a thread's write opens the overflow file; while
    # the folder creates it, the cell writes 100 more chars. Unlocked, the opening
    # write copied the already-cut window: a file of 110 of 204 bytes, called full.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    entered, resume = threading.Event(), threading.Event()
    real_open = OverflowDir.open

    def gated_open(self, max_bytes=None, *, warn=None):
        writer = real_open(self, max_bytes=max_bytes, warn=warn)
        entered.set()
        resume.wait(timeout=5)
        return writer

    monkeypatch.setattr(OverflowDir, "open", gated_open)

    def emit(text):
        return sys.stdout.write(text)

    def wait_open():
        # Only the first write's opening pauses; the other writer goes on.
        assert entered.wait(timeout=5)
        threading.Timer(0.5, resume.set).start()

    agent = _agent(max_stdout=10, file_backed=True)
    result = await agent.runtime.execute_code(
        "emit('seed')\n"
        "task = asyncio.create_task(asyncio.to_thread(emit, 'A' * 100))\n"
        "await asyncio.to_thread(wait_open)\n"
        "emit('B' * 100)\n"
        "await task\n",
        builtins={"emit": emit, "wait_open": wait_open},
        wrap_in_function=True,
        validate=False,
    )
    assert result.success, result.stderr
    match = re.search(r"full output: (.+)]", result.stdout)
    assert match, result.stdout
    contents = Path(match.group(1)).read_text()
    assert len(contents) == 204 and contents.startswith("seed")
    assert sorted(contents[4:]) == sorted("A" * 100 + "B" * 100)


def test_no_temporary_directory_still_runs_the_cell_with_memory_only(tmp_path):
    # capture_folder_failure_runtime.py: the process folder could not be created,
    # and its OSError aborted the cell before it ran. A fresh interpreter: no
    # process folder exists yet.
    code = f"""
        import asyncio, json, tempfile
        import nooa.agent
        from nooa import Agent
        from nooa.config.truncation_config import CaptureConfig, TruncationConfig
        from nooa.unifiedllm import FakeLLMClient

        nooa.agent._auto_tracing_attempted = True
        tempfile.tempdir = {str(tmp_path / "missing")!r}
        config = TruncationConfig(capture=CaptureConfig(max_stdout=10, file_backed=True))
        ran = []
        agent = Agent(llm=FakeLLMClient([]), truncation=config)
        result = asyncio.run(agent.runtime.execute_code(
            "mark(); print('0123456789' * 10, end='')",
            builtins={{"mark": lambda: ran.append(True)}}, validate=False))
        print(json.dumps([bool(ran), result.success, result.stdout]))
    """
    assert json.loads(_run(code, tmp_path)[-1]) == [
        True,
        True,
        "01234\n[... 90 chars omitted; full output not kept]\n56789",
    ]


@needs_fork
def test_a_forked_child_gets_its_own_capture_folder(tmp_path):
    # capture_fork.py: the child inherited the parent's (inert) process folder,
    # so a new capture in the child kept no file.
    code = """
        import asyncio, json, os, re
        import nooa.agent
        from nooa import Agent
        from nooa.config.truncation_config import CaptureConfig, TruncationConfig
        from nooa.unifiedllm import FakeLLMClient

        nooa.agent._auto_tracing_attempted = True
        config = TruncationConfig(capture=CaptureConfig(max_stdout=10, file_backed=True))

        def run(text):
            agent = Agent(llm=FakeLLMClient([]), truncation=config)
            result = asyncio.run(agent.runtime.execute_code(
                f"print({text!r} * 100, end='')", validate=False))
            match = re.search(r"full output: (.+)]", result.stdout)
            return match.group(1) if match else None

        parent = run("p")
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            path = run("c")
            content = open(path).read() if path else None
            os.write(write, json.dumps([path, content]).encode())
            os._exit(0)
        os.close(write)
        os.waitpid(pid, 0)
        child_path, child_content = json.loads(os.read(read, 65536))
        print(json.dumps({
            "child_kept": child_content == "c" * 100,
            "own_folder": child_path is not None
            and os.path.dirname(child_path) != os.path.dirname(parent),
            "parent_intact": open(parent).read() == "p" * 100,
        }))
    """
    assert json.loads(_run(code, tmp_path)[-1]) == {
        "child_kept": True,
        "own_folder": True,
        "parent_intact": True,
    }
