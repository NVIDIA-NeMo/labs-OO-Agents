# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TruncationConfig.overflow_files=False``: no output file anywhere; and handed-out files are given back."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent import work_shell
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.work_shell import Shell, ShellOwner, shell_owner

from nooa.config.truncation_config import TruncationConfig
from nooa.interactive import Done

_BIG = "head -c 1100000 /dev/zero | tr '\\0' a; head -c 1100000 /dev/zero | tr '\\0' b >&2"

_WORK = f"""
import json
from pathlib import Path

async def in_a_job():
    async with Shell(max_output=1000) as sh:
        r = await sh.run({_BIG!r})
    return [r.stdout, r.stderr, r.stdout_path, r.stderr_path]

job = await self.jobs.launch(in_a_job(), name="files")
r = await Shell().run({_BIG!r})
direct = [r.stdout, r.stderr, r.stdout_path, r.stderr_path]
Path(OUT).write_text(json.dumps({{"job": job, "direct": direct}}))
"""


def _output_files(root: Path) -> list[Path]:
    return sorted([*root.rglob("nooa-overflow-*"), *root.rglob("nooa-output-*")])


async def test_switch_off_writes_no_file_for_cells_shell_runs_or_jobs(tmp_path, monkeypatch):
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))  # every temporary file would land here
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    out = workspace / "results.json"
    llm = CellLLM(
        [
            cell("print('h' * 30_000 + 'm' * 10_000 + 't' * 30_000)"),
            cell(f"OUT = {str(out)!r}\n" + _WORK),
            cell("return_result(Done(explanation='ok'))"),
        ]
    )
    agent = AtomAgent(cwd=workspace, llm=llm, truncation=TruncationConfig(overflow_files=False))
    try:
        assert agent._truncation.capture.file_backed  # Atom's cells would write a file
        assert agent._shells.overflow_dir() is None
        assert await asyncio.wait_for(agent.handle({"user_messages": ["go"]}), 120) == Done(
            explanation="ok"
        )
        seen = str(llm.calls[1].messages)
        assert "[... 20,001 chars omitted; full output not kept]" in seen, seen[-3000:]
        assert "full output: " not in seen
        results = json.loads(out.read_text())
        for stdout, stderr, stdout_path, stderr_path in (results["job"], results["direct"]):
            assert stdout_path is None and stderr_path is None
            assert "bytes omitted; full output not kept]" in stdout
            assert "bytes omitted; full output not kept]" in stderr
            assert stdout.startswith("a") and stderr.startswith("b")
        assert _output_files(temp) == [] and _output_files(workspace) == []
    finally:
        await agent.aclose()
    assert _output_files(temp) == []


async def test_a_shell_under_a_switched_off_owner_keeps_memory_only(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    owner = ShellOwner(overflow_files=False)
    token = shell_owner.set(owner)
    try:
        async with Shell(cwd=tmp_path) as sh:
            r = await sh.run(_BIG)
    finally:
        shell_owner.reset(token)
        await owner.aclose()
    assert r.stdout_path is None and r.stderr_path is None
    assert "bytes omitted; full output not kept]" in r.stdout
    assert _output_files(tmp_path) == []


async def test_a_failed_stdin_file_leaves_no_handed_out_file(tmp_path, monkeypatch):
    # model.md C2 (work_shell.py:723): the captures were made before the stdin
    # file, outside the try that closes them, so a failing os.open left both
    # files handed out and unopened: two of the agent's 20 places, for good.
    real_open = os.open

    def failing_open(path, *args, **kwargs):
        if str(path).endswith(".stdin"):
            raise OSError(28, "No space left on device")
        return real_open(path, *args, **kwargs)

    owner = ShellOwner()
    token = shell_owner.set(owner)
    try:
        async with Shell(cwd=tmp_path) as sh:
            await sh.run("true")  # bash is up
            folder = owner.overflow_dir()
            assert folder is not None
            monkeypatch.setattr(work_shell.os, "open", failing_open)
            with pytest.raises(OSError, match="No space"):
                await sh.run("cat", stdin="x")
            monkeypatch.setattr(work_shell.os, "open", real_open)
            assert sorted(os.listdir(folder.path)) == []  # no file left handed out
    finally:
        shell_owner.reset(token)
        await owner.aclose()


async def test_a_capture_that_forgets_the_switch_writes_no_file_under_a_switched_off_agent(
    tmp_path, monkeypatch
):
    # Round H item 1: a new entry point that builds a file-backed capture
    # without clearing file_backed still writes no file: the agent's cell
    # context says the session's files are off, and the buffer honours it.
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    out = workspace / "capture.json"
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from nooa.runtime.output_capture import CaptureBuffer\n"
        "buffer = CaptureBuffer(10, file_backed=True)\n"
        "buffer.write('z' * 100)\n"
        f"Path({str(out)!r}).write_text(json.dumps([buffer.file_path, buffer.getvalue()]))\n"
        "buffer.close()\n"
    )
    llm = CellLLM([cell(code), cell("return_result(Done(explanation='ok'))")])
    agent = AtomAgent(cwd=workspace, llm=llm, truncation=TruncationConfig(overflow_files=False))
    try:
        assert await asyncio.wait_for(agent.handle({"user_messages": ["go"]}), 120) == Done(
            explanation="ok"
        )
        path, rendered = json.loads(out.read_text())
        assert path is None
        assert rendered == "zzzzz\n[... 90 chars omitted; full output not kept]\nzzzzz"
        assert _output_files(temp) == [] and _output_files(workspace) == []
    finally:
        await agent.aclose()
    assert _output_files(temp) == []
