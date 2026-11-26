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
            cell("import sys\nawait asyncio.to_thread(sys.stdout.write, 'r' * 80_000)"),
            cell("raise ValueError('e' * 100_000)"),
            cell(f"OUT = {str(out)!r}\n" + _WORK),
            cell("return_result(Done(explanation='ok'))"),
        ]
    )
    agent = AtomAgent(cwd=workspace, llm=llm, truncation=TruncationConfig(overflow_files=False))
    assert _output_files(temp) == []
    try:
        assert agent._truncation.capture.file_backed  # Atom's cells would write a file
        assert agent.runtime.truncation_config.overflow_files is False
        assert agent._shells.overflow_dir() is None
        assert await asyncio.wait_for(agent.handle({"user_messages": ["go"]}), 120) == Done(
            explanation="ok"
        )
        seen = str(llm.calls[1].messages)  # the cell's print
        assert "[... 20,001 chars omitted; full output not kept]" in seen, seen[-3000:]
        seen = str(llm.calls[2].messages)  # a thread's print
        assert "[... 30,000 chars omitted; full output not kept]" in seen, seen[-3000:]
        seen = str(llm.calls[3].messages)  # a large error
        assert "chars omitted; full output not kept]" in seen and "ValueError" in seen
        assert "full output: " not in str(llm.calls[-1].messages)
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


async def test_a_method_level_switch_off_is_read_at_capture_time(tmp_path, monkeypatch):
    # #580 review: the owner copied the agent's overflow_files once, at session
    # start, so a method-level ``@strategy(truncation=...)`` was ignored. The
    # owner now reads ``effective(TruncationConfig, agent)`` at each capture;
    # the runtime sets the method's merged config in this context variable.
    from nooa.config.effective import _current_truncation_config_var

    class _Agent:  # what the owner reads: the agent's own truncation config
        _truncation = TruncationConfig()  # spilling on

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    agent = _Agent()
    owner = ShellOwner(agent=agent)
    token = shell_owner.set(owner)
    try:
        async with Shell(cwd=tmp_path) as sh:
            method = agent._truncation.merge_with(TruncationConfig(overflow_files=False))
            override = _current_truncation_config_var.set(method)
            try:
                off = await sh.run(_BIG)
                assert owner.overflow_dir() is None
            finally:
                _current_truncation_config_var.reset(override)
            assert off.stdout_path is None and off.stderr_path is None
            assert "bytes omitted; full output not kept]" in off.stdout
            assert _output_files(tmp_path) == []
            on = await sh.run(_BIG)  # outside the method: the agent's own config
            assert on.stdout_path is not None and Path(on.stdout_path).is_file()
    finally:
        shell_owner.reset(token)
        await owner.aclose()


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
