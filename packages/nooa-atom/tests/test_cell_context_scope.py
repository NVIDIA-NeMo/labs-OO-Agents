# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cell context (Shell owner, cell-output folder) is scoped to its generation; owner close always closes the folder."""

import asyncio
from pathlib import Path

from atom_test_agents import python_cell
from nooa_atom.agent.agent import AtomCodeAct
from nooa_atom.agent.work_shell import Shell, ShellOwner, shell_owner

from nooa import Agent
from nooa.config.truncation_config import CaptureConfig, TruncationConfig
from nooa.decorators import strategy
from nooa.runtime.output_capture import capture_overflow
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _cell(code: str, i: int) -> LLMResponse:
    return python_cell(code, f"cell{i}")


class _Session(
    Agent,
    llm=FakeLLMClient(),
    truncation=TruncationConfig(capture=CaptureConfig(file_backed=True)),
):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(llm=FakeLLMClient(responses, strict_exhaustion=True))
        self._shells = ShellOwner()
        self.child: _Session | None = None
        self.seen: list[bool] = []

    def record_context(self) -> None:
        assert self.child is not None
        self.seen.append(capture_overflow.get() is self.child._shells.overflow_dir())

    @strategy(AtomCodeAct())
    async def work(self) -> str: ...


async def test_a_nested_child_generation_leaves_the_parents_cell_context():
    # A parent cell awaits a child agent's generation (same task); the parent's next
    # cell prints past the capture limit. Its file is the parent's, not the child's.
    child = _Session([_cell("return_result('child')", 1)])
    parent = _Session(
        [
            _cell("await self.child.work()", 2),
            _cell("self.record_context(); print('x' * 60000)", 3),
            _cell("return_result('parent')", 4),
        ]
    )
    parent.child = child
    outer = capture_overflow.get(), shell_owner.get()
    try:
        assert await parent.work() == "parent"
        assert parent.seen == [False]  # the second cell does not see the child's folder
        parent_dir, child_dir = parent._shells.overflow_dir(), child._shells.overflow_dir()
        assert parent_dir is not None and child_dir is not None
        saved = list(Path(parent_dir.path).glob("nooa-output-*"))
        assert len(saved) == 1 and saved[0].read_text() == "x" * 60000 + "\n"
        assert not list(Path(child_dir.path).glob("nooa-output-*"))
        await child._shells.aclose()
        assert saved[0].exists()  # the child's close keeps the parent's output
        assert (capture_overflow.get(), shell_owner.get()) == outer  # reset after the call
    finally:
        await parent._shells.aclose()
        await child._shells.aclose()
        await parent.aclose()
        await child.aclose()


async def test_a_cancelled_owner_close_still_closes_its_folder():
    owner = ShellOwner()
    token = shell_owner.set(owner)
    sh = Shell()
    try:
        result = await sh.run("head -c 1100000 /dev/zero")
    finally:
        shell_owner.reset(token)
    folder = owner.overflow_dir()
    assert folder is not None and result.stdout_path is not None
    file = Path(result.stdout_path)
    assert file.exists()
    entered, release = asyncio.Event(), asyncio.Event()
    session = sh._session
    assert session is not None
    original_close = session.close

    async def slow_close() -> None:  # hold the Shell's close at a real await
        entered.set()
        await release.wait()
        await original_close()

    session.close = slow_close  # type: ignore[method-assign]
    closer = asyncio.create_task(owner.aclose())
    await entered.wait()
    closer.cancel()
    try:
        await closer
    except asyncio.CancelledError:
        pass
    assert folder.closed and not file.exists()  # closed despite the cancellation
    release.set()
    assert sh._closing is not None
    await sh._closing
