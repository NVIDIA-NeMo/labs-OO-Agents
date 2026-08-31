# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Advanced: durable workflow resumption — pause a run and continue it later.

Most workflows run to completion in one process. This example covers the cases where
a workflow must stop and continue later: a worker restart, an external dependency,
a scheduled handoff, or a human approval.

Use a storage manager to save the agent's state before stopping. Later, create a
fresh agent, restore the snapshot, provide any newly available input, and call the
workflow again. Completed steps are kept in agent fields, so the workflow skips them
and continues with the next unfinished step.

Here, a human threshold is the reason for the pause. The example resumes in a
separate Python process to demonstrate that the saved progress survives a process
boundary. Watch the "running ..." lines: the child runs only the remaining step.

    uv run python examples/advanced/durable_hitl.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from nooa.storage.sqlite import SQLiteStorageManager
from nooa.util.quickstart import *

PAYLOAD = "A shipment of 400 units arrived two weeks late and 12 units are damaged."


class SuspendForInput(Exception):
    """Raised when only a human can supply the next value."""


class PipelineAgent(Agent, llm=llm):
    """You review an operations report."""

    steps_done: dict[str, str]  # snapshotted — this is what makes run() resumable
    inputs: dict[str, str]

    @strategy(PredictStrategy())
    async def summarise(self, payload: str) -> str:
        """Summarise the report in one short phrase."""
        ...

    @strategy(PredictStrategy())
    async def verdict(self, payload: str, threshold: str) -> str:
        """Give a one-line verdict on the report against the operator's threshold."""
        ...

    async def run(self, payload: str) -> str:
        if "summary" not in self.steps_done:
            print("  running summarise")
            self.steps_done["summary"] = await self.summarise(payload)

        if "threshold" not in self.inputs:
            raise SuspendForInput("what risk threshold should apply?")

        if "verdict" not in self.steps_done:
            print("  running verdict")
            self.steps_done["verdict"] = await self.verdict(payload, self.inputs["threshold"])
        return self.steps_done["verdict"]


async def until_a_human_is_needed(db: str) -> str:
    storage = SQLiteStorageManager(db)
    try:
        agent = PipelineAgent(storage=storage)
        agent.steps_done, agent.inputs = {}, {}
        try:
            await agent.run(PAYLOAD)
        except SuspendForInput as needed:
            print(f"[parent] suspended: {needed}")
            return storage.save_snapshot(agent)
        raise AssertionError("expected the orchestrator to suspend")
    finally:
        storage.close()  # release the lock, or the child hits SessionAlreadyActiveError


async def resume_and_finish(db: str, snapshot_id: str) -> None:
    storage = SQLiteStorageManager(db)
    try:
        agent = PipelineAgent(storage=storage)  # fresh object, fresh process
        storage.restore_snapshot(snapshot_id, agent)
        print(f"[child]  restored: {sorted(agent.steps_done)}")

        # A real host would read this from a form, a queue, or a prompt.
        agent.inputs["threshold"] = "flag anything above 2% damage"

        print(f"[child]  verdict : {await agent.run(PAYLOAD)}")
    finally:
        storage.close()


if __name__ == "__main__":
    if "--resume" in sys.argv:
        asyncio.run(resume_and_finish(sys.argv[3], sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "durable.db")
            snapshot_id = asyncio.run(until_a_human_is_needed(db))
            print("[parent] re-launching as a separate process ...\n")
            sys.stdout.flush()  # otherwise the child's output overtakes the parent's
            subprocess.run([sys.executable, __file__, "--resume", snapshot_id, db], check=True)
