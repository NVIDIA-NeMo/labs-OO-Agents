# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Advanced: durable human-in-the-loop — suspend a run, resume it in a new process.

Orchestrators are pure Python, but a method's progress lives on the call stack and
`AgentSnapshot` captures agent *attributes*, not the stack. So a `run()` that pauses
for a human cannot be picked up again after the process exits.

Keeping progress in a snapshotted field instead of in locals fixes that: re-entering
the orchestrator after `restore_snapshot()` skips whatever is already done.

Watch the "running ..." lines. The child process prints only the step it actually
had to perform — the earlier one came back from the snapshot.

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
        storage.close()  # the child cannot open the database until we let go


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
