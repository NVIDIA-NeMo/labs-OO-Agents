# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advanced: durable human-in-the-loop — suspend a run, resume it in a new process.

AGENTS.md prescribes that orchestrators are pure Python: a real method body calling
generation methods for each step. But a method's progress lives on the call stack,
and AgentSnapshot captures agent *attributes*, not the stack — so an orchestrator
that pauses for a human cannot be resumed once the process exits.

The fix needs no new machinery: keep the orchestrator's progress in a snapshotted
field instead of in locals, so re-entering it after a restore skips finished work.

This runs two steps, suspends for an operator value, snapshots, then re-launches
itself as a genuinely separate process which restores and finishes. The child
reports its own LLM call count — 1, not 3 — proving finished steps were not re-run.

    uv run python examples/advanced/durable_hitl.py
"""

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from nooa import Agent, strategy
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient
from nooa.unifiedllm.unifiedllm import LLMResponse


def scripted(*replies: str) -> FakeLLMClient:
    """A hermetic LLM: no API key, no network, and an exact call count to assert on."""
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content=r,
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": r},
            )
            for r in replies
        ]
    )


class SuspendForInput(Exception):
    """Raised by the orchestrator when only a human can supply the next value."""

    def __init__(self, field: str, question: str) -> None:
        self.field = field
        self.question = question
        super().__init__(question)


class PipelineAgent(Agent):
    """You review an operations report."""

    # The step ledger. Ordinary annotated fields, so AgentSnapshot captures them —
    # which is the whole reason the orchestrator below can be resumed.
    steps_done: dict[str, str]
    inputs: dict[str, str]

    @strategy(PredictStrategy())
    async def step_one(self, payload: str) -> str:
        """Summarise the payload in one short phrase."""
        ...

    @strategy(PredictStrategy())
    async def step_two(self, payload: str) -> str:
        """Name the single biggest risk in the payload, in one short phrase."""
        ...

    @strategy(PredictStrategy())
    async def step_three(self, payload: str, threshold: str) -> str:
        """Give a one-line verdict on the payload against the operator's threshold."""
        ...

    async def run(self, payload: str) -> str:
        """Pure-Python orchestrator: progress lives in self.steps_done, not on the stack."""
        if "step_one" not in self.steps_done:
            self.steps_done["step_one"] = await self.step_one(payload)
        if "step_two" not in self.steps_done:
            self.steps_done["step_two"] = await self.step_two(payload)

        # Nothing to compute here — only a human can set this. The orchestrator unwinds
        # and the host decides when, and in which process, to resume.
        if "threshold" not in self.inputs:
            raise SuspendForInput("threshold", "What risk threshold should apply?")

        if "step_three" not in self.steps_done:
            self.steps_done["step_three"] = await self.step_three(payload, self.inputs["threshold"])
        return self.steps_done["step_three"]


PAYLOAD = "A shipment of 400 units arrived two weeks late and 12 units are damaged."


async def first_leg(db_path: str) -> str:
    """Run until a human is needed, snapshot, and release the database."""
    llm = scripted('{"value": "late shipment, some damage"}', '{"value": "supplier reliability"}')
    storage = SQLiteStorageManager(db_path)
    try:
        agent = PipelineAgent(llm=llm, storage=storage)
        agent.steps_done = {}
        agent.inputs = {}
        try:
            await agent.run(PAYLOAD)
        except SuspendForInput as suspend:
            snapshot_id = storage.save_snapshot(agent)
            print(f"[parent] steps done      : {sorted(agent.steps_done)}")
            print(f"[parent] SUSPENDED       : {suspend.question}")
            print(f"[parent] LLM calls here  : {llm.call_count}")
            print(f"[parent] snapshot saved  : {snapshot_id[:8]}")
            return snapshot_id
        raise AssertionError("expected the orchestrator to suspend for operator input")
    finally:
        # SQLiteStorageManager holds a lock on db_path; the child process cannot open
        # it until we let go.
        storage.close()


async def second_leg(db_path: str, snapshot_id: str) -> None:
    """A different process: restore the ledger and finish the run."""
    llm = scripted('{"value": "Accept, with a supplier review."}')
    storage = SQLiteStorageManager(db_path)
    try:
        agent = PipelineAgent(llm=llm, storage=storage)  # fresh object, fresh process
        storage.restore_snapshot(snapshot_id, agent)
        print(f"[child]  restored        : {sorted(agent.steps_done)}")

        # A real host would read this from a form, a queue, or a CLI prompt.
        agent.inputs["threshold"] = "flag anything above 2% damage"
        print("[child]  operator supplied: threshold")

        verdict = await agent.run(PAYLOAD)
        print(f"[child]  verdict          : {verdict}")
        print(f"[child]  LLM calls here   : {llm.call_count}  <- not 3: finished steps skipped")
    finally:
        storage.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "durable_hitl.db")
        snapshot_id = asyncio.run(first_leg(db_path))
        print("[parent] re-launching as a separate process ...\n")
        # stdout is block-buffered when it is not a terminal, so without this the
        # child's output overtakes the parent's and the narrative reads backwards.
        sys.stdout.flush()
        subprocess.run([sys.executable, __file__, "--resume", snapshot_id, db_path], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        asyncio.run(second_leg(sys.argv[3], sys.argv[2]))
    else:
        main()
