# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 08: Context blocks — pin live state into the LLM's system prompt.

uv run python examples/quickstart/08_context_blocks.py
"""

from nooa.agentdoc import spec
from nooa.util.quickstart import *


class NoteTakingAgent(Agent, llm=llm):
    """Agent that stores notes and answers questions about them."""

    def __init__(self):
        super().__init__()
        self._notes: list[str] = []
        spec(self, "context", hidden=False)  # Expose context management to LLM

    # SW1: deterministic helpers the LLM (and you) can call
    def add_note(self, text: str) -> None:
        """Add a note to the collection."""
        self._notes.append(text)

    def render_notes(self) -> str:
        """Render all stored notes as a formatted list."""
        return "\n".join(f"- {n}" for n in self._notes) or "No notes yet."

    # SW3: generation methods
    async def record(self, note: str) -> str:
        """Store this note using add_note and confirm what was saved."""
        ...

    async def answer(self, question: str) -> str:
        """Answer the question using the notes visible in your context."""
        ...


@autorun
async def main():
    agent = NoteTakingAgent()

    # Dynamic block: the expression is re-evaluated every LLM turn,
    # so the LLM always sees the latest notes without re-passing them
    from nooa import Context

    agent.context["notes"] = Context(expr="self.render_notes()")

    notes = [
        "Deploy uses blue-green strategy with 5-minute health checks.",
        "Database migrations run before traffic shifts.",
        "Rollback is automatic if error rate exceeds 1% for 2 minutes.",
    ]

    for note in notes:
        await agent.record(note)

    # The LLM sees all three notes in its context — no need to pass them explicitly
    answer = await agent.answer("What triggers an automatic rollback?")
    print(f"Answer: {answer}")

    # Static block: pin a value once — useful for specs, plans, decisions.
    # Unlike set_dynamic, this is evaluated once and never re-evaluated.
    agent.context["policy"] = "Always prefer rollback over forward-fix during incidents."
    answer2 = await agent.answer("Should we try to fix forward or roll back?")
    print(f"Answer: {answer2}")
