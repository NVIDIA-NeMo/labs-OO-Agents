# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Note-taking agent for testing Context API usage.

This test verifies that the LLM can:
1. Discover the Notes interface (add_note, render_notes)
2. Realize it can pin self.notes.render_notes() into context
3. Use that persisted context to answer questions about stored notes
"""

from nooa import Agent
from nooa.agentdoc import spec
from nooa.context_blocks import DynamicContext


class Notes:
    """Interface for storing and rendering notes."""

    def __init__(self):
        self._notes: list[str] = []

    def add_note(self, note: str) -> None:
        """Add a note to the collection."""
        self._notes.append(note)

    def render_notes(self) -> str:
        """Render all stored notes as a formatted string suitable for context."""
        if not self._notes:
            return "No notes stored."
        return "\n".join(f"- {note}" for note in self._notes)

    def __len__(self) -> int:
        return len(self._notes)


class NoteTakingAgent(Agent):
    """An agent that takes notes and answers questions about them.

    Available state:
    - self.notes: Notes interface with add_note(note) and render_notes() methods
    - self.context: Context API for persisting information across calls
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notes = Notes()
        spec(self, "context", hidden=False)

    async def process_note(self, note: str) -> str:
        """Store the given note and ensure notes are visible in context for future calls.

        Use self.notes to store and render notes.
        Use self.context to make rendered notes persist across calls.
        """
        ...

    async def answer_question(self, question: str) -> str:
        """Answer a question using notes available in context."""
        ...


class NoteTakingTestWrapper(Agent):
    """Wrapper for note-taking capability tests.

    Provides a single entry point that:
    1. Adds multiple notes via process_note()
    2. Asks a question via answer_question()
    3. Returns the answer for scoring
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inner_agent: NoteTakingAgent | None = None

    async def run_test(
        self,
        notes: list[str],
        question: str,
    ) -> dict:
        """Run a note-taking test sequence.

        Args:
            notes: List of notes to add
            question: Question to ask about the notes

        Returns:
            Dict with test results for scoring
        """
        self._inner_agent = NoteTakingAgent(llm=self._llm)

        for note in notes:
            await self._inner_agent.process_note(note)

        answer = await self._inner_agent.answer_question(question)

        context_keys = list(self._inner_agent.context.keys())
        dynamic_blocks = {
            k: v.expr
            for k, v in self._inner_agent.context_manager._raw_items()
            if isinstance(v, DynamicContext)
        }

        return {
            "answer": answer,
            "notes_count": len(self._inner_agent.notes),
            "context_blocks": context_keys,
            "dynamic_blocks": dynamic_blocks,
        }
