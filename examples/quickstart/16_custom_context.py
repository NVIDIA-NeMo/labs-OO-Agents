# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quickstart 16: Use an independent context API and assembly policy.

uv run python examples/quickstart/16_custom_context.py
"""

import re

from nooa import Agent, Block, CacheBoundary, EventQuery, context_text, select_context_events
from nooa.context_blocks import Role
from nooa.util.quickstart import BaseModel, autorun, llm


class ResearchContext:
    """A context API designed for this agent, independent of NOOA's ContextApi."""

    def __init__(self):
        self._corpus: dict[str, str] = {}
        self._selected: dict[str, str] = {}

    def add(self, source: str, text: str) -> None:
        self._corpus[source] = text

    def search(self, query: str) -> list[str]:
        """Select relevant documents and return their source IDs."""
        terms = set(re.findall(r"\w+", query.casefold()))
        scores = {
            source: len(terms & set(re.findall(r"\w+", f"{source} {text}".casefold())))
            for source, text in self._corpus.items()
        }
        best = max(scores.values(), default=0)
        self._selected = {
            source: self._corpus[source]
            for source, score in scores.items()
            if score == best and score > 0
        }
        return list(self._selected)

    def selected(self) -> dict[str, str]:
        return dict(self._selected)


class ResearchContextView:
    """Turn ResearchContext state into the context for one model call."""

    async def assemble(self, owner, call):
        yield Block(
            key="research_api",
            content=(
                "Use execute_python to call "
                "self.research_context.search(query: str) -> list[str]. "
                "Search results become selected research context. "
                "Answer only from selected research and cite source IDs."
            ),
        )
        for event in select_context_events(owner.events, call=call):
            yield event
        yield CacheBoundary()
        selected = owner.research_context.selected()
        if selected:
            yield Block(
                key="selected_research",
                content=context_text(selected, call=call),
                role=Role.USER,
            )


class ResearchAnswer(BaseModel):
    answer: str
    sources: list[str]


class ResearchAgent(
    Agent,
    llm=llm,
    context_view=ResearchContextView(),
    event_query=EventQuery.current_call(),
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.research_context = ResearchContext()

    async def investigate(self, question: str) -> ResearchAnswer:
        """Research and answer the question using the ResearchContext API."""
        ...


async def main():
    agent = ResearchAgent()
    agent.research_context.add("aurora-brief", "Project Aurora's verification code is Q7-MANGO.")
    agent.research_context.add("borealis-brief", "Project Borealis meets in Oslo.")

    # A custom view replaces default assembly, so built-in context is not included.
    agent.context["ignored_builtin_context"] = "THIS MUST NOT REACH THE MODEL"

    result = await agent.investigate("What is Project Aurora's verification code?")
    print(result)


if __name__ == "__main__":
    autorun(main)
