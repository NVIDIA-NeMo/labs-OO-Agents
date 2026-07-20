# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 10: Skills — inject curated context into an agent.

uv run python examples/quickstart/10_skills.py
"""

from pathlib import Path

from nooa import TextSkill
from nooa.util.quickstart import *

ASSETS = Path(__file__).parent.parent / "assets"


class FrontendAgent(Agent, llm=llm):
    """Agent with a single file-based skill."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontend_design = TextSkill(path=ASSETS / "frontend-design")

    async def respond(self, prompt: str) -> str:
        """Respond to a user message."""
        ...


class GenericAgent(Agent, llm=llm):
    """Agent that loads all skills from a directory by scanning for SKILL.md."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load every subdirectory containing a SKILL.md as a TextSkill
        for entry in sorted(ASSETS.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").exists():
                setattr(self, entry.name.replace("-", "_"), TextSkill(path=entry))

    async def respond(self, prompt: str) -> str:
        """Respond to a user message."""
        ...


@autorun
async def main():
    agent = FrontendAgent()
    result = await agent.respond("Can you build a responsive layout for a landing page?")
    print(result)
    agent = GenericAgent()
    result = await agent.respond("Can you build a responsive layout for a landing page?")
    print(result)
