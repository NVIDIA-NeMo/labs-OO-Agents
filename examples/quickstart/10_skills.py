# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 10: Skills — inject curated context into an agent.

uv run python examples/quickstart/10_skills.py
"""

from pathlib import Path

from nooa import TextSkill
from nooa.skill_registry import SkillRegistry
from nooa.util.quickstart import *

ASSETS = Path(__file__).parent.parent / "assets"


class FrontendAgent(Agent, llm=llm):
    """Use self.frontend_design.files to inspect packaged files.

    Use self.frontend_design.shell for file access and script execution.
    """

    frontend_design: TextSkill

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontend_design = TextSkill(path=ASSETS / "frontend-design")

    async def respond(self, prompt: str) -> str:
        """Respond to a user message."""
        ...


class GenericAgent(Agent, llm=llm):
    """Agent that registers and activates a model-facing skill."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skills = SkillRegistry(self)
        self.skills.register(
            "local.frontend_design",
            TextSkill(path=ASSETS / "frontend-design"),
        )
        self.skills.activate(["local.frontend_design"])

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
