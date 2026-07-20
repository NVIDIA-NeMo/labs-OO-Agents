# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 15 — Export an ATIF trajectory.

Demonstrates how to export ATIF trajectories from OO agents: call
``enable_atif()`` once and every agent run writes a trajectory file you
can hand to downstream tooling (SFT pipelines, evals, dashboards).

Run it::

    uv run python examples/quickstart/14_atif_trajectory.py
"""

from pathlib import Path
from typing import Literal

from nooa.atif import enable_atif
from nooa.util.quickstart import *  # noqa: F403

# Categories defined once and reused as the contract for both the
# per-ticket classifier and the aggregate counts.
Category = Literal["bug", "feature", "question"]


@strategy(PredictStrategy())
async def categorize_ticket(ticket: str) -> Category:
    """Classify a single support ticket into its category."""
    ...


class TicketTriageAgent(Agent, llm=llm):
    """Triage support tickets into per-category counts."""

    def __init__(self):
        super().__init__()
        self.tickets = [
            "The app crashes when I open the settings page.",
            "Can you add dark mode support?",
            "How do I export my data to CSV?",
            "Login fails with a 500 error after the latest update.",
            "Please support keyboard shortcuts for power users.",
        ]

    def get_tickets(self) -> list[str]:
        """Return the open support tickets to triage."""
        return self.tickets

    async def triage(self) -> dict[Category, int]:
        """Categorize every open ticket and return the count per category.

        Classify each ticket with the async ``categorize_ticket`` helper.
        """
        ...


@autorun
async def main():
    # One call instruments every Agent created afterwards; trajectories
    # land under `output_dir/<AgentClass>/<session_id>.json`.
    enable_atif(output_dir="logs/atif")

    agent = TicketTriageAgent()
    counts = await agent.triage()
    print(f"Triage result: {counts}")

    latest = max(
        Path("logs/atif/TicketTriageAgent").glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    print(f"ATIF trajectory written to: {latest}")
