# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 03: SW1 + SW3 — regular Python methods as LLM tools.

uv run python examples/quickstart/03_codeact_tools.py
"""

from typing import TypedDict

from nooa.util.quickstart import *


class Result(TypedDict):
    can_fulfill: bool
    total_cost: float
    unavailable_items: list[str]


class InventoryAgent(Agent, llm=llm):
    """You are an agent that checks inventory using deterministic helper methods."""

    def __init__(self):
        super().__init__()
        self.inventory = {
            "apple": {"stock": 50, "price": 0.75},
            "banana": {"stock": 30, "price": 0.50},
            "orange": {"stock": 0, "price": 0.80},  # Out of stock
        }

    # SW1: Deterministic Python - automatically available as "tools" for the LLM
    def get_stock(self, item: str) -> int:
        """Get current stock for an item."""
        return self.inventory.get(item, {}).get("stock", 0)

    def get_price(self, item: str) -> float:
        """Get price for an item."""
        return self.inventory.get(item, {}).get("price", 0.0)

    # SW3: Generation method - LLM implements this, calling SW1 methods as needed
    async def can_fulfill_order(self, items: list[str], budget: float) -> Result:
        """Check if order can be fulfilled within budget."""
        ...


@autorun
async def main():
    agent = InventoryAgent()
    result = await agent.can_fulfill_order(["apple", "banana", "orange"], budget=5.0)
    print(f"Can fulfill: {result['can_fulfill']}")
    print(f"Total cost: {result['total_cost']}")
    print(f"Unavailable items: {result['unavailable_items']}")
