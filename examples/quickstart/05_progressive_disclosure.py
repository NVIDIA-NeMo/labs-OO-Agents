# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 05: Progressive disclosure — LLM uses doc() to explore unknown types.

uv run python examples/quickstart/05_progressive_disclosure.py
"""

from typing import Any

from nooa.util.quickstart import *

_WAREHOUSE = {
    "ART-001": Artwork("Starry Night Print", "Van Gogh Studio", appraised_value=15000.0),
    "STK-001": StockHolding("NVDA", shares=100, price_per_share=875.50),
    "JWL-001": Jewelry("Diamond Ring", carats=2.5, rate_per_carat=8000.0),
    "COL-001": Collectible("Vintage Baseball Card", base_value=5000.0, condition="excellent"),
    "ART-002": Artwork("Modern Abstract", "Local Artist", appraised_value=2500.0),
    "STK-002": StockHolding("AAPL", shares=50, price_per_share=185.25),
}


def get_item(item_id: str) -> Any:
    """Retrieve an item from the warehouse by ID."""
    return _WAREHOUSE.get(item_id)


class WarehouseAppraiser(Agent, llm=llm):
    """Agent that appraises items without knowing their types ahead of time."""

    get_item = staticmethod(get_item)

    async def appraise_item(self, item_id: str) -> float:
        """Get the monetary value of an item."""
        ...


@autorun
async def main():
    appraiser = WarehouseAppraiser()
    test_items = ["ART-001", "STK-001", "JWL-001", "COL-001"]

    for item_id in test_items:
        value = await appraiser.appraise_item(item_id)
        item = get_item(item_id)
        item_type = type(item).__name__
        print(f"  {item_id} ({item_type}): ${value:,.2f}")
