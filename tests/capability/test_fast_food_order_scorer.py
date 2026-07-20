# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fast-food order capability scorer."""

from eval_pipeline.models import ScoringContext
from tests.capability.agents.order import FastFoodOrderScorer

EXPECTED_EXTRA_MAYO = {
    "order_submitted": True,
    "order_items": [
        {
            "product_id": 1002,
            "modifications": {
                "additions": ["mayo"],
                "removals": ["lettuce"],
                "special_instructions": ["extra mayo"],
            },
        },
        {"product_id": 2001, "size": "medium"},
    ],
}


def score(expected, actual):
    return FastFoodOrderScorer().score(
        ScoringContext(task_id="fast-food", input="", expected=expected, actual=actual)
    )


def test_extra_mayo_as_addition_only_matches_expected_special_instruction():
    actual = {
        "order_submitted": True,
        "order_items": [
            {
                "product_id": 1002,
                "modifications": {"additions": ["mayo"], "removals": ["lettuce"]},
            },
            {"product_id": 2001, "size": "medium"},
        ],
    }

    result = score(EXPECTED_EXTRA_MAYO, actual)

    assert result.score == 1.0
    assert result.reasoning == "Order state matches"


def test_extra_mayo_as_special_instruction_only_matches_expected_addition():
    expected = {
        "order_submitted": True,
        "order_items": [
            {
                "product_id": 1002,
                "modifications": {"additions": ["mayo"], "removals": ["lettuce"]},
            },
        ],
    }
    actual = {
        "order_submitted": True,
        "order_items": [
            {
                "product_id": 1002,
                "modifications": {
                    "removals": ["lettuce"],
                    "special_instructions": ["extra mayo"],
                },
            },
        ],
    }

    result = score(expected, actual)

    assert result.score == 1.0
    assert result.reasoning == "Order state matches"


def test_missing_coke_still_fails():
    expected = {
        "order_submitted": True,
        "order_items": [
            {
                "product_id": 1001,
                "modifications": {
                    "additions": ["cheese", "bacon"],
                    "removals": ["tomato"],
                },
            },
            {"product_id": 2001, "size": "large"},
            {"product_id": 3001, "size": "medium"},
        ],
    }
    actual = {
        "order_submitted": True,
        "order_items": [
            {
                "product_id": 1001,
                "modifications": {
                    "additions": ["cheese", "bacon"],
                    "removals": ["tomato"],
                },
            },
            {"product_id": 2001, "size": "large"},
        ],
    }

    result = score(expected, actual)

    assert result.score == 0.0
    assert "product_id': 3001" in result.reasoning


def test_cancel_flow_still_requires_canceled_flag():
    expected = {"order_canceled": True, "order_items": []}
    actual = {"order_items": []}

    result = score(expected, actual)

    assert result.score == 0.0
    assert "order_canceled" in result.reasoning


def test_independent_order_items_match_regardless_of_order():
    expected = {
        "order_submitted": True,
        "order_items": [
            {"product_id": 3001, "size": "medium"},
            {"product_id": 2001, "size": "large"},
        ],
    }
    actual = {
        "order_submitted": True,
        "order_items": [
            {"product_id": 2001, "size": "large"},
            {"product_id": 3001, "size": "medium"},
        ],
    }

    result = score(expected, actual)

    assert result.score == 1.0
    assert result.reasoning == "Order state matches"


def test_duplicate_order_item_still_fails_after_order_normalization():
    expected = {
        "order_submitted": True,
        "order_items": [
            {"product_id": 1001},
            {"product_id": 2001, "size": "medium"},
        ],
    }
    actual = {
        "order_submitted": True,
        "order_items": [
            {"product_id": 2001, "size": "medium"},
            {"product_id": 1001},
            {"product_id": 2001, "size": "medium"},
        ],
    }

    result = score(expected, actual)

    assert result.score == 0.0
    assert "order_items" in result.reasoning
