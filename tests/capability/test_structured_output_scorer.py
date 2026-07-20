# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for structured output capability scorer."""

from eval_pipeline.models import ScoringContext
from tests.capability.agents.structured_output import StructuredCombinedValueScorer

EXPECTED_RECOMMEND = {
    "user": {"name": "Alice Johnson", "age": 28, "email": "alice.j@example.com"},
    "review": {
        "product_name": "Wireless Headphones XP-200",
        "rating": 5,
        "would_recommend": True,
        "key_points": [
            "Crystal clear sound quality",
            "8+ hours battery life",
            "Exceptional comfort",
            "Premium build quality",
        ],
    },
    "summary": "Alice Johnson (28) highly recommends the Wireless Headphones XP-200 with a 5-star rating",
}


EXPECTED_NOT_RECOMMEND = {
    "user": {"name": "Bob Martinez", "age": 35, "email": "bob.martinez@techcorp.com"},
    "review": {
        "product_name": "SmartWatch Pro Series 3",
        "rating": 2,
        "would_recommend": False,
        "key_points": [
            "Sleek modern design",
            "Basic fitness tracking works well",
            "Poor battery life",
            "Clunky app integration",
            "Reasonable price point",
        ],
    },
    "summary": "Bob Martinez (35) does not recommend the SmartWatch Pro Series 3 due to battery and app issues",
}


def score(expected, actual):
    return StructuredCombinedValueScorer().score(
        ScoringContext(task_id="structured", input="", expected=expected, actual=actual)
    )


def test_summary_does_not_need_to_repeat_positive_rating():
    actual = {
        **EXPECTED_RECOMMEND,
        "summary": "Alice Johnson (28) highly recommends the Wireless Headphones XP-200.",
    }

    result = score(EXPECTED_RECOMMEND, actual)

    assert result.score == 1.0
    assert result.reasoning == "Nested Pydantic values match"


def test_summary_does_not_need_to_repeat_negative_rating():
    actual = {
        **EXPECTED_NOT_RECOMMEND,
        "summary": "Bob Martinez (35) would not recommend the SmartWatch Pro Series 3.",
    }

    result = score(EXPECTED_NOT_RECOMMEND, actual)

    assert result.score == 1.0
    assert result.reasoning == "Nested Pydantic values match"


def test_summary_still_requires_recommendation_direction():
    actual = {
        **EXPECTED_NOT_RECOMMEND,
        "summary": "Bob Martinez (35) reviewed the SmartWatch Pro Series 3.",
    }

    result = score(EXPECTED_NOT_RECOMMEND, actual)

    assert result.score == 0.0
    assert "summary missing negative recommendation" in result.reasoning


def test_review_rating_is_still_checked_as_structured_value():
    actual = {
        **EXPECTED_RECOMMEND,
        "review": {**EXPECTED_RECOMMEND["review"], "rating": 4},
        "summary": "Alice Johnson (28) highly recommends the Wireless Headphones XP-200.",
    }

    result = score(EXPECTED_RECOMMEND, actual)

    assert result.score == 0.0
    assert "review.rating: expected 5, got 4" in result.reasoning
