# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured output capability tests using PredictStrategy with Pydantic models."""

from typing import Annotated

from pydantic import BaseModel, Field

from nooa import Agent, strategy
from nooa.strategies import PredictStrategy


class UserInfo(BaseModel):
    """User information extracted from text."""

    name: str = Field(..., description="Full name of the user")
    age: int = Field(..., description="Age in years")
    email: str = Field(..., description="Email address")


class ReviewInfo(BaseModel):
    """Product review information extracted from text."""

    product_name: str = Field(..., description="Name of the product being reviewed")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    would_recommend: bool = Field(..., description="Whether the reviewer recommends this product")
    key_points: list[str] = Field(..., description="Key points from the review")


class CombinedResult(BaseModel):
    """Combined extraction result with user and review information."""

    user: UserInfo = Field(..., description="Extracted user information")
    review: ReviewInfo = Field(..., description="Extracted review information")
    summary: str = Field(..., description="Brief summary combining user context and review")


class PredictAgent(Agent):
    """Agent that tests PredictStrategy with composed Pydantic models.

    This agent demonstrates extracting multiple structured pieces from rich text
    and composing them into a single result.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @strategy(PredictStrategy())
    async def extract_user_info(
        self, text: Annotated[str, "Text containing user information"]
    ) -> UserInfo:
        """Extract user information from the given text."""
        ...

    @strategy(PredictStrategy())
    async def extract_review_info(
        self, text: Annotated[str, "Text containing review information"]
    ) -> ReviewInfo:
        """Extract product review information from the given text."""
        ...

    @strategy(PredictStrategy())
    async def combine_extraction(self, user: UserInfo, review: ReviewInfo) -> CombinedResult:
        """Combine user and review information into a single result with summary.

        Create a CombinedResult that includes:
        - The user information
        - The review information
        - A summary that mentions the user's name, age, product, and recommendation
        """
        ...

    async def process_review(self, text: str) -> CombinedResult:
        """Orchestrator method that extracts and combines structured information.

        This regular (non-generation) method:
        1. Extracts user info using extract_user_info()
        2. Extracts review info using extract_review_info()
        3. Combines them using combine_extraction()

        Args:
            text: Rich text containing both user and review information

        Returns:
            CombinedResult object with all extracted and combined information
        """
        # Step 1: Extract user information
        user_info = await self.extract_user_info(text)

        # Step 2: Extract review information
        review_info = await self.extract_review_info(text)

        # Step 3: Combine them with LLM-generated summary
        result = await self.combine_extraction(user_info, review_info)

        return result


class StructuredCombinedValueScorer:
    # Score nested structured extraction values without requiring exact generated prose.
    _CONCEPTS = {
        "Wireless Headphones XP-200": [
            ("sound quality", ("sound", "quality")),
            ("8+ hours battery life", ("battery", "8", "hour")),
            ("comfort", ("comfort",)),
            ("premium build quality", ("premium", "build")),
        ],
        "SmartWatch Pro Series 3": [
            ("sleek modern design", ("sleek", "design")),
            ("basic fitness tracking", ("fitness", "tracking")),
            ("poor battery life", ("battery", ("poor", "disappointing", "daily", "barely"))),
            ("clunky app integration", ("app", ("clunky", "sync", "integration"))),
            ("reasonable price", ("price", ("reasonable", "fair"))),
        ],
        "Coffee Maker Deluxe Pro": [
            ("good coffee", ("coffee", ("good", "rich", "flavor"))),
            ("programmable timer", ("programmable", "timer")),
            ("easy to clean", (("clean", "dishwasher"),)),
            ("fair price", ("price", ("fair", "reasonable"))),
            ("better insulation", (("insulation", "insulated"), ("carafe", "hot"))),
        ],
    }

    def score(self, ctx):
        from eval_pipeline.models import ScoreResult

        expected = self._to_dict(ctx.expected)
        actual = self._to_dict(ctx.actual)
        errors = []
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return ScoreResult(
                score=0.0,
                reasoning=f"Expected and actual must be mappings, got {type(expected).__name__} and {type(actual).__name__}",
            )
        self._check_user(expected.get("user", {}), actual.get("user", {}), errors)
        expected_review = expected.get("review", {})
        actual_review = actual.get("review", {})
        self._check_review(expected_review, actual_review, errors)
        self._check_summary(expected, actual, errors)
        ok = not errors
        return ScoreResult(
            score=1.0 if ok else 0.0,
            reasoning="Nested Pydantic values match" if ok else "; ".join(errors[:6]),
            metadata={"errors": errors, "error_count": len(errors)},
        )

    def _to_dict(self, value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if isinstance(value, dict):
            return {k: self._to_dict(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_dict(v) for v in value]
        return value

    def _check_user(self, expected, actual, errors):
        for key in ("name", "age", "email"):
            if actual.get(key) != expected.get(key):
                errors.append(
                    f"user.{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
                )

    def _check_review(self, expected, actual, errors):
        for key in ("product_name", "rating", "would_recommend"):
            if actual.get(key) != expected.get(key):
                errors.append(
                    f"review.{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
                )
        product = expected.get("product_name")
        actual_points = actual.get("key_points")
        if not isinstance(actual_points, list) or not all(
            isinstance(p, str) for p in actual_points
        ):
            errors.append("review.key_points: expected a list of strings")
            return
        actual_text = self._normalize(" ".join(actual_points))
        missing = []
        for label, requirements in self._CONCEPTS.get(product, []):
            if not self._requirements_met(requirements, actual_text):
                missing.append(label)
        if missing:
            errors.append(f"review.key_points missing concepts: {missing}")

    def _check_summary(self, expected, actual, errors):
        summary = actual.get("summary")
        if not isinstance(summary, str):
            errors.append("summary: expected string")
            return
        summary_norm = self._normalize(summary)
        user = expected.get("user", {})
        review = expected.get("review", {})
        required = []
        required.extend(self._normalize(str(user.get("name", ""))).split())
        required.append(str(user.get("age", "")))
        required.extend(self._normalize(str(review.get("product_name", ""))).split())
        # The model is asked for a brief summary mentioning name, age, product,
        # and recommendation. The rating is validated on review.rating, but is
        # not required to be repeated in summary prose.
        missing = [token for token in required if token and token not in summary_norm]
        if missing:
            errors.append(f"summary missing required values: {missing}")
        recommends = review.get("would_recommend")
        if recommends is True and "recommend" not in summary_norm:
            errors.append("summary missing recommendation")
        if recommends is False and not (
            "not recommend" in summary_norm
            or "would not recommend" in summary_norm
            or "wouldnt recommend" in summary_norm
        ):
            errors.append("summary missing negative recommendation")

    def _normalize(self, text):
        import re

        text = text.lower().replace("+", " ")
        tokens = re.findall(r"[a-z0-9]+", text)
        stems = []
        for token in tokens:
            if token.endswith("ing") and len(token) > 5:
                token = token[:-3]
            elif token.endswith("ed") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            stems.append(token)
        return " ".join(stems)

    def _requirements_met(self, requirements, text):
        for requirement in requirements:
            if isinstance(requirement, tuple):
                if not any(self._normalize(str(option)) in text for option in requirement):
                    return False
            elif self._normalize(str(requirement)) not in text:
                return False
        return True


__all__ = [
    "CombinedResult",
    "PredictAgent",
    "ReviewInfo",
    "StructuredCombinedValueScorer",
    "UserInfo",
]
