# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public declarations and result objects for decision strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

type Criterion = str | dict[str, JsonValue] | list[JsonValue] | None
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


@dataclass(frozen=True, slots=True)
class Instructions:
    """Describe what a decision output should predict."""

    value: Criterion


@dataclass(frozen=True, slots=True, init=False)
class Criteria:
    """Describe decision outcomes positionally or by their Python values.

    Positional entries describe ordered score levels or choices. ``by_value``
    describes outcomes whose Python values matter, such as booleans, enum
    members, or literals.

    Examples:
        ``Criteria("calm", "frustrated", "angry")``
        ``Criteria(by_value={True: "urgent", False: "not urgent"})``
    """

    value: tuple[Criterion, ...] | dict[Any, Criterion]

    def __init__(
        self,
        *entries: Criterion,
        by_value: dict[Any, Criterion] | None = None,
    ) -> None:
        """Create criteria from ordered entries or an explicit value mapping.

        Args:
            *entries: Descriptions in the same order as score levels or choices.
            by_value: Descriptions keyed by the corresponding Python outcome.

        Raises:
            TypeError: If positional entries and ``by_value`` are both supplied.
        """
        if entries and by_value is not None:
            raise TypeError("Criteria accepts positional entries or by_value, not both")
        value: tuple[Criterion, ...] | dict[Any, Criterion]
        value = dict(by_value) if by_value is not None else tuple(entries)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class Threshold:
    """Probability cutoff used to interpret a decision locally."""

    value: float

    def __post_init__(self) -> None:
        """Reject cutoffs that cannot represent a probability."""
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise ValueError("Threshold must be finite and between 0 and 1")


class DecisionModelRequiredError(RuntimeError):
    """The declared result requires evidence unavailable from an LLM fallback."""


class Decision[T](BaseModel):
    """Base class for a normalized decision result."""

    value: T


class BooleanDecision(Decision[bool]):
    """Boolean value plus the backend probability of true."""

    type: Literal["boolean"] = "boolean"
    probability_true: Probability
    threshold: Probability = 0.5

    @model_validator(mode="after")
    def _value_matches_threshold(self) -> BooleanDecision:
        """Ensure the boolean value agrees with its probability and threshold."""
        expected = self.probability_true >= self.threshold
        if self.value is not expected:
            raise ValueError("value must equal probability_true >= threshold")
        return self


class ChoiceDecision[E: Enum](Decision[E | None]):
    """Selected enum member, distribution, and optional acceptance cutoff."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["choice"] = "choice"
    selected: E
    probabilities: dict[E, Probability]
    confidence: Probability
    threshold: Probability | None = None

    @model_validator(mode="after")
    def _value_matches_threshold(self) -> ChoiceDecision[E]:
        """Ensure the accepted value agrees with the selected-choice threshold."""
        selected_probability = self.probabilities.get(self.selected)
        if selected_probability is None:
            raise ValueError("probabilities must contain the selected choice")
        expected = (
            self.selected
            if self.threshold is None or selected_probability >= self.threshold
            else None
        )
        if self.value != expected:
            raise ValueError("value does not match the selected choice and threshold")
        return self


class ScoreDecision(Decision[float]):
    """Numeric rubric score with its distribution and legend."""

    type: Literal["score"] = "score"
    probabilities: dict[int, Probability]
    legend: dict[int, Criterion]
    confidence: Probability


__all__ = [
    "BooleanDecision",
    "ChoiceDecision",
    "Criteria",
    "Criterion",
    "Decision",
    "DecisionModelRequiredError",
    "Instructions",
    "Probability",
    "ScoreDecision",
    "Threshold",
]
