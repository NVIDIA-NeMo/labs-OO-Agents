# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router test agent with sub-agents that compute actual values.

The subagents perform real computations that the router must aggregate.
This prevents "cheating" by just returning expected agent names.

Uses TypedDict for explicit return types to catch errors at type-check time.
"""

from typing import TypedDict

from nooa import Agent

# ============================================================================
# Typed return structures - makes expected output explicit
# ============================================================================


class AnalyzerResult(TypedDict):
    """Result from AnalyzerSubAgent.analyze()"""

    sum: float
    mean: float
    count: int


class ValidatorResult(TypedDict):
    """Result from ValidatorSubAgent.validate()"""

    all_positive: bool
    no_duplicates: bool
    is_sorted: bool


class RouterResult(TypedDict):
    """Result from RouterTestWrapper.process()"""

    agents_called: list[str]
    results: dict[str, AnalyzerResult | ValidatorResult | str]


# ============================================================================
# Sub-agents that perform actual computations
# ============================================================================


class AnalyzerSubAgent(Agent):
    """Agent that computes statistics from numeric data."""

    async def analyze(self, values: list[float]) -> AnalyzerResult:
        """Compute statistics for the given numeric values.

        Args:
            values: List of numbers to analyze

        Returns:
            AnalyzerResult with computed statistics:
            - sum: Sum of all values
            - mean: Average of values
            - count: Number of values
        """
        ...


class TransformerSubAgent(Agent):
    """Agent that transforms data to different formats."""

    async def transform(self, values: list[float], format: str) -> str:
        """Transform the values to the specified format.

        Args:
            values: List of numbers to transform
            format: One of "CSV", "JSON", "SUM"

        Returns:
            String representation in the requested format:
            - CSV: "1,2,3,4,5"
            - JSON: "[1, 2, 3, 4, 5]"
            - SUM: "Total: 15"
        """
        ...


class ValidatorSubAgent(Agent):
    """Agent that validates data against rules."""

    async def validate(self, values: list[float]) -> ValidatorResult:
        """Validate the values against common rules.

        Args:
            values: List of numbers to validate

        Returns:
            ValidatorResult with validation results:
            - all_positive: True if all values > 0
            - no_duplicates: True if all values are unique
            - is_sorted: True if values are in ascending order
        """
        ...


# ============================================================================
# Router that delegates to subagents
# ============================================================================


class RouterTestWrapper(Agent):
    """Router that delegates to specialist subagents and aggregates their results.

    The router must actually call the subagents and return their computed results.
    Simply guessing the answer won't work - the subagents do real computation.
    """

    # Subagent classes available for instantiation
    AnalyzerSubAgent = AnalyzerSubAgent
    TransformerSubAgent = TransformerSubAgent
    ValidatorSubAgent = ValidatorSubAgent

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process(self, user_message: str, values: list[float]) -> RouterResult:
        """Route the request to appropriate subagent(s) and return their results.

        Parse the user message to determine which subagent(s) to call:
        - "analyze" -> call AnalyzerSubAgent.analyze(values)
        - "transform to X" -> call TransformerSubAgent.transform(values, format="X")
        - "validate" -> call ValidatorSubAgent.validate(values)
        - Multiple requests like "analyze and validate" -> call both, use asyncio.gather

        Create subagent instances with: agent = self.AnalyzerSubAgent(llm=self._llm)

        Return a RouterResult with:
        - agents_called: List of agent names that were called (e.g., ["Analyzer", "Validator"])
        - results: Dict mapping agent name to its computed result

        Example return for "analyze this":
        {
            "agents_called": ["Analyzer"],
            "results": {
                "Analyzer": {"sum": 15, "mean": 3.0, "count": 5}
            }
        }
        """
        ...
