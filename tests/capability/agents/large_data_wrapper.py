# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large data test wrapper - generates test data on-the-fly."""

from nooa import Agent


class LargeDataTestWrapper(Agent):
    """Wrapper that generates large test data dynamically.

    This avoids storing 2MB+ of repeated strings in git.
    Test data is generated on-the-fly with configurable sizes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = None  # Will be set by call_agent

    def _generate_large_data(
        self,
        base_pattern: str,
        repetitions: int,
        marker: str | None = None,
        marker_position: int | None = None,
    ) -> str:
        """Generate large test data with optional marker insertion.

        Args:
            base_pattern: Pattern to repeat (e.g., "Lorem ipsum dolor sit amet. ")
            repetitions: How many times to repeat
            marker: Optional marker to insert
            marker_position: Position to insert marker (in repetitions, not chars)

        Returns:
            Generated data string
        """
        if marker and marker_position is not None:
            # Build with marker at specific position
            before = base_pattern * marker_position
            after = base_pattern * (repetitions - marker_position)
            return before + marker + after
        else:
            return base_pattern * repetitions

    async def call_agent(
        self,
        test_case: str,
        marker: str | None = None,
        pattern: str | None = None,
        context_chars: int | None = None,
    ) -> dict | str:
        """Generate test data and call appropriate agent method.

        Args:
            test_case: Which test to run ("find_found", "find_not_found", "extract", "count")
            marker: Marker to find (for find/extract tests)
            pattern: Pattern to count (for count test)
            context_chars: Context chars for extract test

        Returns:
            Result from the agent method
        """
        from .large_data import LargeDataAgent

        # Create agent instance
        if self.agent is None:
            self.agent = LargeDataAgent(llm=self._llm)

        # Generate test data based on test case
        if test_case == "find_found":
            # 790KB data with marker at position 570,000
            # Data: (base * 10000) + marker + (after_base * 5000)
            base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            before = base * 10000
            after = " Sed do eiusmod tempor incididunt ut labore." * 5000
            data = before + marker + after
            return await self.agent.find_marker(data, marker)

        elif test_case == "find_not_found":
            # 540KB data without marker
            base = "The quick brown fox jumps over the lazy dog. "
            data = base * 12000
            return await self.agent.find_marker(data, marker)

        elif test_case == "extract":
            # 376KB data with marker at position 184,000
            # Data: (base * 8000) + marker + (after_base * 8000)
            base = "Background noise text. "
            before = base * 8000
            after = " More background noise. " * 8000
            data = before + marker + after
            return await self.agent.extract_around_marker(data, marker, context_chars)

        elif test_case == "count":
            # 320KB data with pattern repeated 20,000 times
            base = f"Pattern: {pattern}. "
            data = base * 20000
            return await self.agent.count_pattern(data, pattern)

        else:
            raise ValueError(f"Unknown test case: {test_case}")
