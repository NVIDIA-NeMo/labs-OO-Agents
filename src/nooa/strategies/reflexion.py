# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflexion strategy - generate → reflect → improve loop.

A composite strategy that wraps another strategy (default: PurePythonStrategy)
and adds self-critique to iteratively improve outputs.

Flow:
1. Run base strategy to generate initial result
2. Reflect on result using structured output
3. If satisfactory → return result
4. If not → add reflection feedback to events, repeat

Inspired by: Methodic's Reflexion strategy
"""

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from nooa.agentdoc import truncating_pformat
from nooa.decorators import strategy
from nooa.errors import GenerationError
from nooa.events import Feedback
from nooa.strategies.base import (
    GenerationStrategy,
    RuntimeServices,
    build_sampling_kwargs,
)
from nooa.strategies.template import TemplateStrategy

if TYPE_CHECKING:
    from nooa.config.strategy_config import ReflexionConfig
    from nooa.config.truncation_config import TruncationConfig
    from nooa.context_blocks import DynamicContext
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)

# === Reflection Output Model ===


class ReflectionOutput(BaseModel):
    """Structured output for reflection evaluation.

    The LLM uses this to evaluate whether the generated result is satisfactory
    and what improvements could be made.
    """

    is_satisfactory: bool = Field(
        description="True if the result fully addresses the task requirements"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Problems identified with the current result",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific improvements to make",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the evaluation",
    )


class ReflexionStrategy(GenerationStrategy):
    """Reflexion strategy: generate → reflect → improve loop.

    Wraps a base strategy (default: PurePythonStrategy) and adds self-critique.
    The LLM evaluates its own output and iteratively improves based on feedback.

    Use this for tasks requiring high quality:
    - Complex analysis that benefits from multiple passes
    - Tasks where correctness is critical
    - Outputs that need comprehensive coverage

    Configuration:
        base: The underlying generation strategy (default: PurePythonStrategy)
        config: ReflexionConfig with max_iterations (default: 3)

    Example:
        from nooa.config.strategy_config import ReflexionConfig

        @strategy(ReflexionStrategy(config=ReflexionConfig(max_iterations=2)))
        def analyze(self, data: str) -> dict:
            '''Thoroughly analyze the data.'''
    """

    def __init__(
        self,
        base: GenerationStrategy | None = None,
        config: "ReflexionConfig | None" = None,
    ):
        """Initialize Reflexion strategy.

        Args:
            base: Base strategy for code generation.
                  Defaults to PurePythonStrategy().
            config: ReflexionConfig with iteration limits and sampling params.
        """
        from nooa.config.strategy_config import ReflexionConfig as _RC

        # Lazy import to avoid circular dependency
        from nooa.strategies.pure_python import PurePythonStrategy

        self.base = base if base is not None else PurePythonStrategy()
        self.config = config or _RC()

    def _build_sampling_kwargs(self) -> dict[str, Any]:
        """Build sampling kwargs for llm calls, excluding None values."""
        return build_sampling_kwargs(self.config)

    @property
    def name(self) -> str:
        """Strategy name."""
        return f"REFLEXION[{self.base.name}]"

    def get_block_overrides(self) -> dict[str, "str | DynamicContext | None"]:
        """Delegate to base strategy's block overrides."""
        return self.base.get_block_overrides()

    @property
    def requires_lock(self) -> bool:
        """Inherit lock requirement from base strategy."""
        return self.base.requires_lock

    @strategy(TemplateStrategy())
    async def reflection_prompt(self, runtime: RuntimeServices, task: str, result: str) -> str:
        """Critically evaluate the result of the task execution.

        Original task: {task}

        Result produced:
        {result}

        Evaluate:
        1. Does this result fully address the task requirements?
        2. Are there any errors, inaccuracies, or inconsistencies?
        3. What specific improvements could be made?

        Be strict but fair. Only mark as satisfactory if the result truly meets all requirements."""
        ...

    async def execute(self, runtime: RuntimeServices, call: "CurrentCall") -> Any:
        """Execute Reflexion strategy.

        Runs the base strategy, then reflects on the result. If not satisfactory,
        adds feedback to events and reruns the base strategy.

        Args:
            runtime: RuntimeServices providing LLM, execution, and event management.
            call: CurrentCall with method details and arguments.

        Returns:
            The result after reflection deems it satisfactory, or the best
            result after max_reflections iterations.

        Raises:
            GenerationError: If base strategy fails on all attempts.
        """
        result = None
        last_error: Exception | None = None

        for iteration in range(self.config.max_iterations):
            logger.debug(
                f"[REFLEXION iter={iteration + 1}/{self.config.max_iterations}] "
                f"Running base strategy: {self.base.name}"
            )

            try:
                # Run base strategy via execute_nested for proper instrumentation
                result = await runtime.execute_nested(self.base, call)

                # Reflect on result
                reflection = await self._reflect(runtime, call, result)

                logger.debug(
                    f"[REFLEXION iter={iteration + 1}] "
                    f"Reflection: satisfactory={reflection.is_satisfactory}, "
                    f"issues={len(reflection.issues)}"
                )

                if reflection.is_satisfactory:
                    logger.debug(
                        f"[REFLEXION] Satisfactory result after {iteration + 1} iteration(s)"
                    )
                    return result

                # Not satisfactory - add feedback for next iteration
                if iteration < self.config.max_iterations - 1:
                    feedback = self._format_reflection_feedback(reflection, call)
                    runtime.event_manager.add(Feedback(content=feedback))
                    logger.debug("[REFLEXION] Added feedback, continuing to next iteration")

            except GenerationError as e:
                # Base strategy failed - track error but try again if iterations remain
                last_error = e
                logger.debug(f"[REFLEXION iter={iteration + 1}] Base strategy failed: {e}")
                if iteration < self.config.max_iterations - 1:
                    # Add error context for next iteration
                    runtime.event_manager.add(
                        Feedback(
                            content=f"Previous attempt failed: {e}\n\n"
                            f"Please try a different approach for `{call.method_name}`."
                        )
                    )

        # Exhausted reflections
        if result is not None:
            logger.warning(
                f"[REFLEXION] Returning result after {self.config.max_iterations} iterations "
                f"(not fully satisfactory)"
            )
            return result

        # No successful result at all
        if last_error:
            raise GenerationError(
                f"REFLEXION generation failed after {self.config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )
        else:
            raise GenerationError(
                f"REFLEXION generation failed after {self.config.max_iterations} attempts with no result."
            )

    async def _reflect(
        self,
        runtime: RuntimeServices,
        call: "CurrentCall",
        result: Any,
    ) -> ReflectionOutput:
        """Generate reflection on the result.

        Uses structured output to get a consistent evaluation format.

        Args:
            runtime: RuntimeServices for LLM access.
            call: The method call being evaluated.
            result: The result to reflect on.

        Returns:
            ReflectionOutput with satisfaction status and feedback.
        """
        # Build task description
        task_parts = [f"Method: `{call.method_name}`"]
        if call.signature:
            task_parts.append(f"Signature: {call.signature}")
        if call.docstring:
            task_parts.append(f"Docstring: {call.docstring}")
        task_description = "\n".join(task_parts)

        # Format result for reflection
        result_str = self._format_result_for_reflection(result, runtime.truncation_config)

        # Build reflection prompt using @strategy method with TemplateStrategy
        prompt = await self.reflection_prompt(runtime, task=task_description, result=result_str)

        # Add reflection request to events temporarily
        runtime.event_manager.add(
            Feedback(content=prompt),
            record=True,
        )

        # Generate reflection with structured output
        response, _event_id = await runtime.generate(
            tools=None,
            output_model=ReflectionOutput,
            **self._build_sampling_kwargs(),
        )

        # Parse response
        if isinstance(response.content, ReflectionOutput):
            return response.content
        elif isinstance(response.content, dict):
            return ReflectionOutput(**response.content)
        else:
            # Fallback: assume NOT satisfactory to trigger retry
            logger.warning(
                "[REFLEXION] Could not parse reflection response, assuming not satisfactory to retry"
            )
            return ReflectionOutput(
                is_satisfactory=False,
                reasoning="Could not parse reflection response",
                issues=["Failed to parse reflection - retrying"],
            )

    def _format_result_for_reflection(self, result: Any, tc: "TruncationConfig") -> str:
        """Format result for the reflection prompt.

        Args:
            result: The result to format.
            tc: Truncation config controlling how much of the result to show.

        Returns:
            String representation suitable for LLM evaluation.
        """
        if result is None:
            return "None (no value returned)"

        # Show as much of the result as possible — the LLM must evaluate its quality.
        # event_format provides max_string/max_length/max_depth so nested strings
        # within structured results are bounded by cfg.event_format rather than
        # pformat's hidden 150-char fallback.
        return truncating_pformat(
            result,
            **tc.event_format.model_dump(),
        )

    def _format_reflection_feedback(
        self,
        reflection: ReflectionOutput,
        call: "CurrentCall",
    ) -> str:
        """Format reflection as feedback for the next iteration.

        Args:
            reflection: The reflection evaluation.
            call: The method call context.

        Returns:
            Formatted feedback string for events.
        """
        parts = [
            f"Your implementation of `{call.method_name}` needs improvement.",
            "",
        ]

        if reflection.reasoning:
            parts.append(f"**Evaluation:** {reflection.reasoning}")
            parts.append("")

        if reflection.issues:
            parts.append("**Issues identified:**")
            for issue in reflection.issues:
                parts.append(f"- {issue}")
            parts.append("")

        if reflection.suggestions:
            parts.append("**Suggestions for improvement:**")
            for suggestion in reflection.suggestions:
                parts.append(f"- {suggestion}")
            parts.append("")

        parts.append(
            f"Please create an improved implementation of `{call.method_name}` "
            f"that addresses these issues."
        )

        return "\n".join(parts)
