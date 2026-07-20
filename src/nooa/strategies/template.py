# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TemplateStrategy - String templating without LLM calls.

This is the foundational strategy that all other strategies build upon.
No LLM call, just template rendering with Python expression evaluation.
"""

from typing import TYPE_CHECKING

from nooa.strategies.base import GenerationStrategy

if TYPE_CHECKING:
    from nooa.strategies.base import RuntimeServices
    from nooa.strategies.current_call import CurrentCall


class TemplateStrategy(GenerationStrategy):
    """Template rendering strategy using runtime.expand_variables().

    This is the "simplest" strategy - no LLM call, just string templating
    with Python expression evaluation.

    Use for prompt building, message formatting, and any string templating
    needs within strategies.

    Example:
        @strategy(TemplateStrategy())
        async def error_msg(self, runtime: RuntimeServices, method: str, line: int) -> str:
            '''Syntax error in `{method}` at line {line}.

            Available tools: {len(self.tools)}
            Current iteration: {call.iteration}'''
            ...

        # Calling:
        msg = await self.error_msg(runtime, method="analyze", line=42)
        # Result: "Syntax error in `analyze` at line 42.\n\nAvailable tools: 3\nCurrent iteration: 1"

    Template Variables:
        - Method parameters become template variables
        - {self.xxx} accesses agent attributes (via runtime.agent)
        - {call.xxx} accesses CurrentCall attributes
        - Supports Python expressions: {len(self.tools)}, {self.doc.show()}, etc.

    Features:
        - Full expression power via runtime.expand_variables()
        - No LLM call (requires_lock = False)
        - Fast rendering (<1ms typically)
        - Appears in traces as [TEMPLATE] strategy
    """

    @property
    def name(self) -> str:
        """Strategy name for tracing."""
        return "TEMPLATE"

    @property
    def traceable(self) -> bool:
        """No tracing — template rendering is not an LLM call."""
        return False

    @property
    def requires_lock(self) -> bool:
        """No LLM call needed, can run concurrently."""
        return False

    async def execute(
        self,
        runtime: "RuntimeServices",
        call: "CurrentCall",
    ) -> str:
        """Render template from call.docstring using call.kwargs as variables.

        The template comes from the method's docstring, and variables come from
        the method's parameters (stored in call.kwargs).

        Args:
            runtime: Runtime services (provides expand_variables and agent context).
            call: Current call context with docstring (template) and kwargs (variables).

        Returns:
            Rendered template string with all variables expanded.

        Raises:
            ValueError: If template references undefined variables.
            Any exception from expand_variables evaluation.

        Example:
            call = CurrentCall(
                method_name="build_prompt",
                docstring="Hello {name}, you have {count} messages.",
                kwargs={"name": "Alice", "count": 5}
            )
            result = await strategy.execute(runtime, call)
            # Result: "Hello Alice, you have 5 messages."
        """
        template = call.docstring or ""

        # Empty template - return empty string
        if not template:
            return ""

        # Build context for template expansion.
        # tc is injected AFTER kwargs so it always refers to the truncation config —
        # even if a method has a parameter named "tc", the config wins.
        context = {
            "self": runtime.agent,  # For {self.xxx} expressions
            "call": call,  # For {call.xxx} expressions
            **call.kwargs,  # Method parameters as variables
            "tc": runtime.truncation_config,  # For {original_call.format_parameters_as_code(tc=tc)}
        }

        # Expand using runtime's powerful expand_variables
        # This supports Python expressions like {len(self.tools)}, {self.doc.show()}, etc.
        return await runtime.expand_variables(
            template,
            extra_context=context,
            error_mode="raise",
        )
