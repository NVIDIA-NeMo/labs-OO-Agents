# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes for generation strategies.

GenerationStrategy ABC - all strategies inherit from this.
RuntimeServices Protocol - what strategies can use from the runtime.
"""

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nooa.metaclass import AgentMeta

if TYPE_CHECKING:
    from nooa.config.truncation_config import TruncationConfig
    from nooa.context_blocks import DynamicContext
    from nooa.runtime.restrictions import RestrictionsConfig
    from nooa.strategies.current_call import CurrentCall


@runtime_checkable
class RuntimeServices(Protocol):
    """Protocol defining services available to strategies.

    Strategies receive this interface instead of the full runtime,
    providing a clean dependency boundary.

    Properties:
        agent: The agent instance being executed.
        event_manager: Event manager for conversation management.

    Methods:
        generate(): Build messages from context + events, call LLM.
        execute_code(): Execute validated Python code in sandbox.

    Note:
        Nested ellipsis method calls work implicitly - generated code can call
        `await self.other_method()` and lock inheritance is handled automatically.
    """

    @property
    def agent(self) -> Any:
        """The agent instance."""
        ...

    @property
    def event_manager(self) -> Any:
        """Event manager for conversation management."""
        ...

    @property
    def truncation_config(self) -> "TruncationConfig":
        """Truncation configuration for the current agent."""
        ...

    async def generate(
        self,
        *,
        tools: list[Any] | None = None,
        output_model: type | None = None,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Build messages from context + events, call LLM.

        Runtime builds messages automatically from:
        - System message: context blocks + strategy.strategy_prompt
        - Events: conversation events

        Creates an LLMOutput event and adds it to event manager.

        Args:
            tools: Optional list of tool definitions.
            output_model: Optional Pydantic model for structured output.
            **kwargs: Additional LLM options (temperature, etc.).

        Returns:
            Tuple of (LLMResponse, event_id) where:
            - LLMResponse has content, reasoning, usage
            - event_id can be used for event_manager.update() (e.g., strip reasoning)
        """
        ...

    async def execute_code(
        self,
        code: str,
        *,
        builtins: dict[str, Any] | None = None,
        validate: bool = True,
        wrap_in_function: bool = False,
        timeout: float | None = 90.0,
        tool_call_id: str | None = None,
        execution_count: int = 1,
        restrictions: "RestrictionsConfig | None" = None,
    ) -> Any:
        """Execute Python code with namespace + strategy builtins.

        Args:
            code: Python code to execute.
            builtins: Strategy-provided functions (return_result, message, etc.)
            validate: Run planning language validation first.
            wrap_in_function: Wrap code in an async function for proper await support.
            timeout: Max seconds before aborting execution (None = no limit).
            tool_call_id: LLM's tool call ID for trace correlation (OpenAI format).
            execution_count: Execution number for Jupyter-style "Cell In[N]" filename.
            restrictions: Code execution restrictions (blocked modules/calls).
                None uses defaults from RestrictionsConfig().

        Returns:
            ExecutionResult with stdout, error, defined_methods.
        """
        ...

    async def execute_nested(
        self,
        strategy: "GenerationStrategy",
        call: "CurrentCall",
    ) -> Any:
        """Execute nested strategy within current generation session.

        Use for composite strategies (Reflexion, PlanExecute) that wrap
        other strategies. Inherits parent's generation lock (no deadlock).

        The nested strategy runs within the current session:
        - Inherits lock (won't deadlock)
        - Proper hook instrumentation
        - Events are shared

        Args:
            strategy: The nested strategy to execute.
            call: The call context (can be same or modified).

        Returns:
            Result from the nested strategy.
        """
        ...

    def get_generation_id(self) -> str | None:
        """Get the current generation session ID.

        Returns the innermost generation_id from the stack, or None if
        not in a generation session. Used for correlating events with
        their generation context.

        Returns:
            Current generation ID or None if not in a generation session.

        Note:
            Never raises exceptions - returns None for empty stack.
        """
        ...

    def get_parent_generation_id(self) -> str | None:
        """Get the parent generation session ID (for nested calls).

        Returns the second-to-last generation_id from the stack, or None
        if this is the root generation or not in a generation session.

        Returns:
            Parent generation ID or None if root or not in a session.

        Note:
            Never raises exceptions - returns None for stack with <2 entries.
        """
        ...

    async def expand_variables(
        self,
        text: str,
        extra_context: dict[str, Any] | None = None,
        error_mode: str = "show",
    ) -> str:
        """Expand {expression} placeholders in text using runtime context.

        Args:
            text: Text with {expression} placeholders
            extra_context: Additional variables for evaluation
            error_mode: How to handle errors ("show", "silent", "raise")

        Returns:
            Text with expressions evaluated and substituted
        """
        ...


class GenerationStrategy(ABC, metaclass=AgentMeta):
    """Abstract base class for generation strategies with automatic method wrapping.

    Each strategy is a self-contained unit that knows how to:
    - Generate prompts for the LLM
    - Process LLM responses
    - Execute resulting code
    - Handle retries and errors

    Strategies own their configuration (max_iterations, etc.)
    instead of relying on external ExecutionConfig.

    Strategy methods with ellipsis bodies are auto-generated (same as Agent methods).
    Strategy methods are NEVER traced (no .runtime attribute).

    Example:
        @strategy(PurePythonStrategy(max_iterations=5))
        def analyze(self, data: str) -> dict:
            '''Analyze data.'''
    """

    @property
    def name(self) -> str:
        """Strategy name (e.g., 'PURE_PYTHON', 'STRUCTURED_OUTPUT').

        Default: class name. Override for custom names.
        """
        return self.__class__.__name__

    def get_block_overrides(self) -> dict[str, "str | DynamicContext | None"]:
        """Return strategy-specific context block overrides.

        Returns:
            Dict mapping block keys to values:
            - str: Static content
            - DynamicContext("expr"): Re-evaluated each turn
            - None: Remove block
            Empty dict by default (no overrides).

        Example:
            def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
                return {
                    "strategy_prompt": DynamicContext("strategy.strategy_instructions(runtime)")
                }
        """
        return {}

    def get_block_order(self) -> list[str] | None:
        """Return desired ordering of system-role block keys, or None for default.

        When a list is returned, system blocks are reordered to match:
        - Listed keys appear first, in the given order
        - Unlisted system blocks keep their relative order and follow after

        Applied after Phase 5 (scoped context) and before events (Phase 6).
        Only affects Role.SYSTEM blocks; event blocks are always last.

        Returns:
            List of block keys in desired order, or None to keep default ordering.
        """
        return None

    @property
    def traceable(self) -> bool:
        """Whether generation hooks should fire for this strategy.

        Default is True. Override to False for strategies that don't call
        the LLM (e.g. TemplateStrategy) to suppress noisy trace spans.
        """
        return True

    @property
    def requires_lock(self) -> bool:
        """Whether this strategy needs exclusive generation access.

        Default is True - most strategies need serialized access to prevent
        concurrent LLM calls from interleaving events. Override to False
        for Methodic-style stateless strategies that can run concurrently.
        """
        return True

    @abstractmethod
    async def execute(self, runtime: RuntimeServices, call: "CurrentCall") -> Any:
        """Execute the strategy to generate code for a method call.

        Args:
            runtime: RuntimeServices providing LLM, execution, and event management.
            call: CurrentCall with method details and arguments.

        Returns:
            The result of executing the generated code.

        Raises:
            GenerationError: If generation fails after max retries.
            GenerationAborted: If agent explicitly aborts.
        """
        ...

    async def call_with_instrumentation(
        self,
        callable_obj: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        runtime: RuntimeServices,
        method_name: str,
    ) -> Any:
        """Helper to call a method with before/after method_invocation hooks.

        Handles both sync and async callables with proper exception handling
        and automatic hook instrumentation.

        Args:
            callable_obj: The callable to invoke (sync or async)
            args: Positional arguments
            kwargs: Keyword arguments
            runtime: RuntimeServices providing agent context
            method_name: Name of the method for hook instrumentation

        Returns:
            Result from the callable

        Example:
            result = await self.call_with_instrumentation(
                bound_method, args, kwargs,
                runtime=runtime, method_name="process"
            )
        """
        from uuid import uuid4

        from nooa.runtime.hooks import call_after_hook, call_before_hook

        invocation_id = str(uuid4())
        hook_context = call_before_hook(
            "before_method_invocation",
            agent=runtime.agent,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            invocation_id=invocation_id,
        )

        result = None
        exception = None
        try:
            if inspect.iscoroutinefunction(callable_obj):
                result = await callable_obj(*args, **kwargs)
            else:
                result = callable_obj(*args, **kwargs)
            return result
        except Exception as e:
            exception = e
            raise
        finally:
            call_after_hook(
                "after_method_invocation",
                hook_context,
                agent=runtime.agent,
                method_name=method_name,
                result=result,
                exception=exception,
                invocation_id=invocation_id,
            )


def build_sampling_kwargs(config: "Any") -> dict[str, "Any"]:
    """Build sampling kwargs for LLM calls, excluding None values.

    Shared by PredictStrategy, CodeActStrategy, and ReflexionStrategy.
    """
    return {
        k: v
        for k, v in {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }.items()
        if v is not None
    }
