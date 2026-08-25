# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-tool CodeAct strategy."""

from typing import TYPE_CHECKING, Any

from nooa.context_blocks import DynamicContext
from nooa.decorators import strategy
from nooa.events import Error
from nooa.strategies.base import RuntimeServices
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.template import TemplateStrategy

if TYPE_CHECKING:
    from nooa.config.strategy_config import CodeActConfig
    from nooa.strategies.current_call import CurrentCall


class CodeActExperimental(CodeActStrategy):
    """Single-provider-tool CodeAct variant with in-cell completion.

    The model receives only ``execute_python`` as a provider tool. ``return_result``
    remains available inside Python cells, where it completes the task. Bare
    expressions do not complete the task, and trailing strings are suppressed
    to avoid echoing prose as if it were a result.
    """

    def __init__(
        self,
        config: "CodeActConfig | None" = None,
        *,
        error_formatter: Any = None,
    ) -> None:
        from nooa.config.strategy_config import CodeActConfig

        effective = config or CodeActConfig()
        # With no provider-level return_result tool, plain text must stay non-terminal.
        effective = effective.model_copy(update={"text_only_stop_behavior": "synthetic_comment"})
        super().__init__(config=effective, error_formatter=error_formatter)

    @property
    def name(self) -> str:
        return "CODEACT_EXPERIMENTAL"

    def get_block_overrides(self) -> dict[str, Any]:
        """Keep the stable execution contract separate from changing REPL locals."""
        overrides = super().get_block_overrides()
        overrides["repl_locals"] = DynamicContext("strategy.repl_locals_context(runtime)")
        return overrides

    def get_block_order(self) -> list[str] | None:
        """Place live locals immediately after the stable execution context."""
        order = super().get_block_order() or []
        index = order.index("execution_context") + 1
        return [*order[:index], "repl_locals", *order[index:]]

    async def repl_locals_context(self, runtime: RuntimeServices) -> str:
        """Render method parameters and persistent names available in the next cell."""
        call = getattr(runtime, "current_call", None)
        local_names: set[str] = set()
        if call is not None:
            local_names.update(call.bound_parameters())
            live_locals = call.execution_locals or call.session_locals
            if live_locals:
                local_names.update(live_locals)
        names = "\n".join(f"- `{name}`" for name in sorted(local_names))
        return f"## Locals\n\nAvailable in the next cell:\n\n{names}"

    def _always_available_text(self) -> str:
        return (
            "Always available without import: `self`, `print()`, `pprint()`, `doc()`, "
            "`return_result()`, plus stdlib `asyncio` and `typing`."
        )

    def _python_tool_name(self) -> str:
        return "execute_python"

    def _build_tools(self, return_type: Any, method_name: str) -> list[Any]:
        del return_type, method_name
        return [self._build_execute_python_tool()]

    def _supports_return_result(self) -> bool:
        return False

    def _available_tool_names(self) -> str:
        return "execute_python"

    def _python_output_value(self, result: Any) -> Any:
        if result.has_return and not result.error:
            if not result.explicit_return and isinstance(result.returned_value, str):
                return None
            return result.returned_value
        return None

    @strategy(TemplateStrategy())
    async def strategy_instructions(self, runtime: RuntimeServices) -> str:
        """
        ## Strategy

        You have a persistent Python session. Parameters are pre-loaded as locals,
        and names defined in one cell remain available in later cells. Use `await`
        directly, `print`/`pprint` to inspect values, and `doc(obj)` for APIs.

        **Your only tool is `execute_python(code)`.** You must call it each turn;
        plain-text replies do not execute work or finish the task.

        To finish, call `return_result(value)` inside `execute_python`; this immediately
        submits a value matching the method's annotated return type. A bare final
        expression does not finish the task. In particular, a trailing string is not
        shown; use `print(text)` when you want to inspect prose before submitting it.

        Use Python for arithmetic, iteration, transforms, and batches rather than
        manually constructing large outputs. Define reusable helpers at the top of
        a cell. Existing methods on `self` may be called with `await` when async.

        ## Restrictions (will throw)

        - `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`
        - `globals`, `locals`, `vars`, `asyncio.run`, `loop.run_until_complete`
        - Attaching callables to the agent: `self.foo = fn`, `setattr(self, 'foo', fn)`, `type(self).foo = fn`
        """
        ...

    @strategy(TemplateStrategy())
    async def _tool_use_reminder(self, runtime: RuntimeServices, reason: str) -> str:
        """{reason} Call `execute_python(code)`. To finish, call `return_result(value)` inside the cell."""
        ...

    @staticmethod
    def _add_text_only_correction(runtime: RuntimeServices, call: "CurrentCall") -> None:
        runtime.event_manager.add(
            Error(
                content=(
                    "Your last reply was plain text with no tool call, so it was dropped. "
                    f"To finish `{call.method_name}`, call `execute_python` with "
                    "`return_result(value)` inside the cell. To continue working, "
                    "call `execute_python` with the next computation."
                )
            )
        )
