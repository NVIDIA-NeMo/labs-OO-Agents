# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-tool CodeAct V2 strategy."""

import inspect
from html import escape
from types import ModuleType
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import DynamicContext
from nooa.decorators import strategy
from nooa.events import Error
from nooa.strategies.base import RuntimeServices
from nooa.strategies.codeact import (
    CodeActStrategy,
    TextOnlyResponseAction,
    TextOnlyResponseContext,
)
from nooa.strategies.template import TemplateStrategy

if TYPE_CHECKING:
    from nooa.config.strategy_config import CodeActConfig
    from nooa.strategies.current_call import CurrentCall


class CodeActV2(CodeActStrategy):
    """Single-provider-tool CodeAct variant with in-cell completion.

    The model receives only ``python_cell`` as a provider tool. ``return_result``
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
        super().__init__(
            config=config,
            error_formatter=error_formatter,
            on_text_only=self._retry_text_only_response,
        )

    @property
    def name(self) -> str:
        return "CODEACT_V2"

    def get_block_overrides(self) -> dict[str, Any]:
        """Put the execution contract on the tool and keep only runtime context blocks."""
        overrides = super().get_block_overrides()
        overrides["strategy_prompt"] = None
        overrides["execution_context"] = None
        overrides["python_cell_context"] = DynamicContext("strategy.python_cell_context(runtime)")
        overrides["python_cell_state"] = DynamicContext(
            "strategy.python_cell_state_context(runtime)"
        )
        return overrides

    def get_static_block_keys(self) -> set[str]:
        """Exclude the removed strategy prompt from the cacheable context prefix."""
        return (super().get_static_block_keys() - {"strategy_prompt", "execution_context"}) | {
            "python_cell_context"
        }

    def get_block_order(self) -> list[str] | None:
        """Place live locals immediately after the stable execution context."""
        order = [key for key in (super().get_block_order() or []) if key != "strategy_prompt"]
        index = order.index("execution_context")
        return [
            *order[:index],
            "python_cell_context",
            "python_cell_state",
            *order[index + 1 :],
        ]

    async def python_cell_context(self, runtime: RuntimeServices) -> str:
        """Render the available namespace once, as commented Python declarations."""
        agent_module = inspect.getmodule(type(runtime.agent))
        if agent_module is None:
            return ""

        from nooa.agentdoc.visibility import iter_agent_mro_modules

        context = self._extract_module_context(agent_module, agent=runtime.agent)
        return self._render_execution_context_stub(
            context,
            {module.__name__ for module in iter_agent_mro_modules(type(runtime.agent))},
            self.config.restrictions.blocked_modules,
        )

    def _format_execution_context_stub(self, code: list[str], in_scope_only: list[str]) -> str:
        """Keep guidance and runtime names inside the same Python-style block."""
        lines = [
            "```python",
            "# Python cell context",
            "# Already in scope inside python_cell(); state persists across cells.",
            "# Use these names directly; do not re-import or re-define them.",
            "# Use doc(name) for details about any type or function.",
            "",
            *code,
        ]
        if in_scope_only:
            lines.extend(("", "# Other bound names: " + ", ".join(sorted(in_scope_only))))
        names = ", ".join(self._always_available_builtins())
        lines.extend(
            (
                "",
                "# Runtime helpers (already available):",
                f"# {names}",
                "# Standard-library modules asyncio and typing are also available.",
                "```",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _python_cell_state_label(value: Any, *, max_chars: int = 160) -> str:
        """Return a bounded single-line label safe inside the XML context block."""
        text = str(value).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
        if len(text) > max_chars:
            text = f"{text[: max_chars - 1]}…"
        return escape(text, quote=False)

    @staticmethod
    def _cell_state(call: "CurrentCall | None") -> dict[str, dict[str, str]]:
        """Use one visibility rule for compact context and the complete state builtin."""
        live_locals = None if call is None else (call.execution_locals or call.session_locals)
        inputs = {} if call is None else call.bound_parameters()
        input_names = set(inputs)
        local_types = {
            str(name): type((live_locals or {}).get(name, value)).__name__
            for name, value in inputs.items()
        }
        import_names: dict[str, str] = {}
        if live_locals:
            names = sorted(name for name in live_locals if isinstance(name, str))
            for name in names:
                value = live_locals[name]
                if name == "Out" or name.startswith("_") or name in input_names:
                    continue
                if isinstance(value, ModuleType):
                    import_names[name] = value.__name__
                    continue
                if isinstance(value, type) or callable(value):
                    continue
                local_types[name] = type(value).__name__
        return {"cell_locals": local_types, "cell_imports": import_names}

    async def python_cell_state_context(self, runtime: RuntimeServices) -> str:
        """Render compact working state without repeating inputs or output history."""
        state = self._cell_state(getattr(runtime, "current_call", None))
        local_items = sorted(state["cell_locals"].items())
        import_items = sorted(state["cell_imports"].items())

        lines = ["## Python cell state"]

        if import_items:
            visible_imports = import_items[:20]
            omitted = len(import_items) - len(visible_imports)
            suffix = (
                f' (+{omitted} more; `print(python_cell_state()["cell_imports"])`)'
                if omitted
                else ""
            )
            imports = ", ".join(
                f"{self._python_cell_state_label(name, max_chars=80)} → "
                f"{self._python_cell_state_label(module_name, max_chars=80)}"
                for name, module_name in visible_imports
            )
            lines.extend(("", f"Cell imports: {imports}{suffix}"))

        if local_items:
            visible = local_items[:20]
            suffix = (
                f" (+{len(local_items) - len(visible)} more; `print(python_cell_state())`)"
                if len(local_items) > 20
                else ""
            )
            items = ", ".join(
                f"{self._python_cell_state_label(name, max_chars=80)} "
                f"({self._python_cell_state_label(type_name, max_chars=80)})"
                for name, type_name in visible
            )
            lines.extend(
                (
                    "",
                    "Cell locals (includes method inputs; reuse unchanged values): "
                    f"{items}{suffix}",
                )
            )
        else:
            lines.extend(("", "Cell locals (includes method inputs): none"))
        return "\n".join(lines)

    def _build_builtins(self, runtime: RuntimeServices, call: "CurrentCall") -> dict[str, Any]:
        builtins = super()._build_builtins(runtime, call)

        def python_cell_state() -> dict[str, dict[str, str]]:
            """Return the complete name-to-type inventory for this call's cell state."""
            return self._cell_state(call)

        builtins["python_cell_state"] = python_cell_state
        return builtins

    def _always_available_builtins(self) -> tuple[str, ...]:
        return (*super()._always_available_builtins(), "python_cell_state()")

    def _python_tool_name(self) -> str:
        return "python_cell"

    def _build_execute_python_tool(self) -> Any:
        """Build the sole provider tool, including its complete operating contract."""
        tool = super()._build_execute_python_tool()
        tool.description = f"""Execute one cell in the current method call's Python session.

Parameters are pre-loaded as locals. Names defined in one cell remain available in
later cells of this call; reuse them instead of recreating unchanged values. The
caller controls whether locals survive after the method returns, so follow the agent's
application-specific state guidance. {self._always_available_text()} Use
`await` directly. This is your only provider tool: call it on every turn because
plain-text replies do not execute work or finish the task.

To finish, call `return_result(value)` inside the cell. It immediately submits a
value matching the method's annotated return type. A bare final expression does not
finish the task. In particular, a trailing string is not shown; use `print(text)`
when you want to inspect prose before submitting it.

Use Python for arithmetic, iteration, transforms, and batches rather than manually
constructing large outputs. Define reusable helpers at the top of a cell. Existing
methods on `self` may be called with `await` when async.

If `self` exposes delegation, inspect its documentation with `doc(self.delegate)`.
Use bounded objectives when an independent context helps; run independent work
with `asyncio.gather` and inspect each result. For single-shot extraction or
classification, use a documented `@strategy(PredictStrategy())` helper.

Restrictions (will throw):
{self._restrictions_text()}
"""
        return tool

    def _build_tools(self, return_type: Any, method_name: str) -> list[Any]:
        del return_type, method_name
        return [self._build_execute_python_tool()]

    def _supports_return_result(self) -> bool:
        return False

    def _available_tool_names(self) -> str:
        return "python_cell"

    def _python_output_value(self, result: Any) -> Any:
        if result.has_return and not result.error:
            if not result.explicit_return and isinstance(result.returned_value, str):
                return None
            return result.returned_value
        return None

    @strategy(TemplateStrategy())
    async def _tool_use_reminder(self, runtime: RuntimeServices, reason: str) -> str:
        """{reason} Call `python_cell(code)`. To finish, call `return_result(value)` inside the cell."""
        ...

    @staticmethod
    def _retry_text_only_response(context: TextOnlyResponseContext) -> TextOnlyResponseAction:
        return TextOnlyResponseAction.retry(
            Error(
                content=(
                    "Your last reply was plain text with no tool call. It was preserved, "
                    "but a bare message cannot end the turn or run code. "
                    f"To finish `{context.call.method_name}`, call `python_cell` with "
                    "`return_result(value)` inside the cell. To continue working, "
                    "call `python_cell` with the next computation."
                )
            )
        )
