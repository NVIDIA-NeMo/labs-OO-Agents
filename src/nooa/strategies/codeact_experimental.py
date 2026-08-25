# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-tool CodeAct strategy."""

import inspect
from html import escape
from types import ModuleType
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
        from nooa.config.strategy_config import CodeActConfig

        effective = config or CodeActConfig()
        # With no provider-level return_result tool, plain text must stay non-terminal.
        effective = effective.model_copy(update={"text_only_stop_behavior": "synthetic_comment"})
        super().__init__(config=effective, error_formatter=error_formatter)

    @property
    def name(self) -> str:
        return "CODEACT_EXPERIMENTAL"

    def get_block_overrides(self) -> dict[str, Any]:
        """Put the execution contract on the tool and keep only runtime context blocks."""
        overrides = super().get_block_overrides()
        overrides["strategy_prompt"] = None
        overrides["python_cell_context"] = DynamicContext(
            "strategy.python_cell_context(runtime)"
        )
        overrides["python_cell_state"] = DynamicContext(
            "strategy.python_cell_state_context(runtime)"
        )
        return overrides

    def get_static_block_keys(self) -> set[str]:
        """Exclude the removed strategy prompt from the cacheable context prefix."""
        return (super().get_static_block_keys() - {"strategy_prompt"}) | {
            "python_cell_context"
        }

    def get_block_order(self) -> list[str] | None:
        """Place live locals immediately after the stable execution context."""
        order = [key for key in (super().get_block_order() or []) if key != "strategy_prompt"]
        index = order.index("execution_context")
        return [
            *order[:index],
            "python_cell_context",
            "execution_context",
            "python_cell_state",
            *order[index + 1 :],
        ]

    async def python_cell_context(self, runtime: RuntimeServices) -> str:
        """Render static module capabilities available in generated Python cells."""
        agent_module = inspect.getmodule(type(runtime.agent))
        if agent_module is None:
            return ""

        from nooa.runtime.restrictions import is_from_blocked_module

        context = self._extract_module_context(agent_module, agent=runtime.agent)
        modules = sorted(
            (name, value.__name__)
            for name, value in context.items()
            if isinstance(value, ModuleType)
            and not is_from_blocked_module(value, self.config.restrictions.blocked_modules)
        )
        if not modules:
            return ""

        labels = ", ".join(
            f"`{name}`" if name == module_name else f"`{name}` → `{module_name}`"
            for name, module_name in modules
        )
        return "\n".join(
            (
                "## Python cell context",
                "",
                f"Module capabilities already in scope: {labels}.",
                "Use them directly; do not re-import them.",
            )
        )

    @staticmethod
    def _python_cell_state_label(value: Any, *, max_chars: int = 160) -> str:
        """Return a bounded single-line label safe inside the XML context block."""
        text = str(value).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
        if len(text) > max_chars:
            text = f"{text[: max_chars - 1]}…"
        return escape(text, quote=False)

    async def python_cell_state_context(self, runtime: RuntimeServices) -> str:
        """Render compact working state without repeating inputs or output history."""
        call = getattr(runtime, "current_call", None)
        live_locals = None if call is None else (call.execution_locals or call.session_locals)
        inputs = {} if call is None else call.bound_parameters()
        input_names = set(inputs)
        local_types = {str(name): type(value).__name__ for name, value in inputs.items()}
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
        local_items = sorted(local_types.items())
        import_items = sorted(import_names.items())

        agent = runtime.agent
        lines = ["## Python cell state"]
        shell = getattr(agent, "shell", None)
        if shell is not None and (cwd := getattr(shell, "cwd", None)) is not None:
            lines.extend(("", f"`self.shell.cwd`: {self._python_cell_state_label(cwd)}"))
        elif (cwd := getattr(agent, "cwd", None)) is not None:
            lines.extend(("", f"`self.cwd`: {self._python_cell_state_label(cwd)}"))

        persistent_vars = getattr(agent, "vars", None)
        if persistent_vars:
            count = len(persistent_vars)
            lines.append(
                f"`self.v`: {count} persistent var{'s' if count != 1 else ''} — "
                "inspect: `print(self.v.items())`; "
                "remove one: `del self.v.<name>`; clear all: `self.v.clear()`"
            )
        elif hasattr(agent, "v"):
            lines.append("`self.v`: none")

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
            lines.extend(("", f"Cell locals (includes method inputs): {items}{suffix}"))
        else:
            lines.extend(("", "Cell locals (includes method inputs): none"))
        return "\n".join(lines)

    def _build_builtins(self, runtime: RuntimeServices, call: "CurrentCall") -> dict[str, Any]:
        builtins = super()._build_builtins(runtime, call)

        def python_cell_state() -> dict[str, dict[str, str]]:
            """Return the complete name-to-type inventory for persistent and cell state."""
            persistent = getattr(runtime.agent, "vars", {})
            live = call.execution_locals or call.session_locals or {}
            inputs = call.bound_parameters()
            input_names = set(inputs)
            visible = {
                name: value
                for name, value in live.items()
                if isinstance(name, str)
                and name != "Out"
                and not name.startswith("_")
                and name not in input_names
                and not isinstance(value, type)
                and not callable(value)
            }
            return {
                "self.v": {str(name): type(value).__name__ for name, value in persistent.items()},
                "cell_locals": {
                    **{str(name): type(value).__name__ for name, value in inputs.items()},
                    **{
                        name: type(value).__name__
                        for name, value in visible.items()
                        if not isinstance(value, ModuleType)
                    },
                },
                "cell_imports": {
                    name: value.__name__
                    for name, value in visible.items()
                    if isinstance(value, ModuleType)
                },
            }

        builtins["python_cell_state"] = python_cell_state
        return builtins

    def _always_available_text(self) -> str:
        return (
            "Always available without import: `self`, `print()`, `pprint()`, `doc()`, "
            "`python_cell_state()`, `return_result()`, plus stdlib `asyncio` and `typing`."
        )

    def _python_tool_name(self) -> str:
        return "python_cell"

    def _build_execute_python_tool(self) -> Any:
        """Build the sole provider tool, including its complete operating contract."""
        tool = super()._build_execute_python_tool()
        tool.description = """Execute one cell in a persistent Python session.

Parameters are pre-loaded as locals, and names defined in one cell remain available
in later cells. Already available without import: `self`, `print()`, `pprint()`,
`doc()`, `python_cell_state()`, `return_result()`, `asyncio`, and `typing`. Use
`await` directly. This is your only provider tool: call it on every turn because
plain-text replies do not execute work or finish the task.

To finish, call `return_result(value)` inside the cell. It immediately submits a
value matching the method's annotated return type. A bare final expression does not
finish the task. In particular, a trailing string is not shown; use `print(text)`
when you want to inspect prose before submitting it.

Use Python for arithmetic, iteration, transforms, and batches rather than manually
constructing large outputs. Define reusable helpers at the top of a cell. Existing
methods on `self` may be called with `await` when async.

Restrictions (will throw):
- `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`
- `globals`, `locals`, `vars`, `asyncio.run`, `loop.run_until_complete`
- Attaching callables to the agent: `self.foo = fn`, `setattr(self, "foo", fn)`,
  `type(self).foo = fn`
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
    def _add_text_only_correction(runtime: RuntimeServices, call: "CurrentCall") -> None:
        runtime.event_manager.add(
            Error(
                content=(
                    "Your last reply was plain text with no tool call, so it was dropped. "
                    f"To finish `{call.method_name}`, call `python_cell` with "
                    "`return_result(value)` inside the cell. To continue working, "
                    "call `python_cell` with the next computation."
                )
            )
        )
