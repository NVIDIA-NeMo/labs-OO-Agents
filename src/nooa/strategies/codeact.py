# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct strategy - LLM uses execute_python tool + structured output.

Flow:
1. Build prompt with task + agent state + available methods
2. Call LLM with execute_python tool AND output_model (structured output schema)
3. If LLM calls execute_python tool → execute code, add result to events, continue loop
4. If LLM returns structured output → validate and return result
5. Handle errors and retries

This combines the flexibility of tool-based interaction with structured final outputs.
The LLM can reason in natural language and execute Python code via tool calls
until ready to return a structured result matching the method's return type.

Reference: "Executable Code Actions Elicit Better LLM Agents" (Wang et al.)
"""

import ast
import inspect
import json
import logging
import types
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    get_args,
    get_origin,
)
from uuid import uuid4

from pydantic import BaseModel, create_model
from pydantic import ValidationError as PydanticValidationError

from nooa.agentdoc._structured import format_type as _format_type
from nooa.context_blocks import DynamicContext, ResultStatus, ToolCallEvent, ToolResult
from nooa.context_blocks.exceptions import BlockSyntaxError
from nooa.decorators import strategy
from nooa.errors import GenerationError
from nooa.events import (
    AfterTurn,
    BeforeTurn,
    DebugTrace,
    Error,
    ExecutionSignal,
    PythonOutput,
    Task,
    TextOnlyReply,
)
from nooa.runtime.harness_metrics import get_harness_metrics
from nooa.runtime.hooks import call_after_hook, call_before_hook
from nooa.strategies.base import RuntimeServices, build_sampling_kwargs
from nooa.strategies.codeact_errors import format_validation_error
from nooa.strategies.composite import CompositeStrategy
from nooa.strategies.generated_code import (
    ExecutionNamespaceBuilder,
    GeneratedCodeValidator,
    HelperFunctionManager,
)
from nooa.strategies.template import TemplateStrategy
from nooa.strategy_validation import (
    InvariantError,
    run_postconditions,
    run_preconditions,
)
from nooa.unifiedllm.types import Tool, ToolCall

if TYPE_CHECKING:
    from nooa.config.strategy_config import CodeActConfig
    from nooa.errors.formatting import IPythonErrorFormatter
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)


# Small, deterministic expression subset accepted inside constructor-string
# arguments.  The values supplied to these callables have already been reduced
# to plain data by ``_safe_constructor_arg``; callbacks and object attributes
# therefore cannot cross into this compatibility path.
_SAFE_CONSTRUCTOR_CALLS = {
    "abs": abs,
    "all": all,
    "any": any,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "sum": sum,
}


class _ReturnResultSignal(ExecutionSignal):
    """Signal raised when return_result() is called from within execute_python code.

    This is an internal signal (not an error) that indicates the LLM has computed
    the final result and wants to return it inline, rather than making a separate
    return_result tool call.

    The exception-based approach allows return_result() to work anywhere in the code
    (not just at the end) and immediately stops execution, similar to a return statement.

    Inherits from ExecutionSignal so actor.py can distinguish it from actual errors.
    """

    def __init__(self, result: dict[str, Any]):
        self.result = result
        super().__init__("return_result() called")


def _as_comment(text: str) -> str:
    """Render *text* as Python comment lines (one ``#`` prefix per line)."""
    return "\n".join(f"# {line}" if line else "#" for line in text.splitlines())


def _prepend_comment(tool_calls: list[ToolCall], text: str) -> list[ToolCall]:
    """Return a copy of *tool_calls* with *text* prepended as a comment to the
    first execute_python code block.  Other tool calls are left unchanged.
    """
    result: list[ToolCall] = []
    prepended = False
    preview = _as_comment(text)
    for tc in tool_calls:
        if not prepended and tc.name == "execute_python":
            try:
                args = json.loads(tc.arguments)
                original_code = args.get("code", "")
                args["code"] = f"{preview}\n{original_code}"
                tc = replace(tc, arguments=json.dumps(args))
                prepended = True
                get_harness_metrics().content_prepended_as_comment()
            except json.JSONDecodeError:
                logger.debug(
                    "[CODEACT] _prepend_comment: skipping execute_python with unparseable arguments (tool_call_id=%s)",
                    tc.id,
                )
        result.append(tc)
    return result


@dataclass
class _ToolCallsResult:
    """Result of processing tool calls in a single turn."""

    completed: bool = False
    final_value: Any = None


@dataclass
class _TurnState:
    """Mutable state yielded by CodeActSession.turn()."""

    success: bool = False
    is_final: bool = False
    exception: str | None = None


@dataclass
class CodeActSession:
    """Tracks state for a single CodeAct generation session."""

    max_iterations: int | None
    max_retries: int
    target_method_name: str
    event_manager: Any  # EventManager reference for Out[n] access
    iteration: int = 0
    error_count: int = 0
    consecutive_text_only: int = 0
    session_locals: dict[str, Any] = field(default_factory=dict)
    out_accessor: Any = field(default=None)  # OutAccessor instance, created lazily
    sandbox_executor: Any = field(default=None)  # SandboxedExecutor when backend="sandbox"

    def __post_init__(self) -> None:
        """Initialize OutAccessor for Jupyter-style Out[n] access."""
        from nooa.runtime.out_accessor import OutAccessor

        self.out_accessor = OutAccessor(event_manager=self.event_manager)
        # Make Out available in session namespace for LLM code
        self.session_locals["Out"] = self.out_accessor

    def is_exhausted(self) -> bool:
        if self.max_iterations is not None and self.iteration >= self.max_iterations:
            return True
        return self.error_count >= self.max_retries

    def record_iteration(self) -> None:
        self.iteration += 1

    def record_error(self) -> None:
        self.error_count += 1

    def record_text_only(self) -> None:
        self.consecutive_text_only += 1

    def reset_text_only(self) -> None:
        self.consecutive_text_only = 0

    def record_output(self, execution_count: int, value: Any) -> None:
        """Record an execution output for Out[n] access."""
        if self.out_accessor is not None:
            self.out_accessor.record(execution_count, value)

    def build_failure_error(self) -> GenerationError:
        if self.error_count >= self.max_retries:
            return GenerationError(
                f"Generation failed after {self.error_count} errors (max_retries={self.max_retries}). "
                f"Unable to generate valid code for `{self.target_method_name}`."
            )
        return GenerationError(
            f"Generation failed after {self.iteration} iterations (max_iterations={self.max_iterations}). "
            f"Unable to complete `{self.target_method_name}`."
        )

    @asynccontextmanager
    async def turn(
        self,
        event_manager: Any,
        call_method_name: str,
        strategy_name: str,
        generation_id: str,
        parent_generation_id: str | None,
        turn_number: int,
    ) -> AsyncIterator[_TurnState]:
        """Emit BeforeTurn/AfterTurn around a turn body, yielding _TurnState."""
        event_manager.add(
            BeforeTurn(
                method_name=call_method_name,
                strategy=strategy_name,
                generation_id=generation_id,
                parent_generation_id=parent_generation_id,
                turn_number=turn_number,
            ),
            record=False,
        )
        state = _TurnState()
        try:
            yield state
        except Exception as exc:
            if state.exception is None:  # Don't overwrite caller-set value
                state.exception = type(exc).__name__
            raise
        finally:
            event_manager.add(
                AfterTurn(
                    method_name=call_method_name,
                    strategy=strategy_name,
                    generation_id=generation_id,
                    parent_generation_id=parent_generation_id,
                    turn_number=turn_number,
                    is_final=state.is_final,
                    success=state.success,
                    exception_type=state.exception,
                ),
                record=False,
            )


def _iter_agent_attrs(agent: Any) -> Iterator[Any]:
    """Yield non-hidden attribute values from an agent (class then instance)."""
    from nooa.agentdoc.visibility import is_hidden_field

    cls = type(agent)
    for name in dir(cls):
        if name.startswith("__"):
            continue
        try:
            val = getattr(cls, name, None)
            if val is None or callable(val):
                continue
            if is_hidden_field(cls, name):
                continue
            yield val
        except Exception:
            pass
    for name, val in getattr(agent, "__dict__", {}).items():
        if val is None or name.startswith("__"):
            continue
        if is_hidden_field(cls, name):
            continue
        yield val


class CodeActStrategy(CompositeStrategy):
    """CodeAct strategy: LLM uses execute_python tool + structured output.

    Combines the flexibility of tool-based interaction with structured final outputs.
    The LLM can reason in natural language and execute Python code via tool calls
    until ready to return a structured result matching the method's return type.

    Key differences from PurePythonStrategy:
    - LLM can reason in natural language between code executions
    - Code is executed via explicit tool calls (not raw output)
    - Final response must be structured output matching return type

    Configuration (fields of CodeActConfig, passed via config=):
        max_iterations: Maximum number of tool call iterations
        max_retries: Maximum consecutive errors before failure

    Example:
        @strategy(CodeActStrategy())
        def analyze(self, data: str) -> AnalysisResult:
            '''Analyze data and return structured results.'''
            ...

        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
        def quick_task(self, x: int) -> dict:
            '''Task with custom iteration limit.'''
            ...
    """

    def __init__(
        self,
        config: "CodeActConfig | None" = None,
        *,
        error_formatter: "IPythonErrorFormatter | None" = None,
    ):
        """Initialize CodeAct strategy.

        Args:
            config: CodeActConfig with iteration limits, timeouts, and sampling params.
                    Defaults to CodeActConfig() with standard defaults.
            error_formatter: Custom error formatter for LLM feedback. Defaults to IPythonErrorFormatter.
                Any object with a `format(error, code) -> str` method works.

        Note:
            Prefill is always enabled and uses InspectInputsPrefill internally.
        """
        from nooa.config.strategy_config import CodeActConfig as _CC

        self.config = config or _CC()
        self.error_formatter = error_formatter

    def _build_sampling_kwargs(self) -> dict[str, Any]:
        """Build sampling kwargs for llm calls, excluding None values."""
        return build_sampling_kwargs(self.config)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "CODEACT"

    def _get_truncation_config(self, runtime: RuntimeServices) -> Any:
        """Get truncation config from runtime's agent.

        Args:
            runtime: RuntimeServices instance with agent reference

        Returns:
            TruncationConfig from agent, or default if not available
        """
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        return getattr(runtime.agent, "_truncation", DEFAULT_TRUNCATION_CONFIG)

    def get_block_overrides(self) -> dict[str, "str | DynamicContext | None"]:
        overrides: dict[str, str | DynamicContext | None] = {
            "strategy_prompt": DynamicContext("strategy.strategy_instructions(runtime)"),
            "execution_context": DynamicContext("strategy.execution_context(runtime)"),
        }
        # Tell the agent about the active sandbox guardrails (only when enabled).
        if self.config.execution_backend == "sandbox" and self.config.sandbox.context_block:
            overrides["sandbox"] = DynamicContext("strategy.sandbox_context(runtime)")
        return overrides

    def get_static_block_keys(self) -> set[str]:
        """Block keys from get_block_overrides() that should be in the static partition."""
        # The sandbox constraints are fixed for the session -> static (cacheable) prefix.
        return {"execution_context", "strategy_prompt", "sandbox"}

    def get_block_order(self) -> list[str] | None:
        """Place doc(self) after execution_context so the LLM sees instructions first."""
        return [
            "system_prompt",
            "strategy_prompt",
            "sandbox",
            "execution_context",
            "self",
        ]

    async def sandbox_context(self, runtime: RuntimeServices) -> str:
        """Render the active sandbox constraints as an agent-facing context block."""
        from nooa.runtime.sandbox.context_block import render_sandbox_block

        return render_sandbox_block(self.config.sandbox, cell_timeout=self.config.cell_timeout)

    async def execution_context(self, runtime: RuntimeServices) -> str:
        """Generate execution context block showing available imports and types.

        This is a separate context block that appears in the system prompt,
        documenting what symbols are available in execute_python().
        """
        agent_module = inspect.getmodule(type(runtime.agent))
        if not agent_module:
            return """## Execution Context

Standard Python builtins and agent instance (`self`) are available."""

        # Extract context to see what's available
        context = self._extract_module_context(agent_module, agent=runtime.agent)

        # Filter out blocked modules so the LLM doesn't see unavailable symbols
        from nooa.agentdoc.visibility import iter_agent_mro_modules

        blocked = self.config.restrictions.blocked_modules

        # Names of all modules across the agent's MRO that contribute symbols —
        # used to classify a symbol as "defined here" vs "imported". This mirrors
        # the merged namespace built in _extract_module_context so the displayed
        # types/functions reflect parent-module symbols too.
        own_module_names = {m.__name__ for m in iter_agent_mro_modules(type(runtime.agent))}

        return self._render_execution_context_stub(context, own_module_names, blocked)

    def _render_execution_context_stub(
        self, context: dict[str, Any], own_module_names: set[str], blocked: Any
    ) -> str:
        """Render the execution scope as a Python stub (`.pyi`-style).

        Symbols are rendered as the top of a module would be: ``import`` lines
        for dependencies, and ``class X: ...`` / ``def f(...) -> T: ...`` stubs
        for things defined in the agent's own (or an ancestor) module. Functions
        are rendered uniformly — whether a function is LLM-backed (a ``@strategy``
        standalone) or plain Python is an implementation detail; the ``async``
        keyword in the signature already carries the calling contract.
        """
        import sys

        from nooa.runtime.restrictions import is_from_blocked_module

        module_lines: list[str] = []  # `import x` / `import x as y`
        from_imports: dict[str, set[str]] = {}  # module -> {names} for imported symbols
        in_scope_only: list[str] = []  # names with no faithful import path
        defined_classes: list[tuple[str, Any]] = []  # (name, obj) defined in agent module
        functions: list[tuple[str, Any]] = []  # (name, obj) — all callables, unified

        def record_import(obj: Any, name: str) -> None:
            """Add a `from <module> import <name>` if it's faithfully importable.

            We only emit an import we know would actually resolve: the object
            must be reachable as ``<module>.<name>``. Prefer the shortest public
            path (``from pydantic import BaseModel`` over ``pydantic.main``) by
            walking the dotted prefixes of ``__module__``. Type aliases and oddly
            re-exported names (whose ``__module__`` doesn't actually expose them)
            fall back to a plain in-scope listing rather than a fabricated,
            unrunnable import.
            """
            mod = getattr(obj, "__module__", None)
            if not mod:
                in_scope_only.append(name)
                return
            parts_ = mod.split(".")
            for i in range(1, len(parts_) + 1):
                candidate = ".".join(parts_[:i])
                if getattr(sys.modules.get(candidate), name, None) is obj:
                    from_imports.setdefault(candidate, set()).add(name)
                    return
            in_scope_only.append(name)

        for name, obj in context.items():
            if is_from_blocked_module(obj, blocked):
                continue
            if isinstance(obj, types.ModuleType):
                actual_name = getattr(obj, "__name__", name)
                module_lines.append(
                    f"import {actual_name} as {name}" if actual_name != name else f"import {name}"
                )
            elif callable(obj):
                # Functions/classes defined in the agent's own or an ancestor
                # module → stubs. `@strategy` generation standalones carry
                # _needs_generation and belong with the agent's code even if
                # their wrapper's __module__ differs, so they always render as
                # stubs (async signature included). Everything else imported →
                # an import line.
                obj_module = getattr(obj, "__module__", None)
                defined_here = obj_module in own_module_names or getattr(
                    obj, "_needs_generation", False
                )
                if isinstance(obj, type):
                    if defined_here:
                        defined_classes.append((name, obj))
                    else:
                        record_import(obj, name)
                elif defined_here:
                    functions.append((name, obj))
                else:
                    record_import(obj, name)
            else:
                in_scope_only.append(name)

        code: list[str] = []

        for line in sorted(module_lines):
            code.append(line)
        for mod in sorted(from_imports):
            names = ", ".join(sorted(from_imports[mod]))
            code.append(f"from {mod} import {names}")

        if defined_classes:
            if code:
                code.append("")
            for name, _ in sorted(defined_classes, key=lambda p: p[0]):
                code.append(f"class {name}: ...")

        if functions:
            if code:
                code.append("")
            code.append(self._render_function_specs(functions))

        parts = [
            "## Execution Context",
            "",
            "These names are already in scope inside `execute_python()` (state "
            "persists across cells) — call them, don't re-import or re-define. "
            "Use `doc(name)` to inspect any type or function in detail.",
            "",
            "```python",
            "\n".join(code) if code else "# (no module-level symbols)",
            "```",
        ]

        if in_scope_only:
            parts.append(f"Also in scope: {', '.join(sorted(in_scope_only))}.")

        parts.append(
            "Always available without import: `self`, `print()`, `pprint()`, `doc()`, "
            "`return_result()`, plus stdlib `asyncio` and `typing`."
        )

        return "\n".join(parts)

    @staticmethod
    def _render_function_specs(functions: list[tuple[str, Any]]) -> str:
        """Render module-level functions as full signatures + docstrings.

        Reuses ``agentdoc.doc()`` (the same machinery that renders method specs
        in ``doc(type(self))``) so module-level functions — including
        ``@strategy`` standalone generation functions — get an ``async`` marker,
        a typed signature, and their docstring instead of a bare name.

        ``inline_depth=0`` keeps the output to signatures + docstrings without
        re-expanding referenced type bodies; those types already appear under
        **Available types** / **Imported items**, so expanding them here would
        only duplicate content.

        Falls back to a bare comma-joined name list if ``doc()`` raises, so a
        rendering failure can never break prompt construction.
        """
        ordered = sorted(functions, key=lambda pair: pair[0])
        try:
            from nooa.agentdoc import doc

            return doc(*[obj for _, obj in ordered], inline_depth=0)
        except Exception:
            return ", ".join(name for name, _ in ordered)

    @strategy(TemplateStrategy())
    async def strategy_instructions(self, runtime: RuntimeServices) -> str:
        """
        ## Strategy

        Jupyter-like Python session. Parameters pre-loaded as locals; state persists across cells. Use `await` directly, `print`/`pprint` to debug, `doc(obj)` to inspect types. You MUST call a tool each turn — **plain-text responses do NOT end the session**. To finish, call `return_result(value)`. Repeated text-only responses will abort the run with an error.

        **Your two tools:**
        - `execute_python(code)` — run a code cell
        - `return_result(value)` — submit your final answer (also callable from inside `execute_python`)

        ## When to use which tool

        Use `return_result(...)` directly for simple answers determinable from the inputs alone (yes/no, one field, a single lookup).

        Use `execute_python(...)` for lists/batches, arithmetic, multi-step computation, transforms, or iteration. Always iterate in code — never construct large arrays by hand.

        For language tasks (classification, extraction, interpretation), use LLM reasoning — answer directly via `return_result`, or delegate to a `@strategy(PredictStrategy())` standalone function (see below). Don't keyword-match or regex.

        ## Returning computed results

        After computing in code, call `return_result(variable)` **from within** `execute_python()`. This passes the variable directly. Do NOT re-type computed values in a separate `return_result` tool call.

        ## Helpers

        Define helpers at the top of the cell and call them by name. Existing methods on `self` are usable via `await self.method(...)`. Helpers persist as REPL locals across cells in this session.

        ```python
        def normalize(x):
            return x.strip().lower()

        cleaned = [normalize(v) for v in values]
        ```

        ## Fan-out generation

        For per-item LLM work over a list, decorate a standalone async function with `@strategy(PredictStrategy())` and an ellipsis body. `asyncio.gather` runs the calls in parallel.

        ```python
        @strategy(PredictStrategy())
        async def detect_language(message: str) -> str:
            \"\"\"Return the ISO 639-1 language code for {{message}} (e.g. 'en', 'fr', 'de', 'ja').\"\"\"
            ...

        codes = await asyncio.gather(*(detect_language(m) for m in messages))
        return_result(codes)
        ```

        For iterative sub-tasks that need code execution, use `@strategy(CodeActStrategy())`. The sub-task must be strictly simpler than the current call to avoid infinite recursion.

        ## Restrictions (will throw)

        - `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`
        - `globals`, `locals`, `vars`, `asyncio.run`, `loop.run_until_complete`
        - Attaching callables to the agent: `self.foo = fn`, `setattr(self, 'foo', fn)`, `type(self).foo = fn`
        """
        ...

    @strategy(TemplateStrategy())
    async def _tool_use_reminder(self, runtime: RuntimeServices, reason: str) -> str:
        """{reason} Use `execute_python(code)` to run code, or `return_result(...)` to submit your answer."""
        ...

    @staticmethod
    def _add_text_only_correction(runtime: RuntimeServices, call: "CurrentCall") -> None:
        """Add a model-visible correction after a text-only turn.

        Mirrors PredictStrategy's validation-retry feedback (``Error``,
        ``Role.USER``): instead of silently dropping the turn, tell the model
        what it did and what to do, so it self-corrects on the next turn. The
        consecutive-text-only backstop still aborts after repeated text-only replies.
        """
        runtime.event_manager.add(
            Error(
                content=(
                    f"Your last reply was plain text with no tool call, so it was "
                    f"dropped — a bare message cannot end the turn or run code. "
                    f"To finish `{call.method_name}`, call `return_result(value)`. "
                    f"To do more work, call `execute_python(code)`. "
                    f"Re-issue your response now as one of those tool calls."
                )
            )
        )

    @staticmethod
    def _mark_text_only_recovered(runtime: RuntimeServices) -> None:
        """Flip the most recent unrecovered ``TextOnlyReply`` to recovered=True.

        Called when a real tool call lands after one or more text-only replies —
        the correction worked. Lets capture/replay distinguish benign,
        self-corrected replies from ones that needed an abort.
        """
        # Flip every unrecovered text-only reply since the last real progress,
        # not just the most recent — multiple consecutive ones can precede a
        # single tool call, and all were rescued by it. Stop at the first
        # already-recovered one (older runs are already resolved).
        for tag in reversed(runtime.event_manager.keys()):
            event = runtime.event_manager.get(tag)
            if not isinstance(event, TextOnlyReply):
                continue
            if event.recovered:
                break
            runtime.event_manager.update(tag, recovered=True)

    @strategy(TemplateStrategy())
    async def _build_task_message(
        self, runtime: RuntimeServices, original_call: "CurrentCall"
    ) -> str:
        """
        ## Task: {original_call.method_name}

        {original_call.docstring}

        You are executing `{original_call.method_name}` — code runs in the Execution Context above. Calling `self.{original_call.method_name}(...)` would recurse.
        """
        ...

    # Keys added by the session itself that should not leak to the caller.
    _SESSION_INTERNAL_KEYS = frozenset({"Out", "__repl_captured_locals__"})

    @staticmethod
    def _sync_session_locals(call: "CurrentCall", session: "CodeActSession") -> None:
        """Write back session_locals to caller's dict, filtering session internals."""
        if call.session_locals is None:
            return
        filtered = {
            k: v
            for k, v in session.session_locals.items()
            if k not in CodeActStrategy._SESSION_INTERNAL_KEYS
        }
        call.session_locals.clear()
        call.session_locals.update(filtered)

    def _create_sandbox_executor(
        self, runtime: RuntimeServices, call: "CurrentCall", builtins: dict[str, Any]
    ) -> Any:
        """Build the per-session sandboxed executor (fails closed on unavailable guards)."""
        from nooa.runtime.sandbox.executor import SandboxedExecutor

        # ``_call`` is referenced by prefill/introspection cells (doc(_call.return_type));
        # inject it alongside the framework builtins so the worker namespace has it.
        framework_builtins = {**builtins, "_call": call}
        return SandboxedExecutor(
            runtime.agent,
            self.config.sandbox,
            cell_timeout=self.config.cell_timeout,
            framework_builtins=framework_builtins,
            restrictions=self.config.restrictions,
        )

    async def _close_sandbox(self, session: "CodeActSession") -> None:
        """Tear down the session's sandbox worker, if any."""
        executor = getattr(session, "sandbox_executor", None)
        if executor is None:
            return
        session.sandbox_executor = None
        try:
            await executor.aclose()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the result
            logger.debug(f"[CODEACT] sandbox teardown error (ignored): {exc}")

    async def execute(self, runtime: RuntimeServices, call: "CurrentCall") -> Any:
        """Execute CodeAct strategy with two-tool approach.

        Uses two tools:
        - execute_python(code): Run Python code for computation
        - return_result(...): Return the final structured answer

        Args:
            runtime: RuntimeServices providing LLM, execution, and event management.
            call: CurrentCall with method details and arguments.

        Returns:
            Validated structured data matching return type.

        Raises:
            GenerationError: If generation fails after max retries/iterations.
        """
        # Guarantee the sandbox worker is torn down on EVERY exit — including the
        # GenerationError paths inside the loop — not just the success returns.
        session_holder: dict[str, Any] = {}
        try:
            return await self._run_generation(runtime, call, session_holder)
        finally:
            session = session_holder.get("session")
            if session is not None:
                await self._close_sandbox(session)

    async def _run_generation(
        self,
        runtime: RuntimeServices,
        call: "CurrentCall",
        session_holder: dict[str, Any],
    ) -> Any:
        """Body of :meth:`execute`; ``session_holder`` receives the session for teardown."""
        # Return type is pre-resolved by _execute_with_generation (handles PEP 563).
        return_type = call.return_type
        if return_type is None:
            raise GenerationError(
                f"Method `{call.method_name}` has no return type annotation. "
                f"CodeActStrategy requires a return type (Pydantic model or basic type)."
            )

        # Initialize session
        session = CodeActSession(
            max_iterations=self.config.max_iterations,
            max_retries=self.config.max_retries,
            target_method_name=call.method_name,
            event_manager=runtime.event_manager,
        )
        # Publish to the caller so execute()'s finally can tear down the worker.
        session_holder["session"] = session

        # Seed session_locals from caller-provided dict (persistent stack)
        if call.session_locals is not None:
            session.session_locals.update(call.session_locals)

        # Build builtins for code execution
        _init_hm = get_harness_metrics()
        with _init_hm.timer("time_session_init"):
            builtins = self._build_builtins(runtime, call)

            # Sandbox backend: create the per-session guarded executor. Fails
            # closed here (before any cell) if a requested guardrail can't be
            # enforced; the worker itself is forked lazily on the first cell.
            if self.config.execution_backend == "sandbox":
                session.sandbox_executor = self._create_sandbox_executor(runtime, call, builtins)

            # Build both tools
            execute_python_tool = self._build_execute_python_tool()
            return_result_tool = self._build_return_result_tool(return_type, call.method_name)
            tools = [execute_python_tool, return_result_tool]

            # Use the task event's tag as the call ID so the LLM sees a stable reference.
            # _build_task_message reads only method_name/docstring, so it's safe to build
            # before the event lands and assign the returned tag back to call.id.
            task_content = await self._build_task_message(runtime, original_call=call)
            tag = runtime.event_manager.add(Task(prompt=task_content))
            object.__setattr__(call, "id", tag)
            # Method-local preconditions run before generation and fail fast
            # (raise to abort the call); see nooa.strategy_validation.
            run_preconditions(runtime.agent, call, self.config.preconditions)

        logger.info(
            f"[CODEACT] Starting session for {call.method_name}: "
            f"max_iterations={self.config.max_iterations}, max_retries={self.config.max_retries}"
        )

        # Run prefill (always enabled - inspects inputs with truncation)
        try:
            with _init_hm.timer("time_prefill"):
                await self._run_prefill(runtime, call, builtins, session)
        except Exception as e:
            logger.warning(f"[CODEACT] Prefill error (continuing): {e}")
            runtime.event_manager.add(Error(content=f"Prefill error: {e}"))

        # Get generation_id for turn events
        generation_id = runtime.get_generation_id()
        if generation_id is None:
            raise RuntimeError(
                "get_generation_id() returned None - strategy execution requires a generation context. "
                "This indicates the runtime was not properly initialized via _in_generation_session()."
            )
        parent_generation_id = runtime.get_parent_generation_id()
        turn_number = 0

        # CodeAct loop
        while not session.is_exhausted():
            turn_number += 1

            async with session.turn(
                runtime.event_manager,
                call.method_name,
                self.name,
                generation_id,
                parent_generation_id,
                turn_number,
            ) as turn_state:
                # Use "auto" for tool_choice - "required" can cause 500 errors
                # on some models (e.g., Nemotron) when there's conversation events
                tool_choice = "auto"

                logger.debug(
                    f"[CODEACT] Loop iteration: iter={session.iteration}/{session.max_iterations}, "
                    f"err={session.error_count}/{session.max_retries}, tool_choice={tool_choice}"
                )

                response = None
                event_id = None
                try:
                    # generate() rebuilds the conversation from event_manager each call,
                    # so events added in prior iterations change what the LLM sees.
                    response, event_id = await runtime.generate(
                        tools=tools,
                        tool_choice=tool_choice,
                        **self._build_sampling_kwargs(),
                    )
                except BlockSyntaxError as e:
                    self._handle_block_syntax_error(e, session, runtime)
                    continue

                except Exception as e:
                    # LLM API errors (rate limits, connection errors, timeouts, etc.)
                    # Note: unifiedllm already has retry logic with exponential backoff
                    # for 429, 500, 502, 503, 504 errors. This handles cases where
                    # all retries are exhausted or the error is not retryable.
                    session.record_error()
                    error_name = type(e).__name__
                    cause_parts = []
                    exc: BaseException | None = e
                    seen_ids: set[int] = set()
                    while exc is not None and id(exc) not in seen_ids:
                        seen_ids.add(id(exc))
                        cause_parts.append(f"{type(exc).__name__}: {exc}")
                        exc = exc.__cause__ or exc.__context__
                    cause_chain = " <- ".join(cause_parts)
                    error_msg = (
                        f"LLM API error (attempt {session.error_count}/{session.max_retries}): "
                        f"{cause_chain}"
                    )
                    get_harness_metrics().llm_api_error(cause_chain[:500])
                    runtime.event_manager.add(Error(content=error_msg))
                    logger.warning(
                        f"[CODEACT] LLM API error (iter={session.iteration}, err={session.error_count}): "
                        f"{cause_chain}",
                        exc_info=True,
                    )
                    turn_state.exception = error_name
                    if session.is_exhausted():
                        turn_state.is_final = True
                        raise GenerationError(
                            f"LLM API error after {session.max_retries} retries. "
                            f"Original error: {error_name}: {e}"
                        ) from e

                # Skip rest of turn if LLM call failed
                if response is None:
                    continue

                # ── Post-response cleanup (CodeAct) ──────────────────────
                # Intercept point: strategy-specific response transforms.
                # Handles text-only→synthetic, comment prepend, tool call
                # translation. Consider making extensible in the future.
                if response.finish_reason == "tool_calls" and response.tool_calls:
                    tool_calls = response.tool_calls
                    # If the LLM also emitted message content alongside the tool
                    # call(s), preserve it by prepending it as a comment at the
                    # top of the first execute_python code block.
                    if response.content:
                        content = response.content
                        text = (
                            content.model_dump_json()
                            if isinstance(content, BaseModel)
                            else str(content)
                        )
                        if text.strip():
                            tool_calls = _prepend_comment(tool_calls, text)
                    # A real tool call counts as progress: reset the consecutive
                    # text-only guard (issue 185) before executing, so a single
                    # exec mid-stream rescues the run from accidental drift.
                    if session.consecutive_text_only > 0:
                        self._mark_text_only_recovered(runtime)
                    session.reset_text_only()
                    result = await self._process_tool_calls(
                        tool_calls,
                        runtime,
                        builtins,
                        session,
                        call,
                        return_type,
                        event_id or "",
                    )
                    if result.completed:
                        turn_state.success = True
                        turn_state.is_final = True
                        self._sync_session_locals(call, session)
                        return result.final_value
                    continue

                # ── Text-only response (no tool call) ──────────────────────
                # Normalize content for both branches below.
                _raw_content = response.content
                _text = (
                    _raw_content.model_dump_json()
                    if isinstance(_raw_content, BaseModel)
                    else str(_raw_content)
                    if _raw_content
                    else ""
                )
                _has_text = bool(_text.strip())

                # Route A: "return_result" mode — treat stop as a done signal
                # and route through return_result() validation. Handles both
                # stop+content and stop+no-content in one branch.
                if (
                    response.finish_reason == "stop"
                    and self.config.text_only_stop_behavior == "return_result"
                    and (_has_text or not _raw_content)
                ):
                    session.record_iteration()
                    # Capture the drift faithfully for /bug + replay (recorded but
                    # Role.METADATA, so it never reaches the model). Replaces the
                    # old lossy DebugTrace; preserves the verbatim content (even
                    # when empty) before the offending event is removed.
                    drift_tag = runtime.event_manager.add(
                        TextOnlyReply(
                            content=_text,
                            finish_reason=str(response.finish_reason),
                            route="return_result",
                            consecutive_text_only=session.consecutive_text_only + 1,
                        )
                    )
                    runtime.event_manager.remove(event_id)
                    synthetic_id = f"synthetic_{uuid4().hex[:8]}"
                    result_value = _text if _has_text else None
                    synthetic_tool_call = ToolCall(
                        id=synthetic_id,
                        name="return_result",
                        arguments=json.dumps({"result": result_value}),
                    )
                    get_harness_metrics().stop_to_return_result(result_value)
                    logger.info(
                        f"[CODEACT] finish_reason='stop' "
                        f"({'content=' + str(len(_text)) + ' chars' if _has_text else 'no content'}) "
                        f"→ synthetic return_result(). Routing through validation."
                    )
                    result = await self._process_tool_calls(
                        [synthetic_tool_call],
                        runtime,
                        builtins,
                        session,
                        call,
                        return_type,
                        event_id or "",
                    )
                    if result.completed:
                        # Recovered via the synthetic return_result — the drift was
                        # benign. Mark it so capture/replay can distinguish recovered
                        # drifts from ones that needed a correction.
                        runtime.event_manager.update(drift_tag, recovered=True)
                        turn_state.success = True
                        turn_state.is_final = True
                        self._sync_session_locals(call, session)
                        return result.final_value
                    # Validation failed — give the model a visible correction (the
                    # PredictStrategy pattern: a Role.USER event it sees on the next
                    # turn) instead of silently dropping the turn, then continue.
                    # The abort below is only a backstop for repeated non-compliance.
                    session.record_text_only()
                    self._add_text_only_correction(runtime, call)
                    max_text_only = self.config.max_consecutive_text_only
                    if max_text_only > 0 and session.consecutive_text_only >= max_text_only:
                        get_harness_metrics().text_only_loop_abort()
                        preview = _text if _has_text else "(empty)"
                        turn_state.is_final = True
                        raise GenerationError(
                            f"CodeAct aborted: LLM returned plain text without a tool call "
                            f"{session.consecutive_text_only} times in a row "
                            f"(max_consecutive_text_only={max_text_only}) for "
                            f"`{call.method_name}`. The agent likely thinks it is done — "
                            f"it must call `return_result(...)` to finish. "
                            f"Last text: {preview!r}"
                        )

                    continue

                # Route B: "synthetic_comment" mode — convert text to a no-op
                # execute_python comment that preserves content in traces.
                elif _has_text:
                    session.record_iteration()
                    # Capture the drift faithfully for /bug + replay (recorded but
                    # Role.METADATA, never shown to the model). Replaces the old
                    # lossy DebugTrace.
                    runtime.event_manager.add(
                        TextOnlyReply(
                            content=_text,
                            finish_reason=str(response.finish_reason),
                            route="synthetic_comment",
                            consecutive_text_only=session.consecutive_text_only + 1,
                        )
                    )
                    runtime.event_manager.remove(event_id)
                    synthetic_id = f"synthetic_{uuid4().hex[:8]}"
                    runtime.event_manager.add(
                        ToolCallEvent(
                            tool_call_id=synthetic_id,
                            name="execute_python",
                            arguments={"code": _as_comment(_text)},
                            result=ToolResult(
                                tool_call_id=synthetic_id,
                                content="status: commentary only — task is NOT finished. You must call return_result() to complete.",
                                result_status=ResultStatus.COMPLETE,
                            ),
                            metadata={"synthetic": True, "synthetic_type": "text_response"},
                        )
                    )
                    runtime.event_manager.add(
                        PythonOutput(
                            tool_call_id=synthetic_id,
                            execution_count=session.iteration or 1,
                            execution_status=ResultStatus.COMPLETE,
                            metadata={"synthetic": True, "synthetic_type": "text_response"},
                        )
                    )
                    get_harness_metrics().text_to_synthetic()
                    logger.debug(
                        f"[CODEACT] Text-only response ({len(_text)} chars) "
                        f"converted to synthetic comment."
                    )
                    # No extra Error correction here: Route B already injects a
                    # synthetic execute_python comment whose ToolResult tells
                    # the model the task isn't finished. Adding a "your reply had no
                    # tool call" Error would contradict that synthetic tool call and
                    # confuse the model. The backstop below still aborts on repeated
                    # non-compliance.
                    session.record_text_only()
                    max_text_only = self.config.max_consecutive_text_only
                    if max_text_only > 0 and session.consecutive_text_only >= max_text_only:
                        get_harness_metrics().text_only_loop_abort()
                        preview = _text
                        turn_state.is_final = True
                        raise GenerationError(
                            f"CodeAct aborted: LLM returned plain text without a tool call "
                            f"{session.consecutive_text_only} times in a row "
                            f"(max_consecutive_text_only={max_text_only}) for "
                            f"`{call.method_name}`. The agent likely thinks it is done — "
                            f"it must call `return_result(...)` to finish. "
                            f"Last text: {preview!r}"
                        )
                    continue

                # Empty response - error
                get_harness_metrics().empty_response()
                session.record_error()
                # Capture raw LLM response for debugging before removing the event
                _debug_parts = [
                    f"finish_reason={response.finish_reason!r}",
                    f"content={response.content!r}",
                    f"tool_calls={response.tool_calls!r}",
                ]
                raw = getattr(response, "raw_response", None)
                if raw is not None:
                    output = getattr(raw, "output", None)
                    if output is not None:
                        _debug_parts.append(f"raw_response.output={output!r}")
                runtime.event_manager.add(
                    DebugTrace(content=f"Empty response: {'; '.join(_debug_parts)}")
                )
                # Remove the empty assistant event - APIs reject empty content
                runtime.event_manager.remove(event_id)
                # Reasoning models that exhaust max_tokens produce empty output.
                # Retrying won't help — abort immediately with an actionable message.
                if response.finish_reason == "length":
                    turn_state.is_final = True
                    raise GenerationError(
                        "Empty response: the model used all available output tokens "
                        "on reasoning and had none left for a tool call. "
                        "This typically means `max_tokens` is too low for a "
                        "reasoning model (e.g. GPT-5.5, o-series). "
                        "Increase `max_tokens` in the model config "
                        "(16384+ recommended for reasoning models)."
                    )
                feedback = await self._tool_use_reminder(runtime, reason="Empty response received.")
                runtime.event_manager.add(Error(content=feedback))

        # Loop exhausted without success
        runtime.event_manager.add(
            AfterTurn(
                method_name=call.method_name,
                strategy=self.name,
                generation_id=generation_id,
                parent_generation_id=parent_generation_id,
                turn_number=turn_number,
                is_final=True,
                success=False,
                exception_type="GenerationError",
            ),
            record=False,
        )
        self._sync_session_locals(call, session)
        raise session.build_failure_error()

    def _translate_tool_call_to_code(
        self,
        tool_name: str,
        args: dict[str, Any],
        builtins: dict[str, Any],
        session: CodeActSession,
        runtime: RuntimeServices,
    ) -> str | None:
        """Translate an unknown tool call into execute_python code.

        Weaker models sometimes call agent methods directly as tool calls
        instead of using execute_python(). This translates those into
        equivalent execute_python code, teaching the model the correct
        pattern for subsequent turns.

        Returns Python code string if translatable, None otherwise.
        """
        # Determine how to call the function
        call_target = None
        is_async = False

        # Check if it's a method on the agent (accessible via self.)
        agent = runtime.agent
        if (
            not tool_name.startswith("_")
            and hasattr(agent, tool_name)
            and callable(getattr(agent, tool_name))
        ):
            call_target = f"self.{tool_name}"
            is_async = inspect.iscoroutinefunction(getattr(agent, tool_name))
        # Check if it's a known builtin (module-level function, etc.)
        elif tool_name in builtins and callable(builtins.get(tool_name)):
            call_target = tool_name
            is_async = inspect.iscoroutinefunction(builtins[tool_name])
        # Check session locals (previously defined functions)
        elif tool_name in session.session_locals and callable(
            session.session_locals.get(tool_name)
        ):
            call_target = tool_name
            is_async = inspect.iscoroutinefunction(session.session_locals[tool_name])

        if call_target is None:
            return None

        # Build argument string from the parsed args dict
        arg_parts = []
        for k, v in args.items():
            arg_parts.append(f"{k}={v!r}")
        args_str = ", ".join(arg_parts)

        # Generate code — use await for async methods to pass CodeAct validation
        call_expr = f"await {call_target}({args_str})" if is_async else f"{call_target}({args_str})"
        code = f"result = {call_expr}\nprint(result)"
        return code

    async def _process_tool_calls(
        self,
        tool_calls: list[Any],
        runtime: RuntimeServices,
        builtins: dict[str, Any],
        session: CodeActSession,
        call: "CurrentCall",
        return_type: Any,
        event_id: str,
    ) -> _ToolCallsResult:
        """Process tool calls from a single LLM turn.

        Executes tool calls sequentially, stopping at the first error.
        Returns a _ToolCallsResult indicating whether the task completed.
        """
        # Handle tool calls - process ALL tool calls sequentially
        # Some LLMs return multiple tool calls in one response even when
        # parallel_tool_calls=false. We execute them in order, with each
        # cell's output available to subsequent cells via session_locals.
        session.record_iteration()

        # Remove the empty LLMOutput that runtime.generate() created
        # and replace it with a proper ToolCallEvent that includes tool_calls
        runtime.event_manager.remove(event_id)

        num_tool_calls = len(tool_calls)
        if num_tool_calls > 1:
            logger.debug(f"[CODEACT] Processing {num_tool_calls} tool calls sequentially")

        # Process each tool call in order, stopping at the first error.
        # If one cell fails, subsequent cells likely depend on its output
        # and would cascade into confusing errors.
        for tool_call in tool_calls:
            # Parse arguments
            try:
                args = json.loads(tool_call.arguments)
            except json.JSONDecodeError as e:
                session.record_error()
                runtime.event_manager.add(
                    Error(content=f"Invalid arguments for tool `{tool_call.name}`: {e}")
                )
                # Stop processing remaining tool calls - let LLM fix this first
                return _ToolCallsResult()

            # Add ToolCallEvent to record the tool call (result will be nested later)
            tool_call_event_id = runtime.event_manager.add(
                ToolCallEvent(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    arguments=args,
                    result=None,  # Will be updated after execution
                )
            )

            # Handle based on tool name
            if tool_call.name == "execute_python":
                # Execute Python code
                result = await self._handle_execute_python(
                    runtime,
                    tool_call,
                    args,
                    builtins,
                    session,
                    call,
                    return_type,
                    tool_call_event_id=tool_call_event_id,
                )
                if result is None or getattr(result, "error", None) is not None:
                    # Error occurred (already handled) or code execution failed -
                    # stop processing remaining tool calls
                    return _ToolCallsResult()

                # Check if return_result() was called inline
                if isinstance(result, tuple) and result[0] == "TASK_COMPLETE":
                    # Task completed via inline return_result()
                    return _ToolCallsResult(completed=True, final_value=result[1])

            elif tool_call.name == "return_result":
                # Return the final result
                try:
                    validated, error_msg = self._handle_return_result(
                        runtime, tool_call, args, return_type, session, call
                    )
                except GenerationError:
                    # _handle_return_result raises GenerationError when validation
                    # fails AND session is exhausted. Ensure the ToolCallEvent has
                    # an error result before re-raising so the event is never left
                    # with result=None in the DB (which corrupts the next session's
                    # context render).
                    runtime.event_manager.update(
                        tool_call_event_id,
                        result=ToolResult(
                            tool_call_id=tool_call.id,
                            content="return_result validation failed (session exhausted).",
                            result_status=ResultStatus.ERROR,
                        ),
                    )
                    raise
                if error_msg is None:
                    # Rewrite the tool_call arguments to show correct syntax
                    # when coercion changed the value (teaches model the right format).
                    corrected_args = self._corrected_return_args(validated, args)
                    runtime.event_manager.update(
                        tool_call_event_id,
                        arguments=corrected_args,
                        result=ToolResult(
                            tool_call_id=tool_call.id,
                            content="Result accepted.",
                            result_status=ResultStatus.COMPLETE,
                        ),
                    )
                    logger.info("[CODEACT] Task completed successfully via return_result")
                    return _ToolCallsResult(completed=True, final_value=validated)
                # Validation error - update with error result
                runtime.event_manager.update(
                    tool_call_event_id,
                    result=ToolResult(
                        tool_call_id=tool_call.id,
                        content=f"Invalid result: {error_msg}\n"
                        f"Please call return_result again with valid arguments. "
                        f"Tip: if you computed the result in execute_python(), you can call "
                        f"return_result(variable) from within the code instead.",
                        result_status=ResultStatus.ERROR,
                    ),
                )
                # Stop processing remaining tool calls
                return _ToolCallsResult()

            else:
                # Unknown tool — attempt to translate to execute_python.
                # Weaker models sometimes call agent methods directly as tool
                # calls instead of wrapping them in execute_python().
                translated_code = (
                    self._translate_tool_call_to_code(
                        tool_call.name, args, builtins, session, runtime
                    )
                    if self.config.translate_tool_calls
                    else None
                )
                if translated_code is not None:
                    get_harness_metrics().tool_call_translated(tool_call.name)
                    logger.debug(
                        f"[CODEACT] Translated tool call '{tool_call.name}' -> execute_python"
                    )
                    # Update ToolCallEvent to reflect the translation
                    runtime.event_manager.update(
                        tool_call_event_id,
                        name="execute_python",
                        arguments={"code": translated_code},
                    )
                    translated_args = {"code": translated_code}
                    result = await self._handle_execute_python(
                        runtime,
                        tool_call,
                        translated_args,
                        builtins,
                        session,
                        call,
                        return_type,
                        tool_call_event_id=tool_call_event_id,
                    )
                    if result is None or getattr(result, "error", None) is not None:
                        return _ToolCallsResult()
                    if isinstance(result, tuple) and result[0] == "TASK_COMPLETE":
                        return _ToolCallsResult(completed=True, final_value=result[1])
                else:
                    # Truly unknown tool — not translatable
                    session.record_error()
                    runtime.event_manager.update(
                        tool_call_event_id,
                        result=ToolResult(
                            tool_call_id=tool_call.id,
                            content=f"Unknown tool `{tool_call.name}`. "
                            f"Available tools: execute_python, return_result",
                            result_status=ResultStatus.ERROR,
                        ),
                    )
                    # Stop processing remaining tool calls
                    return _ToolCallsResult()

        # All tool calls processed without completion or error-break
        return _ToolCallsResult()

    @staticmethod
    def _handle_block_syntax_error(
        e: BlockSyntaxError,
        session: "CodeActSession",
        runtime: RuntimeServices,
    ) -> None:
        """Handle a BlockSyntaxError raised during generate().

        The LLM created a context block with invalid Python syntax.
        This is recoverable — remove the bad block, add error feedback
        so the LLM can fix it, and record an iteration (not an error
        retry, since retrying with the bad block would just fail again).
        """
        get_harness_metrics().block_syntax_error(f"block '{e.key}': {e.original_error}")
        logger.warning(f"[CODEACT] Block syntax error in block '{e.key}': {e.original_error}")

        # Remove the bad block so subsequent attempts can proceed
        try:
            _ctx = getattr(runtime, "context", None)
            if _ctx is not None:
                _ctx.remove(e.key)
                logger.debug(f"[CODEACT] Removed bad block '{e.key}' from context")
        except Exception as remove_err:
            logger.debug(f"[CODEACT] Could not remove block '{e.key}': {remove_err}")

        # Add helpful error feedback for the LLM
        _field = getattr(e, "field", "expr")
        error_msg = (
            f"Error: Block '{e.key}' has invalid Python syntax.\n"
            f"The {_field} parameter must be a valid Python expression.\n"
            f"  Invalid {_field}: {e.expr[:100]}{'...' if len(e.expr) > 100 else ''}\n"
            f"  Error: {e.original_error}\n\n"
            f"To fix this, use context.set() with value= for static content:\n"
            f'  context.set("{e.key}", value="your content here")\n'
            f"Or use a valid Python expression:\n"
            f'  context.set("{e.key}", expr="self.my_variable")'
        )
        runtime.event_manager.add(Error(content=error_msg))

        # Record as an iteration (not error) since this is fixable feedback
        session.record_iteration()

    async def _handle_execute_python(
        self,
        runtime: RuntimeServices,
        tool_call: Any,
        args: dict[str, Any],
        builtins: dict[str, Any],
        session: CodeActSession,
        call: Any,
        return_type: Any,
        tool_call_event_id: str,
    ) -> Any | None:
        """Handle execute_python tool call with deferred output pattern.

        The deferred output pattern ensures tool result is nested in ToolCallEvent
        even when nested agent calls occur during execution:

        1. Update ToolCallEvent.result with "status: executing" immediately
        2. Execute code (nested agent events may be added here)
        3. Update ToolCallEvent.result status to "complete" or "error"
        4. Add PythonOutput with actual output content

        Returns the execution result, a tuple ("TASK_COMPLETE", result) if return_result()
        was called inline, or None if an error occurred.
        """
        method_name = call.method_name
        code = args.get("code", "")

        # Strip markdown fences early — validator and helper binding below
        # run ast.parse() which fails on fenced input. (runtime.execute_code
        # also strips fences as a safety net for non-CodeAct code paths.)
        from nooa.runtime.response_cleanup import strip_code_fences

        code, fence_token = strip_code_fences(code)
        if fence_token:
            get_harness_metrics().fence_removal(fence_token)

        if not code.strip():
            session.record_error()
            # Update ToolCallEvent with error result
            runtime.event_manager.update(
                tool_call_event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content="status: error",
                    result_status=ResultStatus.ERROR,
                ),
            )
            runtime.event_manager.add(
                PythonOutput(
                    tool_call_id=tool_call.id,
                    execution_count=session.iteration,
                    stdout="",
                    stderr="Execution error: empty code provided.",
                    value=None,
                    explicit_return=False,
                    execution_status=ResultStatus.ERROR,
                )
            )
            return None

        # Update ToolCallEvent with executing status immediately - BEFORE code execution
        # This ensures result is nested even if nested agents add events
        runtime.event_manager.update(
            tool_call_event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content="status: executing",
                result_status=ResultStatus.COMPLETE,  # Will update to error if needed
            ),
        )

        # Execute the code (pass tool_call.id for trace correlation)
        # Nested agent calls may add their events to the event manager during this execution
        with get_harness_metrics().timer("time_code_execution"):
            result = await self._execute_code(
                runtime, code, builtins, session, method_name, tool_call_id=tool_call.id
            )

        # Determine final status
        final_status = ResultStatus.ERROR if result.error else ResultStatus.COMPLETE

        # Track execute_python outcome
        hm = get_harness_metrics()
        hm.exec_python(success=not result.error)
        if result.error:
            hm.exec_error(
                type(result.error).__name__, str(result.error)[:500], session.iteration, code[:200]
            )

        # Update ToolCallEvent with final status
        runtime.event_manager.update(
            tool_call_event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=f"status: {final_status.value}",
                result_status=final_status,
            ),
        )

        # Merge captured locals into session for REPL-style persistence
        if result.captured_locals:
            session.session_locals.update(result.captured_locals)
            logger.debug(f"[CODEACT] Captured locals: {list(result.captured_locals.keys())}")

        # Re-inject return type names into session_locals if the model clobbered them.
        # Defense-in-depth: even if the E501 validator misses an assignment that
        # shadows the return type, this ensures they're always fresh for the next cell.
        # Handles complex types like Optional[Answer], list[Answer] by extracting
        # all protected names and re-injecting from the execution builtins.
        if return_type is not None:
            from nooa.runtime.code_validator import _collect_type_names

            protected_names = _collect_type_names(return_type, builtins)
            for name in protected_names:
                original = builtins.get(name)
                if original is not None and session.session_locals.get(name) is not original:
                    session.session_locals[name] = original

        # Check if return_result() was called inline (signals task completion)
        if result.signal and isinstance(result.signal, _ReturnResultSignal):
            logger.debug("[CODEACT] Detected inline return_result() call")

            # Validate the return_result (called inline within execute_python, not as separate tool)
            try:
                validated, validation_error = self._handle_return_result(
                    runtime,
                    tool_call,
                    result.signal.result,  # Extract the result dict from the signal
                    return_type,
                    session,
                    call,
                )
            except GenerationError:
                # _handle_return_result raises GenerationError when validation
                # fails AND session is exhausted. Ensure the ToolCallEvent has
                # an error result before re-raising so the event is never left
                # with result=None in the DB.
                runtime.event_manager.update(
                    tool_call_event_id,
                    result=ToolResult(
                        tool_call_id=tool_call.id,
                        content="return_result validation failed (session exhausted).",
                        result_status=ResultStatus.ERROR,
                    ),
                )
                raise

            if validation_error:
                # Update result to reflect validation failure
                runtime.event_manager.update(
                    tool_call_event_id,
                    result=ToolResult(
                        tool_call_id=tool_call.id,
                        content="status: error",
                        result_status=ResultStatus.ERROR,
                    ),
                )

            error_text = ""
            if result.error:
                line_offset = getattr(result, "wrapper_line_offset", 0)
                error_text = self._format_error(result.error, line_offset=line_offset)
            stderr = result.stderr
            if error_text:
                stderr = (
                    f"{stderr}\nExecution error:\n{error_text}"
                    if stderr
                    else f"Execution error:\n{error_text}"
                )
            if validation_error:
                stderr = (
                    f"{stderr}\nreturn_result validation error: {validation_error}"
                    if stderr
                    else f"return_result validation error: {validation_error}"
                )

            runtime.event_manager.add(
                PythonOutput(
                    tool_call_id=tool_call.id,
                    execution_count=session.iteration,
                    stdout=result.stdout,
                    stderr=stderr,
                    value=result.returned_value if result.has_return else None,
                    explicit_return=result.explicit_return,
                    execution_status=ResultStatus.ERROR if validation_error else final_status,
                    images=result.images,
                )
            )

            if validation_error is None:
                # Success! Return special tuple to signal completion.
                # Emit a synthetic return_result ToolCallEvent so the final
                # answer appears in the trajectory (otherwise the inline
                # path leaves no trace of the value).  Mirrors PredictStrategy's
                # _replace_with_tool_call pattern in predict.py.
                self._emit_synthetic_inline_return(runtime, validated)
                logger.info("[CODEACT] Task completed successfully via inline return_result()")
                return ("TASK_COMPLETE", validated)

            # Validation failed - error included in PythonOutput
            return None

        # Check if code used an explicit Python `return` statement with a value that matches
        # expected type. This handles the case where the LLM uses `return {...}` instead of
        # `return_result(...)`. We only auto-complete for EXPLICIT returns, not bare expressions.
        if result.explicit_return and result.has_return and not result.error:
            logger.debug(
                "[CODEACT] Detected explicit return statement - attempting auto-completion"
            )
            try:
                success, validated = self._try_validate_return_value(
                    result.returned_value,
                    return_type,
                    method_name,
                )
                if success:
                    # Add execution output event
                    runtime.event_manager.add(
                        PythonOutput(
                            tool_call_id=tool_call.id,
                            execution_count=session.iteration,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            value=result.returned_value,
                            explicit_return=result.explicit_return,
                            execution_status=ResultStatus.COMPLETE,
                            images=result.images,
                        )
                    )
                    get_harness_metrics().explicit_return_completed()
                    # Emit a synthetic return_result ToolCallEvent so the
                    # final answer is visible in the trajectory; see the
                    # inline-return_result path above.
                    self._emit_synthetic_inline_return(runtime, validated)
                    logger.info("[CODEACT] Auto-completed task from explicit return statement")
                    return ("TASK_COMPLETE", validated)
                # Validation failed - continue with normal flow
                logger.debug(
                    "[CODEACT] Auto-completion validation failed (type mismatch), continuing loop"
                )
            except Exception as e:
                logger.debug(f"[CODEACT] Auto-completion validation error: {e}, continuing loop")

        # Note: Bare expressions (IPython-style) are shown as "Out[n]:" but do NOT auto-complete.
        # Only explicit `return x` statements can auto-complete the task.

        # Format error if present
        error_text = ""
        if result.error:
            line_offset = getattr(result, "wrapper_line_offset", 0)
            if (
                isinstance(result.error, PydanticValidationError)
                and result.returned_value is not None
            ):
                error_text = format_validation_error(
                    result.error, return_type, result.returned_value, runtime.truncation_config
                )
            else:
                error_text = self._format_error(result.error, line_offset=line_offset)

        # Add PythonOutput with actual output and value
        runtime.event_manager.add(
            PythonOutput(
                tool_call_id=tool_call.id,
                execution_count=session.iteration,
                stdout=result.stdout,
                stderr=result.stderr,
                error=error_text,
                value=result.returned_value if result.has_return and not result.error else None,
                explicit_return=result.explicit_return if result.has_return else False,
                execution_status=final_status,
                images=result.images,
            )
        )

        logger.debug(
            f"[CODEACT] execute_python complete. "
            f"stdout_len={len(result.stdout)}, "
            f"has_return={result.has_return}, "
            f"error={result.error is not None}"
        )

        return result

    def _handle_return_result(
        self,
        runtime: RuntimeServices,
        tool_call: Any,
        args: dict[str, Any],
        return_type: Any,
        session: CodeActSession,
        call: Any,
    ) -> tuple[Any, str | None]:
        """Validate return_result arguments and return the result.

        Pure validation - does NOT update ToolCallEvent.result. Caller is responsible
        for updating the ToolCallEvent with the result when return_result is called
        as a tool (vs inline within execute_python code).

        Args should be: {"result": <value matching return_type>}

        Returns:
            Tuple of (validated_result, error_message):
            - (value, None) on success
            - (None, "error message") on validation failure
        """
        method_name = call.method_name
        # Generate execution ID and get current generation_id for correlation
        execution_id = str(uuid4())
        current_generation_id = runtime.get_generation_id()

        # Call before hook
        hook_context = call_before_hook(
            "before_tool_execution",
            agent=runtime.agent,
            tool_name="return_result",
            arguments=args,
            execution_id=execution_id,
            generation_id=current_generation_id,
        )

        validated = None
        exception = None
        error_msg = None
        normalized_args: dict[str, Any] = {}

        try:
            # Special case: None return type
            # Accept empty args, result=None, or no result key - all valid for -> None methods
            # Note: `-> None` annotation gives `None` (the value), not `type(None)` (NoneType)
            if return_type is None or return_type is type(None):
                # For None return type, just verify result is None (or missing)
                result_value = args.get("result", None)
                if result_value is not None:
                    error_msg = (
                        f"Method returns None but got result={result_value!r}. "
                        f"Call return_result() with no arguments."
                    )
                    return (None, error_msg)
                run_postconditions(runtime.agent, None, call, self.config.postconditions)
                return (None, None)  # Success - return None

            # Normalize args to always have "result" key
            # ── Return result normalization ──────────────────────────
            # Intercept point: normalizes return_result args before
            # Pydantic validation. Handles args wrapping, GPT-4o
            # double-quote fix, variable ref resolution, JSON parsing.
            # Consider making extensible in the future.

            # Be flexible: accept both return_result(result=...) and return_result(field1=..., field2=...)
            if "result" not in args and len(args) > 0:
                # LLM passed direct fields (e.g., sum=100, mean=20)
                # Wrap them as the result value
                get_harness_metrics().args_normalized()
                normalized_args: dict[str, Any] = {"result": args}
            else:
                # Already has "result" key, use as-is
                normalized_args = args

            # Fix GPT-4o-mini double-quoting bug: LLM sometimes wraps string in extra quotes
            # with escaped newlines. Detect and unwrap: "\"code\\n\"" -> "code\n"
            # Safe because legitimate code never starts/ends with " AND contains \\n
            if "result" in normalized_args and isinstance(normalized_args["result"], str):
                result_val = normalized_args["result"]
                if result_val.startswith('"') and result_val.endswith('"') and "\\n" in result_val:
                    # Unwrap the extra quotes and decode escaped characters
                    get_harness_metrics().gpt4o_double_quote_fix(result_val)
                    normalized_args["result"] = result_val[1:-1].encode().decode("unicode_escape")

            # Create wrapper model: class ReturnResultModel(BaseModel): result: T
            # For non-Pydantic types (e.g. pd.DataFrame), falls back to Any in the model
            ReturnResultModel, is_pydantic_validated = self._create_return_model(
                return_type, method_name
            )

            # Handle case where LLM passes a string instead of actual object.
            # Don't transform if the expected return type IS str — the string should stay as-is.
            if (
                "result" in normalized_args
                and isinstance(normalized_args["result"], str)
                and return_type is not str
            ):
                result_str = normalized_args["result"]
                # Variable reference resolution: the LLM sometimes calls return_result
                # as a tool with a variable name (e.g. {"result": "results"}) instead of
                # calling return_result(results) from within execute_python code.
                # Resolve the variable from the session namespace if it exists.
                if result_str.isidentifier() and result_str in session.session_locals:
                    get_harness_metrics().variable_ref_resolved(result_str)
                    logger.debug(
                        "[CODEACT] Resolved variable reference %r from session locals "
                        "in return_result tool call",
                        result_str,
                    )
                    normalized_args["result"] = session.session_locals[result_str]
                else:
                    parsed = self._maybe_parse_json_string(result_str)
                    if parsed is not result_str:
                        normalized_args["result"] = parsed
                    else:
                        # Constructor-call coercion: if the string looks like a Python
                        # constructor call (e.g. "Answer(answer=1, reason='...')"), eval it
                        # in the session namespace where the type is available.
                        coerced = self._maybe_eval_constructor_string(
                            result_str, return_type, session
                        )
                        normalized_args["result"] = coerced

            # Validate using Pydantic
            validated_model = ReturnResultModel(**normalized_args)
            validated = getattr(validated_model, "result")  # noqa: B009

            # For non-Pydantic types, Pydantic accepted Any — do isinstance check.
            # Unwrap Annotated[T, ...] first; isinstance() can't take Annotated as 2nd arg.
            if not is_pydantic_validated and validated is not None:
                base_type, _ = self._extract_annotated_description(return_type)
                if not isinstance(validated, base_type):
                    type_name = getattr(base_type, "__name__", str(base_type))
                    raise TypeError(
                        f"Expected an instance of {type_name}, "
                        f"but got {type(validated).__name__}.\n"
                        f"Hint: Use execute_python() to construct the {type_name} object, "
                        f"then call return_result(variable) from within the code."
                    )

            # Method-local postconditions: deterministic invariants declared on
            # the strategy config that Pydantic return-type validation can't
            # express (e.g. live agent state). An ``InvariantError`` is caught by
            # the handler below and routed as model-correctable retry feedback;
            # any other exception surfaces as an infrastructure error.
            run_postconditions(runtime.agent, validated, call, self.config.postconditions)

            return (validated, None)

        except (
            PydanticValidationError,
            InvariantError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as e:
            exception = e
            get_harness_metrics().validation_error(type(e).__name__, str(e)[:500])
            session.record_error()
            # Pass actual value for better "Got: {...}" error messages
            actual_value = normalized_args.get("result") if normalized_args else None
            error_msg = format_validation_error(
                e, return_type, actual_value, runtime.truncation_config
            )
            logger.debug(f"[CODEACT] return_result validation failed: {e}")

            if session.is_exhausted():
                raise GenerationError(
                    f"return_result validation failed after {session.max_retries} attempts.\n"
                    f"Last error:\n{error_msg}"
                ) from e

            return (None, error_msg)

        finally:
            # Call after hook
            call_after_hook(
                "after_tool_execution",
                hook_context,
                agent=runtime.agent,
                tool_name="return_result",
                arguments=args,
                result=validated,
                exception=exception,
                execution_id=execution_id,
            )

    def _emit_synthetic_inline_return(self, runtime: RuntimeServices, value: Any) -> None:
        """Emit a synthetic ``return_result`` :class:`ToolCallEvent` for the
        inline-completion path.

        CodeAct supports two completion patterns:

        - **Explicit tool_call**: the LLM emits ``return_result(...)`` as
          a separate tool_call alongside ``execute_python``. CodeAct's
          dispatch loop handles this and emits its own
          :class:`ToolCallEvent` (see ``_handle_return_result``).
        - **Inline**: the LLM emits only ``execute_python(...)`` and the
          generated code itself calls ``return_result(...)`` (which
          raises :class:`TaskCompleteSignal`), OR the code uses an
          explicit ``return X`` statement that auto-completes.

        In the inline path, the LLM never issued a ``return_result``
        tool_call from its perspective, so the final answer would
        otherwise be invisible in any trajectory built from the
        framework's event stream (e.g. the ATIF exporter). To preserve
        observability, we emit a synthetic ``ToolCallEvent`` with the
        captured value.

        Mirrors :meth:`PredictStrategy._replace_with_tool_call` in
        ``predict.py``. The event carries
        ``metadata.synthetic = True`` and
        ``metadata.synthetic_type = "codeact_inline_return"`` so
        downstream consumers can distinguish framework-emitted markers
        from genuine LLM tool_calls if desired.
        """
        tool_call_id = f"codeact_inline_{uuid4().hex[:8]}"
        # _jsonable() recurses unguarded — a self-referential dict/list
        # or any other serialization failure here would turn a valid
        # completion into a failed run. The synthetic event is purely
        # observability metadata, so fail-open: fall back to a str()
        # repr on any exception (including RecursionError).
        try:
            serialized_value: Any = self._jsonable(value)
        except Exception:  # noqa: BLE001
            logger.debug(
                "[CODEACT] Failed to JSON-coerce inline return value; falling back to str() repr",
                exc_info=True,
            )
            serialized_value = str(value)
        runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call_id,
                name="return_result",
                arguments={"result": serialized_value},
                result=ToolResult(
                    tool_call_id=tool_call_id,
                    content="Result accepted (inline).",
                    result_status=ResultStatus.COMPLETE,
                ),
                metadata={
                    "synthetic": True,
                    "synthetic_type": "codeact_inline_return",
                },
            )
        )

    def _jsonable(self, value: Any) -> Any:
        """JSON-coerce a Python value for the synthetic-return ``arguments``.

        Mirrors :meth:`PredictStrategy._jsonable`. Handles Pydantic
        models, dicts, sets, lists/tuples, then falls back to
        ``json.dumps`` round-trip; non-serialisable values become their
        ``str()`` repr so the synthetic event never raises.
        """
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, set):
            converted = [self._jsonable(v) for v in value]
            return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _maybe_parse_json_string(self, value: Any) -> Any:
        """Parse a JSON or Python literal string, otherwise return as-is.

        Some LLMs return structured data as a string instead of actual object values.
        This handles both JSON syntax and Python literal syntax (e.g., lists with
        single quotes like "['a', 'b']").
        """
        if not isinstance(value, str):
            return value

        # Check if it looks like a structured value (starts with { or [)
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            # Try JSON first (more strict)
            try:
                parsed = json.loads(value)
                get_harness_metrics().json_auto_parsed("json")
                return parsed
            except json.JSONDecodeError:
                pass

            # Try Python literal syntax (handles single quotes, etc.)
            try:
                parsed = ast.literal_eval(value)
                get_harness_metrics().json_auto_parsed("literal_eval")
                return parsed
            except (ValueError, SyntaxError):
                pass  # Not valid Python literal, return as-is

        return value

    def _corrected_return_args(self, validated: Any, original_args: dict) -> dict:
        """Return corrected tool_call arguments showing correct JSON syntax.

        When coercion transformed the result (e.g. from a constructor-call
        string to an actual object), rewrite the arguments dict so the
        model's conversation history shows what it should have passed.
        """
        try:
            if isinstance(validated, BaseModel):
                corrected_result = validated.model_dump(mode="json")
            elif validated is None:
                return original_args
            else:
                # For non-Pydantic types, try JSON round-trip
                json.dumps(validated)
                corrected_result = validated
        except (TypeError, ValueError):
            return original_args

        return {"result": corrected_result}

    @staticmethod
    def _safe_constructor_arg(node: ast.AST, session_locals: dict[str, Any]) -> Any:
        """Decode one constructor argument without executing Python code.

        Literal constants/containers are decoded directly. Bare names retain
        pass-by-reference semantics by resolving directly from the REPL namespace.
        Containers are rebuilt recursively so they can contain those references
        without evaluating Python expressions. A short allowlist covers common
        deterministic expressions emitted by models, such as ``min(...)``; those
        helpers operate only on copied plain data. Everything else, including
        attributes and arbitrary callables, is rejected.
        """

        if isinstance(node, ast.Name):
            if node.id not in session_locals:
                raise ValueError(f"unknown constructor argument name: {node.id}")
            # Dictionary lookup does not invoke any behavior on the referenced
            # object. The trusted return type receives the same object identity.
            return session_locals[node.id]

        if isinstance(node, ast.Constant):
            return CodeActStrategy._copy_constructor_data(node.value)

        if isinstance(node, ast.List):
            return [
                CodeActStrategy._safe_constructor_arg(item, session_locals) for item in node.elts
            ]

        if isinstance(node, ast.Tuple):
            return tuple(
                CodeActStrategy._safe_constructor_arg(item, session_locals) for item in node.elts
            )

        if isinstance(node, ast.Set):
            return {
                CodeActStrategy._copy_constructor_data(
                    CodeActStrategy._safe_constructor_arg(item, session_locals)
                )
                for item in node.elts
            }

        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for key_node, value_node in zip(node.keys, node.values, strict=True):
                if key_node is None:
                    expanded = CodeActStrategy._safe_constructor_arg(value_node, session_locals)
                    if type(expanded) is not dict:
                        raise ValueError("literal ** expansion must be an exact dict")
                    for key, item in expanded.items():
                        result[CodeActStrategy._copy_constructor_data(key)] = item
                else:
                    key = CodeActStrategy._copy_constructor_data(
                        CodeActStrategy._safe_constructor_arg(key_node, session_locals)
                    )
                    item = CodeActStrategy._safe_constructor_arg(value_node, session_locals)
                    result[key] = item
            return result

        # literal_eval supports numeric signs and complex-number literals without
        # admitting general arithmetic or overloaded session objects.
        if isinstance(node, (ast.UnaryOp, ast.BinOp)):
            try:
                return CodeActStrategy._copy_constructor_data(ast.literal_eval(node))
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
                raise ValueError("unsupported constructor numeric expression") from exc

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _SAFE_CONSTRUCTOR_CALLS.get(node.func.id)
            if fn is None:
                raise ValueError(f"unsupported constructor argument call: {node.func.id}")
            args = [
                CodeActStrategy._copy_constructor_data(
                    CodeActStrategy._safe_constructor_arg(arg, session_locals)
                )
                for arg in node.args
            ]
            kwargs: dict[str, Any] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    expanded = CodeActStrategy._copy_constructor_data(
                        CodeActStrategy._safe_constructor_arg(keyword.value, session_locals)
                    )
                    if type(expanded) is not dict or not all(type(key) is str for key in expanded):
                        raise ValueError("safe call **kwargs must be a string-keyed dict")
                    kwargs.update(expanded)
                else:
                    kwargs[keyword.arg] = CodeActStrategy._copy_constructor_data(
                        CodeActStrategy._safe_constructor_arg(keyword.value, session_locals)
                    )
            result = fn(*args, **kwargs)
            return CodeActStrategy._copy_constructor_data(result)

        raise ValueError(f"unsupported constructor argument syntax: {type(node).__name__}")

    @staticmethod
    def _copy_constructor_data(value: Any) -> Any:
        """Copy an exact built-in scalar/container value or reject it.

        Used only before invoking a fixed deterministic helper. Exact-type checks
        prevent arbitrary references from reaching those calls through conversion
        or iteration hooks.
        """

        if value is None or type(value) in (bool, int, float, complex, str, bytes):
            return value
        if type(value) is list:
            return [CodeActStrategy._copy_constructor_data(item) for item in value]
        if type(value) is tuple:
            return tuple(CodeActStrategy._copy_constructor_data(item) for item in value)
        if type(value) is set:
            return {CodeActStrategy._copy_constructor_data(item) for item in value}
        if type(value) is frozenset:
            return frozenset(CodeActStrategy._copy_constructor_data(item) for item in value)
        if type(value) is dict:
            return {
                CodeActStrategy._copy_constructor_data(key): CodeActStrategy._copy_constructor_data(
                    item
                )
                for key, item in value.items()
            }
        raise ValueError(f"constructor argument {type(value).__name__} is not plain data")

    def _maybe_eval_constructor_string(self, value: str, return_type: Any, session: Any) -> Any:
        """Parse a string that looks like a Python constructor call.

        Detects patterns like 'ClassName(field=value, ...)' where ClassName
        matches the expected return type. Only plain literal/data arguments and
        a small deterministic expression subset are accepted. Returns the
        constructed object on success, or the original string on failure.

        This handles a common LLM failure mode where the model calls
        return_result as a tool with the constructor as a string argument
        instead of calling return_result(ClassName(...)) from within
        execute_python.
        """
        stripped = value.strip()

        # Must look like a constructor call: Identifier(...)
        # Quick check before parsing
        paren_idx = stripped.find("(")
        if paren_idx <= 0 or not stripped.endswith(")"):
            return value

        candidate_name = stripped[:paren_idx].strip()
        if not candidate_name.isidentifier():
            return value

        # Only the trusted return type may be constructed. Preserve safe aliases
        # without allowing an arbitrary session-local factory to become code.
        type_name = getattr(return_type, "__name__", None)
        if not isinstance(return_type, type):
            return value
        if (
            candidate_name != type_name
            and session.session_locals.get(candidate_name) is not return_type
        ):
            return value

        try:
            parsed = ast.parse(stripped, mode="eval").body
            if not isinstance(parsed, ast.Call) or not isinstance(parsed.func, ast.Name):
                return value
            if parsed.func.id != candidate_name:
                return value

            args: list[Any] = []
            for arg in parsed.args:
                if isinstance(arg, ast.Starred):
                    expanded = self._safe_constructor_arg(arg.value, session.session_locals)
                    if type(expanded) not in (list, tuple):
                        raise ValueError("constructor *args must be an exact list or tuple")
                    args.extend(expanded)
                else:
                    args.append(self._safe_constructor_arg(arg, session.session_locals))
            kwargs: dict[str, Any] = {}
            for keyword in parsed.keywords:
                if keyword.arg is None:
                    expanded = self._safe_constructor_arg(keyword.value, session.session_locals)
                    if type(expanded) is not dict or not all(type(key) is str for key in expanded):
                        raise ValueError("constructor **kwargs must be a string-keyed dict")
                    kwargs.update(expanded)
                else:
                    kwargs[keyword.arg] = self._safe_constructor_arg(
                        keyword.value, session.session_locals
                    )

            result = return_type(*args, **kwargs)
            get_harness_metrics().constructor_string_coerced(candidate_name)
            logger.debug(
                "[CODEACT] Coerced constructor-call string %r into %s instance",
                stripped[:80],
                type(result).__name__,
            )
            return result
        except Exception:
            logger.debug(
                "[CODEACT] Failed to parse constructor string %r, returning as-is",
                stripped[:80],
            )
            return value

    def _try_validate_return_value(
        self,
        value: Any,
        return_type: Any,
        method_name: str,
    ) -> tuple[bool, Any]:
        """Try to validate a value against the expected return type.

        Returns:
            Tuple of (success: bool, validated_value: Any)
            - (True, value) if validation succeeded
            - (False, None) if validation failed

        Does NOT add any events to the event manager - used for silent validation checks.
        """
        try:
            # Special case: None return type
            # Note: `-> None` annotation gives `None` (the value), not `type(None)` (NoneType)
            if return_type is None or return_type is type(None):
                if value is None:
                    return (True, None)
                return (False, None)

            # Create wrapper model: class ReturnResultModel(BaseModel): result: T
            # For non-Pydantic types, falls back to Any in the model
            ReturnResultModel, is_pydantic_validated = self._create_return_model(
                return_type, method_name
            )

            # Handle case where value is a JSON string
            # BUT: Don't parse if the expected return type IS str - the string should stay as-is
            if isinstance(value, str) and return_type is not str:
                value = self._maybe_parse_json_string(value)

            # Validate using Pydantic
            validated_model = ReturnResultModel(result=value)
            validated = getattr(validated_model, "result")  # noqa: B009

            # For non-Pydantic types, do isinstance check instead.
            # Unwrap Annotated[T, ...] first; isinstance() can't take Annotated as 2nd arg.
            if not is_pydantic_validated and validated is not None:
                base_type, _ = self._extract_annotated_description(return_type)
                if not isinstance(validated, base_type):
                    return (False, None)

            return (True, validated)

        except (PydanticValidationError, ValueError, TypeError, json.JSONDecodeError):
            return (False, None)

    def _build_execute_python_tool(self) -> Tool:
        """Build the execute_python tool definition."""

        def execute_python(code: str) -> str:
            """Execute Python code in the agent's environment.

            Variables and helper functions persist across calls.
            Access `self` (agent instance), task parameters, and all
            pre-imported modules (see Execution Context).

            Args:
                code: Python code to execute.

            Returns:
                Execution result including stdout, errors, and any returned values.
            """
            return ""

        return Tool(
            name="execute_python",
            description=(
                "Execute Python code in the agent's environment. "
                "Variables persist across calls. "
                "Access `self`, task parameters, and all pre-imported modules "
                "(see Execution Context). "
                "Use `return <value>` to capture results."
            ),
            callable=execute_python,
            parameters_model=None,  # Auto-generate from callable signature
        )

    def _is_pydantic_compatible(self, return_type: Any) -> bool:
        """Check if a type can be represented in the return_result tool's JSON schema.

        Probes both create_model() AND model_json_schema(). No cache — the probe is
        microseconds and avoids id() GC reuse bugs and global mutable state.

        The result decides whether the type goes into the tool's JSON schema or falls
        back to ``Any``. Probing only create_model() is insufficient: a model can accept
        a field yet be unable to *serialize* it to JSON Schema — e.g. a Pydantic model
        with a ``pd.DataFrame`` / ``np.ndarray`` field (``arbitrary_types_allowed``).
        That only fails at model_json_schema() time, so without probing it here the tool
        schema build crashes at method-invocation. Probing it lets such types fall back
        to the ``Any`` schema (the model constructs the value in execute_python instead).

        Returns True for Pydantic models, dataclasses, basic types, generics, etc.
        Returns False for types like pd.DataFrame, np.ndarray, custom classes without
        Pydantic support, and models with non-JSON-serializable fields.
        """
        try:
            probe = create_model("_PydanticCompatProbe", result=(return_type, ...))
            probe.model_json_schema()
            return True
        except Exception:  # noqa: BLE001 — feasibility probe; any failure → fall back to Any
            return False

    def _create_return_model(self, return_type: Any, method_name: str) -> tuple[Any, bool]:
        """Create a Pydantic model for return_result validation.

        Returns:
            Tuple of (model_class, is_pydantic_validated).
            If the return type is Pydantic-compatible, returns the proper model and True.
            If not, returns a model with Any and False (caller should do isinstance check).
        """
        if self._is_pydantic_compatible(return_type):
            model = create_model(
                f"{method_name.title().replace('_', '')}ReturnResult",
                result=(return_type, ...),
            )
            return model, True
        else:
            # Fall back to Any — accept any value, let caller do isinstance check
            model = create_model(
                f"{method_name.title().replace('_', '')}ReturnResult",
                result=(Any, ...),
            )
            return model, False

    def _extract_annotated_description(self, type_hint: Any) -> tuple[Any, str | None]:
        """Extract the base type and first string metadata from Annotated.

        If the type is Annotated[T, "description", ...], extracts T and "description".
        If already has Field, returns (type_hint, None) to avoid double-wrapping.

        Returns:
            (base_type_or_original, description) where description is the first string metadata or None
        """
        from pydantic.fields import FieldInfo

        origin = get_origin(type_hint)

        if origin is Annotated:
            args = get_args(type_hint)
            base_type = args[0]  # First arg is always the actual type
            metadata = args[1:]  # Rest are metadata

            # Check if already has Field/FieldInfo - if so, don't extract
            for item in metadata:
                if isinstance(item, FieldInfo):
                    return type_hint, None

            # Find first string in metadata
            for item in metadata:
                if isinstance(item, str):
                    return base_type, item

            return base_type, None

        # Not annotated
        return type_hint, None

    def _render_return_type_doc(self, return_type: Any, *, max_chars: int = 1200) -> str | None:
        """Render ``doc(return_type)`` for an opaque (non-JSON-schemable) return type.

        Used to give the model construction guidance for types that fall back to the
        ``Any`` tool schema (pd.DataFrame, np.ndarray, custom classes). A
        ``spec.define_doc(<type>)`` adapter makes this concise; otherwise it's the
        type's default introspection. Best-effort: returns None if doc() is unavailable
        or fails — never blocks tool construction. Truncated to keep the prompt lean.
        """
        try:
            from nooa.agentdoc import doc as _doc
            from nooa.agentdoc.adapters import register_all

            # Activate doc() adapters for installed libs (pandas, plotly, …) so the
            # rendering is concise rather than the library's full constructor docstring.
            # Idempotent and installed-gated; only runs on this (opaque-type) path.
            register_all()
            rendered = _doc(return_type)
        except Exception:  # noqa: BLE001 — doc() is advisory; never break tool build
            return None
        if not rendered or not isinstance(rendered, str):
            return None
        rendered = rendered.strip()
        if len(rendered) > max_chars:
            rendered = (
                rendered[:max_chars].rstrip() + "\n… (call doc(<type>) for the full definition)"
            )
        return rendered

    def _build_return_result_tool(self, return_type: Any, method_name: str) -> Tool:
        """Build the return_result tool with schema matching the return type.

        Always uses a consistent schema: {result: <return_type>}
        Runtime parsing is flexible to accept multiple calling conventions.

        Automatically extracts string descriptions from Annotated[T, "description"]
        and converts them to Pydantic Field descriptions in the tool schema.

        Special case: When return_type is None, the tool accepts no parameters
        (or optional result=None) to indicate task completion.
        """
        from pydantic import Field

        def return_result(result: Any = None) -> Any:
            """Return the final result for the task.

            Call this tool when you have computed the final answer.
            The result must match the expected return type.
            """
            # This callable won't actually be called - we handle it in the execute loop
            return result

        # Handle None return type specially - no required result parameter
        # Note: `-> None` annotation gives `None` (the value), not `type(None)` (NoneType)
        if return_type is None or return_type is type(None):
            # For None return type, result is optional with default None
            ReturnResultModel = create_model(
                f"{method_name.title().replace('_', '')}ReturnResult",
                result=(type(None), None),  # Optional, defaults to None
            )
            description = (
                "Signal task completion. "
                "Call this when you have finished the task. "
                "No parameters required."
            )
        else:
            # Extract description from Annotated if present
            base_type, annotated_desc = self._extract_annotated_description(return_type)

            # Check if the type is Pydantic-compatible; fall back to Any if not
            # (e.g. pd.DataFrame, np.ndarray, custom classes without Pydantic support)
            schema_type = base_type if self._is_pydantic_compatible(base_type) else Any

            # Build the result field with description if available
            if annotated_desc:
                field = Field(..., description=annotated_desc)
                ReturnResultModel = create_model(
                    f"{method_name.title().replace('_', '')}ReturnResult",
                    result=(schema_type, field),
                )
            else:
                ReturnResultModel = create_model(
                    f"{method_name.title().replace('_', '')}ReturnResult",
                    result=(schema_type, ...),
                )

            # Always mention the actual type in the description, even if schema uses Any
            type_name = _format_type(return_type)
            if schema_type is Any and schema_type is not return_type:
                description = (
                    f"Return the final result for the task. "
                    f"Call this ONLY when you have computed the final answer. "
                    f"Expected return type: {type_name}. "
                    f"IMPORTANT: This type cannot be passed directly via this tool. "
                    f"Construct the object in execute_python() and call "
                    f"return_result(variable) from within the code instead."
                )
                # Opaque types (pd.DataFrame, np.ndarray, custom classes) carry no JSON
                # schema, so the model has nothing structural to go on. Proactively fold in
                # the type's doc() so it knows how to construct it without spending a turn
                # calling doc() itself. A `spec.define_doc(<type>)` adapter (e.g. pandas)
                # makes this concise; otherwise it's the type's default introspection.
                type_doc = self._render_return_type_doc(base_type)
                if type_doc:
                    description += f"\n\nReturn type reference:\n{type_doc}"
            else:
                description = (
                    f"Return the final result for the task. "
                    f"Call this ONLY when you have computed the final answer. "
                    f"Expected return type: {type_name}. "
                    f"Tip: prefer calling return_result(variable) from within execute_python() "
                    f"to pass computed results directly."
                )

        return Tool(
            name="return_result",
            description=description,
            callable=return_result,
            parameters_model=ReturnResultModel,
        )

    async def _run_prefill(
        self,
        runtime: RuntimeServices,
        call: "CurrentCall",
        builtins: dict[str, Any],
        session: CodeActSession,
    ) -> None:
        """Run prefill code before the main generation loop.

        Executes prefill code as separate synthetic tool calls. Each prefill type
        runs as its own execution, demonstrating variable persistence across turns.
        This helps the LLM understand that computed variables remain available.

        Runs two types of prefill code (as separate executions):
        1. InspectInputsPrefill: Auto-generated code to inspect input parameters
        2. Pre-ellipsis code: User-defined setup code before the `...` marker
        """

        prefill_builtins = {**builtins, "_call": call}

        # 1. Prefill plugin (parameter inspection by default).
        # ``CodeActConfig.prefill`` controls this: default is
        # ``InspectInputsPrefill()``, ``None`` disables, custom Prefill
        # replaces. See ``strategies.prefill.Prefill`` protocol.
        prefill = self.config.prefill
        if prefill is not None:
            truncation_config = self._get_truncation_config(runtime)
            inspect_code = prefill.get_code(call, config=truncation_config)

            if inspect_code:
                await self._execute_prefill_step(
                    runtime,
                    inspect_code,
                    prefill_builtins,
                    session,
                    call.method_name,
                    prefill_type="inspect_inputs",
                )

        # 2. Pre-ellipsis code (user-defined setup before ...)
        # Runs as separate execution - LLM sees variables persist from step 1
        if call.pre_ellipsis_code:
            logger.debug(f"[CODEACT] Pre-ellipsis code: {len(call.pre_ellipsis_code)} chars")
            await self._execute_prefill_step(
                runtime,
                call.pre_ellipsis_code,
                prefill_builtins,
                session,
                call.method_name,
                prefill_type="pre_ellipsis",
            )

    async def _execute_prefill_step(
        self,
        runtime: RuntimeServices,
        code: str,
        builtins: dict[str, Any],
        session: CodeActSession,
        method_name: str,
        prefill_type: str,
    ) -> None:
        """Execute a single prefill step as a synthetic tool call.

        Each prefill step appears as a separate code execution in events,
        helping the LLM understand that variables persist across turns.
        """
        get_harness_metrics().prefill(prefill_type)
        logger.debug(f"[CODEACT] Running prefill ({prefill_type}) for {method_name}")

        # Create synthetic tool call
        prefill_id = f"prefill_{uuid4().hex[:8]}"
        prefill_event_id = runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=prefill_id,
                name="execute_python",
                arguments={"code": code},
                result=None,  # Will be updated after execution
                metadata={"prefill": True, "prefill_type": "inspect_inputs"},
            )
        )

        # Update with executing status immediately (deferred output pattern)
        runtime.event_manager.update(
            prefill_event_id,
            result=ToolResult(
                tool_call_id=prefill_id,
                content="status: executing",
                result_status=ResultStatus.COMPLETE,  # Will update to error if needed
            ),
        )

        # Execute the code
        result = await self._execute_code(
            runtime, code, builtins, session, method_name, tool_call_id=prefill_id
        )

        # Merge captured locals into session (persists for next steps and LLM turns)
        if result.captured_locals:
            session.session_locals.update(result.captured_locals)
            logger.debug(
                f"[CODEACT] Prefill ({prefill_type}) captured locals: "
                f"{list(result.captured_locals.keys())}"
            )

        # Update ToolCallEvent with final status
        final_status = ResultStatus.ERROR if result.error else ResultStatus.COMPLETE
        runtime.event_manager.update(
            prefill_event_id,
            result=ToolResult(
                tool_call_id=prefill_id,
                content=f"status: {final_status.value}",
                result_status=final_status,
            ),
        )

        # Format error if present
        error_text = ""
        if result.error:
            line_offset = getattr(result, "wrapper_line_offset", 0)
            error_text = self._format_error(result.error, line_offset=line_offset)

        # Add execution output as user message (execution_count=0 since before main loop)
        runtime.event_manager.add(
            PythonOutput(
                tool_call_id=prefill_id,
                execution_count=0,  # Prefill is before main loop
                stdout=result.stdout,
                stderr=result.stderr,
                error=error_text,
                value=result.returned_value if result.has_return else None,
                explicit_return=result.explicit_return if result.has_return else False,
                execution_status=final_status,
                images=result.images,
                metadata={"prefill": True, "prefill_type": prefill_type},
            )
        )

        if result.error:
            logger.warning(f"[CODEACT] Prefill ({prefill_type}) execution error: {result.error}")

    async def _execute_code(
        self,
        runtime: RuntimeServices,
        code: str,
        builtins: dict[str, Any],
        session: CodeActSession,
        target_method_name: str,
        tool_call_id: str | None = None,
    ) -> Any:
        """Execute Python code via the runtime."""
        from nooa.events import ExecutionResult

        logger.debug(
            "[CODEACT] Executing code (iter=%s, err=%s, chars=%s)",
            session.iteration,
            session.error_count,
            len(code),
        )

        # Validate REPL policy (no classes, await async methods)
        validator = GeneratedCodeValidator()
        validation_errors = validator.validate(code, runtime.agent)
        if validation_errors:
            error_msg = "Code validation failed:\n" + "\n".join(
                f"• {err}" for err in validation_errors
            )
            logger.warning(f"[CODEACT] Validation errors: {validation_errors}")
            return ExecutionResult(stdout="", error=Exception(error_msg), defined_methods={})

        # Sandbox backend: the worker owns the persistent namespace and compiles
        # its own helpers, so we skip the parent-side namespace/helper build and
        # delegate the cell to the guarded worker process. Routing through
        # execute_code (rather than the executor directly) keeps the execute_python
        # middleware, before/after_code_execution hooks and events firing.
        if session.sandbox_executor is not None:
            from nooa.runtime.debug_handler import code_exec_context

            with code_exec_context(code):
                return await runtime.execute_code(
                    code,
                    validate=True,  # run restrictions/cell-guard validation on the parent
                    wrap_in_function=True,
                    timeout=self.config.cell_timeout,
                    tool_call_id=tool_call_id,
                    execution_count=session.iteration,
                    restrictions=self.config.restrictions,
                    sandbox_executor=session.sandbox_executor,
                )

        # Build execution namespace
        strategy_extras: dict[str, Any] = {
            "CodeActStrategy": type(self),
        }
        try:
            from nooa.strategies.predict import PredictStrategy

            strategy_extras["PredictStrategy"] = PredictStrategy
        except ImportError:
            pass

        namespace = ExecutionNamespaceBuilder.build(
            runtime.agent, extra={**builtins, **session.session_locals, **strategy_extras}
        )

        # Pre-compile helper function defs so @strategy decorators can see
        # _generated_source at decoration time. Helpers are never attached to the agent.
        helper_compiler = HelperFunctionManager()
        helper_result = helper_compiler.apply(
            code,
            runtime.agent,
            session.session_locals,
            namespace=namespace,
        )

        if helper_result.errors:
            error_msg = "Failed to define helper function(s):\n" + "\n".join(
                f"- {e}" for e in helper_result.errors
            )
            logger.warning(f"[CODEACT] Helper compile errors: {helper_result.errors}")
            return ExecutionResult(stdout="", error=Exception(error_msg), defined_methods={})

        if helper_result.installed:
            logger.debug(f"[CODEACT] Compiled helpers: {helper_result.installed}")

        # Execute with session locals
        execution_builtins = {**builtins, **session.session_locals}

        # Track this execution so the /activity slash command can report that
        # the agent is currently running a code cell (vs blocked on an LLM call).
        from nooa.runtime.debug_handler import code_exec_context

        with code_exec_context(code):
            return await runtime.execute_code(
                code,
                builtins=execution_builtins,
                validate=True,
                wrap_in_function=True,
                timeout=self.config.cell_timeout,
                tool_call_id=tool_call_id,
                execution_count=session.iteration,
                restrictions=self.config.restrictions,
            )

    def _format_error(
        self, error: Exception, code: str | None = None, *, line_offset: int = 0
    ) -> str:
        """Format an error for display using the configured formatter."""
        if self.error_formatter is not None:
            # Custom formatters may or may not support line_offset
            try:
                return self.error_formatter.format(error, code, line_offset=line_offset)
            except TypeError:
                # Formatter doesn't accept line_offset
                return self.error_formatter.format(error, code)

        from nooa.errors.formatting import format_error_for_llm

        return format_error_for_llm(error, code, line_offset=line_offset)

    def _extract_module_context(
        self, agent_module: types.ModuleType, agent: Any | None = None
    ) -> dict[str, Any]:
        """Extract relevant items from agent's module for execution context.

        This makes the execution environment behave as if code was written directly
        in the method body, with access to all module-level imports and type definitions.

        Args:
            agent_module: The agent's defining module
            agent: Optional agent instance to check for skill/tool attributes

        Returns:
            Dict of names that should be available during code execution
        """
        context: dict[str, Any] = {}

        from nooa.agentdoc.visibility import (
            filter_module_globals,
            filter_mro_module_globals,
            iter_agent_mro_modules,
        )

        # Merge module-level globals across the agent's MRO so a parent defined
        # in another module contributes its module-level symbols too. The
        # "defined here" check below treats any MRO module as own, not just the
        # leaf module. Fall back to the single (leaf) module when no agent
        # instance is available.
        if agent is not None:
            filtered = filter_mro_module_globals(type(agent))
            own_module_names = {m.__name__ for m in iter_agent_mro_modules(type(agent))}
        else:
            filtered = filter_module_globals(agent_module)
            own_module_names = {agent_module.__name__}

        # 1. Module-level imports and definitions
        for name, obj in filtered.items():
            # Include imported modules (import os, import json)
            if isinstance(obj, types.ModuleType):
                context[name] = obj
                continue

            # Include imported classes/functions
            # Check if it was imported (not defined in an MRO module)
            obj_module = getattr(obj, "__module__", None)
            if obj_module and obj_module not in own_module_names:
                if isinstance(obj, type) or callable(obj):
                    context[name] = obj
                    continue

        # 2. Module-level definitions: classes AND functions (incl. standalone
        #    @strategy wrappers) defined in the agent's own (or ancestor's)
        #    module. Visible by default — filter_module_globals has already
        #    dropped names hidden via @hidden, Annotated[..., hidden], or
        #    `with hidden:`. Functions were previously excluded here (only
        #    `type` instances were kept), so a module-level helper or standalone
        #    generation function was invisible to the agent and absent from
        #    exec_globals.
        for name, obj in filtered.items():
            if name in context:
                continue
            obj_module = getattr(obj, "__module__", None)
            if obj_module in own_module_names and callable(obj):
                context[name] = obj

        # 3. Auto-import classes for skill/tool instances found on the agent
        if agent is not None:
            self._import_dynamic_classes(agent, agent_module, context)

        # Note: Return type annotations are automatically included by steps 1-2
        # since they must be known symbols in the module namespace

        return context

    @staticmethod
    def _import_dynamic_classes(
        agent: Any, agent_module: types.ModuleType, context: dict[str, Any]
    ) -> None:
        """Auto-import classes for agent attributes not already in the execution context.

        When agents use dynamically generated tools/skills (e.g. from mcp_nooa or
        skills_nooa), their classes won't be in the module's imports. This discovers
        them from the agent's attributes and makes them available for doc()/isinstance().

        Discovered types are added to ``context`` (which feeds exec_globals) but NOT
        written to ``agent_module.__dict__`` — mutating the module is a side effect
        that permanently pollutes the namespace (gl-78).

        Package-agnostic: works for any class, not just known packages.
        """
        try:
            seen = set(context.keys())

            for attr_value in _iter_agent_attrs(agent):
                cls = type(attr_value)
                name = cls.__name__

                if name in seen or cls.__module__ == "builtins":
                    continue

                context[name] = cls
                seen.add(name)
        except Exception:
            pass  # Convenience feature — never break execution

    def _build_builtins(self, runtime: RuntimeServices, call: "CurrentCall") -> dict[str, Any]:
        """Build execution builtins including module context.

        This creates an execution environment that behaves as if the LLM's code
        was written directly in the method body, with access to:
        - Module-level imports
        - Module-level type definitions (Pydantic models, etc.)
        - Strategy builtins (return_result)
        - Method parameters
        """

        def return_result(*args: Any, **kwargs: Any) -> None:
            """Submit the final answer from within execute_python code.

            Usage:
                - return_result(value)               # single positional -> "result" field
                - return_result(field1=v1, field2=v2)  # named fields of the return type

            Positional and keyword arguments cannot be mixed, except for return
            types with a ``kind`` discriminator field (see the implementation
            comment below).
            """
            if args:
                if len(args) > 1:
                    raise ValueError("return_result() takes at most 1 positional argument")
                if kwargs:
                    # Mixing positional + keyword args is only allowed when the
                    # return type has a "kind" discriminator field: the positional
                    # is routed to "kind" and the kwargs fill the rest, so
                    # return_result("respond", content="hi") is equivalent to
                    # return_result(kind="respond", content="hi"). Do NOT remove
                    # this branch: it is relied on in production (the TUI's
                    # RespondResult and skills-sw SKILL.md).
                    model_fields = getattr(call.return_type, "model_fields", {})
                    if "kind" not in model_fields:
                        raise ValueError("Cannot mix positional and keyword arguments")
                    if "kind" in kwargs:
                        raise ValueError(
                            "return_result() got kind both positionally and by keyword"
                        )
                    result = dict(kwargs)
                    result["kind"] = list(args)[0]
                    raise _ReturnResultSignal(result=result)
                # Single positional argument - treat as the 'result' field
                raise _ReturnResultSignal(result={"result": list(args)[0]})
            else:
                # Keyword arguments - pass as-is
                raise _ReturnResultSignal(result=kwargs)

        # Start with module context (imports, type definitions)
        agent_module = inspect.getmodule(type(runtime.agent))
        builtins: dict[str, Any] = {}
        if agent_module:
            builtins.update(self._extract_module_context(agent_module, agent=runtime.agent))

        # Add strategy builtins (these override any module-level names)
        builtins.update(
            {
                "return_result": return_result,
            }
        )

        # Add method parameters as variables.
        # call.kwargs is already the fully merged positional+keyword mapping
        # (built by _execute_with_generation before the strategy is invoked).
        builtins.update(call.kwargs)

        return builtins
