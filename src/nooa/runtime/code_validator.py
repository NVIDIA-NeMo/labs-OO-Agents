# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified AST validation for agent-generated code.

This module provides a single entry point for all code validation:
- SecurityValidator: Guardrails against common footguns (forbidden builtins,
  restricted imports, direct class/dunder mutation)
- BlockingCallValidator: Prevents blocking calls that freeze the event loop
- REPLPolicyValidator: Enforces REPL-style coding conventions

Security model — read before extending:
    These validators are **guardrails, not a security boundary**. They operate on
    a static AST and reduce the chance that an LLM *accidentally* corrupts runtime
    state, freezes the event loop, or reaches for a surprising module. They do
    **not** and **cannot** contain adversarial code: a static Python checker is
    trivially bypassable (e.g. ``open()`` for arbitrary file I/O, dynamic module
    loading via ``importlib.util``/``importlib.machinery``, reflection, C-extension
    loading). Do not treat the deny-lists as a jail or add checks under the belief
    that they close a real escape — that is unwinnable whack-a-mole.

    The actual containment boundary is OS-level isolation (container / VM / gVisor,
    e.g. NVIDIA OpenShell). Run agents that execute generated code inside one.
    See the README security note and ``runtime/restrictions.py``.

Usage:
    validator = UnifiedCodeValidator()
    context = ValidationContext(
        code=code,
        available_names=["self", "asyncio"],
        restricted_imports=frozenset({"os", "sys"}),  # deny list
        blocked_modules=DEFAULT_BLOCKED_MODULES,
    )
    validator.validate(code, context)  # Raises ValidationError on failure
"""

# =============================================================================
# Error Code Registry
# =============================================================================
# The codes below match what the validators actually emit. Every code is emitted
# with severity="error" (the historical "W" prefix on the infinite-loop check was
# renamed to E303 to reflect that it is an error, not a warning).
# SecurityValidator (_SecurityVisitor):
#   E001 — Forbidden builtin / attribute call (exec, eval, compile, __import__,
#          input, globals, locals, breakpoint; their aliases; calls that could
#          modify runtime restrictions; getattr() of any such name)
#   E002 — Restricted or blocked import (module in restricted_imports or blocked_modules)
#   E003 — Wildcard import ('from ... import *')
#   E004 — Recursive self-call (self.<method>() listed in forbidden_self_calls)
#   E005 — Process/control-flow termination (raise SystemExit/KeyboardInterrupt,
#          sys.exit()/os._exit()/os.abort() and their aliases)
#   E101 — Forbidden dunder attribute access (__class__, __subclasses__, etc.)
#          and access to '__builtins__'
#   E102 — Base-class/super dunder access that bypasses Agent runtime guards
#          (object.__setattr__, type.__setattr__, super().__setattr__, etc.)
#   E104 — setattr()/delattr()/getattr() targeting a dunder attribute name
# REPLPolicyValidator:
#   E301 — Missing await on an async method call
#   E303 — Potential infinite loop ('while True' without break/return)
# ClassAssignmentValidator:
#   E401 — Forbidden class attribute assignment (ClassName.x = ..., type(self).x = ...)
#   E402 — Forbidden class-level setattr() (setattr(ClassName, ...), setattr(type(self), ...))
# BlockingCallValidator:
#   E310 — Blocking call that would freeze the event loop
# ReturnTypeShadowValidator:
#   E501 — Local class/function definition or assignment shadows the method's return type
# =============================================================================
import ast
import inspect
import logging
import types
from dataclasses import dataclass, field
from typing import Annotated, Any, ForwardRef, Literal, Protocol, get_args, get_origin

from nooa.errors import RestrictedCodeError as ValidationError
from nooa.runtime.restrictions import (
    RestrictionsConfig,
    match_blocked_module,
)

logger = logging.getLogger(__name__)

# Re-export ValidationError for convenience
__all__ = [
    "UnifiedCodeValidator",
    "SecurityValidator",
    "BlockingCallValidator",
    "REPLPolicyValidator",
    "ClassAssignmentValidator",
    "ReturnTypeShadowValidator",
    "ValidationContext",
    "ValidationError",
    "ValidationIssue",
]


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class ValidationIssue:
    """Single validation issue with location and severity."""

    line: int
    col: int
    message: str
    severity: Literal["error", "warning"] = "error"
    code: str = ""  # Error code like "E001", "W001"
    fix_hint: str | None = None
    doc_link: str | None = None  # Link to documentation


@dataclass
class ValidationContext:
    """Shared context for all validators."""

    code: str = ""
    agent_class: type | None = None
    available_names: set[str] = field(default_factory=set)
    importable_modules: set[str] = field(
        default_factory=set
    )  # Deprecated: use restricted_imports deny list instead
    forbidden_self_calls: set[str] = field(default_factory=set)
    execution_count: int = 1
    agent: Any = None  # Agent instance for method introspection
    exec_globals: dict[str, Any] = field(default_factory=dict)
    restricted_imports: frozenset[str] = field(default_factory=frozenset)
    blocked_modules: frozenset[str] = field(default_factory=frozenset)
    # Return type of the currently executing generation method, when known.
    # Drives ReturnTypeShadowValidator: if the generated code redefines a class
    # whose name is part of this annotation, the resulting __repl_wrapper__-scoped
    # class will not pass return_result Pydantic validation (see issue gl-143).
    return_type: Any = None


class Validator(Protocol):
    """Protocol for individual validators."""

    def validate(self, tree: ast.AST, context: ValidationContext) -> list[ValidationIssue]:
        """Validate AST and return list of issues."""
        ...


# =============================================================================
# Security Validator
# =============================================================================

# Functions that are always forbidden in generated code
FORBIDDEN_BUILTINS = frozenset(
    {
        # Dynamic code execution (security risk)
        "exec",
        "eval",
        "compile",
        "__import__",
        # Blocking stdin operations (cause hangs)
        "input",
        "breakpoint",
        # Namespace access (security risk)
        "globals",
        "locals",
        "vars",  # Similar to locals(), gives access to __dict__
        # Process termination
        "exit",
        "quit",
        # Restriction mutation (prevents agent from loosening its own restrictions)
        "set_restricted_imports",
        "get_restricted_imports",
    }
)

# Function names forbidden as attribute calls (e.g. `mod.set_restricted_imports()`)
FORBIDDEN_ATTR_CALLS = frozenset(
    {
        "set_restricted_imports",
        "get_restricted_imports",
    }
)

# Dunder attributes commonly used to reach runtime internals (introspection
# ladders like __class__ -> __subclasses__). Blocking them trims easy footguns;
# it is not a containment guarantee (see the module docstring's security model).
DANGEROUS_DUNDER_ATTRS = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__dict__",
    }
)


class SecurityValidator:
    """Guardrail checks over generated code (not a security boundary).

    Flags common footguns so the LLM fails fast with a clear message rather than
    corrupting runtime state or silently doing something surprising. This is
    defense-in-depth, not containment — see the module docstring's security
    model. Real isolation comes from running the agent in an OS-level sandbox.

    Checks for:
    - Forbidden builtins (exec, eval, compile, __import__, input, breakpoint, globals, locals)
    - Restricted/blocked imports (modules in restricted_imports or blocked_modules deny lists)
    - Import * (always forbidden)
    - Direct dunder attribute access (__class__, __bases__, etc.)
    - Recursive self-calls (infinite recursion prevention)
    - Aliased forbidden builtins
    - Forbidden attribute calls (set_restricted_imports, get_restricted_imports)
    """

    def validate(self, tree: ast.AST, context: ValidationContext) -> list[ValidationIssue]:
        """Validate security rules and return issues."""
        visitor = _SecurityVisitor(context)
        visitor.visit(tree)
        return visitor.issues


class _SecurityVisitor(ast.NodeVisitor):
    """AST visitor for security validation."""

    def __init__(self, context: ValidationContext):
        self.context = context
        self.issues: list[ValidationIssue] = []
        # Track aliases: local_name -> original_forbidden_name
        self.forbidden_aliases: dict[str, str] = {}
        # Track aliases like `import os as o; o._exit()` / `import sys as s; s.exit()`.
        self.module_aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> Any:
        """Check import statements."""
        for alias in node.names:
            available, deny_tier = self._is_module_available(alias.name)
            if not available:
                self.issues.append(self._make_import_error(node, alias.name, deny_tier))

            root_module = alias.name.split(".", 1)[0]
            if root_module in ("sys", "os"):
                self.module_aliases[alias.asname or root_module] = root_module

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Check from-import statements."""
        # from X import * is always forbidden
        if any(alias.name == "*" for alias in node.names):
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message="'from ... import *' is forbidden for security reasons",
                    code="E003",
                )
            )
            self.generic_visit(node)
            return

        # Check if module is available
        module_name = node.module or ""
        available, deny_tier = self._is_module_available(module_name)
        if not available:
            self.issues.append(self._make_import_error(node, module_name, deny_tier))
        else:
            # Track aliases of forbidden builtins and imported process-termination calls.
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name in FORBIDDEN_BUILTINS:
                    self.forbidden_aliases[local_name] = alias.name
                if (module_name, alias.name) in (
                    ("sys", "exit"),
                    ("os", "_exit"),
                    ("os", "abort"),
                ):
                    self.forbidden_aliases[local_name] = f"{module_name}.{alias.name}"

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Check direct forbidden calls
            if func_name in FORBIDDEN_BUILTINS:
                self.issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"{func_name}() is forbidden - it blocks or allows code execution",
                        code="E001",
                    )
                )
            # Check aliased forbidden calls
            elif func_name in self.forbidden_aliases:
                original = self.forbidden_aliases[func_name]
                if original in ("sys.exit", "os._exit", "os.abort"):
                    self._add_process_termination_call_issue(
                        node, f"{func_name}() (alias for {original})"
                    )
                else:
                    self.issues.append(
                        ValidationIssue(
                            line=node.lineno,
                            col=node.col_offset,
                            message=f"{func_name}() is forbidden (alias for {original})",
                            code="E001",
                        )
                    )
            # Check setattr/delattr/getattr with dunder names or forbidden attr calls
            elif func_name in ("setattr", "delattr", "getattr"):
                self._check_attr_modification_with_dunder(node, func_name)
                self._check_getattr_forbidden_attr_call(node, func_name)

        # Check for forbidden self.method() calls (prevents recursion)
        if self.context.forbidden_self_calls and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                method_name = node.func.attr
                if method_name in self.context.forbidden_self_calls:
                    self.issues.append(
                        ValidationIssue(
                            line=node.lineno,
                            col=node.col_offset,
                            message=f"calling self.{method_name}() is forbidden - "
                            "this would cause infinite recursion",
                            code="E004",
                        )
                    )

        # Check for forbidden attribute calls (e.g. mod.set_restricted_imports())
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_ATTR_CALLS:
                self.issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"{node.func.attr}() is forbidden - "
                        "this could modify runtime security restrictions",
                        code="E001",
                    )
                )
            # Process-termination attribute calls: sys.exit(), os._exit(), os.abort(), plus aliases.
            elif isinstance(node.func.value, ast.Name):
                module_name = self.module_aliases.get(node.func.value.id, node.func.value.id)
                if (module_name, node.func.attr) in (
                    ("sys", "exit"),
                    ("os", "_exit"),
                    ("os", "abort"),
                ):
                    call = f"{node.func.value.id}.{node.func.attr}()"
                    if node.func.value.id != module_name:
                        call += f" (alias for {module_name}.{node.func.attr})"
                    self._add_process_termination_call_issue(node, call)

        self.generic_visit(node)

    def _add_process_termination_call_issue(self, node: ast.Call, call: str) -> None:
        self.issues.append(
            ValidationIssue(
                line=node.lineno,
                col=node.col_offset,
                message=(
                    f"{call} is forbidden - it terminates at the process/control-flow "
                    "level and can cancel sibling tasks. Use break, a flag, a helper "
                    "return, or return_result() to stop."
                ),
                code="E005",
            )
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Check attribute access for dangerous dunders."""
        if node.attr in DANGEROUS_DUNDER_ATTRS:
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Access to '{node.attr}' is forbidden - "
                    "this could bypass security restrictions",
                    code="E101",
                )
            )
        # Forbid base-class dunder accesses like `object.__setattr__` and
        # `type.__setattr__` — they bypass Agent.__setattr__ via the C-level slot.
        elif (
            isinstance(node.value, ast.Name)
            and node.value.id in ("object", "type")
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Access to '{node.value.id}.{node.attr}' is forbidden - "
                    "this would bypass agent runtime guards",
                    code="E102",
                )
            )
        # Forbid `super(...).__setattr__(...)` and similar — super() routes
        # to the parent class's __setattr__, bypassing Agent.__setattr__.
        elif (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super"
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Access to 'super().{node.attr}' is forbidden - "
                    "this would bypass agent runtime guards",
                    code="E102",
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        """Check name access for __builtins__."""
        if node.id == "__builtins__":
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message="Access to '__builtins__' is forbidden - "
                    "this could bypass security restrictions",
                    code="E101",
                )
            )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> Any:
        """Flag `raise SystemExit` / `raise SystemExit(...)`.

        SystemExit (and KeyboardInterrupt) are BaseException, not Exception, so
        a generated cell raising one escapes the runtime's per-cell error
        handling and can cancel sibling tasks (e.g. a surrounding TaskGroup).
        To stop a cell, use break, a flag, a helper return, or return_result().
        This is a fast-fail for the literal form; the runtime also converts any
        SystemExit/KeyboardInterrupt that reaches it into an execution error.
        """
        exc = node.exc
        # `raise SystemExit` (Name) or `raise SystemExit(...)` (Call of a Name)
        name = None
        if isinstance(exc, ast.Name):
            name = exc.id
        elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        if name in ("SystemExit", "KeyboardInterrupt"):
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f"raise {name} is forbidden - it terminates the cell at "
                        "the process/control-flow level and can cancel sibling "
                        "tasks. Use break, a flag, a helper return, or "
                        "return_result() to stop."
                    ),
                    code="E005",
                )
            )
        self.generic_visit(node)

    def _is_module_available(self, module_name: str) -> tuple[bool, str | None]:
        """Check if module is importable under the deny-list model.

        A module is denied if it matches either blocked_modules (tier 1)
        or restricted_imports (tier 2). Everything else is allowed.

        Returns:
            (available, deny_tier) — deny_tier is "blocked" or "restricted"
            when available is False, None otherwise.
        """
        # Tier 1: always deny blocked modules (event-loop hazards)
        if self.context.blocked_modules:
            if match_blocked_module(module_name, self.context.blocked_modules) is not None:
                return False, "blocked"
        # Tier 2: deny restricted imports
        if self.context.restricted_imports:
            if match_blocked_module(module_name, self.context.restricted_imports) is not None:
                return False, "restricted"
        return True, None

    def _check_attr_modification_with_dunder(self, node: ast.Call, func_name: str) -> None:
        """Check if setattr/delattr is being used with dunder attribute names."""
        # setattr(obj, name, value) or delattr(obj, name)
        # The attribute name is the second argument
        if len(node.args) < 2:
            return

        attr_arg = node.args[1]

        # Check if it's a string literal
        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            attr_name = attr_arg.value
            if attr_name in DANGEROUS_DUNDER_ATTRS:
                self.issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"{func_name}() with '{attr_name}' is forbidden - "
                        "this could bypass security restrictions",
                        code="E104",
                    )
                )
            elif attr_name.startswith("__") and attr_name.endswith("__"):
                self.issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"{func_name}() with dunder attribute '{attr_name}' is forbidden - "
                        "this could bypass security restrictions",
                        code="E104",
                    )
                )

    def _check_getattr_forbidden_attr_call(self, node: ast.Call, func_name: str) -> None:
        """Flag getattr(obj, 'name') where name is in FORBIDDEN_ATTR_CALLS."""
        if func_name != "getattr" or len(node.args) < 2:
            return
        attr_arg = node.args[1]
        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            if attr_arg.value in FORBIDDEN_ATTR_CALLS:
                self.issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"getattr() with '{attr_arg.value}' is forbidden - "
                        "this could modify runtime security restrictions",
                        code="E001",
                    )
                )

    def _make_import_error(
        self, node: ast.Import | ast.ImportFrom, module_name: str, deny_tier: str | None = None
    ) -> ValidationIssue:
        """Create error for restricted or blocked import."""
        tier = deny_tier or "restricted"
        if tier == "blocked":
            msg = (
                f"import of '{module_name}' is blocked. "
                f"This module can freeze the event loop and is not allowed in agent code."
            )
        else:
            msg = (
                f"import of '{module_name}' is restricted. "
                f"This module is in the restricted_imports deny list. "
                f"Also forbidden: eval(), exec(), compile(), __import__(), "
                f"input(), globals(), locals(), breakpoint()"
            )
        return ValidationIssue(
            line=node.lineno,
            col=node.col_offset,
            message=msg,
            code="E002",
        )


# =============================================================================
# REPL Policy Validator
# =============================================================================
class REPLPolicyValidator:
    """Validates REPL-style coding conventions.

    Checks for:
    - Missing await on async method calls
    """

    def validate(self, tree: ast.AST, context: ValidationContext) -> list[ValidationIssue]:
        """Validate REPL policy rules and return issues."""
        visitor = _REPLPolicyVisitor(context)
        visitor.visit(tree)
        return visitor.issues


class _REPLPolicyVisitor(ast.NodeVisitor):
    """AST visitor for REPL policy validation."""

    def __init__(self, context: ValidationContext):
        self.context = context
        self.issues: list[ValidationIssue] = []
        self.async_method_names: set[str] = set()
        self.parent_map: dict[ast.AST, ast.AST] = {}
        self._collect_async_methods()

    def visit(self, node: ast.AST) -> Any:
        """Build parent map while visiting."""
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        return super().visit(node)

    def visit_While(self, node: ast.While) -> Any:
        """Check for infinite loops (while True without break/return)."""
        if self._is_infinite_loop(node) and not self._has_exit_statement(node):
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().infinite_loop()
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message="Potential infinite loop detected (while True without break/return) - "
                    "add a break condition or use an iteration limit",
                    code="E303",
                    severity="error",
                )
            )
        self.generic_visit(node)

    def _is_infinite_loop(self, node: ast.While) -> bool:
        """Check if while loop has a constant True condition."""
        # while True:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            return True
        # while 1:
        if isinstance(node.test, ast.Constant) and node.test.value == 1:
            return True
        # while not False:
        if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
            if isinstance(node.test.operand, ast.Constant) and node.test.operand.value is False:
                return True
        return False

    def _has_exit_statement(self, node: ast.While) -> bool:
        """Check if while loop body has a break, return, or raise at the loop level.

        Does NOT count:
        - break/return/raise inside nested function/class definitions
        - break inside nested loops (only exits the inner loop)
        """
        for child in node.body:
            if self._has_exit_in_subtree(child, check_break=True):
                return True
        return False

    def _has_exit_in_subtree(self, node: ast.AST, check_break: bool = True) -> bool:
        """Check if node or its children (excluding nested defs/loops) have exit statements.

        Args:
            node: AST node to check
            check_break: If True, count Break as exit. Set to False when entering nested loops
                        since their breaks don't exit the outer loop.
        """
        # Direct exit statements
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Break) and check_break:
            return True

        # Don't recurse into function/class definitions - their exits don't affect outer loop
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False

        # For nested loops, don't count their breaks as exiting the outer loop
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # Check the loop body, but breaks inside only exit THIS loop, not outer
            for child in node.body:
                if self._has_exit_in_subtree(child, check_break=False):
                    return True
            for child in node.orelse:
                if self._has_exit_in_subtree(child, check_break=False):
                    return True
            return False

        # Check children
        for subnode in ast.iter_child_nodes(node):
            if self._has_exit_in_subtree(subnode, check_break):
                return True

        return False

    def visit_Call(self, node: ast.Call) -> Any:
        """Check for missing await on async method calls."""
        if not self.async_method_names:
            self.generic_visit(node)
            return

        # Check self.method() calls
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        if not isinstance(node.func.value, ast.Name):
            self.generic_visit(node)
            return

        if node.func.value.id != "self":
            self.generic_visit(node)
            return

        method_name = node.func.attr
        if method_name not in self.async_method_names:
            self.generic_visit(node)
            return

        # Check if already awaited or in a gather-friendly context
        if not self._is_awaited_or_gathered(node):
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().missing_await(method_name)
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Method `{method_name}` is async and must be called with `await`",
                    code="E301",
                    fix_hint=f"await self.{method_name}(...)",
                )
            )

        self.generic_visit(node)

    def _collect_async_methods(self) -> None:
        """Collect names of async methods from agent."""
        from nooa.agentdoc.visibility import is_hidden_method

        agent = self.context.agent
        if not agent:
            return

        # Check class-level methods
        for attr_name in dir(agent.__class__):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(agent.__class__, attr_name, None)
                if attr and inspect.iscoroutinefunction(attr):
                    if not is_hidden_method(attr):
                        self.async_method_names.add(attr_name)
            except Exception:
                continue

        # Check instance-level methods
        for attr_name in dir(agent):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(agent, attr_name)
                if callable(attr) and inspect.iscoroutinefunction(attr):
                    if not is_hidden_method(attr):
                        self.async_method_names.add(attr_name)
            except Exception:
                continue

    def _is_awaited_or_gathered(self, node: ast.AST) -> bool:
        """Check if call is awaited or in a gather-friendly context."""
        current = node
        while current in self.parent_map:
            parent = self.parent_map[current]

            # Direct await
            if isinstance(parent, ast.Await) and getattr(parent, "value", None) == current:
                return True

            # In list/generator comprehension (for gather patterns)
            if isinstance(parent, (ast.ListComp, ast.GeneratorExp)):
                return True

            current = parent

        return False


# =============================================================================
# Class Assignment Validator
# =============================================================================
class ClassAssignmentValidator:
    """Detect dangerous class attribute assignments like ClassName.method = ...

    When LLM-generated code assigns directly to a class (instead of an instance),
    it corrupts all subsequent instances that share that class definition.
    This validator blocks patterns like:
    - ParentAgent.method = lambda: ...
    - SubAgentClass.work = factory_result
    - ClassName.attr += value

    Safe patterns that are NOT blocked:
    - self.attr = value (instance assignment)
    - obj.attr = value (non-class object)
    - data["key"] = value (subscript, not attribute)
    """

    def validate(self, tree: ast.AST, context: ValidationContext) -> list[ValidationIssue]:
        """Validate AST for class assignment patterns."""
        visitor = _ClassAssignmentVisitor(context)
        visitor.visit(tree)
        return visitor.issues


class _ClassAssignmentVisitor(ast.NodeVisitor):
    """AST visitor for detecting class attribute assignments."""

    def __init__(self, context: ValidationContext):
        self.context = context
        self.issues: list[ValidationIssue] = []
        self.known_class_names = self._collect_class_names()
        # Track variables assigned from type(self) - these hold class references
        self.class_ref_vars: set[str] = set()
        # Track variables assigned from self - type(var) is equivalent to type(self)
        self.self_ref_vars: set[str] = set()

    def _collect_class_names(self) -> set[str]:
        """Collect names that refer to classes in the execution context."""
        names: set[str] = set()
        agent = self.context.agent
        if not agent:
            return names

        # The agent's class and ALL parent classes via MRO
        # This prevents LLM from assigning to BaseAgent.method when agent extends BaseAgent
        for cls in type(agent).__mro__:
            if cls is object:
                continue
            names.add(cls.__name__)

        # Class attributes that are themselves classes (sub-agent classes)
        from nooa.agentdoc.visibility import is_hidden_field

        for attr_name in dir(type(agent)):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getattr(type(agent), attr_name, None)
                if isinstance(attr, type):
                    if not is_hidden_field(type(agent), attr_name):
                        names.add(attr_name)
            except Exception:
                continue

        return names

    def _is_type_self_call(self, node: ast.expr) -> bool:
        """Check if expression is type(self) or type(self_alias) call.

        Returns True for:
        - type(self)
        - type(var) where var was assigned from self (e.g., agent = self; type(agent))
        """
        if not isinstance(node, ast.Call):
            return False
        if not isinstance(node.func, ast.Name):
            return False
        if node.func.id != "type":
            return False
        if len(node.args) != 1:
            return False
        arg = node.args[0]
        if not isinstance(arg, ast.Name):
            return False
        # Direct type(self) call
        if arg.id == "self":
            return True
        # type(var) where var is an alias for self
        if arg.id in self.self_ref_vars:
            return True
        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for ClassName.attr = value and track self/type(self) assignments."""
        # Track variables assigned from self (e.g., agent = self)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.self_ref_vars.add(target.id)

        # Track variables assigned from type(self)
        if self._is_type_self_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_ref_vars.add(target.id)

        for target in node.targets:
            self._check_class_attribute_target(target, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Check for ClassName.attr += value patterns."""
        self._check_class_attribute_target(node.target, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check for ClassName.attr: Type = value patterns."""
        if node.value is not None:  # Has assignment, not just annotation
            # Track variables assigned from self (e.g., agent: MyAgent = self)
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if isinstance(node.target, ast.Name):
                    self.self_ref_vars.add(node.target.id)

            # Track variables assigned from type(self)
            if self._is_type_self_call(node.value):
                if isinstance(node.target, ast.Name):
                    self.class_ref_vars.add(node.target.id)

            self._check_class_attribute_target(node.target, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for setattr(ClassName, 'attr', value) patterns."""
        if isinstance(node.func, ast.Name) and node.func.id == "setattr":
            if len(node.args) >= 2:
                obj_arg = node.args[0]

                # Check setattr(ClassName, ...)
                if isinstance(obj_arg, ast.Name):
                    if obj_arg.id in self.known_class_names:
                        self.issues.append(
                            ValidationIssue(
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Cannot use setattr() on class '{obj_arg.id}'. "
                                f"This would corrupt all instances. "
                                f"Use setattr(self, ...) to modify the instance instead.",
                                code="E402",
                                severity="error",
                                fix_hint=f"setattr(self, ...) instead of setattr({obj_arg.id}, ...)",
                            )
                        )
                    elif obj_arg.id in self.class_ref_vars:
                        self.issues.append(
                            ValidationIssue(
                                line=node.lineno,
                                col=node.col_offset,
                                message=f"Cannot use setattr() on '{obj_arg.id}' (assigned from type(self)). "
                                f"This would corrupt all instances. "
                                f"Use setattr(self, ...) to modify the instance instead.",
                                code="E402",
                                severity="error",
                                fix_hint="setattr(self, ...) instead",
                            )
                        )

                # Check setattr(type(self), ...)
                elif self._is_type_self_call(obj_arg):
                    self.issues.append(
                        ValidationIssue(
                            line=node.lineno,
                            col=node.col_offset,
                            message="Cannot use setattr() on type(self). "
                            "This would corrupt all instances. "
                            "Use setattr(self, ...) to modify the instance instead.",
                            code="E402",
                            severity="error",
                            fix_hint="setattr(self, ...) instead of setattr(type(self), ...)",
                        )
                    )

        self.generic_visit(node)

    def _check_class_attribute_target(
        self, target: ast.expr, node: ast.Assign | ast.AugAssign | ast.AnnAssign
    ) -> None:
        """Check if assignment target is a class attribute."""
        if not isinstance(target, ast.Attribute):
            return

        # Check for type(self).attr = value (inline pattern)
        if self._is_type_self_call(target.value):
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message="Cannot assign to type(self) attribute. "
                    "This would corrupt all instances. "
                    "Use 'self.attr = ...' to assign to the instance instead.",
                    code="E401",
                    severity="error",
                    fix_hint=f"self.{target.attr} = ... instead of type(self).{target.attr} = ...",
                )
            )
            return

        # Only check direct Name.attr patterns (not chained like self.obj.attr)
        if not isinstance(target.value, ast.Name):
            return

        name = target.value.id

        # Skip 'self' - that's instance assignment, which is fine
        if name == "self":
            return

        # Check if name is a known class
        if name in self.known_class_names:
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Cannot assign to class attribute '{name}.{target.attr}'. "
                    f"This would corrupt all instances. "
                    f"Use 'self.{target.attr} = ...' to assign to the instance instead.",
                    code="E401",
                    severity="error",
                    fix_hint=f"self.{target.attr} = ... instead of {name}.{target.attr} = ...",
                )
            )
        # Check if name is a variable holding a class reference (from type(self))
        elif name in self.class_ref_vars:
            self.issues.append(
                ValidationIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Cannot assign to '{name}.{target.attr}' ('{name}' holds a class reference from type(self)). "
                    f"This would corrupt all instances. "
                    f"Use 'self.{target.attr} = ...' to assign to the instance instead.",
                    code="E401",
                    severity="error",
                    fix_hint=f"self.{target.attr} = ... instead of {name}.{target.attr} = ...",
                )
            )


# =============================================================================
# Return-Type Shadow Validator
# =============================================================================
def _collect_type_names(annotation: Any, namespace: dict[str, Any] | None = None) -> set[str]:
    """Collect concrete class names referenced by a return-type annotation.

    Walks ``Annotated[...]``, ``list[T]``, ``dict[K, V]``, ``T | U``, ``Optional[T]``,
    etc., and returns the names of any classes encountered. Standard typing
    constructs like ``Union``, ``Optional`` and the ``typing`` module itself
    are skipped — only names of concrete user-visible classes are returned.
    Builtin types (``str``, ``int``, ``UnionType``, ...) are skipped at every
    level so they don't pollute the protected set.

    String forward references (``"Answer"``, ``ForwardRef("Answer")``) — which
    can leak through when ``from __future__ import annotations`` is in effect
    and the framework's ``get_type_hints()`` call raised on construction — are
    resolved against ``namespace`` if provided. If the name isn't in the
    namespace, the walker degrades to a no-op for that branch rather than
    raising; we'd rather under-protect than crash on a perfectly fine helper
    class definition.
    """

    def is_user_class(t: Any) -> bool:
        if not isinstance(t, type):
            return False
        module = getattr(t, "__module__", None)
        # Skip stdlib type machinery: builtins (``str``, ``int``, ``list``),
        # runtime typing internals (``types.UnionType``, ``typing`` aliases),
        # and the abstract collection types (``collections.abc.Iterable`` /
        # ``Callable`` show up as the origin of ``Iterable[T]`` / ``Callable[..., T]``).
        # None of these are names an agent would meaningfully redefine.
        return module not in {"builtins", "types", "typing", "collections.abc"}

    def resolve_forward_ref(name: str) -> Any:
        """Look up a string name in the agent's exec_globals, if available."""
        if namespace is None:
            return None
        return namespace.get(name)

    names: set[str] = set()

    def visit(node: Any) -> None:
        if node is None or node is type(None):
            return
        # String forward reference — resolve via the namespace.
        if isinstance(node, str):
            resolved = resolve_forward_ref(node)
            if resolved is not None:
                visit(resolved)
            return
        # ForwardRef objects (constructed by typing internals) wrap a name.
        if isinstance(node, ForwardRef):
            resolved = resolve_forward_ref(node.__forward_arg__)
            if resolved is not None:
                visit(resolved)
            return
        # Unwrap Annotated[T, ...] to its base type.
        if get_origin(node) is Annotated:
            args = get_args(node)
            if args:
                visit(args[0])
            return
        # Generic alias like list[Answer], dict[str, Answer], Foo | Bar.
        origin = get_origin(node)
        if origin is not None:
            if is_user_class(origin):
                names.add(origin.__name__)
            for arg in get_args(node):
                visit(arg)
            return
        # Bare class.
        if is_user_class(node):
            names.add(node.__name__)

    visit(annotation)
    return names


class ReturnTypeShadowValidator:
    """Reject code that shadows the method's return type name.

    Catches three patterns:
    1. ``class Answer(BaseModel): ...`` — local class shadows the return type
       (creates a distinct type that fails isinstance; see gl-143)
    2. ``def Answer(...): ...`` — local function shadows the name
    3. ``Answer = ...`` — assignment overwrites the type reference (the model
       may set it to None or a wrong value, breaking later return_result calls)

    Generated code runs inside ``async def __repl_wrapper__():`` (see
    ``runtime/actor.py``), so any local binding of the return type name
    creates a scoped shadow that breaks ``return_result()`` validation.

    The validator looks up the class names referenced by the method's declared
    return type (via ``ValidationContext.return_type``) and rejects any local
    definition or assignment that would shadow them. Helpers with unrelated
    names (``def gcd(...)``, ``x = 42``) are not affected.
    """

    def validate(self, tree: ast.AST, context: ValidationContext) -> list[ValidationIssue]:
        protected = _collect_type_names(context.return_type, context.exec_globals)
        if not protected:
            return []

        # ``UnifiedCodeValidator.validate`` always parses with ``ast.parse(code)``,
        # which returns an ``ast.Module``. The Validator protocol uses ``ast.AST``
        # for flexibility; narrow here so .body access type-checks.
        if not isinstance(tree, ast.Module):
            return []

        issues: list[ValidationIssue] = []
        for node in tree.body:
            kind: str | None = None
            name: str | None = None
            if isinstance(node, ast.ClassDef):
                kind = "class"
                name = node.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
                name = node.name
            elif isinstance(node, ast.Assign):
                # Check if any assignment target shadows a protected name
                for target in node.targets:
                    shadowed = self._assignment_target_names(target) & protected
                    if shadowed:
                        kind = "assignment"
                        name = next(iter(shadowed))
                        break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in protected:
                    kind = "assignment"
                    name = node.target.id
            if kind is None or name is None:
                continue
            if name not in protected:
                continue

            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().return_type_redefined(name)

            if kind == "assignment":
                issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            f"Cannot reassign '{name}' — it is the return type of this "
                            f"method. Overwriting it will break return_result() validation. "
                            f"Use '{name}(...)' directly to construct your result."
                        ),
                        code="E501",
                        severity="error",
                        fix_hint=(
                            f"remove '{name} = ...' — '{name}' is already available; "
                            f"construct it directly with {name}(...)"
                        ),
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            f"Cannot redefine '{name}' here — it is already in scope as "
                            f"the return type of this method. A local {kind} definition "
                            f"shadows it with a __repl_wrapper__-scoped {kind}, and "
                            f"return_result() will reject the resulting value as the wrong "
                            f"type. Use the existing '{name}' (already imported) instead."
                        ),
                        code="E501",
                        severity="error",
                        fix_hint=(
                            f"remove the local '{kind} {name}(...)' — '{name}' is "
                            f"already available; construct it directly with "
                            f"{name}(...)"
                        ),
                    )
                )
        return issues

    @staticmethod
    def _assignment_target_names(target: ast.AST) -> set[str]:
        """Extract all names from an assignment target (handles tuple unpacking and starred)."""
        names: set[str] = set()
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Starred):
            # *Answer in tuple unpacking — unwrap to get the Name
            if isinstance(target.value, ast.Name):
                names.add(target.value.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
                elif isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
                    names.add(elt.value.id)
        return names


# =============================================================================
# Blocking Call Validator
# =============================================================================
class BlockingCallValidator:
    """Validates code for blocking calls that would freeze the event loop.

    Resolves AST names against exec_globals to determine module of origin.
    Replaces AsyncSafetyValidator with runtime-aware name resolution instead
    of string matching.

    Note: This validator does NOT check for missing ``await`` on async calls.
    For example, ``asyncio.sleep(1)`` (without ``await``) passes this validator
    because ``asyncio.sleep`` is not a blocking call — it's an async function
    called incorrectly. The missing ``await`` is caught by REPLPolicyValidator
    (error code E301). Both validators must be active for complete coverage.
    """

    def __init__(
        self,
        restrictions: RestrictionsConfig | None = None,
    ):
        self.restrictions = restrictions or RestrictionsConfig()

    def validate(self, tree: ast.AST, context: ValidationContext) -> list[ValidationIssue]:
        visitor = _BlockingCallVisitor(
            exec_globals=context.exec_globals,
            restrictions=self.restrictions,
        )
        visitor.visit(tree)
        return visitor.issues


class _BlockingCallVisitor(ast.NodeVisitor):
    """AST visitor that detects blocking calls using runtime-resolved names."""

    def __init__(
        self,
        exec_globals: dict[str, Any],
        restrictions: RestrictionsConfig,
    ):
        self.exec_globals = exec_globals
        self.blocked_modules = restrictions.blocked_modules
        self.blocked_calls = restrictions.blocked_calls
        self.issues: list[ValidationIssue] = []
        # Track local variables assigned from constructors on blocked-call modules.
        # Maps var_name -> (module_name, class_name)
        self.tracked_locals: dict[str, tuple[str, str]] = {}

    def _resolve_module_from_call(self, node: ast.Call) -> str | None:
        """Resolve the module of a chained call like asyncio.get_event_loop().

        For `asyncio.get_event_loop().run_until_complete(...)`, the inner call
        `asyncio.get_event_loop()` has its function's value as the `asyncio` Name.
        We resolve that to the asyncio module.
        """
        if isinstance(node.func, ast.Attribute):
            return self._resolve_module(node.func.value)
        return None

    def _resolve_module(self, node: ast.expr) -> str | None:
        """Resolve an AST expression to its module name via exec_globals.

        For non-module objects, falls back to obj.__module__. Same caveat as
        is_from_blocked_module(): safe with curated block lists but could
        over-match if a broadly-used module like "io" were added.
        """
        if isinstance(node, ast.Name):
            obj = self.exec_globals.get(node.id)
            if isinstance(obj, types.ModuleType):
                return obj.__name__
            return getattr(obj, "__module__", None)
        if isinstance(node, ast.Attribute):
            # For chained attributes like os.path.join, resolve the leftmost name
            return self._resolve_module(node.value)
        return None

    def _add_issue(self, node: ast.AST, module_name: str, call_name: str) -> None:
        self.issues.append(
            ValidationIssue(
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=(
                    f"{module_name}.{call_name}() blocks the event loop and is not "
                    f"allowed in agent code. Use 'await' with an async alternative "
                    f"or an appropriate agent tool."
                ),
                code="E310",
                severity="error",
            )
        )

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Track local variables assigned from constructors on blocked-call modules."""
        # Track: t = threading.Thread(...) -> tracked_locals["t"] = ("threading", "Thread")
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            module_name = self._resolve_module(node.value.func.value)
            if module_name:
                matched = match_blocked_module(module_name, self.blocked_calls)
                if matched:
                    class_name = node.value.func.attr
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.tracked_locals[target.id] = (matched, class_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Check each call against blocked modules and blocked calls."""
        if isinstance(node.func, ast.Attribute):
            # e.g., subprocess.run(), time.sleep(), t.join()
            module_name = self._resolve_module(node.func.value)
            call_name = node.func.attr

            if module_name:
                # Check fully blocked modules
                matched = match_blocked_module(module_name, self.blocked_modules)
                if matched:
                    self._add_issue(node, matched, call_name)
                    return self.generic_visit(node)

                # Check partially blocked calls
                matched = match_blocked_module(module_name, self.blocked_calls)
                if matched:
                    if call_name in self.blocked_calls[matched]:
                        self._add_issue(node, matched, call_name)
                        return self.generic_visit(node)

            # Check local variable tracking (for Thread.join, Lock.acquire etc.)
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if var_name in self.tracked_locals:
                    tracked_module, class_name = self.tracked_locals[var_name]
                    if tracked_module in self.blocked_calls:
                        blocked = self.blocked_calls[tracked_module]
                        dotted = f"{class_name}.{call_name}"
                        if dotted in blocked or call_name in blocked:
                            self._add_issue(node, tracked_module, call_name)

            # Check chained calls: e.g. asyncio.get_event_loop().run_until_complete()
            if isinstance(node.func.value, ast.Call):
                chained_module = self._resolve_module_from_call(node.func.value)
                if chained_module:
                    matched = match_blocked_module(chained_module, self.blocked_calls)
                    if matched and call_name in self.blocked_calls[matched]:
                        self._add_issue(node, matched, call_name)

        elif isinstance(node.func, ast.Name):
            # e.g., run(['ls']) where run = subprocess.run
            obj = self.exec_globals.get(node.func.id)
            if obj is not None:
                obj_module = getattr(obj, "__module__", None)
                if obj_module:
                    matched_blocked = match_blocked_module(obj_module, self.blocked_modules)
                    if matched_blocked:
                        self._add_issue(node, matched_blocked, node.func.id)
                    else:
                        matched_calls = match_blocked_module(obj_module, self.blocked_calls)
                        if matched_calls:
                            fn_name = getattr(obj, "__name__", node.func.id)
                            if fn_name in self.blocked_calls[matched_calls]:
                                self._add_issue(node, matched_calls, fn_name)

        self.generic_visit(node)


# =============================================================================
# Unified Code Validator
# =============================================================================
class UnifiedCodeValidator:
    """Orchestrates multiple validators with consistent error handling.

    This is the main entry point for code validation. It runs all validators
    and formats errors in IPython-style with source context.

    By default, includes SecurityValidator and BlockingCallValidator,
    which are the checks performed by execute_code(). The REPLPolicyValidator
    (class definitions, missing await) is used by strategies separately.
    """

    def __init__(
        self,
        validators: list[Validator] | None = None,
        *,
        include_repl_policy: bool = False,
        restrictions: RestrictionsConfig | None = None,
    ):
        """Initialize with validators.

        Args:
            validators: List of validators to use. If provided, overrides defaults.
            include_repl_policy: If True, include REPLPolicyValidator (class defs,
                missing await). Default False since strategies handle this separately.
            restrictions: Code execution restrictions (blocked modules/calls).
                None uses defaults from RestrictionsConfig().
        """
        if validators is not None:
            self.validators: list[Validator] = validators
        else:
            self.validators = [
                SecurityValidator(),
                BlockingCallValidator(
                    restrictions=restrictions,
                ),
                ClassAssignmentValidator(),
                ReturnTypeShadowValidator(),
            ]
            if include_repl_policy:
                self.validators.append(REPLPolicyValidator())

    def validate(
        self,
        code: str,
        context: ValidationContext,
        *,
        stop_on_first_error: bool = True,
    ) -> None:
        """Validate code against all registered validators.

        Args:
            code: Python source code to validate
            context: Validation context with settings
            stop_on_first_error: If True, stop at first error (default)

        Raises:
            ValidationError: With IPython-style formatting
        """
        # Handle empty/whitespace code
        if not code or not code.strip():
            return

        # Update context with code
        context.code = code

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValidationError(f"Syntax error: {e}", original_exception=e) from e

        # Run validators
        all_issues: list[ValidationIssue] = []
        for validator in self.validators:
            issues = validator.validate(tree, context)
            all_issues.extend(issues)

            if stop_on_first_error and any(i.severity == "error" for i in issues):
                break

        # Format and raise errors
        errors = [i for i in all_issues if i.severity == "error"]
        if errors:
            raise ValidationError(self._format_error(code, errors[0], context))

        # Log warnings
        warnings = [i for i in all_issues if i.severity == "warning"]
        for warning in warnings:
            logger.warning(f"Validation warning [{warning.code}]: {warning.message}")

    def _format_error(self, code: str, issue: ValidationIssue, context: ValidationContext) -> str:
        """Format error in IPython style with source context."""
        lines = code.split("\n")
        source_line = lines[issue.line - 1] if 1 <= issue.line <= len(lines) else ""

        cell_name = f"Cell In[{context.execution_count}]"
        indent = "    "
        caret = " " * issue.col + "^"

        parts = [
            f"{cell_name}, line {issue.line}",
            f"{indent}{source_line}",
            f"{indent}{caret}",
            issue.message,
        ]

        if issue.fix_hint:
            parts.append(f"\nFix: {issue.fix_hint}")

        if issue.doc_link:
            parts.append(f"\nSee: {issue.doc_link}")

        return "\n".join(parts)


# =============================================================================
# Import Pre-processing
# =============================================================================
def strip_redundant_imports(code: str, available_names: set[str]) -> tuple[str, list[str]]:
    """Remove import statements where all imported names are already in scope.

    LLMs habitually prepend imports (``from typing import Literal``,
    ``from strategy import ...``) even when those names are pre-loaded.
    Rather than erroring, we silently strip these lines so both the
    validator and the runtime see clean code.

    Only strips an import when *every* name it would introduce is already
    present in ``available_names``.  Imports that bring genuinely new names
    are left untouched (and will be caught by the security validator if
    the module is forbidden).

    Returns:
        Tuple of (cleaned_code, stripped_statements) where stripped_statements
        is a list of the full original import source lines that were removed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    indices_to_remove: set[int] = set()

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Import):
            # ``import X`` / ``import X as Y`` — check alias or module name
            all_present = all(
                (alias.asname or alias.name) in available_names for alias in node.names
            )
            if all_present:
                indices_to_remove.add(i)

        elif isinstance(node, ast.ImportFrom):
            # ``from X import a, b`` — check each imported name
            if any(alias.name == "*" for alias in node.names):
                continue  # never strip star imports
            all_present = all(
                (alias.asname or alias.name) in available_names for alias in node.names
            )
            if all_present:
                indices_to_remove.add(i)

    if not indices_to_remove:
        return code, []

    # Collect the original source text for each stripped import (for telemetry).
    source_lines_for_stmts = code.splitlines()
    stripped_statements: list[str] = []
    for i, node in enumerate(tree.body):
        if i in indices_to_remove:
            # Reconstruct the import statement from its source line range.
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno) - 1
            stmt_lines = source_lines_for_stmts[start : end + 1]
            # Strip leading/trailing whitespace for a clean record.
            stripped_statements.append("\n".join(stmt_lines).strip())

    # Collect the 1-based line numbers covered by each removed import node.
    # Use lineno/end_lineno so multi-line imports are fully removed.
    lines_to_remove: set[int] = set()
    for i, node in enumerate(tree.body):
        if i in indices_to_remove:
            for line_num in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                lines_to_remove.add(line_num)

    # Handle the semicolon edge case: a kept node starts on the same physical
    # line as a removed import (e.g. `from typing import Literal; x = 1`).
    # We reconstruct using the original source character ranges so that
    # inline comments and multi-line formatting are preserved exactly.
    source_lines = code.splitlines(keepends=True)

    # Group kept nodes by starting line, but only for mixed lines.
    kept_on_removed: dict[int, list[ast.stmt]] = {}
    for i, node in enumerate(tree.body):
        if i not in indices_to_remove and node.lineno in lines_to_remove:
            kept_on_removed.setdefault(node.lineno, []).append(node)

    # For each mixed line, extract the kept nodes' text from the original source.
    # For the last (rightmost) kept node, we take from its col_offset to end of
    # the physical line — this picks up any trailing comment as well as the
    # opening of a multi-line expression.  For preceding nodes we take the exact
    # character range [col_offset:end_col_offset] to avoid including the import.
    # Multi-line kept nodes: only the first line is reconstructed here;
    # continuation lines are not in lines_to_remove so they emit normally below.
    reconstructed: dict[int, list[str]] = {}
    for mixed_line, nodes in kept_on_removed.items():
        raw_line = source_lines[mixed_line - 1].rstrip("\n\r")
        nodes_sorted = sorted(nodes, key=lambda n: n.col_offset)

        # Effective end column on this line: multi-line nodes extend to EOL.
        def eff_end(n: ast.stmt, line: str) -> int:
            end_lineno = n.end_lineno or n.lineno
            return len(line) if end_lineno > n.lineno else (n.end_col_offset or 0)

        max_end = max(eff_end(n, raw_line) for n in tree.body if n.lineno == mixed_line)

        parts: list[str] = []
        for j, node in enumerate(nodes_sorted):
            is_rightmost_last = j == len(nodes_sorted) - 1 and eff_end(node, raw_line) == max_end
            if is_rightmost_last or (node.end_lineno or node.lineno) > node.lineno:
                # Take to end of physical line: captures trailing comment and
                # the opening of any parenthesised multi-line expression.
                parts.append(raw_line[node.col_offset :])
            else:
                # Exact range only — more nodes follow on this line.
                parts.append(raw_line[node.col_offset : node.end_col_offset])
        reconstructed[mixed_line] = parts

    # Reconstruct source by dropping only removed lines, preserving comments,
    # blank lines, and original line numbering so validation error messages
    # reference the correct lines when shown back to the LLM.
    # Continuation lines of multi-line kept nodes are not in lines_to_remove
    # and emit verbatim via the else branch below.
    result_parts: list[str] = []
    for line_num, line in enumerate(source_lines, start=1):
        if line_num in reconstructed:
            for text in reconstructed[line_num]:
                result_parts.append(text + "\n")
        elif line_num in lines_to_remove:
            pass  # pure import line — drop it
        else:
            result_parts.append(line)
    return "".join(result_parts), stripped_statements


# =============================================================================
# Convenience Functions
# =============================================================================
def validate_code(
    code: str,
    *,
    agent_class: type | None = None,
    available_names: list[str] | None = None,
    importable_modules: set[str] | None = None,
    restricted_imports: frozenset[str] | None = None,
    blocked_modules: frozenset[str] | None = None,
    forbidden_self_calls: list[str] | None = None,
    execution_count: int = 1,
    agent: Any = None,
    return_type: Any = None,
) -> None:
    """Convenience function to validate code.

    This wraps UnifiedCodeValidator for backwards compatibility and convenience.

    Args:
        code: Python source code to validate
        agent_class: Agent class for decorator checking
        available_names: Names available in scope
        importable_modules: Deprecated — use restricted_imports instead
        restricted_imports: Deny list of module names. None uses RestrictionsConfig default.
        blocked_modules: Hard-blocked modules. None uses RestrictionsConfig default.
        forbidden_self_calls: Method names that can't be called on self
        execution_count: Execution count for Cell In[N] format
        agent: Agent instance for method introspection
        return_type: Return type annotation of the executing method, used by
            ReturnTypeShadowValidator to reject local class definitions that
            would shadow the return type in __repl_wrapper__ scope.

    Raises:
        ValidationError: If validation fails
    """
    from nooa.runtime.restrictions import RestrictionsConfig

    rc = RestrictionsConfig()
    context = ValidationContext(
        code=code,
        agent_class=agent_class,
        available_names=set(available_names or []),
        importable_modules=importable_modules or set(),
        restricted_imports=restricted_imports
        if restricted_imports is not None
        else rc.restricted_imports,
        blocked_modules=blocked_modules if blocked_modules is not None else rc.blocked_modules,
        forbidden_self_calls=set(forbidden_self_calls or []),
        execution_count=execution_count,
        agent=agent,
        return_type=return_type,
    )
    validator = UnifiedCodeValidator()
    validator.validate(code, context)
