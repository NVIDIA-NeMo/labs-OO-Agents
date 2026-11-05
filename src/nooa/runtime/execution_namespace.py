# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental coherent cell compiler; not selected by production strategies.

The async wrapper preserves Python's return/await/finally semantics. Only its
bindings are lifted into one authoritative globals dictionary. Runtime names
are resolved through Python's builtin fallback, *after* ordinary lexical and
class-namespace lookup, rather than rewriting identifiers or refreshing old
function objects. No user definition is executed during compilation.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextvars
import linecache
import symtable
import types
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

_INTERNAL = "_nooa_ns_"
_SOURCE = _INTERNAL + "source"
_DELETE = _INTERNAL + "delete"
RUNTIME_NAMES = frozenset(
    {
        "self",
        "_call",
        "print",
        "pprint",
        "show",
        "return_result",
        "python_cell_state",
        "Out",
    }
)


def framework_name_allowed(name: str, reserved: frozenset[str]) -> bool:
    """Only auto-discovered provider names, never protected runtime facilities."""
    return name not in reserved and not name.startswith(_INTERNAL) and name != "__builtins__"


def _source_decorator(source: str):
    def attach(function):
        function._generated_source = source
        return function

    return attach


class _CellBindings(ast.NodeVisitor):
    """Collect binding syntax in the wrapper, never nested lexical locals.

    symtable alone includes PEP-709 inlined comprehension iteration variables
    on 3.12/3.13. Intersecting it with this scope-aware syntax pass excludes
    those variables while keeping comprehension walrus bindings.
    """

    def __init__(self):
        self.names: set[str] = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node):
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(self, node):
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)

    def _defaults(self, args):
        for value in [*args.defaults, *args.kw_defaults]:
            if value is not None:
                self.visit(value)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)
        self._defaults(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        # Function annotations are expressions in the enclosing scope (unlike
        # variable annotations). The compiler handles future annotations.
        for arg in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ]:
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._defaults(node.args)

    def visit_ClassDef(self, node):
        self.names.add(node.name)
        for value in [*node.bases, *node.keywords, *node.decorator_list]:
            self.visit(value)

    def visit_comprehension(self, node):
        # The target belongs to the comprehension, not to the cell.
        self.visit(node.iter)
        for condition in node.ifs:
            self.visit(condition)

    def visit_AnnAssign(self, node):
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)
        # Local variable annotations are intentionally unevaluated.


class _LocalAnnotations(ast.NodeTransformer):
    """Normalize only cell-local name annotations before global lifting."""

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_AnnAssign(self, node):
        if not isinstance(node.target, ast.Name):
            return node  # attribute/subscript annotations already have local semantics
        if node.value is None:
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


class _ScopeGlobals(ast.NodeVisitor):
    """Global declarations in exactly one statement scope, including branches."""

    def __init__(self):
        self.names: set[str] = set()

    def visit_Global(self, node):
        self.names.update(node.names)

    def visit_FunctionDef(self, node):
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


class _GlobalDeletes(ast.NodeTransformer):
    """Tombstone executed global del, not locals/class attributes/nonlocals.

    The wrapper already declares all cell bindings global. In nested scopes,
    a deleted Name is local unless explicitly declared global (Python's own
    rule). This needs no name-load rewriting or global-store interception.
    """

    def __init__(self):
        self.global_names: set[str] = set()
        self.class_name: str | None = None

    def _binding_name(self, name: str) -> str:
        # CPython mangles identifier operands, but not the string constant in
        # our deletion callback. Keep that constant in the same binding space.
        if self.class_name and name.startswith("__") and not name.endswith("__"):
            prefix = self.class_name.lstrip("_")
            if prefix:  # an all-underscore class name disables mangling
                return "_" + prefix + name
        return name

    def visit_FunctionDef(self, node):
        previous = self.global_names
        declarations = _ScopeGlobals()
        for statement in node.body:
            declarations.visit(statement)
        self.global_names = {self._binding_name(name) for name in declarations.names}
        try:
            return self.generic_visit(node)
        finally:
            self.global_names = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        previous = self.class_name
        self.class_name = node.name
        try:
            return self.visit_FunctionDef(node)
        finally:
            self.class_name = previous

    def visit_Delete(self, node):
        statements = []

        def target(value):
            if isinstance(value, (ast.Tuple, ast.List)):
                for item in value.elts:
                    target(item)
            elif isinstance(value, ast.Name) and self._binding_name(value.id) in self.global_names:
                statements.append(
                    ast.copy_location(
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id=_DELETE, ctx=ast.Load()),
                                args=[ast.Constant(self._binding_name(value.id))],
                                keywords=[],
                            )
                        ),
                        value,
                    )
                )
            else:
                statements.append(ast.copy_location(ast.Delete(targets=[self.visit(value)]), value))

        for value in node.targets:
            target(value)
        return statements


class _SourceMetadata(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        source = ast.unparse(node)
        self.generic_visit(node)
        # Innermost: user decorator expressions still evaluate in source order;
        # metadata is attached before any user decorator (including @strategy).
        node.decorator_list.append(
            ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_SOURCE, ctx=ast.Load()),
                    args=[ast.Constant(source)],
                    keywords=[],
                ),
                node,
            )
        )
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def _check_reserved(table: symtable.SymbolTable, reserved: frozenset[str]) -> None:
    for symbol in table.get_symbols():
        name = symbol.get_name()
        if name.startswith(_INTERNAL) or name == "__builtins__":
            raise SyntaxError(f"{name!r} is reserved for the cell runtime")
        if (
            name in reserved
            and symbol.is_global()
            and (symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace())
        ):
            raise SyntaxError(f"Cannot write or delete reserved runtime global {name!r}")
    for child in table.get_children():
        _check_reserved(child, reserved)


@dataclass(frozen=True)
class CompiledCell:
    code: types.CodeType
    wrapper_name: str
    bindings: frozenset[str]
    dependencies: frozenset[str]


def compile_cell(source: str, filename: str, reserved: frozenset[str]) -> CompiledCell:
    """Compile once; definitions/defaults/decorators run only on cell execution."""
    name = _INTERNAL + "cell_" + uuid4().hex
    # Parse before wrapping: indenting source text changes multiline literals
    # and escaped newlines. Move the original nodes, including their locations,
    # into a synthetic async function so await/return keep function semantics.
    tree = ast.parse(source, filename=filename)
    wrapper = ast.parse("async def " + name + "(): pass", filename=filename).body[0]
    wrapper.body = tree.body or wrapper.body
    wrapper.end_lineno = wrapper.body[-1].end_lineno
    wrapper.end_col_offset = wrapper.body[-1].end_col_offset
    tree.body = [wrapper]
    # First validate genuine Python scoping (including nonlocal and global
    # annotation diagnostics) before normalization changes the compiler input.
    # Unparsed text is only for scope analysis; executable nodes stay original.
    original_table = symtable.symtable(ast.unparse(tree), filename, "exec").get_children()[0]
    bindings = _CellBindings()
    for statement in wrapper.body:
        bindings.visit(statement)
    lifted = sorted(name for name in bindings.names if original_table.lookup(name).is_local())
    normalizer = _LocalAnnotations()
    wrapper.body = [normalizer.visit(statement) for statement in wrapper.body]
    if lifted:
        wrapper.body.insert(0, ast.copy_location(ast.Global(names=lifted), wrapper.body[0]))
    ast.fix_missing_locations(tree)
    # Re-run the actual compiler's scope analysis after lifting. This catches
    # writes in helpers, including explicit globals, without mistaking method
    # parameters/class attributes/comprehension targets for runtime globals.
    table = symtable.symtable(ast.unparse(tree), filename, "exec").get_children()[0]
    _check_reserved(table, reserved)
    dependencies: set[str] = set()
    pending = list(table.get_children())
    while pending:
        scope = pending.pop()
        dependencies.update(
            symbol.get_name() for symbol in scope.get_symbols() if symbol.is_global()
        )
        pending.extend(scope.get_children())
    metadata = _SourceMetadata()
    wrapper.body = [metadata.visit(statement) for statement in wrapper.body]
    _GlobalDeletes().visit(wrapper)
    ast.fix_missing_locations(tree)
    return CompiledCell(
        compile(tree, filename, "exec", dont_inherit=True),
        name,
        frozenset(bindings.names),
        frozenset(dependencies),
    )


@dataclass
class CallBindings:
    owner: ExecutionNamespace
    values: Mapping[str, Any]
    active: bool = True
    cell_task: asyncio.Task | None = None


_calls: contextvars.ContextVar[tuple[CallBindings, ...]] = contextvars.ContextVar(
    "execution_namespace_calls", default=()
)


class _RuntimeBuiltins(dict):
    """Use Python's own scope lookup; never replace names by their spelling.

    An ordinary globals dict is essential: CPython's class LOAD_NAME bypasses
    dict.__missing__ on globals. Builtin fallback supports a mapping for both
    LOAD_NAME and LOAD_GLOBAL, including adaptive specialization.
    """

    def __init__(self, owner: ExecutionNamespace):
        super().__init__((k, v) for k, v in builtins.__dict__.items() if k not in owner.reserved)
        self.owner = owner

    def __getitem__(self, name):
        # Auto-discovered framework classes are fallback availability, never
        # user globals. A normal Python global store (even same-object rebind)
        # therefore wins without instrumentation or object-graph repair.
        if name not in self.owner.deleted_names and name in self.owner.framework:
            return self.owner.framework[name]
        return super().__getitem__(name)

    def __missing__(self, name):
        if name in self.owner.reserved:
            for call in reversed(_calls.get()):
                if call.owner is self.owner:
                    if name in call.values:
                        return call.values[name]
                    break
            raise RuntimeError(f"Runtime name {name!r} has no originating call context")
        raise KeyError(name)


def _discard_sources(entries: dict[str, tuple]) -> None:
    # Entries contain only source text/metadata, never the namespace. A reused
    # filename may now belong to another owner (even with identical contents).
    for filename, entry in entries.items():
        if linecache.cache.get(filename) is entry:
            del linecache.cache[filename]


class ExecutionNamespace:
    """Experimental per-owner namespace. Values are never snapshot state.

    Lifecycle selection and output/completion capabilities belong to the
    runtime integration; this module does not enable them automatically.
    """

    __nosnapshot__ = True

    def __init__(self, *, reserved: frozenset[str], initial: Mapping[str, Any] | None = None):
        self.reserved = reserved
        self.identity = uuid4().hex
        self._source_entries: dict[str, tuple] = {}
        # Helpers retain globals -> builtins -> owner, so their traceback source
        # stays available. Discarded owners release only their own cache entries.
        weakref.finalize(self, _discard_sources, self._source_entries)
        self.globals = {k: v for k, v in dict(initial or {}).items() if k not in reserved}
        self.globals["__builtins__"] = _RuntimeBuiltins(self)
        self.globals[_SOURCE] = _source_decorator
        self.globals[_DELETE] = self.delete_global
        self.framework: dict[str, Any] = {}
        self._framework_names: set[str] = set()
        self.deleted_names: set[str] = set()
        self._foreground: CallBindings | None = None
        self._initialized = False
        self.user_names: set[str] = set()
        self.dependency_names: set[str] = set()
        self.epoch = 0

    def update_framework(self, values: Mapping[str, Any]) -> None:
        """Replace only explicitly identified dynamic-class availability.

        Removal never touches user globals/aliases. Historical provider names
        stay excluded from initial seeding, including deactivation before the
        first cell. Static module/core seeds are never refreshed here.
        """
        self.framework = {
            name: value
            for name, value in values.items()
            if framework_name_allowed(name, self.reserved)
        }
        self._framework_names.update(self.framework)

    def initialize(self, values: Mapping[str, Any], *, static_names=()) -> None:
        """Seed the initial non-dynamic environment once, never refresh it.

        Core runtime spellings remain static even if an instance's class name
        happens to collide with one. Explicit initial/user globals always win.
        """
        if not self._initialized:
            for name, value in values.items():
                if (
                    name not in self.reserved
                    and name != "__builtins__"
                    and (name not in self._framework_names or name in static_names)
                ):
                    self.globals.setdefault(name, value)
            self._initialized = True

    def delete_global(self, name: str) -> None:
        """Executed explicit global del suppresses future framework fallback.

        A user store/import can still bind the name normally. Ordinary Python
        builtins are unaffected; deleting a local/class name never calls this.
        Failed deletion has no effect. Implicit exception-alias cleanup remains
        normal Python cleanup, not an explicit provider tombstone.
        """
        if name in self.globals:
            del self.globals[name]
        elif name in self.deleted_names or name not in self.framework:
            raise NameError(f"name {name!r} is not defined", name=name)
        self.deleted_names.add(name)

    def advertised_framework(
        self, values: Mapping[str, Any], *, dynamic_names=()
    ) -> dict[str, Any]:
        """Advertise availability, never an obsolete provider over user state.

        Static seeds are intentionally not refreshed: new/replaced module
        exports must not be advertised as though that refresh happened. Dynamic
        additions are supplied before the next cell executes. Identity checks
        here never inspect/repr a user value.
        """
        visible = {}
        for name, value in values.items():
            if not framework_name_allowed(name, self.reserved):
                continue
            if name in self.globals:
                if self.globals[name] is value:
                    visible[name] = value
            elif name not in self.deleted_names and (
                not self._initialized or name in dynamic_names or name in self.reserved
            ):
                visible[name] = value
        return visible

    def bind_call(self, values: Mapping[str, Any]) -> None:
        call = self._foreground
        if call is None or not call.active or not _calls.get() or _calls.get()[-1] is not call:
            raise RuntimeError("No originating active namespace call in this task context")
        bound = {name: value for name, value in values.items() if name in self.reserved}
        if "return_result" in bound:
            from nooa.runtime.actor import _current_call_var

            # None is the honest identity of a direct low-level execute_code
            # call, not a wildcard allowing entry into an actual generation.
            generation = _current_call_var.get()
            submit = bound["return_result"]

            def authorized_return(*args, **kwargs):
                try:
                    task = asyncio.current_task()
                except RuntimeError:
                    task = None
                if (
                    not call.active
                    or call is not self._foreground
                    or not _calls.get()
                    or _calls.get()[-1] is not call
                    or _current_call_var.get() is not generation
                    or task is not call.cell_task
                    or call.cell_task is None
                ):
                    raise RuntimeError(
                        "return_result belongs to an active foreground cell; "
                        "nested generations, background tasks and finished-call aliases cannot complete it"
                    )
                return submit(*args, **kwargs)

            bound["return_result"] = authorized_return
        call.values = bound

    @contextmanager
    def executing_cell(self):
        call = self._foreground
        if call is None or not call.active or not _calls.get() or _calls.get()[-1] is not call:
            raise RuntimeError("No originating active namespace call in this task context")
        if call.cell_task is not None:
            raise RuntimeError("Concurrent/reentrant cell execution in the same namespace")
        call.cell_task = asyncio.current_task()
        try:
            yield
        finally:
            call.cell_task = None

    def inventory(self) -> dict[str, Any]:
        """Bounded names/types only; no repr, deep walks or object callbacks."""
        names = sorted(name for name in self.user_names if name in self.globals)
        visible: dict[str, str] = {}
        budget = 1800  # leaves room for dict punctuation and omitted count (<2 KB)
        for index, name in enumerate(names[:20]):
            label = name if len(name) <= 80 else name[:70] + f"…#{index}"
            kind = type.__getattribute__(type(self.globals[name]), "__name__")[:80]
            cost = len(label) + len(kind) + 8
            if cost > budget:
                break
            visible[label] = kind
            budget -= cost
        return {"names": visible, "omitted": len(names) - len(visible)}

    def cleanup_snapshot(self) -> tuple[int, dict[str, Any]]:
        # Bound the work too: a large/uncertain namespace is retained wholesale.
        names = self.user_names
        if len(names) > 200:
            return self.epoch, {}
        return self.epoch, {name: self.globals[name] for name in names if name in self.globals}

    def cleanup_candidates(self, snapshot: tuple[int, dict[str, Any]]) -> list[str]:
        """Only primitive bindings with demonstrably absent helper dependencies.

        Any opaque retained object (classes, decorated helpers, containers,
        modules, async helpers) makes dependency analysis uncertain: retain all.
        This intentionally favors false negatives over breaking live helpers.
        """
        _, values = snapshot
        primitive = (str, bytes, int, float, bool, type(None))
        dependencies = set(self.dependency_names)
        if dependencies & {"getattr", "globals", "locals", "vars", "eval", "exec", "__globals__"}:
            return []
        for value in values.values():
            if type(value) in primitive:
                continue
            if type(value) is not types.FunctionType or value.__globals__ is not self.globals:
                return []
            if value.__closure__ or value.__code__.co_flags & (0x20 | 0x80 | 0x200):
                return []  # generators/coroutines may still be running
            pending = [value.__code__]
            visited = 0
            while pending:
                code = pending.pop()
                visited += 1
                if visited > 100:
                    return []
                dependencies.update(code.co_names)
                if set(code.co_names) & {
                    "getattr",
                    "globals",
                    "locals",
                    "vars",
                    "eval",
                    "exec",
                    "create_task",
                    "ensure_future",
                    "call_later",
                    "call_soon",
                }:
                    return []
                pending.extend(c for c in code.co_consts if type(c) is types.CodeType)
        return sorted(
            name
            for name, value in values.items()
            if type(value) in primitive and name not in dependencies and len(name) <= 80
        )[:20]

    def apply_cleanup(
        self, snapshot: tuple[int, dict[str, Any]], names: list[str], *, active_work: bool
    ) -> bool:
        epoch, values = snapshot
        if active_work or self._foreground is not None or self.epoch != epoch:
            return False
        current = {name: self.globals[name] for name in self.user_names if name in self.globals}
        if current.keys() != values.keys() or any(
            current[name] is not value for name, value in values.items()
        ):
            return False
        if not set(names) <= set(self.cleanup_candidates(snapshot)):
            return False
        for name in names:
            self.delete_global(name)  # no close/cancel; also prevent provider resurrection
        if names:
            self.epoch += 1
        return True

    @contextmanager
    def enter(self, values: Mapping[str, Any]) -> Iterator[CallBindings]:
        if self._foreground is not None:
            raise RuntimeError("Concurrent/reentrant foreground entry into the same namespace")
        call = CallBindings(self, dict(values))
        token = _calls.set((*_calls.get(), call))
        self._foreground = call
        try:
            yield call
        finally:
            call.active = False
            self._foreground = None
            _calls.reset(token)

    def cell(self, source: str, filename: str = "<experimental-cell>"):
        compiled = compile_cell(source, filename, self.reserved)
        # Publish only after successful compilation, but before execution so
        # runtime failures and retained tracebacks can still recover the source.
        entry = (len(source), None, source.splitlines(keepends=True), filename)
        self._source_entries[filename] = entry
        linecache.cache[filename] = entry
        self.user_names.update(compiled.bindings)
        self.dependency_names.update(compiled.dependencies)
        self.epoch += 1
        exec(compiled.code, self.globals)
        function = self.globals.pop(compiled.wrapper_name)
        return function()
