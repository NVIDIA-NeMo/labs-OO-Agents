# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard accidental rebinding of CodeAct's framework helpers, not adversarial code."""

import ast
import unicodedata
from collections.abc import Iterable


def _declares_completion_globals(
    nodes: Iterable[ast.AST], reserved: frozenset[str]
) -> frozenset[str]:
    """Collect reserved global declarations without recursion or entering nested scopes."""
    declared: set[str] = set()
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Global):
            declared.update(reserved.intersection(node.names))
        elif not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            pending.extend(ast.iter_child_nodes(node))
    return frozenset(declared)


class _CompletionBindings(ast.NodeVisitor):
    """Find writes to the cell/global binding while preserving independent locals."""

    def __init__(self, reserved: frozenset[str]) -> None:
        """Start in the cell scope; None denotes a comprehension's private scope."""
        self.reserved = reserved
        self.scopes: list[frozenset[str] | None] = [reserved]
        self._queued: list[tuple[ast.AST, list[frozenset[str] | None]]] | None = None
        self.errors: list[str] = []

    def visit(self, node: ast.AST) -> None:
        """Walk iteratively, retaining each child's lexical scope and source order."""
        if self._queued is not None:
            self._queued.append((node, self.scopes.copy()))
            return
        pending = [(node, self.scopes.copy())]
        try:
            while pending:
                current, self.scopes = pending.pop()
                children: list[tuple[ast.AST, list[frozenset[str] | None]]] = []
                self._queued = children
                super().visit(current)
                pending.extend(reversed(children))
        finally:
            self._queued = None

    def _binding(
        self,
        name: str | None,
        node: ast.expr | ast.stmt | ast.pattern | ast.alias | ast.ExceptHandler,
        *,
        named_expr: bool = False,
    ) -> None:
        """Report a reserved binding; walruses bind outside comprehension scopes."""
        scope = self.scopes[-1]
        if named_expr:
            scope = next(scope for scope in reversed(self.scopes) if scope is not None)
        if name is not None and scope and name in scope:
            self.errors.append(
                f"Line {node.lineno}: `{name}` is reserved by CodeAct. "
                "Do not rebind or delete it; use a different variable name such as `result`."
            )

    def visit_Name(self, node: ast.Name) -> None:
        """Handle assignment, deletion, unpacking, loop and context-manager targets."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._binding(node.id, node)

    def visit_alias(self, node: ast.alias) -> None:
        """Imports bind their alias, or the first component of a dotted name."""
        self._binding(node.asname or node.name.split(".")[0], node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Exception targets are bindings even though AST stores their names as text."""
        self._binding(node.name, node)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        """Handle both capture and as-pattern bindings."""
        self._binding(node.name, node)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        """Handle a starred sequence-pattern capture."""
        self._binding(node.name, node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        """Handle mapping-rest captures as well as nested patterns."""
        self._binding(node.rest, node)
        self.generic_visit(node)

    def _body(self, body: list[ast.stmt]) -> None:
        """Enter a function/class scope; explicit global writes still affect the helper."""
        self.scopes.append(_declares_completion_globals(body, self.reserved))
        for statement in body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Defaults, decorators and annotations run outside the function's local scope."""
        self._binding(node.name, node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._body(node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """A class name binds outside its body; class attributes are independent locals."""
        self._binding(node.name, node)
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        self._body(node.body)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Lambda defaults run outside its private parameter/body scope."""
        self.visit(node.args)
        self.scopes.append(frozenset())
        self.visit(node.body)
        self.scopes.pop()

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """A comprehension walrus writes into its enclosing non-comprehension scope."""
        self._binding(node.target.id, node.target, named_expr=True)
        self.visit(node.value)

    def visit_ListComp(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        """Only the first iterable runs outside a comprehension's private scope."""
        self.visit(node.generators[0].iter)
        self.scopes.append(None)
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes.pop()

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def _check_cell_syntax(tree: ast.Module, filename: str) -> None:
    """Compile, never execute, in an async cell scope while keeping source line numbers."""
    wrapper = ast.AsyncFunctionDef(
        name="__completion_guard__",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=tree.body,
        decorator_list=[],
        type_params=[],
    )
    ast.copy_location(wrapper, tree.body[0])
    module = ast.Module(body=[wrapper], type_ignores=[])
    compile(module, filename, "exec")


def completion_binding_errors(
    code: str, *, reserved: frozenset[str], filename: str = "<codeact-cell>"
) -> list[str]:
    """Reject rebinding before helper compilation; propagate compiler-only SyntaxErrors."""
    # Python normalizes identifiers; inspect the original source to retain locations.
    source = code if code.isascii() else unicodedata.normalize("NFKC", code)
    if not any(name in source for name in reserved):
        return []
    try:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # Preserve the normal execution path's syntax-error diagnostics.
        visitor = _CompletionBindings(reserved)
        visitor.visit(tree)
        if visitor.errors:
            _check_cell_syntax(tree, filename)
    except RecursionError:
        return [
            "Cell nests too deeply to check for reserved-name rebinding; "
            "split the deeply nested code into smaller statements, then retry."
        ]
    return visitor.errors
