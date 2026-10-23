# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard accidental rebinding of CodeAct's completion helper, not adversarial code."""

import ast
from collections.abc import Iterable


def _declares_completion_global(nodes: Iterable[ast.AST]) -> bool:
    """Find global declarations in this scope, excluding nested Python scopes."""
    for node in nodes:
        if isinstance(node, ast.Global) and "return_result" in node.names:
            return True
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if _declares_completion_global(ast.iter_child_nodes(node)):
                return True
    return False


class _CompletionBindings(ast.NodeVisitor):
    """Find writes to the cell/global binding while preserving independent locals."""

    def __init__(self) -> None:
        """Start in the cell scope; None denotes a comprehension's private scope."""
        self.scopes: list[bool | None] = [True]
        self.errors: list[str] = []

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
        if name == "return_result" and scope:
            self.errors.append(
                f"Line {node.lineno}: `return_result` is reserved for submitting the final "
                "answer. Do not rebind or delete it; use a variable such as `result`, "
                "then call `return_result(result)`."
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
        self.scopes.append(_declares_completion_global(body))
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
        self.scopes.append(False)
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


def completion_binding_errors(code: str) -> list[str]:
    """Reject CodeAct helper rebinding before imports or helper compilation can run."""
    try:
        tree = ast.parse(code)
        visitor = _CompletionBindings()
        visitor.visit(tree)
    except SyntaxError:
        return []  # Preserve the normal execution path's syntax-error diagnostics.
    except RecursionError:
        return [
            "Cell nests too deeply to check for `return_result` rebinding; "
            "split the deeply nested code into smaller statements, then retry."
        ]
    return visitor.errors
