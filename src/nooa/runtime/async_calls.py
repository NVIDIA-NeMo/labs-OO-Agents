# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static missing-``await`` rules shared by the code validators and the error formatter.

One place decides (a) how an attribute chain such as ``self.shell.read`` is
resolved to a live callable without running descriptors, and (b) which
un-awaited calls count as a mistake. Used by ``GeneratedCodeValidator``
(nooa.strategies.generated_code), the E301 check in
``nooa.runtime.code_validator`` and ``nooa.errors.formatting``.

Rules:

- ``self.method()`` (an async method directly on the agent) is flagged unless it
  is awaited, is inside a list/generator comprehension, or is passed straight to
  ``asyncio.gather``/``create_task``/``wait_for``/``ensure_future``/``shield``.
- Attribute chains (``self.shell.read()``) are flagged only on clear misuse: the
  call's value is discarded (a bare expression statement), or the call result is
  used immediately as if awaited (attribute access, subscript, or iteration on
  it). Every other use -- assignment, ``append(...)``, literals, comprehensions,
  call arguments -- passes, because a false positive blocks a valid cell.

This module has no import-time dependencies on the rest of nooa.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

GATHER_FUNCTIONS = frozenset({"gather", "create_task", "wait_for", "ensure_future", "shield"})


def resolve_static_chain(node: ast.expr, namespace: dict[str, Any]) -> object | None:
    """Resolve ``a.b.c`` against live objects without invoking descriptors.

    Each step uses ``inspect.getattr_static``; a missing attribute or a
    ``property`` along the way ends resolution (returns None).
    """
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in namespace:
        return None
    value: Any = namespace[node.id]
    for name in reversed(names):
        try:
            value = inspect.getattr_static(value, name)
        except AttributeError:
            return None
        if isinstance(value, property):
            return None
    return value


def is_coroutine_function(value: object) -> bool:
    """``inspect.iscoroutinefunction`` that sees through static/classmethod and wrappers."""
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    try:
        if inspect.iscoroutinefunction(value):
            return True
        return inspect.iscoroutinefunction(inspect.unwrap(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _attribute_chain_length(func: ast.expr) -> int:
    length = 0
    while isinstance(func, ast.Attribute):
        length += 1
        func = func.value
    return length if isinstance(func, ast.Name) and func.id == "self" else 0


def async_self_chain(func: ast.expr, agent: Any) -> str | None:
    """Return ``"self.shell.read"`` when *func* is an attribute chain (two or more
    attributes) on the agent whose final attribute is a visible coroutine
    function; otherwise None. ``self.method`` itself is covered by
    :func:`collect_async_method_names`."""
    if agent is None or _attribute_chain_length(func) < 2:
        return None
    value = resolve_static_chain(func, {"self": agent})
    if value is None or not is_coroutine_function(value):
        return None
    try:
        from nooa.agentdoc.visibility import is_hidden_method

        if is_hidden_method(value):
            return None
    except Exception:
        pass
    return ast.unparse(func)


def collect_async_method_names(agent: Any) -> set[str]:
    """Names of the agent's visible async methods (class- and instance-level)."""
    from nooa.agentdoc.visibility import is_hidden_method

    names: set[str] = set()
    if not agent:
        return names
    for owner, getter in (
        (agent.__class__, lambda name: getattr(agent.__class__, name, None)),
        (agent, lambda name: getattr(agent, name)),
    ):
        for attr_name in dir(owner):
            if attr_name.startswith("__"):
                continue
            try:
                attr = getter(attr_name)
                if attr and callable(attr) and inspect.iscoroutinefunction(attr):
                    if not is_hidden_method(attr):
                        names.add(attr_name)
            except Exception:
                continue
    return names


def build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def is_gather_argument(node: ast.AST, parent: ast.AST | None) -> bool:
    """True when *node* is passed directly to asyncio.gather/create_task/wait_for/
    ensure_future/shield (or a bare ``gather(...)`` etc.), which await it."""
    if not isinstance(parent, ast.Call) or node not in parent.args:
        return False
    func = parent.func
    if isinstance(func, ast.Attribute):
        return func.attr in GATHER_FUNCTIONS
    if isinstance(func, ast.Name):
        return func.id in GATHER_FUNCTIONS
    return False


def is_awaited_or_gathered(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
    """Exemptions for a direct ``self.method()`` call (the pre-#447 rule plus
    gather/create_task arguments)."""
    current = node
    while current in parent_map:
        parent = parent_map[current]
        if isinstance(parent, ast.Await) and parent.value is current:
            return True
        if isinstance(parent, (ast.ListComp, ast.GeneratorExp)):
            return True
        if is_gather_argument(current, parent):
            return True
        current = parent
    return False


def is_clear_misuse(node: ast.Call, parent_map: dict[ast.AST, ast.AST]) -> bool:
    """True when an un-awaited call is certainly a mistake: its value is discarded
    (bare expression statement) or used as if awaited (``.attr``, ``[i]``, or
    iterated by ``for`` / a comprehension)."""
    parent = parent_map.get(node)
    if isinstance(parent, ast.Expr):
        return parent.value is node
    if isinstance(parent, (ast.Attribute, ast.Subscript)):
        return parent.value is node
    if isinstance(parent, (ast.For, ast.comprehension)):
        return parent.iter is node
    return False


def missing_await_calls(
    tree: ast.AST, agent: Any, async_method_names: set[str] | None = None
) -> list[tuple[ast.Call, str, bool]]:
    """Every call in *tree* that should have been awaited.

    Returns ``(call, name, is_chain)`` where *name* is ``"self.shell.read"`` for
    attribute chains and the bare method name for ``self.method()``.
    """
    if agent is None:
        return []
    if async_method_names is None:
        async_method_names = collect_async_method_names(agent)
    parent_map = build_parent_map(tree)
    found: list[tuple[ast.Call, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = async_self_chain(node.func, agent)
        if chain is not None:
            if is_clear_misuse(node, parent_map):
                found.append((node, chain, True))
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in async_method_names
            and not is_awaited_or_gathered(node, parent_map)
        ):
            found.append((node, node.func.attr, False))
    return found


def missing_await_message(name: str, *, chain: bool) -> str:
    """First line of the missing-await error for *name*."""
    if chain:
        return f"`{name}` is async and must be called with `await`."
    return f"Method `{name}` is async and must be called with `await`"


def missing_await_example(name: str, *, chain: bool) -> str:
    """The fixed call, e.g. ``await self.shell.read(...)``."""
    return f"await {name}(...)" if chain else f"await self.{name}(...)"
