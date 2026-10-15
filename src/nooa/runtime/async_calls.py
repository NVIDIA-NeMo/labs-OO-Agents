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
- Held coroutines (:func:`held_coroutine_uses`, flow-sensitive per scope):
  after ``r = self.repo.symbols(...)`` a later ``r.attr``, ``r[i]``,
  ``len(r)`` or iteration of ``r`` before ``r`` is awaited or rebound is
  flagged. ``await r``, gather/create_task/wait, collecting ``r`` or passing it
  to any call, and the coroutine API (``r.close()``, ``r.send``, ``r.cr_*``)
  pass. Branches and loops are handled conservatively (doubt means no flag).

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


# ---------------------------------------------------------------------------
# Flow-sensitive rule: a name holding an un-awaited coroutine, used as a result
# ---------------------------------------------------------------------------

# Attributes that really exist on a coroutine object (dunders are exempt too).
COROUTINE_API = frozenset(
    {
        "close",
        "send",
        "throw",
        "cr_await",
        "cr_code",
        "cr_frame",
        "cr_origin",
        "cr_running",
        "cr_suspended",
    }
)
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)


def _coroutine_call_chain(value: ast.expr | None, agent: Any) -> str | None:
    """``"self.repo.symbols"`` when *value* is an un-awaited attribute-chain call."""
    if isinstance(value, ast.Call):
        return async_self_chain(value.func, agent)
    return None


def _iter_scope_nodes(node: ast.AST):
    """Yield *node*'s descendants without entering nested function/class/lambda scopes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, _SCOPE_NODES):
            yield from _iter_scope_nodes(child)


def _touched_names(node: ast.AST) -> set[str]:
    """Every name read, bound, deleted or declared global/nonlocal in *node*
    (its own scope; a nested ``def``/``class`` contributes only its own name)."""
    names: set[str] = set()
    for sub in [node, *_iter_scope_nodes(node)]:
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, (ast.Global, ast.Nonlocal)):
            names.update(sub.names)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in sub.names)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
        elif isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
            names.add(sub.name)
        elif isinstance(sub, ast.MatchMapping) and sub.rest:
            names.add(sub.rest)
    return names


def _bound_names(node: ast.AST) -> set[str]:
    """Names bound, deleted or declared global/nonlocal by *node* (own scope)."""
    names: set[str] = set()
    for sub in [node, *_iter_scope_nodes(node)]:
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, (ast.Global, ast.Nonlocal)):
            names.update(sub.names)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in sub.names)
    return names


def _awaited_names(node: ast.AST) -> set[str]:
    return {
        sub.value.id
        for sub in [node, *_iter_scope_nodes(node)]
        if isinstance(sub, ast.Await) and isinstance(sub.value, ast.Name)
    }


def _result_uses(expr: ast.AST, held: dict[str, str]) -> list[tuple[ast.Name, str]]:
    """Uses of a held name in *expr* as if it were the call's result:
    ``name.attr`` (not the coroutine API), ``name[...]``, ``len(name)``,
    iteration by a comprehension, or ``a, b = name`` (checked by the caller).
    Lambdas and generator bodies (lazy) are not entered; a comprehension's own
    target names shadow held names inside it."""
    found: list[tuple[ast.Name, str]] = []

    def is_held(node: ast.AST, live: dict[str, str]) -> bool:
        return isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in live

    def visit(node: ast.AST, live: dict[str, str]) -> None:
        if not live or isinstance(node, _SCOPE_NODES):
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            first = node.generators[0]
            if is_held(first.iter, live):
                found.append((first.iter, live[first.iter.id]))
            else:
                visit(first.iter, live)
            if isinstance(node, ast.GeneratorExp):
                return  # the rest runs lazily, maybe after an await
            inner = {k: v for k, v in live.items() if k not in _bound_names(node)}
            parts: list[ast.AST] = []
            for i, gen in enumerate(node.generators):
                if i:
                    parts.append(gen.iter)
                parts.extend(gen.ifs)
            parts.extend([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
            for part in parts:
                if isinstance(part, ast.expr):
                    visit(part, inner)
            for gen in node.generators[1:]:
                if is_held(gen.iter, inner):
                    found.append((gen.iter, inner[gen.iter.id]))
            return
        if isinstance(node, ast.Attribute) and is_held(node.value, live):
            attr = node.attr
            if attr not in COROUTINE_API and not (attr.startswith("__") and attr.endswith("__")):
                found.append((node.value, live[node.value.id]))
        elif isinstance(node, ast.Subscript) and is_held(node.value, live):
            found.append((node.value, live[node.value.id]))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and is_held(node.args[0], live)
        ):
            found.append((node.args[0], live[node.args[0].id]))
        for child in ast.iter_child_nodes(node):
            visit(child, live)

    visit(expr, held)
    return found


def _check_simple(stmt: ast.stmt, held: dict[str, str]) -> list[tuple[ast.Name, str]]:
    # A name awaited or walrus-bound anywhere in the statement is skipped:
    # evaluation order inside one statement is not worth modelling.
    skip = _awaited_names(stmt) | {
        sub.target.id for sub in [stmt, *_iter_scope_nodes(stmt)] if isinstance(sub, ast.NamedExpr)
    }
    live = {k: v for k, v in held.items() if k not in skip}
    found = _result_uses(stmt, live)
    if (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id in live
        and any(isinstance(t, (ast.Tuple, ast.List)) for t in stmt.targets)
    ):
        found.append((stmt.value, live[stmt.value.id]))
    return found


def _new_bindings(stmt: ast.stmt, agent: Any) -> dict[str, str]:
    """``name = self.x.y(...)`` (also ``a = b = ...``, annotated, and tuple
    targets matched element-wise against a tuple of calls)."""
    out: dict[str, str] = {}
    if isinstance(stmt, ast.Assign):
        targets, value = stmt.targets, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets, value = [stmt.target], stmt.value
    else:
        return out
    for target in targets:
        if isinstance(target, ast.Name):
            chain = _coroutine_call_chain(value, agent)
            if chain:
                out[target.id] = chain
        elif (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and not any(isinstance(e, ast.Starred) for e in (*target.elts, *value.elts))
        ):
            for t, v in zip(target.elts, value.elts, strict=True):
                chain = _coroutine_call_chain(v, agent)
                if isinstance(t, ast.Name) and chain:
                    out[t.id] = chain
    return out


def _scan_block(body: list[ast.stmt], held: dict[str, str], agent: Any, found: list) -> None:
    held = dict(held)
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for name in _touched_names(stmt):  # decorators/defaults/the bound name
                held.pop(name, None)
            continue
        compound_parts = _compound_parts(stmt)
        if compound_parts is None:
            found.extend(_check_simple(stmt, held))
            for name in _bound_names(stmt) | _awaited_names(stmt):
                held.pop(name, None)
            held.update(_new_bindings(stmt, agent))
            continue
        header, blocks = compound_parts
        # Header expressions run first, in the state before the statement.
        for expr in header:
            live = {
                k: v for k, v in held.items() if k not in _awaited_names(expr) | _bound_names(expr)
            }
            found.extend(_result_uses(expr, live))
        if isinstance(stmt, (ast.For, ast.AsyncFor)) and isinstance(stmt.iter, ast.Name):
            if stmt.iter.id in held and stmt.iter.id not in _awaited_names(stmt.iter):
                found.append((stmt.iter, held[stmt.iter.id]))
        # Bodies: straight-line inside, starting from what is certainly still
        # held. Loops drop every name the loop touches (a later iteration may
        # see a rebinding); except/else/finally/match arms start empty.
        if isinstance(stmt, _LOOP_NODES):
            entry = {k: v for k, v in held.items() if k not in _touched_names(stmt)}
        else:
            header_touched = (
                set().union(*(_bound_names(e) | _awaited_names(e) for e in header))
                if header
                else set()
            )
            entry = {k: v for k, v in held.items() if k not in header_touched}
        for block, precise in blocks:
            _scan_block(block, entry if precise else {}, agent, found)
        # After the statement keep only names it never mentions.
        for name in _touched_names(stmt):
            held.pop(name, None)


def _compound_parts(stmt: ast.stmt):
    """``(header expressions, [(block, starts_from_entry_state)])`` for compound
    statements; None for simple statements."""
    if isinstance(stmt, ast.If):
        return [stmt.test], [(stmt.body, True), (stmt.orelse, True)]
    if isinstance(stmt, ast.While):
        return [stmt.test], [(stmt.body, True), (stmt.orelse, False)]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter], [(stmt.body, True), (stmt.orelse, False)]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in stmt.items], [(stmt.body, True)]
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        blocks = [(stmt.body, True)] + [(h.body, False) for h in stmt.handlers]
        return [], blocks + [(stmt.orelse, False), (stmt.finalbody, False)]
    if isinstance(stmt, ast.Match):
        return [stmt.subject], [(case.body, False) for case in stmt.cases]
    return None


def held_coroutine_uses(tree: ast.AST, agent: Any) -> list[tuple[ast.Name, str]]:
    """Uses of a name that holds an un-awaited coroutine as if it were the result.

    Flow-sensitive within each function scope (and the module body): a name is
    *held* after ``name = self.x.y(...)`` (an attribute-chain coroutine call;
    ``self.method()`` is already flagged at the call). It stops being held when
    it is awaited, rebound or deleted. A later ``name.attr`` (except the
    coroutine API: ``close``/``send``/``throw``/``cr_*``/dunders), ``name[...]``,
    ``len(name)``, ``for _ in name``, comprehension iteration or ``a, b = name``
    while held is reported as ``(name_node, "self.x.y")``. Passing the name to
    any call, collecting it, gathering or awaiting it is fine. Branches start
    from the state before them, loops drop every name they mention, handler and
    match arms start empty, and after any compound statement every name it
    mentions is forgotten, so doubtful cases are not flagged.
    """
    if agent is None:
        return []
    found: list[tuple[ast.Name, str]] = []
    scopes: list[list[ast.stmt]] = []
    if isinstance(tree, ast.Module):
        scopes.append(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(node.body)
    for body in scopes:
        _scan_block(body, {}, agent, found)
    seen: set[int] = set()
    unique = []
    for node, chain in sorted(found, key=lambda f: (f[0].lineno, f[0].col_offset)):
        if id(node) not in seen:
            seen.add(id(node))
            unique.append((node, chain))
    return unique


def held_coroutine_message(name: str, chain: str) -> str:
    """First line of the error for a held, un-awaited coroutine."""
    return f"`{name}` holds the un-awaited `{chain}(...)`; write `{name} = await {chain}(...)`."


def held_coroutine_example(name: str, chain: str) -> str:
    return f"{name} = await {chain}(...)"
