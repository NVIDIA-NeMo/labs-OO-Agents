# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dict-like context manager for agent context blocks.

Provides a simple dict-like API for LLM-generated code to manage
what information appears in the model's context.

Usage:
    self.context["notes"] = "Here are my notes..."              # fixed, volatile suffix
    self.context["rules"] = Context("...", prefix=True)          # fixed, cacheable prefix
    self.context["status"] = Context(expr="self.format_status()")  # expression
    value = self.context["notes"]                                # read
    del self.context["notes"]                                    # remove
    "notes" in self.context                                      # check

Reads return what was stored: a literal value, or ``Context(expr=...)`` for an
expression block. Context views evaluate expressions; the manager never does.
"""

from collections.abc import Iterator, KeysView
from typing import Any

from nooa.context_blocks import Context, DynamicContext

_SENTINEL = object()


class ContextManager:
    """Dict-like API for managing context blocks.

    Stores context blocks as key -> value mappings. Values are either static
    values (any type) or DynamicContext markers (for expressions re-evaluated each turn).

    Declarations live in _blocks. Reading an expression block returns its
    declaration as ``Context(expr=...)``, not a value.

    Every key is an ordinary user block. Blocks a context view builds itself,
    such as the system prompt, are not stored here and cannot be changed here;
    a stored block with the same name is just another block.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, Any | DynamicContext] = {}
        self._static: dict[str, bool] = {}
        self.disabled_keys: set[str] = set()

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a context block via dict syntax.

        Accepts the same value types as declarative ``context={}`` dicts:
        - ``str``: Literal text, placed in volatile suffix.
        - ``Context(value=...|expr=..., prefix=bool)``: Full control over content and placement.
        - ``DynamicContext("expr")``: Expression block (deprecated, use Context(expr=...)).
        - ``None``: Suppress the block from prompt rendering.

        Args:
            key: Block key (unique identifier).
            value: Block value (str, Context, DynamicContext, None, or any pformat-able object).
        """
        if value is None:
            self.disable(key)
            self._blocks.pop(key, None)
            self._static.pop(key, None)
            return
        if isinstance(value, Context):
            if value.is_dynamic:
                if value.prefix:
                    self.set_static(key, expr=value.expr)
                else:
                    self.set_dynamic(key, value.expr)
            elif value.prefix:
                self.set_static(key, value.value)
            else:
                self.set_dynamic(key, value=value.value)
            return
        if isinstance(value, DynamicContext):
            self.set_dynamic(key, value.expr)
            return
        self.set_dynamic(key, value=value)

    def set(
        self, key: str, value: str | None = None, *, expr: str | None = None, prefix: bool = False
    ) -> None:
        """Set a context block — convenience method with keyword arguments.

        Equivalent to ``context_manager[key] = Context(value, expr=expr, prefix=prefix)``
        but friendlier for callers who prefer explicit keyword args.

        Args:
            key: Block key.
            value: Literal text content (mutually exclusive with expr).
            expr: Python expression re-evaluated each LLM turn.
            prefix: Place in cacheable prefix if True, volatile suffix if False.
        """
        if value is not None and expr is not None:
            raise TypeError("set() takes value or expr, not both")
        if value is None and expr is None:
            self[key] = None
            return
        if expr is not None:
            self[key] = Context(expr=expr, prefix=prefix)
        else:
            if prefix:
                self[key] = Context(value, prefix=True)
            else:
                self[key] = value

    def set_static(self, key: str, value: Any = _SENTINEL, *, expr: str | None = None) -> None:
        """Set a static context block (placed in the cacheable prefix).

        Static blocks live in the stable, cacheable prefix of the prompt. The
        ``static`` partition and the value *kind* are independent axes: a static
        block can hold a plain value OR a re-evaluated expression. Pass
        ``value`` for a plain, unchanging value. Pass ``expr`` to register a
        block that is re-evaluated every turn yet still rendered in the cacheable
        prefix.

        Args:
            key: Block key (unique identifier).
            value: Plain value to store (mutually exclusive with ``expr``).
            expr: Python expression re-evaluated each turn (keyword-only,
                mutually exclusive with ``value``).

        Raises:
            TypeError: If both ``value`` and ``expr`` are given, neither is
                given, or ``value`` is a DynamicContext (use ``expr=`` instead).
        """
        if expr is not None and _SENTINEL is not value:
            raise TypeError("Cannot specify both value and expr=")

        if expr is not None:
            self._blocks[key] = DynamicContext(expr)
        elif _SENTINEL is not value:
            if isinstance(value, DynamicContext):
                raise TypeError(
                    f"Use self.context.set_static({key!r}, expr={value.expr!r}) "
                    f"instead of passing a DynamicContext as value"
                )
            self._blocks[key] = value
        else:
            raise TypeError("set_static() requires either value or expr=")

        self._static[key] = True
        self.disabled_keys.discard(key)

    def set_dynamic(self, key: str, expr: str | None = None, *, value: Any = _SENTINEL) -> None:
        """Set a dynamic context block (placed in the volatile suffix).

        Accepts either an expression string (positional, re-evaluated each turn)
        or a plain value (keyword-only ``value=``).

        Args:
            key: Block key (unique identifier).
            expr: Python expression to evaluate each turn.
            value: Plain value to store in the dynamic partition (keyword-only).

        Raises:
            TypeError: If both expr and value= are provided.
        """
        if expr is not None and _SENTINEL is not value:
            raise TypeError("Cannot specify both expr and value=")

        if expr is not None:
            self._blocks[key] = DynamicContext(expr)
        elif _SENTINEL is not value:
            self._blocks[key] = value
        else:
            raise TypeError("set_dynamic() requires either expr or value=")
        self._static[key] = False
        self.disabled_keys.discard(key)

    def is_static(self, key: str) -> bool:
        """Return True if the block is in the static (cacheable) partition."""
        return self._static.get(key, False)

    def __getitem__(self, key: str) -> Any:
        """Return a block's declaration: its literal value, or ``Context(expr=...)``.

        Expressions are not evaluated; call the expression's code for its current value.

        Raises:
            KeyError: If key not found.
        """
        return self._declared(key, self._blocks[key])

    def _declared(self, key: str, value: Any) -> Any:
        if isinstance(value, DynamicContext):
            return Context(expr=value.expr, prefix=self.is_static(key))
        return value

    def __delitem__(self, key: str) -> None:
        """Remove a context block.

        Raises:
            KeyError: If key not found.
        """
        if key not in self._blocks:
            raise KeyError(key)
        del self._blocks[key]
        self._static.pop(key, None)
        self.disabled_keys.discard(key)

    def __contains__(self, key: object) -> bool:
        """Check if a context block exists."""
        return key in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def keys(self) -> KeysView[str]:
        """Return block keys."""
        return self._blocks.keys()

    def disable(self, *keys: str) -> None:
        """Suppress named blocks from prompt construction without deleting them.

        Disabled keys are omitted from user blocks, decorator/scoped context,
        and skill blocks. Blocks a view builds itself are not affected; views
        document their own options. Disabling is reversible with :meth:`enable`.
        """
        self.disabled_keys.update(keys)

    def enable(self, *keys: str) -> None:
        """Re-enable named blocks previously suppressed with :meth:`disable`."""
        for key in keys:
            self.disabled_keys.discard(key)

    def is_enabled(self, key: str) -> bool:
        """Return True if a block key is eligible for rendering."""
        return key not in self.disabled_keys

    def is_disabled(self, key: str) -> bool:
        """Return True if a block key is currently suppressed."""
        return key in self.disabled_keys

    def disabled(self) -> "set[str]":
        """Return a copy of currently suppressed block keys."""
        return set(self.disabled_keys)

    def declarations(self) -> tuple[tuple[str, Any], ...]:
        """Return an immutable snapshot of raw block declarations.

        Dynamic declarations remain :class:`DynamicContext` objects. Context
        views use this method to materialize manager state without depending on
        the manager's private storage.
        """
        return tuple(self._blocks.items())

    def get(self, key: str, default: Any = None) -> Any:
        """Get a block value, returning default if not found.

        Like dict.get() — returns default instead of raising KeyError.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, *args: Any) -> Any:
        """Remove and return a block value.

        Like dict.pop() — returns default if provided, raises KeyError otherwise.
        """
        if key not in self._blocks:
            if args:
                return args[0]
            raise KeyError(key)

        value = self._declared(key, self._blocks[key])
        del self._blocks[key]
        self._static.pop(key, None)
        self.disabled_keys.discard(key)
        return value
