# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dict-like context manager for agent context blocks.

Provides a simple dict-like API for LLM-generated code to manage
what information appears in the system prompt.

Usage:
    self.context["notes"] = "Here are my notes..."              # literal
    self.context.set_dynamic("status", "self.format_status()")  # expression
    value = self.context["notes"]                                # read
    del self.context["notes"]                                    # remove
    "notes" in self.context                                      # check

Cache lifecycle for expression blocks:
    set_dynamic("key", "expr")  → stores ExpressionContextBlock, invalidates cache
    _prepare_context() runs     → evaluates expr, calls _update_resolved({"key": value})
    self.context["key"]         → returns cached value from _dynamic_cache
"""

from collections.abc import ItemsView, Iterator, KeysView
from typing import Any

from nooa.context_blocks import (
    Context,
    ContextBlock,
    DynamicContext,
    ExpressionContextBlock,
    LiteralContextBlock,
)
from nooa.context_blocks.exceptions import DynamicNotResolvedError, ProtectedBlockError

_SENTINEL = object()


class ContextManager:
    """Dict-like API for managing context blocks.

    Stores one canonical typed record per key. Each record owns both independent
    axes: literal/expression content and cacheable-prefix/volatile-suffix placement.

    Expression values are resolved into _dynamic_cache after each
    _prepare_context() run. Protection, disabled state, and that cache remain
    runtime-only concerns rather than fields on the serializable block record.

    Protected blocks (system_prompt, self, state) are registered via
    set_protected() / set_dynamic_protected() and cannot be overwritten
    by the LLM-facing API (set / set_dynamic / __setitem__ / __delitem__).
    """

    def __init__(self) -> None:
        self._blocks: dict[str, ContextBlock] = {}
        self.protected_keys: set[str] = set()
        self._dynamic_cache: dict[str, Any] = {}
        self.disabled_keys: set[str] = set()

    @staticmethod
    def _make_block(key: str, value: Any, *, prefix: bool) -> ContextBlock:
        """Normalize one supported input value into a canonical block record."""
        if isinstance(value, DynamicContext):
            return ExpressionContextBlock(key=key, expr=value.expr, prefix=prefix)
        return LiteralContextBlock(key=key, value=value, prefix=prefix)

    def _store_block(self, block: ContextBlock, *, protected: bool = False) -> None:
        """Store a canonical block and refresh runtime-only bookkeeping."""
        self._blocks[block.key] = block
        if protected:
            self.protected_keys.add(block.key)
        self.disabled_keys.discard(block.key)
        self._invalidate(block.key)

    def restore_block(self, block: ContextBlock) -> None:
        """Restore one validated block without inferring content or placement."""
        self._store_block(block, protected=block.key in self.protected_keys)

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

        Raises:
            ProtectedBlockError: If key is protected and value is not None/Context.
        """
        if value is None:
            self.disable(key)
            if key in self._blocks and key not in self.protected_keys:
                del self._blocks[key]
            return
        if isinstance(value, Context):
            if value.is_dynamic:
                assert value.expr is not None
                normalized: Any = DynamicContext(value.expr)
            else:
                normalized = value.value
            self._store_block(
                self._make_block(key, normalized, prefix=value.prefix),
                protected=key in self.protected_keys,
            )
            return
        if isinstance(value, DynamicContext):
            self._store_block(
                self._make_block(key, value, prefix=False),
                protected=key in self.protected_keys,
            )
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
        prefix (the same shape framework blocks like ``<self>`` use).

        Args:
            key: Block key (unique identifier).
            value: Plain value to store (mutually exclusive with ``expr``).
            expr: Python expression re-evaluated each turn (keyword-only,
                mutually exclusive with ``value``).

        Raises:
            ProtectedBlockError: If key is protected.
            TypeError: If both ``value`` and ``expr`` are given, neither is
                given, or ``value`` is a DynamicContext (use ``expr=`` instead).
        """
        if key in self.protected_keys:
            raise ProtectedBlockError(key, "modify")

        if expr is not None and _SENTINEL is not value:
            raise TypeError("Cannot specify both value and expr=")

        if expr is not None:
            block = self._make_block(key, DynamicContext(expr), prefix=True)
        elif _SENTINEL is not value:
            if isinstance(value, DynamicContext):
                raise TypeError(
                    f"Use self.context.set_static({key!r}, expr={value.expr!r}) "
                    f"instead of passing a DynamicContext as value"
                )
            block = self._make_block(key, value, prefix=True)
        else:
            raise TypeError("set_static() requires either value or expr=")
        self._store_block(block)

    def set_dynamic(self, key: str, expr: str | None = None, *, value: Any = _SENTINEL) -> None:
        """Set a dynamic context block (placed in the volatile suffix).

        Accepts either an expression string (positional, re-evaluated each turn)
        or a plain value (keyword-only ``value=``).

        Args:
            key: Block key (unique identifier).
            expr: Python expression to evaluate each turn.
            value: Plain value to store in the dynamic partition (keyword-only).

        Raises:
            ProtectedBlockError: If key is protected.
            TypeError: If both expr and value= are provided.
        """
        if key in self.protected_keys:
            raise ProtectedBlockError(key, "modify")

        if expr is not None and _SENTINEL is not value:
            raise TypeError("Cannot specify both expr and value=")

        if expr is not None:
            block = self._make_block(key, DynamicContext(expr), prefix=False)
        elif _SENTINEL is not value:
            block = self._make_block(key, value, prefix=False)
        else:
            raise TypeError("set_dynamic() requires either expr or value=")
        self._store_block(block)

    def is_static(self, key: str) -> bool:
        """Return True if the block is in the static (cacheable) partition."""
        block = self._blocks.get(key)
        return block.prefix if block is not None else False

    def __getitem__(self, key: str) -> Any:
        """Get the value of a context block.

        Literal blocks return their original value directly from the canonical
        record. Expression blocks return the last value in _dynamic_cache.

        Raises:
            KeyError: If key not found.
            DynamicNotResolvedError: If accessing a DynamicContext block before the
                first LLM turn (expression hasn't been evaluated yet).
        """
        if key not in self._blocks:
            raise KeyError(key)

        block = self._blocks[key]

        if isinstance(block, LiteralContextBlock):
            return block.value

        if key not in self._dynamic_cache:
            raise DynamicNotResolvedError(key, block.expr)
        return self._dynamic_cache[key]

    def __delitem__(self, key: str) -> None:
        """Remove a context block.

        Raises:
            KeyError: If key not found.
            ProtectedBlockError: If key is protected.
        """
        if key not in self._blocks:
            raise KeyError(key)
        if key in self.protected_keys:
            raise ProtectedBlockError(key, "remove")
        del self._blocks[key]
        self.disabled_keys.discard(key)
        self._invalidate(key)

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

        Disabled keys are omitted no matter which source would provide them:
        framework defaults (``system_prompt``, ``self``, ``state``), strategy
        blocks (``strategy_prompt``, ``execution_context``), user blocks,
        decorator/scoped context, or skill-registered blocks. Disabling is
        reversible with :meth:`enable` and is allowed for protected blocks.
        """
        self.disabled_keys.update(keys)
        for key in keys:
            self._invalidate(key)

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

    def _raw_items(self) -> ItemsView[str, ContextBlock]:
        """Return raw key-record pairs.

        Internal method for context_builder — not part of the LLM-facing API.
        Use keys() + __getitem__ for resolved access.
        """
        return self._blocks.items()

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
        if key in self.protected_keys:
            raise ProtectedBlockError(key, "remove")

        # Get value before removal
        block = self._blocks[key]
        if isinstance(block, ExpressionContextBlock):
            value = self._dynamic_cache.get(key, DynamicContext(block.expr))
        else:
            value = block.value

        del self._blocks[key]
        self.disabled_keys.discard(key)
        self._invalidate(key)
        return value

    def _invalidate(self, key: str) -> None:
        """Invalidate the cached resolved value for an expression block.

        Called on set, set_dynamic, delete, and pop to ensure
        stale cache entries are cleared.
        """
        self._dynamic_cache.pop(key, None)

    def apply_override(self, key: str, value: "Any | DynamicContext | None") -> None:
        """Apply a single context block override, routing through the correct API.

        Handles both protected and unprotected keys. Used by Agent._apply_context_dict
        to apply class-level and instance-level context overrides at init time.

        Overrides preserve the static/dynamic partition of the block they replace.
        If the key doesn't exist yet, DynamicContext overrides go to dynamic,
        plain values go to dynamic (matching __setitem__ behavior).
        """
        is_protected = key in self.protected_keys
        existing = self._blocks.get(key)
        is_static_block = existing.prefix if existing is not None else False

        if value is None:
            self.disable(key)
            if not is_protected and key in self._blocks:
                # Remove block definition without clearing disabled_keys
                # (pop() would discard from disabled_keys, undoing the disable)
                del self._blocks[key]
        elif isinstance(value, Context):
            # Route through __setitem__ which handles all Context cases
            self[key] = value
        elif isinstance(value, DynamicContext):
            if is_protected:
                if is_static_block:
                    self._store_block(self._make_block(key, value, prefix=True), protected=True)
                else:
                    self._store_block(self._make_block(key, value, prefix=False), protected=True)
            else:
                self.set_dynamic(key, value.expr)
        else:
            if is_protected:
                if is_static_block:
                    self._store_block(self._make_block(key, value, prefix=True), protected=True)
                else:
                    self._store_block(self._make_block(key, value, prefix=False), protected=True)
            else:
                self[key] = value

    def _update_resolved(self, resolved: dict[str, Any]) -> None:
        """Cache resolved expression-block values.

        Called by _prepare_context() after evaluating all expressions. Literal
        block values are read directly from their canonical records.
        """
        self._dynamic_cache.update(resolved)

    # ------------------------------------------------------------------
    # Internal protected-block API (used by Agent.__init__, not by LLMs)
    # ------------------------------------------------------------------

    def set_static_protected(self, key: str, value: Any = None, *, expr: str | None = None) -> None:
        """Register a protected block in the static (cacheable) partition.

        Protected blocks cannot be modified by the LLM-facing API.

        Args:
            key: Block key.
            value: Plain value to store.
            expr: Python expression to evaluate each turn (keyword-only).
        """
        normalized = DynamicContext(expr) if expr is not None else value
        self._store_block(self._make_block(key, normalized, prefix=True), protected=True)

    def set_dynamic_protected(
        self, key: str, expr: str | None = None, *, value: Any = _SENTINEL
    ) -> None:
        """Register a protected block in the dynamic (volatile) partition.

        Protected blocks cannot be modified by the LLM-facing API.
        Expressions are re-evaluated each LLM turn.

        Args:
            key: Block key.
            expr: Python expression to evaluate each turn.
            value: Plain value to store (keyword-only).
        """
        if expr is not None:
            block = self._make_block(key, DynamicContext(expr), prefix=False)
        elif _SENTINEL is not value:
            block = self._make_block(key, value, prefix=False)
        else:
            raise TypeError("set_dynamic_protected() requires either expr or value=")
        self._store_block(block, protected=True)

    def remove_protected(self, key: str) -> None:
        """Remove a protected block (used by _apply_context_dict with value=None).

        Args:
            key: Block key to remove.

        Raises:
            KeyError: If key not found.
        """
        if key not in self._blocks:
            raise KeyError(key)
        del self._blocks[key]
        self.protected_keys.discard(key)
        self.disabled_keys.discard(key)
        self._invalidate(key)
