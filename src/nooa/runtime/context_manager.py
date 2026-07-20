# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dict-like context manager for agent context blocks.

Provides a simple dict-like API for LLM-generated code to manage
what information appears in the system prompt.

Usage:
    self.context["notes"] = "Here are my notes..."              # static
    self.context.set_dynamic("status", "self.format_status()")  # dynamic
    value = self.context["notes"]                                # read
    del self.context["notes"]                                    # remove
    "notes" in self.context                                      # check

Cache lifecycle for DynamicContext blocks:
    set_dynamic("key", "expr")  → stores DynamicContext in _blocks, invalidates cache
    _prepare_context() runs     → evaluates expr, calls _update_resolved({"key": value})
    self.context["key"]         → returns cached value from _dynamic_cache
"""

from collections.abc import ItemsView, Iterator, KeysView
from typing import Any

from nooa.context_blocks import Context, DynamicContext
from nooa.context_blocks.exceptions import DynamicNotResolvedError, ProtectedBlockError

_SENTINEL = object()


class ContextManager:
    """Dict-like API for managing context blocks.

    Stores context blocks as key -> value mappings. Values are either static
    values (any type) or DynamicContext markers (for expressions re-evaluated each turn).

    Single source of truth:
    - Static blocks: value lives in _blocks only. __getitem__ reads from _blocks.
    - DynamicContext blocks: DynamicContext marker in _blocks, resolved value in _dynamic_cache.
      Cache is populated by _update_resolved() after each _prepare_context() run,
      and invalidated on set_dynamic() or __setitem__().

    Protected blocks (system_prompt, self, state) are registered via
    set_protected() / set_dynamic_protected() and cannot be overwritten
    by the LLM-facing API (set / set_dynamic / __setitem__ / __delitem__).
    """

    def __init__(self) -> None:
        self._blocks: dict[str, Any | DynamicContext] = {}
        self.protected_keys: set[str] = set()
        self._dynamic_cache: dict[str, Any] = {}
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

        Raises:
            ProtectedBlockError: If key is protected and value is not None/Context.
        """
        if value is None:
            self.disable(key)
            if key in self._blocks and key not in self.protected_keys:
                del self._blocks[key]
                self._static.pop(key, None)
            return
        if isinstance(value, Context):
            if value.is_dynamic:
                if value.prefix:
                    if key in self.protected_keys:
                        self.set_static_protected(key, expr=value.expr)
                    else:
                        self.set_static(key, expr=value.expr)
                else:
                    if key in self.protected_keys:
                        self.set_dynamic_protected(key, value.expr)
                    else:
                        self.set_dynamic(key, value.expr)
            else:
                if value.prefix:
                    if key in self.protected_keys:
                        self.set_static_protected(key, value.value)
                    else:
                        self.set_static(key, value.value)
                else:
                    if key in self.protected_keys:
                        self.set_dynamic_protected(key, value=value.value)
                    else:
                        self.set_dynamic(key, value=value.value)
            return
        if isinstance(value, DynamicContext):
            if key in self.protected_keys:
                self.set_dynamic_protected(key, value.expr)
            else:
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
        self._invalidate(key)

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
            self._blocks[key] = DynamicContext(expr)
        elif _SENTINEL is not value:
            self._blocks[key] = value
        else:
            raise TypeError("set_dynamic() requires either expr or value=")
        self._static[key] = False
        self.disabled_keys.discard(key)
        self._invalidate(key)

    def is_static(self, key: str) -> bool:
        """Return True if the block is in the static (cacheable) partition."""
        return self._static.get(key, False)

    def __getitem__(self, key: str) -> Any:
        """Get the value of a context block.

        Static blocks: Returns the original value directly from _blocks.
        DynamicContext blocks: Returns the last resolved value from _dynamic_cache.

        Raises:
            KeyError: If key not found.
            DynamicNotResolvedError: If accessing a DynamicContext block before the
                first LLM turn (expression hasn't been evaluated yet).
        """
        if key not in self._blocks:
            raise KeyError(key)

        value = self._blocks[key]

        # Static block: return directly (single source of truth)
        if not isinstance(value, DynamicContext):
            return value

        # DynamicContext block: return from cache, or raise if not yet resolved
        if key not in self._dynamic_cache:
            raise DynamicNotResolvedError(key, value.expr)
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
        self._static.pop(key, None)
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

    def _raw_items(self) -> ItemsView[str, Any]:
        """Return raw key-value pairs including DynamicContext markers.

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
        raw = self._blocks[key]
        if isinstance(raw, DynamicContext):
            value = self._dynamic_cache.get(key, raw)
        else:
            value = raw

        del self._blocks[key]
        self._static.pop(key, None)
        self.disabled_keys.discard(key)
        self._invalidate(key)
        return value

    def _invalidate(self, key: str) -> None:
        """Invalidate cached resolved value for a dynamic block.

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
        is_static_block = self._static.get(key, False)

        if value is None:
            self.disable(key)
            if not is_protected and key in self._blocks:
                # Remove block definition without clearing disabled_keys
                # (pop() would discard from disabled_keys, undoing the disable)
                del self._blocks[key]
                self._static.pop(key, None)
        elif isinstance(value, Context):
            # Route through __setitem__ which handles all Context cases
            self[key] = value
        elif isinstance(value, DynamicContext):
            if is_protected:
                if is_static_block:
                    self.set_static_protected(key, expr=value.expr)
                else:
                    self.set_dynamic_protected(key, value.expr)
            else:
                self.set_dynamic(key, value.expr)
        else:
            if is_protected:
                if is_static_block:
                    self.set_static_protected(key, value)
                else:
                    self.set_dynamic_protected(key, value=value)
            else:
                self[key] = value

    def _update_resolved(self, resolved: dict[str, Any]) -> None:
        """Cache resolved DynamicContext block values.

        Called by _prepare_context() after evaluating all DynamicContext expressions.
        Only DynamicContext block values should be cached — static blocks are read
        directly from _blocks.
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
        if expr is not None:
            self._blocks[key] = DynamicContext(expr)
        elif isinstance(value, DynamicContext):
            self._blocks[key] = value
        else:
            self._blocks[key] = value
        self.protected_keys.add(key)
        self._static[key] = True
        self.disabled_keys.discard(key)
        self._invalidate(key)

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
            self._blocks[key] = DynamicContext(expr)
        elif _SENTINEL is not value:
            self._blocks[key] = value
        else:
            raise TypeError("set_dynamic_protected() requires either expr or value=")
        self.protected_keys.add(key)
        self._static[key] = False
        self.disabled_keys.discard(key)
        self._invalidate(key)

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
        self._static.pop(key, None)
        self.disabled_keys.discard(key)
        self._invalidate(key)
