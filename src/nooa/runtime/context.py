# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ContextApi — LLM-facing Skill wrapper for agent context blocks.

ContextApi is always present on every Agent as self.context, but hidden from the LLM
by default. Subclasses opt in by calling spec(self, "context", hidden=False) in __init__.

ContextApi wraps agent.context_manager (ContextManager) — the state always lives there.
"""

from collections.abc import Iterator, Set
from typing import TYPE_CHECKING, Any

from nooa.runtime.context_manager import ContextManager
from nooa.skill import Skill

_MISSING = object()

if TYPE_CHECKING:
    from nooa.agent import Agent

__all__ = ["ContextApi", "ContextManager"]


class ContextApi(Skill):
    """Dict-like API for managing what appears in your system prompt.

    Use self.context to add, update, or remove named blocks in your prompt.
    Static blocks are set once; dynamic blocks are re-evaluated each LLM turn.
    Framework blocks (system_prompt, self) are protected and cannot be changed.

    Static blocks (set once):
        self.context["notes"] = "Remember: user prefers JSON output"
        self.context["status"] = f"Processed {n} items"

    Dynamic blocks (re-evaluated each turn):
        self.context.set_dynamic("status", "f'Items: {len(self.results)}'")

    Read / check / remove:
        value = self.context["notes"]          # raises KeyError if missing
        value = self.context.get("notes")      # returns None if missing
        "notes" in self.context                # True/False
        del self.context["notes"]              # remove
        self.context.pop("notes", None)        # remove safely

    Examples:
        # Persist notes across turns
        self.context["plan"] = "Step 1: collect, Step 2: summarise"

        # Dynamic status updated every turn
        self.context.set_dynamic("progress", "f'{self.done}/{self.total} done'")

        # Clean up when done
        del self.context["plan"]

    Load this library:
        doc(self.context)
    """

    def __init__(self, agent: "Agent"):
        self._context: ContextManager = agent.context_manager

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a context block — same value types as declarative ``context={}`` dicts.

        Accepts:
        - ``str``: Literal text, placed in volatile suffix.
        - ``Context(value=...|expr=..., prefix=bool)``: Full control over content and placement.
        - ``DynamicContext("expr")``: Expression block (deprecated, use Context(expr=...)).
        - ``None``: Suppress the block from prompt rendering.
        """
        self._context[key] = value

    def set_dynamic(self, key: str, expr: str | None = None, *, value: Any = _MISSING) -> None:
        """Set a dynamic context block (placed in the volatile suffix).

        .. deprecated::
            Use ``self.context["key"] = Context(expr="...")`` or
            ``self.context["key"] = "value"`` instead.

        Args:
            key: Block key.
            expr: Python expression to evaluate each turn.
            value: Plain value to store in the dynamic partition (keyword-only).
        """
        import warnings

        warnings.warn(
            "set_dynamic() is deprecated. Use self.context[key] = Context(expr=...) "
            "or self.context[key] = 'value' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if value is not _MISSING:
            self._context.set_dynamic(key, value=value)
        else:
            self._context.set_dynamic(key, expr)

    def set_static(self, key: str, value: Any = _MISSING, *, expr: str | None = None) -> None:
        """Set a static context block (placed in the cacheable prefix).

        .. deprecated::
            Use ``self.context["key"] = Context("value", prefix=True)`` or
            ``self.context["key"] = Context(expr="...", prefix=True)`` instead.

        Args:
            key: Block key.
            value: Plain value to store in the static partition.
            expr: Python expression to evaluate each turn (keyword-only,
                mutually exclusive with ``value``).
        """
        import warnings

        warnings.warn(
            "set_static() is deprecated. Use self.context[key] = Context(value, prefix=True) "
            "or Context(expr=..., prefix=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if expr is not None and value is not _MISSING:
            raise TypeError("Cannot specify both value and expr=")
        if expr is not None:
            self._context.set_static(key, expr=expr)
        elif value is not _MISSING:
            self._context.set_static(key, value)
        else:
            raise TypeError("set_static() requires either value or expr=")

    def __getitem__(self, key: str) -> Any:
        # Protected keys are hidden from iteration (__contains__/keys()/len)
        # but readable by name — the LLM already sees their rendered content
        # in the prompt; mutations are guarded by ProtectedBlockError.
        return self._context[key]

    def __delitem__(self, key: str) -> None:
        del self._context[key]

    def __contains__(self, key: object) -> bool:
        return key in self._context and key not in self._context.protected_keys

    def __len__(self) -> int:
        return sum(1 for k in self._context if k not in self._context.protected_keys)

    def __iter__(self) -> Iterator[str]:
        return (k for k in self._context if k not in self._context.protected_keys)

    def keys(self) -> Set[str]:
        """Return user (non-protected) block keys."""
        return self._context.keys() - self._context.protected_keys

    def disable(self, *keys: str) -> None:
        """Suppress named blocks from future prompts without deleting them.

        .. deprecated::
            Use ``self.context["key"] = None`` instead.
        """
        import warnings

        warnings.warn(
            "disable() is deprecated. Use self.context[key] = None instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._context.disable(*keys)

    def enable(self, *keys: str) -> None:
        """Re-enable a previously suppressed block.

        .. deprecated::
            Use ``self.context["key"] = Context(...)`` or ``self.context["key"] = "value"`` instead.
        """
        import warnings

        warnings.warn(
            "enable() is deprecated. Assign a value to re-enable: self.context[key] = Context(...)",
            DeprecationWarning,
            stacklevel=2,
        )
        self._context.enable(*keys)

    def disabled(self) -> "set[str]":
        """Return the set of block keys currently suppressed from prompts."""
        return self._context.disabled()

    def is_enabled(self, key: str) -> bool:
        """Return True if a block key is currently eligible for rendering."""
        return self._context.is_enabled(key)

    def set(
        self, key: str, value: str | None = None, *, expr: str | None = None, prefix: bool = False
    ) -> None:
        """Set a context block — convenience method with keyword arguments.

        Equivalent to ``self.context[key] = Context(value, expr=expr, prefix=prefix)``
        but friendlier for LLMs and developers who prefer explicit keyword args.

        Args:
            key: Block key.
            value: Literal text content (mutually exclusive with expr).
            expr: Python expression re-evaluated each LLM turn.
            prefix: Place in cacheable prefix if True, volatile suffix if False.

        Examples:
            self.context.set("role", "You are an expert")
            self.context.set("status", expr="f'{self.done}/{self.total}'")
            self.context.set("shell", expr="doc(self.shell)", prefix=True)
            self.context.set("old_block", None)  # suppress
        """
        if value is not None and expr is not None:
            raise TypeError("set() takes value or expr, not both")
        if value is None and expr is None:
            self._context[key] = None
            return
        from nooa.context_blocks import Context

        if expr is not None:
            self._context[key] = Context(expr=expr, prefix=prefix)
        else:
            if prefix:
                self._context[key] = Context(value, prefix=True)
            else:
                self._context[key] = value

    def all_keys(self) -> "set[str]":
        """Return all block keys including protected framework and well-known strategy keys.

        Useful for discovering what keys can be suppressed via ``self.context["key"] = None``.

        Well-known keys:
            - ``system_prompt`` — agent class docstring
            - ``self`` — doc(type(self)) introspection
            - ``state`` — current instance field values
            - ``strategy_prompt`` — strategy instructions
            - ``execution_context`` — available imports/types (CodeAct only)
        """
        known_strategy_keys = {"strategy_prompt", "execution_context"}
        return set(self._context.keys()) | known_strategy_keys

    def get(self, key: str, default: Any = None) -> Any:
        """Get a block value, returning default if not found."""
        return self._context.get(key, default)

    def pop(self, key: str, *args: Any) -> Any:
        """Remove and return a block value."""
        return self._context.pop(key, *args)

    def __repr__(self) -> str:
        return f"ContextApi({len(self)} blocks)"
