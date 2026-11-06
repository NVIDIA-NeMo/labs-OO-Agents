# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared attribute-access facade over an owner's existing ``vars`` mapping.

This generalizes the ``TodoVars`` proxy formerly defined in ``nooa.tools.todo``;
it does not implement a new persistence backend. Core ``Todo.v`` uses this proxy;
applications may opt into it for ``self.v``. ``InteractiveAgent.v`` still uses
its separate ``AgentVars`` proxy and reserved-name rules. It lives beside ``SnapshotVars``
so the core Todo tool does not depend on the CLI or an interactive session host.
Snapshot storage and its owner determine when values are saved and restored.
"""

from typing import Any


class PersistentVars:
    """Durable named state backed by an agent or Todo.

    Choose the scope deliberately:

    - ``self.v`` — identity, stable preferences, environment facts, capabilities,
      and long-running coordination shared across tasks and sessions.
    - ``todo.v`` — plans, findings, artifacts, checkpoints, and verification
      metadata belonging to one task.
    - Cell locals — transient scratch data that does not need durable ownership.

    Examples::

        self.v.user_name = "Ada"           # create or update cross-task state
        todo.v.commit = "abc123"            # create or update task state
        print(self.v.user_name)              # read one
        print(todo.v.items())                # inspect all in one scope
        del todo.v.commit                    # remove one
        self.v.clear()                       # remove all in one scope

    Values must be snapshot-serializable (for example dicts, lists, strings,
    numbers, or Pydantic models); unsupported live objects are not stored.
    This facade does not install ``self.v`` on agents or enable disk persistence;
    an application must supply the owner and configure its snapshot storage.
    Names used by helpers (``keys``, ``items``, ``get``, ``set``, ``clear``) must
    be read with ``get(name)`` and written with ``set(name, value)``. Attribute
    access to those names refers to the method, not the stored value.
    """

    def __init__(self, owner: Any):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._owner.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_") or any(key in cls.__dict__ for cls in type(self).__mro__):
            raise AttributeError(f"{key!r} is reserved by PersistentVars; use set({key!r}, value)")
        self._owner.vars[key] = value

    def __delattr__(self, key: str) -> None:
        if key.startswith("_") or any(key in cls.__dict__ for cls in type(self).__mro__):
            raise AttributeError(f"{key!r} is reserved by PersistentVars")
        try:
            del self._owner.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._owner.vars

    def keys(self) -> list[str]:
        """Return the names stored in this scope."""
        return list(self._owner.vars.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return name-value pairs in this scope for inspection."""
        return list(self._owner.vars.items())

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value, or ``default`` when its name is absent."""
        return self._owner.vars.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value by key, including names reserved by helper methods."""
        self._owner.vars[key] = value

    def clear(self) -> None:
        """Remove every value from this scope."""
        self._owner.vars.clear()

    def __repr__(self) -> str:
        return repr(dict(self._owner.vars.items()))
