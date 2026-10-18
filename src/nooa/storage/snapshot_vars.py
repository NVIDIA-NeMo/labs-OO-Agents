# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SnapshotVars — a dict-like store that only keeps snapshot-serializable values.

Agent ``vars`` (``self.v``) and per-todo ``vars`` (``t.v``) are snapshot-backed:
they must round-trip through :func:`serialize`. A single non-serializable value
used to abort the *whole* snapshot, silently losing all durable state on the
next resume.

``SnapshotVars`` moves that check to *write* time. Assigning a value that can't
be serialized logs a warning and **skips the store** — so the container only
ever holds serializable data and the snapshot can never choke on it. Because it
is a ``MutableMapping`` (not a ``dict`` subclass) decorated with
``@snapshotable``, the serializer routes it through the snapshotable branch and
captures its internal ``_data`` (already clean), instead of the strict dict
branch.

Pydantic models can embed ``SnapshotVars`` as a field (``Todo.vars``). Python-mode
``model_dump()`` and ``model_copy()`` keep the ``SnapshotVars`` object. JSON mode
(``model_dump(mode="json")``, ``model_dump_json()``), which is how items cross
sessions, carries plain JSON only: a value that is not JSON data (a pydantic
model, a tuple, ...) raises an error naming its key. Validation accepts a
``SnapshotVars``, a plain dict or ``None`` and never imports or constructs a
class: a payload must not be able to choose what gets instantiated. Non-JSON
values survive only the session snapshot (``nooa.storage.serialization``), whose
allowlist comes from code.
"""

import enum
import logging
from collections.abc import Iterator, MutableMapping
from typing import Any

from nooa.errors.storage import SerializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import serialize

logger = logging.getLogger(__name__)

# Key of a removed tagged JSON form that carried its own deserialization
# allowlist; payloads using it are refused instead of being restored.
_TAGGED_KEY = "__snapshot_vars__"


def _is_json_native(value: Any) -> bool:
    """Whether *value* is plain JSON data (no tuples, models or other objects)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return not isinstance(value, enum.Enum)
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_native(v) for k, v in value.items())
    return False


@snapshotable
class SnapshotVars(MutableMapping):
    """A dict-like store that silently drops non-snapshot-serializable values.

    Behaves like a ``dict`` for all reads/writes/iteration, but a write whose
    value can't be snapshot-serialized is **not stored** — it logs a warning
    instead. This keeps the agent's durable ``vars`` always snapshot-clean, so
    one bad value can't take down the whole snapshot (and with it every other
    var) on the next resume.

    Note for agents: a value that fails this check is *not persisted* — it won't
    survive ``/exit`` + resume. If you need durability, store something
    serializable (a dict/str/number/Pydantic model), not a live object
    (clients, connections, sockets, callables).
    """

    _data: "dict[str, Any]"

    def __init__(self, data: "dict[str, Any] | None" = None) -> None:
        self._data = {}
        if data:
            for k, v in data.items():
                self[k] = v

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            logger.warning(
                "SnapshotVars: key %r (%s) is not a string and will NOT be persisted "
                "(it won't survive /exit + resume): snapshots are JSON, which requires "
                "string keys",
                key,
                type(key).__name__,
            )
            return
        try:
            serialize(value)
        except (SerializationError, TypeError, ValueError, RecursionError) as exc:
            logger.warning(
                "SnapshotVars: value for %r (%s) is not snapshot-serializable and "
                "will NOT be persisted (it won't survive /exit + resume): %s",
                key,
                type(value).__name__,
                exc,
            )
            return
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"SnapshotVars({self._data!r})"

    # ── pydantic integration ─────────────────────

    def to_json_data(self) -> dict[str, Any]:
        """Return the plain-JSON form used by pydantic's JSON mode.

        Raises ``ValueError`` naming the first key whose value is not JSON data
        (a pydantic model, a tuple, ...): only plain JSON crosses sessions.
        """
        for key, value in self._data.items():
            if not _is_json_native(value):
                raise ValueError(
                    f"var {key!r} holds a {type(value).__name__}, which is not plain JSON; "
                    "Todo vars sent to another session must be JSON data "
                    "(dict, list, str, int, float, bool, None)"
                )
        return dict(self._data)

    @classmethod
    def from_data(cls, value: Any) -> "SnapshotVars":
        """Accept a ``SnapshotVars`` as-is, a plain dict or ``None``.

        Never deserializes snapshot envelopes, so validating a model that embeds
        ``SnapshotVars`` cannot import or construct arbitrary classes.
        """
        if isinstance(value, SnapshotVars):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            if len(value) == 1 and _TAGGED_KEY in value:
                raise ValueError(
                    "tagged SnapshotVars payloads are not accepted; vars must be plain JSON"
                )
            return cls(value)
        raise TypeError(f"expected a dict or SnapshotVars, got {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            cls.from_data,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.to_json_data(), when_used="json"
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        return {"type": "object"}
