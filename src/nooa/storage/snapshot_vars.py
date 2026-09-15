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
"""

import logging
from collections.abc import Iterator, MutableMapping
from typing import Any

from nooa.errors.storage import SerializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import serialize

logger = logging.getLogger(__name__)


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
