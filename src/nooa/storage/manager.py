# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage manager protocol for agent persistence.

Defines the StorageManager interface — the single object users pass to
agents for full persistence support. It centralizes persistable events
(via an EventBackend) and state snapshots.

The Agent owns its EventManager; storage owns only the EventBackend
underneath. This separation means swapping storage (``/clear``,
``/session new``, etc.) preserves subscribers and middleware on the
agent's stable EventManager — only the persistence target changes.

"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.runtime.event_backend import EventBackend


@runtime_checkable
class StorageManager(Protocol):
    """Unified storage interface for agent persistence.

    Implementations centralize all agent storage:
    - Event persistence (owns the EventBackend)
    - Snapshot save and restore

    The StorageManager is the single object users pass to agents
    for full persistence support.

    Both ``save_snapshot`` and ``restore_snapshot`` receive the agent
    directly. The implementation reads/writes the agent's internals
    however it sees fit (JSON, DB rows, protobuf, etc.).

    Example (``PostgresStorageManager`` is a hypothetical
    implementation shown only to illustrate the interface; the storage
    backend shipped with this package is ``SQLiteStorageManager``)::

        storage = PostgresStorageManager("postgresql://...")
        agent = MyAgent(storage=storage)

        # Events stream automatically through agent.event_manager,
        # which writes to storage.event_backend. Snapshot explicitly:
        snapshot_id = agent.save()

        # Restore:
        agent = MyAgent.load(snapshot_id, storage=storage)
    """

    @property
    def event_backend(self) -> "EventBackend":
        """The EventBackend for this storage.

        Provides persistence for events (store, get, query). The Agent
        owns the EventManager; this backend is the persistence target
        the manager writes through. Swapping storage means swapping
        this backend underneath the agent's stable EventManager.
        """
        ...

    def save_snapshot(self, agent: "Agent") -> str:
        """Save a snapshot of agent state.

        The implementation reads the agent's internals directly and
        serializes them. See class docstring for available internals.

        Args:
            agent: The agent to snapshot.

        Returns:
            A snapshot_id that can be used to load this snapshot later.
                The format is implementation-defined (UUID, file path,
                DB key, etc.).

        Raises:
            SerializationError: If a context-block value can't be
                serialized by this implementation. Non-serializable
                agent attributes are not fatal: they are logged with a
                warning and skipped rather than raising. To exclude a
                field from snapshots deliberately, annotate it with
                ``Annotated[T, nosnapshot]`` or set ``__nosnapshot__ =
                True`` on its class (see ``storage/markers.py``).
        """
        ...

    def restore_snapshot(self, snapshot_id: str, agent: "Agent") -> None:
        """Restore agent state from a previously saved snapshot.

        The implementation reads the snapshot from storage and applies
        it to the agent by writing to its internals (context blocks,
        method registry, attributes, event manager metadata).

        This is the symmetric counterpart to ``save_snapshot``:
        save reads the agent and writes to storage, restore reads
        storage and writes to the agent.

        Args:
            snapshot_id: The ID returned by ``save_snapshot()``.
            agent: The freshly constructed agent to restore into.

        Raises:
            SnapshotNotFoundError: If snapshot_id is not found in storage.
        """
        ...
