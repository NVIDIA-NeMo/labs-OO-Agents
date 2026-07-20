# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory StorageManager — no persistence, drop-in for current behavior."""

from typing import TYPE_CHECKING

from nooa.runtime.event_backend import EventBackend, InMemoryBackend

if TYPE_CHECKING:
    from nooa.agent import Agent


class InMemoryStorageManager:
    """StorageManager that keeps everything in memory.

    This is the default when no storage is configured. Events live in
    an ``InMemoryBackend`` (same as today), and save/restore are no-ops
    that raise — there's nowhere to persist to.

    This allows us to centralize storage in StorageManager, rather than
    having to special-case "no storage provided".
    """

    def __init__(self) -> None:
        self._event_backend: EventBackend = InMemoryBackend()

    @property
    def event_backend(self) -> EventBackend:
        return self._event_backend

    def save_snapshot(self, agent: "Agent") -> str:
        from nooa.errors.storage import StorageNotConfiguredError

        raise StorageNotConfiguredError(
            "InMemoryStorageManager does not support persistence. "
            "Pass a persistent StorageManager to enable save/restore."
        )

    def restore_snapshot(self, snapshot_id: str, agent: "Agent") -> None:
        from nooa.errors.storage import StorageNotConfiguredError

        raise StorageNotConfiguredError(
            "InMemoryStorageManager does not support persistence. "
            "Pass a persistent StorageManager to enable save/restore."
        )
