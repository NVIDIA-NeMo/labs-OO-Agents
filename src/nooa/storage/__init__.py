# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent persistence - StorageManager protocol and implementations."""

from __future__ import annotations

from typing import Any

_MODULE_BY_NAME = {
    "AgentSnapshot": "nooa.storage.snapshot",
    "InMemoryStorageManager": "nooa.storage.in_memory",
    "SKIP": "nooa.storage.serialization",
    "SQLiteStorageManager": "nooa.storage.sqlite",
    "SessionAlreadyActiveError": "nooa.storage.sqlite",
    "StorageManager": "nooa.storage.manager",
    "deserialize": "nooa.storage.serialization",
    "delete_sqlite_database": "nooa.storage.sqlite",
    "nosnapshot": "nooa.storage.markers",
    "serialize": "nooa.storage.serialization",
    "snapshotable": "nooa.storage.markers",
    "snapshot_from_json": "nooa.storage.json_snapshot",
    "snapshot_to_json": "nooa.storage.json_snapshot",
}

__all__ = [
    "AgentSnapshot",
    "InMemoryStorageManager",
    "SKIP",
    "SQLiteStorageManager",
    "SessionAlreadyActiveError",
    "StorageManager",
    "deserialize",
    "delete_sqlite_database",
    "nosnapshot",
    "serialize",
    "snapshotable",
    "snapshot_from_json",
    "snapshot_to_json",
]


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
