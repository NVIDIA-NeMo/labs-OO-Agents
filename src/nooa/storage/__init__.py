# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent persistence — StorageManager protocol and implementations."""

from nooa.storage.durable_execution import (
    CorruptExecutionStateError,
    EffectClass,
    ExecutionAlreadyRunningError,
    ExecutionClaim,
    ExecutionConflictError,
    ExecutionKey,
    ExecutionReconciliationError,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionStore,
    ExecutionTransition,
    LeaseLostError,
    SQLiteExecutionStore,
    initialize_execution_schema,
    request_hash,
)
from nooa.storage.in_memory import InMemoryStorageManager
from nooa.storage.json_snapshot import snapshot_from_json, snapshot_to_json
from nooa.storage.manager import StorageManager
from nooa.storage.markers import nosnapshot, snapshotable
from nooa.storage.serialization import SKIP, deserialize, serialize
from nooa.storage.snapshot import AgentSnapshot
from nooa.storage.sqlite import (
    SessionAlreadyActiveError,
    SQLiteStorageManager,
    delete_sqlite_database,
)

__all__ = [
    "AgentSnapshot",
    "CorruptExecutionStateError",
    "EffectClass",
    "ExecutionAlreadyRunningError",
    "ExecutionClaim",
    "ExecutionConflictError",
    "ExecutionKey",
    "ExecutionReconciliationError",
    "ExecutionRecord",
    "ExecutionStatus",
    "ExecutionStore",
    "ExecutionTransition",
    "InMemoryStorageManager",
    "LeaseLostError",
    "SKIP",
    "SQLiteStorageManager",
    "SQLiteExecutionStore",
    "SessionAlreadyActiveError",
    "StorageManager",
    "deserialize",
    "delete_sqlite_database",
    "nosnapshot",
    "serialize",
    "snapshotable",
    "snapshot_from_json",
    "snapshot_to_json",
    "initialize_execution_schema",
    "request_hash",
]
