# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for durable operation ownership and replay."""

import multiprocessing
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nooa.storage import (
    CorruptExecutionStateError,
    EffectClass,
    ExecutionAlreadyRunningError,
    ExecutionConflictError,
    ExecutionKey,
    ExecutionReconciliationError,
    ExecutionStatus,
    ExecutionStore,
    LeaseLostError,
    SQLiteExecutionStore,
    SQLiteStorageManager,
)


def _claim_in_process(database: str, start, results) -> None:
    store = SQLiteExecutionStore(database)
    try:
        start.wait()
        try:
            claim = store.claim(ExecutionKey("process-workflow", "tool-1"), {"value": 7})
            results.put("winner" if claim.executable else "replay")
        except ExecutionAlreadyRunningError:
            results.put("loser")
    finally:
        store.close()


def test_terminal_success_is_replayed_without_reexecuting(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "run.db")
    key = ExecutionKey("workflow-1", "tool-1")

    claim = store.claim(key, {"value": 7}, effect_class=EffectClass.IDEMPOTENT)
    record = store.complete_success(claim, {"answer": 42})
    replay = store.claim(key, {"value": 7})

    assert record.status is ExecutionStatus.SUCCEEDED
    assert not replay.executable
    assert replay.record.result == {"answer": 42}
    assert replay.record.attempt == 1


def test_same_operation_id_rejects_different_request(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "run.db")
    key = ExecutionKey("workflow-1", "tool-1")
    store.claim(key, {"value": 7})

    with pytest.raises(ExecutionConflictError):
        store.claim(key, {"value": 8})


def test_same_operation_id_rejects_different_effect_class(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "run.db")
    key = ExecutionKey("workflow-1", "tool-1")
    store.claim(key, {"value": 7}, effect_class=EffectClass.IRREVERSIBLE)

    with pytest.raises(ExecutionConflictError):
        store.claim(key, {"value": 7}, effect_class=EffectClass.PURE)


def test_live_lease_allows_only_one_owner(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "run.db")
    key = ExecutionKey("workflow-1", "tool-1")
    store.claim(key, {"value": 7}, lease_seconds=60)

    with pytest.raises(ExecutionAlreadyRunningError):
        store.claim(key, {"value": 7}, lease_seconds=60)


def test_expired_lease_fences_stale_worker(tmp_path):
    now = [100.0]
    store = SQLiteExecutionStore(tmp_path / "run.db", clock=lambda: now[0])
    key = ExecutionKey("workflow-1", "tool-1")
    first = store.claim(key, {"value": 7}, lease_seconds=5)

    now[0] = 106.0
    second = store.claim(key, {"value": 7}, lease_seconds=5)

    assert second.record.attempt == 2
    assert second.record.lease_generation > first.record.lease_generation
    with pytest.raises(LeaseLostError):
        store.complete_success(first, "stale")
    assert store.complete_success(second, "fresh").result == "fresh"


@pytest.mark.parametrize(
    "effect_class",
    [EffectClass.TRANSACTIONAL, EffectClass.COMPENSATABLE, EffectClass.IRREVERSIBLE],
)
def test_unsafe_expired_lease_becomes_unknown(tmp_path, effect_class):
    now = [100.0]
    store = SQLiteExecutionStore(tmp_path / "run.db", clock=lambda: now[0])
    key = ExecutionKey("workflow-1", f"{effect_class}-operation")
    first = store.claim(key, {"value": 7}, effect_class=effect_class, lease_seconds=5)

    now[0] = 106.0
    recovery = store.claim(key, {"value": 7}, effect_class=effect_class, lease_seconds=5)

    assert not recovery.executable
    assert recovery.record.status is ExecutionStatus.UNKNOWN
    assert recovery.record.attempt == 1
    with pytest.raises(LeaseLostError):
        store.complete_success(first, "stale")


def test_renew_extends_lease_and_expired_owner_cannot_renew(tmp_path):
    now = [100.0]
    store = SQLiteExecutionStore(tmp_path / "run.db", clock=lambda: now[0])
    key = ExecutionKey("workflow-1", "tool-1")
    claim = store.claim(key, {}, lease_seconds=5)

    now[0] = 104.0
    renewed = store.renew(claim, lease_seconds=10)
    assert renewed.record.lease_expires_at == 114.0

    now[0] = 115.0
    with pytest.raises(LeaseLostError):
        store.renew(renewed, lease_seconds=10)


def test_expired_owner_cannot_commit_without_a_takeover(tmp_path):
    now = [100.0]
    store = SQLiteExecutionStore(tmp_path / "run.db", clock=lambda: now[0])
    claim = store.claim(ExecutionKey("workflow-1", "tool-1"), {}, lease_seconds=5)

    now[0] = 105.0

    with pytest.raises(LeaseLostError):
        store.complete_success(claim, "late")


def test_failed_transaction_start_releases_shared_lock():
    connection = sqlite3.connect(":memory:")
    lock = threading.RLock()
    store = SQLiteExecutionStore(connection, lock=lock)
    connection.close()

    with pytest.raises(sqlite3.ProgrammingError):
        store.claim(ExecutionKey("workflow", "operation"), {})

    acquired_from_another_thread: list[bool] = []

    def acquire_lock() -> None:
        acquired = lock.acquire(timeout=1)
        acquired_from_another_thread.append(acquired)
        if acquired:
            lock.release()

    worker = threading.Thread(target=acquire_lock)
    worker.start()
    worker.join(timeout=2)

    assert acquired_from_another_thread == [True]


def test_unknown_outcome_is_terminal_and_preserves_reason(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "run.db")
    key = ExecutionKey("workflow-1", "external-call-1")
    claim = store.claim(key, {"charge": 100}, effect_class=EffectClass.IRREVERSIBLE)

    store.complete_unknown(claim, "transport closed after remote acceptance")
    replay = store.claim(key, {"charge": 100}, effect_class=EffectClass.IRREVERSIBLE)

    assert replay.record.status is ExecutionStatus.UNKNOWN
    assert replay.record.error == {
        "type": "UnknownOutcome",
        "message": "transport closed after remote acceptance",
    }
    assert not replay.executable


def test_unknown_outcome_can_only_be_resolved_explicitly(tmp_path):
    store = SQLiteExecutionStore(tmp_path / "run.db")
    key = ExecutionKey("workflow-1", "external-call-1")
    claim = store.claim(key, {"charge": 100}, effect_class=EffectClass.IRREVERSIBLE)
    store.complete_unknown(claim, "response was lost")

    resolved = store.reconcile_success(key, {"charge_id": "ch_123"})
    assert resolved.status is ExecutionStatus.SUCCEEDED
    assert resolved.result == {"charge_id": "ch_123"}
    assert (
        store.claim(key, {"charge": 100}, effect_class=EffectClass.IRREVERSIBLE).record.status
        is ExecutionStatus.SUCCEEDED
    )

    with pytest.raises(ExecutionReconciliationError):
        store.reconcile_failure(key, RuntimeError("cannot change a resolved operation"))


def test_corrupt_terminal_payload_fails_closed(tmp_path):
    path = tmp_path / "run.db"
    store = SQLiteExecutionStore(path)
    key = ExecutionKey("workflow-1", "tool-1")
    claim = store.claim(key, {})
    store.complete_success(claim, {"answer": 42})
    store._conn.execute(
        "UPDATE durable_operations SET result = ? WHERE workflow_id = ? AND operation_id = ?",
        ("{not-json", key.workflow_id, key.operation_id),
    )
    store._conn.commit()

    with pytest.raises(CorruptExecutionStateError):
        store.get(key)


def test_v1_database_migrates_additive_operation_tables(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_version (version) VALUES (1)")
    connection.commit()
    connection.close()

    with SQLiteStorageManager(path) as storage:
        version = storage._conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 2
        claim = storage.execution_store.claim(ExecutionKey("workflow", "operation"), {})
        assert claim.executable


def test_transition_history_is_append_only(tmp_path):
    now = [100.0]
    store = SQLiteExecutionStore(tmp_path / "run.db", clock=lambda: now[0])
    key = ExecutionKey("workflow-1", "tool-1")
    first = store.claim(key, {}, lease_seconds=5)
    now[0] = 106.0
    second = store.claim(key, {}, lease_seconds=5)
    store.complete_success(second, {"answer": 42})

    history = store.history(key)
    assert [transition.detail for transition in history] == [
        "claimed",
        "expired lease reclaimed",
        "completed",
    ]
    assert [transition.lease_generation for transition in history] == [1, 2, 2]
    with pytest.raises(LeaseLostError):
        store.complete_success(first, "stale")


def test_two_connections_share_lease_and_fencing(tmp_path):
    path = tmp_path / "run.db"
    first = SQLiteExecutionStore(path)
    second = SQLiteExecutionStore(path)
    key = ExecutionKey("workflow-1", "tool-1")

    claim = first.claim(key, {"value": 7}, lease_seconds=60)
    with pytest.raises(ExecutionAlreadyRunningError):
        second.claim(key, {"value": 7}, lease_seconds=60)

    assert first.complete_failure(claim, RuntimeError("boom")).status is ExecutionStatus.FAILED
    assert second.claim(key, {"value": 7}).record.status is ExecutionStatus.FAILED

    first.close()
    second.close()


def test_concurrent_claims_have_one_winner(tmp_path):
    path = tmp_path / "run.db"
    key = ExecutionKey("workflow-1", "tool-1")

    def claim() -> str:
        store = SQLiteExecutionStore(path)
        try:
            return "winner" if store.claim(key, {"value": 7}).executable else "replay"
        except ExecutionAlreadyRunningError:
            return "loser"
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: claim(), range(2)))

    assert sorted(outcomes) == ["loser", "winner"]


def test_claim_is_process_safe(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    path = str(tmp_path / "run.db")
    workers = [
        context.Process(target=_claim_in_process, args=(path, start, results)) for _ in range(2)
    ]

    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=15)

    assert all(worker.exitcode == 0 for worker in workers)
    assert sorted(results.get(timeout=2) for _ in workers) == ["loser", "winner"]


def test_store_accepts_existing_connection():
    connection = sqlite3.connect(":memory:")
    store = SQLiteExecutionStore(connection)
    assert isinstance(store, ExecutionStore)
    claim = store.claim(ExecutionKey("workflow", "operation"), {})
    assert claim.executable
    connection.close()


def test_sqlite_storage_manager_exposes_shared_execution_store(tmp_path):
    with SQLiteStorageManager(tmp_path / "agent.db") as storage:
        key = ExecutionKey("workflow", "operation")
        claim = storage.execution_store.claim(key, {"value": 7})
        storage.execution_store.complete_success(claim, {"answer": 42})

        assert storage.execution_store.get(key).result == {"answer": 42}
