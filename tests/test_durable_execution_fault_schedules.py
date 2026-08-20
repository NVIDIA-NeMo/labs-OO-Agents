# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large deterministic fault schedule for durable execution recovery."""

import random

import pytest

from nooa.storage import (
    EffectClass,
    ExecutionKey,
    ExecutionStatus,
    LeaseLostError,
    SQLiteExecutionStore,
)


def test_10_000_seeded_recovery_schedules():
    now = [0.0]
    store = SQLiteExecutionStore(":memory:", clock=lambda: now[0])
    effect_classes = tuple(EffectClass)

    for seed in range(10_000):
        randomizer = random.Random(seed)
        effect_class = randomizer.choice(effect_classes)
        key = ExecutionKey("seeded-faults", str(seed))
        request = {"seed": seed, "payload": randomizer.randrange(1_000_000)}
        now[0] = float(seed * 10)
        first = store.claim(key, request, effect_class=effect_class, lease_seconds=2)

        if randomizer.choice((True, False)):
            store.complete_success(first, {"seed": seed})
            replay = store.claim(key, request, effect_class=effect_class)
            assert not replay.executable
            assert replay.record.status is ExecutionStatus.SUCCEEDED
            continue

        now[0] += 3.0
        recovery = store.claim(key, request, effect_class=effect_class, lease_seconds=2)
        with pytest.raises(LeaseLostError):
            store.complete_success(first, {"stale": seed})

        if effect_class in {EffectClass.PURE, EffectClass.IDEMPOTENT}:
            assert recovery.executable
            assert recovery.record.attempt == 2
            store.complete_success(recovery, {"seed": seed})
        else:
            assert not recovery.executable
            assert recovery.record.status is ExecutionStatus.UNKNOWN

        terminal = store.claim(key, request, effect_class=effect_class)
        assert not terminal.executable
        assert terminal.record.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.UNKNOWN}
