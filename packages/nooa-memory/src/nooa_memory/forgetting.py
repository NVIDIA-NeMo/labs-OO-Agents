# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forgetting — first-class, on two timescales.

* Online (no LLM): each memory's *retention* decays with time since last access
  and is slowed by ``strength`` (which retrieval bumps). This is the
  Ebbinghaus/MemoryBank ``R = exp(-Δt / S)`` curve, evaluated lazily on read.
* Offline (in reflection): memories whose retention has fallen below threshold
  (and which are old enough, not a protected type, and not high-importance) are
  pruned — archived (tombstoned) by default, or hard-deleted.

"""

from __future__ import annotations

import math
from collections.abc import Callable

from nooa_memory.config import ForgetPolicy
from nooa_memory.schema import Memory, MemoryType, _now
from nooa_memory.store import MemoryStore

_HOUR = 3600.0


def retention(memory: Memory, now: float, cfg: ForgetPolicy) -> float:
    """Ebbinghaus retention in (0, 1]: ``exp(-Δt / S)``.

    ``S`` (memory stability) scales with ``strength`` so frequently-recalled
    memories decay much more slowly. Fully recent memories return ~1.0.
    """
    dt_hours = max(now - memory.last_accessed_at, 0.0) / _HOUR
    stability = cfg.decay_half_life_hours * (1.0 + math.log1p(max(memory.strength, 0)))
    if stability <= 0:
        return 0.0
    return math.exp(-dt_hours / stability)


class ForgettingEngine:
    """Computes online retention and performs offline pruning."""

    def __init__(
        self, store: MemoryStore, config: ForgetPolicy, *, owner: str | None = None
    ) -> None:
        self.store = store
        self.config = config
        # Pruning only ever touches this owner's + unowned memories.
        self.owner = owner

    def is_protected(self, memory: Memory) -> bool:
        if memory.type.value in self.config.protected_types:
            return True
        # An unresolved commitment must never be silently forgotten.
        if memory.type is MemoryType.TODO and memory.status == "open":
            return True
        # High-importance memories are never auto-forgotten.
        return memory.importance >= 8.0

    def should_prune(self, memory: Memory, now: float | None = None) -> bool:
        if not self.config.enabled or memory.archived:
            return False
        if self.is_protected(memory):
            return False
        now = _now() if now is None else now
        age_hours = (now - memory.created_at) / _HOUR
        if age_hours < self.config.prune_min_age_hours:
            return False
        return retention(memory, now, self.config) < self.config.prune_activation_threshold

    def prune(
        self,
        *,
        now: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[str]:
        """Archive/delete all memories that have decayed below threshold.

        Returns the ids that were pruned. ``should_stop`` is checked between
        items (idle reflection interrupts here too).
        """
        if not self.config.enabled:
            return []
        now = _now() if now is None else now
        pruned: list[str] = []
        for m in self.store.all_memories(owner=self.owner):
            if should_stop is not None and should_stop():
                break
            if self.should_prune(m, now):
                if self.config.archive_vs_delete == "delete":
                    self.store.delete(m.id)
                else:
                    self.store.archive(m.id)
                pruned.append(m.id)
        return pruned


# expose for tests / external use
__all__ = ["retention", "ForgettingEngine", "MemoryType"]
