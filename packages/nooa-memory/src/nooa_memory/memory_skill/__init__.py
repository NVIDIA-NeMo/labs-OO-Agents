# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Long-term memory as a skill — a thin external adapter over the memory subsystem.

Registers as ``nemo.memory`` (so it lands on the agent as ``self.memory``). It reuses
the existing ``MemoryToolsMixin`` conscious tools verbatim and installs the existing
``MemoryManager`` in ``attach`` — no memory-internal architecture changes.
"""

from __future__ import annotations

from typing import Any

from nooa.skill import Skill
from nooa_memory.config import MemoryConfig
from nooa_memory.manager import MemoryManager, MemoryToolsMixin
from nooa_memory.monitoring import MemoryStats
from nooa_memory.reflection import ReflectionReport

__all__ = ["MemorySkill"]


class MemorySkill(MemoryToolsMixin, Skill):
    """Long-term memory you own and curate.

    Write durable, reusable knowledge with
    ``self.memory.remember(content, type=..., importance=CRITICAL|HIGH|MEDIUM|LOW|TRIVIAL, tags=[...])``;
    retrieve with ``self.memory.recall(query)`` (associative) or ``self.memory.search(query)``
    (term-focused, graph spread disabled); refine with ``self.memory.update_memory(id, ...)``
    / ``self.memory.forget(id)``; link with ``self.memory.associate(a, b, relation)``;
    read a reference with ``self.memory.deref("kind:key")``; and consolidate with
    ``self.memory.reflect()``. ``self.memory.stats()`` returns usage counters.
    """

    __nosnapshot__ = True  # rebuilt + re-attached on resume

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        reasoner: Any | None = None,
        reconciler: Any | None = None,
    ) -> None:
        # Optional args (discovery-safe: instantiated zero-arg). Defaults to enabled.
        # reasoner/reconciler are the generative reflection hooks (see
        # memory/generative.py) — passed through to the manager at attach.
        self._mgr: MemoryManager | None = None
        self._reasoner = reasoner
        self._reconciler = reconciler
        config = config or MemoryConfig(enabled=True)
        # Mounted as a skill, the tools live on self.memory — render the guide
        # accordingly unless the host explicitly chose a prefix.
        if config.api_prefix == "self.":
            config = config.merge_with(api_prefix="self.memory.")
        self._config = config

    def attach(self, agent: Any) -> None:
        super().attach(agent)  # sets self._agent
        # The manager owns guide injection (rendered with this skill's prefix).
        self._mgr = MemoryManager.install(
            agent,
            config=self._config,
            reasoner=self._reasoner,
            reconciler=self._reconciler,
        )

    def detach(self) -> None:
        if self._mgr is not None:
            self._mgr.uninstall()
            self._mgr = None
        super().detach()

    # The single host hook the inherited MemoryToolsMixin bodies resolve through.
    def _memory_or_raise(self) -> MemoryManager:
        mgr = self._mgr
        if mgr is None or not mgr.config.enabled:
            raise RuntimeError("MemorySkill is not attached/enabled.")
        return mgr

    # remember / recall / search / update_memory / forget / associate: inherited verbatim.

    def reflect(self) -> ReflectionReport:
        """Consolidate memories: merge duplicates, form links, prune, reconsolidate."""
        return self._memory_or_raise().reflect()

    def stats(self) -> MemoryStats:
        """A snapshot of how this agent has used its memory (writes/recalls/…)."""
        return self._memory_or_raise().memory_stats()
