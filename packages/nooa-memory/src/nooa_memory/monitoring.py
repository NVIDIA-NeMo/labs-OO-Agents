# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-usage monitoring, debug logging, and counters.

Reuses the framework's existing observability instead of inventing a new one:

* **logging** — a dedicated ``nooa_memory`` logger; turn it up with
  ``logging.getLogger("nooa_memory").setLevel(logging.DEBUG)`` or the
  framework's ``enable_logging``.
* **event bus** — emits ``MemoryWritten`` / ``MemoryRecalled`` / ``MemoryInjected``
  / ``ReflectionCompleted`` events on the agent's ``EventManager`` with the
  ``RUNTIME_EVENT`` role, so they show up for any existing event/telemetry
  subscriber but never enter the LLM context.
* **counters** — a ``MemoryStats`` snapshot for programmatic monitoring (used by
  the benchmark to compare memory usage across runs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role


# ---------------------------------------------------------------------------
# Runtime events (RUNTIME_EVENT role => never rendered into LLM context)
# ---------------------------------------------------------------------------
class MemoryWritten(EventBase):
    """A memory was encoded (or a duplicate reinforced)."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    memory_id: str = ""
    mem_type: str = "info"
    op: str = "add"  # "add" | "reinforce"
    importance: float = 5.0


class MemoryRecalled(EventBase):
    """Memories were recalled for a query."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    query: str = ""
    n_results: int = 0
    hops: int = 0
    channel: str = "recalled"  # recalled | searched
    memory_ids: list[str] = []  # which memories surfaced (traces show the ids)


class MemoryInjected(EventBase):
    """Spontaneous-association block was (re)injected into context."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    n_memories: int = 0
    chars: int = 0
    memory_ids: list[str] = []


class ReflectionStarted(EventBase):
    """A consolidation pass began (idle / manual / post_task)."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    trigger: str = "manual"


class ReflectionCompleted(EventBase):
    """An offline consolidation pass finished (possibly interrupted)."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT
    merged: int = 0
    edges_added: int = 0
    rescored: int = 0
    pruned: int = 0
    created: int = 0
    trigger: str = "manual"
    interrupted: bool = False
    stopped_in: str = ""  # op name when interrupted ("" = ran to completion)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
@dataclass
class MemoryStats:
    """Running counters of how the agent has used its memory this process."""

    writes: int = 0  # new memories encoded
    reinforced: int = 0  # dedup-on-write hits (existing memory strengthened)
    recalls: int = 0  # recall() / search() calls
    recalled_items: int = 0  # total memories returned by recalls
    injections: int = 0  # spontaneous context-block (re)injections
    injected_chars: int = 0  # cumulative chars injected
    reflections: int = 0  # consolidation passes
    merged: int = 0  # duplicates merged during reflection
    edges_added: int = 0  # graph edges formed during reflection
    pruned: int = 0  # memories forgotten (archived/deleted)
    store_size: int = 0  # active memories in the store (set on snapshot)
    todos_open: int = 0  # open todo memories (set on snapshot)
    todos_done: int = 0  # closed todos, done + dropped (set on snapshot)
    refs_resolved: int = 0  # reference resolutions that came back LIVE
    refs_dangling: int = 0  # ...that fell back to the write-time snapshot
    cross_owner_recalls: int = 0  # recalls that explicitly widened the owner scope
    injection_ms_total: float = 0.0  # cumulative per-turn injection latency

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"writes={self.writes} reinforced={self.reinforced} recalls={self.recalls} "
            f"recalled_items={self.recalled_items} injections={self.injections} "
            f"injected_chars={self.injected_chars} reflections={self.reflections} merged={self.merged} "
            f"edges_added={self.edges_added} pruned={self.pruned} store_size={self.store_size} "
            f"todos={self.todos_open}open/{self.todos_done}closed "
            f"refs={self.refs_resolved}live/{self.refs_dangling}dangling "
            f"cross_owner_recalls={self.cross_owner_recalls} "
            f"injection_ms={self.injection_ms_total:.0f}"
        )
