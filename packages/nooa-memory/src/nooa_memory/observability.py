# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived observability: per-memory usage stats and store-level KPIs.

Everything here is computed from the self-contained records (their per-channel
counters + structured ``access_log``) plus the store's ``maintenance_log`` —
shared by the TUI memory explorer, the web viewer's Memory tab, and benchmark
reports. Each KPI maps to a diagnostic question ("is the store used at all?",
"is retrieval earning its context budget?") — see the design plan §5.1.
"""

from __future__ import annotations

import math
from statistics import median

from nooa_memory.forgetting import ForgettingEngine, retention
from nooa_memory.schema import Memory, MemoryType, _now, role_of

_HOUR = 3600.0
_YOUNG_HOURS = 7 * 24.0  # never-fetched only counts memories older than this


def prune_eta(
    memory: Memory, forgetting: ForgettingEngine, now: float | None = None
) -> float | None:
    """Unix timestamp when the forgetting policy would prune this memory.

    None = never under the current policy (protected, or pruning disabled).
    Solves ``retention(dt) < threshold`` for the Ebbinghaus curve.
    """
    cfg = forgetting.config
    if not cfg.enabled or forgetting.is_protected(memory):
        return None
    threshold = cfg.prune_activation_threshold
    if threshold <= 0.0 or threshold >= 1.0:
        return None
    stability_hours = cfg.decay_half_life_hours * (1.0 + math.log1p(max(memory.strength, 0)))
    return memory.last_accessed_at + stability_hours * _HOUR * math.log(1.0 / threshold)


def _fetches(memory: Memory) -> int:
    """Total deliberate + spontaneous fetches (excludes the creation seed)."""
    return (
        memory.recalled_count + memory.searched_count + memory.injected_count + memory.deref_count
    )


def _injected_then_used(memory: Memory) -> bool | None:
    """Was an injected surface later followed by a deliberate use?

    ``None`` means unverifiable: the access log is a capped ring, so on a
    heavily-accessed memory every ``injected`` entry may have rotated out
    even though ``injected_count`` (uncapped) is positive. Only a surviving
    injected entry can answer the question either way.
    """
    injected_at: float | None = None
    for entry in memory.access_log:
        if entry.channel == "injected" and injected_at is None:
            injected_at = entry.ts
        elif (
            injected_at is not None
            and entry.ts >= injected_at
            and entry.channel in ("recalled", "searched", "deref")
        ):
            return True
    return None if injected_at is None else False


def per_memory_usage(
    memory: Memory, *, forgetting: ForgettingEngine, now: float | None = None
) -> dict:
    """The Usage panel for one memory (TUI detail pane / viewer record detail)."""
    now = _now() if now is None else now
    last = memory.access_log[-1] if memory.access_log else None
    ranked = [e for e in memory.access_log if e.rank is not None]
    scored = [e for e in memory.access_log if e.score is not None]
    return {
        "fetches": _fetches(memory),
        "recalled": memory.recalled_count,
        "searched": memory.searched_count,
        "injected": memory.injected_count,
        "reinforced": memory.reinforced_count,
        "deref": memory.deref_count,
        "last_channel": last.channel if last else None,
        "last_ts": last.ts if last else None,
        "last_session_ref": last.session_ref if last else None,
        "last_trace_ref": last.trace_ref if last else None,
        "mean_rank": round(sum(e.rank for e in ranked) / len(ranked), 2) if ranked else None,
        "mean_score": round(sum(e.score for e in scored) / len(scored), 4) if scored else None,
        # Flag only on positive evidence (an injected entry still in the log
        # with no later deliberate use) — never when rotation makes it unknowable.
        "injected_never_used": memory.injected_count > 0 and _injected_then_used(memory) is False,
        "retention": round(retention(memory, now, forgetting.config), 4),
        "prune_eta": prune_eta(memory, forgetting, now),
        "strength": memory.strength,
        "reinforcement_count": memory.reinforcement_count,
    }


def store_kpis(
    store, *, forgetting: ForgettingEngine, owner: str | None = None, now: float | None = None
) -> dict:
    """The dashboard payload: each KPI maps to a diagnostic question."""
    now = _now() if now is None else now
    mems = store.all_memories(owner=owner)
    total = len(mems)
    if total == 0:
        return {"total": 0, "maintenance": store.maintenance_history(5)}

    by_type: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    for m in mems:
        by_type[m.type.value] = by_type.get(m.type.value, 0) + 1
        # Group by ROLE: hierarchical owners would otherwise add one dashboard
        # row per session instance.
        owner_key = role_of(m.owner) or "(unowned)"
        by_owner[owner_key] = by_owner.get(owner_key, 0) + 1

    # Is the store used at all?
    old_enough = [m for m in mems if (now - m.created_at) / _HOUR > _YOUNG_HOURS]
    never_fetched = [m for m in old_enough if _fetches(m) == 0]
    fetch_counts = sorted((_fetches(m) for m in mems), reverse=True)
    total_fetches = sum(fetch_counts)
    top10 = max(1, math.ceil(total * 0.10))
    concentration = sum(fetch_counts[:top10]) / total_fetches if total_fetches else 0.0

    # Is writing calibrated?
    reinforces = sum(m.reinforcement_count for m in mems)

    # Is retrieval earning its budget? Rate over VERIFIABLE memories only —
    # ring rotation makes _injected_then_used None, and counting those either
    # way would bias the rate (see per_memory_usage).
    injected = [m for m in mems if m.injected_count > 0]
    verdicts = [_injected_then_used(m) for m in injected]
    verifiable = [v for v in verdicts if v is not None]
    injected_used = [v for v in verifiable if v]

    # Do todos work?
    todos = [m for m in mems if m.type is MemoryType.TODO]
    open_todos = [t for t in todos if t.status == "open"]
    open_ages_h = [(now - t.created_at) / _HOUR for t in open_todos]

    # Does sharing work?
    cross_reads = sum(
        1
        for m in mems
        for e in m.access_log
        if e.reader_owner
        and m.owner
        and role_of(e.reader_owner) != role_of(m.owner)  # instances aren't cross-owner
    )

    return {
        "total": total,
        "by_type": by_type,
        "by_owner": by_owner,
        "with_references": sum(1 for m in mems if m.references),
        "never_fetched_pct": round(100.0 * len(never_fetched) / len(old_enough), 1)
        if old_enough
        else None,
        "fetch_concentration_top10pct": round(concentration, 3),
        "total_fetches": total_fetches,
        "dedup_reinforces": reinforces,
        "injected_memories": len(injected),
        "injected_used_rate": round(len(injected_used) / len(verifiable), 3)
        if verifiable
        else None,
        "todos_open": len(open_todos),
        "todos_closed": len(todos) - len(open_todos),
        "todo_median_open_age_hours": round(median(open_ages_h), 1) if open_ages_h else None,
        "cross_owner_reads": cross_reads,
        "maintenance": store.maintenance_history(5),
    }
