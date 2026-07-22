# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection — offline consolidation after a task.

A pure-Python orchestrator running ordered ops that mirror the biological
sequence (clean -> abstract -> renormalise -> forget). The deterministic steps
need no LLM and run by default:

  1. dedup / merge      — fold near-identical memories into a canonical record
  2. edge formation     — link memories whose embeddings are close
  3. re-score importance — bump salient/frequently-recalled memories
  4. prune / forget      — archive memories that have decayed (ForgettingEngine)

An optional ``reasoner`` callable enables the generative abstraction step (abstract
episodes -> skills); when absent that step is skipped so reflection stays fully
offline-testable.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

from pydantic import BaseModel

from nooa_memory.config import ForgetPolicy, ReflectionPolicy
from nooa_memory.embeddings import Embedder
from nooa_memory.forgetting import ForgettingEngine
from nooa_memory.retrieval import _sigmoid, base_level_activation
from nooa_memory.schema import EdgeType, Memory, _now
from nooa_memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _never() -> bool:
    return False


class ReflectionReport(BaseModel):
    """Summary of what a consolidation pass changed (for logs/auditability)."""

    merged: int = 0
    edges_added: int = 0
    rescored: int = 0
    pruned: int = 0
    created: int = 0
    reconciled: int = 0  # clusters where an updated/current value superseded older ones
    superseded: int = 0  # memories archived because they were outdated
    interrupted: bool = False  # a should_stop probe fired before completion
    stopped_in: str | None = None  # the op during/after which the stop was observed
    duration_ms: float = 0.0


class ReflectionEngine:
    """Runs the offline consolidation ops over a memory store."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        config: ReflectionPolicy,
        forget_config: ForgetPolicy,
        *,
        owner: str | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        # Consolidation only ever touches this owner's + unowned memories —
        # it must never merge/archive another agent's records in a shared store.
        self.owner = owner
        self._forgetting = ForgettingEngine(store, forget_config, owner=owner)

    def consolidate(
        self,
        *,
        reasoner: Callable[[list[Memory]], list[Memory]] | None = None,
        reconciler: Callable[[list[Memory]], tuple[Memory | None, list[str]]] | None = None,
    ) -> ReflectionReport:
        return self.consolidate_interruptible(
            should_stop=_never, reasoner=reasoner, reconciler=reconciler
        )

    def consolidate_interruptible(
        self,
        *,
        should_stop: Callable[[], bool],
        reasoner: Callable[[list[Memory]], list[Memory]] | None = None,
        reconciler: Callable[[list[Memory]], tuple[Memory | None, list[str]]] | None = None,
    ) -> ReflectionReport:
        """Run the consolidation ops, checking ``should_stop`` between items.

        Every op commits per item, so stopping anywhere leaves a consistent
        store; the returned report says what completed and where the stop was
        observed. The ops are idempotent over an already-consolidated store —
        an interrupted pass followed by a full pass converges to the same
        state as a never-interrupted one.
        """
        t0 = time.perf_counter()
        report = ReflectionReport()

        def _merge() -> None:
            report.merged = self._merge_duplicates(should_stop)

        def _recon() -> None:
            report.reconciled, report.superseded = self._reconsolidate(reconciler, should_stop)

        def _edges() -> None:
            report.edges_added = self._form_edges(should_stop)

        def _rescore() -> None:
            report.rescored = self._rescore_importance(should_stop)

        def _abstract() -> None:
            report.created = self._abstract_episodes(reasoner, should_stop)

        def _prune() -> None:
            report.pruned = len(self._forgetting.prune(should_stop=should_stop))

        ops: list[tuple[str, Callable[[], None]]] = [("merge_duplicates", _merge)]
        if reconciler is not None:
            ops.append(("reconsolidate", _recon))
        ops.extend([("form_edges", _edges), ("rescore_importance", _rescore)])
        if reasoner is not None:
            ops.append(("abstract", _abstract))
        ops.append(("prune", _prune))

        for name, run in ops:
            if should_stop():
                report.interrupted = True
                report.stopped_in = report.stopped_in or name
                break
            run()
            if should_stop():
                # The flag fired while (or right after) this op ran; the op
                # broke between items, so everything committed is consistent.
                report.interrupted = True
                report.stopped_in = name
                break
        report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return report

    # ------------------------------------------------------------------
    # NREM: dedup / merge
    # ------------------------------------------------------------------
    def _merge_duplicates(self, should_stop: Callable[[], bool] = _never) -> int:
        merged: set[str] = set()
        count = 0
        # Stable order so merges are deterministic; the iteration anchor is canonical.
        anchors = sorted(
            self.store.all_memories(owner=self.owner), key=lambda m: (m.created_at, m.id)
        )
        for m in anchors:
            if should_stop():
                break
            if m.id in merged:
                continue
            emb = self.store.get_embedding(m.id)
            if emb is None:
                continue
            dirty = False
            for nid, cos in self.store.knn(emb, 6, owner=self.owner):
                if nid == m.id or nid in merged or cos < self.config.merge_threshold:
                    continue
                dup = self.store.get(nid)
                if dup is None or dup.type != m.type:
                    continue
                # Fold the duplicate into the canonical memory.
                m.reinforcement_count += 1 + dup.reinforcement_count
                m.strength += dup.strength
                m.importance = max(m.importance, dup.importance)
                m.salience = max(m.salience, dup.salience)
                m.confidence = max(m.confidence, dup.confidence)
                for e in dup.edges:
                    if e.target_id != m.id:
                        m.add_edge(e.target_id, e.type, e.weight)
                m.add_edge(dup.id, EdgeType.REFINES, 1.0)  # provenance
                self.store.archive(dup.id)
                merged.add(nid)
                dirty = True
                count += 1
            if dirty:
                self.store.save(m)
        return count

    # ------------------------------------------------------------------
    # reconsolidation: resolve updated/contradicted facts (keep the current one)
    # ------------------------------------------------------------------
    def _reconsolidate(
        self,
        reconciler: Callable[[list[Memory]], tuple[Memory | None, list[str]]],
        should_stop: Callable[[], bool] = _never,
    ) -> tuple[int, int]:
        """Cluster related memories and let ``reconciler`` resolve outdated values.

        ``reconciler(cluster)`` (cluster sorted oldest->newest) returns
        ``(consolidated_current_memory_or_None, ids_to_archive)``. We archive the
        superseded memories and store the consolidated current one — so retrieval
        stops surfacing stale values. Mirrors memory reconsolidation.
        """
        reconciled = 0
        superseded = 0
        clusters_done = 0
        visited: set[str] = set()
        for m in sorted(
            self.store.all_memories(owner=self.owner), key=lambda x: (x.created_at, x.id)
        ):
            if should_stop():
                break  # checked BEFORE each reconciler (LLM) call
            if m.id in visited:
                continue
            emb = self.store.get_embedding(m.id)
            if emb is None:
                continue
            cluster = [m]
            visited.add(m.id)
            for nid, cos in self.store.knn(emb, self.config.recon_max_cluster, owner=self.owner):
                if nid in visited or cos < self.config.recon_threshold:
                    continue
                c = self.store.get(nid)
                if c is not None:
                    cluster.append(c)
                    visited.add(nid)
            if len(cluster) < 2:
                continue
            if clusters_done >= self.config.max_clusters_per_reflection:
                break  # LLM-cost ceiling; the rest wait for the next run
            clusters_done += 1
            cluster.sort(key=lambda x: (x.created_at, x.id))  # oldest -> newest
            try:
                consolidated, archive_ids = reconciler(cluster)
            except Exception:
                continue
            valid = {c.id for c in cluster}
            archived_here = [aid for aid in (archive_ids or []) if aid in valid]
            if not archived_here:
                continue
            for aid in archived_here:
                self.store.archive(aid)
                superseded += 1
            if consolidated is not None:
                if self.owner is not None and not consolidated.owner:
                    consolidated.owner = self.owner
                for c in cluster:
                    consolidated.add_edge(c.id, EdgeType.REFINES, 1.0)
                self.store.add(consolidated, self.embedder.embed(consolidated.embedding_text()))
            reconciled += 1
        return reconciled, superseded

    # ------------------------------------------------------------------
    # form associative edges between close memories
    # ------------------------------------------------------------------
    def _form_edges(self, should_stop: Callable[[], bool] = _never) -> int:
        added = 0
        for m in self.store.all_memories(owner=self.owner):
            if should_stop():
                break
            emb = self.store.get_embedding(m.id)
            if emb is None:
                continue
            existing = {e.target_id for e in m.edges}
            new = 0
            for nid, cos in self.store.knn(
                emb, self.config.max_edges_per_node + 1, owner=self.owner
            ):
                if nid == m.id or nid in existing:
                    continue
                if cos < self.config.edge_threshold or cos >= self.config.merge_threshold:
                    continue
                m.add_edge(nid, EdgeType.RELATED, weight=round(cos, 4))
                existing.add(nid)
                new += 1
                if new >= self.config.max_edges_per_node:
                    break
            if new:
                self.store.save(m)
                added += new
        return added

    # ------------------------------------------------------------------
    # renormalise importance (salience + access-frequency aware)
    # ------------------------------------------------------------------
    def _rescore_importance(self, should_stop: Callable[[], bool] = _never) -> int:
        now = _now()
        n = 0
        for m in self.store.all_memories(owner=self.owner):
            if should_stop():
                break
            base = _sigmoid(base_level_activation(m.access_log, now, 0.5))
            new_imp = 0.5 * m.importance + 3.0 * m.salience + 2.0 * base
            new_imp = max(0.0, min(10.0, new_imp))
            # Never drop explicitly-protected memories below the forgetting boundary.
            if m.importance >= 8.0:
                new_imp = max(new_imp, 8.0)
            if not math.isclose(new_imp, m.importance, abs_tol=1e-6):
                m.importance = new_imp
                self.store.save(m)
                n += 1
        return n

    # ------------------------------------------------------------------
    # REM: optional generative abstraction (needs an LLM-backed reasoner)
    # ------------------------------------------------------------------
    def _abstract_episodes(
        self,
        reasoner: Callable[[list[Memory]], list[Memory]],
        should_stop: Callable[[], bool] = _never,
    ) -> int:
        episodes = [
            m for m in self.store.all_memories(owner=self.owner) if m.type.value == "episode"
        ]
        if not episodes or should_stop():
            return 0  # never START an LLM call when stopping
        sliced = episodes[: self.config.max_episodes_per_reflection]
        # A reasoner failure (e.g. an empty LLM reply) must be LOUD and retried
        # once — a silent ``return 0`` left consolidation dead for a whole run
        # without any signal (cd82: 17/17 failed calls, zero log lines).
        try:
            new_memories = reasoner(sliced)
        except Exception as exc:
            logger.warning("memory consolidation reasoner failed (%s) — retrying once", exc)
            if should_stop():
                return 0
            try:
                new_memories = reasoner(sliced)
            except Exception as exc2:
                logger.warning(
                    "memory consolidation reasoner failed twice (%s) — skipping this pass", exc2
                )
                return 0
        if should_stop():
            return 0  # stop observed while the reasoner ran: discard, commit nothing
        created = 0
        for nm in new_memories or []:
            if should_stop():
                break
            if self.owner is not None and not nm.owner:
                nm.owner = self.owner
            emb = self.embedder.embed(nm.embedding_text())
            # Link the abstraction back only to episodes the reasoner saw.
            for ep in sliced:
                nm.add_edge(ep.id, EdgeType.DERIVED_FROM, 1.0)
            self.store.add(nm, emb)
            created += 1
        return created
