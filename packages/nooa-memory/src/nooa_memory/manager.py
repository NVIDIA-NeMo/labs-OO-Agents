# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MemoryManager — the additive integration layer.

``MemoryManager.install(agent, config=...)`` wires an opt-in memory subsystem
onto any existing agent with **zero core edits**, mirroring
``agents/summarization.py``:

* pre-turn **spontaneous association** — a dynamic context block refreshed on
  ``BeforeTurn`` (cadence configurable) via ``ContextManager.set_dynamic``.
* **write-on-event** — ``on(<event>)`` subscriptions encode salient events.
* post-task **reflection** — ``intercept("agent_call")`` gated to the top-level
  call runs consolidation after the method returns.

``MemoryToolsMixin`` exposes the conscious tools (``recall``/``search``/
``remember``/``associate``) so they render in ``doc(self)``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from nooa.agent import Agent
from nooa_memory.config import MemoryConfig
from nooa_memory.descriptors import to_numeric, to_status
from nooa_memory.embeddings import Embedder, get_embedder
from nooa_memory.forgetting import ForgettingEngine
from nooa_memory.monitoring import (
    MemoryInjected,
    MemoryRecalled,
    MemoryStats,
    MemoryWritten,
    ReflectionCompleted,
    ReflectionStarted,
)
from nooa_memory.references import capture, parse_ref, render, resolve
from nooa_memory.reflection import ReflectionEngine, ReflectionReport
from nooa_memory.retrieval import RetrievalEngine, derive_queries
from nooa_memory.schema import (
    AccessRecord,
    EdgeType,
    Memory,
    MemoryRef,
    MemoryType,
    role_of,
)
from nooa_memory.store import MemoryStore
from nooa_memory.tracing_bridge import current_trace_ref, install_tracing_bridge

logger = logging.getLogger(__name__)
# Dedicated logger for memory-usage monitoring/debug (raise to DEBUG to trace ops).
mem_logger = logging.getLogger("nooa_memory")

# Salience/importance defaults per auto-written event type.
_EVENT_SALIENCE: dict[str, tuple[float, float]] = {
    # event_type: (salience, importance)
    "Error": (0.9, 7.0),
    "Notification": (0.5, 5.0),
    "Message": (0.4, 5.0),
}

# The instruction injected into the agent's context when memory is installed —
# a TEMPLATE: {api} is the host's tool prefix ("self." for the mixin install,
# "self.memory." for the skill), so the guide never documents a missing method.
# This is the heart of the design: the AGENT owns and curates its memory.
MEMORY_SCHEMA_GUIDE = """\
## Your long-term memory

You have a persistent long-term memory that YOU own and curate: you decide what
to keep and how to structure it. The runtime also auto-writes a few operational
memories (salient events, task episodes) — curate those like your own: refine or
forget them when they are wrong. Store: {store} (your identity: {owner}).

- WRITE durable, reusable knowledge — `{api}remember(content, type=..., importance=..., tags=[...])`.
  Store distilled facts/skills/decisions, ONE self-contained item each — never raw
  transcripts or chit-chat. A near-duplicate write reinforces the existing memory
  and returns ITS id instead of creating a new one — if your new wording is
  sharper, follow up with `{api}update_memory(id, content=...)`.
- REFINE — when a memory becomes more accurate, `{api}update_memory(id, ...)`; when it
  is wrong or obsolete, `{api}forget(id)`; link related memories with
  `{api}associate(id_a, id_b, relation)` where relation is one of: related |
  supports | contradicts | refines | derived_from | created_by | causes |
  precedes | part_of | triggers.
- RECALL — `{api}recall(query)` (associative: semantic + keyword + recency + graph)
  or `{api}search(query)` (term-focused, graph spread disabled). Recall before work
  that touches prior decisions, preferences, plans, or files; skip it for one-off
  questions. Recalled and injected memories are labeled [type#id] — pass that id
  (the 8-char prefix suffices) to update_memory/forget/associate. Injected
  "Recalled memories" are hints, not ground truth.
- PASS BY REFERENCE — when a memory is about live state (a var, a file, a todo),
  store a short description plus `references=["var:plan", "file:docs/spec.md"]`
  instead of pasting the value: references are re-read fresh at recall time
  (`{api}deref("var:plan")` reads one on demand), so the memory cannot go stale.

Memory schema — the fields YOU set: type, importance, tags, title, references,
and (todos only) status:
  type:        info       — a durable fact, preference, rule, or convention
               skill      — a reusable, verified procedure / how-to
               episode    — what happened in a specific task or event
               intent     — a trigger-based reminder ("when X happens, do Y")
               todo       — a durable commitment to act, tracked until resolved;
                            close it with {api}update_memory(id, status="DONE")
                            (or "DROPPED" if abandoned)
               reflection — an insight distilled from several episodes
  importance:  CRITICAL | HIGH | MEDIUM | LOW | TRIVIAL — how much this should influence future decisions
  tags:        salient entities/keywords for retrieval (names, files, dates, topics)
For task tracking WITHIN this session use self.todo (when available); use
type="todo" only for commitments that must survive the session.
Memories you write are yours; on a shared store, `{api}recall(query, owner="*")`
also searches what other agents learned (read-only — you cannot edit theirs).
"""


def render_schema_guide(*, api: str = "self.", store: str = "", owner: str = "") -> str:
    """Render the guide for a host: its tool prefix, store path, and identity."""
    return MEMORY_SCHEMA_GUIDE.format(
        api=api, store=store or "(in-memory)", owner=owner or "(shared)"
    )


class MemoryManager:
    """Owns the store + engines and the agent hooks for one agent."""

    def __init__(
        self,
        agent: Agent,
        config: MemoryConfig | None = None,
        *,
        embedder: Embedder | None = None,
        reasoner: Callable[[list[Memory]], list[Memory]] | None = None,
        reconciler: Callable[[list[Memory]], tuple[Memory | None, list[str]]] | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or MemoryConfig()
        # Writer identity for shared stores: explicit config wins (hosts like
        # the TUI pass their stable per-agent key; "" writes to the shared
        # namespace); the honest library default is the agent's class name.
        self.owner = self.config.owner if self.config.owner is not None else type(agent).__name__
        self.embedder = embedder or get_embedder(self.config.embedding)
        self.store = self._make_store(agent)
        self.retrieval = RetrievalEngine(
            self.store,
            self.embedder,
            self.config.retrieval,
            access_log_cap=self.config.observability.access_log_cap,
        )
        # Engines consolidate at ROLE scope: instance scope would fragment
        # knowledge per session (and reflection must fold across instances).
        self.reflection_engine = ReflectionEngine(
            self.store,
            self.embedder,
            self.config.reflection,
            self.config.forget,
            owner=self.role,
        )
        self.forgetting = ForgettingEngine(self.store, self.config.forget, owner=self.role)
        self._reasoner = reasoner
        self._reconciler = reconciler
        self.stats = MemoryStats()
        # Set by hosts (TUI session id, bench run id); lands on AccessRecords.
        self.session_ref: str | None = None
        self._last_injected_ids: list[str] = []

        # hook state
        self._unsubs: list[Callable[[], None]] = []
        self._pending: list[asyncio.Task[Any]] = []
        self._call_depth = 0
        self._reflecting = False
        self._last_query_hash: int | None = None
        self._primed = False

        self._install_hooks()

    def _make_store(self, agent: Agent) -> MemoryStore:
        path = self.config.path or self._default_path(agent)
        return MemoryStore(path, vector_config=self.config.vector, embedding_dim=self.embedder.dim)

    # ------------------------------------------------------------------
    # install / uninstall
    # ------------------------------------------------------------------
    @classmethod
    def install(
        cls, agent: Agent, *, config: MemoryConfig | None = None, **kwargs: Any
    ) -> MemoryManager:
        """Attach memory to ``agent``. Stored as ``agent._memory`` (lifetime tied)."""
        existing = getattr(agent, "_memory", None)
        if isinstance(existing, MemoryManager):
            existing.uninstall()
        manager = cls(agent, config=config, **kwargs)
        agent._memory = manager  # type: ignore[attr-defined]
        return manager

    @staticmethod
    def _default_path(agent: Agent) -> str:
        from pathlib import Path

        return str(Path(".nooa") / "memory" / "memory.sqlite")

    def _install_hooks(self) -> None:
        em = self.agent.event_manager
        if not self.config.enabled:
            return  # additive guarantee: install is inert when disabled
        # Tell the agent it owns its memory + the schema to write it with.
        if self.config.instruct:
            try:
                self.agent.context_manager.set_static(
                    self.config.instruct_block_key,
                    render_schema_guide(
                        api=self.config.api_prefix, store=self.store.path, owner=self.owner
                    ),
                )
            except Exception:
                mem_logger.debug("memory: could not inject instruction block", exc_info=True)
        # reset per-task injection state on task boundaries
        self._unsubs.append(em.on("Task", self._on_task))
        if self.config.spontaneous.enabled:
            self._unsubs.append(em.on("BeforeTurn", self._on_before_turn))
        for evt in self.config.write.on_events:
            self._unsubs.append(em.on(evt, self._on_write_event))
        if self.config.reflection.enabled and self.config.reflection.trigger == "post_task":
            self._unsubs.append(em.intercept("agent_call", self._reflect_middleware))
        # Bridge memory events into tracing spans (no-op without opentelemetry).
        self._unsubs.extend(install_tracing_bridge(self))

    def uninstall(self) -> None:
        """Remove all hooks and cancel pending background work."""
        if self.config.instruct:
            try:
                del self.agent.context_manager[self.config.instruct_block_key]
            except Exception:
                pass
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()
        for task in self._pending:
            if not task.done():
                task.cancel()
        self._pending.clear()
        self.store.close()
        if getattr(self.agent, "_memory", None) is self:
            del self.agent._memory  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # conscious operations (used by MemoryToolsMixin + internally)
    # ------------------------------------------------------------------
    def remember(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.INFO,
        importance: float | None = None,
        salience: float = 0.0,
        title: str | None = None,
        tags: list[str] | None = None,
        source_task_ref: str | None = None,
        dedup: bool = True,
        references: list[str] | None = None,
    ) -> str:
        """Encode a memory (dedup-on-write). Returns the memory id."""
        mem = Memory(
            content=content,
            type=type,
            title=title,
            importance=self.config.write.default_importance if importance is None else importance,
            salience=salience,
            tags=tags or [],
            source_task_ref=source_task_ref,
            owner=self.owner,
            references=[capture(self.agent, self.store, r) for r in references or []],
        )
        emb = self.embedder.embed(mem.embedding_text())

        if dedup and self.store.count() > 0:
            # Dedup within own scope only — never reinforce another agent's memory.
            for nid, cos in self.store.knn(emb, self.config.write.dedup_top_k, owner=self.role):
                if cos >= self.config.write.dedup_threshold:
                    existing = self.store.get(nid)
                    if existing is not None and existing.type == type:
                        existing.log_access(
                            self._access("reinforced"),
                            cap=self.config.observability.access_log_cap,
                        )
                        existing.reinforcement_count += 1
                        existing.importance = max(existing.importance, mem.importance)
                        existing.salience = max(existing.salience, mem.salience)
                        self.store.save(existing)
                        self.stats.reinforced += 1
                        self._emit(
                            MemoryWritten(
                                memory_id=existing.id,
                                mem_type=type.value,
                                op="reinforce",
                                importance=existing.importance,
                            )
                        )
                        mem_logger.debug(
                            "memory.reinforce id=%s type=%s", existing.id[:8], type.value
                        )
                        return existing.id  # NOOP: reinforced the duplicate

        self.store.add(mem, emb)
        self.stats.writes += 1
        self._emit(
            MemoryWritten(
                memory_id=mem.id, mem_type=type.value, op="add", importance=mem.importance
            )
        )
        mem_logger.debug(
            "memory.write id=%s type=%s imp=%.1f", mem.id[:8], type.value, mem.importance
        )
        return mem.id

    @property
    def role(self) -> str:
        """The role part of this manager's hierarchical owner (``role[@instance]``)."""
        return role_of(self.owner)

    def _resolve_owner(self, owner: str | None) -> str | None:
        """Read-scope policy (hierarchical owners, design §5b).

        None = my ROLE (+ unowned) — every instance of this agent; '*' =
        everyone; a bare role = that role's instances; ``role@inst`` = that
        exact instance.
        """
        if owner is None:
            return self.role
        if owner == "*":
            return None
        return owner

    def _access(self, channel: str) -> AccessRecord:
        """Template access record carrying this manager's read context."""
        return AccessRecord(
            ts=time.time(),
            channel=channel,
            reader_owner=self.owner,
            session_ref=self.session_ref,
            trace_ref=current_trace_ref(),
        )

    def _assert_writable(self, m: Memory) -> None:
        # Role-based: any instance of the same role curates the role's memories.
        if m.owner != "" and role_of(m.owner) != self.role:
            raise PermissionError(
                f"memory {m.id[:8]} belongs to {m.owner!r}; cross-owner writes are not allowed"
            )

    def recall(
        self,
        query: str,
        k: int | None = None,
        *,
        hops: int | None = None,
        owner: str | None = None,
        channel: str = "recalled",
    ) -> list[Memory]:
        """Associative + keyword recall, scored and ranked."""
        if owner is not None and owner != self.owner:
            self.stats.cross_owner_recalls += 1
        res = self.retrieval.recall(
            query,
            k=k,
            hops=hops,
            owner=self._resolve_owner(owner),
            access=self._access(channel),
        )
        eff_hops = self.config.retrieval.hops if hops is None else hops
        self.stats.recalls += 1
        self.stats.recalled_items += len(res)
        self._emit(
            MemoryRecalled(
                query=query[:200],
                n_results=len(res),
                hops=eff_hops,
                channel=channel,
                memory_ids=[m.id for m in res],
            )
        )
        mem_logger.debug("memory.recall q=%r hops=%d -> %d results", query[:60], eff_hops, len(res))
        return res

    def _find(self, memory_id: str) -> Memory | None:
        """Exact id, or the unique 8-char prefix rendered in recalled lines."""
        resolved = self.store.resolve_id(memory_id)
        return self.store.get(resolved) if resolved else None

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        references: list[str] | None = None,
    ) -> bool:
        """Refine an existing memory in place (re-embeds if content changed)."""
        m = self._find(memory_id)
        if m is None:
            return False
        self._assert_writable(m)
        content_changed = content is not None and content != m.content
        if content is not None:
            m.content = content
        if importance is not None:
            m.importance = max(0.0, min(10.0, importance))
        if type is not None:
            m.type = type
            # Switching taxonomy: a todo opens its lifecycle; nothing else has one.
            if m.type is MemoryType.TODO and m.status is None:
                m.status = "open"
            elif m.type is not MemoryType.TODO:
                m.status = None
        if status is not None:
            if m.type is not MemoryType.TODO:
                raise ValueError("status applies only to todo memories")
            m.status = to_status(status)
        if tags is not None:
            m.tags = tags
        if references is not None:
            m.references = [capture(self.agent, self.store, r) for r in references]
        m.reinforcement_count += 1
        m.log_access(
            self._access("reinforced"),
            reinforce=False,
            cap=self.config.observability.access_log_cap,
        )
        if content_changed:
            self.store.add(m, self.embedder.embed(m.embedding_text()))  # re-embed + replace
        else:
            self.store.save(m)
        self._emit(
            MemoryWritten(
                memory_id=m.id, mem_type=m.type.value, op="update", importance=m.importance
            )
        )
        mem_logger.debug("memory.update id=%s", m.id[:8])
        return True

    def forget(self, memory_id: str) -> bool:
        """Archive (tombstone) a memory the agent judges wrong or obsolete."""
        m = self._find(memory_id)
        if m is None:
            return False
        self._assert_writable(m)
        self.store.archive(m.id)
        self.stats.pruned += 1
        self._emit(
            MemoryWritten(
                memory_id=m.id, mem_type=m.type.value, op="forget", importance=m.importance
            )
        )
        mem_logger.debug("memory.forget id=%s", m.id[:8])
        return True

    def explain(self, query: str, *, k: int = 10, owner: str | None = None) -> list[dict]:
        """Dry-run recall: the scored candidate table (no touch, no logging)."""
        return self.retrieval.explain(query, k=k, owner=self._resolve_owner(owner))

    def _emit(self, event: Any) -> None:
        """Emit a runtime memory event on the agent's bus (never enters LLM context)."""
        try:
            self.agent.event_manager.add(event, record=False)
        except Exception:
            pass

    def memory_stats(self) -> MemoryStats:
        """Snapshot of how the agent has used its memory (also refreshes store_size)."""
        self.stats.store_size = self.store.count()
        todos = [m for m in self.store.all_memories(owner=self.role) if m.type is MemoryType.TODO]
        self.stats.todos_open = sum(1 for t in todos if t.status == "open")
        self.stats.todos_done = len(todos) - self.stats.todos_open
        return self.stats

    def log_summary(self) -> None:
        """Log a one-line memory-usage summary at INFO."""
        mem_logger.info("memory stats: %s", self.memory_stats().summary())

    def deref(self, ref: str) -> str:
        """Resolve a ``kind:key`` reference against live state and render it."""
        kind, key = parse_ref(ref)
        resolved = resolve(self.agent, self.store, MemoryRef(kind=kind, key=key))
        self._count_resolution(resolved.status)
        if kind == "memory":
            # Agents pass the 8-char prefixes that recalled lines render
            # ([type#id8]) — resolve exactly like references._lookup does, so
            # the memory that rendered LIVE is the one whose access gets logged.
            try:
                full_id = self.store.resolve_id(key)
            except ValueError:
                full_id = None  # ambiguous prefix: resolve() rendered DANGLING
            target = self.store.get(full_id) if full_id else None
            if target is not None:
                target.log_access(
                    self._access("deref"), cap=self.config.observability.access_log_cap
                )
                self.store.save(target)
        return render(resolved)

    def _count_resolution(self, status: str) -> None:
        if status == "LIVE":
            self.stats.refs_resolved += 1
        else:
            self.stats.refs_dangling += 1

    def associate(self, a_id: str, b_id: str, relation: str = "related") -> None:
        """Add a directed graph edge ``a_id -> b_id`` (ids may be unique prefixes)."""
        etype = EdgeType(relation)  # raises on an unknown relation (no silent RELATED)
        src = self._find(a_id)
        if src is not None:
            self._assert_writable(src)  # the edge mutates the source's neighborhood
            a_id = src.id
        dst = self._find(b_id)
        if dst is not None:
            b_id = dst.id
        self.store.add_edge(a_id, b_id, etype)

    def reflect(self, *, trigger: str = "manual") -> ReflectionReport:
        """Run a consolidation pass (background thread when configured).

        With ``config.reflection.background=True`` the pass runs on a daemon
        thread and this returns immediately with an empty report — the caller's
        turn must not pay for serial consolidation LLM calls (600-1,500s/game
        on the ARC fleet). Stats/bookkeeping update when the pass completes.
        """
        if self.config.reflection.background:
            thread = getattr(self, "_manual_reflect_thread", None)
            if thread is not None and thread.is_alive():
                return ReflectionReport()  # a pass is already running — don't stack

            self._emit(ReflectionStarted(trigger=trigger))

            def _run() -> None:
                report = self.reflection_engine.consolidate(
                    reasoner=self._reasoner, reconciler=self._reconciler
                )
                self._finish_reflection(report, trigger=trigger)

            thread = threading.Thread(target=_run, name="memory-reflect", daemon=True)
            self._manual_reflect_thread = thread
            thread.start()
            return ReflectionReport()

        self._emit(ReflectionStarted(trigger=trigger))
        report = self.reflection_engine.consolidate(
            reasoner=self._reasoner, reconciler=self._reconciler
        )
        return self._finish_reflection(report, trigger=trigger)

    def reflect_interruptible(
        self, should_stop: Callable[[], bool], *, trigger: str = "idle"
    ) -> ReflectionReport:
        """A consolidation pass that stops within one item of ``should_stop``.

        The idle-reflection entry point (executor-friendly, sync): per-item
        commits make an interrupted pass safe, and the report says where the
        stop was observed. Same bookkeeping as ``reflect()``.
        """
        self._emit(ReflectionStarted(trigger=trigger))
        report = self.reflection_engine.consolidate_interruptible(
            should_stop=should_stop, reasoner=self._reasoner, reconciler=self._reconciler
        )
        return self._finish_reflection(report, trigger=trigger)

    def _finish_reflection(self, report: ReflectionReport, *, trigger: str) -> ReflectionReport:
        self.stats.reflections += 1
        self.stats.merged += report.merged
        self.stats.edges_added += report.edges_added
        self.stats.pruned += report.pruned + report.superseded
        self._emit(
            ReflectionCompleted(
                merged=report.merged,
                edges_added=report.edges_added,
                rescored=report.rescored,
                pruned=report.pruned,
                created=report.created,
                trigger=trigger,
                interrupted=report.interrupted,
                stopped_in=report.stopped_in or "",
                duration_ms=report.duration_ms,
            )
        )
        self.store.log_maintenance("reflect", {"trigger": trigger, **report.model_dump()})
        mem_logger.info("memory.reflect trigger=%s %s", trigger, report.model_dump())
        return report

    # ------------------------------------------------------------------
    # spontaneous association (pre-turn injection)
    # ------------------------------------------------------------------
    def recall_for_context(self) -> str:
        """Derive a query from agent state and format recalled memories."""
        if not self.config.enabled or not self.config.spontaneous.enabled:
            return ""
        queries = derive_queries(self.agent, self.config.spontaneous)
        return self._format_recall(queries)

    def _memory_lines(self, m: Memory) -> list[str]:
        head = (m.title or m.content).replace("\n", " ").strip()
        tag = f"{m.type.value}:{m.status}" if m.type is MemoryType.TODO else m.type.value
        lines = [f"- [{tag}#{m.id[:8]}] {head}"]
        # References resolve fresh against live state (pass-by-reference).
        for ref in m.references[:4]:
            resolved = resolve(self.agent, self.store, ref)
            self._count_resolution(resolved.status)
            lines.append(f"    {render(resolved)}")
        return lines

    def _open_todos(self, k: int) -> list[Memory]:
        """Open todos by importance (desc) then age (oldest first), capped at k."""
        todos = [
            m
            for m in self.store.all_memories(owner=self.role)
            if m.type is MemoryType.TODO and m.status == "open"
        ]
        todos.sort(key=lambda m: (-m.importance, m.created_at))
        return todos[:k]

    def _format_recall(self, queries: list[str]) -> str:
        if not queries:
            return ""
        sp = self.config.spontaneous
        k = sp.top_k or self.config.retrieval.top_k
        seen: dict[str, Memory] = {}
        for q in queries:
            for m in self.retrieval.recall(q, k=k, touch=False, owner=self.role):
                if sp.inject_open_todos == "off" and m.type is MemoryType.TODO:
                    continue  # todos are reachable via explicit recall/search only
                seen.setdefault(m.id, m)
        shown = list(seen.values())[:k]
        lines: list[str] = []
        if shown:
            lines.append("## Recalled memories (associative)")
            for m in shown:
                lines.extend(self._memory_lines(m))
        extra: list[Memory] = []
        if sp.inject_open_todos == "always":
            listed = {m.id for m in shown}
            extra = [t for t in self._open_todos(sp.open_todos_k) if t.id not in listed]
            if extra:
                lines.append("## Open todos")
                for t in extra:
                    lines.extend(self._memory_lines(t))
        if not lines:
            self._last_injected_ids = []
            return ""
        injected = shown + extra
        self._last_injected_ids = [m.id for m in injected]
        if self.config.observability.log_injections:
            # Logged WITHOUT touching: injection stays invisible to ACT-R.
            for m in injected:
                m.log_access(self._access("injected"), cap=self.config.observability.access_log_cap)
                self.store.save(m)
        text = "\n".join(lines)
        if len(text) > sp.context_char_budget:
            text = text[: sp.context_char_budget].rstrip() + " …"
        return text

    def _on_task(self, event: object) -> None:
        self._primed = False
        self._last_query_hash = None

    def _on_before_turn(self, event: object) -> None:
        sp = self.config.spontaneous
        if not self.config.enabled or not sp.enabled:
            return
        if sp.inject_cadence == "per_task" and self._primed:
            return
        queries = derive_queries(self.agent, sp)
        if not queries:
            return
        qhash = hash(tuple(queries))
        if sp.inject_cadence == "self_gated" and qhash == self._last_query_hash:
            return
        t0 = time.perf_counter()
        text = self._format_recall(queries)
        self.stats.injection_ms_total += (time.perf_counter() - t0) * 1000.0
        cm = self.agent.context_manager
        key = sp.context_block_key
        if text:
            cm.set_dynamic(key, value=text)
            self.stats.injections += 1
            self.stats.injected_chars += len(text)
            self._emit(
                MemoryInjected(
                    n_memories=text.count("\n- "),
                    chars=len(text),
                    memory_ids=list(self._last_injected_ids),
                )
            )
            mem_logger.debug("memory.inject %d chars (%d memories)", len(text), text.count("\n- "))
        elif key in cm:
            del cm[key]
        self._last_query_hash = qhash
        self._primed = True

    # ------------------------------------------------------------------
    # write-on-event
    # ------------------------------------------------------------------
    def _on_write_event(self, event: object) -> None:
        if not self.config.enabled:
            return
        text = ""
        for attr in ("content", "prompt", "text"):
            val = getattr(event, attr, None)
            if isinstance(val, str) and val.strip():
                text = val
                break
        if not text:
            return
        salience, importance = _EVENT_SALIENCE.get(type(event).__name__, (0.3, 5.0))
        if salience < self.config.write.salience_min:
            return
        try:
            self.remember(
                text,
                type=MemoryType.INFO,
                importance=importance,
                salience=salience,
                source_task_ref=type(event).__name__,
            )
        except Exception:
            logger.warning("memory: write-on-event failed", exc_info=True)

    # ------------------------------------------------------------------
    # post-task reflection
    # ------------------------------------------------------------------
    async def _reflect_middleware(self, ctx: Any, nxt: Any) -> Any:
        self._call_depth += 1
        try:
            ctx = await nxt(ctx)
        finally:
            self._call_depth -= 1

        if self._reflecting:
            return ctx
        is_top = self._call_depth == 0
        if self.config.reflection.only_top_level and not is_top:
            return ctx
        methods = self.config.reflection.entrypoint_methods
        if methods and getattr(ctx, "method_name", None) not in methods:
            return ctx

        if self.config.reflection.background:
            task = asyncio.create_task(self._async_reflect(ctx))
            self._pending.append(task)
            task.add_done_callback(
                lambda t: self._pending.remove(t) if t in self._pending else None
            )
        else:
            self._consolidate_after_task(ctx)
        return ctx

    async def _async_reflect(self, ctx: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._consolidate_after_task, ctx)

    def _consolidate_after_task(self, ctx: Any) -> None:
        if self._reflecting:
            return
        self._reflecting = True
        try:
            if self.config.write.write_episodic:
                self._write_episode(ctx)
            self.reflect(trigger="post_task")  # counts + emits + logs the consolidation
        except Exception:
            logger.warning("memory: reflection failed", exc_info=True)
        finally:
            self._reflecting = False

    def _write_episode(self, ctx: Any) -> None:
        from nooa.runtime.middleware import _AGENT_RESULT_NOT_SET

        method = getattr(ctx, "method_name", "") or "task"
        result = getattr(ctx, "result", _AGENT_RESULT_NOT_SET)
        result_str = "" if result is _AGENT_RESULT_NOT_SET else str(result)
        content = f"Episode: {method}\nResult: {result_str[:500]}"
        self.remember(
            content,
            type=MemoryType.EPISODE,
            importance=5.0,
            salience=0.5,
            title=f"episode:{method}",
            source_task_ref=method,
        )


# ---------------------------------------------------------------------------
# Conscious tools mixin
# ---------------------------------------------------------------------------
class MemoryToolsMixin:
    """Mixes the conscious memory tools into an Agent so they show in ``doc(self)``.

    Usage::

        class MyAgent(MemoryToolsMixin, Agent, llm=llm): ...
        MemoryManager.install(agent, config=MemoryConfig(enabled=True))
    """

    def _memory_or_raise(self) -> MemoryManager:
        mem = getattr(self, "_memory", None)
        if mem is None or not mem.config.enabled:
            raise RuntimeError(
                "Memory is not installed/enabled. Call MemoryManager.install(agent, "
                "config=MemoryConfig(enabled=True))."
            )
        return mem

    def _tool_enabled(self, name: str) -> MemoryManager:
        mem = self._memory_or_raise()
        if name not in mem.config.tools:
            raise RuntimeError(f"Memory tool {name!r} is disabled in MemoryConfig.tools.")
        return mem

    @staticmethod
    def _as_type(type: str) -> MemoryType:
        return MemoryType(type)  # raises ValueError on an unknown type (no silent fallback)

    def remember(
        self,
        content: str,
        *,
        type: str = "info",
        importance: str = "MEDIUM",
        tags: list[str] | None = None,
        title: str | None = None,
        references: list[str] | None = None,
    ) -> str:
        """Write a NEW long-term memory you author: {content}

        Record durable, reusable knowledge (NOT raw transcripts), one item per call.
        type: info | skill | episode | intent | todo | reflection.
        A todo is a durable commitment tracked until you close it with
        update_memory(id, status="DONE").
        importance: CRITICAL | HIGH | MEDIUM | LOW | TRIVIAL.
        tags: salient keywords/entities for later retrieval.
        references: pointers to live state this memory is ABOUT, as "kind:key" —
        var:<name> | context:<block> | file:<relative-path> | todo:<id> |
        memory:<id>. Referenced values are re-read fresh at recall time, so
        prefer a reference over pasting a value that may change.
        Returns the memory id.
        """
        mem = self._tool_enabled("remember")
        return mem.remember(
            content,
            type=self._as_type(type),
            importance=to_numeric("importance", importance),
            tags=tags,
            title=title,
            references=references,
        )

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        references: list[str] | None = None,
    ) -> bool:
        """Refine one of your existing memories {memory_id} (correct or sharpen it).

        Pass only the fields to change; {memory_id} may be the 8-char prefix shown
        in recalled lines. importance: CRITICAL | HIGH | MEDIUM | LOW | TRIVIAL.
        status (todo memories only): OPEN | DONE | DROPPED — close a todo with "DONE".
        references: replaces the memory's "kind:key" pointers when given.
        Returns True if the memory was found.
        """
        mem = self._tool_enabled("update_memory")
        return mem.update(
            memory_id,
            content=content,
            importance=to_numeric("importance", importance) if importance is not None else None,
            type=self._as_type(type) if type else None,
            tags=tags,
            status=status,
            references=references,
        )

    def forget(self, memory_id: str) -> bool:
        """Forget (archive) a memory {memory_id} you judge wrong or obsolete. Returns True if found."""
        mem = self._tool_enabled("forget")
        return mem.forget(memory_id)

    def recall(self, query: str, k: int = 5, owner: str | None = None) -> list[Memory]:
        """Recall memories associatively related to {query} (similarity + graph).

        owner: None = your own memories (+ shared); "*" = every agent sharing
        this store; "<name>" = that specific agent's (read-only).
        """
        mem = self._tool_enabled("recall")
        return mem.recall(query, k=k, owner=owner)

    def search(self, query: str, k: int = 5, owner: str | None = None) -> list[Memory]:
        """Term-focused recall for {query}: dense + keyword retrieval, graph spread disabled.

        owner: None = your own memories (+ shared); "*" = every agent sharing
        this store; "<name>" = that specific agent's (read-only).
        """
        mem = self._tool_enabled("search")
        # term-focused recall: 0 hops, keyword + dense, no graph spread
        return mem.recall(query, k=k, hops=0, owner=owner, channel="searched")

    def deref(self, ref: str) -> str:
        """Read the CURRENT value behind a reference {ref} ("kind:key").

        Kinds: var:<name> | context:<block> | file:<relative-path> | todo:<id>
        | memory:<id>. Returns the live value, or the write-time snapshot
        marked DANGLING when the target no longer resolves.
        """
        mem = self._tool_enabled("deref")
        return mem.deref(ref)

    def associate(self, a_id: str, b_id: str, relation: str = "related") -> None:
        """Link memory {a_id} to {b_id} with a directed {relation} edge.

        relation: related | supports | contradicts | refines | derived_from |
        created_by | causes | precedes | part_of | triggers (unknown -> error).
        Ids may be the 8-char prefixes shown in recalled memory lines.
        """
        mem = self._tool_enabled("associate")
        mem.associate(a_id, b_id, relation)
