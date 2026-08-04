# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle-reflection runner: consolidate memory while the user isn't looking.

Owns the lifecycle state machine from
``docs/design/memory-system/design-idle-reflection.md`` §3::

    IDLE --(response done)--> DEBOUNCE --> RUNNING --> IDLE
                                  |            |
                                  +--(input)---+--> interrupted --> IDLE

The runner lives on the agent thread's event loop: ``on_response_done()`` is
called by the dispatcher when a turn ends and the prompt returns to the user;
``interrupt()`` is awaited wherever new input is about to reach the agent.
The consolidation itself runs in an executor thread via
``MemoryManager.reflect_interruptible``, whose per-item commits make an
interrupted (or briefly overlapping) pass safe by construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:  # optional-import pattern mirrored from nooa_memory.tracing_bridge
    from opentelemetry import trace as _ot_trace
except ImportError:  # pragma: no cover - exercised only without the tracing extra
    _ot_trace = None

if TYPE_CHECKING:
    from nooa_memory.manager import MemoryManager
    from nooa_memory.reflection import ReflectionReport

    from .config import TUIConfig

logger = logging.getLogger(__name__)

# Indicator animation: a small wave gliding right, one step per repaint tick.
_WAVE = "▁▂▃▅▃▂▁"
_TICK_S = 0.15  # repaint cadence (and wave step) while a run is active
_FLASH_S = 0.3  # lifetime of the single "reflection interrupted" flash frame


class ReflectionRunner:
    """Debounced, promptly-interruptible idle reflection for one agent."""

    def __init__(
        self,
        agent: Any,
        manager: MemoryManager,
        config: TUIConfig,
        *,
        enabled: bool,
        episode_writer: Callable[[str], str | None] | None = None,
    ) -> None:
        self._agent = agent
        self._manager = manager
        self._config = config
        self.enabled = enabled
        # Generative phase 1: LLM writes an episode about the recent session
        # window before consolidation (which also feeds the reasoner).
        self._episode_writer = episode_writer
        self._self_writing = False  # our own episode write is not "dirt"
        self._stop = threading.Event()
        self._task: asyncio.Task | None = None
        self._state: str = "idle"  # idle | debounce | running
        self._flash_until = 0.0
        self.last_report: ReflectionReport | None = None
        # Dirty counter: memory mutations since the last consolidation pass.
        # Fed by the MemoryWritten subscription below (add/update/forget/
        # reinforce); ReflectionCompleted is deliberately NOT subscribed, so a
        # run never re-triggers itself. Seeded from store state: the runner is
        # REBUILT whenever memory is reconfigured (e.g. `/reflection on`), and
        # a fresh counter must not forget writes that predate it — that would
        # gate idle runs off exactly when the user just asked for them.
        self.dirty = self._initial_dirt()
        # Repaint hook, wired lazily by TUIApplication (thread-safe callable).
        self.invalidate: Callable[[], None] | None = None
        self._unsubscribe: Callable[[], None] | None = agent.event_manager.on(
            "MemoryWritten", self._on_memory_written
        )

    # ------------------------------------------------------------------
    # dirty signal
    # ------------------------------------------------------------------
    def _initial_dirt(self) -> int:
        """Memories touched since the last consolidation (pre-existing dirt)."""
        try:
            history = self._manager.store.maintenance_history(50)
            last = max((h["ts"] for h in history if h["kind"] == "reflect"), default=0.0)
            return sum(
                1
                for m in self._manager.store.all_memories(owner=self._manager.role)
                if max(m.created_at, m.last_accessed_at) > last
            )
        except Exception:  # a broken store must not take bootstrap down
            logger.warning("reflection: could not seed the dirty counter", exc_info=True)
            return 0

    def _on_memory_written(self, _event: object) -> None:
        if self._self_writing:
            return  # the runner's own episode write must not re-trigger a run
        self.dirty += 1

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    def _agent_idle(self) -> bool:
        """No queued user input, no running jobs, no pending channel work."""
        # Both lookups are host-optional surfaces: plain Agents (tests,
        # embedders) have neither queue; a missing queue means nothing pending.
        q = getattr(self._agent, "_user_messages_in", None)
        if q is not None and not q.is_empty():
            return False
        qm = getattr(self._agent, "queue_manager", None)
        if qm is None:
            return True
        if any(h.state == "running" for h in qm._handles):
            return False
        for name, ch in qm._channels.items():
            if name == "user_messages":
                continue
            if ch.mode == "queue" and not ch.is_empty():
                return False
        return True

    def on_response_done(self) -> None:
        """Schedule a debounced run. Called on the agent loop when a turn ends.

        Start gate (§3): enabled ∧ dirty ∧ agent-otherwise-idle, and no run
        already pending — otherwise this is a no-op.
        """
        if not self.enabled or self.dirty <= 0 or not self._agent_idle():
            return
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._state = "debounce"
        self._task = asyncio.ensure_future(self._debounce_then_run())

    def run_now(self) -> bool:
        """Start a run immediately (``/reflection now``): no debounce, no
        dirty gate, no enabled check — an explicit user request runs whatever
        is configured. Returns False when a run is already pending/active.
        Must be called on the agent loop (like ``on_response_done``)."""
        if self._task is not None and not self._task.done():
            return False
        self._stop.clear()
        self._state = "running"  # indicator shows immediately
        self._task = asyncio.ensure_future(self._run_to_idle())
        return True

    async def _debounce_then_run(self) -> None:
        try:
            await asyncio.sleep(self._config.reflection_debounce_s)
            # Re-check the gate: the world may have changed during debounce.
            if self._stop.is_set() or self.dirty <= 0 or not self._agent_idle():
                return
            await self._run()
        finally:
            self._state = "idle"
            self._invalidate()

    async def _run_to_idle(self) -> None:
        try:
            await self._run()
        finally:
            self._state = "idle"
            self._invalidate()

    async def _run(self) -> None:
        self._state = "running"
        # Snapshot semantics: writes landing while the run executes
        # re-increment the counter for the next idle window.
        self.dirty = 0
        # Event access happens on the agent loop; the executor only gets text.
        events_text = ""
        if self._episode_writer is not None:
            from nooa_memory.generative import render_recent_events

            events_text = render_recent_events(self._agent)
        ticker = asyncio.ensure_future(self._tick())
        try:
            self.last_report = await asyncio.get_running_loop().run_in_executor(
                None, self._reflect_in_thread, events_text
            )
        except Exception:
            # Surfaced loudly, but a failed consolidation must never take
            # the session down — the store is consistent per item.
            logger.exception("memory.reflection idle run failed")
        finally:
            ticker.cancel()

    def _reflect_in_thread(self, events_text: str = "") -> ReflectionReport:
        """The one sync function handed to the executor.

        Phase 1 (generative only): the LLM writes an episode about the recent
        session window — then consolidation runs, so the reasoner sees the
        fresh episode in the same pass. The span is opened *inside* the
        executor thread: OTel context does not flow into executors, so opening
        it on the loop would orphan the ``memory.reflected`` event the tracing
        bridge attaches to the current span. Degrades without opentelemetry.
        """
        if _ot_trace is None:
            self._write_episode(events_text)
            return self._manager.reflect_interruptible(self._stop.is_set, trigger="idle")
        db_path = self._manager.store.path
        attributes = {
            "memory.trigger": "idle",
            "memory.owner": self._manager.owner,
            "memory.db_path": str(Path(db_path).resolve()) if db_path != ":memory:" else db_path,
        }
        tracer = _ot_trace.get_tracer("nooa_cli.tui.reflection")
        with tracer.start_as_current_span("memory.reflection", attributes=attributes):
            self._write_episode(events_text)
            return self._manager.reflect_interruptible(self._stop.is_set, trigger="idle")

    def _write_episode(self, events_text: str) -> None:
        """Generative phase 1 — contained: a failed episode write never blocks
        consolidation, and the stop flag is honored before the LLM call."""
        if self._episode_writer is None or self._stop.is_set() or not events_text:
            return
        try:
            episode = self._episode_writer(events_text)
        except Exception:
            logger.warning("memory.reflection episode write failed", exc_info=True)
            return
        if episode is None or self._stop.is_set():
            return  # not noteworthy, or the stop arrived while the LLM ran
        from nooa_memory import MemoryType

        self._self_writing = True
        try:
            self._manager.remember(
                episode,
                type=MemoryType.EPISODE,
                title="session episode",
                salience=0.5,
                source_task_ref="idle_reflection",
            )
        finally:
            self._self_writing = False

    async def interrupt(self, grace: float | None = None) -> None:
        """Stop any pending or active run; return within *grace* regardless.

        A pending debounce is cancelled silently (the indicator was never
        shown). A running pass gets the stop flag — observed between items —
        and up to *grace* seconds to wind down; after that the caller proceeds
        anyway: per-item commits make a short overlap safe (design §5).
        """
        grace = self._config.reflection_grace_s if grace is None else grace
        task = self._task
        if task is None or task.done():
            return
        if self._state == "debounce":
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            # A task cancelled before its first step never runs its finally —
            # reset here so the state can't stick in "debounce".
            self._task = None
            self._state = "idle"
            return
        self._stop.set()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), grace)
        if not task.done() or (self.last_report is not None and self.last_report.interrupted):
            self._flash_until = time.monotonic() + _FLASH_S

    def teardown(self) -> None:
        """Unsubscribe and cancel without waiting (off / reconfigure / detach)."""
        self._stop.set()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        self._task = None
        self._state = "idle"
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # ------------------------------------------------------------------
    # indicator
    # ------------------------------------------------------------------
    def indicator_frame(self) -> str:
        """Status-bar segment: a gliding wave while RUNNING, one 'interrupted'
        flash frame right after a cancel, empty (zero width) otherwise."""
        if self._state == "running" and not self._stop.is_set():
            shift = int(time.monotonic() / _TICK_S) % len(_WAVE)
            return f"✦ reflecting {_WAVE[shift:]}{_WAVE[:shift]}"
        if time.monotonic() < self._flash_until:
            return "✦ reflection interrupted"
        return ""

    async def _tick(self) -> None:
        while True:
            self._invalidate()
            await asyncio.sleep(_TICK_S)

    def _invalidate(self) -> None:
        if self.invalidate is not None:
            self.invalidate()
