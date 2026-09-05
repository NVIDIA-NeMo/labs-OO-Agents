# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF v1.7 exporter — event-driven state machine.

Consumes the framework's own event stream (Task, BeforeTurn,
SystemPrompt, LLMComplete, LLMOutput, ToolCallEvent, PythonOutput,
AfterTurn, etc.) and assembles an in-memory :class:`Trajectory`
Pydantic model that is serialized to disk atomically. Handles
single-trajectory runs as well as nesting, concurrency, compaction,
and multimodal input.
"""

from __future__ import annotations

import base64
import binascii
import contextvars
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from uuid import uuid4

from nooa.atif.schema import (
    AgentSchema,
    ContentPart,
    FinalMetricsSchema,
    ImageSource,
    MetricsSchema,
    ObservationResultSchema,
    ObservationSchema,
    StepObject,
    SubagentTrajectoryRef,
    ToolCallSchema,
    Trajectory,
)
from nooa.context_blocks.roles import Role
from nooa.standalone import _atif_exporter_var

if TYPE_CHECKING:
    from nooa.context_blocks.events import EventBase, ToolCallEvent
    from nooa.events import (
        AfterTurn,
        BeforeTurn,
        Error,
        LLMComplete,
        LLMOutput,
        Notification,
        PythonOutput,
        Reasoning,
        Summary,
        SystemPrompt,
        Task,
    )
    from nooa.runtime.event_manager import EventManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-turn pending state
# ---------------------------------------------------------------------------


class _PendingStep:
    """Mutable accumulator for one in-flight ATIF agent step.

    One instance per ``generation_id`` between its ``BeforeTurn`` and
    ``AfterTurn``. Sealed to a ``StepObject`` at ``AfterTurn`` time.
    """

    __slots__ = (
        "extra",
        "generation_id",
        "llm_call_count",
        "message",
        "method_name",
        "metrics",
        "model_name",
        "parent_generation_id",
        "reasoning_content",
        "started_at",
        "strategy_name",
        "tool_calls",
        "turn_number",
    )

    def __init__(
        self,
        *,
        generation_id: str,
        parent_generation_id: str | None,
        method_name: str,
        strategy_name: str,
        turn_number: int,
    ) -> None:
        self.generation_id = generation_id
        self.parent_generation_id = parent_generation_id
        self.method_name = method_name
        self.strategy_name = strategy_name
        self.turn_number = turn_number
        self.started_at = datetime.now(UTC)
        self.model_name: str | None = None
        self.metrics: MetricsSchema | None = None
        self.tool_calls: list[ToolCallSchema] = []
        self.reasoning_content: str | None = None
        self.message: str = ""
        self.extra: dict[str, Any] = {}
        self.llm_call_count: int = 0  # bumped by LLMComplete


class _DispatchStep(NamedTuple):
    """A buffered delegation step: a parent's reference to an embedded
    sub-trajectory (standalone or sub-agent), flushed once the system step
    exists. ``extra`` carries per-handoff metadata (e.g. step-range offset)."""

    name: str
    trajectory_id: str
    session_id: str | None
    timestamp: str
    extra: dict[str, Any] | None = None


# Dispatch table: event_type string → AtifExporter handler-method name.
#
# The exporter does NOT subscribe to each of these individually. Instead it
# subscribes once via ``event_manager.on("*", self._dispatch_event)`` and
# routes by ``event_type`` here. This way custom EventBase subclasses
# defined outside the framework (or added in future releases) flow into
# the trajectory via the role-based fallback in ``_dispatch_event``.
_HANDLER_DISPATCH: dict[str, str] = {
    "Task": "on_task",
    "BeforeAgentCall": "on_before_agent_call",
    "AfterAgentCall": "on_after_agent_call",
    "BeforeTurn": "on_before_turn",
    "SystemPrompt": "on_system_prompt",
    "LLMComplete": "on_llm_complete",
    "LLMOutput": "on_llm_output",
    "Reasoning": "on_reasoning",
    "ToolCallEvent": "on_tool_call_event",
    "PythonOutput": "on_python_output",
    "AfterTurn": "on_after_turn",
    "Error": "on_error",
    "Notification": "on_notification",
    "Summary": "on_summary",
}


_DATA_URL_RE = re.compile(r"^data:(?P<media_type>image/[\w+-]+);base64,(?P<data>.+)$")
_MEDIA_TYPE_TO_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso8601_utc(dt: datetime | None = None) -> str:
    """ISO 8601 UTC timestamp (with millisecond precision and Z suffix).

    ``dt`` is normalised to UTC first: the ``Z`` suffix asserts UTC, so
    formatting a local-time value verbatim would mislabel it.  Event
    timestamps arrive naive (local), while the default is already aware —
    without the conversion a single trajectory mixes both, leaving steps
    out of chronological order by the local UTC offset.  A naive value is
    interpreted as local time, which is what the event system produces.
    """
    dt = datetime.now(UTC) if dt is None else dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _parse_arguments(args: Any) -> dict[str, Any]:
    """Coerce a tool_call's arguments field to a dict (it may arrive as JSON string)."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
        return parsed if isinstance(parsed, dict) else {"_raw": args}
    return {}


def _format_python_output_content(po: PythonOutput) -> str:
    """Render a PythonOutput event as an observation content string.

    Mirrors the spec example shape: stdout / stderr / returned_value
    concatenated with section headers.  Empty stdout/stderr/value
    yields an empty string — caller skips attaching an observation
    in that case.
    """
    parts: list[str] = []
    if po.stdout:
        parts.append(f"stdout:\n{po.stdout}")
    if po.stderr:
        parts.append(f"stderr:\n{po.stderr}")
    if po.error:
        parts.append(f"error:\n{po.error}")
    if po.value is not None and po.value != "None":
        parts.append(f"returned_value:\n{po.value}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class AtifExporter:
    """Event-driven ATIF v1.7 trajectory exporter.

    Activated via :func:`nooa.atif.install.install_atif` or
    :func:`nooa.atif.install.atif_scope`. Tests may also drive
    the handlers directly without going through EventManager.

    State machine:
      * Trajectory root is initialised in ``__init__``.
      * ``BeforeTurn`` opens a ``_PendingStep`` keyed by
        ``generation_id``.
      * ``LLMComplete`` fills metrics, model_name, tool_calls,
        reasoning_content on the matching pending step.
      * ``LLMOutput`` sets the pending step's assistant message text.
      * ``ToolCallEvent`` (creation) registers the tool_call_id ⇆
        Python event-reference mapping; tool_calls[i] entries are
        already in place from ``LLMComplete``.
      * ``PythonOutput`` registers observation content for an
        ``execute_python`` call (the dominant case).
      * ``AfterTurn`` closes the pending step: builds
        ``observation.results[]`` (PythonOutput-first, then
        ToolCallEvent.result.content as fallback), assigns
        ``step_id``, appends to ``steps``, atomic-writes.

    Crash-safety: every ``AfterTurn`` writes the current trajectory
    atomically.  When the agent_call terminates (top-level
    ``AfterTurn(is_final=True, parent_generation_id is None)``), the
    final ``final_metrics`` are computed and a final write happens.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        session_id: str | None,
        agent_name: str,
        agent_version: str,
        agent_model_name: str | None = None,
        agent_tool_definitions: list[dict[str, Any]] | None = None,
        agent_extra: dict[str, Any] | None = None,
        trajectory_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._finalized = False

        self._trajectory = Trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id or str(uuid4()),
            agent=AgentSchema(
                name=agent_name,
                version=agent_version,
                model_name=agent_model_name,
                tool_definitions=agent_tool_definitions,
                extra=agent_extra,
            ),
            steps=[],
        )

        # Per-turn state, keyed by generation_id.
        self._pending: dict[str, _PendingStep] = {}

        # Tool-call indexes (lifetime = trajectory; older ids are kept
        # in case an event arrives late). Memory is bounded by per-trial
        # tool_call count, which is small for any real trajectory.
        self._python_outputs: dict[str, PythonOutput] = {}
        self._tool_call_events: dict[str, ToolCallEvent] = {}

        # Auto-incrementing step_id assigned when a pending step closes.
        self._next_step_id = 1

        # Tracks active tool_call by generation_id: the most recently
        # opened ToolCallEvent that has not yet produced a PythonOutput.
        # Used to find the "enclosing tool_call" for Case C subagent
        # attachment (so the subagent_trajectory_ref lands on the right
        # observation result).
        self._enclosing_tool_call_id: str | None = None

        # Case C state: child EventManager → (child sub-exporter, list of unsubscribers).
        # Child sub-exporters share this exporter's trajectory file (no disk
        # writes of their own — they accumulate in memory and on _detach_child
        # we lift their trajectory into self._trajectory.subagent_trajectories).
        self._children: dict[int, tuple[AtifExporter, list[Callable[[], None]], str | None]] = {}

        # Pending subagent_trajectory_ref entries keyed by enclosing
        # parent tool_call_id. Populated by _detach_child; consumed by
        # _build_step_from_pending when the enclosing tool_call's step
        # is finalized.
        self._pending_subagent_refs: dict[str, list[SubagentTrajectoryRef]] = {}

        # Whether this exporter writes to disk. Subagent sub-exporters
        # share the parent's file (via the parent's _write), so they set
        # this to False to avoid double writes.
        self._writes_to_disk = True

        # Cascade-binding token. At most one is set at a time; they differ
        # only in who releases it: install (atif_scope, until uninstall),
        # entrypoint (per top-level agent call), or run (per generation turn,
        # a fallback for turns not preceded by an agent-call event).
        self._install_token: contextvars.Token | None = None
        self._run_token: contextvars.Token | None = None
        self._entrypoint_token: contextvars.Token | None = None

        # System-prompt state. The framework fires SystemPrompt AFTER the
        # Task event (see runtime/actor.py — render-at-call-time invariant),
        # but ATIF wants source="system" at step_id=1 ahead of the user task.
        # Solution: buffer Task events arriving before the first SystemPrompt;
        # flush them after the system step is emitted.
        self._system_step_emitted: bool = False
        self._system_content_hash: int | None = None
        self._buffered_tasks: list[Task] = []
        # Sub-agent / standalone handoff dispatch steps that occurred before the
        # first SystemPrompt (a pure-Python orchestrator that delegated before
        # any generation turn). Buffered like Tasks and flushed after the system
        # step, ahead of the task, to preserve chronology. Each entry is a
        # ``_DispatchStep`` (name, trajectory_id, session_id, timestamp, extra).
        self._buffered_dispatches: list[_DispatchStep] = []
        # Per-run agent-call state (only one outer call binds an exporter at a
        # time — same-agent nested calls return early). Set by
        # on_before_agent_call, consumed by the matching on_after_agent_call:
        # the call_id that bound us, the parent exporter to embed into (or None
        # if top-level), and this trajectory's step count at call start (for
        # the handoff step-range offset).
        self._arm_call_id: str | None = None
        self._adoptive_parent: Any = None
        self._embed_start_step: int = 0
        self._saved_writes_to_disk: bool = True
        # Drift: when a later SystemPrompt differs from the first, stash the
        # new content for the next agent step's extra. Cleared on emit.
        self._pending_system_drift: str | None = None

    # =========================================================== handlers

    def _dispatch_event(self, event: EventBase) -> None:
        """Route every event off the EventManager's wildcard subscription.

        Specific event types listed in :data:`_HANDLER_DISPATCH` route to
        their dedicated handlers (``on_task``, ``on_llm_complete``, etc.).

        Custom :class:`EventBase` subclasses defined outside the framework
        (or added in future releases) fall through to
        :meth:`_record_generic_event`, which decides whether they belong
        in the trajectory based on their ``_role``:

        - ``Role.RUNTIME_EVENT`` / ``Role.METADATA`` — internal markers
          the framework explicitly hides from the LLM; skip them in ATIF
          for consistency.
        - ``Role.USER`` / ``Role.ASSISTANT`` / ``Role.TOOL`` — events
          the LLM saw; render as a generic step.

        Without this dispatcher, a hardcoded allow-list of event types
        would silently drop any new event a downstream user wires into
        their EventManager.
        """
        handler_name = _HANDLER_DISPATCH.get(event.event_type)
        if handler_name is not None:
            getattr(self, handler_name)(event)
            return
        self._record_generic_event(event)

    def _record_generic_event(self, event: EventBase) -> None:
        """Emit a generic step for a custom EventBase subclass.

        Internal events (``Role.RUNTIME_EVENT`` / ``Role.METADATA``) are
        skipped — they are not part of the LLM-facing conversation, so
        they have no place in the trajectory.

        Role → ATIF source:
          - USER → ``"user"``
          - ASSISTANT → ``"agent"`` with ``llm_call_count=0``
            (deterministic-dispatch per ATIF v1.7 §II, since the event
            was emitted without an enclosing LLM call)
          - TOOL → ``"user"`` (closest spec analog for in-band tool
            results; ATIF doesn't have a TOOL source)

        ``message`` is the framework's own ``pformat`` rendering (the
        same string the LLM would see); ``extra.event_type`` and
        ``extra.event_role`` preserve the original event identity so
        downstream consumers can reconstruct the source.
        """
        role = getattr(event, "_role", None)
        if role in (Role.RUNTIME_EVENT, Role.METADATA, None):
            return  # internal — not LLM-visible

        from nooa.agentdoc import pformat as _pformat

        try:
            message = _pformat(event)
        except Exception:  # noqa: BLE001
            # pformat customisation in user code shouldn't break tracing.
            message = type(event).__name__

        if role == Role.ASSISTANT:
            source = "agent"
            extra_step_fields: dict[str, Any] = {"llm_call_count": 0}
        else:  # USER, TOOL, or anything unexpected — render as user step
            source = "user"
            extra_step_fields = {}

        with self._lock:
            self._append_step(
                StepObject(
                    step_id=self._next_step_id,
                    timestamp=_iso8601_utc(getattr(event, "timestamp", None)),
                    source=source,
                    message=message,
                    extra={
                        "event_type": event.event_type,
                        "event_role": getattr(role, "value", str(role)),
                    },
                    **extra_step_fields,
                )
            )
            self._next_step_id += 1
            self._write()

    def on_task(self, event: Task) -> None:
        """``Task`` ⇒ user step (the initial prompt).

        If ``Task.images`` is non-empty, the message is rendered as a
        ``ContentPart[]`` array combining text + image references, and
        the image bytes are written to ``images/`` next to the
        trajectory file (Case "Multimodal").

        When :class:`SystemPrompt` hasn't been seen yet (the common
        case — Task fires before the first LLM call, SystemPrompt
        fires right after ``_build_messages``), Task is BUFFERED. The
        matching :meth:`on_system_prompt` flushes the buffer once the
        system step is emitted at ``step_id=1``. If no SystemPrompt
        ever arrives (synthetic-event tests, agents that skip LLM
        calls), :meth:`close` flushes the buffer as user steps with no
        system step prepended.
        """
        with self._lock:
            if not self._system_step_emitted:
                self._buffered_tasks.append(event)
                return
            self._emit_task_step(event)

    def _emit_task_step(self, event: Task) -> None:
        """Render and append one buffered/incoming Task as a user step."""
        message = self._build_message_with_images(event.prompt, event.images)
        self._append_step(
            StepObject(
                step_id=self._next_step_id,
                timestamp=_iso8601_utc(event.timestamp),
                source="user",
                message=message,
            )
        )
        self._next_step_id += 1
        self._write()

    def on_system_prompt(self, event: SystemPrompt) -> None:
        """``SystemPrompt`` ⇒ emit ``source: "system"`` at ``step_id=1``.

        Fires whenever the runtime renders messages for an LLM call.
        Behaviour:

        - **First occurrence**: emit a system step at ``step_id=1``;
          mark ``_system_step_emitted``; flush any buffered Tasks (they
          become ``step_id=2..N``); record the content hash for drift.
        - **Subsequent occurrences with identical content**: no-op
          (the LLM saw the same system prompt — no information added).
        - **Subsequent occurrences with different content**: stash the
          new content in ``_pending_system_drift``;
          :meth:`on_llm_complete` will annotate the next agent step
          with ``extra.system_prompt_changed = True`` and
          ``extra.system_prompt = <new content>``.

        Static blocks are supposed to be stable across turns
        (``metadata.static = True``), so drift is the exception — a
        signal that something mutated the static system content
        mid-run.
        """
        with self._lock:
            content = event.content or ""
            if not self._system_step_emitted:
                self._append_step(
                    StepObject(
                        step_id=self._next_step_id,
                        timestamp=_iso8601_utc(event.timestamp),
                        source="system",
                        message=content,
                    )
                )
                self._next_step_id += 1
                self._system_step_emitted = True
                self._system_content_hash = hash(content)
                # Flush delegation dispatch steps that ran before this first
                # turn — chronologically they precede the task below.
                for dispatch in self._buffered_dispatches:
                    self._emit_dispatch_step(dispatch)
                self._buffered_dispatches.clear()
                # Flush any tasks that arrived before the system step.
                for buffered in self._buffered_tasks:
                    self._emit_task_step(buffered)
                self._buffered_tasks.clear()
                self._write()
                return
            if hash(content) != self._system_content_hash:
                self._pending_system_drift = content

    def on_before_agent_call(self, event: Any) -> None:
        """``BeforeAgentCall`` ⇒ bind the cascade var to this exporter.

        The event fires on the agent's own EventManager, so each exporter
        binds with itself. Three cases by what the var currently holds:

        - ``self`` — a method of this agent is already running (same-agent
          nested call, or atif_scope's install binding); no-op.
        - ``None`` — top-level run: bind self, write own file as usual.
        - another exporter ``parent`` — this agent runs nested under
          ``parent``: bind self (so this agent's own children nest under it)
          and mark for embedding into ``parent`` (suppress own file). The
          handoff is recorded on the matching :meth:`on_after_agent_call`.
        """
        with self._lock:
            cur = _atif_exporter_var.get()
            if cur is self:
                return
            try:
                self._entrypoint_token = _atif_exporter_var.set(self)
            except Exception:  # noqa: BLE001
                self._entrypoint_token = None
            self._arm_call_id = getattr(event, "call_id", None) or ""
            self._adoptive_parent = cur
            self._embed_start_step = len(self._trajectory.steps)
            self._saved_writes_to_disk = self._writes_to_disk
            if cur is not None:
                # Nested under a parent run ⇒ embed there, not a separate file.
                # Restored on the matching on_after_agent_call so suppression is
                # scoped to this call (a reused instance may later run top-level).
                self._writes_to_disk = False

        # Note: concurrent calls into the *same* agent instance (e.g.
        # ``asyncio.gather(agent.m(a), agent.m(b))``) are not supported — a
        # single instance shares one event history/pending state. Parallel work
        # should use distinct sub-agent instances.

    def on_after_agent_call(self, event: Any) -> None:
        """``AfterAgentCall`` ⇒ release the binding (success or exception).

        For a nested run, lift this agent's trajectory into the parent's
        ``subagent_trajectories[]`` and emit a handoff reference step carrying
        the step-range this call produced. For a top-level run, finalize the
        trajectory if a generation turn never did (pure-Python orchestrator).
        """
        with self._lock:
            cid = getattr(event, "call_id", None) or ""
            if cid != self._arm_call_id:
                return  # not the call that bound us (same-agent nested / unmatched)
            if self._entrypoint_token is not None:
                try:
                    _atif_exporter_var.reset(self._entrypoint_token)
                except (LookupError, ValueError):
                    pass
                self._entrypoint_token = None
            parent = self._adoptive_parent
            start_step = self._embed_start_step
            self._arm_call_id = None
            self._adoptive_parent = None
            self._writes_to_disk = self._saved_writes_to_disk
            if parent is not None and hasattr(parent, "_embed_subagent_handoff"):
                parent._embed_subagent_handoff(self, start_step, len(self._trajectory.steps))
            else:
                # Top-level: finalize a pure-Python orchestrator that never
                # opened a generation turn (no on_after_turn finalize fired).
                self._finalize_if_needed(success=getattr(event, "success", True))

    def on_before_turn(self, event: BeforeTurn) -> None:
        """``BeforeTurn`` ⇒ open a pending agent step keyed by generation_id.

        For Case B (same-agent nested generation,
        ``parent_generation_id is not None``), the resulting step is
        emitted into the SAME trajectory (B-flatten policy). The pending
        step's ``extra`` carries both ``generation_id`` and
        ``parent_generation_id`` so consumers can reconstruct the
        nesting tree.

        At a top-level turn boundary it also arms the standalone cascade
        (``_run_token``) — a fallback for turns not preceded by a
        ``BeforeAgentCall`` event, skipped when any binding already exists.
        """
        with self._lock:
            self._pending[event.generation_id] = _PendingStep(
                generation_id=event.generation_id,
                parent_generation_id=event.parent_generation_id,
                method_name=event.method_name,
                strategy_name=event.strategy,
                turn_number=event.turn_number,
            )
            if (
                event.turn_number == 1
                and event.parent_generation_id is None
                and self._install_token is None
                and self._run_token is None
                and self._entrypoint_token is None
            ):
                self._run_token = _atif_exporter_var.set(self)

    def on_llm_complete(self, event: LLMComplete) -> None:
        """``LLMComplete`` ⇒ fill metrics / tool_calls / model_name / reasoning."""
        with self._lock:
            ps = self._pending.get(event.generation_id)
            if ps is None:
                logger.debug(
                    "atif: LLMComplete for unknown generation_id=%s (ignored)",
                    event.generation_id,
                )
                return
            ps.model_name = event.model_name or ps.model_name
            ps.metrics = MetricsSchema(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
                extra={"reasoning_tokens": event.reasoning_tokens}
                if event.reasoning_tokens
                else None,
            )
            ps.tool_calls = [
                ToolCallSchema(
                    tool_call_id=tc["tool_call_id"],
                    function_name=tc["function_name"],
                    arguments=_parse_arguments(tc.get("arguments")),
                )
                for tc in event.tool_calls
            ]
            if event.reasoning_content:
                # Accumulate alongside any Reasoning events that fired pre-LLM.
                if ps.reasoning_content:
                    ps.reasoning_content += "\n" + event.reasoning_content
                else:
                    ps.reasoning_content = event.reasoning_content
            ps.llm_call_count += 1
            # Per-turn dynamic context envelope (re-rendered every LLM call
            # from current agent state — see runtime/actor.py
            # _extract_trailing_context_envelope). Stashed on the agent
            # step's extra so SFT pipelines can reconstruct the exact bytes
            # the LLM saw at this turn without storing the full per-turn
            # messages list.
            if event.dynamic_context:
                ps.extra["dynamic_context"] = event.dynamic_context
            # System-prompt drift: a later SystemPrompt event saw different
            # static content. Static blocks are supposed to be stable
            # (metadata.static=True), so drift is a signal that something
            # mutated them mid-run. Annotate the agent step that consumed
            # the new content.
            if self._pending_system_drift is not None:
                ps.extra["system_prompt_changed"] = True
                ps.extra["system_prompt"] = self._pending_system_drift
                self._system_content_hash = hash(self._pending_system_drift)
                self._pending_system_drift = None

    def on_llm_output(self, event: LLMOutput) -> None:
        """``LLMOutput`` ⇒ set the assistant message text on the current pending step.

        We use the most-recently-opened pending step. If multiple are
        active (concurrent generation), keyed dispatch handles it; this
        fallback covers the simple linear case.
        """
        with self._lock:
            ps = self._most_recent_pending()
            if ps is None:
                logger.debug("atif: LLMOutput with no pending step (ignored)")
                return
            ps.message = event.content

    def on_tool_call_event(self, event: ToolCallEvent) -> None:
        """``ToolCallEvent`` ⇒ index by tool_call_id for later result lookup.

        Also marks this tool_call_id as the *currently enclosing* call,
        so any standalone-generation subagent spawned inside its
        execution (Case C) attaches its trajectory ref here.

        If the event represents a tool_call that the LLM did NOT emit
        (e.g. CodeAct's synthetic ``return_result`` for the
        inline-completion path — see ``codeact.py
        _emit_synthetic_inline_return``), append it to the current
        pending step's ``tool_calls``. This makes framework-emitted
        completion markers visible in the trajectory even though
        ``LLMComplete.tool_calls`` did not include them.
        """
        with self._lock:
            self._tool_call_events[event.tool_call_id] = event
            self._enclosing_tool_call_id = event.tool_call_id

            ps = self._most_recent_pending()
            if ps is None:
                return
            existing = {tc.tool_call_id for tc in ps.tool_calls}
            if event.tool_call_id not in existing:
                ps.tool_calls.append(
                    ToolCallSchema(
                        tool_call_id=event.tool_call_id,
                        function_name=event.name,
                        arguments=event.arguments,
                        extra=(
                            {
                                "synthetic": True,
                                "synthetic_type": event.metadata.get("synthetic_type"),
                            }
                            if event.metadata.get("synthetic")
                            else None
                        ),
                    )
                )

    def on_python_output(self, event: PythonOutput) -> None:
        """``PythonOutput`` ⇒ register an observation source for this tool_call_id.

        The corresponding tool_call has now completed, so clear the
        enclosing-tool-call tracker if it matches.
        """
        with self._lock:
            self._python_outputs[event.tool_call_id] = event
            if self._enclosing_tool_call_id == event.tool_call_id:
                self._enclosing_tool_call_id = None

    def on_reasoning(self, event: Reasoning) -> None:
        """``Reasoning`` ⇒ accumulate on the current pending step's reasoning_content."""
        with self._lock:
            ps = self._most_recent_pending()
            if ps is None:
                return
            if ps.reasoning_content:
                ps.reasoning_content += "\n" + event.content
            else:
                ps.reasoning_content = event.content

    def on_after_turn(self, event: AfterTurn) -> None:
        """``AfterTurn`` ⇒ finalize the pending step, attach observation, append, write."""
        with self._lock:
            ps = self._pending.pop(event.generation_id, None)
            if ps is None:
                logger.debug(
                    "atif: AfterTurn for unknown generation_id=%s (ignored)",
                    event.generation_id,
                )
                return

            step = self._build_step_from_pending(ps)
            self._append_step(step)
            self._next_step_id += 1

            # Top-level final turn ⇒ finalize the whole trajectory.
            if event.is_final and event.parent_generation_id is None:
                self._finalize_trajectory(success=event.success)
                # Release the run-scoped contextvar binding (paired with
                # the push in :meth:`on_before_turn`). The install-time
                # binding, if any, is left in place — atif_scope owns
                # that and resets it at uninstall.
                if self._run_token is not None:
                    try:
                        _atif_exporter_var.reset(self._run_token)
                    except (LookupError, ValueError):
                        # Token-out-of-context (a child task rebound the var);
                        # best-effort cleanup.
                        pass
                    self._run_token = None

            self._write()

    def on_error(self, event: Error) -> None:
        """``Error`` ⇒ next-turn feedback rendered as a user step."""
        with self._lock:
            self._append_step(
                StepObject(
                    step_id=self._next_step_id,
                    timestamp=_iso8601_utc(event.timestamp),
                    source="user",
                    message=event.content,
                    extra={"event_kind": "error"},
                )
            )
            self._next_step_id += 1
            self._write()

    def on_notification(self, event: Notification) -> None:
        """``Notification`` ⇒ external user-facing signal."""
        with self._lock:
            self._append_step(
                StepObject(
                    step_id=self._next_step_id,
                    timestamp=_iso8601_utc(event.timestamp),
                    source="user",
                    message=event.description,
                    extra={"event_kind": "notification", "source": event.source},
                )
            )
            self._next_step_id += 1
            self._write()

    def on_summary(self, event: Summary) -> None:
        """``Summary`` ⇒ compaction system step (spec §VII).

        Marks every prior step as ``is_copied_context=True`` and emits
        a system step with ``extra.context_management.boundary="replace"``.
        """
        with self._lock:
            # Mark prior steps as copied across the boundary.
            for prior in self._trajectory.steps:
                if prior.is_copied_context is None:
                    prior.is_copied_context = True
            self._append_step(
                StepObject(
                    step_id=self._next_step_id,
                    timestamp=_iso8601_utc(event.timestamp),
                    source="system",
                    message="Context compaction performed",
                    observation=ObservationSchema(
                        results=[ObservationResultSchema(content=event.summary_text or "")]
                    ),
                    extra={
                        "context_management": {
                            "type": "compaction",
                            "boundary": "replace",
                        },
                        "summary_tag": event.summary_tag,
                        "replaced_range": list(event.replaced_range),
                    },
                )
            )
            self._next_step_id += 1
            self._write()

    # ============================================================ lifecycle

    def finalize_on_exception(self, exc: BaseException) -> None:
        """Mark the trajectory as crashed and atomically write.

        Called from the activation layer's exception path, not directly
        from any handler.  Also releases the run-scoped
        ContextVar binding (paired with the push in
        :meth:`on_before_turn`) — without this, a crashed run would
        leak its exporter onto the ContextVar for the remainder of
        the async context.

        If the crash happens before the runtime ever fired a
        :class:`SystemPrompt` (e.g. before the first ``_build_messages``
        call), any buffered :class:`Task` events are flushed as user
        steps so the on-disk trajectory still records the prompt.
        """
        with self._lock:
            # Flush orphan-buffered dispatch steps + Tasks (crashed or finished
            # before the first LLM call).
            if self._buffered_dispatches or self._buffered_tasks:
                self._system_step_emitted = True
                for dispatch in self._buffered_dispatches:
                    self._emit_dispatch_step(dispatch)
                self._buffered_dispatches.clear()
                for buffered in self._buffered_tasks:
                    self._emit_task_step(buffered)
                self._buffered_tasks.clear()
            extra = dict(self._trajectory.extra or {})
            extra["crashed"] = True
            extra["exception_type"] = type(exc).__name__
            extra["exception_message"] = str(exc)[:500]
            self._trajectory = self._trajectory.model_copy(update={"extra": extra})
            self._write()
            if self._run_token is not None:
                try:
                    _atif_exporter_var.reset(self._run_token)
                except (LookupError, ValueError):
                    pass
                self._run_token = None
            if self._entrypoint_token is not None:
                try:
                    _atif_exporter_var.reset(self._entrypoint_token)
                except (LookupError, ValueError):
                    pass
                self._entrypoint_token = None

    def get_trajectory(self) -> Trajectory:
        """Return a snapshot of the current trajectory (for tests)."""
        with self._lock:
            return self._trajectory.model_copy(deep=True)

    def close(self) -> None:
        """Release any held ContextVar bindings and flush pending state.

        Idempotent. Called by :func:`install_atif`'s ``uninstall()`` and
        by test fixtures that drive the exporter directly without going
        through ``install_atif``. Without this, a test (or a real
        crashed run) that fired ``on_before_turn`` but no matching
        ``on_after_turn`` would leak the run-scoped token onto the
        ContextVar for the rest of the async context.

        Also flushes any Tasks buffered for a SystemPrompt that never
        arrived (synthetic-event tests, agents that bypass the LLM).
        They're emitted as user steps with no system step prepended —
        graceful degradation when the framework didn't fire the
        expected SystemPrompt.
        """
        with self._lock:
            if self._run_token is not None:
                try:
                    _atif_exporter_var.reset(self._run_token)
                except (LookupError, ValueError):
                    pass
                self._run_token = None
            if self._install_token is not None:
                try:
                    _atif_exporter_var.reset(self._install_token)
                except (LookupError, ValueError):
                    pass
                self._install_token = None
            if self._entrypoint_token is not None:
                try:
                    _atif_exporter_var.reset(self._entrypoint_token)
                except (LookupError, ValueError):
                    pass
                self._entrypoint_token = None
            # Flush orphan-buffered dispatch steps + Tasks (no SystemPrompt
            # arrived).
            if self._buffered_dispatches or self._buffered_tasks:
                # Mark "no system step" so subsequent events emit directly
                # instead of re-buffering.
                self._system_step_emitted = True
                for dispatch in self._buffered_dispatches:
                    self._emit_dispatch_step(dispatch)
                self._buffered_dispatches.clear()
                for buffered in self._buffered_tasks:
                    self._emit_task_step(buffered)
                self._buffered_tasks.clear()

    # ============================================================ subagent

    def _attach_child(self, event_manager: EventManager, child_agent_name: str = "") -> None:
        """Case C: attach this exporter to a child EventManager.

        Creates an in-memory child :class:`AtifExporter` that subscribes
        to all the same events on the child EM. On
        :meth:`_detach_child` the child's trajectory is lifted into
        ``self._trajectory.subagent_trajectories[]`` and a
        :class:`SubagentTrajectoryRef` is queued for the parent's
        currently-enclosing tool_call.
        """
        with self._lock:
            key = id(event_manager)
            if key in self._children:
                return  # idempotent
            child = AtifExporter._make_subagent(
                agent_name=child_agent_name or "standalone",
                agent_version=self._trajectory.agent.version,
                session_id=self._trajectory.session_id,
                trajectory_id=str(uuid4()),
            )
            # Subscribe once via the wildcard so custom user events on the
            # child EventManager also flow into the child trajectory.
            unsubscribers: list[Callable[[], None]] = [
                event_manager.on("*", child._dispatch_event),
            ]
            # Record the enclosing parent tool_call_id at attach time so
            # the eventual ref attaches to the right observation.
            enclosing = self._enclosing_tool_call_id
            self._children[key] = (child, unsubscribers, enclosing)

    def _detach_child(self, event_manager: EventManager) -> None:
        """Case C: detach + lift the child trajectory into this exporter."""
        with self._lock:
            key = id(event_manager)
            entry = self._children.pop(key, None)
            if entry is None:
                return
            child, unsubscribers, enclosing_tool_call_id = entry
            for unsub in unsubscribers:
                try:
                    unsub()
                except Exception:  # noqa: BLE001
                    logger.debug("atif: failed to unsubscribe child handler", exc_info=True)
            child_trajectory = child.get_trajectory()
            # Lift child trajectory into root.subagent_trajectories[].
            root = self._trajectory
            children_list = list(root.subagent_trajectories or [])
            children_list.append(child_trajectory)
            self._trajectory = root.model_copy(update={"subagent_trajectories": children_list})
            # Reference the embedded child from a parent observation rather
            # than leaving it orphaned.
            if child_trajectory.trajectory_id:
                if enclosing_tool_call_id:
                    # Ran inside a tool call: ref it on that tool_call's observation.
                    self._pending_subagent_refs.setdefault(enclosing_tool_call_id, []).append(
                        SubagentTrajectoryRef(
                            trajectory_id=child_trajectory.trajectory_id,
                            session_id=child_trajectory.session_id,
                        )
                    )
                else:
                    # Pure-Python orchestrator (no tool call): a dispatch step
                    # carrying the ref.
                    self._dispatch_or_buffer(
                        _DispatchStep(
                            name=child_trajectory.agent.name,
                            trajectory_id=child_trajectory.trajectory_id,
                            session_id=child_trajectory.session_id,
                            timestamp=_iso8601_utc(),
                            extra={"event_kind": "standalone_dispatch"},
                        )
                    )
            self._write()

    def _embed_subagent_handoff(self, child: AtifExporter, start_step: int, end_step: int) -> None:
        """Embed a nested sub-*Agent*'s trajectory and record a handoff.

        Called from the child's :meth:`on_after_agent_call` when its run was
        nested under this exporter. The child accumulates ONE trajectory across
        all its calls (an OO agent shares event history), so we upsert it by
        ``trajectory_id`` and emit a handoff reference step whose ref records
        the step-range this particular call produced.
        """
        with self._lock:
            child_traj = child.get_trajectory()
            tid = child_traj.trajectory_id
            if not tid:
                return
            self._upsert_subagent(child_traj)
            self._dispatch_or_buffer(
                _DispatchStep(
                    name=child_traj.agent.name,
                    trajectory_id=tid,
                    session_id=child_traj.session_id,
                    timestamp=_iso8601_utc(),
                    extra={
                        "event_kind": "subagent_handoff",
                        "subagent_step_range": [start_step, end_step],
                    },
                )
            )
            self._write()

    def _upsert_subagent(self, child_traj: Trajectory) -> None:
        """Insert *child_traj* into ``subagent_trajectories[]``, or replace the
        existing entry with the same ``trajectory_id`` (a reused sub-agent's
        accumulating trajectory)."""
        children = list(self._trajectory.subagent_trajectories or [])
        for i, existing in enumerate(children):
            if existing.trajectory_id == child_traj.trajectory_id:
                children[i] = child_traj
                break
        else:
            children.append(child_traj)
        self._trajectory = self._trajectory.model_copy(update={"subagent_trajectories": children})

    def _dispatch_or_buffer(self, dispatch: _DispatchStep) -> None:
        """Emit a delegation dispatch step now, or buffer it until the system
        step exists (a delegation before the first generation turn)."""
        if self._system_step_emitted:
            self._emit_dispatch_step(dispatch)
        else:
            self._buffered_dispatches.append(dispatch)

    def _emit_dispatch_step(self, dispatch: _DispatchStep) -> None:
        """Append a deterministic-dispatch step (``llm_call_count=0``) whose
        observation references an embedded sub-trajectory (standalone or
        sub-agent)."""
        kind = (dispatch.extra or {}).get("event_kind")
        if kind == "subagent_handoff":
            message = f"Handoff to sub-agent `{dispatch.name}`."
        else:
            message = f"Called standalone generation function `{dispatch.name}`."
        ref_extra = {k: v for k, v in (dispatch.extra or {}).items() if k != "event_kind"} or None
        self._append_step(
            StepObject(
                step_id=self._next_step_id,
                timestamp=dispatch.timestamp,
                source="agent",
                message=message,
                observation=ObservationSchema(
                    results=[
                        ObservationResultSchema(
                            subagent_trajectory_ref=[
                                SubagentTrajectoryRef(
                                    trajectory_id=dispatch.trajectory_id,
                                    session_id=dispatch.session_id,
                                    extra=ref_extra,
                                )
                            ],
                        )
                    ]
                ),
                llm_call_count=0,
                extra={"event_kind": kind, "subagent_name": dispatch.name},
            )
        )
        self._next_step_id += 1
        self._write()

    def _finalize_if_needed(self, *, success: bool | None) -> None:
        """Finalize a top-level trajectory that no generation turn finalized
        (a pure-Python orchestrator). Flushes buffered dispatches first. No-op
        if already finalized."""
        with self._lock:
            if self._finalized:
                return
            if self._buffered_dispatches or self._buffered_tasks:
                self._system_step_emitted = True
                for dispatch in self._buffered_dispatches:
                    self._emit_dispatch_step(dispatch)
                self._buffered_dispatches.clear()
                for buffered in self._buffered_tasks:
                    self._emit_task_step(buffered)
                self._buffered_tasks.clear()
            self._finalize_trajectory(success=success)
            self._write()

    @classmethod
    def _make_subagent(
        cls,
        *,
        agent_name: str,
        agent_version: str,
        session_id: str | None,
        trajectory_id: str,
    ) -> AtifExporter:
        """Create an in-memory child exporter (no disk writes)."""
        # Use a "null" path; this exporter never writes.
        child = cls.__new__(cls)
        child.path = Path(os.devnull)
        child._lock = threading.RLock()
        child._finalized = False
        child._trajectory = Trajectory(
            session_id=session_id,
            trajectory_id=trajectory_id,
            agent=AgentSchema(name=agent_name, version=agent_version),
            steps=[],
        )
        child._pending = {}
        child._python_outputs = {}
        child._tool_call_events = {}
        child._next_step_id = 1
        child._enclosing_tool_call_id = None
        child._children = {}
        child._pending_subagent_refs = {}
        child._writes_to_disk = False
        child._install_token = None
        child._run_token = None
        child._entrypoint_token = None
        child._arm_call_id = None
        child._adoptive_parent = None
        child._embed_start_step = 0
        child._saved_writes_to_disk = False
        child._system_step_emitted = False
        child._system_content_hash = None
        child._buffered_tasks = []
        child._buffered_dispatches = []
        child._pending_system_drift = None
        return child

    # =========================================================== internals

    def _most_recent_pending(self) -> _PendingStep | None:
        """Pick the most recently opened pending step.

        In the linear (non-concurrent) case this is the only pending step.
        """
        if not self._pending:
            return None
        return next(reversed(self._pending.values()))

    def _build_step_from_pending(self, ps: _PendingStep) -> StepObject:
        """Convert a closed pending step into a ``StepObject``.

        Attaches observation results for each tool_call from
        PythonOutput (preferred) or ToolCallEvent.result (fallback).
        Carries ``generation_id`` and ``parent_generation_id`` in
        ``extra`` so consumers can reconstruct Case-B nesting.
        """
        # Build observation.results[] from PythonOutput + ToolCallEvent.result.
        observation_results: list[ObservationResultSchema] = []
        for tc in ps.tool_calls:
            content = self._lookup_observation_content(tc.tool_call_id)
            subagent_refs = self._subagent_refs_for_tool_call(tc.tool_call_id)
            if content is None and not subagent_refs:
                # No observation found; tool_call still in flight, or no
                # observation produced. Skip the result entry; the tool_call
                # itself is still listed on the step.
                continue
            observation_results.append(
                ObservationResultSchema(
                    source_call_id=tc.tool_call_id,
                    content=content,
                    subagent_trajectory_ref=subagent_refs or None,
                )
            )

        observation = (
            ObservationSchema(results=observation_results) if observation_results else None
        )

        extra = dict(ps.extra) if ps.extra else {}
        # Preserve nesting structure (Case B-flatten).
        extra["generation_id"] = ps.generation_id
        if ps.parent_generation_id is not None:
            extra["parent_generation_id"] = ps.parent_generation_id

        return StepObject(
            step_id=self._next_step_id,
            timestamp=_iso8601_utc(ps.started_at),
            source="agent",
            model_name=ps.model_name,
            message=ps.message,
            reasoning_content=ps.reasoning_content,
            tool_calls=ps.tool_calls or None,
            observation=observation,
            metrics=ps.metrics,
            # Preserve 0 (deterministic-dispatch semantics per ATIF v1.7
            # §II) — a truthy fallback would drop it to None.
            llm_call_count=ps.llm_call_count,
            extra=extra,
        )

    def _subagent_refs_for_tool_call(self, tool_call_id: str) -> list[SubagentTrajectoryRef]:
        """Return refs to child trajectories spawned under *tool_call_id*.

        Populated by :meth:`_detach_child` (Case C). Empty when no
        standalone generation was attached to this tool call.
        """
        return list(self._pending_subagent_refs.get(tool_call_id, []))

    def _lookup_observation_content(self, tool_call_id: str) -> str | None:
        """Resolve observation content for a tool_call_id.

        Order:
          1. ``PythonOutput`` (richest: stdout/stderr/value/error).
          2. ``ToolCallEvent.result.content`` (for non-execute_python).
        Returns ``None`` when no observation has been recorded yet.
        """
        po = self._python_outputs.get(tool_call_id)
        if po is not None:
            content = _format_python_output_content(po)
            if content:
                return content
            # PythonOutput with empty stdout/stderr/value: still record it as
            # an explicit empty observation so consumers can distinguish
            # "no observation yet" from "observation was empty".
            return ""

        tce = self._tool_call_events.get(tool_call_id)
        if tce is not None and tce.result is not None and tce.result.content:
            return tce.result.content

        return None

    def _append_step(self, step: StepObject) -> None:
        """Append a step to the root trajectory (no-validation fast path)."""
        # We mutate self._trajectory.steps in place. Pydantic v2 models with
        # frozen=False allow this; appending to a list field is supported.
        self._trajectory.steps.append(step)

    def _finalize_trajectory(self, *, success: bool | None) -> None:
        """Compute final_metrics and lock the trajectory as finalized."""
        agent_metrics = [
            s.metrics
            for s in self._trajectory.steps
            if s.source == "agent" and s.metrics is not None
        ]
        total_prompt = sum(m.prompt_tokens or 0 for m in agent_metrics)
        total_completion = sum(m.completion_tokens or 0 for m in agent_metrics)
        total_cached = sum(m.cached_tokens or 0 for m in agent_metrics)
        total_cost = sum(m.cost_usd or 0.0 for m in agent_metrics)
        total_reasoning = sum((m.extra or {}).get("reasoning_tokens", 0) for m in agent_metrics)

        fm = FinalMetricsSchema(
            total_prompt_tokens=total_prompt if agent_metrics else None,
            total_completion_tokens=total_completion if agent_metrics else None,
            total_cached_tokens=total_cached if agent_metrics else None,
            total_cost_usd=round(total_cost, 6) if agent_metrics else None,
            total_steps=len(self._trajectory.steps),
            extra=({"reasoning_output_tokens": total_reasoning} if total_reasoning else None),
        )
        # model_copy keeps the rest of the trajectory immutable in shape.
        self._trajectory = self._trajectory.model_copy(update={"final_metrics": fm})
        if success is False:
            extra = dict(self._trajectory.extra or {})
            extra.setdefault("crashed", False)  # explicit signal in success=False path
            self._trajectory = self._trajectory.model_copy(update={"extra": extra})
        self._finalized = True

    def _write(self) -> None:
        """Atomically write the trajectory JSON to disk.

        No-op on in-memory subagent sub-exporters (where
        ``_writes_to_disk`` is False) — the root exporter handles
        persistence once the child is lifted.
        """
        if not self._writes_to_disk:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            # JSON is UTF-8 by definition (RFC 8259 section 8.1). Without an
            # explicit encoding this uses the locale default, and non-ASCII
            # model output raises into the swallow below, losing the file.
            tmp.write_text(
                self._trajectory.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            # Tracing must never break the run; log and move on.
            logger.exception("atif: failed to write trajectory to %s", self.path)

    def _build_message_with_images(
        self, text: str, images: list[dict[str, Any]] | None
    ) -> str | list[ContentPart]:
        """Render a text + image list as an ATIF v1.6+ ContentPart array.

        - Text-only ⇒ returns the raw string (unchanged behaviour).
        - With images: returns ``[ContentPart(text), ContentPart(image), ...]``,
          writing data-URL images to ``images/`` next to the trajectory file.

        Image dicts that don't match a known shape (e.g. opaque URLs)
        are skipped silently to avoid corrupting otherwise-valid
        trajectories.  In-memory subagent exporters skip image writes
        (no path to anchor to).
        """
        if not images:
            return text
        parts: list[ContentPart] = []
        if text:
            parts.append(ContentPart(type="text", text=text))
        for img in images:
            cp = self._image_dict_to_content_part(img)
            if cp is not None:
                parts.append(cp)
        if not parts:
            return text
        return parts

    def _image_dict_to_content_part(self, img: dict[str, Any]) -> ContentPart | None:
        """Convert one OpenAI-style image dict into an ATIF ContentPart.

        Handles ``{type: "image_url", "image_url": {"url": ...}}`` where the
        URL is either a ``data:image/<fmt>;base64,...`` blob (written to
        disk) or a regular URL/path (referenced as-is). Other shapes
        return ``None`` so unknown formats are dropped rather than
        crashing.
        """
        if not isinstance(img, dict):
            return None
        url = ""
        if img.get("type") == "image_url":
            inner = img.get("image_url") or {}
            if isinstance(inner, dict):
                url = inner.get("url", "")
            elif isinstance(inner, str):
                url = inner
        elif img.get("type") == "image" and isinstance(img.get("source"), dict):
            # Already in ATIF-ish shape (rare but worth supporting).
            src = img["source"]
            mt = src.get("media_type")
            path = src.get("path")
            if mt in _MEDIA_TYPE_TO_EXT and path:
                return ContentPart(
                    type="image",
                    source=ImageSource(media_type=cast(Any, mt), path=path),
                )
            return None
        if not url:
            return None

        match = _DATA_URL_RE.match(url)
        if match:
            mt = match.group("media_type")
            if mt not in _MEDIA_TYPE_TO_EXT or not self._writes_to_disk:
                return None
            ext = _MEDIA_TYPE_TO_EXT[mt]
            try:
                blob = base64.b64decode(match.group("data"), validate=True)
            except (ValueError, binascii.Error):
                return None
            images_dir = self.path.parent / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{uuid4().hex[:12]}.{ext}"
            (images_dir / fname).write_bytes(blob)
            return ContentPart(
                type="image",
                source=ImageSource(media_type=cast(Any, mt), path=f"images/{fname}"),
            )

        # Plain URL or path passthrough; guess media_type from extension.
        ext = os.path.splitext(url)[1].lstrip(".").lower()
        mt = next((m for m, e in _MEDIA_TYPE_TO_EXT.items() if e == ext), None)
        if mt is None:
            return None
        return ContentPart(
            type="image",
            source=ImageSource(media_type=cast(Any, mt), path=url),
        )
