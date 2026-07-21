# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event types for the event pipeline.

Events flow through EventManager.add() for:
- Recording events (if record=True)
- Notifying subscribers via on()
- Telemetry (future: hooks)

Design: phase-2-strategy-middleware.md

NVIDIA OO Agents events extend context-blocks EventBase and add specialized types
for the code generation pipeline (task, error, feedback).

Rendering: Events are rendered using pformat(event). Fields with repr=False
are excluded. Private fields (_field with PrivateAttr) are also excluded.

Type names follow "Type Names are Prompts" - no redundant "Event" suffix.
"""

from collections.abc import Callable
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, Field, field_serializer
from pydantic_core import PydanticSerializationError, to_json

from nooa.agentdoc import spec
from nooa.context_blocks import EventBase as EventBase
from nooa.context_blocks import ResultStatus as ResultStatus
from nooa.context_blocks.models import Role

# Sentinel value to distinguish "no return" from "return None"
_NO_RETURN = object()


# === Execution Signals ===


class ExecutionSignal(BaseException):
    """Base class for control flow signals (not errors).

    Signals are used for non-error control flow like return_result().
    They should NOT be recorded as errors in traces.

    Inherits from BaseException (not Exception) so signals are not caught
    by 'except Exception:' blocks in LLM-generated code. This follows
    Python's convention for control flow exceptions like KeyboardInterrupt,
    SystemExit, and GeneratorExit.

    Subclasses can carry additional data (e.g., the result value).
    """

    pass


# === NVIDIA OO Agents-specific events ===


class Task(EventBase):  # type: ignore[misc]
    """Task prompt event - added at start of generation."""

    _role: ClassVar[Role] = Role.USER

    prompt: Annotated[
        str, spec(max_string=None), Field(description="Task prompt describing what to do")
    ]
    images: list[dict[str, Any]] = Field(
        default_factory=list,
        repr=False,
        description="Multimodal content blocks (images, audio, files) attached to this task",
    )


class Message(EventBase):  # type: ignore[misc]
    """User-facing message from generated code via message()."""

    _role: ClassVar[Role] = Role.ASSISTANT

    content: Annotated[str, Field(description="User-facing message content")]


class Reasoning(EventBase):  # type: ignore[misc]
    """Chain-of-thought from generated code via the removed reasoning() builtin.

    Legacy: nothing emits this event anymore. The class is kept so stored
    traces from before the builtin's removal still deserialize and render.
    """

    _role: ClassVar[Role] = Role.ASSISTANT

    content: Annotated[str, Field(description="Chain-of-thought reasoning content")]


class Error(EventBase):  # type: ignore[misc]
    """Error for LLM retry."""

    _role: ClassVar[Role] = Role.USER

    content: Annotated[str, Field(description="Error message for LLM retry")]


class DebugTrace(EventBase):  # type: ignore[misc]
    """Debug info stored in traces but never shown to the LLM.

    Use for capturing raw LLM responses, internal state, or diagnostics
    that are useful for post-hoc debugging but should not influence generation.
    """

    _role: ClassVar[Role] = Role.METADATA

    content: Annotated[str, Field(description="Debug information for trace inspection")]


class Feedback(EventBase):  # type: ignore[misc]
    """Execution feedback when target method not yet defined."""

    _role: ClassVar[Role] = Role.USER

    content: Annotated[str, Field(description="Execution feedback content")]


class TextOnlyReply(EventBase):  # type: ignore[misc]
    """A turn where the LLM replied with plain text instead of a tool call.

    Recorded (so it lands in the events table and is queryable / replayable)
    but ``Role.METADATA`` — never rendered to the model, so capturing a
    text-only reply does not change generation. This is the durable, structured record that
    powers ``/bug`` capture and time-travel replay; the model-visible
    correction is a separate ``Error``/``Feedback`` event (``Role.USER``).

    Replaces the lossy ``DebugTrace`` previously written on the CodeAct
    text-only path, which truncated the content and could not be relied on by
    downstream consumers.
    """

    _role: ClassVar[Role] = Role.METADATA

    content: Annotated[
        str, spec(max_string=None), Field(description="Verbatim text the model emitted")
    ] = ""
    finish_reason: Annotated[
        str, Field(description="LLM finish_reason for the text-only turn (e.g. 'stop')")
    ] = ""
    route: Annotated[
        str,
        Field(description="Handling route: 'return_result' or 'synthetic_comment'"),
    ] = ""
    consecutive_text_only: Annotated[
        int, Field(description="Count of consecutive text-only turns including this one")
    ] = 0
    recovered: Annotated[
        bool,
        Field(
            description=(
                "Set True on a later turn if the model issued a real tool call "
                "after this text-only reply (i.e. the corrective feedback worked)."
            )
        ),
    ] = False


class LLMOutput(EventBase):  # type: ignore[misc]
    """Raw LLM output - code (PURE_PYTHON), JSON (STRUCTURED_OUTPUT), or tool calls (CODEACT)."""

    _role: ClassVar[Role] = Role.ASSISTANT

    content: Annotated[
        str, spec(max_string=None), Field(description="LLM response content (code or JSON)")
    ]


class PythonOutput(EventBase):  # type: ignore[misc]
    """Output from execute_python - appears as user message in events.

    This event captures the actual output from code execution, separate from
    the tool result status. It allows nested agent calls without breaking
    message ordering requirements (tool results must immediately follow tool calls).
    """

    _role: ClassVar[Role] = Role.USER

    tool_call_id: Annotated[str, Field(description="ID of the tool call that produced this output")]
    execution_status: Annotated[
        ResultStatus,
        Field(description="Execution status (ResultStatus.COMPLETE or ResultStatus.ERROR)"),
    ]
    execution_count: Annotated[
        int,
        Field(
            repr=False, description="Jupyter-style execution count (1, 2, 3, ...) for Out[n] access"
        ),
    ]
    stdout: Annotated[str, spec(max_string=None)] = Field(
        default="", description="Captured stdout from execution"
    )
    stderr: Annotated[str, spec(max_string=None)] = Field(
        default="", description="Captured stderr from execution"
    )
    error: str = Field(default="", description="Formatted error message if execution failed")
    value: Any = Field(default=None, description="Python object returned (for Out[n] access)")
    explicit_return: bool = Field(
        default=False,
        repr=False,
        description="True if value from explicit `return x`, False if implicit",
    )
    images: list[dict[str, Any]] = Field(
        default_factory=list,
        repr=False,
        description="Image content blocks captured via show() during execution",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("value")
    def _serialize_value(self, value: Any, _info: Any) -> Any:
        """Render values pydantic-core can't JSON-encode as a bounded ``pformat`` string.

        ``value`` is ``Any`` + ``arbitrary_types_allowed``, so a cell can return an
        object with no pydantic-core JSON serializer (e.g. a ``CancelledError`` from
        an awaited Task, a coroutine, a lock). ``model_dump_json()`` (the SQLite
        event store) would then raise ``PydanticSerializationError`` and wedge the
        turn. We probe with ``to_json`` so pydantic-native types (datetime, UUID,
        Decimal, set, ...) still round-trip structurally, and only fall back to
        ``pformat`` — bounded by ``FormatConfig`` defaults, since this hook can't
        reach the live ``TruncationConfig`` — for genuinely un-encodable objects.
        """
        if value is None:
            return value
        try:
            to_json(value)
        except (PydanticSerializationError, TypeError, ValueError):
            from nooa.agentdoc import pformat
            from nooa.config.truncation_config import FormatConfig

            return pformat(value, **FormatConfig().model_dump())
        return value


class BeforeTurn(EventBase):  # type: ignore[misc]
    """Event emitted before each LLM generation turn.

    This event is emitted before every call to runtime.generate(), which happens
    in strategy loops (e.g., CodeAct's multi-turn loop). It's used to trigger
    event management policies before each generation turn.

    Uses Role.RUNTIME_EVENT to indicate it's never recorded in conversation events.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    method_name: Annotated[str, Field(description="Name of the method being generated")]
    strategy: Annotated[str, Field(description="Strategy name")]
    generation_id: Annotated[str, Field(description="Generation session ID")]
    parent_generation_id: str | None = Field(
        default=None, description="Parent generation ID if nested"
    )
    turn_number: Annotated[
        int, Field(description="Turn number within this generation session (1, 2, 3, ...)")
    ]


class AfterTurn(EventBase):  # type: ignore[misc]
    """Event emitted after each LLM generation turn.

    Symmetric with BeforeTurn. Emitted after every generation turn completes.
    When is_final=True, this is the last turn and the method has completed.

    Uses Role.RUNTIME_EVENT to indicate it's never recorded in conversation events.
    Use this for per-turn cleanup, event management, or post-generation analysis.

    Turn numbering:
    - turn_number=1, is_final=False: First turn completed, more turns expected
    - turn_number=N, is_final=True: Final turn, method complete
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    method_name: Annotated[str, Field(description="Name of the method being generated")]
    strategy: Annotated[str, Field(description="Strategy name")]
    generation_id: Annotated[str, Field(description="Generation session ID")]
    parent_generation_id: str | None = Field(
        default=None, description="Parent generation ID if nested"
    )
    turn_number: Annotated[
        int, Field(description="Turn number within this generation session (1, 2, 3, ...)")
    ]
    is_final: Annotated[bool, Field(description="True if this is the final turn (method complete)")]
    success: bool | None = Field(
        default=None,
        description="True if method completed successfully (only set when is_final=True)",
    )
    exception_type: str | None = Field(
        default=None, description="Exception type if failed (only set when is_final=True)"
    )


class BeforeAgentCall(EventBase):  # type: ignore[misc]
    """Emitted when any agent method invocation begins (generation, pure-Python,
    or sync). Pub-sub complement to the ``agent_call`` middleware, symmetric with
    :class:`BeforeTurn`. ``Role.RUNTIME_EVENT`` so it is never recorded.

    ``is_top_level`` is the agent-instance nesting axis (no agent active before
    this call) — subscribers wanting "outermost run" filter on it, not
    ``parent_call_id`` (the call stack is also pushed for sync/``@no_trace``).
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    method_name: Annotated[str, Field(description="Name of the agent method being invoked")]
    call_id: Annotated[str, Field(description="Unique ID for this agent-method call")]
    parent_call_id: str | None = Field(
        default=None, description="Nearest traced ancestor call ID, or None at the top level"
    )
    is_top_level: Annotated[
        bool,
        Field(description="True iff no agent was active (_parent_agent_var) before this call"),
    ]
    needs_generation: bool = Field(
        default=False, description="True for ellipsis (LLM) methods; False for pure-Python/sync"
    )


class AfterAgentCall(EventBase):  # type: ignore[misc]
    """Completion of an agent method (success or exception), symmetric with
    :class:`BeforeAgentCall`. ``Role.RUNTIME_EVENT`` (never recorded)."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    method_name: Annotated[str, Field(description="Name of the agent method that was invoked")]
    call_id: Annotated[str, Field(description="Unique ID for this agent-method call")]
    parent_call_id: str | None = Field(
        default=None, description="Nearest traced ancestor call ID, or None at the top level"
    )
    is_top_level: Annotated[
        bool,
        Field(description="True iff no agent was active (_parent_agent_var) before this call"),
    ]
    needs_generation: bool = Field(
        default=False, description="True for ellipsis (LLM) methods; False for pure-Python/sync"
    )
    success: Annotated[bool, Field(description="True if the method returned normally")] = True
    exception_type: str | None = Field(
        default=None, description="Exception type name if the method raised"
    )


class TuiSessionResumed(EventBase):  # type: ignore[misc]
    """Emitted once after a TUI agent is reconstituted from a snapshot.

    Fired after ``restore_latest_snapshot()`` completes (both the ``-c``
    startup resume in ``bootstrap`` and the ``/session resume`` command), before
    the first turn runs — so the agent's snapshot-backed state (``self.v``,
    todos, ...) is already in place. Skills can subscribe via
    ``event_manager.on("TuiSessionResumed", handler)`` to run post-restore setup
    (e.g. ``agent_mesh`` reconnecting and reclaiming its handle).

    Always emitted on session startup/resume; ``restored`` distinguishes an
    actual snapshot restore (True) from a fresh session with nothing to restore
    (False), so handlers can no-op when there's nothing to reconstitute.

    Uses Role.RUNTIME_EVENT — never recorded in conversation events, never in
    LLM context.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    session_id: Annotated[str, Field(description="The resumed/started session id")]
    restored: Annotated[
        bool,
        Field(description="True if a snapshot was actually restored into the agent"),
    ]


class TuiSessionCleared(EventBase):  # type: ignore[misc]
    """Emitted once after a TUI agent's working state is reset by ``/clear``.

    Fired at the end of ``_reset_agent_working_state()`` — after todos, vars,
    user context blocks, and the shell have been reset, with the agent in its
    clean post-clear state. The symmetric counterpart to ``TuiSessionResumed``:
    skills subscribe via ``event_manager.on("TuiSessionCleared", handler)`` to
    re-initialize session-scoped state (caches, connections, ...) for the fresh
    session.

    Uses Role.RUNTIME_EVENT — never recorded in conversation events, never in
    LLM context.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    session_id: Annotated[
        str | None,
        Field(default=None, description="The new (post-clear) session id, if known"),
    ]


class LLMCallStart(EventBase):  # type: ignore[misc]
    """Emitted immediately before the LLM round-trip (``acall()``) in
    ``runtime.generate()``.

    Observable via ``event_manager.on("LLMCallStart", fn)`` — fire-and-forget,
    for telemetry / activity surfacing (e.g. the TUI status line showing
    "waiting on model"). Pairs 1:1 with a later ``LLMCallEnd`` carrying the
    same ``generation_id`` + ``turn_number``.

    Uses Role.RUNTIME_EVENT — never recorded in conversation events, never
    shown to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    method_name: Annotated[str, Field(description="Name of the method being generated")]
    strategy: Annotated[str, Field(description="Strategy name")]
    generation_id: Annotated[str, Field(description="Generation session ID")]
    turn_number: Annotated[
        int, Field(description="Turn number within this generation session (1, 2, 3, ...)")
    ]


class LLMCallEnd(EventBase):  # type: ignore[misc]
    """Emitted immediately after the LLM round-trip returns (or fails) in
    ``runtime.generate()``. Symmetric with ``LLMCallStart``.

    Observable via ``event_manager.on("LLMCallEnd", fn)``. ``success`` is
    False (and ``exception_type`` set) when ``acall()`` raised — the event
    still fires so observers can clear their "waiting on model" state.

    Uses Role.RUNTIME_EVENT — never recorded in conversation events.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    method_name: Annotated[str, Field(description="Name of the method being generated")]
    strategy: Annotated[str, Field(description="Strategy name")]
    generation_id: Annotated[str, Field(description="Generation session ID")]
    turn_number: Annotated[
        int, Field(description="Turn number within this generation session (1, 2, 3, ...)")
    ]
    success: Annotated[bool, Field(description="True if the LLM call returned normally")] = True
    exception_type: str | None = Field(
        default=None, description="Exception type name if the call raised"
    )


class Notification(EventBase):  # type: ignore[misc]
    """Generic "something happened" signal rendered into the LLM context.

    Not tied to any specific producer. The framework never emits this
    event itself — input channels emit :class:`QueueOutput` on ``put()``
    (see ``runtime/channels.py``). ``Notification`` is a user-emitted
    signal you can ``event_manager.add(...)`` yourself for long-running
    job completions, timer ticks, external webhook deliveries, or any
    other asynchronous signal the agent should know about.

    - ``source`` identifies the producer — convention is a short
      namespaced string like ``"queue:user_messages"``, ``"job:12345"``,
      ``"timer:daily-cron"``. The outer dispatcher keys off this to
      decide which handler to run next.
    - ``description`` is a free-form string for the LLM; include enough
      to make the notification self-describing in the event stream.
    """

    _role: ClassVar[Role] = Role.USER

    source: Annotated[
        str, Field(description="Origin of the notification (e.g. 'queue:user_messages')")
    ]
    description: Annotated[
        str, Field(description="Human-readable description of what happened")
    ] = ""


class Summary(EventBase):  # type: ignore[misc]
    """Collapsed events - with optional summary text.

    Represents a range of events that have been archived and optionally summarized.
    Supports tree traversal for progressive disclosure via EventManager methods.

    - summary_text="..." → LLM-generated summary of collapsed events
    - summary_text=None  → events collapsed without summary (truncation)

    The children_tags field is hidden from rendering (repr=False) to save tokens.
    Agents can access collapsed events via events[summary.children_tags].

    String tags:
    - Individual events: "1", "2", "3", etc.
    - Summary events: "2..40" (range that was collapsed)
    - events["5"] always works, even after event is archived
    """

    _role: ClassVar[Role] = Role.ASSISTANT  # LLM's own recap of past actions

    # The tag for this summary event (e.g., "2..40") - REQUIRED
    summary_tag: Annotated[
        str,
        Field(description="String tag for this summary (e.g., '2..40')"),
    ]
    # Original range for view ordering (still int-based internally)
    replaced_range: Annotated[
        tuple[int, int],
        Field(description="(start_seq, end_seq) for view ordering"),
    ]
    children_tags: list[str] = Field(
        default_factory=list, repr=False, description="Tags of directly replaced events"
    )
    summary_text: Annotated[str | None, spec(max_string=None)] = Field(
        default=None, description="LLM-generated summary, or None for truncation"
    )
    doc: str = Field(default="", description="Usage hint for accessing collapsed events")


class LLMComplete(EventBase):  # type: ignore[misc]
    """Emitted after each LLM round-trip completes.

    Carries the LLMResponse metadata (tokens, cost, model_name,
    structured tool_calls list, reasoning_content) as a structured
    event so downstream consumers don't need ``intercept('llm_call')``
    just to observe LLM metrics. The ATIF exporter relies on this
    event for ``step.metrics`` and ``step.tool_calls``.

    Uses ``Role.RUNTIME_EVENT`` to keep it out of LLM context — this
    is observability metadata, never rendered to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    model_name: Annotated[
        str,
        Field(description="Model identifier the LLM call was routed to"),
    ] = ""
    prompt_tokens: Annotated[
        int,
        Field(description="Total input tokens (incl. cached) sent to the model"),
    ] = 0
    completion_tokens: Annotated[
        int,
        Field(description="Total tokens generated by the model"),
    ] = 0
    cached_tokens: Annotated[
        int,
        Field(description="Subset of prompt_tokens served from prompt cache"),
    ] = 0
    reasoning_tokens: Annotated[
        int,
        Field(description="Reasoning tokens (subset of completion_tokens for thinking models)"),
    ] = 0
    cost_usd: Annotated[
        float,
        Field(description="Estimated monetary cost of this LLM call in USD"),
    ] = 0.0
    tool_calls: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "Structured tool_calls list from LLMResponse.tool_calls. "
                "Each entry: {tool_call_id, function_name, arguments}. "
                "Canonical runtime ids (call_* for OpenAI Responses)."
            )
        ),
    ] = []  # noqa: RUF012  # default_factory not needed; events are immutable post-construction
    reasoning_content: Annotated[
        str,
        Field(description="Chain-of-thought string from the response, when provided"),
    ] = ""
    generation_id: Annotated[
        str,
        Field(
            description=(
                "Matches the surrounding BeforeTurn/AfterTurn generation_id; "
                "used by consumers (e.g. ATIF exporter) to pair the metrics "
                "with the right turn under concurrent generation."
            )
        ),
    ] = ""
    dynamic_context: Annotated[
        str,
        Field(
            description=(
                "Rendered content of the trailing ``<context>…</context>`` "
                "envelope that wraps dynamic SYSTEM-role blocks at the end "
                "of the messages list (``CachedBlockFormatter`` design). "
                "Re-rendered every LLM call from current agent state; "
                "captured here so downstream consumers (e.g. ATIF exporter "
                "writing it onto ``step.extra.dynamic_context``) can "
                "reconstruct what the LLM saw on each turn without having "
                "to keep the full per-turn messages list. Empty string when "
                "no dynamic blocks are present."
            )
        ),
    ] = ""


class SystemPrompt(EventBase):  # type: ignore[misc]
    """Snapshot of the static SYSTEM-role content for an LLM call.

    Fired by ``runtime.generate()`` right after the messages list is
    built. Carries the rendered ``messages[0].content`` (the static
    system prompt: strategy prompt + ``doc(self)`` + static context
    blocks). The trailing dynamic-context envelope is captured
    separately on :class:`LLMComplete.dynamic_context`.

    Uses ``Role.RUNTIME_EVENT`` to keep it out of LLM context — this is
    pure observability metadata. ATIF consumers emit a single
    ``source: "system"`` step from the first ``SystemPrompt`` per
    trajectory; subsequent occurrences with different content are
    treated as drift on the relevant agent step.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    content: Annotated[
        str,
        spec(max_string=None),
        Field(
            description=(
                "Rendered static system content (``messages[0].content`` "
                "for the LLM call about to happen). Stable across turns "
                "of one generation method when all SYSTEM-role blocks "
                "are marked ``static=True``; drift indicates a mutation "
                "of those blocks mid-run."
            )
        ),
    ] = ""
    generation_id: Annotated[
        str,
        Field(
            description=(
                "Matches the surrounding BeforeTurn/AfterTurn "
                "generation_id; used by consumers to pair the system "
                "snapshot with the right turn under concurrent generation."
            )
        ),
    ] = ""


# === Union of all core event types ===

Event = (
    Task
    | Message
    | Reasoning
    | Error
    | Feedback
    | LLMOutput
    | PythonOutput
    | Notification
    | Summary
    | BeforeTurn
    | AfterTurn
    | TuiSessionResumed
    | TuiSessionCleared
    | LLMComplete
    | SystemPrompt
    | TextOnlyReply
)


# === ExecutionResult ===


class ExecutionResult(BaseModel):
    """Result of code execution via RuntimeServices.execute_code()."""

    stdout: str = Field(default="", description="Captured stdout from execution")
    stderr: str = Field(default="", description="Captured stderr from execution")
    error: Exception | None = Field(default=None, description="Exception if execution failed")
    signal: ExecutionSignal | None = Field(
        default=None, description="Control flow signal (not an error), e.g. return_result()"
    )
    defined_methods: dict[str, Callable[..., Any]] = Field(
        default_factory=dict, description="Methods defined during execution", exclude=True
    )
    returned_value: Any = Field(default=_NO_RETURN, description="Value returned by code")
    explicit_return: bool = Field(
        default=False, description="True only for explicit `return x` statements"
    )
    captured_locals: dict[str, Any] = Field(
        default_factory=dict,
        description="Local variables captured for REPL-style persistence",
        exclude=True,
    )
    images: list[dict[str, Any]] = Field(
        default_factory=list,
        repr=False,
        description="Image content blocks captured via show() during execution",
    )
    wrapper_line_offset: int = Field(
        default=0,
        description="Number of wrapper lines before user code (for traceback adjustment)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def success(self) -> bool:
        """True if execution completed without error."""
        return self.error is None

    @property
    def has_return(self) -> bool:
        """True if code produced a return value (explicit or implicit)."""
        return self.returned_value is not _NO_RETURN

    def has_method(self, name: str) -> bool:
        """Check if a specific method was defined."""
        return name in self.defined_methods

    def format_output(self, *, fenced: bool = False) -> str:
        """Format stdout/stderr for LLM feedback.

        Args:
            fenced: If True, wrap output in markdown code fences (```).
                   If False, use plain text format.

        Returns:
            Formatted string with stdout/stderr sections, or empty string if no output.
        """
        parts = []
        if self.stdout:
            if fenced:
                parts.append(f"Stdout:\n```\n{self.stdout}\n```")
            else:
                parts.append(f"Stdout:\n{self.stdout}")
        if self.stderr:
            if fenced:
                parts.append(f"Stderr:\n```\n{self.stderr}\n```")
            else:
                parts.append(f"Stderr:\n{self.stderr}")
        return "\n\n".join(parts)
