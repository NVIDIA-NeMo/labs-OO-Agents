# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data that crosses the Session boundary.

Everything here is a pydantic model with data fields only: no callables,
no live objects. Sessions, hosts and parent agents exchange these values;
live agents never cross.
"""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field, SerializeAsAny, ValidationError, field_validator

from nooa.interactive import Done, NeedInput
from nooa.runtime.turn_loop import TurnCancelled  # noqa: F401  (re-exported)

if TYPE_CHECKING:
    from nooa_coder.session.port import SessionPort

SessionStatus = Literal["running", "idle", "retained", "closed", "on_disk"]
ChildStatus = SessionStatus  # a child reports the same statuses as any session


class TaskResult(BaseModel):
    """Structured result of a delegated objective or a benchmark task.

    The field descriptions are what the model reads when it builds one;
    they come from the benchmark agent, which re-exports this class.
    """

    solution_description: str = Field(
        description="What you did and why it solves the problem. Describe root cause and fix."
    )
    evidence: str = Field(
        description=(
            "Concrete evidence that the task is done: what tests passed, "
            "what output was produced, what behavior changed. Not a guess -- "
            "cite the actual results you observed."
        )
    )
    how_to_verify: str = Field(
        title="How to Verify",
        description=(
            "How a verifier can confirm correctness: concrete checks or steps and their "
            "expected results. Include commands when appropriate; a shell command is not required."
        ),
    )
    report: str = Field(
        default="",
        description="Concise human-readable report that a parent or runner shows inline.",
    )


class Receipt(BaseModel):
    """Returned by ``submit()`` and ``steer()``: where an item went and its id.

    ``delivered`` is ``queued`` when the item waits on a channel for a
    turn, ``steered`` when it was handed to the running turn's next model
    call.
    """

    session_id: str
    channel: str
    item_id: str
    delivered: Literal["queued", "steered"]


class ChildRef(BaseModel):
    """A handle on a child session: data plus methods.

    It holds the child's id and a few facts, never the child itself. The
    methods find the parent's port at call time from the running turn, so
    a ``ChildRef`` kept in ``self.v`` still works after a checkpoint and a
    reload, and one that arrives in a ``ChildResult`` can be acted on
    directly (``item.child.send(...)``). Only the child's parent can use
    it: ``parent_id`` is that parent, and the methods raise
    ``ChildFailedError`` in any other session.
    """

    id: str
    name: str
    depth: int
    status: ChildStatus
    parent_id: str | None = None

    async def wait(self) -> Done:
        """Wait for the child's next ``Done`` and return it.

        Raises ``ChildFailedError`` if the child fails or is closed first.
        """
        return await _port(self).wait_child(self.id)

    async def send(self, item: Any, *, channel: str = "user_messages") -> "Receipt":
        """Send an item (text or data) to the child on one of its channels."""
        return await _port(self).send_child(self.id, item, channel=channel)

    async def steer(self, text: str) -> "Receipt":
        """Give the child's running turn extra text; queued if it is idle."""
        return await _port(self).steer_child(self.id, text)

    async def close(self) -> None:
        """Close the child session (its children first)."""
        await _port(self).close_child(self.id)

    @property
    def info(self) -> "SessionInfo":
        """The child's current metadata."""
        return _port(self).child_info(self.id)


def _port(ref: ChildRef) -> "SessionPort":
    """The running turn's port, if its session is ``ref``'s parent."""
    from nooa_coder.session.port import require_port

    port = require_port()
    if ref.parent_id is not None and ref.parent_id != port.id:
        raise ChildFailedError(
            f"Child {ref.id!r} belongs to session {ref.parent_id!r}, not to {port.id!r}"
        )
    return port


class ChildFailedError(RuntimeError):
    """Raised by ``ChildRef.wait()`` when the child fails or is closed before a ``Done``."""


class ChildResult(BaseModel):
    """A child finished a turn with ``Done``; delivered on the parent's ``delegates`` channel.

    A ``done.result`` that arrives as data (a child that returned a dict, or
    a result reloaded from the record) is rebuilt as a ``TaskResult`` when it
    is one. ``delegate()`` gets its ``Done`` through this class too.
    """

    child: ChildRef
    done: Done

    @field_validator("done", mode="after")
    @classmethod
    def _task_result(cls, done: Done) -> Done:
        if isinstance(done.result, dict):
            try:
                result = TaskResult.model_validate(done.result)
            except ValidationError:
                return done
            done = done.model_copy(update={"result": result})
        return done


class ChildQuestion(BaseModel):
    """A child asked a question (its ``NeedInput``); answer with ``item.answer(...)``.

    ``answer_schema`` is the JSON schema of the child's ``answer_type`` when
    it gave one; the answer is then a dict matching it.
    """

    child: ChildRef
    question: str
    options: list[str] | None = None
    answer_schema: dict[str, Any] | None = None

    @classmethod
    def from_need_input(cls, child: ChildRef, need: NeedInput) -> "ChildQuestion":
        """Convert a child's ``NeedInput``; its answer class becomes a JSON schema."""
        schema = need.answer_type.model_json_schema() if need.answer_type is not None else None
        return cls(child=child, question=need.question, options=need.options, answer_schema=schema)

    async def answer(self, answer: Any) -> "Receipt":
        """Send the answer to the child; shorthand for ``self.child.send(answer)``."""
        return await self.child.send(answer)


class ChildFailed(BaseModel):
    """A child's turn failed or the child ended without a result."""

    child: ChildRef
    error: str


class TurnCancelledOutcome(BaseModel):
    """What ``prompt()`` returns when the turn it waited for was cancelled."""

    by: str


class TranscriptEntry(BaseModel):
    """One line of a session's transcript as a host shows it."""

    role: Literal["user", "agent", "question", "cancelled", "note"]
    content: str
    item_id: str | None = None
    timestamp: float


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cost_usd",
)
"""A session's own usage fields; each has an ``attributed_`` twin for its children.

The names match ``nooa.llm_types.LLMUsage``, which a model response carries.
"""


class Usage(BaseModel):
    """Token and cost totals: the session's own and those attributed from its children.

    ``cached_input_tokens`` are input tokens read from the provider's prompt
    cache, ``cache_write_input_tokens`` those written to it, and
    ``reasoning_tokens`` output tokens spent on reasoning.
    ``last_input_tokens`` is the input of this session's latest model call
    (the context in use); it is not a total, so ``own()`` and
    ``with_attributed()`` leave it out.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    attributed_input_tokens: int = 0
    attributed_output_tokens: int = 0
    attributed_cached_input_tokens: int = 0
    attributed_cache_write_input_tokens: int = 0
    attributed_reasoning_tokens: int = 0
    attributed_total_tokens: int = 0
    attributed_cost_usd: float = 0.0
    last_input_tokens: int = 0

    def own(self) -> "Usage":
        """Only this session's own totals."""
        return Usage(**{name: getattr(self, name) for name in USAGE_FIELDS})

    def with_attributed(self) -> "Usage":
        """Own plus attributed totals, as own totals."""
        return Usage(
            **{
                name: getattr(self, name) + getattr(self, f"attributed_{name}")
                for name in USAGE_FIELDS
            }
        )

    def has_attributed(self) -> bool:
        return any(getattr(self, f"attributed_{name}") for name in USAGE_FIELDS)


class SessionInfo(BaseModel):
    """Metadata for one session, live or on disk.

    ``status`` is ``on_disk`` for a session read from the store; the
    registry reports ``running``, ``idle`` or ``retained`` for live ones.
    """

    id: str
    parent_id: str | None = None
    depth: int = 0
    name: str | None = None
    title: str | None = None
    title_is_user_set: bool = False
    workspace: str = ""
    host: str = ""
    agent: str = ""
    model: str = ""
    reasoning: str | None = None
    """The reasoning level chosen with ``set_reasoning``; ``None`` for the model's default."""
    mode: str = "auto"
    status: SessionStatus = "on_disk"
    retained: bool = False
    turn_method: str = "handle"
    created_at: float = 0.0
    last_active: float = 0.0
    turn_count: int = 0
    reply_count: int = 0
    usage: Usage = Field(default_factory=Usage)
    forked_from: str | None = None
    """The session this one was copied from by ``SessionStore.fork``, if any."""


class InUseSession(BaseModel):
    """A stored session some process holds, as ``SessionStore.in_use()`` reports it.

    ``owner`` describes the holder: ``pid N on HOST`` (another machine),
    ``local process (pid N)``, or ``old TUI (pid N)``. ``title``,
    ``parent_id`` and ``workspace`` are empty when the file could not be
    read. ``last_write`` is the file's modification time.
    """

    id: str
    title: str | None = None
    parent_id: str | None = None
    workspace: str = ""
    owner: str
    last_write: float


class CommandInfo(BaseModel):
    """A slash command a session offers."""

    name: str
    description: str = ""
    input_hint: str | None = None


class ModelInfo(BaseModel):
    """The model a session's next call uses, as data (``Session.model_info()``).

    ``reasoning_levels`` is empty when the client declares none;
    ``reasoning_level`` is ``None`` when none was chosen, so the model
    uses ``reasoning_default`` (or its own default when that is ``None``).
    """

    alias: str = ""
    context_window: int | None = None
    reasoning_level: str | None = None
    reasoning_levels: list[str] = Field(default_factory=list)
    reasoning_default: str | None = None


PlanStatus = Literal["pending", "in_progress", "completed"]
PlanPriority = Literal["high", "medium", "low"]


class PlanEntry(BaseModel):
    """One step of the agent's plan, as a host shows it (``Session.plan()``).

    The fields are ACP's plan entry: ``content`` (the step), ``status`` and
    ``priority``. The coding agent derives them from its todos.
    """

    content: str
    status: PlanStatus = "pending"
    priority: PlanPriority = "medium"


class CommandResult(BaseModel):
    """What a slash command returned: text to show, and data for clients that know it.

    ``output_to_agent`` says whether the command meant its output for the
    agent; the host decides whether to send it on.
    """

    text: str
    output_to_agent: bool = False
    data: dict[str, Any] | None = None


class _Update(BaseModel):
    session_id: str


class TurnStartedUpdate(_Update):
    """A turn started; ``item_ids`` are the admitted items it carries."""

    kind: Literal["turn_started"] = "turn_started"
    item_ids: list[str] = Field(default_factory=list)


class TurnEndedUpdate(_Update):
    """A turn ended.

    ``outcome`` is the outcome as data (``Done``/``Waiting`` fields; for a
    question: ``question``, ``options``, ``answer_schema``; for a cancel:
    ``by``; for an error: ``error``). ``result_type`` is the
    ``module:qualname`` of a pydantic ``Done.result``, so a receiver can
    rebuild it. ``usage`` is this turn's own delta.
    """

    kind: Literal["turn_ended"] = "turn_ended"
    outcome_kind: Literal["done", "need_input", "waiting", "cancelled", "error"]
    outcome: dict[str, Any] = Field(default_factory=dict)
    result_type: str | None = None
    usage: Usage = Field(default_factory=Usage)


class ItemAdmittedUpdate(_Update):
    """An item was admitted on a channel."""

    kind: Literal["item_admitted"] = "item_admitted"
    channel: str
    item_id: str
    source: str
    preview: str = ""
    text: str = ""  # the whole item: the string itself, else its JSON


class ItemConsumedUpdate(_Update):
    """An admitted item was taken: by the turn loop, by agent code, or as a steer a model call saw."""

    kind: Literal["item_consumed"] = "item_consumed"
    channel: str
    item_id: str


class CancelledUpdate(_Update):
    """A running turn was cancelled."""

    kind: Literal["cancelled"] = "cancelled"
    by: str
    interrupted: str | None = None


class TitleChangedUpdate(_Update):
    """The session's title changed."""

    kind: Literal["title_changed"] = "title_changed"
    title: str
    user_set: bool


class ModeChangedUpdate(_Update):
    """The session's permission mode changed."""

    kind: Literal["mode_changed"] = "mode_changed"
    mode: str


class ModelChangedUpdate(_Update):
    """The session's model alias changed; the new client applies from the next turn."""

    kind: Literal["model_changed"] = "model_changed"
    model: str


class ReasoningChangedUpdate(_Update):
    """The reasoning level changed (``set_reasoning``); it applies from the next model call."""

    kind: Literal["reasoning_changed"] = "reasoning_changed"
    level: str | None = None


class CommandsChangedUpdate(_Update):
    """The slash commands the session offers changed (a skill loaded, a control added).

    ``commands`` is the new list, as ``Session.commands()`` returns it.
    """

    kind: Literal["commands_changed"] = "commands_changed"
    commands: list[CommandInfo]


class ChildCreatedUpdate(_Update):
    """A child session was created under this one."""

    kind: Literal["child_created"] = "child_created"
    child_id: str
    name: str | None
    depth: int
    retained: bool


class UsageChangedUpdate(_Update):
    """The session's usage changed (a model call, or a child's attributed usage).

    ``usage`` is the new total.
    """

    kind: Literal["usage_changed"] = "usage_changed"
    usage: Usage


class AgentEventUpdate(_Update):
    """One event from the session's agent: tool calls, cell output, messages, model responses.

    Every event the agent's event manager handles is forwarded, runtime
    events (terminal output, file edits) included, in the order they
    happen and ordered with the other updates: a model response arrives
    before the ``UsageChangedUpdate`` it causes. ``event`` is the event
    itself (a pydantic value; listeners must not change it) and
    serialises as its own type.
    """

    kind: Literal["agent_event"] = "agent_event"
    event: SerializeAsAny[EventBase]


class ClosedUpdate(_Update):
    """The session closed."""

    kind: Literal["closed"] = "closed"


SessionEvent = Annotated[
    TurnStartedUpdate
    | TurnEndedUpdate
    | ItemAdmittedUpdate
    | ItemConsumedUpdate
    | CancelledUpdate
    | TitleChangedUpdate
    | ModeChangedUpdate
    | ModelChangedUpdate
    | ReasoningChangedUpdate
    | CommandsChangedUpdate
    | ChildCreatedUpdate
    | UsageChangedUpdate
    | AgentEventUpdate
    | ClosedUpdate,
    Field(discriminator="kind"),
]
"""What ``Session.subscribe()`` listeners receive: data only."""
