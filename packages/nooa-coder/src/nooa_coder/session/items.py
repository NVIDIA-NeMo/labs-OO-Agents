# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data that crosses the Session boundary.

Everything here is a pydantic model with data fields only: no callables,
no live objects. Sessions, hosts and parent agents exchange these values;
live agents never cross.
"""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field

from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role
from nooa.interactive import Done, NeedInput

SessionStatus = Literal["running", "idle", "retained", "closed", "on_disk"]
ChildStatus = Literal["running", "idle", "retained", "closed"]


class TaskResult(BaseModel):
    """Structured result of a delegated objective or a benchmark task."""

    solution_description: str = Field(description="What was done")
    evidence: str = Field(description="What shows that it works")
    how_to_verify: str = Field(description="How someone else can check it")
    report: str = Field(default="", description="Optional longer report")


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
    directly (``item.child.send(...)``).
    """

    id: str
    name: str
    depth: int
    status: ChildStatus


class ChildFailedError(RuntimeError):
    """Raised by ``ChildRef.wait()`` when the child fails or is closed before a ``Done``."""


class ChildResult(BaseModel):
    """A child finished a turn with ``Done``; delivered on the parent's ``delegates`` channel."""

    child: ChildRef
    done: Done


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


class ChildFailed(BaseModel):
    """A child's turn failed or the child ended without a result."""

    child: ChildRef
    error: str


class TurnCancelledOutcome(BaseModel):
    """What ``prompt()`` returns when the turn it waited for was cancelled."""

    by: str


class TranscriptEntry(BaseModel):
    """One line of a session's transcript as a host shows it."""

    role: Literal["user", "agent", "question", "cancelled"]
    content: str
    item_id: str | None = None
    timestamp: float


class Usage(BaseModel):
    """Token and cost totals: the session's own and those attributed from its children."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attributed_input_tokens: int = 0
    attributed_output_tokens: int = 0
    attributed_cost_usd: float = 0.0


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
    mode: str = "auto"
    status: SessionStatus = "on_disk"
    retained: bool = False
    created_at: float = 0.0
    last_active: float = 0.0
    turn_count: int = 0
    usage: Usage = Field(default_factory=Usage)


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


class ChildCreatedUpdate(_Update):
    """A child session was created under this one."""

    kind: Literal["child_created"] = "child_created"
    child_id: str
    name: str | None
    depth: int
    retained: bool


class ClosedUpdate(_Update):
    """The session closed."""

    kind: Literal["closed"] = "closed"


class AgentEventUpdate(_Update):
    """The agent added an event; read it from the agent's events by id."""

    kind: Literal["agent_event"] = "agent_event"
    event_id: str
    event_type: str


SessionEvent = Annotated[
    TurnStartedUpdate
    | TurnEndedUpdate
    | ItemAdmittedUpdate
    | CancelledUpdate
    | TitleChangedUpdate
    | ModeChangedUpdate
    | ChildCreatedUpdate
    | ClosedUpdate
    | AgentEventUpdate,
    Field(discriminator="kind"),
]
"""What ``Session.subscribe()`` listeners receive: data only."""


class TurnCancelled(EventBase):
    """A person, a parent or the host stopped the turn before it finished.

    Appended to the agent's events when a cancel takes effect, after the
    interrupted cell's output, so the model sees at its next turn that it
    was stopped rather than that a cell failed. ``by`` says who stopped it
    (``"user"``, ``"parent:<name>"``, ``"host"``); ``interrupted`` is the
    tag of the interrupted cell's ``PythonOutput``, or ``None`` when the
    turn was stopped between cells (for example during a model call).
    """

    _role: ClassVar[Role] = Role.USER

    by: str
    interrupted: str | None = None
