# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data that crosses the Session boundary.

Everything here is a pydantic model with data fields only: no callables,
no live objects. Sessions, hosts and parent agents exchange these values;
live agents never cross.
"""

import hashlib
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from nooa.agentdoc import spec
from nooa.context_blocks import EventBase
from nooa.interactive import Done, NeedInput, TaskResult  # noqa: F401  (TaskResult re-exported)
from nooa.runtime.turn_loop import TurnCancelled  # noqa: F401  (re-exported)

if TYPE_CHECKING:
    from nooa_atom.session.port import SessionPort

SessionStatus = Literal["running", "working", "idle", "retained", "closed", "on_disk"]
"""``running``: a turn is running. ``working``: its last turn ended ``Waiting``
on a job of its own. ``idle``: live, nothing to do. ``retained``: an idle kept
child (``spawn()``). ``closed``: closed for good. ``on_disk``: not loaded in
this process."""
ChildStatus = SessionStatus  # a child reports the same statuses as any session


class Receipt(BaseModel):
    """Returned by ``submit()``, ``steer()`` and ``ChildRef.send()``: where an item went and its id.

    ``delivered`` is ``queued`` when the item waits on a channel for a
    turn, ``steered`` when it was handed to the running turn's next model
    call.
    """

    session_id: str
    channel: str
    item_id: str
    delivered: Literal["queued", "steered"]


class ChildRef(BaseModel):
    """A handle on a child session: data plus ``wait()``, ``send()`` and ``close()``.

    It holds the child's id, name, path (``/root/review_pr_462``) and a few
    facts, never the child itself. The methods find the parent's port at
    call time from the running turn, so a ``ChildRef`` kept in ``self.v``
    still works after a checkpoint and a reload, and one that arrives in a
    ``ChildResult`` can be acted on directly (``item.child.send(...)``).
    Only the child's parent can control it (``send()``, ``close()``):
    ``parent_id`` is that parent. An ancestor further up may read its
    ``status`` and ``wait()`` for it; other sessions get
    ``ChildFailedError``. Its repr shows a question of the child's that
    you have not answered.
    """

    id: str
    name: str
    path: str = ""
    """Where it is in the session tree: its parent's path, ``/`` and its name."""
    depth: int
    status: ChildStatus
    parent_id: str | None = None
    todo_id: str | None = None
    """The id of the Todo this child was briefed on (provenance only; nothing flows back)."""

    def __getattribute__(self, name: str) -> Any:
        # ``status`` is live in the parent's turn (read from the registry);
        # elsewhere, or when the registry no longer knows the child, it is
        # the stored value, the status when the ref was made. Dumps,
        # snapshots and comparisons read the stored field, so they are stable.
        if name == "status":
            port = _reader_port(self)
            if port is not None:
                with suppress(ChildFailedError):
                    return port.child_status(object.__getattribute__(self, "id"))
        return super().__getattribute__(name)

    def __snapshot_data__(self) -> dict[str, Any]:
        """The stored fields for a snapshot; never asks the port for the live status."""
        return self.model_dump(mode="json")

    @classmethod
    def __restore_snapshot__(cls, data: dict[str, Any]) -> "ChildRef":
        """Rebuild a ref from a snapshot: ``status``, or ``status_when_made`` in older ones."""
        known = {key: value for key, value in data.items() if key in cls.model_fields}
        if "status" not in known and "status_when_made" in data:
            known["status"] = data["status_when_made"]
        return cls.model_validate(known)

    def __repr_args__(self) -> Any:
        for name, value in super().__repr_args__():
            if name == "id":
                continue  # the uuid stays on the ref; the repr shows name and path
            yield (name, self.status) if name == "status" else (name, value)
        port = _parent_port(self)
        question = None
        if port is not None:
            with suppress(ChildFailedError):
                question = port.pending_question(self.id)
        if question is not None:
            yield "pending_question", f"[{question.id}] {question.question}"

    async def wait(self) -> "ChildResult | ChildQuestion":
        """Wait for the child's next ``ChildResult`` or ``ChildQuestion`` and return it.

        Raises ``ChildFailedError`` when the child fails or is closed, and at
        once when waiting could never end: the child is idle with nothing
        queued, or it waits for your answer. A ``ChildMessage`` does not end it.
        """
        return await _port(self, control=False).wait_child(self.id)

    async def send(self, item: Any, *, channel: str = "user_messages") -> "Receipt":
        """Send the child a message: a running turn reads it next, an idle child starts a turn."""
        return await _port(self).send_child(self.id, item, channel=channel)

    async def close(self) -> None:
        """Close the child session for good (its children first)."""
        await _port(self).close_child(self.id)


def question_id(child_id: str, reply: int) -> str:
    """The id of the question a child asked in its ``reply``-th reply.

    ``reply`` counts the child's replies as ``SessionInfo.reply_count`` does
    (its messages and its ended turns). The child's session stores it in the
    ``need_input`` outcome, so the live question and the one read back from
    the record after a reload have the same id.
    """
    return hashlib.sha1(f"{child_id}:{reply}".encode()).hexdigest()[:8]


def _parent_port(ref: ChildRef) -> "SessionPort | None":
    """The running turn's port when it is ``ref``'s parent, else ``None``."""
    from nooa_atom.session.port import current_port

    port = current_port.get()
    if port is None or (ref.parent_id is not None and ref.parent_id != port.id):
        return None
    return port


def _reader_port(ref: ChildRef) -> "SessionPort | None":
    """The running turn's port when it is ``ref``'s parent or an ancestor further up."""
    from nooa_atom.session.port import current_port

    port = current_port.get()
    if port is None or ref.parent_id is None or ref.parent_id == port.id:
        return port
    return port if port.is_ancestor_of(ref.id) else None


def _port(ref: ChildRef, *, control: bool = True) -> "SessionPort":
    """The running turn's port, if its session is ``ref``'s parent.

    With ``control=False`` (``wait()``), an ancestor further up will do too:
    it may watch a deeper session, but only its parent controls it.
    """
    from nooa_atom.session.port import require_port

    port = require_port()
    if ref.parent_id is None or ref.parent_id == port.id:
        return port
    if port.is_ancestor_of(ref.id):
        if not control:
            return port
        raise ChildFailedError(
            f"{ref.path or ref.id} is not your child but a deeper session: you can read "
            "its status and wait() for it; control it through its parent"
        )
    raise ChildFailedError(
        f"Child {ref.path or ref.id!r} belongs to session {ref.parent_id!r}, not to {port.id!r}"
    )


class ChildFailedError(RuntimeError):
    """A child's turn failed, it is closed, or waiting for it could never end."""


class ChildResult(BaseModel):
    """A child finished a turn with ``Done``; read ``item.report`` for its full report, ``item.result`` for its ``TaskResult``."""

    name: str = Field("", description="the child's name")
    path: str = Field("", description="the child's path in the session tree, else its name")
    summary: str = Field("", description="the child's Done.explanation: one status line")
    report: Annotated[str, spec(max_string=1000)] = Field(
        "", description="the full report: result.report, else Done.message"
    )
    result: Annotated[TaskResult | None, spec(max_string=300)] = Field(
        None, description="the child's TaskResult (always one for a worker), else None"
    )
    evidence: Annotated[list[str], spec(max_length=5, max_string=300)] = Field(
        default_factory=list, description="the child's Done.evidence"
    )
    child: ChildRef = Field(description="handle on the child: await item.child.send(...)")
    done: Annotated[Done, spec(hidden=True)]
    """The child's ``Done`` as it arrived. The fields above are derived from it
    and ``child`` on validation and left out of dumps, so a record holds
    ``child`` and ``done`` only."""

    @field_validator("done", mode="after")
    @classmethod
    def _task_result(cls, done: Done) -> Done:
        # A result that arrives as data (a child that returned a dict, or one
        # reloaded from the record) is rebuilt as a TaskResult when it is one.
        if isinstance(done.result, dict):
            try:
                result = TaskResult.model_validate(done.result)
            except ValidationError:
                return done
            done = done.model_copy(update={"result": result})
        return done

    @model_serializer(mode="wrap")
    def _dump(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # A dump keeps the source of truth only; validation derives the rest.
        return {key: value for key, value in handler(self).items() if key in ("child", "done")}

    @model_validator(mode="after")
    def _flatten(self) -> "ChildResult":
        done = self.done
        result = done.result if isinstance(done.result, TaskResult) else None
        self.__dict__.update(
            name=self.child.name,
            path=self.child.path or self.child.name,
            summary=done.explanation,
            report=(result.report if result is not None and result.report else None)
            or done.message
            or "",
            result=result,
            evidence=list(done.evidence),
        )
        return self

    def status_source(self) -> str:
        """Who sent it, for the per-turn ``Unread:`` count (never its content)."""
        return self.path


class ChildQuestion(BaseModel):
    """A kept child asked you a question (its ``NeedInput``); reply with ``await item.child.send(...)``."""

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="names the question until any send() to the child or its next turn",
    )
    name: str = Field("", description="the child's name")
    path: str = Field("", description="the child's path in the session tree, else its name")
    question: Annotated[str, spec(max_string=2000)] = Field(description="the question")
    reason: str | None = Field(None, description="why the child asks")
    child: ChildRef = Field(description="handle on the child: await item.child.send(...)")

    @model_validator(mode="after")
    def _flatten(self) -> "ChildQuestion":
        self.__dict__.update(name=self.child.name, path=self.child.path or self.child.name)
        return self

    @classmethod
    def from_need_input(cls, child: ChildRef, need: NeedInput) -> "ChildQuestion":
        """Convert a child's ``NeedInput``."""
        return cls(child=child, question=need.question, reason=need.reason)

    def status_source(self) -> str:
        """Who sent it, for the per-turn ``Unread:`` count (never its content)."""
        return self.path


class ChildFailed(BaseModel):
    """A child's turn failed or the child ended without a result; read ``item.error``."""

    name: str = Field("", description="the child's name")
    path: str = Field("", description="the child's path in the session tree, else its name")
    error: Annotated[str, spec(max_string=2000)] = Field(description="what went wrong")
    child: ChildRef = Field(description="handle on the child")

    @model_validator(mode="after")
    def _flatten(self) -> "ChildFailed":
        self.__dict__.update(name=self.child.name, path=self.child.path or self.child.name)
        return self

    def status_source(self) -> str:
        """Who sent it, for the per-turn ``Unread:`` count (never its content)."""
        return self.path


# Longest ``self.message()`` text a child's parent receives, in characters.
# A progress note is short; the full report belongs in the child's result.
CHILD_MESSAGE_CHARS = 1_000
CHILD_MESSAGE_CUT = "\n[cut: the full report belongs in the result]"


class ChildMessage(BaseModel):
    """A progress note a child sent with ``self.message()``, not its result; read ``item.text``."""

    name: str = Field("", description="the child's name")
    path: str = Field("", description="the child's path in the session tree, else its name")
    text: Annotated[str, spec(max_string=2000)] = Field(description="the message")
    child: ChildRef = Field(description="handle on the child: await item.child.send(...)")

    @model_validator(mode="after")
    def _flatten(self) -> "ChildMessage":
        self.__dict__.update(name=self.child.name, path=self.child.path or self.child.name)
        return self

    def status_source(self) -> str:
        """Who sent it, for the per-turn ``Unread:`` count (never its content)."""
        return self.path


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
    """A kept child (``spawn()``): it stays open after its results."""
    turn_method: str = "handle"
    todo_id: str | None = None
    """The id of the Todo this child was briefed on (provenance only; nothing flows back)."""
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
    ``priority``. The Atom agent derives them from its todos.
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
    question: ``question``, ``reason``; for a cancel:
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
