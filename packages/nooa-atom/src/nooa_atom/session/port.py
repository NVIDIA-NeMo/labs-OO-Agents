# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent-side port: what an agent can do with its own Session (``self.session``)."""

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any

from nooa import hidden
from nooa.agentdoc import spec
from nooa.interactive import InteractiveAgent
from nooa_atom.session.items import (
    ChildFailedError,
    ChildQuestion,
    ChildRef,
    ChildResult,
    ChildStatus,
    Receipt,
    SessionInfo,
    Usage,
)
from nooa_atom.session.options import reject_removed_options
from nooa_atom.session.session import Session, as_data

if TYPE_CHECKING:
    from nooa_atom.session.registry import SessionRegistry

current_port: contextvars.ContextVar["SessionPort | None"] = contextvars.ContextVar(
    "nooa_atom_current_port", default=None
)
"""The port of the session whose turn is running; set once at the top of each session loop."""


def require_port() -> "SessionPort":
    """The running turn's port; ``ChildRef`` methods resolve it here."""
    port = current_port.get()
    if port is None:
        raise RuntimeError("ChildRef methods can only be used inside a session's turn")
    return port


class SessionPort:
    """Your own session: delegate work to child sessions and manage them.

    A child is another agent session with its own history. You give it a
    short ``description`` (its name and title) and the full ``prompt`` it
    works from, and get back a ``ChildRef``: a small handle you can keep in
    ``self.v`` across turns and sessions. Every child runs the worker turn
    (``handle_work``): each ``Done`` carries a ``TaskResult``.

    - ``kept=False`` (the default) is a one-shot worker: it may not ask
      questions and is closed after its result.
    - ``kept=True`` keeps it: it may ask you questions (``ChildQuestion``,
      answered with ``await item.answer(...)``), takes follow-ups with
      ``await child.send(...)``, and stays open until ``close()`` or the end
      of its tree. A closed child stays closed: ``send()`` raises.

    ``await child.wait()`` returns its next ``ChildResult`` or
    ``ChildQuestion``, and raises ``ChildFailedError`` when it fails, is
    closed, or could never finish (idle with nothing queued, or waiting for
    your answer). Otherwise end your turn with
    ``Waiting(explanation=..., on=["delegates"])``: the outcome arrives on
    ``delegates``, after any ``ChildMessage`` progress notes. An outcome goes
    to one place only: a waiting ``wait()``, else ``delegates``.

    Never predict or make up a pending child's result: end the turn and act
    on the item when it arrives.
    """

    __nosnapshot__ = True

    def __init__(self, session: Session, registry: "SessionRegistry") -> None:
        self._session = session
        self._registry = registry

    @property
    def id(self) -> str:
        """This session's id."""
        return self._session.id

    @property
    def depth(self) -> int:
        """This session's depth in the tree (0 for a root)."""
        return self._session.depth

    @property
    def max_depth(self) -> int:
        """The deepest a child may be; delegating beyond it fails."""
        return self._session.options.max_depth

    @property
    def may_ask(self) -> bool:
        """Whether this session may end a turn with a question.

        A root asks its person; a kept child asks its parent. A one-shot
        worker may not: nobody would answer it.
        """
        session = self._session
        return session.parent_id is None or session.options.kept

    async def delegate(
        self,
        description: str,
        prompt: str,
        *,
        model: str | None = None,
        kept: bool = False,
        permission_mode: str | None = None,
        todo: Any = None,
        **removed: Any,
    ) -> ChildRef:
        """Create a child session that works on ``prompt``; return its ``ChildRef``.

        Args:
            description: Short label; becomes the child's name and title.
            prompt: The full task the child works from.
            model: Model alias for the child; the same as yours when omitted.
            kept: ``False``: a one-shot worker, closed after its first result,
                that may not ask questions. ``True``: it may ask, takes
                follow-ups, and stays open until it is closed or its tree ends.
            permission_mode: Permission mode for the child; yours when omitted.
            todo: A ``Todo`` the child works on: a copy of it becomes the
                active Todo in the child's own list (the agent's
                ``receive_delegated_todo``), the ref records its id, and the
                copy is recorded as the base its changes merge back against.
        """
        _reject_removed("delegate", removed)
        parent = self._session
        options = parent.options.inherit(
            name=description,
            model=model,
            permission_mode=permission_mode,
            kept=kept,
            turn_method=_worker_turn_method(parent, kept),
            todo_id=getattr(todo, "id", None),
        )
        registry = self._registry

        async def hand_over(child: Session) -> None:
            registry.record_todo_base(parent, child.id, todo)
            receive = getattr(child._agent, "receive_delegated_todo", None)
            if callable(receive):
                receive(todo)

        child = await registry.create(
            options,
            parent_id=parent.id,
            initial_items=[("user_messages", prompt)],
            initial_source=_parent_source(parent),
            prepare=hand_over if todo is not None else None,
        )
        return registry.child_ref(child)

    @property
    def name(self) -> str:
        """This session's name, else its title, else ``"root"`` for an unnamed root."""
        session = self._session
        return session.name or session.info.title or ("root" if session.parent_id is None else "")

    def children(self) -> list[ChildRef]:
        """Handles on your children, live and closed."""
        return [
            ChildRef(
                id=info.id,
                name=info.name or "",
                depth=info.depth,
                status=info.status,
                parent_id=info.parent_id,
                todo_id=info.todo_id,
            )
            for info in self._registry.children(self._session.id)
        ]

    async def rename(self, title: str) -> None:
        """Set this session's title (a title the person set is kept)."""
        await self._session.set_title(title, user_set=False)

    def usage(self) -> Usage:
        """Tokens and cost so far: your own and your children's (attributed)."""
        return self._session.info.usage.model_copy()

    # ---- what ChildRef methods call ----------------------------------

    @hidden
    async def wait_child(self, child_id: str) -> ChildResult | ChildQuestion:
        """A child's next outcome (``ChildRef.wait()``); fails fast when it could never come.

        In order: a queued outcome is withdrawn and returned; a closed child
        fails; a busy child (running, ``working``, or with input queued) is
        waited for; otherwise the child waits for this session's answer or is
        idle, and ``ChildFailedError`` says which.
        """
        registry = self._registry
        queued = registry.take_queued_result(self._session, child_id)
        if queued is not None:
            return queued
        child = registry.get(child_id)
        if child is None or child.closing or registry.is_closed(self._session.id, child_id):
            question = registry.pending_question(self._session, child_id)
            if child is None and question is not None:
                raise ChildFailedError(_waits_for_answer(child_id, question))
            raise ChildFailedError(f"Child {child_id!r} is not running")
        if registry.is_idle(child):
            question = registry.pending_question(self._session, child_id)
            if question is not None:
                raise ChildFailedError(_waits_for_answer(child.name or child_id, question))
            raise ChildFailedError(
                f"Child {child.name or child_id!r} is idle with nothing queued, so wait() "
                "would never return; send it work with `await ref.send(...)` first"
            )
        waiter = registry.waiter(self._session, child_id)
        try:
            return await asyncio.shield(waiter)
        except asyncio.CancelledError:
            # The parent's turn was cancelled: the child's outcome must not
            # vanish into an orphaned future; it goes to delegates instead.
            registry.drop_waiter(self._session, child_id, waiter)
            raise

    @hidden
    async def send_child(self, child_id: str, item: Any, *, channel: str) -> Receipt:
        """Admit ``item`` on one of a child's channels (loading the child if needed).

        Text on ``user_messages`` steers a running turn (it reads it before
        its next model call) or, when the child is idle, queues it and starts
        a turn. A new turn of the child ends its pending question.
        """
        child = await self._registry.open_child(self._session, child_id)
        source = _parent_source(self._session)
        if channel != "user_messages":
            return await child.submit(as_data(item), channel=channel, source=source)
        # A message answers a pending question, as the turn it starts would.
        self._registry.clear_question(self._session, child_id)
        if isinstance(item, str):
            return await child.steer(item, source=source)
        return await child.submit(as_data(item), channel=channel, source=source)

    @hidden
    async def answer_child(self, child_id: str, question_id: str, answer: Any) -> Receipt:
        """Answer a child's question (``ChildQuestion.answer()``); it is no longer pending."""
        pending = self._registry.pending_question(self._session, child_id)
        if pending is None or pending.id != question_id:
            current = "it has none" if pending is None else f"it is {pending.id!r}"
            raise ChildFailedError(
                f"Question {question_id!r} is not the child's pending question ({current})"
            )
        return await self.send_child(child_id, answer, channel="user_messages")

    @hidden
    def pending_question(self, child_id: str) -> ChildQuestion | None:
        """The child's unanswered question (shown in a ``ChildRef``'s repr)."""
        return self._registry.pending_question(self._session, child_id)

    @hidden
    async def close_child(self, child_id: str) -> None:
        """Close a child session."""
        await self._registry.close_child(self._session, child_id)

    @hidden
    def child_status(self, child_id: str) -> ChildStatus:
        """A child's live status (``ChildRef.status``)."""
        return self._registry.child_status(self._session, child_id)

    @hidden
    def info(self) -> SessionInfo:
        """This session's metadata, with its live status."""
        return self._registry.live_info(self._session)

    @hidden
    def child_info(self, child_id: str) -> SessionInfo:
        """A child's metadata (for hosts and tests; not on ``ChildRef``)."""
        self._registry.check_owner(self._session, child_id)
        return self._registry.info(child_id)


def _waits_for_answer(name: str, question: ChildQuestion) -> str:
    return (
        f"Child {name!r} is waiting for your answer to [{question.id}] {question.question!r}; "
        "answer that ChildQuestion with `await question.answer(...)` before waiting"
    )


def _reject_removed(method: str, removed: dict[str, Any]) -> None:
    """Raise a one-line ``TypeError`` for a removed keyword (``retain=``, ``mode=``, ``keep=``)."""
    if not removed:
        return
    reject_removed_options(f"{method}()", removed)
    raise TypeError(f"{method}() got an unexpected keyword argument {next(iter(removed))!r}")


def _worker_turn_method(parent: Session, kept: bool) -> str:
    """A child's turn method: the worker turn ``handle_work``.

    An agent class without one runs a one-shot worker with ``handle_batch``
    and a kept child, which may ask, with ``handle``.
    """
    if callable(getattr(parent._agent, "handle_work", None)):
        return "handle_work"
    return "handle" if kept else "handle_batch"


def _parent_source(session: Session) -> str:
    return f"parent:{session.name or session.id}"


def install_port(
    agent: InteractiveAgent,
    session: Session,
    registry: "SessionRegistry",
    *,
    visible: bool | None = None,
) -> SessionPort:
    """Install ``self.session`` on the agent and the channel delegation uses.

    Registers ``delegates`` (child results) when the agent has not, and
    makes the loop set ``current_port`` so ``ChildRef`` methods work from
    any cell.

    The port is in the model docs unless ``visible=False``; when
    ``visible`` is not given, the agent class decides with a
    ``session_port_visible: ClassVar[bool]`` attribute (True when absent),
    so an agent that offers delegation through its own methods can hide
    the port without the host passing anything.
    """
    if visible is None:
        visible = bool(getattr(type(agent), "session_port_visible", True))
    port = SessionPort(session, registry)
    agent.session = port  # type: ignore[attr-defined]
    if not visible:
        spec(agent, "session", hidden=True)
    if "delegates" not in agent.queue_manager.channels():
        agent.queue_manager.queue("delegates")
    session.add_loop_context_hook(lambda: current_port.set(port))
    return port
