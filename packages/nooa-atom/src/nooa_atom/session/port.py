# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent-side port: what an agent can do with its own Session (``self.session``)."""

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any

from nooa import hidden
from nooa.agentdoc import spec
from nooa.interactive import Done, InteractiveAgent
from nooa_atom.session.items import (
    ChildFailedError,
    ChildRef,
    ChildStatus,
    Receipt,
    SessionInfo,
    Usage,
)
from nooa_atom.session.options import ChildMode
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
    ``self.v`` across turns and sessions.

    - Wait for the answer in the same cell::

          child = await self.session.delegate("Review auth", prompt)
          done = await child.wait()          # the child's Done; done.result may be None

    - Or let it run in the background and end your turn with
      ``Waiting(explanation=..., on=["delegates"])``. Its result arrives in
      a later notification on the ``delegates`` channel as a
      ``ChildResult`` (``item.done``), a ``ChildQuestion`` (answer with
      ``await item.answer(...)``) or a ``ChildFailed`` (``item.error``).
      Each ``self.message()`` of the child arrives before them as a
      ``ChildMessage`` (``item.text``); it does not end ``child.wait()``.

    Two separate choices shape a child. ``mode="worker"`` (the default)
    runs ``handle_work`` turns: written for a parent reader, each ``Done``
    carries a ``TaskResult``, and it may ask you a question.
    ``mode="conversation"`` runs the interactive ``handle()``. ``keep=False``
    (the default) closes it after its first result; ``keep=True`` keeps it
    for more messages (``await child.send(...)``) until it is closed or its
    tree ends. A closed child stays closed: ``send()`` raises.

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

    async def delegate(
        self,
        description: str,
        prompt: str,
        *,
        context: Any = None,
        model: str | None = None,
        mode: ChildMode = "worker",
        keep: bool = False,
        permission_mode: str | None = None,
        **removed: Any,
    ) -> ChildRef:
        """Create a child session that works on ``prompt``; return its ``ChildRef``.

        Args:
            description: Short label; becomes the child's name and title.
            prompt: The full task the child works from.
            context: Optional data (a pydantic model or JSON data) the child
                receives with the prompt on its ``context`` channel.
            model: Model alias for the child; the same as yours when omitted.
            mode: ``"worker"`` (the default) runs ``handle_work`` turns that
                report with a ``TaskResult`` and may ask you a question;
                ``"conversation"`` runs the interactive ``handle()``.
            keep: ``False``: closed after its first result. ``True``: stays
                open for more messages until it is closed or its tree ends.
            permission_mode: Permission mode for the child; yours when omitted.
        """
        _reject_removed("delegate", removed)
        parent = self._session
        options = parent.options.inherit(
            name=description,
            model=model,
            permission_mode=permission_mode,
            child_mode=mode,
            keep=keep,
            turn_method=_turn_method(self._session, mode),
        )
        items: list[tuple[str, Any]] = [("user_messages", prompt)]
        if context is not None:
            items.append(("context", as_data(context)))
        child = await self._registry.create(
            options,
            parent_id=parent.id,
            initial_items=items,
            initial_source=_parent_source(parent),
        )
        return self._registry.child_ref(child)

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
    async def wait_child(self, child_id: str) -> Done:
        """Next ``Done`` of a child; raises ``ChildFailedError`` on failure or close.

        Also raises ``ChildFailedError`` when ``child_id`` is not this
        session's child.
        """
        registry = self._registry
        queued = registry.take_queued_result(self._session, child_id)
        if queued is not None:
            return queued
        if registry.get(child_id) is None:
            raise ChildFailedError(f"Child {child_id!r} is not running")
        waiter = registry.waiter(self._session, child_id)
        try:
            return await asyncio.shield(waiter)
        except asyncio.CancelledError:
            # The parent's turn was cancelled: the child's result must not
            # vanish into an orphaned future; it goes to delegates instead.
            registry.drop_waiter(self._session, child_id, waiter)
            raise

    @hidden
    async def send_child(self, child_id: str, item: Any, *, channel: str) -> Receipt:
        """Admit ``item`` on one of a child's channels (loading the child if needed)."""
        child = await self._registry.open_child(self._session, child_id)
        return await child.submit(
            as_data(item), channel=channel, source=_parent_source(self._session)
        )

    @hidden
    async def steer_child(self, child_id: str, text: str) -> Receipt:
        """Steer a child's running turn (a message if it is idle)."""
        child = await self._registry.open_child(self._session, child_id)
        return await child.steer(text, source=_parent_source(self._session))

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
        """A child's metadata."""
        self._registry.check_owner(self._session, child_id)
        return self._registry.info(child_id)


def _reject_removed(method: str, removed: dict[str, Any]) -> None:
    """Raise a clear ``TypeError`` for ``retain=`` (removed) or any unknown keyword."""
    if "retain" in removed:
        raise TypeError(
            f"{method}() no longer takes retain=: choose the turn type with "
            "mode='worker' (the default) or mode='conversation', and the lifetime with "
            "keep=True (stays open for follow-ups) or keep=False (closed after its result)"
        )
    if removed:
        raise TypeError(f"{method}() got an unexpected keyword argument {next(iter(removed))!r}")


def _turn_method(parent: Session, mode: ChildMode) -> str:
    """The child's turn method, from its mode only (never from its lifetime).

    A worker runs ``handle_work``; an agent class without one runs workers
    with ``handle_batch``.
    """
    if mode == "conversation":
        return "handle"
    if mode != "worker":
        raise ValueError(f"mode must be 'worker' or 'conversation', not {mode!r}")
    return (
        "handle_work" if callable(getattr(parent._agent, "handle_work", None)) else "handle_batch"
    )


def _parent_source(session: Session) -> str:
    return f"parent:{session.name or session.id}"


def install_port(
    agent: InteractiveAgent,
    session: Session,
    registry: "SessionRegistry",
    *,
    visible: bool | None = None,
) -> SessionPort:
    """Install ``self.session`` on the agent and the channels delegation uses.

    Registers ``delegates`` (child results) and, on a child, ``context``
    (data from the parent) when the agent has not, and makes the loop set
    ``current_port`` so ``ChildRef`` methods work from any cell.

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
    channels = agent.queue_manager.channels()
    if "delegates" not in channels:
        agent.queue_manager.queue("delegates")
    if session.parent_id is not None and "context" not in channels:
        agent.queue_manager.queue("context")
    session.add_loop_context_hook(lambda: current_port.set(port))
    return port
