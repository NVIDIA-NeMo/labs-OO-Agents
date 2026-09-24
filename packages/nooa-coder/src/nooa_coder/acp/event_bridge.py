# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate a Session's updates and its agent's events into ACP session updates.

One bridge per session id per adapter. It listens in two places: the
Session's updates (``subscribe``: titles, modes, turn ends, cancels,
children, admitted items, close) and the agent's own event stream (tool
cells, messages, file edits, terminal commands, model responses), which
carries runtime events the Session's passthrough leaves out. Both are
synchronous, so updates reach the client in the order things happened.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from acp import (
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_agent_thought_text,
    update_plan,
    update_tool_call,
    update_user_message_text,
)
from acp.helpers import plan_entry
from acp.interfaces import Client
from acp.schema import (
    ContentToolCallContent,
    Cost,
    CurrentModeUpdate,
    PlanEntry,
    SessionInfoUpdate,
    ToolCallLocation,
    UsageUpdate,
)

from nooa.agentdoc import pformat
from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from nooa.events import LLMResponse, PythonOutput
from nooa.interactive import AgentMessage
from nooa_coder.coding.activity import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from nooa_coder.session.items import (
    CancelledUpdate,
    ChildCreatedUpdate,
    ItemAdmittedUpdate,
    ModeChangedUpdate,
    SessionInfo,
    TitleChangedUpdate,
    TurnEndedUpdate,
)

# ACP owns stdout for JSON-RPC; diagnostics belong on stderr, which is where
# the logging default sends them.
logger = logging.getLogger(__name__)

_STOP = object()

# Bound on a rendered Out[n] value; large results belong in the agent's
# context, not repeated in full inside a client tool card.
_MAX_VALUE_CHARS = 10_000

OWN_SOURCES = frozenset({"acp", "user:declined"})
"""Item sources this adapter admits itself; the client already shows those."""

_ECHOED_CHANNELS = frozenset({"user_messages", "steer"})

ToolKey = tuple[str, str]
"""An open tool card: (id of the session that ran it, its tool call id)."""


class BridgedSession(Protocol):
    """What the bridge needs from a Session."""

    id: str
    agent: Any
    info: SessionInfo

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]: ...


@dataclass(frozen=True, slots=True)
class _BestEffortUpdate:
    value: Any


def _fenced_code(text: str, language: str) -> str:
    """Wrap text in a Markdown fence that cannot collide with its contents."""
    longest_run = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _python_content(code: str, output: str | None = None) -> list[ContentToolCallContent]:
    """Render Python source and output as Markdown visible in ACP clients."""
    content = [tool_content(text_block(_fenced_code(code, "python")))]
    if output is not None:
        content.append(tool_content(text_block(_fenced_code(output, "text"))))
    return content


def cancel_text(by: str) -> str:
    """What the conversation says when a turn was stopped by ``by``."""
    return "Stopped at your request." if by == "user" else f"Stopped by {by}."


def question_text(question: str, options: list[str] | None) -> str:
    """A ``NeedInput`` question as the agent's final message of the turn."""
    if not options:
        return question
    return question + "\n\n" + "\n".join(f"- {option}" for option in options)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class ACPEventBridge:
    """Send one session's activity to an ACP client, in order, through one pump task.

    ``resolve_child(child_id)`` finds a live child Session; when given, the
    tool cards of this session's children (and theirs) are mirrored into
    this session with ids ``"{child_id}:{tool_call_id}"``.
    """

    def __init__(
        self,
        session: BridgedSession,
        client: Client,
        *,
        resolve_child: Callable[[str], BridgedSession | None] | None = None,
        own_sources: frozenset[str] = OWN_SOURCES,
    ) -> None:
        self.session = session
        self.agent = session.agent
        self.client = client
        self.session_id = session.id
        self._resolve_child = resolve_child
        self._own_sources = own_sources
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._error: Exception | None = None
        # Set when the pump exits on a BaseException it cannot handle. Nothing
        # resolves flush markers after that, so flush must fail rather than wait.
        self._pump_failure: BaseException | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._open_tools: set[ToolKey] = set()
        self._python_source: dict[ToolKey, str] = {}
        self._terminal_output: dict[ToolKey, str] = {}
        self._cost_usd = 0.0
        self._used: int | None = None  # input tokens of the latest model call
        self._plan: list[PlanEntry] = []
        self._children: list[dict[str, Any]] = []
        self._mirrors: dict[str, list[Callable[[], None]]] = {}
        self._unsubscribers: list[Callable[[], None]] = [
            *self._subscribe_agent(self.agent.event_manager, source=None),
            session.subscribe(self._on_session_update),
        ]
        self._pump_task = asyncio.create_task(self._pump(), name="nooa-acp-events")

    # ---- subscriptions -----------------------------------------------

    def _subscribe_agent(
        self, event_manager: Any, *, source: str | None
    ) -> list[Callable[[], None]]:
        """Wire the handlers onto one agent's events.

        ``source`` is ``None`` for this session's own agent, else the id of
        a child whose tool cards are mirrored (cards only: a child's
        messages, thoughts and usage are not this conversation's).
        """
        handlers: list[tuple[str, Callable[[Any], None]]] = [
            ("ToolCallEvent", lambda event: self._on_tool_call(event, source)),
            ("PythonOutput", lambda event: self._on_python_output(event, source)),
            ("FileEdit", lambda event: self._on_file_edit(event, source)),
            ("TerminalCommandStarted", lambda event: self._on_terminal_started(event, source)),
            ("TerminalCommandOutput", lambda event: self._on_terminal_output(event, source)),
            ("TerminalCommandFinished", lambda event: self._on_terminal_finished(event, source)),
        ]
        if source is None:
            handlers += [
                ("AgentMessage", self._on_agent_message),
                ("LLMResponse", self._on_llm_response),
            ]
        return [event_manager.on(event_type, handler) for event_type, handler in handlers]

    def _mirror(self, child: BridgedSession) -> None:
        """Mirror a child's tool cards (and its own children's) into this session."""
        if child.id in self._mirrors or self._closed:
            return

        def on_child_update(update: Any) -> None:
            kind = getattr(update, "kind", None)
            if kind == "child_created" and self._resolve_child is not None:
                grandchild = self._resolve_child(update.child_id)
                if grandchild is not None:
                    self._mirror(grandchild)
            elif kind == "closed":
                self._unmirror(child.id)

        self._mirrors[child.id] = [
            *self._subscribe_agent(child.agent.event_manager, source=child.id),
            child.subscribe(on_child_update),
        ]

    def _unmirror(self, child_id: str) -> None:
        for unsubscribe in self._mirrors.pop(child_id, []):
            unsubscribe()
        self._fail_tools(
            [key for key in self._open_tools if key[0] == child_id],
            "The subagent ended before this finished.",
            title="Unfinished",
        )

    # ---- keys --------------------------------------------------------

    def _key(self, tool_call_id: str, source: str | None) -> ToolKey:
        return (source or self.session_id, tool_call_id)

    def _wire_id(self, key: ToolKey) -> str:
        session_id, tool_call_id = key
        return tool_call_id if session_id == self.session_id else f"{session_id}:{tool_call_id}"

    # ---- queue -------------------------------------------------------

    def _enqueue(self, update: Any) -> None:
        if not self._closed:
            self._queue.put_nowait(update)

    def publish(self, update: Any) -> None:
        """Queue a host-originated session update on the ordered ACP stream."""
        self._enqueue(update)

    def publish_best_effort(self, update: Any) -> None:
        """Queue bootstrap metadata without poisoning the live event stream."""
        self._enqueue(_BestEffortUpdate(update))

    # ---- session updates ---------------------------------------------

    def _on_session_update(self, update: Any) -> None:
        if isinstance(update, ItemAdmittedUpdate):
            if update.channel in _ECHOED_CHANNELS and update.source not in self._own_sources:
                text = getattr(update, "text", "") or update.preview
                self._enqueue(update_user_message_text(text))
        elif isinstance(update, TitleChangedUpdate):
            self._enqueue(
                SessionInfoUpdate(
                    session_update="session_info_update",
                    title=update.title,
                    updated_at=datetime.now(UTC).isoformat(),
                )
            )
        elif isinstance(update, ModeChangedUpdate):
            self._enqueue(
                CurrentModeUpdate(session_update="current_mode_update", current_mode_id=update.mode)
            )
        elif isinstance(update, CancelledUpdate):
            # Before the prompt's future resolves: the Session emits this first,
            # so the cards are closed before the prompt answers "cancelled".
            self.fail_open_tools(cancel_text(update.by), title="Cancelled")
            self._enqueue(update_agent_message(text_block(cancel_text(update.by))))
        elif isinstance(update, TurnEndedUpdate):
            self._on_turn_ended(update)
        elif isinstance(update, ChildCreatedUpdate):
            self._on_child_created(update)
        elif getattr(update, "kind", None) == "usage_changed":
            self._publish_usage()
        elif getattr(update, "kind", None) == "closed":
            self._close_task = self._close_task or asyncio.ensure_future(self._close())

    def _on_turn_ended(self, update: TurnEndedUpdate) -> None:
        if update.outcome_kind == "need_input":
            # Rendered once per turn, here; the prompt that owns the consumed
            # item decides whether to also open a form.
            question = str(update.outcome.get("question", ""))
            options = update.outcome.get("options")
            self._enqueue(update_agent_message(text_block(question_text(question, options))))
        elif update.outcome_kind == "error":
            # A turn ending on an error does not always write the PythonOutput
            # for a cell it announced; close its card rather than leave it
            # spinning.
            self.fail_open_tools("Did not finish.", title="Unfinished")
        self._publish_plan()

    def _on_child_created(self, update: ChildCreatedUpdate) -> None:
        self._children.append(
            {
                "sessionId": update.child_id,
                "name": update.name,
                "depth": update.depth,
                "retained": update.retained,
            }
        )
        self._enqueue(
            SessionInfoUpdate(
                session_update="session_info_update",
                field_meta={"dev.nooa/children": list(self._children)},
            )
        )
        if self._resolve_child is not None:
            child = self._resolve_child(update.child_id)
            if child is not None:
                self._mirror(child)

    # ---- agent events ------------------------------------------------

    def _on_agent_message(self, event: EventBase) -> None:
        if not isinstance(event, AgentMessage):
            return
        self._enqueue(update_agent_message(text_block(event.content)))

    def _on_tool_call(self, event: EventBase, source: str | None) -> None:
        if (
            not isinstance(event, ToolCallEvent)
            or event.name not in {"execute_python", "python_cell"}
            or event.metadata.get("prefill") is True
            # codeact manufactures an execute_python call to carry a prose-only
            # reply. Nothing ran, so showing it as a Python card would present
            # the model's own text, commented out, as a completed execution.
            or event.metadata.get("synthetic") is True
        ):
            return
        code = event.arguments.get("code", "")
        if not isinstance(code, str):
            code = repr(code)
        key = self._key(event.tool_call_id, source)
        self._open_tools.add(key)
        self._python_source[key] = code
        self._enqueue(
            start_tool_call(
                self._wire_id(key),
                "Running Python",
                # Zed 1.14 treats every ``execute`` tool as a terminal card.
                # A plain-content execute card has neither a terminal nor an
                # output disclosure, so its source cannot be opened. Python is
                # structured Markdown content, not a client-owned terminal.
                kind="other",
                status="in_progress",
                content=_python_content(code),
            )
        )

    def _on_python_output(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, PythonOutput):
            return
        key = self._key(event.tool_call_id, source)
        if key in self._open_tools:
            self._open_tools.discard(key)
            code = self._python_source.pop(key, "")
            parts = [
                part.rstrip() for part in (event.stdout, event.stderr, event.error) if part.strip()
            ]
            # A cell whose last line is a bare expression produces no stdout:
            # the result arrives as ``value`` and codeact shows it to the model
            # as Out[n]. Without this the client is told there was no output
            # while the agent is reasoning from one.
            if event.value is not None:
                rendered = pformat(event.value, max_string=_MAX_VALUE_CHARS, unquote_strings=True)
                parts.append(f"Out[{event.execution_count}]: {rendered}")
            cancelled = event.execution_status is ResultStatus.CANCELLED
            output = "\n".join(parts) or ("Cancelled." if cancelled else "Completed.")
            status: Literal["failed", "completed"] = (
                "failed"
                if event.execution_status in (ResultStatus.ERROR, ResultStatus.CANCELLED)
                else "completed"
            )
            title = (
                "Cancelled"
                if cancelled
                else "Python failed"
                if status == "failed"
                else "Ran Python"
            )
            self._enqueue(
                update_tool_call(
                    self._wire_id(key),
                    title=title,
                    status=status,
                    content=_python_content(code, output),
                )
            )
        if source is None:
            self._publish_plan()

    def _on_file_edit(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, FileEdit):
            return
        tool_call_id = self._wire_id(self._key(f"file-edit-{uuid4()}", source))
        path = event.path
        title = f"{'Created' if event.operation == 'create' else 'Edited'} {Path(path).name}"
        if event.content_complete:
            content = [tool_diff_content(path, event.new_text, event.old_text)]
        else:
            content = [tool_content(text_block(event.diff or "File content was truncated."))]
        line = max(0, event.start_line - 1) if event.start_line is not None else None
        self._enqueue(
            start_tool_call(
                tool_call_id,
                title,
                kind="edit",
                status="completed",
                content=content,
                locations=[ToolCallLocation(path=path, line=line)],
                raw_input={"path": path, "operation": event.operation},
            )
        )

    def _on_terminal_started(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandStarted):
            return
        key = self._key(event.command_id, source)
        self._open_tools.add(key)
        self._terminal_output[key] = ""
        self._enqueue(
            start_tool_call(
                self._wire_id(key),
                f"$ {event.command}",
                kind="execute",
                status="in_progress",
                raw_input={
                    "command": event.command,
                    "working_directory": event.working_directory,
                },
            )
        )

    def _on_terminal_output(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandOutput):
            return
        key = self._key(event.command_id, source)
        if key not in self._open_tools:
            return
        chunk = event.stdout
        if event.stderr:
            chunk += ("\n" if chunk and not chunk.endswith("\n") else "") + event.stderr
        output = self._terminal_output.get(key, "") + chunk
        self._terminal_output[key] = output
        self._enqueue(
            update_tool_call(
                self._wire_id(key),
                status="in_progress",
                content=[tool_content(text_block(output))],
            )
        )

    def _on_terminal_finished(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandFinished):
            return
        key = self._key(event.command_id, source)
        self._open_tools.discard(key)
        output = self._terminal_output.pop(key, "")
        # ACP has no cancelled status, so a stopped command is still "failed" —
        # but it must read as the user's own action, not as a crash.
        reason = "Cancelled by user." if event.cancelled else event.error
        if reason:
            output += ("\n" if output and not output.endswith("\n") else "") + reason
        failed = (
            event.timed_out
            or event.cancelled
            or bool(event.error)
            or (event.exit_code is not None and event.exit_code != 0)
        )
        self._enqueue(
            update_tool_call(
                self._wire_id(key),
                status="failed" if failed else "completed",
                content=[tool_content(text_block(output or "Completed."))],
                raw_output={
                    "exit_code": event.exit_code,
                    "timed_out": event.timed_out,
                    "output_truncated": event.output_truncated,
                },
            )
        )

    def _on_llm_response(self, event: EventBase) -> None:
        if not isinstance(event, LLMResponse):
            return
        reasoning = event.reasoning
        if reasoning:
            self._enqueue(update_agent_thought_text(reasoning))
        usage = event.usage
        if usage is None:
            return
        self._cost_usd += usage.cost_usd
        self._used = usage.input_tokens
        self._publish_usage()

    # ---- derived updates ---------------------------------------------

    def _publish_usage(self) -> None:
        """Context use and cost (own plus what children spent) as a usage update."""
        context_window = getattr(getattr(self.agent, "llm", None), "context_window", None)
        if context_window is None or self._used is None:
            return
        attributed = self.session.info.usage.attributed_cost_usd
        meta: dict[str, Any] | None = None
        status = getattr(self.agent, "get_summarization_status", None)
        if callable(status):
            try:
                meta = {"dev.nooa/context": _json_safe(status())}
            except Exception:
                logger.debug("Could not read the context status", exc_info=True)
        self._enqueue(
            UsageUpdate(
                session_update="usage_update",
                used=self._used,
                size=max(context_window, self._used),
                cost=Cost(amount=self._cost_usd + attributed, currency="USD"),
                field_meta=meta,
            )
        )

    def _publish_plan(self) -> None:
        """Send the agent's todos as an ACP plan when they changed."""
        todo: Any = getattr(self.agent, "todo", None)
        if not callable(getattr(todo, "list_todos", None)):
            return
        active = todo.active() if callable(getattr(todo, "active", None)) else None
        entries = [
            plan_entry(
                item.title,
                status=(
                    "completed"
                    if item.status == "done"
                    else "in_progress"
                    if active is not None and item.id == active.id
                    else "pending"
                ),
            )
            for item in todo.list_todos()
        ]
        if entries == self._plan:
            return
        self._plan = entries
        self._enqueue(update_plan(entries))

    # ---- pump --------------------------------------------------------

    def _stopped_error(self, cause: BaseException) -> RuntimeError:
        error = RuntimeError("ACP event bridge stopped")
        error.__cause__ = cause
        return error

    def _fail_pending_flushes(self, cause: BaseException) -> None:
        """Resolve every queued flush marker once the pump can no longer run."""
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, asyncio.Future) and not item.done():
                item.set_exception(self._stopped_error(cause))
            self._queue.task_done()

    async def _pump(self) -> None:
        try:
            await self._pump_loop()
        except BaseException as exc:
            # CancelledError is a BaseException, so a transport cancelled during
            # client disconnect used to kill this task silently. flush() waits on
            # a marker only the pump resolves, so every later flush — and close(),
            # which flushes first — blocked forever.
            self._pump_failure = exc
            self._fail_pending_flushes(exc)
            raise

    async def _pump_loop(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, asyncio.Future):
                    # Hand the failure to the turn that is flushing, then clear
                    # it. Latching it would mean every later turn ran the model
                    # and edited files while the client saw nothing and got a
                    # stale exception it could not act on.
                    error, self._error = self._error, None
                    if not item.done():
                        if error is None:
                            item.set_result(None)
                        else:
                            item.set_exception(error)
                    continue
                if isinstance(item, _BestEffortUpdate):
                    try:
                        await self.client.session_update(self.session_id, item.value)
                    except Exception:
                        logger.debug(
                            "ACP client rejected best-effort session bootstrap update",
                            exc_info=True,
                        )
                elif self._error is None:
                    try:
                        await self.client.session_update(self.session_id, item)
                    except Exception as exc:
                        # Skip the rest of this turn's updates rather than
                        # hammering a transport that just failed; the next
                        # flush reports the error and resets.
                        self._error = exc
                        logger.warning(
                            "ACP session %s dropped an update after a failed send",
                            self.session_id,
                            exc_info=True,
                        )
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        if self._closed:
            return
        if self._pump_failure is not None:
            raise self._stopped_error(self._pump_failure)
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(future)
        # Race the pump: if it dies without draining this marker, waiting on the
        # marker alone would never return.
        await asyncio.wait({future, self._pump_task}, return_when=asyncio.FIRST_COMPLETED)
        if future.done():
            future.result()
            return
        raise self._stopped_error(self._pump_failure or RuntimeError("pump exited"))

    def _fail_tools(self, keys: list[ToolKey], reason: str, *, title: str | None) -> None:
        for key in keys:
            self._open_tools.discard(key)
            self._terminal_output.pop(key, None)
            code = self._python_source.pop(key, None)
            content = (
                _python_content(code, reason)
                if code is not None
                else [tool_content(text_block(reason))]
            )
            self._enqueue(
                update_tool_call(
                    self._wire_id(key),
                    title=title or ("Python interrupted" if code is not None else None),
                    status="failed",
                    content=content,
                )
            )

    def fail_open_tools(self, reason: str, *, title: str | None = None) -> None:
        """Close out open tool calls, titling them with what actually happened.

        The title is the collapsed-card text, so it is the only thing a user
        sees without expanding. A fixed "Python interrupted" made a deliberate
        cancellation read as a technical failure.
        """
        self._fail_tools(sorted(self._open_tools), reason, title=title)

    async def close(self, *, finish_open: bool = True) -> None:
        """Stop listening and sending. Idempotent.

        ``finish_open`` closes cards still open as "Unfinished"; a bridge
        detached from a session that goes on running leaves them alone.
        """
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._close(finish_open=finish_open))
        await asyncio.shield(self._close_task)

    async def wait_closed(self) -> None:
        """Wait until a close started by the session's ``closed`` update has finished."""
        while self._close_task is None:
            await asyncio.sleep(0)
        await asyncio.shield(self._close_task)

    async def _close(self, *, finish_open: bool = True) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        for child_id in list(self._mirrors):
            for unsubscribe in self._mirrors.pop(child_id):
                unsubscribe()
        # A turn that ended before its PythonOutput — an exception escaping the
        # strategy, say — leaves cards in_progress and their source retained.
        if finish_open:
            self.fail_open_tools("Session closed before this finished.", title="Unfinished")
        else:
            self._open_tools.clear()
            self._python_source.clear()
            self._terminal_output.clear()
        with suppress(Exception):
            await self.flush()
        self._closed = True
        self._queue.put_nowait(_STOP)
        with suppress(BaseException):
            await self._pump_task
