# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate a Session's updates into ACP session updates.

One bridge per session id per adapter. It listens to the Session's
updates only (``subscribe``): titles, modes, turn ends, cancels,
children, admitted items, close, and the agent's events (tool cells,
messages, file edits, terminal commands, model responses) as
``AgentEventUpdate``. The stream is synchronous and ordered, so updates
reach the client in the order things happened. The bridge never holds
the agent: status beyond the updates comes from ``model_info()`` and
``plan()``.
"""

import asyncio
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
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    Cost,
    CurrentModeUpdate,
    PlanEntry,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallLocation,
    UsageUpdate,
    UserMessageChunk,
)

from nooa.agentdoc import pformat
from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from nooa.events import LLMResponse, PythonOutput
from nooa.interactive import AgentMessage, Done, NeedInput, Waiting
from nooa_coder.coding.activity import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from nooa_coder.session.items import (
    USAGE_FIELDS,
    AgentEventUpdate,
    CancelledUpdate,
    ChildCreatedUpdate,
    ItemAdmittedUpdate,
    ItemConsumedUpdate,
    ModeChangedUpdate,
    ModelInfo,
    SessionInfo,
    TitleChangedUpdate,
    TurnEndedUpdate,
)

# ACP owns stdout for JSON-RPC; diagnostics belong on stderr, which is where
# the logging default sends them.
logger = logging.getLogger(__name__)

_STOP = object()

_MAX_TITLE_CODE_CHARS = 80

# Values of return_result(...): the turn's result, not a cell's output.
_TURN_RESULTS = (Done, NeedInput, Waiting)

# Bound on a rendered Out[n] value; large results belong in the agent's
# context, not repeated in full inside a client tool card.
_MAX_VALUE_CHARS = 10_000

OWN_SOURCES = frozenset({"acp", "user:declined"})
"""Item sources this adapter admits itself; the client already shows those."""

_ECHOED_CHANNELS = frozenset({"user_messages", "steer"})

# By (command_truncated, stdin_truncated) of a TerminalCommandStarted.
_TRUNCATION_NOTES = {
    (True, True): "The command and its standard input were truncated for display.",
    (True, False): "The command was truncated for display.",
    (False, True): "Its standard input was truncated for display.",
}

ToolKey = tuple[str, str]


@dataclass
class _OpenCard:
    """What the bridge keeps about one tool card it has opened and not yet closed."""

    kind: Literal["python", "terminal"]
    code: str = ""  # a Python card's source
    output: str = ""  # what a terminal command has streamed so far
    note: str | None = None  # a terminal card's truncation note, kept at the top


"""An open tool card: (id of the session that ran it, its tool call id)."""


class BridgedSession(Protocol):
    """What the bridge needs from a Session."""

    id: str
    info: SessionInfo

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]: ...

    def model_info(self) -> ModelInfo: ...

    def plan(self) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class _BestEffortUpdate:
    value: Any


@dataclass(frozen=True, slots=True)
class _Sent:
    """Resolve ``future`` once the pump reaches it, without taking a flush's error."""

    future: asyncio.Future[None]


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


def _python_title(code: str, status: str | None = None) -> str:
    """``python: <first code line>``, like the shell card's ``$ <command>``.

    A card's title is what a collapsed card shows, so it names the code; a
    status word (``failed``, ``cancelled``...) is appended when the cell did
    not complete.
    """
    first = next((line.strip() for line in code.splitlines() if line.strip()), "")
    if len(first) > _MAX_TITLE_CODE_CHARS:
        first = first[: _MAX_TITLE_CODE_CHARS - 1] + "…"
    title = f"python: {first}" if first else "python"
    return f"{title} ({status})" if status else title


def cancel_text(by: str) -> str:
    """What the conversation says when a turn was stopped by ``by``."""
    return "Stopped at your request." if by == "user" else f"Stopped by {by}."


def question_text(question: str, options: list[str] | None, reason: str | None = None) -> str:
    """A ``NeedInput`` question as the agent's final message of the turn; its reason last."""
    text = question
    if options:
        text += "\n\n" + "\n".join(f"- {option}" for option in options)
    if reason:
        text += "\n\n" + reason
    return text


POOL_INPUT_EVENT_ID = "poolside/inputEventId"
POOL_CLIENT_INPUT_ID = "poolside/clientInputId"


def pool_input_event(item_id: str, client_input_id: str | None = None) -> SessionInfoUpdate:
    """The update Pool's own agent sends when it takes a person's input.

    Pool places a message it queued with ``_poolside/session_steer`` in the
    conversation when it sees the steer's ``inputId`` as
    ``poolside/clientInputId``; without it the message is never shown.
    Every input also carries a ``poolside/inputEventId``: here the item id,
    so it is the same live and when the transcript is replayed. Measured
    with Pool 1.0.16 against its own agent (``pool acp``).
    """
    meta: dict[str, Any] = {POOL_INPUT_EVENT_ID: item_id}
    if client_input_id is not None:
        meta = {POOL_CLIENT_INPUT_ID: client_input_id, **meta}
    return SessionInfoUpdate(session_update="session_info_update", field_meta=meta)


MAX_CHUNK_CHARS = 64_000
"""Longest text one message or thought chunk carries.

A longer text (a large paste, a long answer, a transcript entry replayed by
``session/load``) goes out as several chunks of the same kind, which a
client joins as it joins any stream of chunks. This keeps every message well
under the 1 MiB that WebSocket clients accept by default (``websockets``):
64,000 characters are at most about 384 KiB of JSON, even if every
character needs a six-byte escape.
"""

_TEXT_CHUNKS = (AgentMessageChunk, UserMessageChunk, AgentThoughtChunk)


def split_text_chunk(update: Any) -> list[Any]:
    """``update`` as a list of updates, split when it is a text chunk over ``MAX_CHUNK_CHARS``."""
    if not isinstance(update, _TEXT_CHUNKS) or not isinstance(update.content, TextContentBlock):
        return [update]
    text = update.content.text
    if len(text) <= MAX_CHUNK_CHARS:
        return [update]
    return [
        update.model_copy(
            update={
                "content": update.content.model_copy(
                    update={"text": text[start : start + MAX_CHUNK_CHARS]}
                )
            }
        )
        for start in range(0, len(text), MAX_CHUNK_CHARS)
    ]


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
        self._close_started = asyncio.Event()
        # Sticky: any close(finish_open=False) before the cards are finalised
        # wins, whoever started the close.
        self._finish_open = True
        self._open: dict[ToolKey, _OpenCard] = {}
        self._plan: list[PlanEntry] = []
        self._children: list[dict[str, Any]] = []
        self._mirrors: dict[str, list[Callable[[], None]]] = {}
        # Pool steers waiting for their item to be taken: item id ->
        # (Pool's inputId, the future resolved once Pool has been told).
        self._client_inputs: dict[str, tuple[str, asyncio.Future[None]]] = {}
        # Recently taken item ids, for a steer registered after its item was taken.
        self._taken: dict[str, None] = {}
        # By event type. Cards come from this session's agent and, mirrored,
        # from its children's; messages, thoughts and usage only from its own.
        self._card_handlers: dict[str, Callable[[Any, str | None], None]] = {
            "ToolCallEvent": self._on_tool_call,
            "PythonOutput": self._on_python_output,
            "FileEdit": self._on_file_edit,
            "TerminalCommandStarted": self._on_terminal_started,
            "TerminalCommandOutput": self._on_terminal_output,
            "TerminalCommandFinished": self._on_terminal_finished,
        }
        self._own_handlers: dict[str, Callable[[Any], None]] = {
            "AgentMessage": self._on_agent_message,
            "LLMResponse": self._on_llm_response,
        }
        self._unsubscribers: list[Callable[[], None]] = [
            session.subscribe(self._on_session_update),
        ]
        self._pump_task = asyncio.create_task(self._pump(), name="nooa-acp-events")

    # ---- subscriptions -----------------------------------------------

    def _on_agent_event(self, event: EventBase, source: str | None) -> None:
        """One agent event, from an ``AgentEventUpdate``.

        ``source`` is ``None`` for this session's own agent, else the id of
        a child whose tool cards are mirrored (cards only: a child's
        messages, thoughts and usage are not this conversation's).
        """
        card = self._card_handlers.get(event.event_type)
        if card is not None:
            card(event, source)
            return
        if source is None:
            own = self._own_handlers.get(event.event_type)
            if own is not None:
                own(event)

    def _mirror(self, child: BridgedSession) -> None:
        """Mirror a child's tool cards (and its own children's) into this session."""
        if child.id in self._mirrors or self._closed:
            return

        def on_child_update(update: Any) -> None:
            kind = getattr(update, "kind", None)
            if isinstance(update, AgentEventUpdate):
                self._on_agent_event(update.event, child.id)
            elif isinstance(update, ChildCreatedUpdate):
                # Announced like this session's own children, so the client
                # knows the id before cards arrive under it.
                self._on_child_created(update)
            elif kind == "closed":
                self._unmirror(child.id)

        self._mirrors[child.id] = [child.subscribe(on_child_update)]

    def _unmirror(self, child_id: str) -> None:
        for unsubscribe in self._mirrors.pop(child_id, []):
            unsubscribe()
        self._fail_tools(
            [key for key in self._open if key[0] == child_id],
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
            for part in split_text_chunk(update):
                self._queue.put_nowait(part)

    def publish(self, update: Any) -> None:
        """Queue a host-originated session update on the ordered ACP stream."""
        self._enqueue(update)

    def publish_best_effort(self, update: Any) -> None:
        """Queue bootstrap metadata without poisoning the live event stream."""
        self._enqueue(_BestEffortUpdate(update))

    # ---- session updates ---------------------------------------------

    def client_input_taken(self, item_id: str, client_input_id: str) -> asyncio.Future[None]:
        """Tell Pool when the turn takes ``item_id``, the item of its steer ``client_input_id``.

        Returns a future that resolves once ``pool_input_event`` has gone
        out, after everything queued before it, so the steer's answer can
        follow it as Pool's own agent does. It never resolves when the item
        is never taken (withdrawn, discarded, the session closed): the caller
        also waits for the item's outcome, and calls ``forget_client_input``.
        """
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if item_id in self._taken:
            self._send_client_input(item_id, client_input_id, done)
        else:
            self._client_inputs[item_id] = (client_input_id, done)
        return done

    def forget_client_input(self, item_id: str) -> None:
        self._client_inputs.pop(item_id, None)

    def _send_client_input(
        self, item_id: str, client_input_id: str, done: asyncio.Future[None]
    ) -> None:
        if self._closed:
            if not done.done():
                done.set_result(None)
            return
        self._enqueue(pool_input_event(item_id, client_input_id))
        self._queue.put_nowait(_Sent(done))  # resolved once the update is sent

    def _on_session_update(self, update: Any) -> None:
        if isinstance(update, AgentEventUpdate):
            self._on_agent_event(update.event, None)
        elif isinstance(update, ItemConsumedUpdate):
            self._taken[update.item_id] = None
            while len(self._taken) > 256:
                del self._taken[next(iter(self._taken))]
            waiting = self._client_inputs.pop(update.item_id, None)
            if waiting is not None:
                self._send_client_input(update.item_id, *waiting)
        elif isinstance(update, ItemAdmittedUpdate):
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
            self._start_close()

    def _on_turn_ended(self, update: TurnEndedUpdate) -> None:
        if update.outcome_kind == "need_input":
            # Rendered once per turn, here; the prompt that owns the consumed
            # item decides whether to also open a form.
            question = str(update.outcome.get("question", ""))
            options = update.outcome.get("options")
            reason = update.outcome.get("reason")
            self._enqueue(
                update_agent_message(
                    text_block(question_text(question, options, str(reason) if reason else None))
                )
            )
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
        self._open[key] = _OpenCard("python", code=code)
        self._enqueue(
            start_tool_call(
                self._wire_id(key),
                _python_title(code),
                # Zed 1.14 treats every ``execute`` tool as a terminal card.
                # A plain-content execute card has neither a terminal nor an
                # output disclosure, so its source cannot be opened. Python is
                # structured Markdown content, not a client-owned terminal.
                kind="other",
                status="in_progress",
                content=_python_content(code),
                raw_input={"code": code},
            )
        )

    def _on_python_output(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, PythonOutput):
            return
        key = self._key(event.tool_call_id, source)
        card = self._open.pop(key, None)
        if card is not None:
            code = card.code
            parts = [
                part.rstrip() for part in (event.stdout, event.stderr, event.error) if part.strip()
            ]
            # A cell whose last line is a bare expression produces no stdout:
            # the result arrives as ``value`` and codeact shows it to the model
            # as Out[n]. Without this the client is told there was no output
            # while the agent is reasoning from one. A turn result from
            # return_result(...) is not output: the turn's own messages show it.
            if event.value is not None and not isinstance(event.value, _TURN_RESULTS):
                rendered = pformat(event.value, max_string=_MAX_VALUE_CHARS, unquote_strings=True)
                parts.append(f"Out[{event.execution_count}]: {rendered}")
            cancelled = event.execution_status is ResultStatus.CANCELLED
            output = "\n".join(parts) or ("Cancelled." if cancelled else "Completed.")
            status: Literal["failed", "completed"] = (
                "failed"
                if event.execution_status in (ResultStatus.ERROR, ResultStatus.CANCELLED)
                else "completed"
            )
            title = _python_title(
                code, "cancelled" if cancelled else "failed" if status == "failed" else None
            )
            self._enqueue(
                update_tool_call(
                    self._wire_id(key),
                    title=title,
                    status=status,
                    content=_python_content(code, output),
                    raw_input={"code": code},
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
        content: list[Any]
        if event.content_complete:
            content = [tool_diff_content(path, event.new_text, event.old_text)]
        else:
            content = [tool_content(text_block(event.diff or "File content was truncated."))]
            if event.diff and not event.diff_complete:
                content.append(
                    tool_content(
                        text_block("The diff was truncated: the edit is larger than shown.")
                    )
                )
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
                raw_output={
                    "content_complete": event.content_complete,
                    "diff_complete": event.diff_complete,
                },
            )
        )

    def _on_terminal_started(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandStarted):
            return
        key = self._key(event.command_id, source)
        note = _TRUNCATION_NOTES.get((event.command_truncated, event.stdin_truncated))
        self._open[key] = _OpenCard("terminal", note=note)
        self._enqueue(
            start_tool_call(
                self._wire_id(key),
                f"$ {event.command}",
                kind="execute",
                status="in_progress",
                content=self._terminal_content(key, None),
                raw_input={
                    "command": event.command,
                    "working_directory": event.working_directory,
                    "command_truncated": event.command_truncated,
                    "stdin_truncated": event.stdin_truncated,
                },
            )
        )

    def _terminal_content(self, key: ToolKey, output: str | None) -> list[Any] | None:
        """A terminal card's content: its truncation note, if any, then ``output``."""
        card = self._open.get(key)
        texts = [card.note if card is not None else None, output]
        content = [tool_content(text_block(text)) for text in texts if text is not None]
        return content or None

    def _on_terminal_output(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandOutput):
            return
        key = self._key(event.command_id, source)
        card = self._open.get(key)
        if card is None:
            return
        chunk = event.stdout
        if event.stderr:
            chunk += ("\n" if chunk and not chunk.endswith("\n") else "") + event.stderr
        card.output += chunk
        output = card.output
        self._enqueue(
            update_tool_call(
                self._wire_id(key),
                status="in_progress",
                content=self._terminal_content(key, output),
            )
        )

    def _on_terminal_finished(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandFinished):
            return
        key = self._key(event.command_id, source)
        card = self._open.get(key)
        output = card.output if card is not None else ""
        # ACP has no cancelled status, so a stopped command is still "failed" —
        # but it must read as the user's own action, not as a crash.
        reason = "Cancelled by user." if event.cancelled else event.error
        if reason:
            output += ("\n" if output and not output.endswith("\n") else "") + reason
        content = self._terminal_content(key, output or "Completed.")
        self._open.pop(key, None)
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
                content=content,
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

    # ---- derived updates ---------------------------------------------

    def _publish_usage(self) -> None:
        """Context use and cost (own plus what children spent) as a usage update.

        ACP's usage update has no token counts beyond the context in use, so
        the session's totals (own plus children's, cached and reasoning
        tokens included) go in ``_meta["dev.nooa/usage"]``. Nothing is sent
        before the session's first model call: there is no context in use.
        """
        context_window = self.session.model_info().context_window
        usage = self.session.info.usage
        used = usage.last_input_tokens
        if context_window is None or not used:
            return
        totals = usage.with_attributed()
        meta_totals = totals.model_dump(
            include={name for name in USAGE_FIELDS if name != "cost_usd"}
        )
        meta: dict[str, Any] = {"dev.nooa/usage": meta_totals}
        self._enqueue(
            UsageUpdate(
                session_update="usage_update",
                used=used,
                size=max(context_window, used),
                cost=Cost(amount=totals.cost_usd, currency="USD"),
                field_meta=meta,
            )
        )

    def _publish_plan(self) -> None:
        """Send the agent's plan (``Session.plan()``) as an ACP plan update when it changed."""
        entries = [
            plan_entry(entry.content, status=entry.status, priority=entry.priority)
            for entry in self.session.plan()
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
            elif isinstance(item, _Sent) and not item.future.done():
                item.future.set_result(None)
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
                if isinstance(item, _Sent):
                    if not item.future.done():
                        item.future.set_result(None)
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
            card = self._open.get(key)
            if card is None:
                continue
            if card.kind == "python":
                content = _python_content(card.code, reason)
            else:
                # What the command streamed so far stays above the reason.
                streamed = card.output.rstrip("\n")
                content = self._terminal_content(
                    key, f"{streamed}\n\n{reason}" if streamed else reason
                )
            del self._open[key]
            self._enqueue(
                update_tool_call(
                    self._wire_id(key),
                    # A Python card keeps its code in the title, with the reason.
                    title=_python_title(card.code, (title or "interrupted").lower())
                    if card.kind == "python"
                    else title,
                    status="failed",
                    content=content,
                )
            )

    def fail_open_tools(self, reason: str, *, title: str | None = None) -> None:
        """Close out this session's open tool calls, titling them with what happened.

        The title is the collapsed-card text, so it is the only thing a user
        sees without expanding. A fixed "Python interrupted" made a deliberate
        cancellation read as a technical failure.

        Mirrored children's cards are left alone: a child goes on running
        when this session's turn is cancelled or fails, and its own results
        close them (or ``_unmirror`` does when it ends).
        """
        own = sorted(key for key in self._open if key[0] == self.session_id)
        self._fail_tools(own, reason, title=title)

    async def close(self, *, finish_open: bool = True) -> None:
        """Stop listening and sending. Idempotent.

        ``finish_open`` closes cards still open as "Unfinished"; a bridge
        detached from a session that goes on running leaves them alone.
        ``finish_open=False`` counts even when a close is already under way
        (the session's own ``closed`` update, say), as long as that close
        has not yet finalised the cards.
        """
        if not finish_open:
            self._finish_open = False
        await asyncio.shield(self._start_close())

    def _start_close(self) -> asyncio.Task[None]:
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._close())
            self._close_started.set()
        return self._close_task

    async def wait_closed(self) -> None:
        """Wait until a close started by the session's ``closed`` update has finished."""
        await self._close_started.wait()
        assert self._close_task is not None
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        for child_id in list(self._mirrors):
            for unsubscribe in self._mirrors.pop(child_id):
                unsubscribe()
        # A turn that ended before its PythonOutput — an exception escaping the
        # strategy, say — leaves cards in_progress and their source retained.
        if self._finish_open:
            # Every card, mirrored children's too: nothing will close them now.
            self._fail_tools(
                sorted(self._open), "Session closed before this finished.", title="Unfinished"
            )
        else:
            self._open.clear()
        with suppress(Exception):
            await self.flush()
        self._closed = True
        self._queue.put_nowait(_STOP)
        with suppress(BaseException):
            await self._pump_task
