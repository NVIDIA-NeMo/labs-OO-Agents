# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate a Session's updates into ACP v2 session updates.

One bridge per session id per adapter. It listens to the Session's
updates only (``subscribe``): titles, turn starts and ends, cancels,
children, admitted items, close, and the agent's events (tool cells,
messages, file edits, terminal commands, model responses) as
``AgentEventUpdate``. The stream is synchronous and ordered, so updates
reach the client in the order things happened. The bridge never holds
the agent: status beyond the updates comes from ``model_info()`` and
``plan()``.

Updates are v2 values (``nooa_atom.acp.v2``) sent through a ``Peer``:
messages carry a ``messageId``, tool calls are ``tool_call_update``
upserts, the session's foreground state goes out as ``state_update``,
and each child is a subagent session of its own (``subagent_update``)
whose cards and messages go out under its id.
"""

import asyncio
import logging
import os
import re
import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from nooa.agentdoc import pformat
from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from nooa.events import LLMResponse, PythonOutput, Summary
from nooa.interactive import AgentMessage, Done, NeedInput, Waiting
from nooa.strategies.codeact import MAX_ITERATIONS_MESSAGE, OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from nooa_atom.acp import v2
from nooa_atom.acp.protocol import POOLSIDE_DIALECT, PoolDialect
from nooa_atom.acp.v2 import Peer
from nooa_atom.agent.activity import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from nooa_atom.session.items import (
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
"""Already-visible client input/permission answers and silent decline markers.

Accepted form answers use ``acp:form-answer`` instead, so admission echoes them.
"""

_ECHOED_CHANNELS = frozenset({"user_messages", "steer"})

# By (command_truncated, stdin_truncated) of a TerminalCommandStarted.
_TRUNCATION_NOTES = {
    (True, True): "The command and its standard input were truncated for display.",
    (True, False): "The command was truncated for display.",
    (False, True): "Its standard input was truncated for display.",
}

Verbosity = Literal["silent", "concise", "verbose"]
VERBOSITY_LEVELS = ("silent", "concise", "verbose")

ToolKey = tuple[str, str]


@dataclass
class _OpenCard:
    """What the bridge keeps about one tool card it has opened and not yet closed."""

    kind: Literal["python", "terminal"]
    code: str = ""  # a Python card's full source
    wire_id: str | None = None  # concise cells replace the same rolling card
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
class _Out:
    """One update for the client: ``session_id`` is a mirrored child's, else ``None``."""

    update: dict[str, Any]
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _BestEffortUpdate:
    value: _Out


@dataclass
class _RollingSlot:
    wire_id: str
    key: ToolKey


@dataclass(frozen=True, slots=True)
class _RollingUpdate:
    value: _Out
    epoch: str
    transport_generation: int


@dataclass(frozen=True, slots=True)
class _InputEvent:
    """An input acknowledgement and the future for this specific send attempt."""

    value: dict[str, Any]
    future: asyncio.Future[None]


@dataclass(slots=True)
class _ClientInput:
    client_id: str
    future: asyncio.Future[None]
    taken: bool = False


def _template_pattern(template: str) -> re.Pattern[str]:
    """A pattern matching ``template`` (a ``str.format`` string) with any field values."""
    from string import Formatter

    return re.compile(
        "".join(
            re.escape(literal) + (".*?" if field is not None else "")
            for literal, field, _spec, _conversion in Formatter().parse(template)
        )
    )


_MAX_ITERATIONS = _template_pattern(MAX_ITERATIONS_MESSAGE)

INTERNAL_ERROR = -32603


def limit_stop_reason(message: str) -> str | None:
    """``max_tokens`` / ``max_turn_requests`` for a turn that hit a generation limit, else ``None``.

    The messages CodeActStrategy raises, from its own constants.
    max_retries ("failed after N errors") is repeated invalid output, not a
    limit on turn requests, so it stays an error.
    """
    if OUTPUT_TOKENS_EXHAUSTED_MESSAGE in message:
        return "max_tokens"
    if _MAX_ITERATIONS.search(message):
        return "max_turn_requests"
    return None


_STOP_REASONS = {"done": "end_turn", "need_input": "end_turn", "cancelled": "cancelled"}


def _observe_future(future: asyncio.Future[None]) -> None:
    # Requests/flushes can be cancelled while the pump still owns their send.
    # Retrieving an exception here does not stop an await from raising it.
    if not future.cancelled():
        future.exception()


def _fenced_code(text: str, language: str) -> str:
    """Wrap text in a Markdown fence that cannot collide with its contents."""
    longest_run = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _python_content(code: str, output: str | None = None) -> list[dict[str, Any]]:
    """Render Python source and output as Markdown visible in ACP clients."""
    content = [v2.tool_text(_fenced_code(code, "python"))]
    if output is not None:
        content.append(v2.tool_text(_fenced_code(output, "text")))
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


def question_text(question: str, reason: str | None = None) -> str:
    """A ``NeedInput`` question as the agent's final message of the turn; its reason last."""
    text = question
    if reason:
        text += "\n\n" + reason
    return text


def result_card(kind: str, outcome: dict[str, Any]) -> tuple[str, str | None] | None:
    """The title and body of the card that ends a ``Done``/``Waiting`` turn, else ``None``.

    The title is ``Done: <first line of the explanation>`` (``Waiting: ...``),
    or just ``Done``; a longer explanation is the body.
    """
    labels = {"done": "Done", "waiting": "Waiting"}
    if kind not in labels:
        return None
    explanation = str(outcome.get("explanation") or "").strip()
    first, *rest = explanation.splitlines() or [""]
    title = f"{labels[kind]}: {first}" if first else labels[kind]
    return title, (explanation if rest else None)


def end_line(text: str) -> str:
    """An agent message's ``text``, ending in a blank line.

    Every agent message the bridge sends is complete, but clients that join
    consecutive agent chunks (every v1 client; Pool 1.0.16 even across
    ``messageId`` values) need a paragraph break: a single line break is only
    a soft break in Markdown, so "Not posted.\n" followed by "Trace viewer:"
    rendered as one paragraph, "Not posted. Trace viewer:". A blank line ends
    the paragraph, so the next message (``/usage``, ``/trace-url`` included)
    starts its own.
    """
    if text and not text.endswith("\n\n"):
        return text + ("\n" if text.endswith("\n") else "\n\n")
    return text


def agent_message(text: str, message_id: str | None = None) -> list[dict[str, Any]]:
    """A whole agent message (``end_line``), as chunks of one message id."""
    return v2.message("agent_message", end_line(text), message_id)


POOL_INPUT_EVENT_ID = POOLSIDE_DIALECT.input_event_id
POOL_CLIENT_INPUT_ID = POOLSIDE_DIALECT.client_input_id


def pool_input_event(
    item_id: str,
    client_input_id: str | None = None,
    dialects: tuple[PoolDialect, ...] = (POOLSIDE_DIALECT,),
) -> dict[str, Any]:
    """The update Pool's own agent sends when it takes a person's input.

    Pool places a message it queued with ``_poolside/session_steer`` in the
    conversation when it sees the steer's ``inputId`` as
    ``poolside/clientInputId``; without it the message is never shown.
    Every input also carries a ``poolside/inputEventId``: here the item id,
    so it is the same live and when the transcript is replayed. Measured
    with Pool 1.0.16 against its own agent (``pool acp``). Pool 2 reads the
    same two ids under ``pool/`` keys; ``dialects`` names the key sets sent.
    """
    meta: dict[str, Any] = {}
    for dialect in dialects:
        if client_input_id is not None:
            meta[dialect.client_input_id] = client_input_id
        meta[dialect.input_event_id] = item_id
    return {"sessionUpdate": "session_info_update", "_meta": meta}


MAX_CHUNK_CHARS = v2.TEXT_LIMIT
"""Longest text one message or thought chunk carries (``v2.TEXT_LIMIT``)."""


class ACPEventBridge:
    """Send one session's activity to an ACP client, in order, through one pump task.

    ``resolve_child(child_id)`` finds a live child Session; when given, this
    session's children (and theirs) are followed too: each is announced as
    a subagent (``subagent_update``) and its cards, messages and state go
    out under its own session id.
    """

    def __init__(
        self,
        session: BridgedSession,
        client: Peer,
        *,
        resolve_child: Callable[[str], BridgedSession | None] | None = None,
        own_sources: frozenset[str] = OWN_SOURCES,
    ) -> None:
        self.session = session
        self.client = client
        self.session_id = session.id
        self.verbosity: Verbosity = "verbose"
        # The Pool meta key sets of input events (``pool_input_event``).
        self.pool_dialects: tuple[PoolDialect, ...] = (POOLSIDE_DIALECT,)
        # The foreground state the client was last told, by session id (this
        # one and the followed children): idle, running or requires_action.
        self._states: dict[str, str] = {}
        # Requests to the client the foreground work is waiting on (forms).
        self._asks = 0
        # A local command's foreground work (``begin_local``), while no turn runs.
        self._local = False
        self._turn_running = False
        # What a waiting turn waits on, shown as background tool calls
        # ``job:<name>``; ``None`` until a turn has waited on something.
        self._background: dict[str, str] | None = None
        # Children announced to the client, and their parents.
        self._parents: dict[str, str] = {}
        self._described: set[str] = set()
        self._resolve_child = resolve_child
        self._own_sources = own_sources
        self._loop = asyncio.get_running_loop()
        self._owner_thread = threading.get_ident()
        self._pending: deque[tuple[Callable[[Any], None], Any]] = deque()
        self._pending_lock = threading.Lock()
        self._accepting_updates = True
        self._draining = False
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
        self._rolling: dict[str, _RollingSlot] = {}
        self._helper_names: dict[str, str] = {}
        self._rolling_seen: set[ToolKey] = set()  # ACP execution ids must not be reused
        self._invalid_rolling: set[str] = set()
        self._transport_generation = 0
        self._plan: list[dict[str, Any]] = []
        self._children: list[dict[str, Any]] = []
        self._mirrors: dict[str, list[Callable[[], None]]] = {}
        # Keep the association through send failure until the request forgets it.
        self._client_inputs: dict[str, _ClientInput] = {}
        # Live admission dedup, including a steer's same-ID queued fallback.
        # Replay deliberately goes through publish(), not this set.
        self._echoed_items: set[str] = set()
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
            "Summary": self._on_summary,
        }
        self._unsubscribers: list[Callable[[], None]] = [
            session.subscribe(lambda update: self._dispatch(self._on_session_update, update)),
        ]
        self._pump_task = asyncio.create_task(self._pump(), name="nooa-acp-events")

    def set_verbosity(self, level: str) -> None:
        """Set this ACP attachment's activity detail, without changing agent behavior."""
        if level not in VERBOSITY_LEVELS:
            raise ValueError("Use silent, concise, or verbose.")
        self._drain()
        if level == self.verbosity:
            return
        # A mirrored child can still be running when the parent changes verbosity.
        # Finalise already-visible cards so reduced detail leaves no spinning cards.
        self._fail_tools(list(self._open), "Activity detail changed.", title="Display ended")
        self._rolling.clear()
        self.verbosity = cast(Verbosity, level)

    def _activity_text(self, full: str, concise: str | None = None) -> None:
        """Keep the activity's wire shape, reducing only its text in concise mode."""
        if self.verbosity == "silent":
            return
        text = full if self.verbosity == "verbose" else (concise or full)
        if self.verbosity == "concise":
            text = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if text:
            self._enqueue_all(v2.message("agent_thought", text.rstrip() + "\n\n"))

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
                if isinstance(update.event, AgentMessage):
                    # A child's own conversation, under its id.
                    self._enqueue_all(agent_message(update.event.content), child.id)
            elif isinstance(update, ChildCreatedUpdate):
                # Announced like this session's own children, so the client
                # knows the id before cards arrive under it.
                self._on_child_created(update, parent=child.id)
            elif isinstance(update, ItemAdmittedUpdate):
                if update.channel in _ECHOED_CHANNELS:
                    self._child_message(
                        child.id, update.item_id, getattr(update, "text", "") or update.preview
                    )
            elif kind == "turn_started":
                self._reset_rolling("A new helper turn started.", source=child.id)
                self._set_state("running", child.id)
            elif isinstance(update, TurnEndedUpdate):
                self._reset_rolling("Helper turn ended before this finished.", source=child.id)
                reason = _STOP_REASONS.get(update.outcome_kind, "end_turn")
                if update.outcome_kind == "error":
                    reason = "error"
                self._set_state("idle", child.id, stop_reason=reason)
            elif isinstance(update, CancelledUpdate):
                self._reset_rolling(cancel_text(update.by), source=child.id, title="Cancelled")
            elif kind == "closed":
                self._unmirror(child.id)

        self._mirrors[child.id] = [
            child.subscribe(lambda update: self._dispatch(on_child_update, update))
        ]

    def _unmirror(self, child_id: str) -> None:
        self._reset_rolling("The subagent ended before this finished.", source=child_id)
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

    def _target(self, key: ToolKey) -> str | None:
        """The session a card's updates go to: ``None`` for this one, else the child's id."""
        return None if key[0] == self.session_id else key[0]

    def _discard_rolling(self) -> None:
        """Discard all slots and queued updates, including already-ended epochs."""
        self._transport_generation += 1
        for slot in self._rolling.values():
            self._invalid_rolling.add(slot.wire_id)
            self._open.pop(slot.key, None)
        self._rolling.clear()

    def _reset_rolling(
        self, reason: str, *, source: str | None = None, title: str = "Unfinished"
    ) -> None:
        source_id = source or self.session_id
        slot = self._rolling.get(source_id)
        if slot is not None:
            self._fail_tools([slot.key], reason, title=title)
            self._rolling.pop(source_id, None)

    def _card_title(self, key: ToolKey, card: _OpenCard, status: str | None = None) -> str:
        return _python_title(card.code, status)

    # ---- queue -------------------------------------------------------

    def _dispatch(self, callback: Callable[[Any], None], update: Any) -> None:
        """Admit whole callbacks, so card state and queue writes stay on one loop."""
        with self._pending_lock:
            if not self._accepting_updates:
                return
            self._pending.append((callback, update))
            if threading.get_ident() != self._owner_thread:
                self._loop.call_soon_threadsafe(self._drain)
                return
        self._drain()

    def _drain(self) -> None:
        """An owner update or cleanup must not overtake an admitted worker update."""
        if self._draining:
            return
        self._draining = True
        try:
            while True:
                with self._pending_lock:
                    if not self._pending:
                        break
                    callback, update = self._pending.popleft()
                try:
                    callback(update)
                except (Exception, asyncio.CancelledError):
                    logger.warning("ACP event handler failed", exc_info=True)
        finally:
            self._draining = False

    def _enqueue(self, update: dict[str, Any], session_id: str | None = None) -> None:
        """Queue one v2 update for this session (``session_id``: a followed child's)."""
        if self._closed:
            return
        out = _Out(update, session_id)
        wire_id = update.get("toolCallId")
        if wire_id is not None and any(slot.wire_id == wire_id for slot in self._rolling.values()):
            self._queue.put_nowait(_RollingUpdate(out, wire_id, self._transport_generation))
        else:
            self._queue.put_nowait(out)

    def _enqueue_all(self, updates: list[dict[str, Any]], session_id: str | None = None) -> None:
        for update in updates:
            self._enqueue(update, session_id)

    def publish(self, update: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Queue host-originated v2 session updates on the ordered ACP stream."""
        updates = update if isinstance(update, list) else [update]
        self._dispatch(self._enqueue_all, updates)

    def say(self, text: str) -> None:
        """Queue a whole agent message, as the host says it (not the agent)."""
        self.publish(agent_message(text))

    def publish_best_effort(self, update: dict[str, Any]) -> None:
        """Queue bootstrap metadata without poisoning the live event stream."""
        self._dispatch(
            lambda value: (
                self._queue.put_nowait(_BestEffortUpdate(_Out(value))) if not self._closed else None
            ),
            update,
        )

    # ---- session updates ---------------------------------------------

    def client_input_taken(self, item_id: str, client_input_id: str) -> asyncio.Future[None]:
        """Acknowledge consumption, completing only after the transport send succeeds.

        A repeated registration shares its attempt; conflicting client IDs are
        rejected. Late registration is supported for the last 256 consumed IDs.
        Failed sends can be explicitly retried with ``retry_client_input`` while
        this registration is retained. This is not an execution retry or proof
        of client rendering; transport failure can occur after peer receipt.
        """
        previous = self._client_inputs.get(item_id)
        if previous is not None:
            if previous.client_id != client_input_id:
                raise ValueError("Conflicting client input ID")
            return previous.future
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        done.add_done_callback(_observe_future)
        registration = _ClientInput(client_input_id, done, item_id in self._taken)
        self._client_inputs[item_id] = registration
        if self._closed or self._pump_failure is not None:
            done.set_exception(self._stopped_error(self._pump_failure or RuntimeError("closed")))
        elif registration.taken:
            self._send_client_input(item_id, registration)
        return done

    def client_input_was_taken(self, item_id: str) -> bool:
        registration = self._client_inputs.get(item_id)
        return bool(registration and registration.taken)

    def retry_client_input(self, item_id: str) -> asyncio.Future[None]:
        """Explicitly retry only a failed acknowledgement, using the same wire IDs.

        The caller must first flush to report/reset the ordinary stream error.
        No admission, consumption, transcript replay or new turn is performed.
        This bridge-local facility does not make a repeated Pool steer request
        idempotent; reconnect/persistent client-ID reconciliation is not defined.
        """
        registration = self._client_inputs[item_id]
        if self._closed or self._pump_failure is not None:
            raise self._stopped_error(self._pump_failure or RuntimeError("closed"))
        if self._error is not None:
            raise RuntimeError("Flush the ACP send error before retrying an input event")
        if not registration.taken or not registration.future.done():
            raise ValueError("Input acknowledgement is not a failed completed attempt")
        if registration.future.cancelled() or registration.future.exception() is None:
            raise ValueError("Input acknowledgement is not a failed completed attempt")
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        done.add_done_callback(_observe_future)
        registration.future = done
        self._send_client_input(item_id, registration)
        return done

    def release_client_input_waiter(self, item_id: str, *, keep_pending: bool = False) -> None:
        """Release a request, not a consumed input's pump-owned notification.

        Successful sends are cleaned up; failed sends retain IDs for explicit
        retry. A cancelled request's still-admitted input retains its association
        until consumption/send or bridge close, even if not yet taken.
        """
        registration = self._client_inputs.get(item_id)
        if registration is None:
            return
        if not keep_pending and not registration.taken and not registration.future.done():
            self.forget_client_input(item_id)
            return

        def release_success(future: asyncio.Future[None]) -> None:
            if (
                not future.cancelled()
                and future.exception() is None
                and self._client_inputs.get(item_id) is registration
                and registration.future is future
            ):
                self.forget_client_input(item_id)

        if registration.future.done():
            release_success(registration.future)
        else:
            registration.future.add_done_callback(release_success)

    def forget_client_input(self, item_id: str) -> None:
        registration = self._client_inputs.pop(item_id, None)
        if registration is not None and not registration.future.done():
            registration.future.cancel()

    def _send_client_input(self, item_id: str, registration: _ClientInput) -> None:
        self._queue.put_nowait(
            _InputEvent(
                pool_input_event(item_id, registration.client_id, self.pool_dialects),
                registration.future,
            )
        )

    def _on_session_update(self, update: Any) -> None:
        if isinstance(update, AgentEventUpdate):
            self._on_agent_event(update.event, None)
        elif isinstance(update, ItemConsumedUpdate):
            self._taken[update.item_id] = None
            while len(self._taken) > 256:
                del self._taken[next(iter(self._taken))]
            waiting = self._client_inputs.get(update.item_id)
            if waiting is not None and not waiting.taken:
                waiting.taken = True
                self._send_client_input(update.item_id, waiting)
        elif isinstance(update, ItemAdmittedUpdate):
            if (
                update.channel in _ECHOED_CHANNELS
                and update.source not in self._own_sources
                and update.item_id not in self._echoed_items
            ):
                self._echoed_items.add(update.item_id)
                text = getattr(update, "text", "") or update.preview
                self._enqueue_all(v2.message("user_message", text, update.item_id))
        elif isinstance(update, TitleChangedUpdate):
            self._enqueue(
                {
                    "sessionUpdate": "session_info_update",
                    "title": update.title,
                    "updatedAt": datetime.now(UTC).isoformat(),
                }
            )
        elif isinstance(update, ModeChangedUpdate):
            # v2 has no session modes; the v1 shim shows this, V2Peer drops it.
            self._enqueue({"sessionUpdate": "current_mode_update", "currentModeId": update.mode})
        elif getattr(update, "kind", None) == "turn_started":
            self._reset_rolling("A new turn started.")
            self._turn_running = True
            self._set_state("requires_action" if self._asks else "running")
        elif isinstance(update, CancelledUpdate):
            self._reset_rolling(cancel_text(update.by), title="Cancelled")
            # Before the prompt's future resolves: the Session emits this first,
            # so the cards are closed before the prompt answers "cancelled".
            self.fail_open_tools(cancel_text(update.by), title="Cancelled")
            self._enqueue_all(agent_message(cancel_text(update.by)))
        elif isinstance(update, TurnEndedUpdate):
            self._on_turn_ended(update)
            self._turn_ended_state(update)
        elif isinstance(update, ChildCreatedUpdate):
            self._on_child_created(update)
        elif getattr(update, "kind", None) == "usage_changed":
            self._publish_usage()
        elif getattr(update, "kind", None) == "closed":
            self._start_close()

    def _on_turn_ended(self, update: TurnEndedUpdate) -> None:
        self._reset_rolling("Turn ended before this finished.")
        if update.outcome_kind == "need_input":
            # Rendered once per turn, here; the next prompt answers it as text.
            question = str(update.outcome.get("question", ""))
            reason = update.outcome.get("reason")
            self._enqueue_all(
                agent_message(question_text(question, str(reason) if reason else None))
            )
        elif update.outcome_kind in {"done", "waiting"}:
            # A card, not chat text: the turn's last message stays its reply.
            card = result_card(update.outcome_kind, update.outcome)
            if card is not None:
                self._status_card(*card)
        elif update.outcome_kind == "error":
            # A turn ending on an error does not always write the PythonOutput
            # for a cell it announced; close its card rather than leave it
            # spinning.
            self.fail_open_tools("Did not finish.", title="Unfinished")
        self._publish_plan()

    # ---- foreground state ------------------------------------------

    def _set_state(
        self,
        state: str,
        session_id: str | None = None,
        *,
        stop_reason: str | None = None,
        error: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        """Tell the client a session's foreground state (``None``: this session's).

        Repeats are dropped unless ``force`` (a resume's snapshot). A child's
        state also goes to its parent, in the child's ``subagent_update``.
        """
        target = session_id or self.session_id
        previous = self._states.get(target, "idle")
        if previous == state and not force:
            return
        self._states[target] = state
        update = v2.state_update(state)
        if state == "idle":
            if stop_reason is not None:
                update["stopReason"] = stop_reason
            if error is not None:
                update["error"] = error
        if target == self.session_id and self._background is not None:
            update["background"] = [
                {"type": "tool_call", "toolCallId": tool_call_id}
                for tool_call_id in self._background
            ]
        self._enqueue(update, session_id)
        parent = self._parents.get(target)
        if parent is not None:
            state_info = {key: value for key, value in update.items() if key != "sessionUpdate"}
            self._enqueue(
                {"sessionUpdate": "subagent_update", "sessionId": target, "state": state_info},
                None if parent == self.session_id else parent,
            )

    def state(self, session_id: str | None = None) -> str:
        """The foreground state last sent for this session (or a followed child)."""
        return self._states.get(session_id or self.session_id, "idle")

    def snapshot_state(self) -> None:
        """Send the current foreground state again (``session/resume`` ends with it)."""
        self._dispatch(lambda _: self._set_state(self.state(), force=True), None)

    def begin_local(self) -> None:
        """A local command starts foreground work (no turn runs it)."""

        def begin(_: Any) -> None:
            if self.state() == "idle":
                self._local = True
                self._set_state("running")

        self._dispatch(begin, None)

    def end_local(self, stop_reason: str = "end_turn") -> None:
        """The local command's work ended; idle unless a turn started meanwhile."""

        def end(_: Any) -> None:
            if self._local:
                self._local = False
                if not self._turn_running:
                    self._set_state("idle", stop_reason=stop_reason)

        self._dispatch(end, None)

    def ask_started(self) -> None:
        """A request to the client (a form) blocks the foreground work: requires_action."""

        def started(_: Any) -> None:
            self._asks += 1
            if self.state() == "running":
                self._set_state("requires_action")

        self._dispatch(started, None)

    def ask_ended(self) -> None:
        def ended(_: Any) -> None:
            self._asks = max(0, self._asks - 1)
            if not self._asks and self.state() == "requires_action":
                self._set_state("running")

        self._dispatch(ended, None)

    def _turn_ended_state(self, update: TurnEndedUpdate) -> None:
        """Idle at a turn's end, with its stop reason; what a waiting turn waits on is background."""
        self._turn_running = False
        self._local = False
        outcome = update.outcome_kind
        names = (
            [str(name) for name in update.outcome.get("on") or []] if outcome == "waiting" else []
        )
        if names or self._background:
            self._set_background(names)
        error = None
        if outcome == "error":
            message = str(update.outcome.get("error") or "The turn failed.")
            stop_reason = limit_stop_reason(message)
            if stop_reason is None:
                stop_reason = "error"
                error = v2.error_object(INTERNAL_ERROR, message)
        else:
            stop_reason = _STOP_REASONS.get(outcome, "end_turn")
        self._set_state("idle", stop_reason=stop_reason, error=error)

    def _set_background(self, names: list[str]) -> None:
        """List what a waiting turn waits on as background tool calls; finish the rest."""
        background = self._background if self._background is not None else {}
        self._background = background
        wanted = {f"job:{name}": name for name in names}
        for tool_call_id in list(background):
            if tool_call_id not in wanted:
                del background[tool_call_id]
                self._enqueue(v2.tool_call(tool_call_id, status="completed"))
        for tool_call_id, name in wanted.items():
            if tool_call_id not in background:
                background[tool_call_id] = name
                self._enqueue(
                    v2.tool_call(
                        tool_call_id, title=f"Waiting on {name}", kind="other", status="in_progress"
                    )
                )

    # ---- children ----------------------------------------------------

    def _on_child_created(self, update: ChildCreatedUpdate, *, parent: str | None = None) -> None:
        parent_id = parent or self.session_id
        child_id = update.child_id
        child = self._resolve_child(child_id) if self._resolve_child else None
        if child_id not in self._parents and child_id != parent_id:
            self._parents[child_id] = parent_id
            announce: dict[str, Any] = {
                "sessionUpdate": "subagent_update",
                "sessionId": child_id,
                "title": update.name or "helper",
                "capabilities": {"cancel": {}},
            }
            options = getattr(child, "options", None)
            workspace = getattr(options, "workspace", None)
            if workspace is not None and Path(workspace).is_absolute():
                announce["cwd"] = str(workspace)
            self._enqueue(announce, None if parent_id == self.session_id else parent_id)
        self._helper_names[child_id] = update.name or "helper"
        self._children.append(
            {
                "sessionId": child_id,
                "name": update.name,
                "depth": update.depth,
                "retained": update.retained,
            }
        )
        self._enqueue(
            {
                "sessionUpdate": "session_info_update",
                "_meta": {"dev.nooa/children": list(self._children)},
            }
        )
        if child is not None:
            self._mirror(child)

    def _child_message(self, child_id: str, message_id: str, text: str) -> None:
        """A message the parent sent a child: a ``session_message`` in the child's session."""
        parent = self._parents.get(child_id, self.session_id)
        self._enqueue(
            {
                "sessionUpdate": "session_message",
                "messageId": message_id or v2.new_message_id(),
                "senderSessionId": parent,
                "recipientSessionId": child_id,
                "content": [v2.text(text)],
            },
            child_id,
        )
        if child_id not in self._described and text.strip():
            self._described.add(child_id)
            self._enqueue(
                {
                    "sessionUpdate": "subagent_update",
                    "sessionId": child_id,
                    "description": text[:500],
                },
                None if parent == self.session_id else parent,
            )

    # ---- agent events ------------------------------------------------

    def _on_agent_message(self, event: EventBase) -> None:
        if not isinstance(event, AgentMessage):
            return
        self._enqueue_all(agent_message(event.content))
        self._move_rolling_below()

    def _move_rolling_below(self) -> None:
        """Keep concise activity below the newest message.

        A client shows a card where it was started and changes it in place, so
        a rolling card updated after a message would stay above that message
        and the agent would look idle. A slot whose cell has finished is
        dropped: the next cell starts a new card below. A cell still running
        moves: its old card is closed with a note, and a new card below the
        message gets its later output.
        """
        for source, slot in list(self._rolling.items()):
            card = self._open.get(slot.key)
            if card is None or card.wire_id != slot.wire_id:
                del self._rolling[source]
                continue
            target = self._target(slot.key)
            self._enqueue(
                v2.tool_call(
                    slot.wire_id,
                    status="completed",
                    content=_python_content(card.code, "Continued below."),
                ),
                target,
            )
            slot.wire_id = card.wire_id = f"concise-python-{uuid4()}"
            self._enqueue(
                v2.tool_call(
                    card.wire_id,
                    title=self._card_title(slot.key, card),
                    kind="other",
                    status="in_progress",
                    content=_python_content(card.code),
                    raw_input={"code": card.code},
                ),
                target,
            )

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
        repeated = key in self._rolling_seen
        self._rolling_seen.add(key)
        if self.verbosity == "silent":
            return
        if self.verbosity == "concise":
            # Reused execution ids cannot distinguish old/new results. Ignore
            # repeated admissions rather than pair stale output with new source.
            if repeated:
                return
            # Cells replace only their own source's display slot. Concurrent
            # helpers retain separate cards and cannot overwrite one another.
            slot = self._rolling.get(key[0])
            first = slot is None
            if slot is None:
                slot = _RollingSlot(f"concise-python-{uuid4()}", key)
                self._rolling[key[0]] = slot
            else:
                self._open.pop(slot.key, None)
                slot.key = key
            card = _OpenCard("python", code=code, wire_id=slot.wire_id)
            self._open[key] = card
            title = self._card_title(key, card)
            self._enqueue(
                v2.tool_call(
                    slot.wire_id,
                    title=title,
                    kind="other" if first else None,
                    status="in_progress",
                    content=_python_content(code),
                    raw_input={"code": code},
                ),
                self._target(key),
            )
            return
        self._open[key] = _OpenCard("python", code=code)
        self._enqueue(
            v2.tool_call(
                key[1],
                title=_python_title(code),
                # Zed 1.14 treats every ``execute`` tool as a terminal card.
                # A plain-content execute card has neither a terminal nor an
                # output disclosure, so its source cannot be opened. Python is
                # structured Markdown content, not a client-owned terminal.
                kind="other",
                status="in_progress",
                content=_python_content(code),
                raw_input={"code": code},
            ),
            self._target(key),
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
            title = self._card_title(
                key, card, "cancelled" if cancelled else "failed" if status == "failed" else None
            )
            self._enqueue(
                v2.tool_call(
                    card.wire_id or key[1],
                    title=title,
                    status=status,
                    content=_python_content(code, output),
                    raw_input={"code": code},
                ),
                self._target(key),
            )
        if source is None:
            self._publish_plan()

    def _on_file_edit(self, event: EventBase, source: str | None) -> None:
        if self.verbosity != "verbose" or not isinstance(event, FileEdit):
            return
        key = self._key(f"file-edit-{uuid4()}", source)
        path = event.path
        title = f"{'Created' if event.operation == 'create' else 'Edited'} {Path(path).name}"
        content: list[Any]
        if event.content_complete and os.path.isabs(path):
            content = [v2.diff(path, event.old_text, event.new_text, start_line=event.start_line)]
        elif event.content_complete:
            content = [v2.tool_text(f"Edited {path}")]
        else:
            content = [v2.tool_text(event.diff or "File content was truncated.")]
            if event.diff and not event.diff_complete:
                content.append(
                    v2.tool_text("The diff was truncated: the edit is larger than shown.")
                )
        # v2 lines are 1-based, like the event's.
        location: dict[str, Any] = {"path": path}
        if event.start_line is not None:
            location["line"] = max(1, event.start_line)
        self._enqueue(
            v2.tool_call(
                key[1],
                title=title,
                kind="edit",
                status="completed",
                content=content,
                locations=[location] if os.path.isabs(path) else None,
                raw_input={"path": path, "operation": event.operation},
                raw_output={
                    "content_complete": event.content_complete,
                    "diff_complete": event.diff_complete,
                },
            ),
            self._target(key),
        )

    def _on_terminal_started(self, event: EventBase, source: str | None) -> None:
        if self.verbosity != "verbose" or not isinstance(event, TerminalCommandStarted):
            return
        key = self._key(event.command_id, source)
        note = _TRUNCATION_NOTES.get((event.command_truncated, event.stdin_truncated))
        self._open[key] = _OpenCard("terminal", note=note)
        self._enqueue(
            v2.tool_call(
                key[1],
                title=f"$ {event.command}",
                kind="execute",
                status="in_progress",
                content=self._terminal_content(key, None),
                raw_input={
                    "command": event.command,
                    "working_directory": event.working_directory,
                    "command_truncated": event.command_truncated,
                    "stdin_truncated": event.stdin_truncated,
                },
            ),
            self._target(key),
        )

    def _terminal_content(self, key: ToolKey, output: str | None) -> list[Any] | None:
        """A terminal card's content: its truncation note, if any, then ``output``."""
        card = self._open.get(key)
        texts = [card.note if card is not None else None, output]
        content = [v2.tool_text(text) for text in texts if text is not None]
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
            v2.tool_call(key[1], status="in_progress", content=self._terminal_content(key, output)),
            self._target(key),
        )

    def _on_terminal_finished(self, event: EventBase, source: str | None) -> None:
        if not isinstance(event, TerminalCommandFinished):
            return
        key = self._key(event.command_id, source)
        card = self._open.get(key)
        if card is None:
            return
        output = card.output
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
            v2.tool_call(
                key[1],
                status="failed" if failed else "completed",
                content=content,
                raw_output={
                    "exit_code": event.exit_code,
                    "timed_out": event.timed_out,
                    "output_truncated": event.output_truncated,
                },
            ),
            self._target(key),
        )

    def _on_llm_response(self, event: EventBase) -> None:
        if self.verbosity != "verbose" or not isinstance(event, LLMResponse):
            return
        reasoning = event.reasoning
        if reasoning:
            self._enqueue_all(v2.message("agent_thought", reasoning))

    def _on_summary(self, event: EventBase) -> None:
        if not isinstance(event, Summary) or event.summary_text is None:
            return
        body = event.summary_text if self.verbosity == "verbose" else None
        self._status_card(f"Summarized context ({event.summary_tag})", body)

    def _status_card(self, title: str, body: str | None = None) -> None:
        """A finished card for something the agent did, not said; none when silent."""
        if self.verbosity == "silent":
            return
        self._enqueue(
            v2.tool_call(
                f"status-{uuid4()}",
                title=title,
                kind="other",
                status="completed",
                content=[v2.tool_text(body)] if body else None,
            )
        )

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
            {
                "sessionUpdate": "usage_update",
                "used": used,
                "size": max(context_window, used),
                "cost": {"amount": totals.cost_usd, "currency": "USD"},
                "_meta": meta,
            }
        )

    def _publish_plan(self) -> None:
        """Send the agent's plan (``Session.plan()``) as an ACP plan update when it changed."""
        entries = [
            {"content": entry.content, "priority": entry.priority, "status": entry.status}
            for entry in self.session.plan()
        ]
        if entries == self._plan:
            return
        self._plan = entries
        self._enqueue(v2.plan_update(entries))

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
            elif isinstance(item, _InputEvent) and not item.future.done():
                item.future.set_exception(self._stopped_error(cause))
            self._queue.task_done()

    def _fail_client_inputs(self, cause: BaseException) -> None:
        for registration in self._client_inputs.values():
            if not registration.future.done():
                registration.future.set_exception(self._stopped_error(cause))

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
            self._fail_client_inputs(exc)
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
                            self._discard_rolling()
                            item.set_exception(error)
                    continue
                if isinstance(item, _InputEvent):
                    if self._error is not None:
                        if not item.future.done():
                            item.future.set_exception(self._error)
                        continue
                    try:
                        await self.client.update(self.session_id, item.value)
                    except BaseException as exc:
                        if isinstance(exc, Exception):
                            self._error = exc
                            self._discard_rolling()
                            error = exc
                        else:
                            error = self._stopped_error(exc)
                        if not item.future.done():
                            item.future.set_exception(error)
                        if not isinstance(exc, Exception):
                            raise
                    else:
                        if not item.future.done():
                            item.future.set_result(None)
                    continue
                rolling_epoch = None
                if isinstance(item, _RollingUpdate):
                    if (
                        item.epoch in self._invalid_rolling
                        or item.transport_generation != self._transport_generation
                    ):
                        continue
                    rolling_epoch = item.epoch
                    item = item.value
                if isinstance(item, _BestEffortUpdate):
                    try:
                        await self._send(item.value)
                    except Exception:
                        logger.debug(
                            "ACP client rejected best-effort session bootstrap update",
                            exc_info=True,
                        )
                elif self._error is None:
                    try:
                        await self._send(item)
                    except Exception as exc:
                        # Skip the rest of this turn's updates rather than
                        # hammering a transport that just failed; the next
                        # flush reports the error and resets.
                        self._error = exc
                        if rolling_epoch is not None:
                            self._invalid_rolling.add(rolling_epoch)
                        self._discard_rolling()
                        logger.warning(
                            "ACP session %s dropped an update after a failed send",
                            self.session_id,
                            exc_info=True,
                        )
            finally:
                self._queue.task_done()

    async def _send(self, out: _Out) -> None:
        if out.session_id is None or out.session_id == self.session_id:
            await self.client.update(self.session_id, out.update)
        else:
            await self.client.update(out.session_id, out.update, mirrored=True)

    async def flush(self) -> None:
        self._drain()
        if self._closed:
            return
        if self._pump_failure is not None:
            raise self._stopped_error(self._pump_failure)
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_observe_future)
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
                v2.tool_call(
                    card.wire_id or key[1],
                    # A Python card keeps its code in the title, with the reason.
                    title=self._card_title(key, card, (title or "interrupted").lower())
                    if card.kind == "python"
                    else title,
                    status="failed",
                    content=content,
                ),
                self._target(key),
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
        self._drain()
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
        with self._pending_lock:
            self._accepting_updates = False
        self._drain()
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
        self._rolling.clear()
        with suppress(Exception):
            await self.flush()
        self._closed = True
        self._fail_client_inputs(RuntimeError("closed"))
        self._queue.put_nowait(_STOP)
        with suppress(BaseException):
            await self._pump_task
