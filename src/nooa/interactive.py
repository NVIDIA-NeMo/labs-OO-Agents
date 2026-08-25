# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive agent base: queue-driven turn loop, persistent vars, summarization.

``InteractiveAgent`` is the base class for agents driven by an outer
dispatcher — a terminal UI, a harness, or any host that feeds input
queues and re-enters ``handle()`` once per notification. It provides:

* ``self.user_messages`` (plus ``slash_commands`` / ``system_messages``)
  input queues via ``QueueManager``,
* ``self.v`` — snapshot-backed persistent variables that survive turns
  and sessions,
* ``message()`` — send a Markdown message to the user,
* ``rename_session()`` — update the host's current session title,
* the ``handle()`` → ``RespondResult`` turn protocol,
* token-budget history summarization (``install_summarizer`` /
  ``apply_model_limits``).

``nooa_cli.coding.CodingAgent`` adds the shared coding tools used by
interactive hosts such as the TUI and ACP.
"""

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from nooa import hidden, strategy
from nooa.agentdoc import doc
from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role
from nooa.storage.markers import nosnapshot
from nooa.storage.snapshot_vars import SnapshotVars

with hidden:
    import importlib
    import importlib.util
    from collections.abc import Callable

    from nooa import Agent
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig  # noqa: F401
    from nooa.runtime.channels import Channel, QueueManager, _ChannelReader
    from nooa.runtime.producers_skill import ProducersSkill
    from nooa.strategies import CodeActStrategy
    from nooa.tools.web_publisher import WebPublisher

# Standard library — all visible in REPL
import asyncio  # noqa: F401
import datetime  # noqa: F401
import json  # noqa: F401
import re  # noqa: F401

from nooa.runtime import producers  # noqa: F401
from nooa.runtime.producers import after, cron, monitor, run_job, tail  # noqa: F401

# os is used by this module (NEMO_OO_RICH_URL check) but not useful to expose to
# the agent's REPL — hide it so doc(self) / exec_globals don't advertise it.
with hidden:
    import os

@hidden
class _LazyOptionalImport:
    """Proxy an optional REPL convenience import until generated code uses it."""

    def __init__(self, module_name: str, attr_name: str | None = None) -> None:
        self._module_name = module_name
        self._attr_name = attr_name
        self._loaded: Any | None = None

    def _load(self) -> Any:
        if self._loaded is None:
            module = importlib.import_module(self._module_name)
            self._loaded = getattr(module, self._attr_name) if self._attr_name else module
        return self._loaded

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)

    def __dir__(self) -> list[str]:
        try:
            return sorted(set(super().__dir__()) | set(dir(self._load())))
        except ImportError:
            return super().__dir__()

    def __repr__(self) -> str:
        target = f"{self._module_name}.{self._attr_name}" if self._attr_name else self._module_name
        if self._loaded is None:
            return f"<lazy import {target}>"
        return repr(self._loaded)


@hidden
def _optional_import(module_name: str, attr_name: str | None = None) -> _LazyOptionalImport | None:
    try:
        if importlib.util.find_spec(module_name) is None:
            return None
    except (ImportError, ValueError):
        return None
    return _LazyOptionalImport(module_name, attr_name)


# Optional third-party libraries — visible in REPL (use np, pd, px, go directly).
# Keep the names available without charging import cost before the first prompt.
with hidden:
    _np = _optional_import("numpy")
    _pd = _optional_import("pandas")
    _px = _optional_import("plotly.express")
    _go = _optional_import("plotly.graph_objects")
    _make_subplots = _optional_import("plotly.subplots", "make_subplots")
    _scipy = _optional_import("scipy")
    _sklearn = _optional_import("sklearn")

if _np is not None:
    np = _np  # noqa: F401
if _pd is not None:
    pd = _pd  # noqa: F401
if _px is not None:
    px = _px  # noqa: F401
if _go is not None:
    go = _go  # noqa: F401
if _make_subplots is not None:
    make_subplots = _make_subplots  # noqa: F401
if _scipy is not None:
    scipy = _scipy  # noqa: F401
if _sklearn is not None:
    sklearn = _sklearn  # noqa: F401

with hidden:
    from nooa.unifiedllm import UnifiedLLM


class RespondReason(StrEnum):
    """Reason/action returned by ``handle()`` at the end of a turn."""

    DONE = "DONE"
    NEED_INPUT = "NEED_INPUT"
    WAIT = "WAIT"
    GET_USER_INPUT = "GET_USER_INPUT"


RespondKind = Literal["DONE", "NEED_INPUT", "WAIT", "GET_USER_INPUT"]


class RespondResult(BaseModel):
    """Return value for ``handle()`` — signals what the outer loop should do next.

    Fields:

    - ``kind`` — reason/action enum:
        * ``RespondReason.DONE`` — the current request is complete.
        * ``RespondReason.NEED_INPUT`` — the agent asked a question or needs
          human input before continuing the current request.
        * ``RespondReason.WAIT`` — the agent is waiting for a background job or
          non-user queue/event before it can continue.
        * ``RespondReason.GET_USER_INPUT`` — legacy spelling for waiting on
          human input; prefer ``DONE`` or ``NEED_INPUT``.

      All stop reasons use the same dispatcher wake path: race every declared
      queue/event channel and re-enter ``handle()`` with the first arrival.
      ``kind`` records why the agent stopped; it does not choose a different
      queue primitive.
    - ``explanation`` — short internal status note about why this turn ended.
      This is NOT the reply to the user; it renders as a terse one-line status
      (e.g. ``∴ done: <explanation>``), not as a chat reply.

      User-facing content — answers, greetings, questions for the user — MUST
      be sent via ``self.message(...)`` BEFORE ``return_result``. The rule per
      ``kind``:

        * ``DONE`` — if this turn was handling a user message, you almost
          certainly owe the user a reply: call ``self.message(...)`` first.
          Exceptions: the request was satisfied by a pure side effect the user
          can already see (a slash-command mutation, a UI state change), or you
          already sent a ``self.message(...)`` earlier in the turn.
        * ``NEED_INPUT`` — you are asking the user something. ALWAYS call
          ``self.message(...)`` with the question first; the ``explanation`` is
          not the question.
        * ``WAIT`` — no user reply expected. ``explanation`` IS the intended
          surface: name the job/queue being waited on and why (e.g. "waiting
          on build job #422 to finish before running tests").

      Keep ``explanation`` terse and factual either way — it's a status line
      for the host log, not prose.


    Use ``self.v.<name> = value`` for state that should survive across
    turns (snapshot-backed).

    Correct pattern for a user-message reply — send the reply first, THEN
    signal turn end::

        self.message("Hi! I'm the coding agent. What would you like to work on?")
        return_result(RespondReason.DONE, explanation="greeted the user")

    WRONG — putting the reply in ``explanation`` (the user will not see it as
    a chat reply; only a terse status line is shown)::

        # BAD: no self.message() call, reply stuffed into explanation
        return_result(
            RespondReason.DONE,
            explanation="Hi! I'm the coding agent. What would you like to work on?",
        )

    Correct pattern for WAIT — no ``self.message()`` needed; ``explanation``
    is the status the UI shows::

        return_result(RespondReason.WAIT, explanation="waiting on build job #422")

    The older explicit model form is still valid::

        return_result(RespondResult(kind="DONE", explanation="greeted the user"))
    """

    kind: RespondReason = Field(description="What the outer dispatcher should do next")
    explanation: str = Field(
        min_length=1,
        description=(
            "Short internal status note about why this turn ended (e.g. 'answered "
            "the request', 'waiting on build job #422'). NOT shown to the user as "
            "a chat reply — it renders as a terse status line. For DONE and "
            "NEED_INPUT, send user-facing content via self.message(...) BEFORE "
            "calling return_result; do not put the answer, greeting, or question "
            "here. For WAIT, explanation IS the surfaced status — name the "
            "job/queue being waited on."
        ),
    )

    @field_validator("explanation")
    @classmethod
    def _explanation_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("explanation is required")
        return value

    model_config = {"arbitrary_types_allowed": True}


class AgentMessage(Metadata):
    """A Markdown message sent by the agent to the user via ``self.message()``.

    Interactive hosts share this event so durable sessions can be replayed
    without knowing which frontend produced the response.
    """

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""


class SummarizationConfig(BaseModel):
    """Configuration for history summarization.

    ``max_tokens`` defaults to ``None`` meaning "80% of the LLM's context
    window, resolved at install time." The old 100K absolute was fine when
    models had ~200K context but fired at ~10% usage on 1M-context models
    like Opus 4.8, making summarization feel constant. Set an explicit
    integer to pin a specific threshold.
    """

    policy: Literal["token_budget", "none"] = "token_budget"
    max_tokens: int | None = None
    preserve_recent: int = 10
    target_chars: int = 4000


# Default model used when an agent class is defined without an explicit LLM
# (overridden at instantiation).
DEFAULT_MODEL = "claude-opus-4-8"

with hidden:
    try:
        from nooa.unifiedllm import get_llm_client

        _DEFAULT_LLM = get_llm_client(DEFAULT_MODEL)
    except Exception:
        from nooa.unifiedllm import FakeLLMClient

        _DEFAULT_LLM = FakeLLMClient()


# Summarizer trigger as a fraction of the LLM's context window. This is the
# ONLY budget managed here — event-pile truncation is enforced at the
# runtime level (see ActorRuntime._build_messages) and adapts to whichever
# LLM is actually resolved for each call (including per-call overrides).
_SUMMARIZER_BUDGET_PCT = 0.8


def _summarizer_budget(llm: "UnifiedLLM") -> int:
    """Resolve the summarizer trigger from the LLM's context window.

    Falls back to 100K when the LLM doesn't expose ``context_window`` so
    we still have a functional threshold.
    """
    cw = getattr(llm, "context_window", None)
    return int(cw * _SUMMARIZER_BUDGET_PCT) if cw else 100_000


def apply_model_limits(agent: Agent) -> None:
    """Sync the summarizer trigger against ``agent.llm.context_window``.

    Call after a model switch so the summarizer threshold moves with the
    new context window. Runtime-level event truncation picks up the new
    window automatically on the next ``_build_messages`` call.
    """
    from nooa.config.summarizer_config import TokenBudgetConfig

    summarizer_max = _summarizer_budget(agent.llm)
    for summarizer in getattr(agent, "_summarizers", []):
        current = summarizer.config
        summarizer.config = TokenBudgetConfig(
            max_tokens=summarizer_max,
            preserve_recent=current.preserve_recent,
            target_chars=current.target_chars,
        )


def install_summarizer(config: SummarizationConfig, agent: Agent) -> None:
    """Install a summarizer on the agent based on configuration.

    Args:
        config: Summarization configuration. ``config.max_tokens=None`` (the
            default) resolves to 80% of the agent LLM's context window at
            install time, so the trigger scales with model capability.
        agent: Agent to install summarizer on (inherits LLM, attaches to history)
    """
    if config.policy == "none":
        return

    from nooa.config.summarizer_config import TokenBudgetConfig

    summarizer_max = (
        config.max_tokens if config.max_tokens is not None else _summarizer_budget(agent.llm)
    )

    TokenBudgetSummarizer.install(
        agent,
        config=TokenBudgetConfig(
            max_tokens=summarizer_max,
            preserve_recent=config.preserve_recent,
            target_chars=config.target_chars,
        ),
    )


class AgentVars:
    """Attribute-access proxy for an agent's persistent ``vars`` dict.

    Mirrors ``TodoVars``: write ``self.v.spec = "..."`` instead of
    ``self.vars["spec"] = "..."``. Reads and writes go straight
    through to ``self.vars`` so snapshot serialization is unaffected.

    Use for variables that need to survive across turns and across
    sessions but aren't tied to a specific todo. (For per-todo state,
    use ``self.todo.<id>.v`` — same shape, narrower scope.)

    Values are snapshot-backed: assigning something that can't be
    snapshot-serialized (a live client, socket, callable, ...) logs a
    warning and is **not stored** — it won't survive ``/exit`` + resume.
    Store serializable data (dict/str/number/Pydantic model) instead.
    """

    def __init__(self, agent: Any):
        object.__setattr__(self, "_agent", agent)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._agent.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r} on agent") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self._agent.vars[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self._agent.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r} on agent") from None

    def __contains__(self, key: str) -> bool:
        return key in self._agent.vars

    def __repr__(self) -> str:
        return repr(self._agent.vars)


@hidden
class InteractiveAgent(Agent, llm=_DEFAULT_LLM):
    """Base class for agents driven by an outer dispatcher (TUI, harness, ...).

    Subclass this and implement ``handle()`` to build a custom interactive
    agent. ``message()`` and ``rename_session()`` are provided for free.

    **Input queues.** Every ``InteractiveAgent`` has a ``self.user_messages``
    queue (``InputQueue``) that the host feeds when the human types.
    Subclasses may declare additional queues as instance attributes for
    other producers (long-running job output, monitor streams, etc.).

    Each queue is two objects: an ``InputQueue`` (full producer +
    dispatcher API, hidden from the LLM under ``_<name>_in``) and an
    ``OutputQueue`` reader facade exposed under the public name. The
    LLM can ``await self.user_messages.get()`` to dequeue mid-turn;
    everything else (put, snapshot, qsize, etc.) is dispatcher-only.

    ``handle()`` runs *per turn* — the outer dispatcher calls it with
    the next notification (a ``dict[str, list]``). Use ``self.v`` for
    state that should survive across turns.
    """

    _render_message: Annotated[Callable[[str], None] | None, hidden, nosnapshot]
    _session_manager: Annotated[Any | None, hidden, nosnapshot]
    # QueueManager owns the channel registry. Hidden from the LLM by
    # default — the LLM should access individual channels (e.g.
    # ``self.user_messages``) directly, not through a string-keyed
    # registry lookup.
    queue_manager: Annotated[QueueManager, hidden, nosnapshot]
    # Producer-side Channel (full put / pop_last / snapshot / etc.)
    # — hidden, since the LLM has no business calling those.
    _user_messages_in: Annotated[Channel, hidden, nosnapshot]
    # Slash command results — posted by the host when a @slash_command returns.
    _slash_commands_in: Annotated[Channel, hidden, nosnapshot]
    # Host-owned system/internal prompts, e.g. keep-going continuations or
    # harness instructions.
    _system_messages_in: Annotated[Channel, hidden, nosnapshot]
    # Read-only facade (just .get() / .status() / .name) is what the
    # LLM sees as ``self.user_messages``.
    user_messages: Annotated[_ChannelReader, nosnapshot]
    # Slash command results — the agent receives SlashCommandResult objects here.
    slash_commands: Annotated[_ChannelReader, nosnapshot]
    # System/internal prompts generated by the host.
    system_messages: Annotated[_ChannelReader, nosnapshot]
    # Persistent variables for the LLM — survives across turns AND
    # across sessions (snapshot-backed). Accessed via the ``self.v``
    # proxy for dot-attribute reads/writes (``self.v.spec = "..."``).
    vars: SnapshotVars

    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm or _DEFAULT_LLM, **kwargs)
        self._render_message = None
        self._session_manager = None
        self.vars = SnapshotVars()
        self.queue_manager = QueueManager(event_manager=self.event_manager)
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.user_messages = self._user_messages_in.reader
        self._slash_commands_in = self.queue_manager.queue("slash_commands")
        self.slash_commands = self._slash_commands_in.reader
        self._system_messages_in = self.queue_manager.queue("system_messages")
        self.system_messages = self._system_messages_in.reader
        self.producers = ProducersSkill()
        # Surface pending-queue counts (and a short preview of each item)
        # to the LLM every turn — the agent reads queue depth straight
        # from the ``queues`` context block. Composed via
        # ``QueueManager.status()`` so adding new channels Just Works.
        from nooa import Context

        self.context["queues"] = Context(expr="self.queue_manager.status()")
        if os.environ.get("NEMO_OO_RICH_URL"):
            from nooa.tools.web_publisher import RichOutput

            self.event_manager.register_event_type(RichOutput)
            self.web: Annotated[WebPublisher, nosnapshot] = WebPublisher(
                event_manager=self.event_manager
            )
            # The WebPublisher's doc is static across the session, so
            # it goes into the cacheable prefix with the system prompt.
            from nooa import Context

            self.context["web"] = Context(doc(self.web), prefix=True)

    @property
    def v(self) -> AgentVars:
        """Attribute-access proxy for ``self.vars`` — the agent's
        persistent variable dict.

        Use for state that should survive across turns and sessions
        but isn't tied to a specific todo. Snapshot-backed via
        ``self.vars``.

        Usage::

            self.v.spec = "implement queue mode"
            self.v.cursor = 0
            print(self.v.spec)
            del self.v.cursor

        Compare:
        - REPL locals → cleared between turns.
        - ``self.v.k = v`` → snapshot-backed, survives turns + sessions.
        - ``self.todo.<t>.v.k = v`` → same as ``self.v`` but scoped to
          one todo.
        """
        return AgentVars(self)

    def message(self, text: str, *, echo: bool = False) -> None:
        """Send a Markdown message to the user.

        Each call renders as an independent block — so every call must be a
        complete, self-contained Markdown document.  In particular, never split
        a table across calls: the header row and all data rows must be in the
        same ``message()`` call, otherwise the table will not render correctly.

        Args:
            text: Markdown content to send.
            echo: If True, also ``print()`` the text so the LLM can see it
                in the execution output. By default the LLM does NOT see the
                content of ``message()`` calls — only the user does. Use
                ``echo=True`` when you need to reference the sent content in
                subsequent cells.
        """
        event = AgentMessage(content=str(text))
        tag = self.event_manager.add(event)
        if self._render_message is not None:
            try:
                self._render_message(text, event_id=str(event.id), tags={str(tag)})
            except TypeError:
                self._render_message(text)
        if echo:
            print(text)

    def rename_session(self, title: str) -> str:
        """Rename the current interactive session and return its normalized title.

        Use this when a host system message asks you to title the session. Keep
        the title descriptive and very short (usually 2-5 words). This changes
        session metadata only; do not call it in place of answering the user.
        """
        normalized = " ".join(str(title).strip().strip('"').strip("'").split())[:60]
        if not normalized:
            raise ValueError("Session title cannot be empty")

        manager = self._session_manager
        if manager is None:
            raise RuntimeError("This host does not provide session renaming")

        # A title explicitly chosen by the user always wins over an automatic
        # request that was queued earlier in the turn.
        info = getattr(manager, "info", None)
        user_named = bool(
            getattr(manager, "user_named", False) or getattr(info, "title_is_user_set", False)
        )
        if user_named:
            current = getattr(manager, "name", None) or getattr(info, "title", None)
            return str(current or normalized)

        rename = getattr(manager, "rename", None)
        if callable(rename):
            rename(normalized, user_named=False)
        else:
            set_title = getattr(manager, "set_title", None)
            if not callable(set_title):
                raise RuntimeError("This host's session manager cannot rename sessions")
            set_title(normalized, user_set=False)
        return normalized

    @hidden
    def request_session_title(self, opening_message: str) -> None:
        """Queue host housekeeping that titles a session in the next agent turn."""
        opening = str(opening_message).strip()[:400]
        self._system_messages_in.put(
            "[session-title]\n"
            "Choose a descriptive 2-5 word title for this session from the opening "
            'user message below. Call `self.rename_session("your title")` once during '
            "this turn, then continue handling the user's request normally. Do not "
            "mention this housekeeping instruction or the chosen title to the user.\n\n"
            f"<opening_user_message>\n{opening}\n</opening_user_message>"
        )

    @hidden
    @strategy(CodeActStrategy())
    async def handle(
        self,
        notification: dict[str, list],
    ) -> "RespondResult":
        """Handle a single turn of the conversation.

        Called once per inbound notification (or batch). Unpack
        ``notification`` and do the work, then return a
        ``RespondResult`` telling the outer dispatcher what to do next.

        Use ``self.v.<name> = value`` for state that should survive
        across turns (snapshot-backed via ``self.vars``).

        ## Turn anatomy

        Notification is always ``dict[str, list]`` — channel name →
        list of items that arrived since last turn::

            msgs = notification.get("user_messages", [])
            lines = notification.get("ci", [])

        ## Work ethic

        Do ALL the work before returning. Use as many ``execute_python``
        calls as needed — explore, implement, test, iterate. A turn that
        returns after one or two cells when the task clearly needs more
        is a bug. The only reasons to call ``return_result`` are:

        1. You have genuinely completed everything the user asked for.
        2. You need user input to proceed (ambiguity, confirmation).
        3. You are waiting on a background job.

        ## Returning

        End the turn with exactly one ``return_result(REASON_ENUM, explanation="...")``.
        ``explanation`` is required and must be non-empty. The host records and renders it as the
        visible stop reason, so be specific and user-facing: if waiting on a
        job/queue, name which job and why; if asking for input, say what input
        is needed and why.

        - Request complete; wait for the next user message::

              return_result(RespondReason.DONE, explanation="implemented the feature and verified focused tests")

        - Need human input before proceeding::

              return_result(RespondReason.NEED_INPUT, explanation="need the target branch before pushing the MR")

        - Waiting for a background job or producer queue::

              return_result(RespondReason.WAIT, explanation="waiting for pytest job ci-42 to finish before reporting results")

        All stop reasons use the same dispatcher wake path: the host races every
        declared queue/event channel and re-enters ``handle()`` with the first
        arrival. Use ``kind`` to say why you are stopping, not to select a
        different queue primitive.



        ## Available queues

        The dispatcher delivers the next item via ``notification``.
        You can also dequeue extra items mid-turn (without ending it)
        by awaiting the queue directly::

            extra = await self.user_messages.get()  # blocks until next message

        Recognized ``queue_name`` values:

        - ``"user_messages"`` — text from the human; ``item`` is a str.
        - ``"slash_commands"`` — results from slash commands; ``item`` is a
          ``SlashCommandResult(command, args, value, text)``. Access the
          raw return value via ``item.value``; ``str(item)`` gives the text.
        - ``"system_messages"`` — host-owned system/internal prompts,
          such as keep-going continuations or harness instructions.

        Subclasses may add more (e.g. ``"job_outputs"``); the
        ``<queue_status>`` context block lists the pending count per
        queue each turn. After any stop reason, the dispatcher races every
        declared queue/event and re-enters with the first arrival.
        """
        ...
