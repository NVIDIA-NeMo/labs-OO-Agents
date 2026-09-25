# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive agent base: queue-driven turn loop, persistent vars, summarization.

``InteractiveAgent`` is the base class for agents driven by an outer
dispatcher — a terminal UI, a harness, or any host that feeds input
queues and re-enters ``handle()`` once per notification. It provides:

* ``self.user_messages`` via ``QueueManager``; hosts declare whatever further
  queues they need (``self.queue_manager.queue("name")``),
* ``self.v`` — snapshot-backed persistent variables that survive turns
  and sessions,
* ``message()`` — send a Markdown message to the user,
* the turn protocol: ``handle()`` returns ``Done``, ``NeedInput`` or
  ``Waiting`` (``RespondResult`` is the older form, still accepted);
  ``handle_batch()`` runs unattended turns and returns ``Done`` or ``Waiting``,
* token-budget history summarization (``install_summarizer`` /
  ``apply_model_limits``).

``nooa_cli.coding.CodingAgent`` adds the shared coding tools used by
interactive hosts such as the TUI and ACP.
"""

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from nooa import hidden, strategy
from nooa.agentdoc import doc
from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role
from nooa.storage.markers import nosnapshot
from nooa.storage.snapshot_vars import SnapshotVars

with hidden:
    from collections.abc import Callable

    from nooa import Agent
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig, PredictConfig  # noqa: F401
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

# Optional third-party libraries — visible in REPL (use np, pd, px, go directly)
try:
    import numpy as np  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import pandas as pd  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import plotly.express as px  # noqa: F401  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # noqa: F401  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import scipy  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

try:
    import sklearn  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    pass

with hidden:
    from nooa.unifiedllm import UnifiedLLM


def _non_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


def _blank_to_none(value: str | None) -> str | None:
    return None if value is None else (value.strip() or None)


class Done(BaseModel):
    """Turn result: the work for this turn is finished.

    ``message`` is the preferred way to answer the user on a turn that
    handled a user message: put the reply here and the host shows it.
    ``self.message()`` remains for text you want to show before the turn
    ends. ``explanation`` stays a short status line. ``evidence`` is
    optional: short factual lines a reviewer can check, used when you
    verified something. ``result`` is set only when a delegated objective
    or benchmark task completes; it is then a ``TaskResult`` (defined with
    the bench and session code, so it is typed ``Any`` here).
    """

    explanation: str = Field(description="Status line saying what was finished; not the reply")
    message: str | None = Field(default=None, description="The reply to show the user")
    evidence: list[str] = Field(
        default_factory=list,
        description='Short checkable facts, e.g. "pytest tests/x.py: 24 passed"',
    )
    result: Any | None = Field(
        default=None, description="TaskResult when a delegated or bench task completes"
    )

    _check_explanation = field_validator("explanation")(_non_blank)
    _check_message = field_validator("message")(_blank_to_none)


class NeedInput(BaseModel):
    """Turn result: a question the agent cannot continue without.

    ``question`` is the question itself; the host shows it to the person,
    so do not also send it with ``self.message()``. Set ``options`` for a
    single choice, or ``answer_type`` (a pydantic model class whose fields
    are simple values: str, int, float, bool, or a list of str) for a typed
    answer, or neither for free text. The host turns ``answer_type`` into a
    form and the answer arrives in the next notification as an instance of
    it. ``reason`` optionally says in one sentence why progress is not
    possible or not desirable without the answer.
    """

    question: str = Field(description="The question to show the person")
    reason: str | None = Field(
        default=None, description="Why progress is not possible or not desirable without it"
    )
    options: list[str] | None = Field(
        default=None, min_length=1, description="Choices for a single choice"
    )
    answer_type: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic model class with simple fields; the answer comes back as an instance",
    )

    _check_question = field_validator("question")(_non_blank)

    @model_validator(mode="after")
    def _one_answer_shape(self) -> "NeedInput":
        if self.options is not None and self.answer_type is not None:
            raise ValueError("set options or answer_type, not both")
        return self


class Waiting(BaseModel):
    """Turn result: waiting on a background job or queue, not on a person.

    ``on`` lists what is being waited on by name: a queue channel
    (``"delegates"``, ``"jobs"``), a spawned job's label, a subagent's name.
    ``explanation`` says why, for the host. ``message`` is an optional line
    the host shows the user. The host keeps the request open and runs the
    next turn when one of them delivers.
    """

    explanation: str = Field(description="Why the turn is waiting")
    message: str | None = Field(default=None, description="Line to show the user while waiting")
    on: list[str] = Field(
        min_length=1, description="Names of the channels, jobs or subagents being waited on"
    )

    _check_explanation = field_validator("explanation")(_non_blank)
    _check_message = field_validator("message")(_blank_to_none)

    @field_validator("on")
    @classmethod
    def _names_not_blank(cls, value: list[str]) -> list[str]:
        return [_non_blank(name) for name in value]


class RespondReason(StrEnum):
    """Reason/action returned by ``handle()`` at the end of a turn (older form)."""

    DONE = "DONE"
    NEED_INPUT = "NEED_INPUT"
    WAIT = "WAIT"
    GET_USER_INPUT = "GET_USER_INPUT"


RespondKind = Literal["DONE", "NEED_INPUT", "WAIT", "GET_USER_INPUT"]


class RespondResult(BaseModel):
    """Older return value for ``handle()`` — signals what the outer loop should do next.

    Still accepted by ``handle()`` so existing hosts and agents keep working;
    new code returns ``Done``, ``NeedInput`` or ``Waiting``.

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
    - ``explanation`` — required non-empty short reason why the agent is ending this
      turn, or what external input/background event it is waiting for. The host
      records and renders this line, so make it concrete: name the job/queue
      being waited on, why it matters, or what user input is needed and why.


    Use ``self.v.<name> = value`` for state that should survive across
    turns (snapshot-backed).

    Build from within the LLM's ``execute_python`` code::

        return_result(
            RespondReason.DONE,
            explanation="answered the request; waiting for the next user message",
        )

    The older explicit model form is still valid::

        return_result(RespondResult(kind="DONE", explanation="answered the request"))
    """

    kind: RespondReason = Field(description="What the outer dispatcher should do next")
    explanation: str = Field(
        min_length=1,
        description=("Required: why handle() returned, or what the dispatcher is waiting for."),
    )

    _check_explanation = field_validator("explanation")(_non_blank)

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

    ``max_tokens=None`` uses ``threshold_fraction`` of the usable input window
    (model window minus the effective reply reserve), resolved for each completed
    request. Set an explicit integer to pin a threshold across model switches.
    """

    policy: Literal["token_budget", "none"] = "token_budget"
    max_tokens: int | None = None
    threshold_fraction: float = Field(default=0.75, gt=0, lt=1)
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
_SUMMARIZER_BUDGET_PCT = 0.75


def _summarizer_budget(
    llm: "UnifiedLLM",
    fallback_reserve: int = 0,
    *,
    threshold_fraction: float = _SUMMARIZER_BUDGET_PCT,
) -> int:
    """Resolve the summarizer trigger from the LLM's usable input window.

    Falls back to 100K when the LLM doesn't expose ``context_window`` so
    we still have a functional threshold.
    """
    from nooa.agents.summarization import context_budget

    return context_budget(llm, threshold_fraction, fallback_reserve=fallback_reserve)


def apply_model_limits(agent: Agent) -> None:
    """Sync automatic summarizers against the selected model's usable window.

    Call after a model switch so the summarizer threshold moves with the
    new context window. Runtime-level event truncation picks up the new
    window automatically on the next ``_build_messages`` call.
    """
    for summarizer in getattr(agent, "_summarizers", []):
        if not getattr(summarizer, "_automatic_context_budget", False):
            continue
        summarizer_max = _summarizer_budget(
            agent.llm,
            agent._truncation.response_reserve_tokens,
            threshold_fraction=summarizer._automatic_context_budget_percent,
        )
        current = summarizer.config
        summarizer.config = current.model_copy(update={"max_tokens": summarizer_max})


def install_summarizer(config: SummarizationConfig, agent: Agent) -> None:
    """Install a summarizer on the agent based on configuration.

    Args:
        config: Summarization configuration. ``config.max_tokens=None`` follows
            ``threshold_fraction`` (75% by default) of each request's usable input window.
        agent: Agent to install summarizer on (inherits LLM, attaches to history)
    """
    if config.policy == "none":
        return

    from nooa.config.summarizer_config import TokenBudgetConfig

    summarizer_max = (
        config.max_tokens
        if config.max_tokens is not None
        else _summarizer_budget(
            agent.llm,
            agent._truncation.response_reserve_tokens,
            threshold_fraction=config.threshold_fraction,
        )
    )

    summarizer = TokenBudgetSummarizer.install(
        agent,
        config=TokenBudgetConfig(
            max_tokens=summarizer_max,
            preserve_recent=config.preserve_recent,
            target_chars=config.target_chars,
        ),
    )
    summarizer._automatic_context_budget = config.max_tokens is None
    summarizer._automatic_context_budget_percent = config.threshold_fraction


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
    agent. ``message()`` is provided for free.

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
    # QueueManager owns the channel registry. Hidden from the LLM by
    # default — the LLM should access individual channels (e.g.
    # ``self.user_messages``) directly, not through a string-keyed
    # registry lookup.
    queue_manager: Annotated[QueueManager, hidden, nosnapshot]
    # Producer-side Channel (full put / pop_last / snapshot / etc.)
    # — hidden, since the LLM has no business calling those.
    _user_messages_in: Annotated[Channel, hidden, nosnapshot]
    # Read-only facade (just .get() / .status() / .name) is what the
    # LLM sees as ``self.user_messages``.
    user_messages: Annotated[_ChannelReader, nosnapshot]
    # Persistent variables for the LLM — survives across turns AND
    # across sessions (snapshot-backed). Accessed via the ``self.v``
    # proxy for dot-attribute reads/writes (``self.v.spec = "..."``).
    vars: SnapshotVars

    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm or _DEFAULT_LLM, **kwargs)
        self._render_message = None
        self.vars = SnapshotVars()
        self.queue_manager = QueueManager(event_manager=self.event_manager)
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.user_messages = self._user_messages_in.reader
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

    @hidden
    @strategy(CodeActStrategy())
    async def handle(
        self,
        notification: dict[str, list],
    ) -> Done | NeedInput | Waiting | RespondResult:
        """Handle one interactive turn.

        Called once per inbound notification (or batch). Unpack
        ``notification``, do the work, then end the turn with one typed
        result. Use ``self.v.<name> = value`` for state that must survive
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
        is a bug. End the turn only when the request is complete, when you
        need an answer from the person, or when you are waiting on a
        background job.

        ## Returning

        A turn has exactly one terminal result: the value you return.
        Everything else is intermediate. End the turn with one
        ``return_result(...)`` of one of these:

        - ``Done(message=..., explanation=...)`` — the request is complete.
          ``message`` is the reply the host shows the user; ``explanation``
          is a short status line. Add ``evidence=[...]`` for checks you ran::

              return_result(Done(message="Added the flag and its test.", explanation="added flag"))

          Use ``self.message()`` only for text to show before the turn ends.

        - ``NeedInput(question=...)`` — you cannot continue without an
          answer. ``question`` is the question; the host shows it, so do
          not also send it with ``message()``. Add ``options=[...]`` for a
          single choice, or ``answer_type=SomeModel`` (a pydantic class you
          define with simple fields) for a typed answer. The answer arrives
          in the next notification. ``reason`` optionally says why you
          need it::

              return_result(NeedInput(question="Which branch should I push to?", options=["main", "dev"]))

        - ``Waiting(message=..., explanation=..., on=[...])`` — waiting on a
          background job or queue, not on a person. ``message`` is shown to
          the user; ``on`` names what you wait for: a channel, a job label,
          a subagent::

              return_result(Waiting(message="Tests are running; I will report when they finish.", explanation="tests running", on=["jobs:ci-42"]))

        ``RespondResult`` is the older form and is still accepted.

        ## Available queues

        The dispatcher delivers the next item via ``notification``.
        You can also dequeue extra items mid-turn (without ending it)
        by awaiting the queue directly::

            extra = await self.user_messages.get()  # blocks until next message

        Recognized ``queue_name`` values:

        - ``"user_messages"`` — text from the human; ``item`` is a str.

        Hosts declare the rest. ``CodingAgent`` adds ``"slash_commands"``
        (``SlashCommandResult``) and ``"system_messages"`` (host-owned
        prompts such as keep-going continuations); a harness might add
        ``"job_outputs"``. The
        ``<queue_status>`` context block lists the pending count per
        queue each turn. After any result, the dispatcher races every
        declared queue/event and re-enters with the first arrival.
        """
        ...

    @hidden
    @strategy(CodeActStrategy())
    async def handle_batch(
        self,
        notification: dict[str, list],
    ) -> Done | Waiting:
        """Handle one unattended turn.

        No person is watching this turn and nobody can answer a question,
        so ``NeedInput`` is not allowed here and is rejected as a type
        error. Unpack ``notification`` (channel name → list of items) and
        do all the work.

        A turn has exactly one terminal result: the value you return.
        Everything else is intermediate. End with one ``return_result(...)``:

        - ``Done(explanation=...)`` — the work is finished, or cannot go
          further. If something blocks you, say what in ``explanation``.
          Set ``result`` when the task asks for a structured result.
        - ``Waiting(message=..., explanation=..., on=[...])`` — waiting on a
          background job or queue you started. ``on`` names it.
        """
        ...
