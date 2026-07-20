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
* the ``handle()`` → ``RespondResult`` turn protocol,
* token-budget history summarization (``install_summarizer`` /
  ``apply_model_limits``).

The TUI's ``BaseTUIAgent`` (packages/nooa-tui) is a thin subclass that
adds TUI-specific tooling and event types on top of this class.
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
    from collections.abc import Callable

    from nooa import Agent
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig, PredictConfig  # noqa: F401
    from nooa.runtime.channels import Channel, QueueManager, _ChannelReader
    from nooa.runtime.producers_skill import ProducersSkill
    from nooa.strategies import CodeActStrategy, PredictStrategy
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

    The TUI emits its own ``TUIAgentMessage`` subtype (its explorers and
    session store match on that exact type name); this is the host-neutral
    event used by ``InteractiveAgent`` outside the TUI.
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
    agent. ``message()`` and ``name_session()`` are provided for free.

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
    # Slash command results — posted by the host when a @slash_command returns.
    _slash_commands_in: Annotated[Channel, hidden, nosnapshot]
    # Host-owned system/internal prompts, e.g. keep-going continuations.
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

    @hidden
    @strategy(PredictStrategy(PredictConfig(output_serialization="tool_call")))
    async def name_session(self, user_message: str) -> str:
        """Generate an ultra-short 2-5 word session title.

        Conversation starts with: {user_message}
        """
        ...

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
          such as keep-going continuations.

        Subclasses may add more (e.g. ``"job_outputs"``); the
        ``<queue_status>`` context block lists the pending count per
        queue each turn. After any stop reason, the dispatcher races every
        declared queue/event and re-enters with the first arrival.
        """
        ...
