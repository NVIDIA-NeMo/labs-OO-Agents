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
* the turn protocol: ``handle()`` returns ``Done``, ``NeedInput``, ``NeedInputForm`` or
  ``Waiting``; ``handle_batch()`` runs unattended turns and returns ``Done`` or ``Waiting``,
* token-budget history summarization (``install_summarizer`` /
  ``apply_model_limits``).

``nooa_cli.coding.CodingAgent`` adds the shared coding tools used by
interactive hosts such as the TUI and ACP.
"""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from nooa import hidden, strategy
from nooa.agentdoc import doc
from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role
from nooa.storage.markers import nosnapshot
from nooa.storage.snapshot_vars import SnapshotVars

with hidden:
    from collections.abc import Callable

    from nooa import Agent, no_trace
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig, PredictConfig  # noqa: F401
    from nooa.events import _json_safe
    from nooa.runtime.channels import Channel, ChannelReader, QueueManager
    from nooa.runtime.producers_skill import ProducersSkill
    from nooa.runtime.turn_loop import TurnLoop
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

    @field_serializer("result", when_used="json")
    def _serialize_result(self, value: Any) -> Any:
        """Keep JSON dumps (the event store) from raising on an arbitrary object."""
        return _json_safe(value)


class NeedInput(BaseModel):
    """Conversational question, answered by the next user message; never a dialog.

    Options are suggestions, not a validation constraint. Use NeedInputForm for
    explicit string-question forms. The host displays the question and optional reason.
    """

    model_config = {"extra": "forbid"}
    kind: Literal["question"] = "question"
    question: str
    reason: str | None = None
    options: list[str] | None = Field(default=None, min_length=1)

    _check_question = field_validator("question")(_non_blank)

    @field_validator("options")
    @classmethod
    def _options_not_blank(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else [_non_blank(option) for option in value]


class FormChoice(BaseModel):
    """A stored string value and the label shown in a picker."""

    model_config = {"extra": "forbid", "strict": True}
    value: str
    title: str
    _check_value = field_validator("value")(_non_blank)
    _check_title = field_validator("title")(_non_blank)


class TextQuestion(BaseModel):
    """One text input. Optional omitted/blank answers become the empty string."""

    model_config = {"extra": "forbid", "strict": True}
    kind: Literal["text"] = "text"
    id: str
    label: str
    help: str | None = None
    required: bool = True
    _check_id = field_validator("id")(_non_blank)
    _check_label = field_validator("label")(_non_blank)


class PickOneQuestion(TextQuestion):
    """Choose exactly one listed value (not its display title)."""

    kind: Literal["pick_one"] = "pick_one"
    choices: list[FormChoice] = Field(min_length=1)

    @field_validator("required")
    @classmethod
    def _required_picker(cls, required: bool) -> bool:
        if not required:
            raise ValueError(
                "Optional picker questions are unsupported by Pool 1.0.16 and abort the whole form. "
                "Use an explicit optional TextQuestion with choices in help (text, not a strict "
                "picker), or a required picker if an answer really is required."
            )
        return required

    @field_validator("choices")
    @classmethod
    def _unique_values(cls, choices: list[FormChoice]) -> list[FormChoice]:
        if len({choice.value for choice in choices}) != len(choices):
            raise ValueError("choice values must be unique")
        return choices


class PickOneOrTextQuestion(PickOneQuestion):
    """Choose a listed value or enter a different string."""

    kind: Literal["pick_one_or_text"] = "pick_one_or_text"


FormQuestion = Annotated[
    TextQuestion | PickOneQuestion | PickOneOrTextQuestion, Field(discriminator="kind")
]


class FormResponse(BaseModel):
    """Form outcome; accepted content is a string dictionary, never a typed model.

    Decline and cancel carry no content. Stop aborts the prompt separately.
    Validate protocol shape against the owning request with validate_response().
    Domain validation belongs to the agent, which can ask a targeted follow-up.
    """

    model_config = {"extra": "forbid", "strict": True}
    action: Literal["accept", "decline", "cancel"]
    content: dict[str, str] | None = None

    @model_validator(mode="after")
    def _content_for_action(self) -> "FormResponse":
        if self.action == "accept" and self.content is None:
            raise ValueError("accept requires content")
        if self.action != "accept" and self.content is not None:
            raise ValueError("decline/cancel must not carry content")
        return self


class NeedInputForm(BaseModel):
    """Explicit ordered string questions, not a permission or conversational question.

    Supported descriptors describe text, pick one, and pick one or text. These
    are experimentally verified Pool mappings, not an inventory of all Pool UI
    types. Pickers must be required: Pool 1.0.16 rejects optional non-text fields
    and aborts the whole form. Use an explicitly authored optional TextQuestion
    instead when a text input is appropriate. No implicit conversions, defaults,
    or domain constraints are applied.
    """

    model_config = {"extra": "forbid", "strict": True}
    kind: Literal["form"] = "form"
    heading: str
    reason: str | None = None
    questions: list[FormQuestion] = Field(min_length=1)
    _check_heading = field_validator("heading")(_non_blank)

    @field_validator("questions")
    @classmethod
    def _unique_ids(cls, questions: list[FormQuestion]) -> list[FormQuestion]:
        if len({question.id for question in questions}) != len(questions):
            raise ValueError("question ids must be unique")
        return questions

    def validate_response(self, response: FormResponse) -> FormResponse:
        """Check ids, strings, required presence and strict selection, not domain rules.

        Preserve nonblank text verbatim. Missing or whitespace-only optional
        fields become ""; required blanks fail. Never retry the whole form here.
        """
        response = FormResponse.model_validate(response)
        if response.action != "accept":
            return response
        content = response.content
        assert content is not None
        unknown = content.keys() - {question.id for question in self.questions}
        if unknown:
            raise ValueError(f"Unknown answer ids: {sorted(unknown)}")
        answers: dict[str, str] = {}
        for question in self.questions:
            value = content.get(question.id, "")
            if not value.strip():
                if question.required:
                    raise ValueError(f"{question.id}: a nonblank answer is required")
                value = ""
            elif question.kind == "pick_one" and value not in {
                choice.value for choice in question.choices
            }:
                raise ValueError(f"{question.id}: choose a listed value")
            answers[question.id] = value
        return FormResponse(action="accept", content=answers)


InputRequest = Annotated[NeedInput | NeedInputForm, Field(discriminator="kind")]
"""Structured request outputs require kind=question/form; constructors supply it."""


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
    # Drives handle() from the channels once a host calls turns.start().
    # Hidden: the model must not cancel or await its own turns.
    turns: Annotated[TurnLoop, hidden, nosnapshot]
    # Producer-side Channel (full put / pop_last / snapshot / etc.)
    # — hidden, since the LLM has no business calling those.
    _user_messages_in: Annotated[Channel, hidden, nosnapshot]
    # Read-only facade (just .get() / .status() / .name) is what the
    # LLM sees as ``self.user_messages``.
    user_messages: Annotated[ChannelReader, nosnapshot]
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
        self.event_manager.on_close(self.queue_manager.shutdown)
        self.turns = TurnLoop(self)
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

    @hidden
    @no_trace
    async def aclose(self) -> None:
        """Stop owned turns/jobs before the caller releases shared resources.

        Self-close from an executing turn is rejected; its external owner must
        close after settlement. This avoids joining one's own task indirectly.
        """
        if self.turns.in_turn:
            raise RuntimeError("an executing turn cannot close its own agent")
        # Register for this close epoch; the manager shields and joins cleanup.
        # Last registered runs first, before other components release resources.
        self.event_manager.on_close(self._close_runtime)
        await super().aclose()

    @hidden
    @no_trace
    async def _close_runtime(self) -> None:
        try:
            await self.turns.stop()
        finally:
            await self.queue_manager.shutdown()

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
    ) -> Done | InputRequest | Waiting:
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
          not also send it with ``message()``. Add ``options=[...]`` for
          suggested answers. This never opens a dialog; the next message answers.
          For explicit forms use ``NeedInputForm(heading=...,
          questions=[TextQuestion(id="name", label="Name?")])``. Domain validation
          belongs to the agent; ask targeted follow-ups, never repeat the whole wizard. The answer is a ``FormResponse`` with action
          accept (a string dictionary keyed by ids), decline or cancel. Raw fallback text is
          unvalidated, not accepted form content. The answer arrives
          in the next notification. ``reason`` optionally says why you
          need it::

              return_result(NeedInput(question="Which branch should I push to?", options=["main", "dev"]))

        - ``Waiting(message=..., explanation=..., on=[...])`` — waiting on a
          background job or queue, not on a person. ``message`` is shown to
          the user; ``on`` names what you wait for: a channel, a job label,
          a subagent::

              return_result(Waiting(message="Tests are running; I will report when they finish.", explanation="tests running", on=["jobs:ci-42"]))

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
