# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion check: ask the model whether a turn's ``Done`` really finishes the task.

When a turn method ends with ``Done``, :class:`JudgedCodeActV2` makes one
tool-free model call (the "judge") with a bounded view of the turn: the
original request, the ``Done`` itself and the tail of the turn's cells and
outputs. The judge replies ``CONTINUE: <what is missing>`` or ``STOP:
<reason>``. On ``CONTINUE`` the feedback is appended to the same turn as a
``Notification`` ("Completion check: ...") and the CodeAct loop keeps going,
with the turn's Python locals intact. Anything else, an exception, or the
cap of consecutive ``CONTINUE`` verdicts accepts the ``Done``.
``NeedInput`` and ``Waiting`` are never judged.

This follows the AVO main loop's continuation judge: only the ``CONTINUE:`` /
``STOP:`` prefixes count, unclear output or an exception resolves to stop,
and approvals without progress are capped at three.

Every decision is recorded as a :class:`CompletionCheck` metadata event
(never shown to the model) so trace analysis can see it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role
from nooa.events import LLMResponse, Notification, PythonOutput
from nooa.interactive import Done
from nooa.strategies import CodeActV2

if TYPE_CHECKING:
    from nooa.strategies.base import RuntimeServices
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)

COMPLETION_CHECK_ENV = "NOOA_ATOM_COMPLETION_CHECK"
"""Environment override: ``off``, ``batch`` (default), ``interactive`` or ``all``."""

_MODES: dict[str, frozenset[str]] = {
    "off": frozenset(),
    "batch": frozenset({"handle_batch"}),
    "interactive": frozenset({"handle"}),
    "all": frozenset({"handle", "handle_batch"}),
}

FEEDBACK_SOURCE = "Completion check before your Done was accepted."
"""The ``Notification.source`` of injected judge feedback."""


class CompletionCheckConfig(BaseModel):
    """Which turn methods get a completion check, and its bounds.

    The default checks unattended turns (``handle_batch``: delegated
    children, headless and benchmark runs) and not interactive ones
    (``handle``), where a person reads every reply and can say "keep going".
    """

    methods: frozenset[str] = frozenset({"handle_batch"})
    max_continues: int = Field(default=3, ge=0)
    """Consecutive ``CONTINUE`` verdicts per turn; the next ``Done`` is accepted unjudged."""
    request_chars: int = 8_000
    done_chars: int = 8_000
    tail_items: int = 12
    tail_chars: int = 12_000
    item_chars: int = 2_000

    def applies(self, method_name: str) -> bool:
        return method_name in self.methods

    @classmethod
    def from_mode(cls, mode: str) -> CompletionCheckConfig:
        """``off``, ``batch``, ``interactive`` or ``all``."""
        key = mode.strip().lower()
        if key not in _MODES:
            raise ValueError(f"completion check mode must be one of {sorted(_MODES)}, got {mode!r}")
        return cls(methods=_MODES[key])

    @classmethod
    def from_env(cls) -> CompletionCheckConfig:
        """The default, or the mode named by ``NOOA_ATOM_COMPLETION_CHECK``.

        An unknown value is logged and ignored (the default applies).
        """
        value = os.environ.get(COMPLETION_CHECK_ENV, "").strip()
        if not value:
            return cls()
        try:
            return cls.from_mode(value)
        except ValueError:
            logger.warning("Ignoring %s=%r; using the default", COMPLETION_CHECK_ENV, value)
            return cls()


class CompletionCheck(Metadata):
    """One completion-check decision on a turn's ``Done`` (stored, never shown to the model)."""

    _role: ClassVar[Role] = Role.METADATA

    method_name: str = ""
    decision: Literal["continue", "stop"] = "stop"
    reason: Literal["judge", "unclear", "error", "cap"] = "judge"
    """``judge``: an explicit verdict; ``unclear``: no recognized prefix;
    ``error``: the judge call failed; ``cap``: accepted unjudged after the cap."""
    attempt: int = 0
    """1-based number of this Done within the turn."""
    feedback: str = ""
    raw: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


JUDGE_INSTRUCTIONS = """\
You check whether an AI coding agent has really finished the task it was given, \
before its final answer is accepted. You see the original request, the agent's \
final answer (Done) and the end of its work log. You cannot run tools or code; \
judge only from what is shown.

Reply with exactly one line that starts with one of these prefixes:
CONTINUE: <concrete feedback: which requested items are missing, unverified or wrong, and what to do next>
STOP: <short reason the task is complete, or why more work in this turn cannot help>

Say CONTINUE only for a concrete, fixable gap the agent can still close in this \
turn (a requested step not done, a claimed check never run, an error ignored, a \
required output missing). Do not ask for polish, extra features or work the \
request did not ask for. If the agent reports a genuine blocker it cannot \
resolve, or you are unsure, say STOP."""


def parse_verdict(text: str) -> tuple[Literal["continue", "stop"], str, bool]:
    """``(decision, feedback, recognized)`` from the judge's reply.

    Only a leading ``CONTINUE:`` or ``STOP:`` counts (case-insensitive,
    leading whitespace allowed). ``CONTINUE:`` without feedback is unclear.
    """
    stripped = (text or "").strip()
    upper = stripped.upper()
    if upper.startswith("CONTINUE:"):
        feedback = stripped[len("CONTINUE:") :].strip()
        if feedback:
            return "continue", feedback, True
        return "stop", "", False
    if upper.startswith("STOP:"):
        return "stop", stripped[len("STOP:") :].strip(), True
    return "stop", "", False


def _clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    half = max(limit // 2 - 20, 0)
    return f"{text[:half]}\n... [{len(text) - 2 * half} chars cut] ...\n{text[-half:]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _format_request(call: CurrentCall) -> str:
    notification = call.args[0] if call.args else call.kwargs.get("notification", {})
    if not isinstance(notification, dict):
        return str(notification)
    parts = []
    for channel, items in notification.items():
        for item in items if isinstance(items, list) else [items]:
            data = _jsonable(item)
            text = data if isinstance(data, str) else json.dumps(data, default=str, indent=1)
            parts.append(f"[{channel}]\n{text}")
    return "\n\n".join(parts)


def _format_done(done: Done) -> str:
    data = done.model_dump(mode="json", exclude_none=True)
    return json.dumps(data, default=str, indent=1)


def _cell_code(response: LLMResponse) -> list[str]:
    codes = []
    for tool_call in response.tool_calls or []:
        arguments = tool_call.arguments
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed = None
        code = parsed.get("code") if isinstance(parsed, dict) else None
        codes.append(code if isinstance(code, str) else str(arguments))
    return codes


def _turn_tail(runtime: RuntimeServices, call: CurrentCall, config: CompletionCheckConfig) -> str:
    """The turn's recent cells, outputs and mid-turn notifications, bounded.

    Only events the agent itself was shown are used: its own cell code and
    text, cell outputs, and notifications appended to the turn.
    """
    items: list[str] = []
    seen_task = call.task_tag is None
    for tag, event in runtime.event_manager.items():
        if not seen_task:
            seen_task = tag == call.task_tag
            continue
        if event.event_role is not Role.USER and not isinstance(event, LLMResponse):
            continue
        if isinstance(event, LLMResponse):
            if event.content and event.content.strip():
                items.append(f"[agent text]\n{_clip(event.content, config.item_chars)}")
            for code in _cell_code(event):
                items.append(f"[agent cell]\n{_clip(code, config.item_chars)}")
        elif isinstance(event, PythonOutput):
            body = "\n".join(
                part
                for part in (
                    f"status: {event.execution_status.value}",
                    event.stdout and f"stdout:\n{event.stdout}",
                    event.stderr and f"stderr:\n{event.stderr}",
                    event.error and f"error:\n{event.error}",
                )
                if part
            )
            items.append(f"[cell output]\n{_clip(body, config.item_chars)}")
        elif isinstance(event, Notification):
            text = event.description or ""
            if event.value is not None:
                text += "\n" + json.dumps(_jsonable(event.value), default=str)
            items.append(f"[notification: {event.source}]\n{_clip(text, config.item_chars)}")
        else:
            content = getattr(event, "content", None)
            if isinstance(content, str) and content:
                items.append(f"[{event.event_type}]\n{_clip(content, config.item_chars)}")
    tail = items[-config.tail_items :] if config.tail_items > 0 else []
    omitted = len(items) - len(tail)
    text = "\n\n".join(tail)
    if len(text) > config.tail_chars:
        text = "... [earlier log cut] ...\n" + text[-config.tail_chars :]
    if omitted:
        text = f"[{omitted} earlier log entries not shown]\n\n" + text
    return text


def build_review(
    runtime: RuntimeServices, call: CurrentCall, done: Done, config: CompletionCheckConfig
) -> str:
    """The judge's bounded user message for this turn."""
    return (
        "<original_request>\n"
        f"{_clip(_format_request(call), config.request_chars)}\n"
        "</original_request>\n\n"
        "<final_answer>\n"
        f"{_clip(_format_done(done), config.done_chars)}\n"
        "</final_answer>\n\n"
        "<work_log_tail>\n"
        f"{_turn_tail(runtime, call, config)}\n"
        "</work_log_tail>\n\n"
        "Has the agent finished the original request? Reply with CONTINUE: or STOP: as instructed."
    )


def _judge_llm(runtime: RuntimeServices) -> Any:
    """The model the turn itself is using."""
    from nooa.runtime.actor import _current_llm_var

    llm = _current_llm_var.get()
    return llm if llm is not None else getattr(runtime.agent, "llm", None)


class JudgedCodeActV2(CodeActV2):
    """CodeActV2 whose ``Done`` results go through the completion check.

    The check runs after ``return_result`` validation and the method's
    postconditions, so it only ever sees a ``Done`` the turn would
    otherwise end with. The configuration is the agent's
    ``completion_check`` attribute (:class:`CompletionCheckConfig`); an
    agent without one, or a method the configuration does not name, is
    unchanged.
    """

    async def _process_tool_calls(  # type: ignore[override]
        self,
        tool_calls: list[Any],
        runtime: RuntimeServices,
        builtins: dict[str, Any],
        session: Any,
        call: CurrentCall,
        return_type: Any,
        event_id: str,
        preserve_llm_response: bool = False,
    ) -> Any:
        result = await super()._process_tool_calls(
            tool_calls,
            runtime,
            builtins,
            session,
            call,
            return_type,
            event_id,
            preserve_llm_response=preserve_llm_response,
        )
        if not result.completed or not isinstance(result.final_value, Done):
            return result
        config = getattr(runtime.agent, "completion_check", None)
        if not isinstance(config, CompletionCheckConfig) or not config.applies(call.method_name):
            return result
        if await self._should_continue(runtime, call, session, result.final_value, config):
            return type(result)()  # not completed: the same turn keeps going
        return result

    async def _should_continue(
        self,
        runtime: RuntimeServices,
        call: CurrentCall,
        session: Any,
        done: Done,
        config: CompletionCheckConfig,
    ) -> bool:
        continues = getattr(session, "completion_continues", 0)
        attempt = getattr(session, "completion_attempts", 0) + 1
        session.completion_attempts = attempt
        record = CompletionCheck(method_name=call.method_name, attempt=attempt)

        if continues >= config.max_continues:
            record.reason = "cap"
            self._record(runtime, record)
            return False

        raw = ""
        try:
            llm = _judge_llm(runtime)
            if llm is None:
                raise RuntimeError("no model client for the completion check")
            response = await llm.acall(
                [
                    {"role": "system", "content": JUDGE_INSTRUCTIONS},
                    {"role": "user", "content": build_review(runtime, call, done, config)},
                ]
            )
            raw = response.content or ""
            usage = getattr(response, "usage", None)
            if usage is not None:
                record.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                record.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        except Exception as exc:  # any failure accepts the Done
            logger.warning("Completion check failed; accepting Done: %s", exc)
            record.reason = "error"
            record.raw = _clip(f"{type(exc).__name__}: {exc}", 2_000)
            self._record(runtime, record)
            return False

        decision, feedback, recognized = parse_verdict(raw)
        record.raw = _clip(raw, 4_000)
        record.decision = decision
        record.feedback = feedback
        record.reason = "judge" if recognized else "unclear"
        self._record(runtime, record)
        if decision != "continue":
            return False

        session.completion_continues = continues + 1
        runtime.event_manager.add(
            Notification(
                source=FEEDBACK_SOURCE,
                description=(
                    f"Completion check: {feedback}\n\n"
                    "Keep working on the original request in this turn (your Python "
                    "locals are still here), then end with return_result(...) again."
                ),
            )
        )
        return True

    @staticmethod
    def _record(runtime: RuntimeServices, record: CompletionCheck) -> None:
        try:
            runtime.event_manager.add(record)
        except Exception:
            logger.exception("Could not record the completion check decision")


__all__ = [
    "COMPLETION_CHECK_ENV",
    "CompletionCheck",
    "CompletionCheckConfig",
    "JUDGE_INSTRUCTIONS",
    "JudgedCodeActV2",
    "build_review",
    "parse_verdict",
]
