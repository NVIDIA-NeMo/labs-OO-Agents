# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Built-in keep-going stop audit for the TUI."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from nooa import PredictStrategy, strategy
from nooa.config import PredictConfig
from nooa.unifiedllm import get_llm_client

DEFAULT_KEEP_GOING_MODEL = "nemotron3-nano-30b"
KEEP_GOING_PREFIX = "[keep-going] "


class KeepGoingDecision(BaseModel):
    """Judgement of whether the agent should continue after returning DONE."""

    should_reprompt: bool = Field(
        description="True only when the agent stopped early and should continue autonomously."
    )
    reason: str = Field(
        default="",
        description="One short user-visible reason, under 100 characters.",
    )
    next_action: str = Field(
        default="",
        description="If should_reprompt is true, one concise imperative instruction for the agent.",
    )


class KeepGoingPrompt(BaseModel):
    """System continuation prompt plus the one-line reason shown to the user."""

    prompt: str
    display_reason: str


def _predict_config() -> PredictConfig:
    return PredictConfig(
        max_retries=2,
        max_tokens=300,
        temperature=0,
        output_serialization="tool_call",
    )


@lru_cache(maxsize=16)
def _judge_keep_going_for_model(model: str):
    keep_going_llm = get_llm_client(model)

    @strategy(PredictStrategy(config=_predict_config()), llm=keep_going_llm)
    async def _judge_keep_going(
        last_turns: str,
        todo_state: str,
        queue_state: str,
        return_result: str,
        user_visible_messages: str,
    ) -> KeepGoingDecision:
        """Decide whether an agent that just returned DONE should continue.

        Use only the supplied evidence. Return should_reprompt=true only when
        the agent stopped early and can continue without user input.

        Continue for obvious unfinished autonomous work: open unblocked todos,
        promised edits/tests/checks not done, actionable failures/findings not
        handled, or asking the user to inspect output the agent can inspect.

        Do not continue when the turn is complete, genuinely needs user input,
        needs approval/credentials, or is waiting for a running background job.

        Keep outputs short:
        - reason: one user-visible fragment under 100 chars, e.g. "open todos remain".
        - next_action: one imperative instruction, e.g. "Run the focused tests.".
        If should_reprompt=false, set reason and next_action to empty strings.
        """
        ...

    return _judge_keep_going


async def judge_keep_going(
    last_turns: str,
    todo_state: str,
    queue_state: str,
    return_result: str,
    user_visible_messages: str,
    *,
    model: str = DEFAULT_KEEP_GOING_MODEL,
) -> KeepGoingDecision:
    """Run the keep-going judge with the configured model."""
    judge = _judge_keep_going_for_model(model)
    return await judge(
        last_turns=last_turns,
        todo_state=todo_state,
        queue_state=queue_state,
        return_result=return_result,
        user_visible_messages=user_visible_messages,
    )


async def build_keep_going_prompt(
    agent: Any, result: Any, *, model: str = DEFAULT_KEEP_GOING_MODEL
) -> KeepGoingPrompt | None:
    """Return a synthetic continuation prompt when a DONE result stopped too early."""
    if _result_kind(result) != "DONE":
        return None
    if KEEP_GOING_PREFIX in str(result):
        return None
    decision = await judge_keep_going(**_snapshot(agent, result), model=model)
    if not decision.should_reprompt:
        return None
    reason = _one_line(decision.reason)
    next_action = decision.next_action.strip() or reason
    if not next_action:
        return None
    system_prompt = (
        f"{KEEP_GOING_PREFIX}A stop-reason audit says you are not actually done.\n\n"
        f"Reason: {reason}\n\n"
        f"Next action: {next_action}\n\n"
        "Continue now. Do not ask the user to read logs/output you can inspect yourself."
    )
    return KeepGoingPrompt(prompt=system_prompt, display_reason=reason)


def _one_line(value: str, max_len: int = 100) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _result_kind(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("kind", ""))
    kind = getattr(result, "kind", "")
    return getattr(kind, "value", str(kind)) if kind else ""


def _snapshot(agent: Any, result: Any) -> dict[str, str]:
    return {
        "last_turns": _last_turns(agent, 12),
        "todo_state": _todo_state(agent),
        "queue_state": _safe_call(getattr(getattr(agent, "queue_manager", None), "status", None)),
        "return_result": str(result),
        "user_visible_messages": _messages_from_recent_events(agent, 12),
    }


def _todo_state(agent: Any) -> str:
    todo = getattr(agent, "todo", None)
    if todo is None:
        return ""
    lines: list[str] = []
    try:
        open_todos = todo.list_todos(status="open")
    except Exception:
        open_todos = []
    if open_todos:
        lines.append(f"OPEN TODOS ({len(open_todos)}) — these are unresolved work:")
        for item in open_todos[:10]:
            title = getattr(item, "title", "")
            notes = getattr(item, "notes", "")
            item_id = getattr(item, "id", "")
            line = f"- [{item_id}] {title}"
            if notes:
                line += f" — {notes[:500]}"
            lines.append(line)
    else:
        lines.append("OPEN TODOS (0)")
    full_status = _safe_call(getattr(todo, "status", None))
    if len(full_status) > 3000:
        full_status = full_status[:1500] + "\n…\n" + full_status[-1500:]
    lines.append("\nFULL TODO STATUS (truncated if long):\n" + full_status)
    return "\n".join(lines)


def _last_turns(agent: Any, n: int) -> str:
    event_manager = getattr(agent, "event_manager", None)
    if event_manager is None:
        return ""
    try:
        items = event_manager.items()[-n:]
    except Exception:
        return ""
    lines: list[str] = []
    for tag, event in items:
        text = str(event)
        if len(text) > 2000:
            text = text[:2000] + "…"
        lines.append(f"<{tag}> {type(event).__name__}: {text}")
    return "\n".join(lines)


def _messages_from_recent_events(agent: Any, n: int) -> str:
    event_manager = getattr(agent, "event_manager", None)
    if event_manager is None:
        return ""
    try:
        events = event_manager.values()[-n:]
    except Exception:
        return ""
    snippets: list[str] = []
    for event in events:
        if type(event).__name__ not in {
            "AgentMessage",
            "Message",
            "MessageOutput",
            "TextOutput",
            "MarkdownOutput",
            "TUIAgentMessage",
        }:
            continue
        text = getattr(event, "text", None) or getattr(event, "content", None) or str(event)
        snippets.append(str(text))
    return "\n\n".join(snippets)


def _safe_call(fn: Any) -> str:
    if fn is None:
        return ""
    try:
        return str(fn())
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}>"
