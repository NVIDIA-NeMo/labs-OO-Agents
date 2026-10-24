# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Atom agent's role in a session: its turn policy and its role instructions.

``AtomAgent`` has one turn method, ``handle``. The host picks the role with
the session's ``turn_method`` (``SessionOptions.turn_method``), which is
recorded with the session: ``handle`` is interactive, ``handle_batch``
unattended, ``handle_orchestrate`` an unattended orchestrator, and
``handle_work`` a child (``SessionPort.delegate``/``Subagent`` children). The
agent factory turns that selection into the agent's ``policy`` (the
postconditions its ``handle`` turns enforce) and its ``instructions`` (one
static ``role`` context block in the prompt prefix). A saved session
records the same ``turn_method``, so it resumes with the same role.
"""

from typing import Literal

TurnPolicy = Literal["interactive", "unattended", "orchestrator", "child"]

POLICY_BY_TURN_METHOD: dict[str, TurnPolicy] = {
    "handle": "interactive",
    "handle_batch": "unattended",
    "handle_orchestrate": "orchestrator",
    "handle_work": "child",
}
"""The policy each ``turn_method`` selection means for an Atom agent."""

_UNATTENDED = (
    "Nobody can answer questions, and self.message() reaches nobody: the result is what gets "
    'read. The task is notification["user_messages"]; its data is '
    'notification.get("context", []). End with Done(explanation=..., '
    "result=TaskResult(solution_description=..., evidence=..., how_to_verify=..., "
    "report=...)), also when blocked (say what blocked you). First await or cancel your "
    "running jobs; idle shells and subagents may stay. Leave the tree clean."
)

ROLE_INSTRUCTIONS: dict[TurnPolicy, str] = {
    "interactive": (
        "You are working with a person. Send brief progress notes and the reply with "
        "self.message() (complete Markdown: outcome, evidence, risks), then Done; background "
        "jobs may keep running. Ask with NeedInput(question=...) only when the person must answer."
    ),
    "unattended": "You work on one task unattended. " + _UNATTENDED,
    "orchestrator": (
        "You orchestrate one task unattended: you own the result and subagents do the work. "
        "Split the task into Todo briefs for subagents rather than implementing it; write "
        "runnable acceptance checks first, run them yourself (a report is not proof), send "
        "failures back with send, and loop until the task is met or another round is not "
        "worth it. " + _UNATTENDED
    ),
    "child": (
        "You are a child session working for a parent agent. Your parent's brief, a follow-up "
        'or an answer is in notification["user_messages"]; its first line says whether you may '
        "ask (NeedInput). Your parent reads only your result: end with Done(explanation=..., "
        "result=TaskResult(solution_description=..., evidence=..., how_to_verify=..., "
        "report=...)), also when blocked (say what blocked you). self.message() reaches your "
        "parent as a short note (cut at 1,000 chars): no routine progress notes. First await or "
        "cancel your running jobs; idle shells and subagents may stay. Leave the tree clean."
    ),
}
"""The default role instructions the agent factory gives each policy."""


def role_for(turn_method: str) -> tuple[TurnPolicy, str]:
    """The policy and the default role instructions for a session's ``turn_method``."""
    policy = POLICY_BY_TURN_METHOD.get(turn_method, "interactive")
    return policy, ROLE_INSTRUCTIONS[policy]
