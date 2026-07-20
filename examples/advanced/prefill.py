# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-ellipsis prefill: code before ... runs first, LLM acts on the output.

uv run python examples/advanced/prefill.py
"""

from dataclasses import dataclass, field
from datetime import date

from nooa import Agent
from nooa.util.quickstart import autorun, llm


@dataclass
class Task:
    name: str
    status: str = "pending"
    assignee: str | None = None
    due: date | None = None
    children: list["Task"] = field(default_factory=list)


class Messenger:
    """Tool for notifying team members."""

    async def send(self, to: str, message: str) -> str:
        """Send a message to a team member."""
        # This print appears in the REPL output, visible to the LLM
        print(f"[SENT] To {to}: {message}")
        return f"Message sent to {to}"


class ProjectManager(Agent, llm=llm):
    messenger = Messenger()

    async def check_and_notify(self, project: Task, today: date) -> str:
        """Check project and use self.messenger.send() to notify assignees about issues."""

        def all_tasks(t: Task) -> list[Task]:
            return [t] + [x for c in t.children for x in all_tasks(c)]

        overdue = [t for t in all_tasks(project) if t.due and t.due < today and t.status != "done"]
        unassigned = [t for t in all_tasks(project) if t.status == "pending" and not t.assignee]
        print(f"Overdue tasks: {[(t.name, t.assignee, t.due) for t in overdue]}")
        print(f"Unassigned tasks: {[t.name for t in unassigned]}")
        ...


PROJECT = Task(
    "Launch v2.0",
    "in_progress",
    "Alice",
    date(2026, 2, 1),
    [
        Task("Backend", "done", "Bob", date(2026, 1, 15)),
        Task(
            "Frontend",
            "in_progress",
            "Carol",
            date(2026, 1, 25),
            [
                Task("Login", "done", "Carol", date(2026, 1, 20)),
                Task("Dashboard", "pending", "Dave", date(2026, 1, 28)),  # Overdue
            ],
        ),
        Task("Docs", "pending", None, date(2026, 1, 30)),  # Unassigned
    ],
)


def print_event_sequence(agent: Agent):
    """Print the event sequence from the agent's event manager."""
    events = list(agent.event_manager.values())
    W = 70

    print("\n" + "─" * W)
    print(f"  EVENT SEQUENCE  ({len(events)} events)")
    print("─" * W)

    for i, event in enumerate(events):
        etype = event.event_type
        tool_call_id = getattr(event, "tool_call_id", "") or ""
        tag = "  [prefill]" if tool_call_id.startswith("prefill_") else ""

        print(f"\n[{i}] {etype.upper()}{tag}")

        if etype == "task":
            print(f"    {getattr(event, 'prompt', '')}")

        elif etype == "tool_call":
            code = (getattr(event, "arguments", {}) or {}).get("code", "")
            for line in code.splitlines():
                print(f"    {line}")

        elif etype == "python_output":
            stdout = str(getattr(event, "stdout", "") or "").rstrip()
            error = getattr(event, "error", None)
            value = getattr(event, "value", None)
            if stdout:
                for line in stdout.splitlines():
                    print(f"    {line}")
            if error:
                print(f"    error: {error}")
            if value is not None:
                print(f"    → {value}")

        else:
            content = str(getattr(event, "content", "") or "")
            if content:
                print(f"    {content}")

    print("\n" + "─" * W)
    flow = " → ".join(
        e.event_type
        + (" [prefill]" if (getattr(e, "tool_call_id", "") or "").startswith("prefill_") else "")
        for e in events
    )
    print(f"  {flow}")
    print("─" * W + "\n")


@autorun
async def main():
    agent = ProjectManager()
    result = await agent.check_and_notify(PROJECT, date(2026, 1, 29))
    print(f"\nResult: {result}")
    print_event_sequence(agent)
