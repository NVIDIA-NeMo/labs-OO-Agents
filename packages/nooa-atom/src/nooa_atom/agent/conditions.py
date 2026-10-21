# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy postconditions for the Atom agent's turn methods."""

from typing import Any

from nooa.interactive import (  # noqa: F401  (re-exported)
    ONE_SHOT_QUESTION,
    Done,
    Waiting,
    require_task_result,
)
from nooa.strategy_validation import InvariantError
from nooa_atom.agent.background import live_children


def require_result(agent: Any, result: Any, call: Any) -> None:
    """Reject a batch ``Done`` without a ``result``.

    A batch turn is unattended: its ``self.message()`` output reaches at most
    a parent as a ``ChildMessage``, and the parent that delegated it reads
    ``done.result``. The strategy
    turns the ``InvariantError`` into feedback for the model, which then
    returns again with a ``TaskResult``.
    """
    if isinstance(result, Done) and result.result is None:
        raise InvariantError(
            "An unattended turn must end with Done(explanation=..., result=TaskResult(...)): "
            "set solution_description, evidence and how_to_verify, and put the full "
            "report in report. self.message() is no substitute here; the result is "
            "what the requester reads."
        )


class WaitingTargetNotRunning(InvariantError):
    """A ``Waiting`` names something in ``on`` that is not running, so nothing will arrive."""


# Names that mean the person. The person is not a job: waiting on them is
# NeedInput in an interactive turn and impossible in an unattended one.
PERSON_NAMES = frozenset({"user_messages", "user", "person", "human"})

ONLY_RUNNING = "Waiting is only for running jobs or background agents."
INTERACTIVE_FIX = (
    "If the work is finished, return Done; if you need an answer from the person, return NeedInput."
)
UNATTENDED_FIX = (
    "Send a child work with `await child.send(...)`, spawn one, or finish with Done "
    "and say what is unmet."
)


def require_running_wait(agent: Any, result: Any, call: Any) -> None:
    """Interactive turns (``handle``): ``Waiting.on`` names only running things.

    See ``check_waiting_targets`` for what counts as running. The hint
    points at ``Done`` and ``NeedInput``, the other two ways to end the turn.
    """
    check_waiting_targets(agent, result, fix=INTERACTIVE_FIX)


def require_running_wait_unattended(agent: Any, result: Any, call: Any) -> None:
    """Unattended turns (``handle_batch``, ``handle_orchestrate``): the same check.

    These turn methods cannot return ``NeedInput``; the hint says to send or
    spawn a child, or finish with ``Done`` and say what is unmet.
    """
    check_waiting_targets(agent, result, fix=UNATTENDED_FIX)


def check_waiting_targets(agent: Any, result: Any, *, fix: str) -> None:
    """Reject a ``Waiting`` whose ``on`` names something that is not running.

    The host runs the next turn only when something the turn waits on
    delivers. Each name in ``Waiting.on`` must be a running background job
    or a running or queued background agent:

    - a running job registered with the queue manager, by its channel name
      (``"mesh"`` while the mesh inbox pump runs) or label;
      ``"jobs:<label>"`` names a job by label; ``"jobs"`` means any job
      and needs at least one running;
    - ``"delegates"``: a child of this session is running or has queued
      input, or an item is already buffered on ``delegates`` (a child that
      just finished, or a follow-up sent with ``child.send()`` that has not
      started yet, both count);
    - a child's name or id: that child is running, ``working`` (its last
      turn ended Waiting on its own job) or has queued input.

    A plain name works for a job's label and a child's name alike; one that
    is both is rejected as ambiguous (use ``"jobs:<label>"`` or the child's id).

    - any channel with an item already buffered on it (a job that finished
      during the turn with its result queued, by its channel name), and
      ``"jobs:<label>"`` (or the label) when a ``self.jobs`` item with that
      label is buffered.

    The person (``"user_messages"``) is never a valid target. Anything else
    raises ``WaitingTargetNotRunning``, which the strategy shows the model
    as the error from its ``return_result`` cell; the turn then continues.
    """
    if not isinstance(result, Waiting):
        return
    running = _running_names(agent)
    unmet = [
        name for name in result.on if name in running["ambiguous"] or not _is_running(name, running)
    ]
    if not unmet:
        return
    reasons = []
    for name in unmet:
        if name in PERSON_NAMES:
            reasons.append(f"Waiting on {name!r}, but the person is not a job.")
        elif name in running["ambiguous"]:
            reasons.append(
                f"Waiting on {name!r} is ambiguous: it names both a job and a child. Use "
                f"'jobs:{name}' for the job, or the child's id {running['ambiguous'][name]!r}."
            )
        elif name == "delegates":
            reasons.append(
                "Waiting on 'delegates', but no child is running and nothing is queued "
                "or buffered, so nothing will arrive."
            )
        elif name in running["children"]:
            reasons.append(
                f"Waiting on child {name!r}, but it is not running and has nothing "
                "queued, so nothing will arrive from it."
            )
        elif name == "jobs":
            reasons.append(
                "Waiting on 'jobs', but no job is running and nothing is queued; "
                "see self.jobs.list() for finished ones."
            )
        elif name.startswith("jobs:") or name in running["known_jobs"]:
            label = name.removeprefix("jobs:")
            reasons.append(
                f"Waiting on {name!r}, but no job labelled {label!r} is running: it may "
                f"have finished already (see self.jobs.get({label!r})), or start it with "
                f"self.jobs.run(..., label={label!r})."
            )
        else:
            reasons.append(
                f"Waiting on {name!r}, but no running job or child has that name (nor is it "
                "'jobs' or 'delegates'), so nothing will arrive. If it was a job, it may have "
                f"finished already (see self.jobs.get({name!r})), or start it with "
                f"self.jobs.run(..., label={name!r})."
            )
    now = sorted(running["now"])
    raise WaitingTargetNotRunning(
        " ".join([*reasons, ONLY_RUNNING, fix])
        + " Running right now: "
        + (", ".join(repr(name) for name in now) if now else "nothing")
        + "."
    )


def _running_names(agent: Any) -> dict[str, Any]:
    """What can deliver right now, from the session registry and the job registry.

    ``children`` maps every child's name and id to whether it is running,
    working or has queued input; ``jobs`` holds every running job's channel name and
    label; ``known_jobs`` every job's, running or not; ``now`` the names
    that are accepted; ``ambiguous`` maps a name that is both a ``self.jobs``
    label and a live child's name to that child's id.
    """
    children: dict[str, bool] = {}
    port = getattr(agent, "session", None)
    if port is not None:
        registry = port._registry
        for info in registry.children(port.id):
            live = registry.get(info.id)
            # ``working``: the child's last turn ended Waiting on its own job,
            # so it will run again and deliver (#451 live status).
            active = info.status in ("running", "working") or (
                live is not None and not live.closing and live.has_queued_input()
            )
            for key in (info.id, info.name):
                if key:
                    children[key] = children.get(key, False) or active
    queues = agent.queue_manager
    handles = queues.running_handles()
    jobs = {key for handle in handles for key in (handle.name, handle.label)}
    known_jobs = set(queues.jobs())
    now = {name for name, active in children.items() if active} | jobs
    # A channel with an item already buffered delivers at once: a job that
    # finished during the turn (its result queued on its channel) or a
    # child's outcome on ``delegates``.
    channels = queues.channels()
    now |= {name for name, channel in channels.items() if channel.qsize() > 0}
    # A ``self.jobs`` job that already delivered: its labelled item is buffered.
    job_labels = {handle.label for handle in handles if handle.name == "jobs"}
    if "jobs" in channels:
        for item in channels["jobs"].snapshot():
            label = getattr(item, "label", None)
            if isinstance(label, str) and label:
                now |= {f"jobs:{label}", label}
                job_labels.add(label)
    if any(children.values()):
        now.add("delegates")
    if handles:
        now.add("jobs")
    now -= PERSON_NAMES
    # A name that is both a self.jobs label and a live child's name could mean either.
    live = {child.name: child.id for child in live_children(agent) if child.name}
    ambiguous = {label: live[label] for label in job_labels if label in live}
    return {
        "children": children,
        "jobs": jobs,
        "known_jobs": known_jobs,
        "now": now,
        "ambiguous": ambiguous,
    }


def _is_running(name: str, running: dict[str, Any]) -> bool:
    if name in PERSON_NAMES:
        return False
    if name in running["now"]:
        return True
    return name.startswith("jobs:") and name.removeprefix("jobs:") in running["jobs"]
