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
    delivers. Each name in ``Waiting.on`` must be one of:

    - a running ``self.jobs`` job by its label (``"tests"``) or path
      (``"/root/tests"``), or a job with its result already buffered;
    - one of your children by name, path or id, while it is running,
      ``working`` (its last turn ended Waiting on its own job) or has
      queued input;
    - ``"jobs"`` (any job; needs one running or a result buffered) or
      ``"delegates"`` (a child is running or has queued input, or an item
      is already buffered on ``delegates``);
    - any other running job registered with the queue manager, by its
      channel name or label (``"mesh"`` while the mesh inbox pump runs), or
      any channel with an item already buffered on it.

    A deeper session (``/root/review/verifier``) is not a target: what it
    delivers goes to its parent, so wait on that child. The person
    (``"user_messages"``) is never a valid target. Anything else raises
    ``WaitingTargetNotRunning``, which the strategy shows the model as the
    error from its ``return_result`` cell; the turn then continues.
    """
    if not isinstance(result, Waiting):
        return
    running = _running_names(agent)
    unmet = [name for name in result.on if not _is_running(name, running)]
    if not unmet:
        return
    reasons = []
    for name in unmet:
        if name in PERSON_NAMES:
            reasons.append(f"Waiting on {name!r}, but the person is not a job.")
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
        elif name in running["known_jobs"]:
            reasons.append(
                f"Waiting on {name!r}, but no job labelled {name!r} is running: it may "
                f"have finished already (see self.jobs.get({name!r})), or start it with "
                f"self.jobs.run(..., label={name!r})."
            )
        elif (
            name.startswith(running["path"] + "/")
            and name.count("/") > running["path"].count("/") + 1
        ):
            top = name[len(running["path"]) + 1 :].split("/", 1)[0]
            reasons.append(
                f"Waiting on {name!r}, but it is not your child: what it delivers goes "
                f"to its parent. Wait on your child {top!r} instead."
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

    ``children`` maps each of your children's names, paths and ids to
    whether it is running, working or has queued input; ``known_jobs``
    holds every ``self.jobs`` label, running or not; ``now`` the names that
    are accepted (shown short: a job's label, a child's name); ``also`` the
    other accepted spellings (paths and ids); ``path`` your own path.
    """
    children: dict[str, bool] = {}
    now: set[str] = set()
    also: set[str] = set()
    port = getattr(agent, "session", None)
    path = port.path if port is not None else "/root"
    if port is not None:
        registry = port._registry
        for info in registry.children(port.id):
            live = registry.get(info.id)
            # ``working``: the child's last turn ended Waiting on its own job,
            # so it will run again and deliver (#451 live status).
            active = info.status in ("running", "working") or (
                live is not None and not live.closing and live.has_queued_input()
            )
            child_path = registry.path(info.id)
            for key in (info.id, info.name, child_path):
                if key:
                    children[key] = children.get(key, False) or active
            if active:
                if info.name:
                    now.add(info.name)
                also.update(key for key in (info.id, child_path) if key)
    queues = agent.queue_manager
    handles = queues.running_handles()
    now |= {key for handle in handles for key in (handle.name, handle.label)}
    channels = queues.channels()
    # A channel with an item already buffered delivers at once: a job that
    # finished during the turn (its result queued on its channel) or a
    # child's outcome on ``delegates``.
    now |= {name for name, channel in channels.items() if channel.qsize() > 0}
    jobs = getattr(agent, "jobs", None)
    # Every job's channel name and every self.jobs label, running or not.
    known_jobs = set(queues.jobs())
    if jobs is not None:
        known_jobs |= {info.label for info in jobs.list()}
    if jobs is not None:
        for info in jobs.list():
            if info.state == "running":
                now.add(info.label)
                also.add(info.path)
    # A ``self.jobs`` job that already delivered: its labelled item is buffered.
    if "jobs" in channels:
        for item in channels["jobs"].snapshot():
            label = getattr(item, "label", None)
            if isinstance(label, str) and label:
                now.add(label)
                item_path = getattr(item, "path", None)
                if item_path:
                    also.add(item_path)
    if any(children.values()):
        now.add("delegates")
    if handles:
        now.add("jobs")
    now -= PERSON_NAMES
    return {
        "children": children,
        "known_jobs": known_jobs,
        "now": now,
        "also": also,
        "path": path,
    }


def _is_running(name: str, running: dict[str, Any]) -> bool:
    if name in PERSON_NAMES:
        return False
    return name in running["now"] or name in running["also"]
