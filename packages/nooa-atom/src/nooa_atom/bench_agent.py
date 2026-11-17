# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``atom`` benchmark agent: the Atom agent on a benchmark task.

nooa-bench finds this class through its ``nooa_bench.agents`` entry-point
group (see this package's pyproject.toml), so nooa-bench does not depend on
nooa-atom: ``nemo-harbor --agent-type atom`` and ``nooa-bench run`` work
when both are installed.

The task runs unattended in a session tree (``nooa_atom.run_task``), the
same agent and session layer the ACP server uses. The session is stored in
the task's working directory, in ``.nooa/sessions`` (or in
``NOOA_SESSIONS_DIR``), and can be opened afterwards like any session.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from nooa_atom.hosts.headless import DEFAULT_MAX_TURNS, UNATTENDED_TURN_METHODS, TaskRun, run_task
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions

logger = logging.getLogger(__name__)


def _problem_statement(task_input: dict) -> str:
    """Extract the task text from supported Harbor/benchmark field names."""
    for key in ("user_message", "problem_statement", "task_description"):
        value = task_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        "task_input must include a non-empty user_message, problem_statement, or task_description"
    )


class _RecordedEvents:
    """The root agent's events, read by the runner's trajectory export."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def all_events(self) -> list[Any]:
        return list(self._events)


class AtomBenchAgent:
    """Runs one benchmark task with ``nooa_atom``'s ``AtomAgent``.

    ``llm`` is the runner's model client; the agent uses it and does not
    close it. After ``_run_evaluation``, ``event_manager.all_events()``
    returns the root agent's events for the trajectory export.

    ``run_headless`` is the ``nooa-bench run`` command's entry: one task in a
    workspace, with the turn method and limits the command passes.
    """

    TURN_METHODS: ClassVar[tuple[str, ...]] = UNATTENDED_TURN_METHODS
    DEFAULT_MAX_TURNS: ClassVar[int] = DEFAULT_MAX_TURNS

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm
        self.event_manager = _RecordedEvents([])

    async def _run_evaluation(self, task_input: dict) -> dict:
        """Run the task; return the runner's result dictionary."""
        description = _problem_statement(task_input)
        cwd = task_input.get("working_dir") or next(
            (d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd()
        )
        options = SessionOptions(workspace=Path(cwd), agent_spec=ATOM_AGENT, llm=self.llm)
        run = await run_task(options, description)
        self.event_manager = _RecordedEvents(run.events)
        logger.info("Session %s ended after %d turn(s)", run.session_id, run.turns)
        return _result_dict(run)

    @classmethod
    async def run_headless(
        cls,
        task: str,
        *,
        workspace: Path,
        model: str | None = None,
        max_turns: int | None = None,
        timeout: float | None = None,
        turn_method: str = "handle_batch",
    ) -> dict:
        """Run ``task`` unattended in ``workspace``; return ``nooa-bench run``'s JSON payload."""
        options = SessionOptions(
            workspace=workspace, agent_spec=ATOM_AGENT, model=model, turn_method=turn_method
        )
        outcome = await run_task(
            options,
            task,
            max_turns=cls.DEFAULT_MAX_TURNS if max_turns is None else max_turns,
            timeout=timeout,
        )
        done = outcome.done
        result = outcome.result
        return {
            "status": "done" if done is not None else "stopped",
            "stopped": outcome.stopped,
            "message": done.message if done is not None else None,
            "explanation": done.explanation if done is not None else None,
            "result": result.model_dump() if result is not None else None,
            "session_id": outcome.session_id,
            "turns": outcome.turns,
            "usage": outcome.usage.with_attributed().model_dump(),
        }


def _result_dict(run: TaskRun) -> dict:
    usage = run.usage.with_attributed()
    result = run.result
    error = run.stopped
    if error is None and result is None:
        error = "the agent finished without a TaskResult"
    return {
        "response": result.how_to_verify if result is not None else "",
        "success": result is not None and bool(result.solution_description),
        "result": result.model_dump() if result is not None else None,
        "error": error,
        "session_id": run.session_id,
        "n_input_tokens": usage.input_tokens,
        "n_output_tokens": usage.output_tokens,
    }
