# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``coder`` agent type: the nooa-coder coding agent on a benchmark task.

The task runs unattended in a session tree (``nooa_coder.run_task``), the
same agent and session layer the ACP server uses. The session is stored in
the task's working directory, in ``.nooa/sessions`` (or in
``NOOA_SESSIONS_DIR``), and can be opened afterwards like any session.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from nooa_coder import SessionOptions, TaskRun, run_task
from nooa_coder.session.loader import CODING_AGENT

from nooa_bench.bench_agent import _problem_statement

logger = logging.getLogger(__name__)


class _RecordedEvents:
    """The root agent's events, read by the runner's trajectory export."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def all_events(self) -> list[Any]:
        return list(self._events)


class CoderBenchAgent:
    """Runs one benchmark task with ``nooa_coder``'s ``CodingAgent``.

    ``llm`` is the runner's model client; the agent uses it and does not
    close it. After ``_run_evaluation``, ``event_manager.all_events()``
    returns the root agent's events for the trajectory export.
    """

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm
        self.event_manager = _RecordedEvents([])

    async def _run_evaluation(self, task_input: dict) -> dict:
        """Run the task; return the runner's result dictionary."""
        description = _problem_statement(task_input)
        cwd = task_input.get("working_dir") or next(
            (d for d in ("/testbed", "/app") if os.path.isdir(d)), os.getcwd()
        )
        options = SessionOptions(workspace=Path(cwd), agent_spec=CODING_AGENT, llm=self.llm)
        run = await run_task(options, description)
        self.event_manager = _RecordedEvents(run.events)
        logger.info("Session %s ended after %d turn(s)", run.session_id, run.turns)
        return _result_dict(run)


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
