# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``nooa-bench`` command.

``nooa-bench run`` runs one task with the Atom agent, headless
and without Harbor, for smoke tests and scripts::

    nooa-bench run --workspace DIR --model ALIAS "task text"

Standard output carries only the result, as one JSON object; logs go to
standard error. The exit code is 0 when the agent finished (``Done``) and
1 when the run stopped without finishing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path

import click
from nooa_atom.hosts.headless import DEFAULT_MAX_TURNS


@click.group()
def main() -> None:
    """NOOA benchmark commands."""


@main.command()
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    show_default=True,
    help="Directory the agent works in; its session is stored in .nooa/sessions there.",
)
@click.option(
    "--model",
    envvar="NOOA_MODEL",
    default=None,
    help="Model alias. Or set NOOA_MODEL. Default: the workspace's default model.",
)
@click.option(
    "--max-turns",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_TURNS,
    show_default=True,
    help="Stop when this many turns have ended and the agent is still waiting.",
)
@click.option("--timeout", type=float, default=None, help="Stop after this many seconds.")
@click.argument("task")
def run(
    workspace: Path, model: str | None, max_turns: int, timeout: float | None, task: str
) -> None:
    """Run TASK with the Atom agent, unattended, and print the result as JSON."""
    from nooa_atom import SessionOptions, run_task
    from nooa_atom.session.loader import ATOM_AGENT

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    options = SessionOptions(workspace=workspace.resolve(), agent_spec=ATOM_AGENT, model=model)
    # Anything else that prints goes to standard error: standard output is the result.
    with contextlib.redirect_stdout(sys.stderr):
        outcome = asyncio.run(run_task(options, task, max_turns=max_turns, timeout=timeout))
    done = outcome.done
    result = outcome.result
    payload = {
        "status": "done" if done is not None else "stopped",
        "stopped": outcome.stopped,
        "message": done.message if done is not None else None,
        "explanation": done.explanation if done is not None else None,
        "result": result.model_dump() if result is not None else None,
        "session_id": outcome.session_id,
        "turns": outcome.turns,
        "usage": outcome.usage.with_attributed().model_dump(),
    }
    click.echo(json.dumps(payload, indent=2))
    sys.exit(0 if done is not None else 1)


if __name__ == "__main__":
    main()
