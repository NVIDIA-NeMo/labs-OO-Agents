# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``nooa-bench`` command.

``nooa-bench run`` runs one task with a benchmark agent (``--agent``, by
default the Atom agent), headless and without Harbor, for smoke tests and
scripts::

    nooa-bench run --workspace DIR --model ALIAS "task text"

The agent is loaded through the agent registry (``nooa_bench.agents``), so
the Atom agent needs nooa-atom installed. Standard output carries only the
result, as one JSON object; logs go to standard error. The exit code is 0
when the agent finished (``Done``) and 1 when the run stopped without
finishing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path

import click

from nooa_bench.agents import AgentNotInstalledError, load_agent


@click.group()
def main() -> None:
    """NOOA benchmark commands."""


@main.command()
@click.option(
    "--agent",
    "agent_name",
    default="atom",
    show_default=True,
    help="The benchmark agent to run (an installed nooa_bench.agents entry point).",
)
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
    default=None,
    help="Stop when this many turns have ended and the agent is still waiting. "
    "Default: the agent's (10 for atom).",
)
@click.option("--timeout", type=float, default=None, help="Stop after this many seconds.")
@click.option(
    "--turn-method",
    default="handle_batch",
    show_default=True,
    help="The root's turn method; for atom handle_batch or handle_orchestrate "
    "(the root plans, delegates and verifies instead of doing the work).",
)
@click.argument("task")
def run(
    agent_name: str,
    workspace: Path,
    model: str | None,
    max_turns: int | None,
    timeout: float | None,
    turn_method: str,
    task: str,
) -> None:
    """Run TASK with a benchmark agent, unattended, and print the result as JSON."""
    try:
        agent_class = load_agent(agent_name)
    except AgentNotInstalledError as error:
        raise click.UsageError(str(error)) from None
    run_headless = getattr(agent_class, "run_headless", None)
    if run_headless is None:
        raise click.UsageError(
            f"The {agent_name!r} agent cannot run headless; use nemo-harbor --agent-type {agent_name}."
        )
    turn_methods = tuple(getattr(agent_class, "TURN_METHODS", ()))
    if turn_methods and turn_method not in turn_methods:
        raise click.BadParameter(
            f"{turn_method!r} is not one of {', '.join(turn_methods)}.",
            param_hint="'--turn-method'",
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    # Anything else that prints goes to standard error: standard output is the result.
    with contextlib.redirect_stdout(sys.stderr):
        payload = asyncio.run(
            run_headless(
                task,
                workspace=workspace.resolve(),
                model=model,
                max_turns=max_turns,
                timeout=timeout,
                turn_method=turn_method,
            )
        )
    click.echo(json.dumps(payload, indent=2))
    sys.exit(0 if payload.get("status") == "done" else 1)


if __name__ == "__main__":
    main()
