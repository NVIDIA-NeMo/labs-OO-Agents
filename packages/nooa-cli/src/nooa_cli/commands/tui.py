# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the TUI through the ``nooa tui`` subcommand.

Usage:
    nooa tui --model gpt-4o

Loaded on every ``nooa`` invocation, including ``nooa --help`` — keep this
module's imports light and defer the TUI machinery into the handler.
"""

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import click

_RESUME_LAST = "__last__"


@contextmanager
def _working_directory_project_scope(working_dir: str):
    """Use ``--working-dir`` as the TUI's project configuration scope.

    An explicit ``NEMO_OO_PROJECT_DIR`` remains authoritative. Otherwise the
    TUI reads and writes project settings, registries, and secrets beneath the
    workspace it is actually operating on, not beneath the framework checkout.
    """
    env_name = "NEMO_OO_PROJECT_DIR"
    if env_name in os.environ:
        yield
        return
    os.environ[env_name] = str(Path(working_dir).expanduser().resolve() / ".nooa")
    try:
        yield
    finally:
        os.environ.pop(env_name, None)


@click.command()
@click.option(
    "--model",
    "-m",
    help="LLM model to use (overrides config default)",
)
@click.option(
    "--agent",
    "agent_spec",
    default=None,
    metavar="MODULE:CLASS",
    help=(
        "Custom agent class instead of TUIAgent. "
        "Format: 'module.path:ClassName' or './file.py:ClassName'. "
        "CodingAgent subclasses receive the configured workspace and skills."
    ),
)
@click.option(
    "--working-dir",
    "-w",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    default=".",
    help="Working directory for bash commands",
)
@click.option(
    "--mcp-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
    default=None,
    help="MCP config file (default: .mcp.json in cwd)",
)
@click.option(
    "--llm-config",
    "llm_config_paths",
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Additional LLM registry YAML (repeatable, highest precedence). "
        "Clone private configs locally before passing them."
    ),
)
@click.option(
    "--no-splash",
    is_flag=True,
    help="Skip the splash screen",
)
@click.option(
    "--skills-dir",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    help="Skills directory (can be specified multiple times)",
)
@click.option(
    "--context-limit",
    type=int,
    help="Context limit for summarization",
)
@click.option(
    "--no-trace",
    is_flag=True,
    help="Disable tracing",
)
@click.option(
    "--vi",
    is_flag=True,
    help="Enable vi keybindings in the input prompt",
)
@click.option(
    "--python",
    is_flag=True,
    help="Show agent Python code execution panels",
)
@click.option(
    "--continue",
    "-c",
    "continue_session",
    is_flag=False,
    flag_value=_RESUME_LAST,
    default=None,
    help="Resume a session: -c (last session) or -c <short-hash>",
)
def command(
    model: str | None,
    agent_spec: str | None,
    working_dir: str,
    mcp_file: str | None,
    llm_config_paths: tuple[Path, ...],
    no_splash: bool,
    skills_dir: tuple[str, ...],
    context_limit: int | None,
    no_trace: bool,
    vi: bool,
    python: bool,
    continue_session: str | None,
):
    """Launch the NVIDIA Labs Object Oriented Agents (NOOA) TUI.

    Interactive REPL for chatting with agents, running commands, and managing
    skills and MCP servers.

    Invoke as `nooa tui`.

    Examples:
        nooa tui
        nooa tui --model gpt-4o
        nooa tui --working-dir /path/to/project
        nooa tui --mcp-file .mcp.json
        nooa tui --llm-config .nooa/llm_config.yaml
        nooa tui --agent internal_agents:CodingAgent
        nooa tui --vi
    """
    # Lazy imports preserve the fast path for every other ``nooa`` command.
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.main import main as tui_main

    with _working_directory_project_scope(working_dir):
        config = Config.load(
            model=model,
            agent=agent_spec,
            working_dir=working_dir,
            mcp_file=Path(mcp_file) if mcp_file else None,
            llm_config=list(llm_config_paths) or None,
            no_splash=no_splash,
            skills_dir=list(skills_dir) if skills_dir else None,
            context_limit=context_limit,
            no_trace=no_trace,
            vi=vi,
            python=python,
        )

        continue_last = continue_session == _RESUME_LAST
        resume_session_id = (
            continue_session if continue_session and continue_session != _RESUME_LAST else None
        )

        try:
            asyncio.run(
                tui_main(
                    config=config,
                    continue_last=continue_last,
                    resume_session_id=resume_session_id,
                )
            )
        except KeyboardInterrupt:
            sys.exit(0)


# ``python -m nooa_cli.commands.tui`` remains a script-free fallback.
if __name__ == "__main__":
    command()
