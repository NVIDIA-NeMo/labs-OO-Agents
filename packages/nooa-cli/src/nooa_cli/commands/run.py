# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one NOOA coding-agent turn without a terminal UI."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import click

_RESUME_LAST = "__last__"


@contextmanager
def _project_scope(working_dir: str) -> Iterator[None]:
    """Anchor layered project configuration to the requested workspace."""
    env_name = "NEMO_OO_PROJECT_DIR"
    if env_name in os.environ:
        yield
        return
    os.environ[env_name] = str(Path(working_dir).expanduser().resolve() / ".nooa")
    try:
        yield
    finally:
        os.environ.pop(env_name, None)


def _read_prompt(parts: tuple[str, ...]) -> str:
    """Combine positional prompt words and optional piped standard input."""
    positional = " ".join(parts).strip()
    explicit_stdin = positional == "-"
    if explicit_stdin:
        positional = ""
    read_stdin = explicit_stdin or not sys.stdin.isatty()
    piped = sys.stdin.read() if read_stdin else ""
    prompt = "\n".join(piece for piece in (positional, piped.rstrip("\n")) if piece)
    if not prompt.strip():
        raise click.UsageError("Provide PROMPT or pipe a prompt on standard input.")
    return prompt


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("prompt", nargs=-1, type=click.UNPROCESSED)
@click.option("--model", "-m", help="LLM model to use (overrides the configured default).")
@click.option(
    "--working-dir",
    "-w",
    "-C",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    default=".",
    help="Workspace for agent tools and project configuration.",
)
@click.option("--agent", "agent_spec", metavar="MODULE:CLASS", help="Custom coding-agent class.")
@click.option(
    "--mcp-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
    help="MCP configuration file (default: .mcp.json in the workspace).",
)
@click.option(
    "--llm-config",
    "llm_config_paths",
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Additional LLM registry YAML (repeatable, highest precedence).",
)
@click.option("--context-limit", type=int, help="Context limit for summarization.")
@click.option("--no-trace", is_flag=True, help="Disable tracing.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("text", "json", "jsonl")),
    default="text",
    show_default=True,
    help="Output plain text, one JSON result document, or streaming JSON Lines.",
)
@click.option(
    "--output",
    "output_path",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the final agent response to this file.",
)
@click.option("--quiet", is_flag=True, help="Suppress status diagnostics on stderr.")
@click.option(
    "--continue",
    "continue_session",
    is_flag=True,
    help="Resume the most recent session for this workspace.",
)
@click.option("--resume", "resume_session_id", metavar="ID_OR_PREFIX", help="Resume one session.")
@click.option("--ephemeral", is_flag=True, help="Run without creating a resumable session.")
def command(
    prompt: tuple[str, ...],
    model: str | None,
    working_dir: str,
    agent_spec: str | None,
    mcp_file: str | None,
    llm_config_paths: tuple[Path, ...],
    context_limit: int | None,
    no_trace: bool,
    output_format: str,
    output_path: Path | None,
    quiet: bool,
    continue_session: bool,
    resume_session_id: str | None,
    ephemeral: bool,
) -> None:
    """Run PROMPT once without launching the interactive TUI.

    PROMPT may contain multiple shell words. With no PROMPT, input is read from
    stdin; use a literal '-' to require stdin. When both are present, piped
    input is appended as additional context.
    """
    if continue_session and resume_session_id:
        raise click.UsageError("--continue and --resume cannot be used together")
    if ephemeral and (continue_session or resume_session_id):
        raise click.UsageError("--ephemeral cannot be combined with --continue or --resume")
    text = _read_prompt(prompt)

    # Heavy framework imports stay below the Click boundary so `nooa --help`
    # remains fast and robust when an optional model provider is unavailable.
    from nooa.llm_config import llm_config_chain
    from nooa.secrets import load_secrets_into_env
    from nooa.unifiedllm import reload_registry
    from nooa_cli.headless import run_headless
    from nooa_cli.tui.config import Config

    def emit_event(event: dict[str, object]) -> None:
        if output_format == "jsonl":
            click.echo(json.dumps(event, ensure_ascii=False))

    with _project_scope(working_dir):
        try:
            load_secrets_into_env()
            reload_registry(*llm_config_chain(), *llm_config_paths)
            config = Config.load(
                model=model,
                agent=agent_spec,
                working_dir=working_dir,
                mcp_file=Path(mcp_file) if mcp_file else None,
                llm_config=list(llm_config_paths) or None,
                context_limit=context_limit,
                no_trace=no_trace,
                no_splash=True,
            )
            result = asyncio.run(
                run_headless(
                    text,
                    config=config,
                    ephemeral=ephemeral,
                    continue_session=continue_session,
                    resume_session_id=resume_session_id,
                    on_event=emit_event if output_format == "jsonl" else None,
                )
            )
        except KeyboardInterrupt:
            if output_format == "jsonl":
                cancelled = {
                    "schema_version": 1,
                    "type": "turn.cancelled",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "session_id": None,
                    "result": {
                        "schema_version": 1,
                        "session_id": None,
                        "status": "cancelled",
                        "messages": [],
                        "explanation": "cancelled",
                        "usage": {},
                        "error": None,
                    },
                }
                click.echo(json.dumps(cancelled, ensure_ascii=False))
            raise click.exceptions.Exit(130) from None
        except click.ClickException:
            raise
        except Exception as exc:
            if output_format in {"json", "jsonl"}:
                error = {
                    "schema_version": 1,
                    "session_id": None,
                    "status": "failed",
                    "messages": [],
                    "explanation": str(exc),
                    "usage": {},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                if output_format == "jsonl":
                    error = {
                        "schema_version": 1,
                        "type": "turn.failed",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "session_id": None,
                        "result": error,
                    }
                click.echo(json.dumps(error, ensure_ascii=False))
            else:
                click.echo(f"Error: {exc}", err=True)
            raise click.exceptions.Exit(1) from None

    final_text = "\n\n".join(result.messages)
    if output_path is not None:
        output_path.write_text(final_text, encoding="utf-8")

    if output_format == "json":
        click.echo(json.dumps(result.to_dict(), ensure_ascii=False))
    elif output_format == "text" and final_text:
        click.echo(final_text)

    if not quiet and result.session_id is not None:
        click.echo(f"Session: {result.session_id}", err=True)
    if result.status == "failed":
        if output_format == "text":
            click.echo(f"Error: {result.explanation}", err=True)
        raise click.exceptions.Exit(1)
    if result.status == "blocked":
        raise click.exceptions.Exit(3)
    if result.status == "cancelled":
        raise click.exceptions.Exit(130)


if __name__ == "__main__":
    command()
