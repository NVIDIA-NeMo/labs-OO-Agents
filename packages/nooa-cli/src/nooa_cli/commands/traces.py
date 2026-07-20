# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manage trace and evaluation files.

Usage:
    nooa traces delete                          # Delete, with confirmation
    nooa traces delete -n                       # Dry run
    nooa traces delete --older-than 7           # Only files >7 days old
    nooa traces list                            # List trace directories
    nooa traces stats                           # Show trace file statistics
"""

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click

from nooa_cli._common import find_project_root, format_size

SESSION_ID_RE = re.compile(r'"session\.id"\s*:\s*"[^"]+"')

# Directories to skip when scanning for trace files
EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------


def _discover_trace_dirs(root: Path | None = None) -> list[Path]:
    """Discover configured file-backed trace directories without a recursive scan.

    The live trace viewer stores spans in SQLite via OTLP/Journal ingestion; JSONL
    trace files are an explicit fallback/export path.  Keep this helper bounded to
    locations NeMo OO itself can write: the project trace dir used by TUI
    ``--trace``, the legacy ``./traces`` convention at the project root, and an
    explicit ``TRACE_DIR`` override when present.
    """
    project_root = root.resolve() if root is not None else find_project_root().resolve()
    candidates = [
        _project_trace_dir(project_root),
        project_root / "traces",
    ]

    env_trace_dir = os.getenv("TRACE_DIR")
    if env_trace_dir:
        env_path = Path(env_trace_dir).expanduser()
        candidates.append(env_path if env_path.is_absolute() else Path.cwd() / env_path)

    return sorted({path.resolve() for path in candidates if path.is_dir()})


def _project_trace_dir(root: Path | None = None) -> Path:
    """Return the project-local JSONL trace directory used by TUI ``--trace``."""
    project_root = root.resolve() if root is not None else find_project_root().resolve()
    return project_root / ".nooa" / "traces"


def _find_trace_files(root: Path) -> list[Path]:
    """Find all trace JSONL files, excluding annotation and eval files."""
    trace_files: list[Path] = []
    _collect_files(root, "*.jsonl", trace_files, exclude_non_trace=True)
    return sorted(trace_files)


def _find_eval_files(root: Path) -> list[Path]:
    """Find all .noo-eval.jsonl and .006eval.json files."""
    eval_files: list[Path] = []
    _collect_files(root, "*.noo-eval.jsonl", eval_files)
    _collect_files(root, "*.006eval.json", eval_files)
    return sorted(eval_files)


def _collect_files(root: Path, pattern: str, out: list[Path], *, exclude_non_trace: bool = False):
    """Walk directory tree collecting files matching pattern, skipping excluded dirs."""
    try:
        for f in root.glob(pattern):
            if exclude_non_trace and (
                f.name.endswith(".annotations.jsonl")
                or f.name.endswith(".noo-eval.jsonl")
                or f.name.endswith(".006eval.json")
                or f.name.endswith(".006summary.json")
            ):
                continue
            out.append(f)
        for entry in root.iterdir():
            if entry.is_dir() and entry.name not in EXCLUDE_DIRS:
                _collect_files(entry, pattern, out, exclude_non_trace=exclude_non_trace)
    except PermissionError:
        pass


def _has_session_id(file_path: Path) -> bool:
    """Check if a trace file has session.id in the first 4KB."""
    try:
        with open(file_path, encoding="utf-8") as f:
            chunk = f.read(4096)
    except OSError:
        return True  # Conservatively keep
    return bool(SESSION_ID_RE.search(chunk))


# ---------------------------------------------------------------------------
# CLI group + subcommands
# ---------------------------------------------------------------------------


@click.group()
def command():
    """Manage trace and evaluation files."""


@command.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be deleted without deleting.")
@click.option("--older-than", type=int, metavar="DAYS", help="Only files older than N days.")
@click.option("--all", "delete_all", is_flag=True, help="Delete ALL traces (ignore session.id).")
@click.option("--evals", is_flag=True, help="Also delete eval files (.noo-eval.jsonl).")
@click.option("--evals-only", is_flag=True, help="Only delete eval files (skip traces).")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt.")
def delete(
    directory: str,
    dry_run: bool,
    older_than: int | None,
    delete_all: bool,
    evals: bool,
    evals_only: bool,
    yes: bool,
):
    """Delete trace and evaluation files.

    DIRECTORY is the root directory to scan (default: current directory).

    \b
    By default, traces with a session.id are preserved (they belong to a
    tracked session). Use --all to override this.

    \b
    Examples:
        nooa traces delete                          # Delete, with confirmation
        nooa traces delete -n                       # Dry run (show what would go)
        nooa traces delete --older-than 7           # Only files >7 days old
        nooa traces delete --all -y                 # Delete everything, no prompt
        nooa traces delete --evals                  # Include eval files
    """
    root = Path(directory).resolve()
    click.echo(f"Scanning: {root}")

    all_files: list[tuple[str, Path]] = []

    if not evals_only:
        trace_files = _find_trace_files(root)
        click.echo(f"Found {len(trace_files)} trace files")
        all_files.extend(("trace", f) for f in trace_files)

    if evals or evals_only:
        eval_files = _find_eval_files(root)
        click.echo(f"Found {len(eval_files)} eval files")
        all_files.extend(("eval", f) for f in eval_files)

    if not all_files:
        click.echo("\nNo files found.")
        return

    if older_than:
        cutoff = datetime.now() - timedelta(days=older_than)
        cutoff_ts = cutoff.timestamp()
        all_files = [(t, f) for t, f in all_files if f.stat().st_mtime < cutoff_ts]
        click.echo(f"Filtered to {len(all_files)} files older than {older_than} days")

    to_delete = []
    for file_type, f in all_files:
        if file_type == "eval":
            to_delete.append((file_type, f))
        elif delete_all or not _has_session_id(f):
            to_delete.append((file_type, f))

    if not to_delete:
        click.echo("\nNo files to delete.")
        return

    total_size = sum(f.stat().st_size for _, f in to_delete)
    trace_count = sum(1 for t, _ in to_delete if t == "trace")
    eval_count = sum(1 for t, _ in to_delete if t == "eval")

    click.echo(f"\nFiles to delete: {len(to_delete)} ({format_size(total_size)})")
    if trace_count:
        click.echo(f"  - {trace_count} trace files")
    if eval_count:
        click.echo(f"  - {eval_count} eval files")

    if dry_run:
        click.secho("\n[DRY RUN] Files that would be deleted:", fg="yellow")
        for file_type, f in to_delete[:20]:
            click.echo(f"  [{file_type}] {f}")
        if len(to_delete) > 20:
            click.echo(f"  ... and {len(to_delete) - 20} more")
        return

    if not yes:
        click.confirm(
            f"\nDelete {len(to_delete)} files ({format_size(total_size)})?",
            abort=True,
        )

    click.echo("\nDeleting files...")
    deleted = 0
    for file_type, f in to_delete:
        try:
            f.unlink()
            deleted += 1
            if not sys.stdout.isatty() or deleted <= 20 or deleted == len(to_delete):
                click.echo(f"  Deleted [{file_type}]: {f}")
        except OSError as e:
            click.secho(f"  Error deleting {f}: {e}", fg="red")

    click.secho(f"\nDeleted {deleted} files ({format_size(total_size)} freed).", fg="green")


@command.command("list")
@click.option(
    "--root",
    type=click.Path(exists=True),
    default=None,
    help="Project root to scan (auto-detected by default).",
)
def list_dirs(root: str | None):
    """List discovered trace directories.

    \b
    Examples:
        nooa traces list
        nooa traces list --root /path/to/project
    """
    project_root = (Path(root) if root else find_project_root()).resolve()
    dirs = _discover_trace_dirs(project_root)

    if not dirs:
        click.echo("No trace directories found.")
        return

    click.echo(f"Trace directories (from {project_root}):\n")
    for d in dirs:
        trace_count = len(
            [
                f
                for f in d.glob("*.jsonl")
                if not f.name.endswith(".annotations.jsonl")
                and not f.name.endswith(".noo-eval.jsonl")
            ]
        )
        total_size = sum(
            f.stat().st_size
            for f in d.glob("*.jsonl")
            if not f.name.endswith(".annotations.jsonl") and not f.name.endswith(".noo-eval.jsonl")
        )
        rel = d.relative_to(project_root) if d.is_relative_to(project_root) else d

        count_str = f"{trace_count} traces" if trace_count else "empty"
        size_str = f", {format_size(total_size)}" if total_size else ""

        click.echo(f"  {rel}/  ({count_str}{size_str})")


@command.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
def stats(directory: str):
    """Show statistics about trace files.

    \b
    Examples:
        nooa traces stats
        nooa traces stats ./traces
    """
    root = Path(directory).resolve()
    trace_files = _find_trace_files(root)
    eval_files = _find_eval_files(root)

    if not trace_files and not eval_files:
        click.echo(f"No trace or eval files found in {root}")
        return

    click.echo(f"Trace statistics for: {root}\n")

    if trace_files:
        total_size = sum(f.stat().st_size for f in trace_files)
        with_session = sum(1 for f in trace_files if _has_session_id(f))
        without_session = len(trace_files) - with_session

        now = datetime.now().timestamp()
        ages = [(now - f.stat().st_mtime) / 86400 for f in trace_files]

        click.echo(f"  Trace files:     {len(trace_files)}")
        click.echo(f"  Total size:      {format_size(total_size)}")
        click.echo(f"  With session.id: {with_session}")
        click.echo(f"  Without session: {without_session}")
        if ages:
            click.echo(f"  Oldest:          {max(ages):.0f} days")
            click.echo(f"  Newest:          {min(ages):.1f} days")

    if eval_files:
        total_size = sum(f.stat().st_size for f in eval_files)
        click.echo(f"\n  Eval files:      {len(eval_files)}")
        click.echo(f"  Eval size:       {format_size(total_size)}")
