# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream sources: functions that produce Streams — async-native, non-blocking."""

from __future__ import annotations

import asyncio
import fnmatch as _fnmatch
import os
import re as _re
import sys
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from nooa.agentdoc._truncating_stream import TruncatingStringIO
from nooa.tools._bash_session import BashSession
from nooa_cli.tools.pyp.errors import make_pipe_error
from nooa_cli.tools.pyp.stream import Stream

_STDERR_CAP = 128 * 1024  # 128 KB max stderr buffered


async def _stream_bash_lines(
    cmd: str,
    *,
    timeout: float = 30.0,
    cwd: str | None = None,
    ok_codes: tuple[int, ...] = (0,),
    error_prefix: str = "Command failed",
) -> AsyncIterator[str]:
    """Stream stdout lines from a BashSession command, handling line buffering.

    Shared implementation for run(), rg(), and find() sources.
    Yields complete lines as they arrive. On completion, flushes any
    trailing partial line, then raises PipeError if returncode not in ok_codes.
    """
    bash = BashSession(cwd=cwd or ".")
    await bash.start()
    try:
        buffer = ""
        stderr_buf = ""
        async for stream_name, chunk in bash.run_stream(cmd, timeout=timeout):
            if stream_name == "__done__":
                parts = chunk.split(",")
                returncode = int(parts[0])
                # Flush trailing partial line first
                if buffer:
                    yield buffer
                # Then check for errors
                if returncode not in ok_codes:
                    raise make_pipe_error(
                        f"{error_prefix}: {cmd}",
                        cmd=cmd,
                        returncode=returncode,
                        stderr=stderr_buf,
                    )
                break
            elif stream_name == "stdout":
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield line
            elif stream_name == "stderr":
                if len(stderr_buf) < _STDERR_CAP:
                    stderr_buf += chunk
    finally:
        await bash.close()


def cat(*paths: str | Path, encoding: str = "utf-8") -> Stream:
    """
    Read lines from one or more files, streaming line-by-line (non-blocking).

    Lines are stripped of trailing newlines.

    Usage:
        cat("file.txt").grep("pattern")
        cat("a.txt", "b.txt").sort()
    """

    async def _gen() -> AsyncIterator[str]:
        for p in paths:
            path = Path(p)
            fh = await asyncio.to_thread(open, path, "r", encoding=encoding)
            try:
                while True:
                    line = await asyncio.to_thread(fh.readline)
                    if not line:
                        break
                    yield line.rstrip("\n")
            finally:
                await asyncio.to_thread(fh.close)

    return Stream(_gen(), _steps=[f"cat({', '.join(repr(str(p)) for p in paths)})"])


def run(cmd: str, *, check: bool = True, timeout: float = 30.0, cwd: str | None = None) -> Stream:
    """
    Run a shell command and stream its stdout lines as they arrive (non-blocking).

    Uses BashSession.run_stream() for true streaming — lines are yielded
    as the subprocess emits them, not buffered.

    Args:
        cmd: Shell command string.
        check: If True (default), raise PipeError on non-zero exit.
        timeout: Max seconds to wait (default 30s).
        cwd: Working directory for the command.

    Usage:
        run("ls -la").grep(r"py$")
        run("echo hello", timeout=5).head(1)
    """
    meta = {"returncode": 0, "stderr": "", "cmd": cmd}

    async def _gen() -> AsyncIterator[str]:
        bash = BashSession(cwd=cwd or ".")
        await bash.start()
        try:
            buffer = ""
            _stderr_io = TruncatingStringIO(limit=_STDERR_CAP)
            async for stream_name, chunk in bash.run_stream(cmd, timeout=timeout):
                if stream_name == "__done__":
                    parts = chunk.split(",")
                    returncode = int(parts[0])
                    meta["returncode"] = returncode
                    meta["stderr"] = _stderr_io.getvalue()
                    # Flush trailing partial line first
                    if buffer:
                        yield buffer
                    # Then check for errors
                    if check and returncode != 0:
                        raise make_pipe_error(
                            f"Command failed: {cmd}",
                            cmd=cmd,
                            returncode=returncode,
                            stderr=_stderr_io.getvalue(),
                        )
                    break
                elif stream_name == "stdout":
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        yield line
                elif stream_name == "stderr":
                    _stderr_io.write(chunk)
        finally:
            await bash.close()

    return Stream(_gen(), _steps=[f"run({cmd!r})"], _meta=meta)


def arun(shell_tools, cmd: str, *, timeout: float = 30.0, check: bool = True) -> Stream:
    """
    Stream subprocess output line-by-line as it arrives, using ShellTools.run_stream().

    Args:
        shell_tools: A ShellTools instance (e.g. self.shell).
        cmd: Shell command string.
        timeout: Max seconds to wait.
        check: If True, raise PipeError on non-zero exit.

    Usage:
        arun(self.shell, "make test").grep("FAIL")
        await arun(self.shell, "tail -f log.txt").head(100).collect()
    """

    async def _gen() -> AsyncIterator[str]:
        buffer = ""
        async for event in shell_tools.run_stream(cmd, timeout=timeout):
            if hasattr(event, "text"):
                if event.kind == "stdout":
                    buffer += event.text
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        yield line
            else:
                # Flush remaining partial line
                if buffer:
                    yield buffer
                if check and event.returncode != 0:
                    raise make_pipe_error(
                        f"Command failed: {cmd}",
                        cmd=cmd,
                        returncode=event.returncode,
                    )

    return Stream(_gen(), _steps=[f"arun({cmd!r})"])


def find(
    root: str | Path = ".",
    *,
    name: str | None = None,
    type: str | None = None,
    pattern: str | None = None,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
) -> Stream:
    """
    Walk a directory tree and stream matching paths via ripgrep (non-blocking).

    Uses `rg --files` for fast, .gitignore-aware directory traversal.
    Streams results line-by-line from BashSession.

    Args:
        root: Starting directory.
        name: Glob pattern for file name matching (e.g. "*.py").
        type: "f" for files only, "d" for dirs only (falls back to os.walk).
        pattern: Regex pattern to match full path (applied as Python post-filter).
        exclude: Glob patterns to exclude (e.g. ["*.pyc", "vendor/*"]).
        max_depth: Maximum depth to recurse.
        hidden: Search hidden files/dirs (--hidden).
        no_ignore: Don't respect .gitignore (--no-ignore).

    Usage:
        find(".", name="*.py").grep("test")
        find("src", name="*.rs").wc()
    """
    regex = _re.compile(pattern) if pattern else None

    # rg --files doesn't list directories; fall back for type="d"
    if type == "d":
        root_path = Path(root)
        exclude_set = set(exclude) if exclude else set()

        async def _gen_dirs() -> AsyncIterator[str]:
            def _walk_dirs() -> list[str]:
                results = []
                for dirpath_str, dirnames, _ in os.walk(root_path):
                    dirpath = Path(dirpath_str)
                    if max_depth is not None:
                        rel = dirpath.relative_to(root_path)
                        if len(rel.parts) > max_depth:
                            dirnames.clear()
                            continue
                    if exclude_set:
                        dirnames[:] = [
                            d
                            for d in dirnames
                            if not any(_fnmatch.fnmatch(d, pat) for pat in exclude_set)
                        ]
                    for d in dirnames:
                        path_str = str(dirpath / d)
                        if name and not _fnmatch.fnmatch(d, name):
                            continue
                        if regex and not regex.search(path_str):
                            continue
                        results.append(path_str)
                return results

            paths = await asyncio.to_thread(_walk_dirs)
            for p in paths:
                yield p

        return Stream(_gen_dirs(), _steps=[f"find({str(root)!r}, type='d')"])

    # Build rg --files command
    args = ["rg", "--files"]
    if hidden:
        args.append("--hidden")
    if no_ignore:
        args.append("--no-ignore")
    if max_depth is not None:
        args.append(f"--max-depth={max_depth}")
    if name:
        args.append(f"-g{name}")
    if exclude:
        for pat in exclude:
            args.append(f"-g!{pat}")
    args.append(str(root))

    cmd = " ".join(_shell_quote(a) for a in args)

    async def _gen() -> AsyncIterator[str]:
        async for line in _stream_bash_lines(
            cmd,
            timeout=30.0,
            ok_codes=(0, 1),  # rg --files returns 1 if no files found
            error_prefix="find failed",
        ):
            if regex and not regex.search(line):
                continue
            yield line

    return Stream(_gen(), _steps=[f"find({str(root)!r})"])


def glob(pattern: str, *, root: str | Path = ".") -> Stream:
    """
    Glob for files and stream matching paths (non-blocking).

    Usage:
        glob("**/*.py").grep("test")
    """
    root_path = Path(root)

    async def _gen() -> AsyncIterator[str]:
        paths = await asyncio.to_thread(lambda: [str(p) for p in sorted(root_path.glob(pattern))])
        for p in paths:
            yield p

    return Stream(_gen(), _steps=[f"glob({pattern!r})"])


def rg(
    pattern: str,
    path: str = ".",
    *,
    type_filter: str | None = None,
    include: str | None = None,
    exclude: list[str] | None = None,
    ignore_case: bool = False,
    fixed: bool = False,
    files_only: bool = False,
    context: int = 0,
    max_count: int | None = None,
    hidden: bool = False,
    no_ignore: bool = False,
) -> Stream:
    """
    Search with ripgrep and stream matching lines as they arrive (non-blocking).

    Uses the `rg` binary via BashSession for high-performance regex search.
    Respects .gitignore by default. Streams results line-by-line.

    Args:
        pattern: Regex pattern (or fixed string if fixed=True).
        path: File or directory to search.
        type_filter: Ripgrep type filter (e.g. "py", "rs", "js").
        include: Glob pattern for files to include (e.g. "*.py").
        exclude: Glob patterns to exclude.
        ignore_case: Case-insensitive search (-i).
        fixed: Fixed string search, not regex (-F).
        files_only: Only output file paths with matches (-l).
        context: Lines of context around each match (-C).
        max_count: Max matches per file (-m).
        hidden: Search hidden files/dirs (--hidden).
        no_ignore: Don't respect .gitignore (--no-ignore).

    Usage:
        rg("TODO", type_filter="py").cut(fields=[0], sep=":").sort().uniq()
        rg("def test_", include="*.py").wc()
        rg("FIXME", files_only=True).sort()
    """
    args = ["rg"]
    if ignore_case:
        args.append("-i")
    if fixed:
        args.append("-F")
    if files_only:
        args.append("-l")
    if hidden:
        args.append("--hidden")
    if no_ignore:
        args.append("--no-ignore")
    if context > 0:
        args.append(f"-C{context}")
    if max_count is not None:
        args.append(f"-m{max_count}")
    if type_filter:
        args.append(f"-t{type_filter}")
    if include:
        args.append(f"-g{include}")
    if exclude:
        for pat in exclude:
            args.append(f"-g!{pat}")

    args.append("--")
    args.append(pattern)
    args.append(path)

    cmd = " ".join(_shell_quote(a) for a in args)

    async def _gen() -> AsyncIterator[str]:
        async for line in _stream_bash_lines(
            cmd,
            timeout=30.0,
            ok_codes=(0, 1),  # rg returns 1 for "no matches"
            error_prefix="rg failed",
        ):
            yield line

    return Stream(_gen(), _steps=[f"rg({pattern!r})"])


def stdin() -> Stream:
    """Read lines from sys.stdin as a stream."""

    async def _gen() -> AsyncIterator[str]:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            yield line.rstrip("\n")

    return Stream(_gen(), _steps=["stdin()"])


def lines(text: str) -> Stream:
    """Create a stream from a multiline string."""
    return Stream(iter(text.splitlines()), _steps=["lines(...)"])


def items(iterable: Iterable) -> Stream:
    """Create a stream from any iterable."""
    return Stream(iter(iterable), _steps=["items(...)"])


def empty() -> Stream:
    """Create an empty stream."""
    return Stream(iter([]), _steps=["empty()"])


def seq(start: int = 1, end: int | None = None, *, step: int = 1) -> Stream:
    """
    Generate a numeric sequence (like seq).

    Args:
        start: Start value (inclusive). If end is None, generates 1..start.
        end: End value (inclusive). If None, start is treated as end with start=1.
        step: Step between values (must not be zero).

    Usage:
        seq(5)              # 1, 2, 3, 4, 5
        seq(2, 10, step=2)  # 2, 4, 6, 8, 10
    """
    if step == 0:
        raise ValueError("step cannot be zero")
    if end is None:
        actual_start, actual_end = 1, start
    else:
        actual_start, actual_end = start, end

    def _generate():
        i = actual_start
        if step > 0:
            while i <= actual_end:
                yield str(i)
                i += step
        else:
            while i >= actual_end:
                yield str(i)
                i += step

    return Stream(_generate(), _steps=[f"seq({actual_start}, {actual_end})"])


def _shell_quote(s: str) -> str:
    """Simple shell quoting for arguments."""
    if not s:
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./=")
    if all(c in safe_chars for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"
