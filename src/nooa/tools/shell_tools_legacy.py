# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ShellTools — persistent shell session with file operations.

Replaces ``BashTool`` and ``FileTool`` with a unified skill that maintains
a persistent bash session.  ``cd``, ``export``, environment variables, and
working directory all survive across calls.

Attach to an agent::

    class MyAgent(Agent, llm=llm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.shell = ShellTools(cwd="/path/to/repo")

The LLM discovers tools via ``doc(self.shell)`` and sees the current
working directory in ``pprint(self)`` thanks to ``__repr__``.
"""

import difflib
import logging
import re
import shlex
import unicodedata
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill
from nooa.tools._bash_session import BashSession
from nooa.tools._results import (
    EditResult,
    LsResult,
    RunResult,
    SearchResult,
    StreamDone,
    StreamEvent,
    ViewResult,
    WriteResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_VIEW_LINES = 2000
_TRUNCATED_MSG = (
    "\n<response clipped>\n"
    "NOTE: Output truncated. Use grep to search within the file, "
    "or view with offset/limit to see specific sections."
)

# Directories to skip in ls/find/grep fallback
_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
    }
)


class ShellToolsLegacy(Skill):
    """Persistent shell session with file operations.

    Provides a bash shell where ``cd``, ``export``, and environment changes
    persist across calls, plus file viewing, editing, searching, and
    directory listing.

    Commands are serialized via an internal asyncio.Lock — concurrent
    ``run()`` calls queue and execute one at a time.  For true parallelism,
    create multiple ShellTools instances.

    Tools:
        run(command, ...)       — run a shell command (stateful)
        run_stream(command, ...) — stream stdout/stderr and terminal event
        view(path, ...)         — read a file with line numbers
        edit(path, old, new)    — str_replace with fuzzy fallback
        insert(path, line, content) — insert text at a line number
        write(path, content)    — create or overwrite a file
        grep(pattern, path)     — regex search (ripgrep with grep fallback)
        find(pattern, path)     — find files by glob (ripgrep with find fallback)
        ls(path, depth)         — tree-structured directory listing
    """

    __nosnapshot__ = True

    def __init__(self, cwd: str | Path = ".", init_command: str | None = None) -> None:
        # ``init_command`` (if given) runs once on session start, before any user
        # command, to set up the environment (e.g. activate a conda env).
        self._session = BashSession(cwd=cwd, init_command=init_command)
        self._current_file: str | None = None
        self._current_line: int | None = None
        self._rg_available: bool | None = None

    @property
    @hidden
    def session(self) -> BashSession:
        """The underlying persistent bash session (shared with e.g. RepoTools)."""
        return self._session

    async def _has_rg(self) -> bool:
        """Check whether ripgrep (rg) is available, caching the result."""
        if self._rg_available is None:
            _, _, code = await self._session.run("command -v rg", timeout=5)
            self._rg_available = code == 0
            if not self._rg_available:
                logger.warning(
                    "ripgrep (rg) not found; falling back to grep/find. "
                    "Install ripgrep for better performance: https://github.com/BurntSushi/ripgrep"
                )
        return self._rg_available

    @property
    def cwd(self) -> Path:
        """Current working directory of the shell session."""
        return self._session.cwd

    def __repr__(self) -> str:
        parts = [f"cwd={str(self._session.cwd)!r}"]
        if self._current_file:
            loc = self._current_file
            if self._current_line:
                loc += f":{self._current_line}"
            parts.append(f"file={loc!r}")
        return f"ShellTools({', '.join(parts)})"

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    async def run(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        timeout: Annotated[float, spec(description="Max seconds to wait before timeout")] = 30.0,
    ) -> RunResult:
        '''Run a shell command in the persistent bash session.

        State persists: ``cd``, ``export``, ``source``, aliases, and
        environment variables carry over between calls.

        Always use triple-quoted strings for the command.  Single- or
        double-quoted strings cause SyntaxErrors when the command
        contains newlines or heredocs::

            # BAD — SyntaxError (unterminated string literal):
            await shell.run("cat <<EOF
            hello
            EOF")

            # GOOD — triple-quoted string handles newlines:
            await shell.run("""cat <<EOF
            hello
            EOF""")

        Examples::

            result = await shell.run("""cd src && ls""")
            result = await shell.run("""git diff HEAD~1""")
            result = await shell.run("""python -m pytest tests/ -x""", timeout=300)
        '''
        stdout, stderr, code, timed_out = await self._session.run_with_timeout_flag(
            command, timeout=timeout
        )
        result = RunResult(
            stdout=stdout,
            stderr=stderr,
            returncode=code,
            timed_out=timed_out,
        )
        self._record_bash_metrics(command, result)
        return result

    async def run_stream(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        timeout: Annotated[float, spec(description="Max seconds to wait before timeout")] = 30.0,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        """Stream command output as it arrives, ending with a done event.

        Yields StreamEvent chunks for stdout/stderr incrementally, then a
        final StreamDone with the exit code once the command completes.

        Always use triple-quoted strings for the command (same as ``run``).
        """
        timed_out = False
        exit_code = 0
        async for stream_name, chunk in self._session.run_stream(command, timeout=timeout):
            if stream_name == "__done__":
                parts = chunk.split(",")
                exit_code = int(parts[0])
                timed_out = bool(int(parts[1])) if len(parts) > 1 else False
                break
            yield StreamEvent(kind=stream_name, text=chunk)
        yield StreamDone(kind="done", returncode=exit_code, timed_out=timed_out)

    # ------------------------------------------------------------------
    # view
    # ------------------------------------------------------------------
    async def view(
        self,
        path: Annotated[str, spec(description="File or directory path (relative to cwd)")],
        offset: Annotated[int, spec(description="Start line (1-indexed)")] = 1,
        limit: Annotated[int, spec(description="Max lines to show")] = _MAX_VIEW_LINES,
    ) -> ViewResult:
        """Read a file with numbered lines. If path is a directory, returns a tree listing."""
        resolved = self._resolve(path)

        if resolved.is_dir():
            ls_r = await self.ls(path, depth=2)
            return ViewResult(
                path=path,
                content=ls_r.tree,
                start_line=0,
                end_line=0,
                total_lines=ls_r.num_files,
            )

        if not resolved.is_file():
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().shell_failure(
                "view:file_not_found", f"File not found: {path}", path
            )
            return ViewResult(
                path=path,
                content=f"Error: {path} not found",
                start_line=0,
                end_line=0,
                total_lines=0,
            )

        text = resolved.read_text(errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = max(1, offset)
        end = min(total, start + limit - 1)
        selected = lines[start - 1 : end]

        width = len(str(end))
        numbered = "\n".join(f"{i:{width}d}|{line}" for i, line in enumerate(selected, start))

        truncated = end < total
        if truncated:
            numbered += _TRUNCATED_MSG

        self._current_file = path
        self._current_line = start

        return ViewResult(
            path=path,
            content=numbered,
            start_line=start,
            end_line=end,
            total_lines=total,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # edit
    # ------------------------------------------------------------------
    async def edit(
        self,
        path: Annotated[str, spec(description="File to edit (relative to cwd)")],
        old_str: Annotated[
            str,
            spec(
                description="Exact text to find (must be unique). Fuzzy matching handles minor whitespace/unicode differences."
            ),
        ],
        new_str: Annotated[str, spec(description="Replacement text")],
    ) -> EditResult:
        """Replace text in a file using str_replace with fuzzy fallback.

        Matching strategy:
        1. Exact match (preferred)
        2. Fuzzy match (whitespace + unicode normalization)
        3. Line-number prefixes from view() output are stripped automatically

        Args:
            path: File to edit (relative to cwd).
            old_str: Exact text to find (must be unique). Fuzzy matching handles
                minor whitespace/unicode differences.
            new_str: Replacement text.

        Returns:
            EditResult with diff, lint errors, and success status.
            On failure, returns the closest match as a hint.

        Examples:
            r = await self.shell.edit("src/main.py",
                old_str='print("hello")',
                new_str='print("hello world")')
        """
        resolved = self._resolve(path)
        if not resolved.is_file():
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().shell_failure(
                "edit:file_not_found", f"File not found: {path}", path
            )
            return EditResult(path=path, diff="", success=False, error=f"File not found: {path}")

        content = resolved.read_text(errors="replace")

        # Strip line-number prefixes that agents copy from view() output
        old_str = _strip_line_number_prefixes(old_str)

        # Try exact match first
        count = content.count(old_str)

        if count == 0:
            # Fallback: fuzzy match (whitespace + unicode normalization)
            fuzzy_result = _fuzzy_find_unique(content, old_str)
            if fuzzy_result is None:
                from nooa.runtime.harness_metrics import get_harness_metrics

                get_harness_metrics().shell_failure(
                    "edit:no_match",
                    f"old_str not found in {path}",
                    old_str[:200],
                )
                hint = _find_closest_match(content, old_str)
                error_msg = f"old_str not found in {path}."
                if hint:
                    error_msg += f" Closest match:\n{hint}"
                else:
                    error_msg += " Make sure it matches the file content, including whitespace and indentation."
                return EditResult(path=path, diff="", success=False, error=error_msg)
            # Apply fuzzy-matched edit in normalized space
            norm_content = _normalize_for_fuzzy(content)
            norm_old = _normalize_for_fuzzy(old_str)
            idx, _ = fuzzy_result
            # Reject fuzzy match that starts mid-line (line-based mapping would be inaccurate)
            if idx != 0 and norm_content[idx - 1] != "\n":
                hint = _find_closest_match(content, old_str)
                error_msg = f"old_str not found in {path}."
                if hint:
                    error_msg += f" Closest match:\n{hint}"
                else:
                    error_msg += " Make sure it matches the file content, including whitespace and indentation."
                return EditResult(path=path, diff="", success=False, error=error_msg)
            # Map back to original content using line-based alignment
            norm_lines_before = norm_content[:idx].count("\n")
            orig_lines = content.split("\n")
            # Find the matching chunk in original by line count
            norm_match_lines = norm_old.count("\n") + 1
            orig_end_line = norm_lines_before + norm_match_lines
            orig_chunk = "\n".join(orig_lines[norm_lines_before:orig_end_line])
            if orig_end_line <= len(orig_lines) and norm_lines_before < len(orig_lines):
                # Use slice-based replacement to avoid hitting wrong occurrence
                chunk_start = len("\n".join(orig_lines[:norm_lines_before]))
                if norm_lines_before > 0:
                    chunk_start += 1  # account for the newline separator
                chunk_end = chunk_start + len(orig_chunk)
                new_content = content[:chunk_start] + new_str + content[chunk_end:]
            else:
                return EditResult(
                    path=path,
                    diff="",
                    success=False,
                    error=(
                        f"old_str not found in {path} (fuzzy match alignment failed). "
                        "Make sure it matches the file content."
                    ),
                )
        elif count > 1:
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().shell_failure(
                "edit:multiple_matches",
                f"old_str found {count} times in {path}",
                old_str[:200],
            )
            return EditResult(
                path=path,
                diff="",
                success=False,
                error=(
                    f"old_str found {count} times in {path}. "
                    "It must match exactly once — add surrounding context to make it unique."
                ),
            )
        else:
            # Exact match - apply directly
            new_content = content.replace(old_str, new_str, 1)

        # Pre-edit lint
        pre_errors = await self._lint(resolved)

        # Write the edited content
        resolved.write_text(new_content)

        diff = _unified_diff(path, content, new_content)

        # Post-edit lint — report only new errors
        post_errors = await self._lint(resolved)
        new_errors = sorted(set(post_errors) - set(pre_errors))

        # Update current location
        self._current_file = path
        if old_str in content:
            self._current_line = content[: content.index(old_str)].count("\n") + 1
        else:
            # Fuzzy match was used; estimate from diff
            self._current_line = 1

        return EditResult(
            path=path,
            diff=diff,
            lint_errors=new_errors,
            success=True,
        )

    # insert
    # ------------------------------------------------------------------
    async def insert(self, path: str, line: int, content: str) -> EditResult:
        """Insert content at a specific line number.

        Args:
            path: File to edit (relative to cwd).
            line: Line number to insert before (1-indexed).
                  0 = prepend to file, -1 = append to end.
            content: Text to insert.

        Returns:
            EditResult with diff, lint errors, and success status.

        Examples:
            r = await self.shell.insert("src/main.py", 1, "import os\n")
            r = await self.shell.insert("src/main.py", -1, "\n# end\n")
        """
        resolved = self._resolve(path)
        if not resolved.is_file():
            return EditResult(path=path, diff="", success=False, error=f"File not found: {path}")

        old_content = resolved.read_text(errors="replace")
        lines = old_content.split("\n")

        if line == 0:
            new_content = content + old_content
        elif line == -1:
            new_content = old_content + content
        else:
            # 1-indexed: insert before that line
            idx = max(0, min(line - 1, len(lines)))
            lines.insert(idx, content.rstrip("\n"))
            new_content = "\n".join(lines)

        # Pre-edit lint
        pre_errors = await self._lint(resolved)

        resolved.write_text(new_content)
        diff = _unified_diff(path, old_content, new_content)

        # Post-edit lint
        post_errors = await self._lint(resolved)
        new_errors = sorted(set(post_errors) - set(pre_errors))

        self._current_file = path
        self._current_line = len(lines) if line == -1 else max(1, line)

        return EditResult(path=path, diff=diff, lint_errors=new_errors, success=True)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    async def write(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd)")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> WriteResult:
        """Create or overwrite a file. Creates parent directories if needed."""
        resolved = self._resolve(path)
        created = not resolved.exists()

        diff = ""
        truncation_warning = ""
        if not created:
            old = resolved.read_text(errors="replace")
            diff = _unified_diff(path, old, content)
            # Guard against accidental truncation
            if len(old) > 100 and len(content) < len(old) * 0.7:
                pct = round((1 - len(content) / len(old)) * 100)
                truncation_warning = (
                    f"WARNING: file shrunk by {pct}% "
                    f"({len(old)} -> {len(content)} chars). "
                    "If unintentional, the file may be truncated."
                )

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)

        self._current_file = path
        self._current_line = 1

        result = WriteResult(
            path=path,
            created=created,
            lines=len(content.splitlines()),
            diff=diff,
        )
        if truncation_warning:
            result.diff = truncation_warning + "\n" + result.diff
        return result

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------
    async def grep(
        self,
        pattern: Annotated[str, spec(description="Regex pattern (or literal if literal=True)")],
        path: Annotated[str, spec(description="File or directory to search")] = ".",
        *,
        include: Annotated[str | None, spec(description="Glob filter (e.g. '*.py')")] = None,
        context: Annotated[int, spec(description="Lines of context around each match")] = 0,
        literal: Annotated[bool, spec(description="Treat pattern as literal string")] = False,
        max_matches: Annotated[int, spec(description="Maximum matches to return")] = 100,
        timeout: Annotated[float, spec(description="Timeout in seconds for the search")] = 30.0,
    ) -> SearchResult:
        """Search for a regex pattern in files. Uses ripgrep if available, falls back to grep."""
        if await self._has_rg():
            cmd = self._build_rg_grep_cmd(
                pattern,
                path,
                include=include,
                context=context,
                literal=literal,
                max_matches=max_matches,
            )
        else:
            cmd = self._build_fallback_grep_cmd(
                pattern,
                path,
                include=include,
                context=context,
                literal=literal,
                max_matches=max_matches,
            )

        stdout, _, code = await self._session.run(cmd, timeout=timeout)

        if code == 1:  # no matches
            return SearchResult(matches=[], total_matches=0)

        all_lines = [ln for ln in stdout.splitlines() if ln.strip()] if stdout else []
        if context > 0:
            # With -C, both rg and grep output match lines as "file:num:text"
            # and context lines as "file:num-text". Separators are "--".
            total = sum(1 for ln in all_lines if ln != "--" and re.search(r"(?:^|:)\d+:", ln))
        else:
            total = len(all_lines)
        return SearchResult(
            matches=all_lines[:max_matches],
            total_matches=total,
            truncated=total >= max_matches,
        )

    @staticmethod
    def _build_rg_grep_cmd(
        pattern: str,
        path: str,
        *,
        include: str | None,
        context: int,
        literal: bool,
        max_matches: int,
    ) -> str:
        """Build a ripgrep command string."""
        parts = ["rg", "-n", "--color=never", "--no-heading"]
        if literal:
            parts.append("-F")
        if context > 0:
            parts.extend(["-C", str(context)])
        if include:
            parts.extend(["-g", _sq(include)])
        parts.extend(["-m", str(max_matches), "--", _sq(pattern), _sq(path)])
        return " ".join(parts)

    @staticmethod
    def _build_fallback_grep_cmd(
        pattern: str,
        path: str,
        *,
        include: str | None,
        context: int,
        literal: bool,
        max_matches: int,
    ) -> str:
        """Build a GNU/BSD grep command string as ripgrep fallback."""
        parts = ["grep", "-rn", "-I", "--color=never"]
        for d in sorted(_IGNORE_DIRS):
            parts.extend(["--exclude-dir", _sq(d)])
        if literal:
            parts.append("-F")
        if context > 0:
            parts.extend(["-C", str(context)])
        if include:
            parts.extend(["--include", _sq(include)])
        parts.extend(["-m", str(max_matches), "--", _sq(pattern), _sq(path)])
        return " ".join(parts)

    # ------------------------------------------------------------------
    # find
    # ------------------------------------------------------------------
    async def find(
        self,
        pattern: Annotated[str, spec(description="Glob pattern (e.g. '*.py', 'test_*.py')")],
        path: Annotated[str, spec(description="Directory to search")] = ".",
        *,
        entry_type: Annotated[str, spec(description="'f' for files, 'd' for directories")] = "f",
        max_results: Annotated[int, spec(description="Maximum results to return")] = 200,
    ) -> SearchResult:
        """Find files or directories by glob pattern. Uses ripgrep if available, falls back to find."""
        prune = " ".join(f"-not -path '*/{d}/*'" for d in sorted(_IGNORE_DIRS))
        if entry_type == "d":
            cmd = f"find {_sq(path)} -maxdepth 10 -type d -name {_sq(pattern)} {prune}"
        elif await self._has_rg():
            cmd = f"rg --files -g {_sq(pattern)} {_sq(path)}"
        else:
            cmd = f"find {_sq(path)} -maxdepth 10 -type f -name {_sq(pattern)} {prune}"

        stdout, _, _ = await self._session.run(cmd, timeout=30)
        lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()] if stdout else []
        total = len(lines)
        return SearchResult(
            matches=lines[:max_results],
            total_matches=total,
            truncated=total > max_results,
        )

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------
    async def ls(
        self,
        path: Annotated[str, spec(description="Directory to list")] = ".",
        depth: Annotated[int, spec(description="Maximum depth to recurse")] = 3,
        max_entries: Annotated[int, spec(description="Maximum entries to show")] = 500,
    ) -> LsResult:
        """List directory contents as a tree."""
        resolved = self._resolve(path)
        if not resolved.is_dir():
            return LsResult(path=path, tree=f"Error: {path} is not a directory", num_files=0)

        lines: list[str] = []
        count = 0
        truncated = False

        def _walk(dir_path: Path, prefix: str, cur_depth: int) -> None:
            nonlocal count, truncated
            if truncated or cur_depth > depth:
                return
            try:
                entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return

            entries = [
                e for e in entries if e.name not in _IGNORE_DIRS and not e.name.startswith(".")
            ]

            for i, entry in enumerate(entries):
                if count >= max_entries:
                    truncated = True
                    return
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{connector}{entry.name}{suffix}")
                count += 1
                if entry.is_dir():
                    ext = "    " if is_last else "│   "
                    _walk(entry, prefix + ext, cur_depth + 1)

        lines.append(f"{path}/")
        _walk(resolved, "", 1)

        return LsResult(
            path=path,
            tree="\n".join(lines),
            num_files=count,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Patterns that suggest the LLM is bypassing higher-level tools
    _AVOIDANCE_PATTERNS = (
        (re.compile(r"^sed\s+-i"), "edit"),
        (re.compile(r"^(sed|awk)\s"), "edit"),
        (re.compile(r"^cat\s+\S+"), "view"),
        (re.compile(r"^head\s+"), "view"),
        (re.compile(r"^tail\s+"), "view"),
        (re.compile(r"^(rg|grep)\s+"), "grep"),
        (re.compile(r"^find\s+"), "find"),
    )
    # Shell operators that justify using bash instead of a higher-level tool
    _PIPE_REDIRECT_RE = re.compile(r"[|><]")

    def _record_bash_metrics(self, command: str, result: RunResult) -> None:
        """Record harness metrics for bash invocations."""
        from nooa.runtime.harness_metrics import get_harness_metrics

        hm = get_harness_metrics()

        # Record failures (non-zero exit, timeout)
        if result.timed_out:
            hm.shell_failure("bash:timeout", f"Command timed out: {command[:200]}", command[:200])
        elif result.returncode != 0:
            # Skip exit code 1 for grep/rg (means "no matches", not failure)
            # Skip leading env var assignments (e.g. FOO=bar grep ...) to find actual command
            tokens = command.lstrip().split() if command.strip() else []
            cmd_base = ""
            for tok in tokens:
                if "=" not in tok or tok.startswith("-"):
                    cmd_base = tok
                    break
            if not (result.returncode == 1 and cmd_base in ("grep", "rg")):
                msg = (result.stderr or result.stdout or "")[:200]
                hm.shell_failure(
                    f"bash:exit_{result.returncode}",
                    f"{cmd_base}: {msg}",
                    command[:200],
                )

        # Detect tool avoidance patterns
        # Strip leading env vars, cd prefixes, && chains — check each segment
        for segment in command.split("&&"):
            stripped = segment.strip()
            # Skip cd-only segments
            if stripped.startswith("cd "):
                continue
            for pattern, tool_name in self._AVOIDANCE_PATTERNS:
                if pattern.search(stripped):
                    # Distinguish: pipe/redirect justifies using bash
                    if self._PIPE_REDIRECT_RE.search(stripped):
                        hm.tool_avoided(
                            f"bash({stripped[:100]}) → using {tool_name} via bash (pipe/redirect)"
                        )
                    else:
                        hm.tool_avoided(f"bash({stripped[:100]}) → should use shell.{tool_name}")
                    break

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to the session cwd.

        Safe to call without holding the session lock: captures cwd at call
        time, and callers perform file I/O synchronously (no await between
        resolve and read/write), so no concurrent command can change cwd
        in between.
        """
        p = Path(path)
        return p if p.is_absolute() else self._session.cwd / p

    async def _lint(self, path: Path) -> list[str]:
        """Quick syntax check. Returns list of error strings."""
        errors: list[str] = []
        if path.suffix == ".py":
            stdout, stderr, code = await self._session.run(
                f'python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())"'
                f" {_sq(str(path))}",
                timeout=10,
            )
            if code != 0:
                for line in (stderr or stdout).splitlines():
                    if line.strip():
                        errors.append(line.strip())
        return errors

    async def reset(self) -> None:
        """Kill and restart the bash session, preserving the working directory.

        Use when the session is corrupted (e.g. after a timeout caused by
        unbalanced quotes or a stuck interactive process).
        """
        await self._session.reset()

    async def close(self) -> None:
        """Shut down the persistent bash session."""
        await self._session.close()


def _unified_diff(path: str, old: str, new: str) -> str:
    """Generate a unified diff between old and new content."""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _sq(s: str) -> str:
    """Shell-quote a string."""
    return shlex.quote(s)


def _normalize_for_fuzzy(text: str) -> str:
    """Normalize text for fuzzy matching.

    Strips trailing whitespace per line and normalizes unicode characters
    that models commonly substitute (smart quotes, dashes, special spaces).
    """
    text = unicodedata.normalize("NFKC", text)
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)
    # Smart single quotes to ASCII
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    # Smart double quotes to ASCII
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)
    # Various dashes/hyphens to ASCII hyphen
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    # Special spaces to regular space
    text = re.sub(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", text)
    return text


def _fuzzy_find_unique(content: str, old_str: str) -> tuple[int, int] | None:
    """Find old_str in content using fuzzy normalization.

    Returns (start_index, length_in_normalized_content) if exactly one match
    is found in the normalized content, else None.
    """
    norm_content = _normalize_for_fuzzy(content)
    norm_old = _normalize_for_fuzzy(old_str)
    if not norm_old:
        return None
    count = norm_content.count(norm_old)
    if count != 1:
        return None
    idx = norm_content.index(norm_old)
    return (idx, len(norm_old))


def _strip_line_number_prefixes(text: str) -> str:
    """Strip line-number prefixes that agents copy from view() output.

    Detects patterns like '  42|', ' 7|', '100|' at the start of each line
    and removes them if the majority of lines match.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text
    pattern = re.compile(r"^\s*\d+\|")
    matching = sum(1 for line in lines if pattern.match(line) or not line.strip())
    if matching < len(lines) * 0.8:
        return text
    stripped = []
    for line in lines:
        m = re.match(r"^\s*\d+\|(.*)$", line)
        stripped.append(m.group(1) if m else line)
    return "\n".join(stripped)


def _find_closest_match(content: str, target: str, threshold: float = 0.6) -> str | None:
    """Find the most similar chunk in content to target.

    Returns a short snippet of the closest match, or None if nothing is close enough.
    Caps search at 500 candidate positions to avoid O(n^2) on large files.
    """
    target_lines = target.split("\n")
    content_lines = content.split("\n")
    n = len(target_lines)
    if n == 0 or len(content_lines) < n:
        return None

    best_ratio = 0.0
    best_chunk = None
    max_candidates = min(len(content_lines) - n + 1, 500)
    for i in range(max_candidates):
        chunk = "\n".join(content_lines[i : i + n])
        ratio = difflib.SequenceMatcher(None, target, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_chunk = chunk

    if best_ratio < threshold or best_chunk is None:
        return None
    # Return a truncated preview
    preview = best_chunk[:200]
    if len(best_chunk) > 200:
        preview += "..."
    return preview
