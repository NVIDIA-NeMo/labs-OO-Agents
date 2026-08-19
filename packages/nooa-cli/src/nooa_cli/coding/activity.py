# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in shell tools and events shared by interactive coding hosts."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from difflib import unified_diff
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from uuid import uuid4

from pydantic import Field

from nooa.agentdoc import TruncatingStringIO, hidden, pformat, spec
from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role
from nooa.runtime.event_manager import EventManager
from nooa.skill import Skill
from nooa.tools._bash_session import BashSession
from nooa.tools._results import StreamDone, StreamEvent
from nooa.tools.shell_tools import FileWrite, Match, ShellResult, ShellTools

logger = logging.getLogger(__name__)

_MAX_EVENT_TEXT_CHARS = 10_000
_MAX_COMMAND_OUTPUT_CHARS = 30_000


def _line_count(value: str) -> int:
    if not value:
        return 0
    return value.count("\n") + (0 if value.endswith("\n") else 1)


def _edit_diff(path: str, old_text: str, new_text: str, start_line: int | None) -> str:
    lines = list(
        unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if start_line is not None:
        for index, line in enumerate(lines):
            if line.startswith("@@ "):
                old_count = max(1, _line_count(old_text))
                new_count = max(1, _line_count(new_text))
                lines[index] = f"@@ -{start_line},{old_count} +{start_line},{new_count} @@\n"
                break
    return pformat("".join(lines), max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)


class FileEdit(EventBase):  # type: ignore[misc]
    """A successful structured filesystem edit made by the coding agent."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    path: Annotated[str, Field(description="Absolute path of the edited file")]
    operation: Annotated[
        Literal["create", "update"],
        Field(description="Whether the file was created or updated"),
    ]
    old_text: Annotated[
        str | None,
        Field(description="Bounded affected text before the edit, when available"),
    ] = None
    new_text: Annotated[str, Field(description="Bounded affected text after the edit")] = ""
    start_line: Annotated[
        int | None,
        Field(description="First affected line in the original file, 1-indexed"),
    ] = None
    end_line: Annotated[
        int | None,
        Field(description="Last affected line in the original file, 1-indexed and inclusive"),
    ] = None
    diff: Annotated[str, Field(description="Bounded unified diff of the affected text")] = ""
    content_complete: Annotated[
        bool,
        Field(description="Whether old_text and new_text contain the complete affected text"),
    ] = True


class TerminalCommandStarted(EventBase):  # type: ignore[misc]
    """A command began in a persistent coding-agent terminal."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    command_id: Annotated[str, Field(description="Correlation ID for this command")]
    command: Annotated[str, Field(description="Bounded shell command text")]
    working_directory: Annotated[str, Field(description="Working directory at command start")]
    stdin: Annotated[
        str | None,
        Field(description="Bounded separate stdin text; never stored in session history"),
    ] = None
    command_truncated: Annotated[
        bool,
        Field(description="Whether command was truncated by pformat for this event"),
    ] = False
    stdin_truncated: Annotated[
        bool,
        Field(description="Whether stdin was truncated by pformat for this event"),
    ] = False


class TerminalCommandOutput(EventBase):  # type: ignore[misc]
    """One streaming output chunk from a coding-agent terminal command."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    command_id: Annotated[str, Field(description="Correlation ID for this command")]
    stdout: Annotated[str, Field(description="Output chunk received on stdout")] = ""
    stderr: Annotated[str, Field(description="Output chunk received on stderr")] = ""
    truncated: Annotated[
        bool,
        Field(description="Whether further command output was omitted from activity events"),
    ] = False


class TerminalCommandFinished(EventBase):  # type: ignore[misc]
    """A coding-agent terminal command completed or failed to launch."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    command_id: Annotated[str, Field(description="Correlation ID for this command")]
    exit_code: Annotated[int | None, Field(description="Process exit code when available")] = None
    timed_out: Annotated[bool, Field(description="Whether the command timed out")] = False
    error: Annotated[str, Field(description="Failure before an exit code was available")] = ""
    output_truncated: Annotated[
        bool,
        Field(description="Whether output was omitted from command activity events"),
    ] = False


class ActivityShellTools(Skill):
    """A composed ``ShellTools`` substitute that emits transient activity events.

    Interactive agents opt into this class explicitly. The underlying
    ``ShellTools`` remains independent of event management and host UX.
    """

    def __init__(
        self,
        shell: ShellTools,
        event_manager: EventManager,
    ):
        super().__init__()
        self._shell = shell
        self._event_manager = event_manager

    def __repr__(self) -> str:
        return f"ActivityShellTools(cwd={self.cwd!s})"

    @property
    @hidden
    def cwd(self) -> Path:
        """Current working directory of the wrapped persistent shell."""
        return self._shell.cwd

    @cwd.setter
    def cwd(self, value: str | Path) -> None:
        self._shell.cwd = Path(value)

    @property
    @hidden
    def session(self) -> BashSession:
        """The wrapped shell's persistent bash session."""
        return self._shell.session

    @hidden
    async def close(self) -> None:
        """Close the wrapped shell."""
        await self._shell.close()

    def _resolve_path(self, path: str) -> Path:
        return self._shell._resolve_path(path)

    def _emit(self, event: EventBase) -> None:
        try:
            self._event_manager.add(event)
        except Exception:
            logger.debug("Failed to emit shell activity", exc_info=True)

    @wraps(ShellTools.run)
    async def run(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        *,
        stdin: Annotated[
            str | None, spec(description="Text piped to stdin (replaces heredocs)")
        ] = None,
        timeout: Annotated[float, spec(description="Max seconds")] = 30.0,
    ) -> ShellResult:
        command_id = str(uuid4())
        bounded_command = pformat(command, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
        command_truncated = len(command) > _MAX_EVENT_TEXT_CHARS
        bounded_stdin: str | None = None
        stdin_truncated = False
        if stdin is not None:
            bounded_stdin = pformat(stdin, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
            stdin_truncated = len(stdin) > _MAX_EVENT_TEXT_CHARS
        self._emit(
            TerminalCommandStarted(
                command_id=command_id,
                command=bounded_command,
                working_directory=str(self.cwd),
                stdin=bounded_stdin,
                command_truncated=command_truncated,
                stdin_truncated=stdin_truncated,
            )
        )
        try:
            result = await self._shell.run(command, stdin=stdin, timeout=timeout)
        except BaseException as error:
            self._emit(
                TerminalCommandFinished(
                    command_id=command_id,
                    error=str(error) or type(error).__name__,
                )
            )
            raise

        stdout_buffer = TruncatingStringIO(limit=_MAX_COMMAND_OUTPUT_CHARS // 2)
        stderr_buffer = TruncatingStringIO(limit=_MAX_COMMAND_OUTPUT_CHARS // 2)
        stdout_buffer.write(result.stdout)
        stderr_buffer.write(result.stderr)
        output_truncated = stdout_buffer.was_truncated or stderr_buffer.was_truncated
        if result.stdout or result.stderr:
            self._emit(
                TerminalCommandOutput(
                    command_id=command_id,
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    truncated=output_truncated,
                )
            )
        self._emit(
            TerminalCommandFinished(
                command_id=command_id,
                exit_code=result.returncode,
                timed_out=result.timed_out,
                output_truncated=output_truncated,
            )
        )
        return result

    @wraps(ShellTools.run_stream)
    async def run_stream(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        timeout: Annotated[float, spec(description="Max seconds to wait before timeout")] = 30.0,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        command_id = str(uuid4())
        bounded_command = pformat(command, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
        command_truncated = len(command) > _MAX_EVENT_TEXT_CHARS
        self._emit(
            TerminalCommandStarted(
                command_id=command_id,
                command=bounded_command,
                working_directory=str(self.cwd),
                command_truncated=command_truncated,
            )
        )
        finished = False
        stdout_buffer = TruncatingStringIO(limit=_MAX_COMMAND_OUTPUT_CHARS // 2)
        stderr_buffer = TruncatingStringIO(limit=_MAX_COMMAND_OUTPUT_CHARS // 2)
        stream = self._shell.run_stream(command, timeout=timeout)
        try:
            async for item in stream:
                if isinstance(item, StreamDone):
                    output_truncated = stdout_buffer.was_truncated or stderr_buffer.was_truncated
                    if stdout_buffer.chars_written or stderr_buffer.chars_written:
                        self._emit(
                            TerminalCommandOutput(
                                command_id=command_id,
                                stdout=stdout_buffer.getvalue(),
                                stderr=stderr_buffer.getvalue(),
                                truncated=output_truncated,
                            )
                        )
                    self._emit(
                        TerminalCommandFinished(
                            command_id=command_id,
                            exit_code=item.returncode,
                            timed_out=item.timed_out,
                            output_truncated=output_truncated,
                        )
                    )
                    finished = True
                else:
                    buffer = stdout_buffer if item.kind == "stdout" else stderr_buffer
                    buffer.write(item.text)
                yield item
        except BaseException as error:
            if not finished:
                output_truncated = stdout_buffer.was_truncated or stderr_buffer.was_truncated
                if stdout_buffer.chars_written or stderr_buffer.chars_written:
                    self._emit(
                        TerminalCommandOutput(
                            command_id=command_id,
                            stdout=stdout_buffer.getvalue(),
                            stderr=stderr_buffer.getvalue(),
                            truncated=output_truncated,
                        )
                    )
                self._emit(
                    TerminalCommandFinished(
                        command_id=command_id,
                        error=str(error) or type(error).__name__,
                        output_truncated=output_truncated,
                    )
                )
            raise
        finally:
            await stream.aclose()

    @wraps(ShellTools.read)
    async def read(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd or absolute)")],
        lines: Annotated[
            tuple[int, int] | None,
            spec(description="(start, end) 1-indexed inclusive, or None for whole file"),
        ] = None,
    ) -> Match:
        return await self._shell.read(path, lines)

    @wraps(ShellTools.replace)
    async def replace(
        self,
        target: Annotated[
            Any, spec(description="A Match (from read() or run().matches) or a file path string")
        ],
        old_or_new: Annotated[
            str,
            spec(
                description="For Match: replacement text. For path: text to find (must be unique)"
            ),
        ] = "",
        new: Annotated[
            str | None, spec(description="For path: replacement text. Leave None for Match.")
        ] = None,
    ) -> FileWrite:
        path = target.path if isinstance(target, Match) else target
        if not isinstance(path, str):
            return await self._shell.replace(target, old_or_new, new)

        resolved = (
            Path(target.resolved_path) if isinstance(target, Match) else self._resolve_path(path)
        )
        if isinstance(target, Match):
            old_text = target.text
            start_line = target.start
            end_line = target.end
        else:
            old_text = old_or_new
            start_line = None
            end_line = None
            try:
                with resolved.open("r") as stream:
                    content = stream.read(_MAX_EVENT_TEXT_CHARS + 1)
                if len(content) <= _MAX_EVENT_TEXT_CHARS:
                    offset = content.index(old_text)
                    start_line = content.count("\n", 0, offset) + 1
                    end_line = start_line + max(1, _line_count(old_text)) - 1
            except (OSError, UnicodeError, ValueError):
                pass

        bounded_old = pformat(old_text, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
        old_truncated = len(old_text) > _MAX_EVENT_TEXT_CHARS
        result = await self._shell.replace(target, old_or_new, new)
        # replace() may re-terminate the region, so report what it wrote rather
        # than what it was handed.
        written_text = result.new_text
        bounded_new = pformat(written_text, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
        new_truncated = len(written_text) > _MAX_EVENT_TEXT_CHARS
        self._emit(
            FileEdit(
                path=str(resolved),
                operation="update",
                old_text=bounded_old,
                new_text=bounded_new,
                start_line=start_line,
                end_line=end_line,
                diff=_edit_diff(path, bounded_old, bounded_new, start_line),
                content_complete=not old_truncated and not new_truncated,
            )
        )
        return result

    @wraps(ShellTools.write_file)
    async def write_file(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd or absolute)")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> FileWrite:
        resolved = self._resolve_path(path)
        existed = resolved.exists()
        old_text: str | None = None
        old_complete = True
        if existed:
            try:
                with resolved.open("r") as stream:
                    old_text = stream.read(_MAX_EVENT_TEXT_CHARS + 1)
                old_complete = len(old_text) <= _MAX_EVENT_TEXT_CHARS
                old_text = pformat(old_text, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
            except (OSError, UnicodeError):
                # Observation must not make an otherwise valid overwrite fail.
                old_complete = False

        result = await self._shell.write_file(path, content)
        bounded_content = pformat(content, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
        content_truncated = len(content) > _MAX_EVENT_TEXT_CHARS
        self._emit(
            FileEdit(
                path=str(resolved),
                operation="update" if existed else "create",
                old_text=old_text,
                new_text=bounded_content,
                start_line=1 if existed and old_complete and old_text is not None else None,
                end_line=(
                    max(1, _line_count(old_text))
                    if existed and old_complete and old_text is not None
                    else None
                ),
                diff=_edit_diff(path, old_text or "", bounded_content, 1),
                content_complete=old_complete and not content_truncated,
            )
        )
        return result
