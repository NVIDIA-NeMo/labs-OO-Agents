# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in shell tools and events shared by interactive coding hosts."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from uuid import uuid4

from pydantic import Field

from nooa.agentdoc import hidden, spec
from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role
from nooa.runtime.event_manager import EventManager
from nooa.skill import Skill
from nooa.tools._bash_session import BashSession
from nooa.tools._results import StreamDone, StreamEvent
from nooa.tools.shell_tools import FileWrite, Match, ShellResult, ShellTools

logger = logging.getLogger(__name__)


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
        Field(description="Complete content before the edit, when available"),
    ] = None
    new_text: Annotated[str, Field(description="Complete content after the edit")] = ""


class TerminalCommandStarted(EventBase):  # type: ignore[misc]
    """A command began in a persistent coding-agent terminal."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    command_id: Annotated[str, Field(description="Correlation ID for this command")]
    command: Annotated[str, Field(description="Shell command text")]
    working_directory: Annotated[str, Field(description="Working directory at command start")]
    has_stdin: Annotated[bool, Field(description="Whether stdin was supplied separately")] = False


class TerminalCommandOutput(EventBase):  # type: ignore[misc]
    """One streaming output chunk from a coding-agent terminal command."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    command_id: Annotated[str, Field(description="Correlation ID for this command")]
    stream: Annotated[Literal["stdout", "stderr"], Field(description="Output stream")]
    content: Annotated[str, Field(description="Output chunk")]


class TerminalCommandFinished(EventBase):  # type: ignore[misc]
    """A coding-agent terminal command completed or failed to launch."""

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    command_id: Annotated[str, Field(description="Correlation ID for this command")]
    exit_code: Annotated[int | None, Field(description="Process exit code when available")] = None
    stdout: Annotated[str, Field(description="Captured non-streaming stdout")] = ""
    stderr: Annotated[str, Field(description="Captured non-streaming stderr")] = ""
    timed_out: Annotated[bool, Field(description="Whether the command timed out")] = False
    error: Annotated[str, Field(description="Failure before an exit code was available")] = ""


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
        self._emit(
            TerminalCommandStarted(
                command_id=command_id,
                command=command,
                working_directory=str(self.cwd),
                has_stdin=stdin is not None,
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

        self._emit(
            TerminalCommandFinished(
                command_id=command_id,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
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
        self._emit(
            TerminalCommandStarted(
                command_id=command_id,
                command=command,
                working_directory=str(self.cwd),
            )
        )
        finished = False
        try:
            async for item in self._shell.run_stream(command, timeout=timeout):
                if isinstance(item, StreamDone):
                    self._emit(
                        TerminalCommandFinished(
                            command_id=command_id,
                            exit_code=item.returncode,
                            timed_out=item.timed_out,
                        )
                    )
                    finished = True
                else:
                    self._emit(
                        TerminalCommandOutput(
                            command_id=command_id,
                            stream=item.kind,
                            content=item.text,
                        )
                    )
                yield item
        except BaseException as error:
            if not finished:
                self._emit(
                    TerminalCommandFinished(
                        command_id=command_id,
                        error=str(error) or type(error).__name__,
                    )
                )
            raise

    @wraps(ShellTools.read)
    async def read(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd)")],
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

        resolved = self._resolve_path(path)
        old_text = resolved.read_text()
        result = await self._shell.replace(target, old_or_new, new)
        try:
            new_text = resolved.read_text()
        except (OSError, UnicodeError):
            logger.debug("Failed to observe completed file edit", exc_info=True)
            return result
        self._emit(
            FileEdit(
                path=str(resolved),
                operation="update",
                old_text=old_text,
                new_text=new_text,
            )
        )
        return result

    @wraps(ShellTools.write_file)
    async def write_file(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd)")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> FileWrite:
        resolved = self._resolve_path(path)
        existed = resolved.exists()
        old_text: str | None = None
        if existed:
            try:
                old_text = resolved.read_text()
            except (OSError, UnicodeError):
                # Observation must not make an otherwise valid overwrite fail.
                pass

        result = await self._shell.write_file(path, content)
        self._emit(
            FileEdit(
                path=str(resolved),
                operation="update" if existed else "create",
                old_text=old_text,
                new_text=content,
            )
        )
        return result
