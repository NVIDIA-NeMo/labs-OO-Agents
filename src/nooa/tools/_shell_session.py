# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The persistent-shell interface shared by file tools and their platform adapters."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, Protocol

ShellName = Literal["bash", "powershell"]


class ShellSession(Protocol):
    """Serialized commands, persistent cwd/state, and explicit process ownership.

    ``cwd=`` temporarily changes directories; ``stdin=`` supplies UTF-8 bytes and
    EOF, independently of the command protocol. A timeout returns status 124
    and a separate flag. Streaming ends with ``('__done__', 'code,timeout')``.
    Recovery may restart the shell, losing variables and rerunning its init.
    """

    _keep_background_on_close: bool

    @property
    def shell_name(self) -> ShellName: ...

    @property
    def cwd(self) -> Path: ...

    async def start(self) -> None: ...

    async def run(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int]: ...

    async def run_with_timeout_flag(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int, bool]: ...

    def run_stream(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncGenerator[tuple[str, str]]: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


def create_shell_session(
    cwd: str | Path = ".",
    init_command: str | None = None,
    *,
    shell: Literal["auto", "bash", "powershell"] = "auto",
    keep_background_on_close: bool = False,
) -> ShellSession:
    """Choose native PowerShell on Windows and Bash on Unix, starting lazily."""
    if shell == "auto":
        shell = "powershell" if sys.platform == "win32" else "bash"
    if shell == "bash":
        from nooa.tools._bash_session import BashSession

        return BashSession(cwd, init_command, keep_background_on_close=keep_background_on_close)
    if shell == "powershell":
        from nooa.tools._powershell_session import PowerShellSession

        return PowerShellSession(
            cwd, init_command, keep_background_on_close=keep_background_on_close
        )
    raise ValueError(f"Unknown shell {shell!r}; expected 'auto', 'bash', or 'powershell'")
