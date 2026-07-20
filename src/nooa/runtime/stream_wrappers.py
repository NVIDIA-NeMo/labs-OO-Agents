# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure stream wrapper classes with no agent/LLM dependencies."""

import contextvars
import io
from collections.abc import Iterator
from typing import Any

# Task-local flag to block stdin reads (prevents hangs from input(), sys.stdin.read(), etc.)
# When True for the current async task, any stdin read raises RuntimeError
_block_stdin_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "block_stdin", default=False
)


class ContextVarStream:
    """Stream wrapper that redirects writes to a contextvar buffer when set.

    This provides async-safe output capture: each asyncio task can set its own
    buffer via the contextvar, and writes go to that buffer. When no buffer is
    set, writes go to the original stream.

    This captures ALL output including:
    - print() calls
    - sys.stdout.write() / sys.stderr.write()
    - logging handlers that write to stderr
    - SyntaxWarning / RuntimeWarning (Python 3.12+ writes these via sys.stderr)
    """

    def __init__(
        self,
        original: io.TextIOBase | Any,
        buffer_var: contextvars.ContextVar[io.StringIO | None],
        name: str = "stdout",
    ):
        self._original = original
        self._buffer_var = buffer_var
        self._name = name
        # Preserve original stream attributes for compatibility
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "strict")
        self.newlines = getattr(original, "newlines", None)
        self.mode = getattr(original, "mode", "w")

    def write(self, data: str) -> int:
        """Write to contextvar buffer if set, otherwise to original stream."""
        buffer = self._buffer_var.get()
        if buffer is not None:
            return buffer.write(data)
        return self._original.write(data)

    def writelines(self, lines: list[str]) -> None:
        """Write multiple lines."""
        buffer = self._buffer_var.get()
        if buffer is not None:
            buffer.writelines(lines)
        else:
            self._original.writelines(lines)

    def flush(self) -> None:
        """Flush both buffer and original stream."""
        buffer = self._buffer_var.get()
        if buffer is not None:
            buffer.flush()
        self._original.flush()

    def fileno(self) -> int:
        """Return file descriptor of original stream (for subprocess compatibility)."""
        return self._original.fileno()

    def isatty(self) -> bool:
        """Check if original stream is a TTY."""
        return self._original.isatty()

    def readable(self) -> bool:
        """Streams are not readable."""
        return False

    def writable(self) -> bool:
        """Streams are writable."""
        return True

    def seekable(self) -> bool:
        """Streams are not seekable."""
        return False

    def close(self) -> None:
        """Don't actually close - we're wrapping shared streams."""
        pass

    @property
    def closed(self) -> bool:
        """Check if original stream is closed."""
        return getattr(self._original, "closed", False)

    def __repr__(self) -> str:
        return f"<ContextVarStream({self._name}) wrapping {self._original!r}>"

    def __getattr__(self, name: str) -> Any:
        """Pass through any other attribute accesses to the original stream.

        This ensures compatibility with code that accesses less common stream
        attributes like 'buffer', 'line_buffering', 'name', etc.
        """
        return getattr(self._original, name)


class BlockedStdinWrapper:
    """Stdin wrapper that blocks reads when _block_stdin_var is True.

    This provides async-safe stdin blocking: each asyncio task can enable
    blocking via the contextvar. When blocking is enabled, any read operation
    raises RuntimeError instead of hanging on stdin.

    This catches ALL stdin reads including:
    - input() builtin
    - sys.stdin.read() / readline() / readlines()
    - getpass.getpass()
    - Any library that reads from stdin
    """

    def __init__(self, original: io.TextIOBase | Any):
        self._original = original
        # Preserve original stream attributes for compatibility
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "strict")
        self.newlines = getattr(original, "newlines", None)
        self.mode = getattr(original, "mode", "r")

    def _check_blocked(self) -> None:
        """Raise if stdin reads are blocked for the current async task."""
        if _block_stdin_var.get():
            raise RuntimeError(
                "Reading from stdin is forbidden in agent-generated code. "
                "Use function parameters or return values instead of interactive input."
            )

    def read(self, size: int = -1) -> str:
        """Read from stdin (blocked when in agent code execution)."""
        self._check_blocked()
        return self._original.read(size)

    def readline(self, size: int = -1) -> str:
        """Read a line from stdin (blocked when in agent code execution)."""
        self._check_blocked()
        return self._original.readline(size)

    def readlines(self, hint: int = -1) -> list[str]:
        """Read all lines from stdin (blocked when in agent code execution)."""
        self._check_blocked()
        return self._original.readlines(hint)

    def __iter__(self) -> Iterator[str]:
        """Iterate over stdin lines (blocked when in agent code execution)."""
        self._check_blocked()
        return iter(self._original)

    def __next__(self) -> str:
        """Get next line from stdin (blocked when in agent code execution)."""
        self._check_blocked()
        return next(self._original)

    # Pass-through methods that don't read data
    def fileno(self) -> int:
        """Return file descriptor (for compatibility checks)."""
        return self._original.fileno()

    def isatty(self) -> bool:
        """Check if stdin is a TTY."""
        return self._original.isatty()

    def readable(self) -> bool:
        """Check if stream is readable."""
        return self._original.readable()

    def writable(self) -> bool:
        """Stdin is not writable."""
        return False

    def seekable(self) -> bool:
        """Stdin is not seekable."""
        return False

    def close(self) -> None:
        """Don't actually close - we're wrapping shared streams."""
        pass

    @property
    def closed(self) -> bool:
        """Check if original stream is closed."""
        return getattr(self._original, "closed", False)

    def __repr__(self) -> str:
        blocked = _block_stdin_var.get()
        status = "BLOCKED" if blocked else "open"
        return f"<BlockedStdinWrapper({status}) wrapping {self._original!r}>"

    def __getattr__(self, name: str) -> Any:
        """Pass through any other attribute accesses to the original stream."""
        return getattr(self._original, name)
