# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error types for pyp pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


class PipeError(Exception):
    """A subprocess or transform in the pipeline failed."""

    step: int
    cmd: str | None
    transform: str | None
    returncode: int | None
    stderr: str
    pipeline_repr: str
    input_line: int | None

    def format_error(self) -> str:
        """Format a detailed error message."""
        parts = [str(self.args[0])]
        if self.pipeline_repr:
            parts.append(f"  Pipeline: {self.pipeline_repr}")
        if self.cmd:
            parts.append(f"  Command:  {self.cmd}")
        if self.transform:
            parts.append(f"  Transform: {self.transform}")
        if self.returncode is not None:
            parts.append(f"  Exit:     {self.returncode}")
        if self.stderr:
            parts.append(f"  Stderr:   {self.stderr.rstrip()}")
        if self.input_line is not None:
            parts.append(f"  At line:  {self.input_line}")
        return "\n".join(parts)


def make_pipe_error(
    message: str,
    *,
    step: int = 0,
    cmd: str | None = None,
    transform: str | None = None,
    returncode: int | None = None,
    stderr: str = "",
    pipeline_repr: str = "",
    input_line: int | None = None,
) -> PipeError:
    """Create a PipeError with structured metadata."""
    obj = PipeError(message)
    obj.step = step
    obj.cmd = cmd
    obj.transform = transform
    obj.returncode = returncode
    obj.stderr = stderr
    obj.pipeline_repr = pipeline_repr
    obj.input_line = input_line
    return obj


@dataclass
class Result:
    """Outcome of a fully consumed pipeline."""

    lines: list[str] = field(default_factory=list)
    returncode: int = 0
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True if returncode is 0."""
        return self.returncode == 0

    @property
    def text(self) -> str:
        """Join lines with newlines."""
        return "\n".join(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __bool__(self) -> bool:
        return self.ok and bool(self.lines)

    def __repr__(self) -> str:
        n = len(self.lines)
        ok_str = "ok" if self.ok else f"FAIL(rc={self.returncode})"
        return f"Result({ok_str}, {n} lines)"
