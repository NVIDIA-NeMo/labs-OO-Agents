# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE tool result types."""

from dataclasses import dataclass, field
from typing import Annotated, Literal

from nooa.agentdoc import spec


@dataclass
class RunResult:
    stdout: Annotated[str, spec(description="Command output")]
    stderr: Annotated[str, spec(description="Error output")]
    returncode: Annotated[int, spec(description="Exit code (0 = success)")]
    timed_out: Annotated[bool, spec(description="True if command exceeded timeout")] = False

    @property
    def success(self) -> bool:
        """True when exit code is 0."""
        return self.returncode == 0

    @property
    def text(self) -> str:
        """Formatted output for display."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if self.timed_out:
            parts.append("[timed out]")
        elif self.returncode != 0:
            parts.append(f"[exit code: {self.returncode}]")
        return "\n".join(parts) if parts else "(no output)"

    def __str__(self) -> str:
        return self.text


@dataclass
class ViewResult:
    path: Annotated[str, spec(description="File path")]
    content: Annotated[str, spec(description="File content with line numbers")]
    start_line: Annotated[int, spec(description="First displayed line (1-indexed)")]
    end_line: Annotated[int, spec(description="Last displayed line")]
    total_lines: Annotated[int, spec(description="Total lines in the file")]
    truncated: Annotated[bool, spec(description="True if output was clipped")] = False

    @property
    def text(self) -> str:
        """Formatted output for display."""
        header = f"[{self.path}] lines {self.start_line}-{self.end_line} of {self.total_lines}"
        if self.truncated:
            header += " (truncated)"
        return f"{header}\n{self.content}"

    def __str__(self) -> str:
        return self.text


@dataclass
class EditResult:
    path: Annotated[str, spec(description="File that was edited")]
    diff: Annotated[str, spec(description="Unified diff of changes")]
    lint_errors: Annotated[
        list[str], spec(description="NEW lint errors only (pre-existing filtered out)")
    ] = field(default_factory=list)
    success: Annotated[bool, spec(description="Whether the edit was applied")] = True
    error: Annotated[str, spec(description="Error message if success is False")] = ""

    @property
    def text(self) -> str:
        """Formatted output for display."""
        if not self.success:
            return f"Edit failed: {self.error}"
        parts = [f"Edited {self.path}"]
        if self.diff:
            parts.append(f"\n{self.diff}")
        if self.lint_errors:
            parts.append("\nNew lint errors:")
            for err in self.lint_errors:
                parts.append(f"  {err}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.text


@dataclass
class WriteResult:
    path: Annotated[str, spec(description="File path")]
    created: Annotated[bool, spec(description="True if new file, False if overwritten")]
    lines: Annotated[int, spec(description="Number of lines written")]
    diff: Annotated[str, spec(description="Unified diff (empty for new files)")] = ""

    @property
    def text(self) -> str:
        """Formatted output for display."""
        action = "Created" if self.created else "Wrote"
        return f"{action} {self.path} ({self.lines} lines)"

    def __str__(self) -> str:
        return self.text


@dataclass
class SearchResult:
    matches: Annotated[list[str], spec(description="Matching lines or file paths")]
    total_matches: Annotated[int, spec(description="Total number of matches")]
    truncated: Annotated[bool, spec(description="True if results were capped")] = False

    @property
    def text(self) -> str:
        """Formatted output for display."""
        if not self.matches:
            return "No matches found."
        parts = list(self.matches)
        if self.truncated:
            parts.append(
                f"\n... ({self.total_matches} total matches, showing first {len(self.matches)})"
            )
        else:
            parts.append(f"\n({self.total_matches} matches)")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.text


@dataclass
class StreamEvent:
    kind: Annotated[Literal["stdout", "stderr"], spec(description="Stream chunk event kind")]
    text: Annotated[str, spec(description="Chunk text for the stream event")]


@dataclass
class StreamDone:
    kind: Annotated[Literal["done"], spec(description="Terminal stream event kind")]
    returncode: Annotated[int, spec(description="Final process exit code")]
    timed_out: Annotated[bool, spec(description="True if command exceeded timeout")]


@dataclass
class LsResult:
    path: Annotated[str, spec(description="Directory path")]
    tree: Annotated[str, spec(description="Formatted tree output")]
    num_files: Annotated[int, spec(description="Number of entries shown")]
    truncated: Annotated[bool, spec(description="True if entries were capped")] = False

    @property
    def text(self) -> str:
        """Formatted output for display."""
        header = f"[{self.path}] ({self.num_files} entries)"
        if self.truncated:
            header += " (truncated)"
        return f"{header}\n{self.tree}"

    def __str__(self) -> str:
        return self.text
