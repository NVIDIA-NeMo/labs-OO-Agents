# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job explorer — browse and inspect background QueueManager jobs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from nooa_cli.interactive.runtime import JobSnapshot
from nooa_cli.interactive.state import AgentJobSummary

from .explorer_base import (
    ExplorerConfig,
    ExplorerModel,
    ExplorerView,
    wrap_plain_line,
)
from .subapp import SubviewKeyResult

type JobProjection = JobSnapshot | AgentJobSummary


@dataclass
class JobExplorerRow:
    """One background job in the explorer list."""

    channel: str
    label: str
    state: str
    delivered: int
    queued: int
    values: list[Any]
    search_text: str


def build_job_rows(snapshots: Iterable[JobProjection] | None) -> list[JobExplorerRow]:
    """Build explorer rows from immutable local-runtime projections."""
    rows: list[JobExplorerRow] = []
    for snapshot in snapshots or ():
        values = list(snapshot.values)
        search_parts = [
            snapshot.name,
            snapshot.label,
            snapshot.state,
            *[str(value) for value in values[-20:]],
        ]
        rows.append(
            JobExplorerRow(
                channel=snapshot.name,
                label=snapshot.label,
                state=snapshot.state,
                delivered=len(values),
                queued=snapshot.queued,
                values=values,
                search_text="\n".join(search_parts),
            )
        )
    return rows


class JobExplorerView(ExplorerView):
    """In-app subview for browsing background jobs."""

    def __init__(self, snapshots: Iterable[JobProjection]) -> None:
        rows = build_job_rows(snapshots)
        model = ExplorerModel(rows)
        config = ExplorerConfig(
            title="Job Explorer",
            detail_pane_name="job output",
            empty_message="No background jobs.",
            no_match_message="No jobs matching {query!r}.",
            list_ratio=0.4,
            actions={},
        )
        super().__init__(model, config)

    def format_row(self, row: JobExplorerRow, width: int) -> str:
        state_icon = {
            "running": "⚡",
            "done": "✓",
            "failed": "✗",
            "cancelled": "⊘",
        }.get(row.state, "?")
        line = (
            f"{state_icon} {row.channel:<20} {row.state:<10} "
            f"delivered={row.delivered:<6} queued={row.queued}"
        )
        if row.label != row.channel:
            line += f"  ({row.label})"
        return line[:width]

    def detail_lines(self, row: JobExplorerRow, width: int) -> list[str]:
        width = max(int(width), 20)
        lines: list[str] = [
            f"Channel: {row.channel}",
            f"Label: {row.label}",
            f"State: {row.state}",
            f"Delivered: {row.delivered}  Queued: {row.queued}",
            "",
        ]
        if not row.values:
            lines.append("(no buffered output)")
        else:
            lines.append(f"Output (last {min(len(row.values), 50)} of {len(row.values)}):")
            lines.append("")
            for val in row.values[-50:]:
                for raw in str(val).splitlines() or [""]:
                    lines.extend(wrap_plain_line(raw, width))
        return lines

    def handle_action(self, action: str, row: Any) -> SubviewKeyResult:
        return "ignored"
