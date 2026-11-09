# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media recovery in the runtime: its harness counters.

The counters live in the ``media`` extension of :class:`HarnessMetrics`
(span keys ``harness.media.<name>``), not on every agent's core metrics.
Importing this module registers the extension and the unifiedllm metric
event ``media_rejected_retry``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from nooa.runtime.harness_metrics import (
    HarnessMetrics,
    HarnessMetricsExtension,
    SchemaEntry,
    register_harness_metrics_extension,
    register_llm_metric_event,
)

__all__ = ["MediaHarnessMetrics"]


@register_harness_metrics_extension
class MediaHarnessMetrics(HarnessMetricsExtension):
    """Media recovery counters: rejected attachments and the request image limit."""

    namespace: ClassVar[str] = "media"

    # A provider rejected an attachment; retried with a note in its place (kind)
    rejected_retries: list[str] = Field(default_factory=list)
    image_limit_collapses: list[int] = Field(default_factory=list)  # limit M per recovery
    image_limit_events_collapsed: int = 0

    span_schema: ClassVar[tuple[SchemaEntry, ...]] = (
        SchemaEntry(
            "image_limit_collapsed.count",
            "Image-limit recoveries (oldest images collapsed)",
            "Media Recovery",
            lambda m: len(m.image_limit_collapses),
        ),
        SchemaEntry(
            "image_limit_collapsed.limits",
            "Provider image limits hit",
            "Media Recovery",
            lambda m: m.image_limit_collapses,
            True,
        ),
        SchemaEntry(
            "image_limit_collapsed.events",
            "Image events collapsed for the image limit",
            "Media Recovery",
            lambda m: m.image_limit_events_collapsed,
        ),
        SchemaEntry(
            "rejected_retry.count",
            "Rejected attachments retried without them",
            "Media Recovery",
            lambda m: len(m.rejected_retries),
        ),
        SchemaEntry(
            "rejected_retry.kinds",
            "Rejected attachment kinds",
            "Media Recovery",
            lambda m: m.rejected_retries,
            True,
        ),
    )

    def image_limit_collapsed(self, limit: int, events: int) -> None:
        """The request's images exceeded the provider limit ``limit``; ``events`` were collapsed."""
        self._append_item(self.image_limit_collapses, int(limit))
        self.image_limit_events_collapsed += int(events)

    def rejected_retry(self, kind: Any) -> None:
        """The provider rejected an attachment; the request was retried without it."""
        self._append(self.rejected_retries, str(kind or "unknown"))


def _record_rejected_retry(metrics: HarnessMetrics, kind: Any) -> None:
    metrics.extension(MediaHarnessMetrics).rejected_retry(kind)


register_llm_metric_event("media_rejected_retry", _record_rejected_retry)
