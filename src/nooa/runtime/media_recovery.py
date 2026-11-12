# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media recovery in the runtime: the request image limit and its harness counters.

:class:`ImageLimitRecovery` is a request recovery
(:mod:`nooa.runtime.request_recovery`): when a provider refuses a request for
its image count, it learns the limit, collapses the oldest image events and
asks for a retry; later requests of the session stay under the limit.

The counters live in the ``media`` extension of :class:`HarnessMetrics`
(span keys ``harness.media.<name>``), not on every agent's core metrics.
Importing this module (``nooa.runtime`` does) registers the recovery, the
extension and the unifiedllm metric event ``media_rejected_retry``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from nooa.runtime.harness_metrics import (
    HarnessMetrics,
    HarnessMetricsExtension,
    SchemaEntry,
    get_harness_metrics,
    register_harness_metrics_extension,
    register_llm_metric_event,
)
from nooa.runtime.request_recovery import register_request_recovery

__all__ = ["ImageLimitRecovery", "MediaHarnessMetrics", "image_block_count", "image_limit_recovery"]

logger = logging.getLogger(__name__)


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


def image_block_count(event: Any) -> int:
    """The image content blocks (``image_url``) an event sends to the model."""
    images = getattr(event, "images", None)
    if not isinstance(images, list):
        return 0
    return sum(
        1 for block in images if isinstance(block, dict) and block.get("type") == "image_url"
    )


@register_request_recovery
class ImageLimitRecovery:
    """A provider's per-request image limit: remember it and collapse old image events."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        # The provider's per-request image limit, once a request was refused for it.
        self.limit: int | None = None

    def before_send(self) -> bool:
        """A limit learned earlier in this session: keep this request under it."""
        return self.limit is not None and self.collapse(self.limit) > 0

    def recover(self, exc: BaseException) -> bool | None:
        """Remember the limit, collapse old images; True to retry, None if not recoverable."""
        from nooa.unifiedllm.media_inputs import image_limit

        limit = image_limit(exc)
        if limit is None:
            return None
        self.limit = limit  # the session's later requests stay under it
        if self.collapse(limit) == 0:
            return None  # nothing left to collapse: the same request would fail again
        logger.warning(
            "Provider image limit %d: collapsed older image events and retrying: %s",
            limit,
            str(exc)[:300],
        )
        return True

    def collapse(self, limit: int) -> int:
        """Keep the newest ``limit`` images in the active events; collapse older image events.

        Each image-bearing event (a ``Task`` or ``PythonOutput`` with image
        blocks) older than the newest ``limit`` images is collapsed on its
        own. Only its images go: the summary carries the event's text (the
        cell's printed output, the task prompt) followed by a short note in
        place of the images, and the original stays readable at
        ``self.events["<tag>"]``. Returns the number collapsed.
        """
        from nooa.agentdoc import pformat

        em = self.runtime.event_manager
        # Only the events the view put in the request: an excluded event neither
        # uses the budget nor gets collapsed.
        included = self.runtime._included_event_ids()
        bearing: list[tuple[str, int]] = []
        events: dict[str, Any] = {}
        for tag in em.keys():
            try:
                event = em[tag]
            except Exception:  # noqa: BLE001
                continue
            if included is not None and event.id not in included:
                continue
            count = image_block_count(event)
            if count:
                bearing.append((tag, count))
                events[tag] = event
        kept = 0
        to_collapse: list[str] = []
        for tag, count in reversed(bearing):
            if kept + count <= limit:
                kept += count
            else:
                to_collapse.append(tag)
                kept = limit  # an older, smaller event must not slip in after a skipped one
        collapsed = 0
        for tag in reversed(to_collapse):
            try:
                text = pformat(events[tag])  # images are repr=False: the event's text only
            except Exception:  # noqa: BLE001
                text = ""
            summary_text = (text + "\n" if text else "") + (
                f"[image collapsed (request image limit {limit}); "
                f'original: self.events["{tag}"]; show it again: '
                f'show(Image.from_url(self.events["{tag}"].images[0]["image_url"]["url"]))]'
            )
            try:
                em.collapse(tag, tag, summary_text=summary_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("image-limit collapse failed for %s: %s", tag, exc)
                continue
            collapsed += 1
        if collapsed:
            get_harness_metrics().extension(MediaHarnessMetrics).image_limit_collapsed(
                limit, collapsed
            )
        return collapsed


def image_limit_recovery(runtime: Any) -> ImageLimitRecovery:
    """The runtime's :class:`ImageLimitRecovery` instance."""
    for recovery in runtime._request_recoveries:
        if isinstance(recovery, ImageLimitRecovery):
            return recovery
    raise LookupError("the runtime has no ImageLimitRecovery")
