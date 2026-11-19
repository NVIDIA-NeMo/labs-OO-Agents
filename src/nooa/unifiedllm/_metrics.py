# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The harness-metrics hook for unifiedllm: a ContextVar callback and its recorder.

No reverse import needed: the callback is injected by actor.py at session start.
"""

import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

_llm_metrics_callback: ContextVar[Callable[[str, Any], None] | None] = ContextVar(
    "llm_metrics_callback", default=None
)


def _record_llm_metric(event: str, detail: Any = None) -> None:
    """Fire-and-forget metric recording. No-op if no callback is set.

    Swallows any exception from the callback — instrumentation must never
    break the LLM call flow.
    """
    cb = _llm_metrics_callback.get()
    if cb is not None:
        try:
            cb(event, detail)
        except Exception as e:  # noqa: BLE001
            logger.debug("Metric callback failed for event %r: %s", event, e)
