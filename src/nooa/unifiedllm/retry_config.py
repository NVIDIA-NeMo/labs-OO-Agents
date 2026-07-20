# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict


class RetryConfig(BaseModel):
    """Retry behaviour for LLM API calls."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.3
    rate_limit_extra_retries: int = 3
    rate_limit_base_delay: float = 3.0
    rate_limit_backoff_base: float = 3.0  # was hardcoded as 3.0 in _calculate_delay()
    retryable_status_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504})
    retryable_exceptions: tuple[type[Exception], ...] = (
        asyncio.TimeoutError,
        TimeoutError,
        ConnectionError,
    )
    retry_on_empty_content: bool = False
    on_retry: Callable[[int, Exception, float], None] | None = None
