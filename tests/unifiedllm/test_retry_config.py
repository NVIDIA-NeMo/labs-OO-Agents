# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from nooa.unifiedllm.retry_config import RetryConfig


def test_retry_config_is_pydantic_model():
    assert issubclass(RetryConfig, BaseModel)


def test_retry_config_defaults():
    c = RetryConfig()
    assert c.max_retries == 3
    assert c.base_delay == 1.0
    assert c.max_delay == 60.0
    assert c.exponential_base == 2.0
    assert c.jitter_factor == 0.3
    assert c.rate_limit_extra_retries == 3
    assert c.rate_limit_base_delay == 3.0
    assert c.rate_limit_backoff_base == 3.0
    assert c.retryable_status_codes == frozenset({429, 500, 502, 503, 504})
    assert asyncio.TimeoutError in c.retryable_exceptions
    assert c.retry_on_empty_content is False
    assert c.on_retry is None


def test_retry_config_frozen():
    c = RetryConfig()
    with pytest.raises(ValidationError):
        c.max_retries = 5


def test_retryable_status_codes_is_frozenset():
    c = RetryConfig()
    assert isinstance(c.retryable_status_codes, frozenset)


def test_retryable_exceptions_is_typed_tuple():
    c = RetryConfig()
    assert isinstance(c.retryable_exceptions, tuple)
    for exc_type in c.retryable_exceptions:
        assert isinstance(exc_type, type)
        assert issubclass(exc_type, BaseException)


def test_rate_limit_backoff_base_is_new_field():
    # This field did not exist in the old dataclass
    c = RetryConfig(rate_limit_backoff_base=5.0)
    assert c.rate_limit_backoff_base == 5.0
