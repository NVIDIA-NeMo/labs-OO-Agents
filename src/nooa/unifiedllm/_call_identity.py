# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK-free runtime call identity, independent of request and model settings."""

from contextlib import contextmanager
from contextvars import ContextVar

_call_identity: ContextVar[str | None] = ContextVar("llm_runtime_identity", default=None)


@contextmanager
def call_identity(identity: str | None):
    if identity is not None and (
        not isinstance(identity, str) or not identity or len(identity) > 256
    ):
        raise ValueError("Runtime call identity must be a bounded non-empty string")
    token = _call_identity.set(identity)
    try:
        yield
    finally:
        _call_identity.reset(token)


def current_call_identity() -> str | None:
    return _call_identity.get()
