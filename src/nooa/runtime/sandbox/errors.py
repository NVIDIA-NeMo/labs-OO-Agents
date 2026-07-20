# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exceptions raised by the sandboxed cell-execution backend."""

from __future__ import annotations


class SandboxError(RuntimeError):
    """Base class for sandbox infrastructure errors."""


class SandboxUnavailable(SandboxError):
    """A requested guardrail cannot be enforced on this host.

    Raised at executor start when ``SandboxConfig.require`` is True and the
    kernel lacks a mechanism needed to enforce a requested guardrail. Failing
    closed here is deliberate: the alternative is running untrusted code with
    a guard silently missing.
    """


class CellTimeoutError(SandboxError):
    """A cell exceeded its wall-clock deadline and the worker was killed."""


class CellMemoryError(SandboxError):
    """A cell exceeded its memory cap (address-space or resident-set)."""


class CellSerializationError(SandboxError):
    """A value could not cross the parent/worker process boundary.

    The value (a return value, a brokered tool argument, or a tool result) is
    not picklable. Keep it in the worker namespace and return a JSON/pickle-safe
    summary instead.
    """


class WorkerDiedError(SandboxError):
    """The worker process exited before answering a request."""
