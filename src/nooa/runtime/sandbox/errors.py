# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exceptions raised by the sandboxed cell-execution backend."""

from __future__ import annotations


class SandboxError(RuntimeError):
    """Base class for sandbox infrastructure errors."""


class SandboxExecutionError(RuntimeError):
    """A user-code failure reconstructed from a sandbox worker.

    ``diagnostic`` is rendered inside the worker while its traceback and source
    cache still exist. The parent-side formatter treats it as trusted backend
    output; ``original_type`` preserves the worker exception's type name.
    """

    def __init__(
        self,
        original_type: str,
        message: str,
        diagnostic: str,
        original_error: Exception,
    ) -> None:
        super().__init__(message)
        self.original_type = original_type
        self.diagnostic = diagnostic
        self.original_error = original_error

    def __str__(self) -> str:
        message = super().__str__()
        return f"{self.original_type}: {message}" if message else self.original_type


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
