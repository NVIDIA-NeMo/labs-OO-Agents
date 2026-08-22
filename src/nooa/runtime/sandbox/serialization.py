# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IPC-safe marshaling of execution results across the process boundary.

``ExecutionResult`` carries fields that cannot be pickled (live callables,
arbitrary return values, exceptions). This module converts a worker-side result
into a picklable :class:`ResultDTO` and reconstructs a faithful
``ExecutionResult`` on the parent, following the contract in the design doc:

* ``defined_methods`` / ``captured_locals`` stay in the worker (empty on parent).
* ``returned_value`` crosses only if picklable, else becomes a serialization error.
* ``error`` is reduced to type/message plus a worker-formatted diagnostic and
  reconstructed as a lightweight parent-side exception.
* ``signal`` (``return_result``) is marshaled as a picklable record.
* ``images`` are already dicts.
"""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nooa.agentdoc import TruncatingStringIO
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.runtime.sandbox.errors import CellSerializationError

# ``TruncatingStringIO`` adds a human-readable envelope around retained
# head/tail content. Sandbox IPC has a fixed safety ceiling independent of a
# caller's configured capture budget.
_MAX_ERROR_CONTENT = DEFAULT_TRUNCATION_CONFIG.capture.max_error
_MAX_ERROR_TRANSPORT = _MAX_ERROR_CONTENT + 1_024


def _effective_error_limit(max_error: int | None) -> int:
    """Return the configured content budget clamped to the IPC safety ceiling."""
    requested = _MAX_ERROR_CONTENT if max_error is None else max_error
    return min(requested, _MAX_ERROR_CONTENT)


def _hard_bound_text(value: str, limit: int) -> str:
    """Cap arbitrary text without ever exceeding ``limit`` characters."""
    if len(value) <= limit:
        return value
    marker = "...<truncated>"
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker


def _bounded_error_message(
    value: str,
    *,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> str:
    """Apply the effective capture policy once to a raw message before IPC."""
    content_limit = _effective_error_limit(max_error)
    effective_tail = tail_chars if tail_chars is None or tail_chars < content_limit else None
    if len(value) <= content_limit:
        return value
    stream = TruncatingStringIO(limit=content_limit, tail_chars=effective_tail)
    stream.write(value)
    return _hard_bound_text(stream.getvalue(), _MAX_ERROR_TRANSPORT)


def _bounded_formatted_error(
    value: str,
    *,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> str:
    """Apply capture policy once and enforce a fixed IPC ceiling."""
    content_limit = _effective_error_limit(max_error)
    if len(value) <= content_limit:
        return value
    if value.startswith("<truncated-output>\n") and value.endswith("\n</truncated-output>"):
        # A budget-aware formatter has already retained the configured head and
        # tail. Avoid nesting its envelope, while bounding metadata overhead.
        return _hard_bound_text(value, min(_MAX_ERROR_TRANSPORT, content_limit + 1_024))
    return _bounded_error_message(value, max_error=content_limit, tail_chars=tail_chars)


def is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)
        return True
    except BaseException:
        return False


@dataclass
class ErrorDTO:
    """Picklable surrogate for an exception raised inside a cell."""

    type_name: str
    message: str
    formatted_error: str = ""


@dataclass
class SignalDTO:
    """Picklable surrogate for a ``return_result()`` control-flow signal."""

    result: Any


@dataclass
class ResultDTO:
    """Everything from a worker cell run that can cross the pipe."""

    stdout: str = ""
    stderr: str = ""
    error: ErrorDTO | None = None
    signal: SignalDTO | None = None
    returned_value: Any = None
    has_return: bool = False
    explicit_return: bool = False
    images: list[dict[str, Any]] = field(default_factory=list)
    wrapper_line_offset: int = 0
    defined_method_names: list[str] = field(default_factory=list)


class _SurrogateCellError(Exception):
    """Parent-side reconstruction of a worker exception.

    Preserves the original type name. The worker-rendered diagnostic remains
    separate in :attr:`ErrorDTO.formatted_error` and is copied to the parent
    ``ExecutionResult.formatted_error`` by :func:`dto_to_result`.
    """

    def __init__(self, dto: ErrorDTO):
        super().__init__(dto.message)
        self.original_type = dto.type_name

    def __str__(self) -> str:
        return self.args[0] if self.args else self.original_type


def result_to_dto(
    result: Any,
    *,
    error_formatter: Callable[..., str] | None = None,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> ResultDTO:
    """Convert a worker-side ``ExecutionResult`` into a picklable DTO.

    The presence of a control-flow signal is keyed off ``result.signal`` (not a
    sentinel payload value), and the signal payload is picklability-checked just
    like ``returned_value`` so a non-picklable ``return_result(...)`` yields a
    clean error instead of crashing the worker on ``conn.send``.
    """
    from nooa.events import _NO_RETURN

    dto = ResultDTO(
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        images=list(result.images or []),
        wrapper_line_offset=getattr(result, "wrapper_line_offset", 0),
        defined_method_names=sorted(getattr(result, "defined_methods", {}) or {}),
    )

    if result.error is not None:
        err = result.error
        # User-defined exception formatting must not break the worker send path.
        error_type = type(err).__name__
        try:
            message = str(err) or error_type
        except BaseException:
            message = error_type

        if error_formatter is None:
            from nooa.errors.formatting import format_error_for_llm

            error_formatter = format_error_for_llm

        effective_max_error = _effective_error_limit(max_error)
        try:
            formatter_kwargs: dict[str, Any] = {
                "line_offset": getattr(result, "wrapper_line_offset", 0),
            }
            try:
                signature = inspect.signature(error_formatter)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                for optional_kwargs in (
                    {"max_error": effective_max_error, "tail_chars": tail_chars},
                    {"max_error": effective_max_error},
                    {},
                ):
                    try:
                        signature.bind(err, **formatter_kwargs, **optional_kwargs)
                    except TypeError:
                        continue
                    formatter_kwargs.update(optional_kwargs)
                    break
            formatted_error = error_formatter(err, **formatter_kwargs)
        except BaseException:
            formatted_error = f"{error_type}: {message}"

        dto.error = ErrorDTO(
            type_name=error_type,
            message=_bounded_error_message(
                message,
                max_error=effective_max_error,
                tail_chars=tail_chars,
            ),
            formatted_error=_bounded_formatted_error(
                formatted_error,
                max_error=effective_max_error,
                tail_chars=tail_chars,
            ),
        )
        return dto

    if result.signal is not None:
        payload = getattr(result.signal, "result", None)
        if is_picklable(payload):
            dto.signal = SignalDTO(result=payload)
        else:
            dto.error = ErrorDTO(
                type_name="CellSerializationError",
                message=(
                    "return_result(...) was called with a value that is not picklable and "
                    "cannot cross the sandbox boundary. Return a JSON/pickle-safe value "
                    "(numbers, str, list, dict, ndarray) instead."
                ),
            )
        return dto

    rv = result.returned_value
    if rv is not _NO_RETURN:
        if is_picklable(rv):
            dto.returned_value = rv
            dto.has_return = True
            dto.explicit_return = bool(result.explicit_return)
        else:
            dto.error = ErrorDTO(
                type_name="CellSerializationError",
                message=(
                    f"Return value of type {type(rv).__name__!r} is not picklable and "
                    "cannot cross the sandbox boundary. Keep it in the namespace and "
                    "return a JSON/pickle-safe summary instead."
                ),
            )
    return dto


def _reconstruct_error(err: ErrorDTO) -> Exception:
    """Rebuild a parent-side exception from an :class:`ErrorDTO`.

    Common builtin exceptions (ValueError, KeyError, MemoryError, ...) are
    re-instantiated as their real type so ``_format_error`` and the IPython
    formatter render the faithful ``<Type>: <message>``. The worker-rendered
    diagnostic remains in :attr:`ErrorDTO.formatted_error`; it is not attached to
    the exception. Anything else falls back to :class:`_SurrogateCellError`.
    """
    import builtins as _bi

    if err.type_name == "CellSerializationError":
        exc: Exception = CellSerializationError(err.message)
    elif err.type_name == "SandboxStateError":
        # A cell tried to mutate non-self module-level state; reconstruct the real
        # type (the parent has it) so callers see SandboxStateError, not a surrogate.
        from nooa.runtime.sandbox.readonly import SandboxStateError

        prefix = "SandboxStateError: "
        msg = err.message[len(prefix) :] if err.message.startswith(prefix) else err.message
        exc = SandboxStateError(msg)
    else:
        cls = getattr(_bi, err.type_name, None)
        if isinstance(cls, type) and issubclass(cls, Exception):
            try:
                # Strip a leading "Type: " the message may already carry.
                msg = err.message
                prefix = f"{err.type_name}: "
                if msg.startswith(prefix):
                    msg = msg[len(prefix) :]
                exc = cls(msg)
            except Exception:
                exc = _SurrogateCellError(err)
        else:
            exc = _SurrogateCellError(err)
    return exc


def dto_to_result(dto: ResultDTO, *, signal_factory: Any = None) -> Any:
    """Reconstruct a parent-side ``ExecutionResult`` from a :class:`ResultDTO`.

    ``signal_factory(payload) -> ExecutionSignal`` rebuilds the ``return_result``
    signal from its marshaled payload (supplied by the caller that owns the
    concrete signal type).
    """
    from nooa.events import _NO_RETURN, ExecutionResult

    error: Exception | None = None
    if dto.error is not None:
        error = _reconstruct_error(dto.error)

    signal = None
    if dto.signal is not None and signal_factory is not None:
        signal = signal_factory(dto.signal.result)

    return ExecutionResult(
        stdout=dto.stdout,
        stderr=dto.stderr,
        error=error,
        formatted_error=dto.error.formatted_error if dto.error is not None else "",
        signal=signal,
        defined_methods={},
        returned_value=dto.returned_value if dto.has_return else _NO_RETURN,
        explicit_return=dto.explicit_return,
        captured_locals={},
        images=dto.images,
        wrapper_line_offset=dto.wrapper_line_offset,
    )
