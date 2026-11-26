# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recoveries from a refused model request, registered by the packages that own them.

``ActorRuntime`` creates one instance of each registered recovery per runtime
(in registration order) and, on every model request:

- calls ``before_send()`` once the messages are built (a recovery that learned
  a limit keeps later requests under it, counting only the events the context
  view selected; if it changes stored events, the messages are rebuilt once);
- when the request raises, offers the exception to each recovery in order.
  ``recover(exc)`` returns ``None`` (not this recovery's error: ask the next),
  ``True`` (recovered: rebuild the messages and retry) or ``False`` (this
  recovery's error, but it cannot recover: raise). A recovery that answered
  ``True`` or ``False`` is not asked again for that request.

After the registered recoveries the runtime tries its own context-window
recovery (archive old events, lower the reply budget).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["RequestRecovery", "register_request_recovery", "request_recoveries_for"]


class RequestRecovery(Protocol):
    """One runtime's recovery from a kind of refused request (see the module docs)."""

    def before_send(self) -> bool:
        """Called once a request's messages are built, before it is sent.

        Returns True when it changed stored events, so the runtime rebuilds the
        messages once.
        """
        ...

    def recover(self, exc: BaseException) -> bool | None:
        """None: not mine. True: recovered, retry. False: mine, cannot recover."""
        ...


_FACTORIES: list[Callable[[Any], RequestRecovery]] = []


def register_request_recovery[F: Callable[[Any], RequestRecovery]](factory: F) -> F:
    """Register ``factory(runtime) -> RequestRecovery``; usable as a class decorator.

    Registering the same factory again is a no-op.
    """
    if factory not in _FACTORIES:
        _FACTORIES.append(factory)
    return factory


def request_recoveries_for(runtime: Any) -> list[RequestRecovery]:
    """New instances of the registered recoveries for ``runtime``, in registration order."""
    return [factory(runtime) for factory in _FACTORIES]
