# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convenience helpers for configuring nooa logging.

Library code uses ``logging.getLogger(__name__)`` throughout, producing a
hierarchy rooted at ``nooa``.  Applications that want to see what the
library is doing can call :func:`enable_logging` instead of wiring up
handlers manually::

    from nooa import enable_logging

    # Everything at DEBUG
    enable_logging()

    # Only strategy decisions
    enable_logging(name="nooa.strategies")

    # INFO-level overview
    enable_logging(level=logging.INFO)

The function is intentionally minimal — it adds a single
:class:`~logging.StreamHandler` with a readable format.  For production
use, configure logging via :func:`logging.config.dictConfig` as usual.

Logger hierarchy (all children of ``nooa``)::

    nooa.agent               Agent lifecycle & configuration
    nooa.runtime.actor       Execution engine
    nooa.runtime.hooks       Hook dispatch
    nooa.runtime.code_validator  Code validation / safety checks
    nooa.strategies.codeact  CodeAct strategy
    nooa.strategies.predict  Predict strategy
    nooa.strategies.pure_python  PurePython strategy
    nooa.strategies.reflexion  Reflexion strategy
    nooa.tools.*             Individual tool modules
    nooa.storage.*           Storage backends
    nooa.library_manager     Library/skill loading
    nooa.skill_registry      Skill discovery
"""

import logging
import sys
from typing import IO

_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DEFAULT_DATE_FORMAT = "%H:%M:%S"


def enable_logging(
    level: int = logging.DEBUG,
    name: str = "nooa",
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATE_FORMAT,
    stream: IO[str] | None = None,
) -> None:
    """Attach a :class:`~logging.StreamHandler` to an *nooa* logger.

    This is a convenience for interactive / development use.  It is safe to
    call multiple times — duplicate handlers are not added, but the log
    level is always updated.  Format changes on repeated calls require
    removing the handler first.

    Parameters
    ----------
    level:
        Log level to set on the target logger (default ``DEBUG``).
    name:
        Logger name to configure.  Defaults to the library root
        (``"nooa"``).  Pass a child name to narrow the scope, e.g.
        ``"nooa.strategies"`` or ``"nooa.runtime.actor"``.
    fmt:
        Log format string.
    datefmt:
        Date format string.
    stream:
        Output stream (default ``sys.stderr``).
    """
    target = logging.getLogger(name)

    if stream is None:
        stream = sys.stderr

    # Avoid adding duplicate handlers on repeated calls.
    # Use exact type check — NullHandler inherits from Handler (not
    # StreamHandler in practice, but guard defensively) and has no .stream.
    for h in target.handlers:
        if type(h) is logging.StreamHandler and h.stream is stream:
            target.setLevel(level)
            return

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    target.addHandler(handler)
    target.setLevel(level)
