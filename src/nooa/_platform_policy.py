# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native-Windows refusal that ``import nooa`` runs first.

Standard library only: nothing from nooa is imported here, so the check runs
before any other nooa module loads. See :mod:`nooa._platform` for the policy and
for the POSIX-only feature guards.
"""

from __future__ import annotations

import os
import sys

SKIP_PLATFORM_CHECK_ENV = "NOOA_SKIP_PLATFORM_CHECK"
UNSUPPORTED_PLATFORM_MESSAGE = (
    "nooa does not support native Windows; use WSL 2 "
    "(https://learn.microsoft.com/windows/wsl/install). "
    f"Setting the {SKIP_PLATFORM_CHECK_ENV} environment variable to 1 skips this check: "
    "the pure-Python core and `nooa connect` are tested on native Windows; "
    "other features may work but are unsupported."
)


class UnsupportedPlatformError(ImportError):
    """``import nooa`` on native Windows without ``NOOA_SKIP_PLATFORM_CHECK=1``."""


def is_native_windows() -> bool:
    """True on native Windows (not WSL, which reports ``linux``)."""
    return sys.platform == "win32"


def platform_check_skipped() -> bool:
    """True when the user opted in to running on an unsupported platform."""
    return os.environ.get(SKIP_PLATFORM_CHECK_ENV, "").strip().lower() in {"1", "true", "yes"}


def _print_uncaught_refusal_as_one_line() -> None:
    """Show an uncaught refusal (console scripts, ``python -c``) as one line, not a traceback.

    Only replaces the default hook; code that catches the ImportError is unaffected.
    """
    if sys.excepthook is not sys.__excepthook__:
        return
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, UnsupportedPlatformError):
            sys.stderr.write(f"Error: {exc}\n")
        else:
            previous(exc_type, exc, tb)

    sys.excepthook = hook


def check_platform() -> None:
    """Refuse native Windows at import unless the user opted in."""
    if is_native_windows() and not platform_check_skipped():
        _print_uncaught_refusal_as_one_line()
        raise UnsupportedPlatformError(UNSUPPORTED_PLATFORM_MESSAGE)
