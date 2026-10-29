# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform policy: native Windows is unsupported; use WSL 2.

``import nooa`` on native Windows raises :class:`UnsupportedPlatformError`
unless ``NOOA_SKIP_PLATFORM_CHECK=1`` is set (like Node.js's
``NODE_SKIP_PLATFORM_CHECK``). With the override, the pure-Python core and
``nooa connect`` run and are tested in CI; nothing else is supported there.

Features that need POSIX (bash sessions, the sandbox, process groups, file
locks, POSIX signals) call :func:`require_posix` at the point of use, never at
import time. We do not port those features to Windows; use WSL 2 for them.
"""

from __future__ import annotations

import os
import sys

from nooa.errors import NemoOOAgentsRuntimeError

WSL_HINT = "Native Windows runs only the pure-Python core of nooa; use WSL 2 (https://learn.microsoft.com/windows/wsl/install) for this feature."


SKIP_PLATFORM_CHECK_ENV = "NOOA_SKIP_PLATFORM_CHECK"
UNSUPPORTED_PLATFORM_MESSAGE = (
    "nooa does not support native Windows; use WSL 2 "
    "(https://learn.microsoft.com/windows/wsl/install). "
    f"Setting the {SKIP_PLATFORM_CHECK_ENV} environment variable to 1 skips this check: "
    "only the pure-Python core and `nooa connect` are tested on native Windows, "
    "and other issues found only there will not be fixed."
)


class UnsupportedPlatformError(ImportError):
    """``import nooa`` on native Windows without ``NOOA_SKIP_PLATFORM_CHECK=1``."""


class PosixOnlyFeatureError(NemoOOAgentsRuntimeError, NotImplementedError):
    """A POSIX-only nooa feature was used on native Windows."""

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(f"{feature} requires Linux or macOS. {WSL_HINT}")


def is_native_windows() -> bool:
    """True on native Windows (not WSL, which reports ``linux``)."""
    return sys.platform == "win32"


def require_posix(feature: str) -> None:
    """Raise :class:`PosixOnlyFeatureError` naming *feature* on native Windows."""
    if is_native_windows():
        raise PosixOnlyFeatureError(feature)


def platform_check_skipped() -> bool:
    """True when the user opted in to running on an unsupported platform."""
    return os.environ.get(SKIP_PLATFORM_CHECK_ENV, "").strip().lower() in {"1", "true", "yes"}


def check_platform() -> None:
    """Refuse native Windows at import unless the user opted in (see module doc)."""
    if is_native_windows() and not platform_check_skipped():
        raise UnsupportedPlatformError(UNSUPPORTED_PLATFORM_MESSAGE)
